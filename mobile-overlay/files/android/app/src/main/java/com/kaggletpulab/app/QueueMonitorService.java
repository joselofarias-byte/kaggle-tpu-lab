package com.kaggletpulab.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.SystemClock;

import androidx.core.app.NotificationCompat;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Foreground service that holds one ntfy.sh SSE stream per queued TPU topic
 * while the app is backgrounded, and fires a high-priority notification the
 * moment a kernel reports ready/serving.
 *
 * <p>Why a foreground service: the WebView JS polling loop in NtfyListener is
 * suspended by Android once the app goes to the background (Doze / App
 * Standby), so queue-ready events would otherwise be missed for hours.
 *
 * <p>Notes:
 * <ul>
 *   <li>Uses HttpURLConnection only — no extra native dependencies.</li>
 *   <li>Reconnects with {@code ?since=<message-id>} so no events are lost
 *       across reconnects.</li>
 *   <li>Topics are persisted in SharedPreferences so a system-killed service
 *       resumes monitoring after START_STICKY restart.</li>
 *   <li>Android 15+ caps dataSync foreground services (~6h/24h); onTimeout()
 *       stops promptly and tells the user to relaunch from the app.</li>
 * </ul>
 */
public class QueueMonitorService extends Service {

    public static final String ACTION_ADD = "com.kaggletpulab.app.queuemonitor.ADD_TOPIC";
    public static final String ACTION_REMOVE = "com.kaggletpulab.app.queuemonitor.REMOVE_TOPIC";
    public static final String ACTION_STOP_ALL = "com.kaggletpulab.app.queuemonitor.STOP_ALL";
    public static final String EXTRA_TOPIC = "topic";

    private static final String PREFS = "queue_monitor";
    private static final String KEY_TOPICS = "topics";
    private static final String CHANNEL_QUEUE = "tpu_queue_channel";
    private static final String CHANNEL_READY = "tpu_ready_channel";
    private static final int NOTIF_FOREGROUND_ID = 0x5101;
    private static final int NOTIF_READY_ID_BASE = 0x5200;

    private static final long MAX_WORKER_AGE_MS = 24L * 60 * 60 * 1000; // 24h safety valve
    private static final long IDLE_RECONNECT_MS = 120_000; // reconnect if SSE silent for 2 min
    private static final long BACKOFF_BASE_MS = 5_000;
    private static final long BACKOFF_MAX_MS = 5 * 60_000;

    private final ConcurrentHashMap<String, TopicWorker> workers = new ConcurrentHashMap<>();
    private volatile boolean destroyed = false;

    /** Per-topic SSE worker; holds its own resume cursor. */
    private final class TopicWorker implements Runnable {
        final String topic;
        volatile String lastId;
        volatile HttpURLConnection conn; // set while connected; disconnect() unblocks readLine()
        final long startedAt = SystemClock.elapsedRealtime();
        Thread thread;

        TopicWorker(String topic) {
            this.topic = topic;
        }

        /** Force-unblock a thread stuck in readLine(). interrupt() alone cannot. */
        void shutdown() {
            HttpURLConnection c = conn;
            if (c != null) {
                try { c.disconnect(); } catch (Exception ignored) {}
            }
            if (thread != null) thread.interrupt();
        }

        @Override
        public void run() {
            long backoff = BACKOFF_BASE_MS;
            while (!destroyed && workers.get(topic) == this) {
                if (SystemClock.elapsedRealtime() - startedAt > MAX_WORKER_AGE_MS) break;
                HttpURLConnection c = null;
                try {
                    StringBuilder sb = new StringBuilder("https://ntfy.sh/").append(topic).append("/sse");
                    if (lastId != null) {
                        sb.append("?since=").append(URLEncoder.encode(lastId, "UTF-8"));
                    }
                    c = (HttpURLConnection) new URL(sb.toString()).openConnection();
                    conn = c;
                    c.setRequestProperty("Accept", "text/event-stream");
                    c.setConnectTimeout(15_000);
                    c.setReadTimeout((int) IDLE_RECONNECT_MS);
                    c.connect();
                    int code = c.getResponseCode();
                    if (code != HttpURLConnection.HTTP_OK) {
                        throw new java.io.IOException("ntfy HTTP " + code);
                    }
                    backoff = BACKOFF_BASE_MS; // successful connect resets backoff
                    readStream(c, this);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    // Network error / idle timeout / malformed stream -> backoff & retry.
                } finally {
                    conn = null;
                    if (c != null) c.disconnect();
                }
                if (destroyed || workers.get(topic) != this) break;
                try {
                    Thread.sleep(backoff);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
                backoff = Math.min(backoff * 2, BACKOFF_MAX_MS);
            }
            workers.remove(topic, this);
            onWorkerCountChanged();
        }
    }

    // ------------------------------------------------------------------ lifecycle

    @Override
    public void onCreate() {
        super.onCreate();
        createChannels();
        // Resume topics persisted across process restarts (START_STICKY).
        Set<String> resumed = persistedTopics();
        if (!resumed.isEmpty()) {
            ensureForeground();
            for (String topic : resumed) startWorker(topic);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && intent.getAction() != null) {
            String topic = intent.getStringExtra(EXTRA_TOPIC);
            switch (intent.getAction()) {
                case ACTION_ADD:
                    if (isValidTopic(topic)) {
                        ensureForeground();
                        startWorker(topic);
                    }
                    break;
                case ACTION_REMOVE:
                    if (topic != null) stopWorker(topic);
                    break;
                case ACTION_STOP_ALL:
                    for (String t : workers.keySet()) stopWorker(t);
                    break;
                default:
                    break;
            }
        }
        if (workers.isEmpty()) {
            stopSelf();
            return START_NOT_STICKY;
        }
        updateForegroundNotification();
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        destroyed = true;
        for (TopicWorker w : workers.values()) w.shutdown();
        workers.clear();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    /**
     * Android 15+: dataSync foreground services have a ~6h/24h execution cap.
     * The system calls this a few seconds before killing the service — stop
     * promptly and tell the user to relaunch monitoring from the app.
     */
    @Override
    public void onTimeout(int startId) {
        fireTimeoutNotice();
        stopSelf();
    }

    // ------------------------------------------------------------- worker mgmt

    private void startWorker(String topic) {
        workers.computeIfAbsent(topic, t -> {
            TopicWorker w = new TopicWorker(t);
            Thread th = new Thread(w, "queue-monitor-" + t);
            th.setDaemon(true);
            w.thread = th;
            th.start();
            return w;
        });
        persistTopics();
    }

    private void stopWorker(String topic) {
        TopicWorker w = workers.remove(topic);
        if (w != null) w.shutdown();
        persistTopics();
        onWorkerCountChanged();
    }

    private void onWorkerCountChanged() {
        if (workers.isEmpty()) {
            clearPersistedTopics();
            stopSelf();
        } else {
            updateForegroundNotification();
        }
    }

    private static boolean isValidTopic(String topic) {
        return topic != null && topic.matches("[A-Za-z0-9_-]{1,64}");
    }

    // ------------------------------------------------------------------ SSE

    private void readStream(HttpURLConnection conn, TopicWorker w) throws Exception {
        BufferedReader r = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
        String line;
        String eventName = null;
        StringBuilder data = new StringBuilder();
        while (!destroyed && workers.get(w.topic) == w) {
            // Blocks; throws SocketTimeoutException after IDLE_RECONNECT_MS of
            // silence, which the caller treats as a reconnect trigger.
            line = r.readLine();
            if (line == null) throw new java.io.EOFException("SSE stream closed by server");
            if (line.isEmpty()) {
                dispatchEvent(w, eventName, data.toString());
                eventName = null;
                data.setLength(0);
            } else if (line.charAt(0) == ':') {
                // comment / keepalive — ignore
            } else if (line.startsWith("event:")) {
                eventName = line.substring(6).trim();
            } else if (line.startsWith("data:")) {
                if (data.length() > 0) data.append('\n');
                data.append(line.substring(5).trim());
            }
        }
    }

    private void dispatchEvent(TopicWorker w, String eventName, String data) {
        if (data == null || data.isEmpty()) return;
        if (eventName != null && !"message".equals(eventName)) return; // "open", "keepalive", ...
        try {
            JSONObject outer = new JSONObject(data);
            if (!"message".equals(outer.optString("event"))) return;
            String id = outer.optString("id", null);
            if (id != null && !id.isEmpty()) w.lastId = id; // resume cursor
            String inner = outer.optString("message", null);
            if (inner == null || inner.isEmpty()) return;
            JSONObject ev = new JSONObject(inner);
            String phase = ev.optString("phase", "");
            if ("ready".equals(phase) || "serving".equals(phase)) {
                fireReadyNotification(w.topic, ev.optString("model", ""), ev.optString("endpoint", ""));
                stopWorker(w.topic); // mission accomplished
            } else if ("failed".equals(phase) || "stopped".equals(phase) || "auto-shutdown".equals(phase)) {
                stopWorker(w.topic); // terminal — nothing left to watch
            }
        } catch (Exception ignored) {
            // Malformed line — skip, keep the stream alive.
        }
    }

    // ------------------------------------------------------------ notifications

    private void createChannels() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm == null) return;
        NotificationChannel queue = new NotificationChannel(
                CHANNEL_QUEUE, "Monitor de cola TPU", NotificationManager.IMPORTANCE_LOW);
        queue.setDescription("Estado persistente mientras se espera una TPU");
        NotificationChannel ready = new NotificationChannel(
                CHANNEL_READY, "Alertas de TPU lista", NotificationManager.IMPORTANCE_HIGH);
        ready.setDescription("Avisa en cuanto el kernel TPU queda listo");
        ready.enableVibration(true);
        nm.createNotificationChannel(queue);
        nm.createNotificationChannel(ready);
    }

    private void ensureForeground() {
        Notification n = buildQueueNotification();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIF_FOREGROUND_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(NOTIF_FOREGROUND_ID, n);
        }
    }

    private Notification buildQueueNotification() {
        int count = Math.max(1, workers.size());
        return new NotificationCompat.Builder(this, CHANNEL_QUEUE)
                .setSmallIcon(getApplicationInfo().icon)
                .setContentTitle("Kaggle TPU Lab")
                .setContentText("Siguiendo " + count + " cola" + (count > 1 ? "s" : "") + " de TPU en segundo plano…")
                .setOngoing(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();
    }

    private void updateForegroundNotification() {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null && !workers.isEmpty()) {
            nm.notify(NOTIF_FOREGROUND_ID, buildQueueNotification());
        }
    }

    private PendingIntent openAppIntent() {
        Intent i = new Intent(this, MainActivity.class);
        i.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        return PendingIntent.getActivity(this, 0, i,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
    }

    private void fireReadyNotification(String topic, String model, String endpoint) {
        StringBuilder body = new StringBuilder();
        if (!model.isEmpty()) body.append("Modelo ").append(model).append(" está en línea. ");
        if (!endpoint.isEmpty()) body.append(endpoint);
        if (body.length() == 0) body.append("Tocá para abrir la app.");
        Notification n = new NotificationCompat.Builder(this, CHANNEL_READY)
                .setSmallIcon(getApplicationInfo().icon)
                .setContentTitle("\uD83C\uDF89 ¡TPU lista!")
                .setContentText(body.toString())
                .setStyle(new NotificationCompat.BigTextStyle().bigText(body.toString()))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_MESSAGE)
                .setAutoCancel(true)
                .setDefaults(NotificationCompat.DEFAULT_ALL)
                .setContentIntent(openAppIntent())
                .build();
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(NOTIF_READY_ID_BASE + (topic.hashCode() & 0xff), n);
    }

    private void fireTimeoutNotice() {
        Notification n = new NotificationCompat.Builder(this, CHANNEL_READY)
                .setSmallIcon(getApplicationInfo().icon)
                .setContentTitle("Monitoreo de cola pausado")
                .setContentText("Android detuvo el monitoreo en segundo plano por límite del sistema. Abrí la app para reanudar.")
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .setContentIntent(openAppIntent())
                .build();
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(NOTIF_READY_ID_BASE + 0xfe, n);
    }

    // --------------------------------------------------------------- persistence

    private void persistTopics() {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putStringSet(KEY_TOPICS, new HashSet<>(workers.keySet()))
                .apply();
    }

    private Set<String> persistedTopics() {
        SharedPreferences p = getSharedPreferences(PREFS, MODE_PRIVATE);
        Set<String> s = p.getStringSet(KEY_TOPICS, null);
        if (s == null) return new HashSet<>();
        Set<String> out = new HashSet<>();
        for (String t : s) if (isValidTopic(t)) out.add(t);
        return out;
    }

    private void clearPersistedTopics() {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(KEY_TOPICS).apply();
    }
}
