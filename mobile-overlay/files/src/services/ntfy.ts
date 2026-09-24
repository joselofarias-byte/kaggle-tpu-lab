import { NtfyEvent } from './types';
import { Capacitor } from '@capacitor/core';
import { LocalNotifications } from '@capacitor/local-notifications';

export class NtfyListener {
  private topic: string;
  private since: number;
  private isPolling: boolean = false;
  private pollTimer: any = null;
  private onEventCb: (event: NtfyEvent, formattedText: string) => void;
  private seenIds: Set<string> = new Set();
  private hasNotifiedReady: boolean = false;

  constructor(
    topic: string,
    onEvent: (event: NtfyEvent, formattedText: string) => void,
    sinceSecondsAgo: number = 600
  ) {
    this.topic = topic;
    // Align with launch.py: only look back within last 10 min (600s) to prevent ghost ready events from old runs
    this.since = Math.floor(Date.now() / 1000) - sinceSecondsAgo;
    this.onEventCb = onEvent;
  }

  public async start(intervalMs: number = 4000) {
    if (this.isPolling) return;
    this.isPolling = true;
    this.hasNotifiedReady = false;

    // Request notification permissions on Android 13+
    if (Capacitor.isNativePlatform()) {
      try {
        await LocalNotifications.requestPermissions();
      } catch {
        // Permissions optional
      }
    }

    const loop = async () => {
      if (!this.isPolling) return;
      try {
        await this.pollOnce();
      } catch {
        // Safe loop
      }
      if (this.isPolling) {
        this.pollTimer = setTimeout(loop, intervalMs);
      }
    };

    loop();
  }

  public stop() {
    this.isPolling = false;
    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  }

  public async pollOnce(): Promise<NtfyEvent[]> {
    if (!this.isPolling) return [];

    const sinceParam = this.since > 100000000 ? String(this.since) : '24h';
    const url = `https://ntfy.sh/${this.topic}/json?poll=1&since=${sinceParam}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    try {
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!res.ok) return [];
      const text = await res.text();
      const lines = text.split('\n');
      const events: NtfyEvent[] = [];

      for (const line of lines) {
        if (!this.isPolling) break;
        if (!line.trim()) continue;

        try {
          const raw = JSON.parse(line);
          if (raw.event !== 'message' || !raw.message) continue;

          // 1. Message Deduplication by unique raw.id
          if (raw.id) {
            if (this.seenIds.has(raw.id)) continue;
            this.seenIds.add(raw.id);
            if (this.seenIds.size > 500) {
              const oldestId = this.seenIds.values().next().value;
              if (oldestId) this.seenIds.delete(oldestId);
            }
          }

          if (raw.time) {
            this.since = Math.max(this.since, raw.time);
          }

          const evData: NtfyEvent = JSON.parse(raw.message);
          evData.rawTime = raw.time;
          events.push(evData);

          const formatted = this.formatEvent(evData);
          this.onEventCb(evData, formatted);

          // 2. Notify once when ready or serving
          if (
            (evData.phase === 'ready' || evData.phase === 'serving') &&
            !this.hasNotifiedReady
          ) {
            this.hasNotifiedReady = true;
            this.notifyApp(
              '🎉 ¡TPU lista!',
              `El modelo ${evData.display_name || evData.model || 'LLM'} está en línea. Tocá para abrir el chat.`
            );
          }
        } catch {
          // ignore non-json messages
        }
      }
      return events;
    } catch {
      clearTimeout(timeoutId);
      return [];
    }
  }

  private static readonly PHASE_MAP: Record<string, string> = {
    install: 'Preparando el entorno Python para la TPU...',
    installed: 'Entorno Python y dependencias listos.',
    'mtp-patch-applied': 'Parche MTP aplicado.',
    'mtp-patch-failed': 'No se pudo aplicar MTP; se continúa sin decodificación especulativa.',
    'cache-missing': 'No hay caché de compilación; inicio en frío (+10~15 min)...',
    'weights-mounted': 'Pesos del modelo montados; no hace falta descargarlos.',
    'weights-download': 'Descargando los pesos del modelo...',
    'weights-downloaded': 'Pesos del modelo descargados.',
    'server-launch': 'Iniciando el motor de inferencia y cargando el modelo...',
    loading: 'Cargando los pesos en la memoria de la TPU...',
    warmed: 'Calentamiento completado; abriendo el túnel público...',
    'auto-shutdown': 'Tiempo máximo alcanzado; instancia detenida correctamente.',
  };

  private formatEvent(ev: NtfyEvent): string {
    const p = ev.phase;

    if (p === 'stopped' || p === 'failed') {
      const parts: string[] = [];
      if (ev.message_es) parts.push(ev.message_es);
      if (ev.hint_es) parts.push(`Sugerencia: ${ev.hint_es}`);
      else if (ev.hint) parts.push(`Sugerencia: ${ev.hint}`);
      if (ev.cause && ev.cause !== ev.message_es) parts.push(`Causa: ${ev.cause}`);
      if (ev.step) parts.push(`Etapa: ${ev.step}`);
      if (ev.error_code && ev.error_code !== ev.step) parts.push(`Código: ${ev.error_code}`);
      if (parts.length === 0 && ev.tail) parts.push(ev.tail);
      return parts.length > 0 ? `⚠️ ${parts.join(' · ')}` : '⚠️ El servicio se detuvo sin detalle adicional.';
    }

    if (ev.message_es) return ev.message_es;
    if (NtfyListener.PHASE_MAP[p]) return NtfyListener.PHASE_MAP[p];
    if (p === 'cache-restored') {
      return (ev as any).covers_this_config === false
        ? 'Caché XLA cargada, pero esta configuración requiere compilación en frío.'
        : 'Caché XLA restaurada; arranque rápido.';
    }
    if (p === 'loaded') return `Pesos cargados en ${ev.minutes || '?'} min; iniciando calentamiento...`;
    if (p === 'compiling') {
      return ev.what
        ? `Compilación ${ev.what} completada (${ev.secs || 0}s)`
        : `Compilando gráficos XLA... ${Math.floor((ev.elapsed_s || 0) / 60)} min transcurridos`;
    }
    if (p === 'tunnel-url') return `Endpoint reservado: ${ev.endpoint} (todavía no está listo)`;
    if (p === 'serving' || p === 'ready') {
      return ev.endpoint ? `🎉 Servicio listo: ${ev.endpoint}` : '🎉 Motor de inferencia iniciado.';
    }
    if (p === 'benchmark') return `Rendimiento: ${ev.decode_tok_s} tok/s`;
    if (p === 'heartbeat') return `Servicio activo (${ev.up_min || '?'} min)${ev.endpoint ? ` — ${ev.endpoint}` : ''}`;

    return `[${p}] ${JSON.stringify(ev)}`;
  }

  private async notifyApp(title: string, body: string) {
    if (Capacitor.isNativePlatform()) {
      try {
        await LocalNotifications.schedule({
          notifications: [
            {
              title,
              body,
              id: Math.floor(Math.random() * 10000),
              schedule: { at: new Date(Date.now() + 100) },
              sound: 'default',
            },
          ],
        });
      } catch {
        // fallback
      }
    } else if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(title, { body });
    }
  }
}
