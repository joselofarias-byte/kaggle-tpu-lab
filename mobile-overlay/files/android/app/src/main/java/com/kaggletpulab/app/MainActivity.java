package com.kaggletpulab.app;

import android.os.Bundle;
import android.webkit.CookieManager;
import android.webkit.WebView;
import com.getcapacitor.BridgeActivity;
import java.io.IOException;
import java.net.CookieHandler;
import java.net.URI;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        // Local plugins must be registered before super.onCreate() builds the bridge.
        registerPlugin(QueueMonitorPlugin.class);
        registerPlugin(SecureStoragePlugin.class);
        super.onCreate(savedInstanceState);
        WebView.setWebContentsDebuggingEnabled(true);
        try {
            CookieManager.getInstance().removeAllCookies(null);
            CookieManager.getInstance().flush();
            final CookieHandler originalHandler = CookieHandler.getDefault();
            CookieHandler.setDefault(new CookieHandler() {
                @Override
                public Map<String, List<String>> get(URI uri, Map<String, List<String>> requestHeaders) throws IOException {
                    if (uri != null && uri.getHost() != null && uri.getHost().contains("kaggle.com")) {
                        return Collections.emptyMap();
                    }
                    return originalHandler != null ? originalHandler.get(uri, requestHeaders) : Collections.emptyMap();
                }

                @Override
                public void put(URI uri, Map<String, List<String>> responseHeaders) throws IOException {
                    if (uri != null && uri.getHost() != null && uri.getHost().contains("kaggle.com")) {
                        return;
                    }
                    if (originalHandler != null) {
                        originalHandler.put(uri, responseHeaders);
                    }
                }
            });
        } catch (Exception ignored) {}
    }
}
