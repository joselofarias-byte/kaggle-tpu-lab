package com.kaggletpulab.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.nio.charset.StandardCharsets;
import java.security.KeyStore;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

@CapacitorPlugin(name = "SecureStorage")
public class SecureStoragePlugin extends Plugin {
    private static final String KEY_ALIAS = "KaggleTpuLabSecureStorageV1";
    private static final String PREFS = "ktl_secure_storage_v1";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";

    private SharedPreferences prefs() {
        return getContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    private SecretKey getOrCreateKey() throws Exception {
        KeyStore ks = KeyStore.getInstance("AndroidKeyStore");
        ks.load(null);
        if (ks.containsAlias(KEY_ALIAS)) {
            return ((KeyStore.SecretKeyEntry) ks.getEntry(KEY_ALIAS, null)).getSecretKey();
        }

        KeyGenerator generator = KeyGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
        KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .setUserAuthenticationRequired(false)
                .build();
        generator.init(spec);
        return generator.generateKey();
    }

    @PluginMethod
    public void set(PluginCall call) {
        String key = call.getString("key");
        String value = call.getString("value");
        if (key == null || key.isEmpty() || value == null) {
            call.reject("key y value son obligatorios");
            return;
        }

        try {
            Cipher cipher = Cipher.getInstance(TRANSFORMATION);
            cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey());
            byte[] encrypted = cipher.doFinal(value.getBytes(StandardCharsets.UTF_8));
            String iv = Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP);
            String data = Base64.encodeToString(encrypted, Base64.NO_WRAP);
            boolean committed = prefs().edit().putString(key, iv + ":" + data).commit();
            if (!committed) {
                call.reject("SharedPreferences no confirmó la escritura cifrada");
                return;
            }

            String verify = decrypt(key);
            if (!value.equals(verify)) {
                call.reject("La escritura cifrada no pudo verificarse");
                return;
            }
            call.resolve();
        } catch (Exception e) {
            call.reject("No se pudo cifrar/guardar: " + e.getClass().getSimpleName(), e);
        }
    }

    @PluginMethod
    public void get(PluginCall call) {
        String key = call.getString("key");
        if (key == null || key.isEmpty()) {
            call.reject("key es obligatorio");
            return;
        }
        try {
            JSObject out = new JSObject();
            out.put("value", decrypt(key));
            call.resolve(out);
        } catch (Exception e) {
            call.reject("No se pudo descifrar/leer: " + e.getClass().getSimpleName(), e);
        }
    }

    @PluginMethod
    public void remove(PluginCall call) {
        String key = call.getString("key");
        if (key == null || key.isEmpty()) {
            call.reject("key es obligatorio");
            return;
        }
        prefs().edit().remove(key).commit();
        call.resolve();
    }

    private String decrypt(String key) throws Exception {
        String stored = prefs().getString(key, null);
        if (stored == null) return null;

        int sep = stored.indexOf(':');
        if (sep <= 0 || sep >= stored.length() - 1) {
            throw new IllegalStateException("Formato cifrado inválido");
        }

        byte[] iv = Base64.decode(stored.substring(0, sep), Base64.NO_WRAP);
        byte[] data = Base64.decode(stored.substring(sep + 1), Base64.NO_WRAP);

        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), new GCMParameterSpec(128, iv));
        byte[] clear = cipher.doFinal(data);
        return new String(clear, StandardCharsets.UTF_8);
    }
}
