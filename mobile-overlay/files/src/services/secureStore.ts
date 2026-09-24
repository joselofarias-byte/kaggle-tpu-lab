/**
 * Encrypted credential storage.
 *
 * Secrets (Kaggle API tokens in `ktl_accounts`, kernel API keys in
 * `ktl_sessions`) used to live in Capacitor Preferences / localStorage as
 * plaintext. They now go through `secretStore`:
 *
 * - On native (Android/iOS): Android Keystore via
 *   `plugin nativo SecureStorage`, with the legacy plaintext store
 *   kept as a durability anchor.
 * - On web: there is no Keystore; the legacy plaintext store is used.
 *   Documented limitation — the shipped app is native-only.
 *
 * Durability policy (learned the hard way, v1.1.0–v1.1.2):
 * some devices accept a Keystore write (the call resolves) but cannot read
 * it back after the process restarts — the failure is silent, not a throw,
 * so "try secure, fall back on error" is not enough. Therefore:
 *
 * - Every `set()` writes the legacy plaintext store FIRST (proven durable),
 *   then best-effort mirrors into the secure backend. A lying or missing
 *   Keystore can never break a save or destroy data.
 * - `get()` prefers the legacy anchor (it is written first and therefore
 *   never staler than the secure copy), falling back to the secure backend.
 * - At boot, for each secret key, the store best-effort heals the secure
 *   copy from legacy and wipes the plaintext ONLY when the secure backend
 *   returns exactly what legacy holds — i.e. the secure copy has proven
 *   itself across a restart. On a healthy device this converges to
 *   secure-only storage; on a device with a broken Keystore the app behaves
 *   like the (working) pre-hardening versions and never loses data.
 */
import { Preferences } from '@capacitor/preferences';
import { Capacitor, registerPlugin } from '@capacitor/core';

/** Storage keys that hold secrets and must never be plaintext on native. */
export const SECRET_KEYS = ['ktl_accounts', 'ktl_sessions'] as const;

export interface KeyValueBackend {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
}

/**
 * Pre-fix behavior: Capacitor Preferences, with a localStorage fallback.
 * Kept as the migration source and as the web / last-resort backend.
 */
export class LegacyPlaintextBackend implements KeyValueBackend {
  private get ls(): Storage | null {
    return typeof localStorage !== 'undefined' ? localStorage : null;
  }

  async get(key: string): Promise<string | null> {
    try {
      const { value } = await Preferences.get({ key });
      if (value !== null && value !== undefined) return value;
    } catch {
      /* fall through to localStorage */
    }
    try {
      return this.ls?.getItem(key) ?? null;
    } catch {
      return null;
    }
  }

  async set(key: string, value: string): Promise<void> {
    try {
      await Preferences.set({ key, value });
    } catch {
      this.ls?.setItem(key, value);
    }
  }

  async remove(key: string): Promise<void> {
    try {
      await Preferences.remove({ key });
    } catch {
      /* ignore */
    }
    try {
      this.ls?.removeItem(key);
    } catch {
      /* ignore */
    }
  }
}

/** Puente a nuestro plugin nativo Android Keystore. */
interface NativeSecureStoragePlugin {
  set(options: { key: string; value: string }): Promise<void>;
  get(options: { key: string }): Promise<{ value: string | null }>;
  remove(options: { key: string }): Promise<void>;
}

const NativeSecureStorage = registerPlugin<NativeSecureStoragePlugin>('SecureStorage');

export class KeystoreBackend implements KeyValueBackend {
  async get(key: string): Promise<string | null> {
    const result = await NativeSecureStorage.get({ key });
    return result.value ?? null;
  }

  async set(key: string, value: string): Promise<void> {
    await NativeSecureStorage.set({ key, value });
  }

  async remove(key: string): Promise<void> {
    await NativeSecureStorage.remove({ key });
  }
}

let warnedInsecure = false;

function warnInsecureOnce() {
  if (!warnedInsecure) {
    warnedInsecure = true;
    console.warn(
      '[secureStore] secure backend unreliable — secrets kept in plaintext Preferences. ' +
        'Run `npx cap sync` and rebuild; if the Keystore itself is broken on this device, plaintext is the safe fallback.',
    );
  }
}

/**
 * Tries the primary (secure) backend, falls back to the legacy plaintext
 * backend on error so the app keeps working even if the native plugin is
 * missing (e.g. `cap sync` not run). Logs a one-time warning.
 */
export class FallbackBackend implements KeyValueBackend {
  constructor(
    private primary: KeyValueBackend,
    private fallback: KeyValueBackend,
  ) {}

  /**
   * The secure backend underneath the fallback wrapper. Migration code uses
   * this (not `set()`) so it can verify a write really landed in secure
   * storage before wiping the plaintext copy — `set()` silently degrades to
   * the legacy backend when the Keystore is unavailable, and wiping after
   * such a degraded write would destroy the only copy of the secret.
   */
  get primaryBackend(): KeyValueBackend {
    return this.primary;
  }

  private warnOnce() {
    warnInsecureOnce();
  }

  async get(key: string): Promise<string | null> {
    try {
      return await this.primary.get(key);
    } catch {
      this.warnOnce();
      return this.fallback.get(key);
    }
  }

  async set(key: string, value: string): Promise<void> {
    try {
      await this.primary.set(key, value);
    } catch {
      this.warnOnce();
      await this.fallback.set(key, value);
    }
  }

  async remove(key: string): Promise<void> {
    try {
      await this.primary.remove(key);
    } catch {
      /* ignore */
    }
    // Belt & braces: never leave a plaintext copy behind.
    await this.fallback.remove(key);
  }
}

export interface SecretStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  remove(key: string): Promise<void>;
  /** Eagerly move all known secret keys out of the legacy plaintext store. */
  migrateLegacy(): Promise<void>;
}

/**
 * Build a secret store from explicit backends (used by unit tests with
 * in-memory backends).
 *
 * Durability contract:
 * - `set()` always writes `legacy` first (the durability anchor), then
 *   best-effort mirrors into `secure`. A secure backend that throws — or
 *   worse, one that accepts writes but silently loses them — can never break
 *   a save or destroy the only copy of a secret.
 * - `get()` prefers `legacy`: it is written before `secure` on every set and
 *   wiped only after the secure copy proves itself, so it is never staler
 *   than the secure copy.
 * - At boot (`ready`), each secret key is reconciled: the secure copy is
 *   best-effort healed from legacy, and the plaintext is wiped ONLY when the
 *   secure backend reads back exactly what legacy holds — proof the secure
 *   copy survived a restart. Every public method awaits `ready`, so nothing
 *   races the wipe.
 */
export function createSecretStore(secure: KeyValueBackend, legacy: KeyValueBackend): SecretStore {
  const sameStore = secure === legacy;

  const ready = (async () => {
    if (sameStore) return;
    for (const key of SECRET_KEYS) {
      try {
        const l = await legacy.get(key).catch(() => null);
        if (l === null || l === undefined) continue;
        // Primero comprobamos si el cifrado YA sobrevivió desde un arranque
        // anterior. Sólo eso autoriza a borrar el ancla plaintext.
        const before = await secure.get(key).catch(() => null);
        if (before === l) {
          await legacy.remove(key).catch(() => {});
          continue;
        }
        // Si todavía no coincide, reparamos la copia cifrada pero conservamos
        // el ancla hasta el próximo arranque. Así un backend que "acepta" el
        // set() y luego pierde la clave no puede destruir la única copia.
        try {
          await secure.set(key, l);
          const check = await secure.get(key).catch(() => null);
          if (check !== l) warnInsecureOnce();
        } catch {
          warnInsecureOnce();
        }
      } catch {
        /* best effort */
      }
    }
  })();

  return {
    async get(key: string): Promise<string | null> {
      await ready;
      try {
        const l = await legacy.get(key);
        if (l !== null && l !== undefined) return l;
      } catch {
        /* fall through to secure */
      }
      if (sameStore) return null;
      try {
        return await secure.get(key);
      } catch {
        return null;
      }
    },
    async set(key: string, value: string): Promise<void> {
      await ready;
      // Durable write first. If this throws there is nothing more we can do.
      await legacy.set(key, value);
      if (sameStore) return;
      // Best-effort secure mirror. Must never break the save.
      try {
        await secure.set(key, value);
      } catch {
        warnInsecureOnce();
      }
    },
    async remove(key: string): Promise<void> {
      await ready;
      try {
        await secure.remove(key);
      } catch {
        /* best-effort */
      }
      if (!sameStore) {
        try {
          await legacy.remove(key);
        } catch {
          /* best-effort */
        }
      }
    },
    async migrateLegacy(): Promise<void> {
      await ready;
    },
  };
}

const legacyBackend = new LegacyPlaintextBackend();

function resolveSecureBackend(): KeyValueBackend {
  if (Capacitor.isNativePlatform()) {
    // NOTE: pass the RAW Keystore backend, not a FallbackBackend wrapper.
    // createSecretStore owns the fallback/durability policy itself, and the
    // boot reconcile tells "secure agrees with legacy" apart from "secure is
    // broken" by reading the secure backend directly. Wrapping it in a
    // FallbackBackend would make secure.get() fall back to legacy on throw,
    // so a broken Keystore would look like agreement and the reconcile would
    // wipe the only copy.
    return new KeystoreBackend();
  }
  // Web: no Keystore exists; documented limitation (shipped app is native).
  return legacyBackend;
}

/** App-wide singleton. Secrets only — non-secret config stays in Preferences. */
export const secretStore: SecretStore = createSecretStore(resolveSecureBackend(), legacyBackend);
