// ponytail: minimal official Kaggle API client, direct protobuf JSON endpoints without cookie leakage
import { Capacitor, CapacitorCookies, CapacitorHttp, HttpResponse } from '@capacitor/core';

export interface KaggleStatusResponse {
  status: 'QUEUED' | 'RUNNING' | 'COMPLETE' | 'ERROR' | 'CANCELLED' | 'UNKNOWN';
  failureMessage?: string;
  hasFailure?: boolean;
  isNetworkError?: boolean;
  httpStatus?: number;
  rawStatus?: string | number;
  detail?: string;
}

/**
 * Conservative parser for Kaggle worker state.
 *
 * IMPORTANT: unknown/unrecognized payloads are UNKNOWN, never CANCELLED.
 * A false terminal state makes the Android controller stop monitoring while
 * the Kaggle job may keep consuming accelerator quota.
 *
 * Kaggle's current SDK enum includes:
 * 0 QUEUED, 1 RUNNING, 2 COMPLETE, 3 ERROR,
 * 4 CANCEL_REQUESTED, 5 CANCEL_ACKNOWLEDGED, 6 NEW_SCRIPT.
 */
export function parseKaggleStatusPayload(rawInput: any): KaggleStatusResponse {
  let raw: any = rawInput;
  if (typeof raw === 'string') {
    try { raw = JSON.parse(raw); } catch {
      return { status: 'UNKNOWN', detail: 'Respuesta de estado no JSON' };
    }
  }
  if (!raw || typeof raw !== 'object') {
    return { status: 'UNKNOWN', detail: 'Respuesta de estado vacía o inválida' };
  }

  const statusRaw = raw.status !== undefined ? raw.status : raw.workerStatus;
  const s = String(statusRaw ?? '').trim().toUpperCase();

  let status: KaggleStatusResponse['status'] = 'UNKNOWN';
  if (s.includes('RUNNING') || statusRaw === 1) status = 'RUNNING';
  else if (s.includes('COMPLETE') || statusRaw === 2) status = 'COMPLETE';
  else if (s.includes('ERROR') || statusRaw === 3) status = 'ERROR';
  else if (s.includes('CANCEL') || statusRaw === 4 || statusRaw === 5) status = 'CANCELLED';
  else if (s.includes('QUEUE') || statusRaw === 0) status = 'QUEUED';
  else if (s.includes('NEW_SCRIPT') || statusRaw === 6) status = 'QUEUED';

  return {
    status,
    rawStatus: typeof statusRaw === 'string' || typeof statusRaw === 'number' ? statusRaw : undefined,
    failureMessage: raw.failureMessage || raw.failure_message,
    hasFailure: status === 'ERROR',
    detail: status === 'UNKNOWN' ? 'Kaggle devolvió un estado no reconocido' : undefined,
  };
}

export interface KagglePushResponse {
  ref?: string;
  url?: string;
  error?: string;
}

// ponytail: clean header builder; cleans Bearer/Basic auth, User-Agent, and explicitly strips Cookies
function getAuthHeaders(tokenOrKey: string, username?: string): Record<string, string> {
  const clean = tokenOrKey.trim();
  const baseHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    'User-Agent': 'kaggle-api/v1.7.0',
    'x-cap-user-agent': 'kaggle-api/v1.7.0',
    'Cookie': '', // ponytail: explicitly override Android native CookieManager to prevent 401 Unauthenticated
  };
  if (clean.startsWith('KGAT_') || clean.startsWith('ey')) {
    return { ...baseHeaders, Authorization: `Bearer ${clean}` };
  }
  const authStr = username ? `${username}:${clean}` : clean;
  return { ...baseHeaders, Authorization: `Basic ${btoa(authStr)}` };
}

// ponytail: wipe web cookies to prevent 401 Unauthenticated on api.kaggle.com
export async function clearKaggleCookies(): Promise<void> {
  try {
    if (typeof document !== 'undefined') {
      document.cookie.split(';').forEach((c) => {
        document.cookie = c.replace(/^ +/, '').replace(/=.*/, `=;expires=${new Date().toUTCString()};path=/`);
      });
    }
  } catch {}
  if (Capacitor.isNativePlatform()) {
    try {
      await CapacitorCookies.clearAllCookies();
    } catch {}
  }
}

async function request(
  url: string,
  options: {
    method?: string;
    headers?: Record<string, string>;
    data?: any;
    params?: Record<string, string>;
  }
): Promise<{ status: number; data: any }> {
  const reqHeaders: Record<string, string> = { ...(options.headers || {}) };
  if (url.includes('kaggle.com')) {
    reqHeaders['Cookie'] = '';
  }

  if (Capacitor.isNativePlatform()) {
    const res: HttpResponse = await CapacitorHttp.request({
      url,
      method: options.method || 'GET',
      headers: reqHeaders,
      data: options.data,
      params: options.params,
    });
    return { status: res.status, data: res.data };
  }
  let reqUrl = url;
  if (options.params) {
    reqUrl += `?${new URLSearchParams(options.params).toString()}`;
  }
  const res = await fetch(reqUrl, {
    method: options.method || 'GET',
    headers: reqHeaders,
    body: options.data ? JSON.stringify(options.data) : undefined,
  });
  let data: any;
  try {
    data = await res.json();
  } catch {
    data = await res.text();
  }
  return { status: res.status, data };
}

export class KaggleApi {
  private token: string;
  private username: string;

  constructor(token: string, username: string = '') {
    this.token = token.trim();
    this.username = username.trim();
  }

  private async ensureUsername(): Promise<string> {
    if (this.username && this.username !== 'user' && this.username !== 'kaggle-user') {
      return this.username;
    }
    try {
      const auth = await this.verifyAuth();
      if (auth.isValid && auth.username && auth.username !== 'kaggle-user') {
        this.username = auth.username;
      }
    } catch {}
    return this.username || 'user';
  }

  async verifyAuth(): Promise<{ isValid: boolean; username: string; error?: string }> {
    try {
      const res = await request('https://api.kaggle.com/v1/kernels/list', {
        headers: getAuthHeaders(this.token, this.username),
        params: { group: 'profile', pageSize: '1' },
      });
      if (res.status === 200) {
        let detected = this.username;
        let list: any = res.data;
        if (typeof list === 'string') {
          try { list = JSON.parse(list); } catch {}
        }
        if (Array.isArray(list) && list.length > 0 && list[0]?.ref) {
          const owner = list[0].ref.split('/')[0];
          if (owner) detected = owner;
        }
        return { isValid: true, username: detected || this.username || 'kaggle-user' };
      }
      return { isValid: false, username: this.username, error: `HTTP ${res.status}` };
    } catch (err: any) {
      return { isValid: false, username: this.username, error: err.message || String(err) };
    }
  }

  async pushKernel(slug: string, code: string, datasets: string[]): Promise<KagglePushResponse> {
    await this.ensureUsername();
    const kernelSlug = `${this.username}/${slug}`;
    const res = await request('https://api.kaggle.com/v1/kernels/push', {
      method: 'POST',
      headers: getAuthHeaders(this.token, this.username),
      data: {
        slug: kernelSlug,
        newTitle: slug,
        text: code,
        language: 'python',
        kernelType: 'script',
        isPrivate: true,
        enableGpu: false,
        enableTpu: true,
        enableInternet: true,
        datasetDataSources: datasets,
      },
    });

    const data = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
    if (data?.error || data?.errorNullable) {
      const msg = String(data.error || data.errorNullable);
      if (msg.includes('already') || msg.includes('Conflict') || msg.includes('Maximum batch TPU')) {
        return { ref: kernelSlug, url: `https://www.kaggle.com/code/${kernelSlug}`, error: `Existing task running, adopted (${msg})` };
      }
      throw new Error(msg);
    }
    return { ref: kernelSlug, url: `https://www.kaggle.com/code/${kernelSlug}` };
  }

  async getKernel(slug: string): Promise<{ ntfyTopic?: string; apiKey?: string } | null> {
    await this.ensureUsername();
    try {
      const res = await request('https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel', {
        method: 'POST',
        headers: getAuthHeaders(this.token, this.username),
        data: { userName: this.username, kernelSlug: slug },
      });
      if (res.status === 200 && res.data) {
        let rawStr = '';
        if (typeof res.data === 'string') {
          rawStr = res.data;
        } else {
          try {
            rawStr = JSON.stringify(res.data);
          } catch {
            rawStr = String(res.data);
          }
        }
        const topicMatch = rawStr.match(/["']?ntfy_topic["']?\s*:\s*["'](ktl-[a-zA-Z0-9_-]+)["']/)
          || rawStr.match(/(ktl-[a-zA-Z0-9]{16,32})/);
        const keyMatch = rawStr.match(/["']?api_key["']?\s*:\s*["']((?:sk-|glm-)[a-zA-Z0-9_-]+)["']/)
          || rawStr.match(/((?:sk-|glm-)[a-zA-Z0-9]{16,48})/);

        if (topicMatch || keyMatch) {
          return {
            ntfyTopic: topicMatch ? topicMatch[1] : undefined,
            apiKey: keyMatch ? keyMatch[1] : undefined,
          };
        }
      }
    } catch (err) {
      console.error(`[KaggleApi] getKernel failed for ${this.username}/${slug}:`, err);
    }
    return null;
  }

  async getStatus(slug: string): Promise<KaggleStatusResponse> {
    await this.ensureUsername();
    try {
      const res = await request('https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus', {
        method: 'POST',
        headers: getAuthHeaders(this.token, this.username),
        data: { userName: this.username, kernelSlug: slug },
      });

      if (res.status === 200) {
        return parseKaggleStatusPayload(res.data);
      }

      // Do not turn an HTTP/API problem into a fake terminal state.
      // The kernel may still be alive and consuming TPU quota.
      const detail = typeof res.data === 'string'
        ? res.data.slice(0, 240)
        : (() => {
            try { return JSON.stringify(res.data).slice(0, 240); } catch { return ''; }
          })();
      console.warn(`[KaggleApi] getStatus HTTP ${res.status} for ${this.username}/${slug}: ${detail}`);
      return {
        status: 'UNKNOWN',
        httpStatus: res.status,
        detail: detail || `HTTP ${res.status}`,
      };
    } catch (err: any) {
      const msg = String(err?.message || err || '');
      const low = msg.toLowerCase();
      const isNet =
        low.includes('resolve host') ||
        low.includes('network') ||
        low.includes('enotfound') ||
        low.includes('timeout') ||
        low.includes('failed to fetch') ||
        low.includes('connection');
      console.error(`[KaggleApi] getStatus failed for ${this.username}/${slug}: ${msg.slice(0, 120)}`);
      return {
        status: 'UNKNOWN',
        isNetworkError: isNet,
        detail: msg.slice(0, 240),
      };
    }
  }

  async cancelKernel(slug: string): Promise<boolean> {
    try {
      const res = await request('https://api.kaggle.com/v1/kernels.KernelsApiService/DeleteKernel', {
        method: 'POST',
        headers: getAuthHeaders(this.token, this.username),
        data: { userName: this.username, kernelSlug: slug },
      });
      return res.status === 200 || res.status === 204;
    } catch {
      return false;
    }
  }
}
