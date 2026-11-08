import { Capacitor, CapacitorHttp } from '@capacitor/core';

export interface ModalConnection { url: string; token: string }
export interface ModalProfile {
  id: string; name: string; model: string; gpu: string; cpu: number;
  memory_mib: number; description: string;
}
export interface ModalSession {
  id: string; profile_id: string; model: string; gpu: string; cpu: number; memory_mib: number;
  status: 'STARTING' | 'READY' | 'STOPPING' | 'STOPPED' | 'ERROR' | 'UNKNOWN';
  created_at: string; expires_at: string; idle_minutes: number; max_minutes: number; detail?: string;
}
export interface ModalMessage { role: 'user' | 'assistant' | 'system'; content: string }
export interface ModalStart {
  profile_id: string; max_minutes: number; idle_minutes: number; request_id: string;
}

export function validateModalConnection(connection: ModalConnection): string | null {
  try {
    const url = new URL(connection.url.trim());
    if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) {
      return 'Usa la URL HTTPS del controlador, sin credenciales, parámetros ni fragmentos.';
    }
  } catch { return 'Introduce una URL HTTPS válida para el controlador.'; }
  if (!connection.token.trim() || /\s/.test(connection.token.trim())) return 'Introduce el token del controlador, sin espacios.';
  return null;
}

const invalidResponse = () => new Error('Respuesta del controlador no reconocida. Estado sin confirmar; sincroniza antes de iniciar otra sesión.');
const textField = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const positiveNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0;

export function parseModalSession(value: unknown): ModalSession | null {
  if (value === null) return null;
  const s = value as ModalSession;
  if (!s || !['STARTING', 'READY', 'STOPPING', 'STOPPED', 'ERROR', 'UNKNOWN'].includes(s.status)
    || ![s.id, s.profile_id, s.model, s.gpu].every(textField)
    || ![s.cpu, s.memory_mib, s.max_minutes, s.idle_minutes].every(positiveNumber)
    || typeof s.created_at !== 'string' || !Number.isFinite(Date.parse(s.created_at))
    || typeof s.expires_at !== 'string' || !Number.isFinite(Date.parse(s.expires_at))
    || (s.detail !== undefined && typeof s.detail !== 'string')) throw invalidResponse();
  return s;
}

export class ModalApi {
  private readonly connection: ModalConnection;
  constructor(connection: ModalConnection) {
    const error = validateModalConnection(connection);
    if (error) throw new Error(error);
    this.connection = { url: connection.url.trim().replace(/\/+$/, ''), token: connection.token.trim() };
  }

  private async request(path: string, method = 'GET', data?: unknown, timeout = 30000): Promise<any> {
    const url = `${this.connection.url}/api/${path}`;
    const headers = { Authorization: `Bearer ${this.connection.token}`, 'Content-Type': 'application/json' };
    let status: number;
    let body: any;
    try {
      if (Capacitor.isNativePlatform()) {
        const res = await CapacitorHttp.request({ url, method, headers, data, connectTimeout: 15000, readTimeout: timeout, disableRedirects: true });
        status = res.status;
        body = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
      } else {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
          const res = await fetch(url, { method, headers, body: data === undefined ? undefined : JSON.stringify(data), signal: controller.signal, redirect: 'error', credentials: 'omit' });
          status = res.status;
          body = await res.json();
        } finally { clearTimeout(timer); }
      }
    } catch {
      throw new Error('No se pudo confirmar la respuesta de Modal. La sesión puede seguir consumiendo recursos; sincroniza o revisa el panel de Modal.');
    }
    if (status === 401 || status === 403) throw new Error('Acceso denegado. Revisa el token del controlador.');
    if (status === 409) throw new Error('Ya existe una sesión o hay una operación en curso. Sincroniza antes de reintentar.');
    if (status < 200 || status >= 300) throw new Error(`El controlador respondió HTTP ${status}. No se confirmó la operación; sincroniza antes de reintentar.`);
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw invalidResponse();
    return body;
  }

  async profiles(): Promise<ModalProfile[]> {
    const body = await this.request('profiles');
    if (!Array.isArray(body.profiles) || !body.profiles.length || !body.profiles.every((p: ModalProfile) =>
      p && [p.id, p.name, p.model, p.gpu].every(textField) && typeof p.description === 'string'
      && positiveNumber(p.cpu) && positiveNumber(p.memory_mib))) throw invalidResponse();
    return body.profiles;
  }
  async session(): Promise<ModalSession | null> { return parseModalSession((await this.request('session')).session); }
  async start(options: ModalStart): Promise<ModalSession> {
    if (!Number.isInteger(options.max_minutes) || options.max_minutes < 5 || options.max_minutes > 120
      || !Number.isInteger(options.idle_minutes) || options.idle_minutes < 2 || options.idle_minutes > 30
      || options.idle_minutes > options.max_minutes || !options.profile_id
      || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(options.request_id)) {
      throw new Error('Revisa el perfil y los límites de ejecución e inactividad.');
    }
    const session = parseModalSession((await this.request('session', 'POST', options, 120000)).session);
    if (!session) throw invalidResponse();
    return session;
  }
  async stop(): Promise<ModalSession | null> { return parseModalSession((await this.request('session', 'DELETE', undefined, 60000)).session); }
  async logs(): Promise<string[]> {
    const body = await this.request('logs');
    if (!Array.isArray(body.lines) || !body.lines.every((line: unknown) => typeof line === 'string')) throw invalidResponse();
    return body.lines.slice(-200).map((line: string) => line.slice(0, 2000));
  }
  async chat(messages: ModalMessage[]): Promise<string> {
    const body = await this.request('chat', 'POST', { messages, max_tokens: 512 }, 120000);
    const content = body.choices?.[0]?.message?.content;
    if (typeof content !== 'string') throw invalidResponse();
    return content;
  }
}
