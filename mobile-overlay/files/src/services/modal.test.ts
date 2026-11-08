import { afterEach, describe, expect, it, vi } from 'vitest';
import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { ModalApi, parseModalSession, validateModalConnection } from './modal';

vi.mock('@capacitor/core', async (importOriginal) => ({
  ...await importOriginal<typeof import('@capacitor/core')>(),
  CapacitorHttp: { request: vi.fn() },
}));

const connection = { url: 'https://controller.example', token: 'test-only' };
const session = { id: 'sb-test', profile_id: 'qwen-small', model: 'Qwen/Qwen2.5-0.5B-Instruct', gpu: 'T4', cpu: 2, memory_mib: 8192, status: 'STARTING', created_at: '2026-11-08T20:00:00Z', expires_at: '2026-11-08T20:30:00Z', idle_minutes: 5, max_minutes: 30 };
const options = { profile_id: 'qwen-small', max_minutes: 30, idle_minutes: 5, request_id: '11111111-1111-4111-8111-111111111111' };
const response = (body: unknown, status = 200) => ({ status, json: async () => body });

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
describe('Modal control transport', () => {
  it('requires HTTPS and keeps credentials out of URLs', () => {
    for (const url of ['http://controller.example', 'https://user:password@controller.example', 'https://controller.example?token=x', 'https://controller.example#x', 'not-a-url']) {
      expect(validateModalConnection({ ...connection, url })).toBeTruthy();
    }
    expect(validateModalConnection({ ...connection, token: '' })).toBeTruthy();
    expect(validateModalConnection(connection)).toBeNull();
  });
  it('does not interpret an invalid/missing session as stopped', async () => {
    const fetch = vi.fn().mockResolvedValue(response({}));
    vi.stubGlobal('fetch', fetch);
    await expect(new ModalApi(connection).session()).rejects.toThrow('Estado sin confirmar');
    for (const value of [{}, { ...session, status: 'BOGUS' }, { ...session, created_at: 'yesterday' }, { ...session, memory_mib: -1 }]) {
      expect(() => parseModalSession(value)).toThrow();
    }
    expect(parseModalSession(null)).toBeNull();
  });
  it('authenticates without cookies, refuses redirects, and never starts during polling', async () => {
    const fetch = vi.fn().mockResolvedValue(response({ session }));
    vi.stubGlobal('fetch', fetch);
    const api = new ModalApi(connection);
    expect((await api.session())?.status).toBe('STARTING');
    expect(fetch).toHaveBeenCalledExactlyOnceWith('https://controller.example/api/session', expect.objectContaining({ method: 'GET', credentials: 'omit', redirect: 'error', headers: expect.objectContaining({ Authorization: 'Bearer test-only' }) }));
  });
  it('validates time limits before a paid start and sends the idempotency key', async () => {
    const fetch = vi.fn().mockResolvedValue(response({ session }));
    vi.stubGlobal('fetch', fetch);
    const api = new ModalApi(connection);
    for (const max_minutes of [0, 121, 5.5]) await expect(api.start({ ...options, max_minutes })).rejects.toThrow();
    await expect(api.start({ ...options, idle_minutes: 31 })).rejects.toThrow();
    await expect(api.start({ ...options, idle_minutes: 10, max_minutes: 5 })).rejects.toThrow();
    expect(fetch).not.toHaveBeenCalled();
    await api.start(options);
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual(options);
  });
  it('does not claim termination on an HTTP or network failure', async () => {
    const fetch = vi.fn().mockResolvedValue(response({ error: 'test-only' }, 503));
    vi.stubGlobal('fetch', fetch);
    const api = new ModalApi(connection);
    await expect(api.stop()).rejects.toThrow('No se confirmó');
    fetch.mockRejectedValue(new Error('private token must not escape'));
    await expect(api.stop()).rejects.toThrow('puede seguir consumiendo recursos');
    await expect(api.session()).rejects.not.toThrow('private token');
  });
  it('uses the native HTTP bridge with bearer authentication and no redirects', async () => {
    vi.spyOn(Capacitor, 'isNativePlatform').mockReturnValue(true);
    const request = vi.spyOn(CapacitorHttp, 'request').mockResolvedValue({ status: 200, data: JSON.stringify({ session }), headers: {}, url: 'https://controller.example/api/session' });
    expect((await new ModalApi(connection).session())?.id).toBe('sb-test');
    expect(request).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({
      url: 'https://controller.example/api/session', method: 'GET', disableRedirects: true,
      headers: expect.objectContaining({ Authorization: 'Bearer test-only' }), readTimeout: 30000,
    }));
  });
  it('rejects malformed profiles rather than offering an unchecked resource', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ profiles: [{ id: 'bad', gpu: 'anything' }] })));
    await expect(new ModalApi(connection).profiles()).rejects.toThrow('Estado sin confirmar');
  });
  it('reads logs and chat only from valid responses', async () => {
    const fetch = vi.fn().mockResolvedValue(response({ lines: ['test log'] }));
    vi.stubGlobal('fetch', fetch);
    const api = new ModalApi(connection);
    expect(await api.logs()).toEqual(['test log']);
    fetch.mockResolvedValue(response({ choices: [{ message: { content: 'Hola' } }] }));
    expect(await api.chat([{ role: 'user', content: 'Hola' }])).toBe('Hola');
    fetch.mockResolvedValue(response({ session: null }));
    expect(await api.stop()).toBeNull();
  });
});
