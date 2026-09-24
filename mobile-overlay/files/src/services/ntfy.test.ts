import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { NtfyListener } from './ntfy';
import { NtfyEvent } from './types';

const nowSec = () => Math.floor(Date.now() / 1000);

function ntfyLine(id: string, phase: string, extra: Record<string, unknown> = {}, time?: number): string {
  return JSON.stringify({
    id,
    event: 'message',
    message: JSON.stringify({ phase, ...extra }),
    time: time ?? nowSec(),
    topic: 'ktl-test',
  });
}

function stubFetch(lines: string[]) {
  const seen: string[] = [];
  (globalThis as any).fetch = vi.fn(async (url: string) => {
    seen.push(url);
    return { ok: true, text: async () => lines.join('\n') };
  });
  return seen;
}

/** Drive pollOnce without starting the background loop. */
async function pollOnce(listener: NtfyListener): Promise<NtfyEvent[]> {
  (listener as unknown as { isPolling: boolean }).isPolling = true;
  try {
    return await listener.pollOnce();
  } finally {
    (listener as unknown as { isPolling: boolean }).isPolling = false;
  }
}

describe('NtfyListener.pollOnce', () => {
  beforeEach(() => {
    // notifyApp() touches `window` on non-native platforms.
    (globalThis as any).window = {};
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    delete (globalThis as any).window;
  });

  it('formats known phases via the phase map', async () => {
    const seen: Array<[NtfyEvent, string]> = [];
    const listener = new NtfyListener('ktl-test', (ev, text) => seen.push([ev, text]));
    stubFetch([
      ntfyLine('a1', 'install'),
      ntfyLine('a2', 'weights-download'),
      ntfyLine('a3', 'tunnel-url', { endpoint: 'https://x.trycloudflare.com' }),
    ]);

    await pollOnce(listener);

    expect(seen).toHaveLength(3);
    expect(seen[0][1]).toContain('entorno Python');
    expect(seen[1][1]).toContain('pesos del modelo');
    expect(seen[2][1]).toContain('https://x.trycloudflare.com');
  });

  it('formats serving/ready/heartbeat/benchmark specially', async () => {
    const texts: string[] = [];
    const listener = new NtfyListener('ktl-test', (_ev, text) => texts.push(text));
    stubFetch([
      ntfyLine('b1', 'ready', { endpoint: 'https://x.trycloudflare.com' }),
      ntfyLine('b2', 'heartbeat', { up_min: 42 }),
      ntfyLine('b3', 'benchmark', { decode_tok_s: 113.9 }),
    ]);

    await pollOnce(listener);

    expect(texts[0]).toContain('Servicio listo');
    expect(texts[1]).toContain('42 min');
    expect(texts[2]).toContain('113.9 tok/s');
  });

  it('falls back to [phase] dump for unknown phases', async () => {
    const texts: string[] = [];
    const listener = new NtfyListener('ktl-test', (_ev, text) => texts.push(text));
    stubFetch([ntfyLine('c1', 'mystery-phase', { foo: 1 })]);

    await pollOnce(listener);

    expect(texts[0]).toMatch(/^\[mystery-phase\]/);
  });

  it('deduplicates messages by raw id', async () => {
    const events: NtfyEvent[] = [];
    const listener = new NtfyListener('ktl-test', (ev) => events.push(ev));
    stubFetch([
      ntfyLine('dup', 'install'),
      ntfyLine('dup', 'install'),
      ntfyLine('other', 'install'),
    ]);

    await pollOnce(listener);

    expect(events).toHaveLength(2);
  });

  it('ignores non-message events, empty lines and garbage', async () => {
    const events: NtfyEvent[] = [];
    const listener = new NtfyListener('ktl-test', (ev) => events.push(ev));
    stubFetch([
      '',
      'not json at all',
      JSON.stringify({ id: 'x1', event: 'open' }),
      JSON.stringify({ id: 'x2', event: 'message' }),
      ntfyLine('x3', 'installed'),
    ]);

    await pollOnce(listener);

    expect(events).toHaveLength(1);
    expect(events[0].phase).toBe('installed');
  });

  it('returns [] when the response is not ok', async () => {
    (globalThis as any).fetch = vi.fn(async () => ({ ok: false }));
    const listener = new NtfyListener('ktl-test', () => {});
    expect(await pollOnce(listener)).toEqual([]);
  });
});
