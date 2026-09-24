import { describe, it, expect, vi, afterEach } from 'vitest';
import { modelsUrl, nextProbeDecision, probeEndpoint } from './endpointProbe';

describe('endpoint probe', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('normalizes origins with and without /v1', () => {
    expect(modelsUrl('https://q.example/v1')).toBe('https://q.example/v1/models');
    expect(modelsUrl('https://g.example')).toBe('https://g.example/v1/models');
    expect(modelsUrl('https://g.example/')).toBe('https://g.example/v1/models');
  });

  it('returns true only for HTTP 200 and never throws', async () => {
    (globalThis as any).fetch = vi.fn(async () => ({ status: 200 }));
    await expect(probeEndpoint('https://q.example/v1', 'sk-secret')).resolves.toBe(true);

    (globalThis as any).fetch = vi.fn(async () => {
      throw new Error('network down sk-secret');
    });
    await expect(probeEndpoint('https://q.example/v1', 'sk-secret')).resolves.toBe(false);
  });

  it('marks the endpoint offline only after three misses and restores it later', () => {
    expect(nextProbeDecision('READY', 0, false)).toEqual({ status: 'READY', misses: 1, listed: true });
    expect(nextProbeDecision('READY', 1, false)).toEqual({ status: 'READY', misses: 2, listed: true });
    expect(nextProbeDecision('READY', 2, false)).toEqual({ status: 'OFFLINE', misses: 3, listed: false });
    expect(nextProbeDecision('OFFLINE', 3, true)).toEqual({ status: 'READY', misses: 0, listed: true });
  });
});
