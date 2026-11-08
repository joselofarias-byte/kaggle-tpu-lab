import { KaggleApi } from './kaggle';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { decideEndpointMount, hasRecentServingEvidence, InstanceManager } from './racer';

const base = {
  status: 'RUNNING' as const,
  phase: 'ready' as const,
  sessionEndpointUrl: undefined as string | undefined,
  eventEndpoint: undefined as string | undefined,
  existingBaseUrl: undefined as string | undefined,
};

describe('decideEndpointMount', () => {
  it('mounts a ready event with a valid https URL', () => {
    const v = decideEndpointMount({
      ...base,
      eventEndpoint: 'https://example.trycloudflare.com',
    });
    expect(v).toEqual({ kind: 'mount', url: 'https://example.trycloudflare.com/v1' });
  });

  it('does not double-append /v1', () => {
    const v = decideEndpointMount({
      ...base,
      phase: 'serving',
      sessionEndpointUrl: 'https://example.trycloudflare.com/v1',
    });
    expect(v).toEqual({ kind: 'mount', url: 'https://example.trycloudflare.com/v1' });
  });

  it('accepts http URLs and heartbeat phase', () => {
    const v = decideEndpointMount({
      ...base,
      phase: 'heartbeat',
      eventEndpoint: 'http://10.0.0.2:8000/',
    });
    expect(v.kind).toBe('mount');
    if (v.kind === 'mount') expect(v.url).toBe('http://10.0.0.2:8000/v1');
  });

  it('mounts session-specific READY even if Kaggle still reports QUEUED', () => {
    const v = decideEndpointMount({
      ...base,
      status: 'QUEUED',
      eventEndpoint: 'https://example.trycloudflare.com',
    });
    expect(v).toEqual({ kind: 'mount', url: 'https://example.trycloudflare.com/v1' });
  });

  it('mounts session-specific heartbeat even if Kaggle still reports QUEUED', () => {
    const v = decideEndpointMount({
      ...base,
      status: 'QUEUED',
      phase: 'heartbeat',
      eventEndpoint: 'https://example.trycloudflare.com/v1',
    });
    expect(v).toEqual({ kind: 'mount', url: 'https://example.trycloudflare.com/v1' });
  });

  it('skips tunnel-url phase (URL recorded, never marked READY)', () => {
    const v = decideEndpointMount({
      ...base,
      phase: 'tunnel-url',
      eventEndpoint: 'https://example.trycloudflare.com',
    });
    expect(v).toEqual({ kind: 'skip' });
  });

  it('skips non-http(s) URLs', () => {
    const v = decideEndpointMount({ ...base, eventEndpoint: 'ftp://example.com/x' });
    expect(v).toEqual({ kind: 'skip' });
  });

  it('skips when there is no URL at all', () => {
    expect(decideEndpointMount({ ...base })).toEqual({ kind: 'skip' });
  });

  it('prefers sessionEndpointUrl, then eventEndpoint, then existingBaseUrl', () => {
    const first = decideEndpointMount({
      ...base,
      sessionEndpointUrl: 'https://one.example.com',
      eventEndpoint: 'https://two.example.com',
      existingBaseUrl: 'https://three.example.com',
    });
    expect(first).toEqual({ kind: 'mount', url: 'https://one.example.com/v1' });

    const second = decideEndpointMount({
      ...base,
      eventEndpoint: 'https://two.example.com',
      existingBaseUrl: 'https://three.example.com',
    });
    expect(second).toEqual({ kind: 'mount', url: 'https://two.example.com/v1' });

    const third = decideEndpointMount({
      ...base,
      existingBaseUrl: 'https://three.example.com',
    });
    expect(third).toEqual({ kind: 'mount', url: 'https://three.example.com/v1' });
  });

  it('trims whitespace around the URL', () => {
    const v = decideEndpointMount({
      ...base,
      eventEndpoint: '  https://example.trycloudflare.com  \n',
    });
    expect(v).toEqual({ kind: 'mount', url: 'https://example.trycloudflare.com/v1' });
  });
});


describe('hasRecentServingEvidence', () => {
  const now = 1_800_000_000_000;

  it('accepts a recent heartbeat', () => {
    expect(hasRecentServingEvidence(
      { lastEvent: { phase: 'heartbeat', rawTime: Math.floor(now / 1000) - 60 } as any },
      now,
      900
    )).toBe(true);
  });

  it('accepts recent ready/serving events', () => {
    expect(hasRecentServingEvidence(
      { lastEvent: { phase: 'ready', rawTime: Math.floor(now / 1000) - 120 } as any },
      now,
      900
    )).toBe(true);
    expect(hasRecentServingEvidence(
      { lastEvent: { phase: 'serving', rawTime: Math.floor(now / 1000) - 120 } as any },
      now,
      900
    )).toBe(true);
  });

  it('rejects stale or terminal evidence', () => {
    expect(hasRecentServingEvidence(
      { lastEvent: { phase: 'heartbeat', rawTime: Math.floor(now / 1000) - 901 } as any },
      now,
      900
    )).toBe(false);
    expect(hasRecentServingEvidence(
      { lastEvent: { phase: 'stopped', rawTime: Math.floor(now / 1000) - 10 } as any },
      now,
      900
    )).toBe(false);
  });
});


describe('status reconciliation', () => {
  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

  function setup() {
    const manager = new InstanceManager(vi.fn());
    const session: any = {
      account: 'test', username: 'demo', token: 'test-only', slug: 'qwen38-tpu-serve',
      topic: '', status: 'QUEUED', done: false, events: [], seenBoot: false,
    };
    (manager as any).sessions.set('test', session);
    return { manager, session };
  }

  it('marks a failed poll unknown, keeps monitoring, then recovers to queued', async () => {
    const { manager, session } = setup();
    const getStatus = vi.spyOn(KaggleApi.prototype, 'getStatus').mockResolvedValue({ status: 'UNKNOWN' });
    await (manager as any).monitorLoop();
    expect(session.status).toBe('UNKNOWN');
    expect(session.done).toBe(false);
    expect(manager.getAllSessions()).toHaveLength(1);
    getStatus.mockResolvedValue({ status: 'QUEUED' });
    await (manager as any).monitorLoop();
    expect(session.status).toBe('QUEUED');
    expect(session.done).toBe(false);
  });

  it('preserves a recent session-specific serving confirmation', async () => {
    const { manager, session } = setup();
    session.status = 'RUNNING';
    session.lastEvent = { phase: 'heartbeat', rawTime: Date.now() / 1000 };
    vi.spyOn(KaggleApi.prototype, 'getStatus').mockResolvedValue({ status: 'UNKNOWN' });
    await (manager as any).monitorLoop();
    expect(session.status).toBe('RUNNING');
    expect(session.done).toBe(false);
  });

  it('manual refresh also marks an unconfirmed queued session unknown', async () => {
    vi.useFakeTimers();
    const { manager, session } = setup();
    vi.spyOn(KaggleApi.prototype, 'getStatus').mockResolvedValue({ status: 'UNKNOWN' });
    await manager.refreshAccount({ id: 'test', username: 'demo', token: 'test-only' } as any);
    expect(session.status).toBe('UNKNOWN');
    expect(session.done).toBe(false);
    vi.clearAllTimers();
  });
});
