import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { KaggleAccount, LiveEndpoint, RaceSession } from './types';

/**
 * Simulated controller loop. Kaggle and ntfy are fakes: no session, no TPU.
 * Reproduces the quota bug that survived PR #8: a session-specific READY was
 * mounted, then the next QUEUED poll/refresh deleted the endpoint.
 */
const harness = vi.hoisted(() => ({
  status: 'QUEUED' as string,
  statuses: [] as string[],
  event: undefined as Record<string, unknown> | undefined,
}));

vi.mock('./kaggle', () => ({
  KaggleApi: class {
    async getStatus() {
      harness.statuses.push(harness.status);
      return { status: harness.status };
    }
    async getKernel() {
      return { ntfyTopic: 'topic-live', apiKey: 'sk-live' };
    }
    async cancelKernel() {}
    async pushKernel() {
      return {};
    }
  },
}));

vi.mock('./ntfy', () => ({
  fetchLatestNtfyLifecycleEvent: vi.fn(async () => harness.event),
  NtfyListener: class {
    start() {}
    stop() {}
    async pollOnce() {
      return [];
    }
  },
}));

import { InstanceManager } from './racer';

const managers: InstanceManager[] = [];

const account: KaggleAccount = {
  id: 'acc-1',
  name: 'lab',
  token: 'KGAT_test',
  username: 'labuser',
  isValid: true,
  isChecking: false,
};

function nowSec(): number {
  return Math.floor(Date.now() / 1000);
}

function endpoint(): LiveEndpoint {
  return {
    accountId: 'acc-1',
    accountName: 'lab',
    baseUrl: 'https://live.trycloudflare.com/v1',
    apiKey: 'sk-live',
    model: 'qwen3.8-27b',
    status: 'READY',
    uptimeMinutes: 12,
  };
}

function session(partial: Partial<RaceSession> = {}): RaceSession {
  return {
    account: 'acc-1',
    accountName: 'lab',
    username: 'labuser',
    token: 'KGAT_test',
    kernel: 'labuser/qwen38-tpu-serve',
    slug: 'qwen38-tpu-serve',
    topic: 'topic-live',
    apiKey: 'sk-live',
    model: 'qwen38-27b',
    status: 'QUEUED',
    seenBoot: false,
    done: false,
    events: [],
    ...partial,
  };
}

function createManager(): InstanceManager {
  const mgr = new InstanceManager(() => {});
  managers.push(mgr);
  return mgr;
}

function seed(mgr: InstanceManager, saved: RaceSession): void {
  mgr.restoreSessions([saved]);
  const poll = (mgr as unknown as { pollInterval: ReturnType<typeof setInterval> | null }).pollInterval;
  if (poll) clearInterval(poll);
  (mgr as unknown as { pollInterval: null }).pollInterval = null;
}

function monitor(mgr: InstanceManager): Promise<void> {
  return (mgr as unknown as { monitorLoop: () => Promise<void> }).monitorLoop();
}

beforeEach(() => {
  harness.status = 'QUEUED';
  harness.statuses = [];
  harness.event = undefined;
});

afterEach(() => {
  for (const mgr of managers) {
    const poll = (mgr as unknown as { pollInterval: ReturnType<typeof setInterval> | null }).pollInterval;
    if (poll) clearInterval(poll);
  }
  managers.length = 0;
});

describe('QUEUED + ready wipe regression', () => {
  it('does not wipe an endpoint just mounted from ntfy READY when Kaggle stays QUEUED', async () => {
    const mgr = createManager();
    const saved = session();
    seed(mgr, saved);

    (mgr as unknown as {
      handleNtfyEvent: (s: RaceSession, ev: Record<string, unknown>, text: string, model: string) => void;
    }).handleNtfyEvent(
      saved,
      {
        phase: 'ready',
        endpoint: 'https://live.trycloudflare.com',
        rawTime: nowSec(),
        api_key: 'sk-from-ready',
      },
      'TPU lista',
      'qwen38-27b'
    );

    expect(mgr.getActiveEndpoint()?.baseUrl).toBe('https://live.trycloudflare.com/v1');
    expect(mgr.getSession('acc-1')?.status).toBe('RUNNING');

    await monitor(mgr);
    await monitor(mgr);

    const live = mgr.getSession('acc-1');
    expect(live?.status).toBe('RUNNING');
    expect(live?.endpoint?.baseUrl).toBe('https://live.trycloudflare.com/v1');
    expect(live?.endpoint?.apiKey).toBe('sk-from-ready');
    expect(mgr.getAllEndpoints().map((ep) => ep.baseUrl)).toEqual(['https://live.trycloudflare.com/v1']);
    expect(live?.events.filter((ev) => ev.phase === 'sync-conflict')).toHaveLength(1);
  });

  it('keeps a race winner mounted while heartbeats are fresh and Kaggle says QUEUED', async () => {
    const mgr = createManager();
    const saved = session({
      status: 'WINNER',
      seenBoot: true,
      endpoint: endpoint(),
      endpointUrl: 'https://live.trycloudflare.com/v1',
      lastEvent: { phase: 'heartbeat', rawTime: nowSec() - 20, up_min: 12 },
    });
    seed(mgr, saved);

    await monitor(mgr);

    expect(mgr.getSession('acc-1')?.status).toBe('WINNER');
    expect(mgr.getActiveEndpoint()?.baseUrl).toBe('https://live.trycloudflare.com/v1');
  });

  it('still clears a false READY when the QUEUED report has no fresh serving evidence', async () => {
    const mgr = createManager();
    const saved = session({
      status: 'RUNNING',
      seenBoot: true,
      endpoint: endpoint(),
      endpointUrl: 'https://live.trycloudflare.com/v1',
      lastEvent: { phase: 'tunnel-url', rawTime: nowSec(), endpoint: 'https://live.trycloudflare.com' },
    });
    seed(mgr, saved);

    await monitor(mgr);

    expect(mgr.getAllEndpoints()).toEqual([]);
    expect(mgr.getSession('acc-1')?.endpoint).toBeUndefined();
    expect(mgr.getSession('acc-1')?.status).toBe('QUEUED');
    expect(mgr.getSession('acc-1')?.events.some((ev) => ev.text.includes('Waiting in queue'))).toBe(true);
  });

  it('drops a stale heartbeat and then allows the queue wipe', async () => {
    const mgr = createManager();
    const saved = session({
      status: 'RUNNING',
      seenBoot: true,
      endpoint: endpoint(),
      lastEvent: { phase: 'heartbeat', rawTime: nowSec() - 30 },
    });
    seed(mgr, saved);

    await monitor(mgr);
    expect(mgr.getActiveEndpoint()?.baseUrl).toBe('https://live.trycloudflare.com/v1');

    saved.lastEvent = { phase: 'heartbeat', rawTime: nowSec() - 5000 };
    await monitor(mgr);

    expect(mgr.getAllEndpoints()).toEqual([]);
    expect(mgr.getSession('acc-1')?.status).toBe('QUEUED');
  });

  it('clears a leftover endpoint that is already marked QUEUED and has no fresh evidence', async () => {
    const mgr = createManager();
    const saved = session({
      status: 'QUEUED',
      endpoint: endpoint(),
      lastEvent: { phase: 'ready', rawTime: nowSec() - 5000 },
    });
    seed(mgr, saved);

    await monitor(mgr);

    expect(mgr.getAllEndpoints()).toEqual([]);
    expect(mgr.getSession('acc-1')?.status).toBe('QUEUED');
    expect(mgr.getSession('acc-1')?.events.some((ev) => ev.text.includes('sin actividad reciente'))).toBe(true);
  });

  it('refreshAccount keeps the restored endpoint when the ntfy probe misses', async () => {
    const mgr = createManager();
    const saved = session({
      status: 'RUNNING',
      seenBoot: true,
      endpoint: endpoint(),
      endpointUrl: 'https://live.trycloudflare.com/v1',
      lastEvent: { phase: 'heartbeat', rawTime: nowSec() - 15, up_min: 9 },
    });
    seed(mgr, saved);

    await mgr.refreshAccount(account);

    expect(harness.statuses).toEqual(['QUEUED']);
    expect(mgr.getSession('acc-1')?.status).toBe('RUNNING');
    expect(mgr.getActiveEndpoint()?.baseUrl).toBe('https://live.trycloudflare.com/v1');
    expect(mgr.getSession('acc-1')?.events.at(-1)?.text).toContain('endpoint conservado');
  });

  it('refreshAccount does not consult Kaggle when the ntfy probe already confirms serving', async () => {
    const mgr = createManager();
    seed(mgr, session());
    harness.status = 'QUEUED';
    harness.event = {
      phase: 'ready',
      endpoint: 'https://live.trycloudflare.com',
      rawTime: nowSec(),
      api_key: 'sk-from-probe',
    };

    await mgr.refreshAccount(account);

    expect(harness.statuses).toEqual([]);
    expect(mgr.getSession('acc-1')?.status).toBe('RUNNING');
    expect(mgr.getActiveEndpoint()?.baseUrl).toBe('https://live.trycloudflare.com/v1');
    expect(mgr.getActiveEndpoint()?.apiKey).toBe('sk-from-probe');
  });

  it('refreshAccount still wipes when Kaggle is QUEUED and serving evidence is stale', async () => {
    const mgr = createManager();
    seed(mgr, session({
      status: 'RUNNING',
      seenBoot: true,
      endpoint: endpoint(),
      lastEvent: { phase: 'ready', rawTime: nowSec() - 5000 },
    }));

    await mgr.refreshAccount(account);

    expect(mgr.getAllEndpoints()).toEqual([]);
    expect(mgr.getSession('acc-1')?.status).toBe('QUEUED');
    expect(mgr.getSession('acc-1')?.events.at(-1)?.text).toContain('Queued (QUEUED)');
  });

  it('monitorLoop still keeps the endpoint on a stale terminal status when heartbeats are fresh', async () => {
    const mgr = createManager();
    seed(mgr, session({
      status: 'RUNNING',
      seenBoot: true,
      endpoint: endpoint(),
      lastEvent: { phase: 'serving', rawTime: nowSec() - 40 },
    }));
    harness.status = 'COMPLETE';

    await monitor(mgr);

    expect(mgr.getSession('acc-1')?.done).toBe(false);
    expect(mgr.getSession('acc-1')?.status).toBe('RUNNING');
    expect(mgr.getActiveEndpoint()?.baseUrl).toBe('https://live.trycloudflare.com/v1');
  });
});
