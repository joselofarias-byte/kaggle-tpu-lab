import { describe, it, expect, vi, afterEach } from 'vitest';
import { parseKaggleStatusPayload, KaggleApi } from './kaggle';

describe('parseKaggleStatusPayload', () => {
  it('maps current Kaggle numeric worker states', () => {
    expect(parseKaggleStatusPayload({ workerStatus: 0 }).status).toBe('QUEUED');
    expect(parseKaggleStatusPayload({ workerStatus: 1 }).status).toBe('RUNNING');
    expect(parseKaggleStatusPayload({ workerStatus: 2 }).status).toBe('COMPLETE');
    expect(parseKaggleStatusPayload({ workerStatus: 3 }).status).toBe('ERROR');
    expect(parseKaggleStatusPayload({ workerStatus: 4 }).status).toBe('UNKNOWN');
    expect(parseKaggleStatusPayload({ workerStatus: 5 }).status).toBe('UNKNOWN');
  });

  it('treats NEW_SCRIPT (6) as non-terminal queued/startup state', () => {
    expect(parseKaggleStatusPayload({ workerStatus: 6 }).status).toBe('QUEUED');
    expect(parseKaggleStatusPayload({ workerStatus: 'NEW_SCRIPT' }).status).toBe('QUEUED');
  });

  it('maps textual worker states', () => {
    expect(parseKaggleStatusPayload({ status: 'RUNNING' }).status).toBe('RUNNING');
    expect(parseKaggleStatusPayload({ status: 'QUEUED' }).status).toBe('QUEUED');
    expect(parseKaggleStatusPayload({ status: 'COMPLETE' }).status).toBe('COMPLETE');
    expect(parseKaggleStatusPayload({ status: 'ERROR' }).status).toBe('ERROR');
    expect(parseKaggleStatusPayload({ status: 'CANCEL_REQUESTED' }).status).toBe('UNKNOWN');
    expect(parseKaggleStatusPayload({ status: 'CANCEL_ACKNOWLEDGED' }).status).toBe('UNKNOWN');
    expect(parseKaggleStatusPayload({ status: 'CANCELLED' }).status).toBe('CANCELLED');
  });

  it('never invents CANCELLED for unknown payloads', () => {
    expect(parseKaggleStatusPayload({ workerStatus: 99 }).status).toBe('UNKNOWN');
    expect(parseKaggleStatusPayload({}).status).toBe('UNKNOWN');
    expect(parseKaggleStatusPayload(null).status).toBe('UNKNOWN');
    expect(parseKaggleStatusPayload('not-json').status).toBe('UNKNOWN');
  });
});

describe('status endpoint protobuf defaults', () => {
  it('uses the SDK QUEUED default only for valid successful endpoint payloads', () => {
    expect(parseKaggleStatusPayload({}, true).status).toBe('QUEUED');
    expect(parseKaggleStatusPayload('{}', true).status).toBe('QUEUED');
    expect(parseKaggleStatusPayload({ failureMessage: '' }, true).status).toBe('QUEUED');
    expect(parseKaggleStatusPayload({}).status).toBe('UNKNOWN');
    for (const raw of [null, [], '', 'not-json', { error: 'denied' }, { status: null }, { failureMessage: 'failed' }]) {
      expect(parseKaggleStatusPayload(raw, true).status).toBe('UNKNOWN');
    }
  });
  it('supports numeric strings without accepting partial state names', () => {
    for (const [status, expected] of [['0', 'QUEUED'], ['1', 'RUNNING'], ['2', 'COMPLETE'], ['3', 'ERROR'], ['4', 'UNKNOWN'], ['5', 'UNKNOWN'], ['6', 'QUEUED'], [' queued ', 'QUEUED'], ['NOT_RUNNING', 'UNKNOWN'], ['INCOMPLETE', 'UNKNOWN']]) {
      expect(parseKaggleStatusPayload({ status }).status).toBe(expected);
    }
  });
});


describe('getStatus transport', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('applies the omitted QUEUED default only after HTTP 200', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ status: 200, json: async () => ({}) });
    vi.stubGlobal('fetch', fetchMock);
    const api = new KaggleApi('test-only', 'demo');
    expect((await api.getStatus('qwen38-tpu-serve')).status).toBe('QUEUED');
    fetchMock.mockResolvedValue({ status: 503, json: async () => ({}) });
    expect((await api.getStatus('qwen38-tpu-serve')).status).toBe('UNKNOWN');
    fetchMock.mockRejectedValue(new Error('network unavailable'));
    expect((await api.getStatus('qwen38-tpu-serve')).status).toBe('UNKNOWN');
  });
});
