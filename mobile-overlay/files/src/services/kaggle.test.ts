import { describe, it, expect } from 'vitest';
import { parseKaggleStatusPayload } from './kaggle';

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
