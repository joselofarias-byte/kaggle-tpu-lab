/**
 * Direct liveness probe for an OpenAI-compatible endpoint.
 *
 * Idea from Haz4rdovisk `src-tauri/src/kaggle/probe.rs`: ntfy saying "ready"
 * is not the same as the tunnel answering. The bearer key is sent and never
 * written into the thrown error.
 */

export function modelsUrl(endpoint: string): string {
  const root = endpoint.trim().replace(/\/+$/, '');
  const base = root.endsWith('/v1') ? root : `${root}/v1`;
  return `${base}/models`;
}

/**
 * Three missed probes take a READY endpoint off the "TPU lista" list.
 * A later success puts it back without relaunching the kernel.
 */
export function nextProbeDecision(
  status: string,
  misses: number,
  ok: boolean,
): { status: 'READY' | 'OFFLINE'; misses: number; listed: boolean } {
  if (ok) return { status: 'READY', misses: 0, listed: true };
  const next = misses + 1;
  if (next >= 3 && status === 'READY') return { status: 'OFFLINE', misses: next, listed: false };
  return { status: status === 'OFFLINE' ? 'OFFLINE' : 'READY', misses: next, listed: status === 'READY' };
}

export async function probeEndpoint(endpoint: string, apiKey: string, timeoutMs = 8000): Promise<boolean> {
  if (!endpoint || !apiKey) return false;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(modelsUrl(endpoint), {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: ctrl.signal,
    });
    return res.status === 200;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
