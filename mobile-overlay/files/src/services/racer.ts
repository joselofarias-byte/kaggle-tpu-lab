import { KaggleAccount, LaunchConfig, RaceSession, LiveEndpoint, NtfyEvent, getServedModelName } from './types';
import { KaggleApi } from './kaggle';
import { prepareKernel } from './templates';
import { NtfyListener } from './ntfy';
import { randomApiKey, randomTopicId } from './random';
import { startQueueMonitoring, stopQueueMonitoring } from './queueMonitor';

export type InstanceUpdateCallback = (
  sessions: RaceSession[],
  activeEndpoint?: LiveEndpoint,
  allEndpoints?: LiveEndpoint[]
) => void;

/**
 * Pure mount decision, extracted from `InstanceManager.handleNtfyEvent` so the
 * race veto logic is unit-testable.
 *
 * - `mount`: a genuine serving signal (ready/serving/heartbeat) with a valid
 *   http(s) URL arrived and Kaggle does not report QUEUED → mount it.
 * - `veto-queued`: valid signal + URL, but Kaggle explicitly reports QUEUED →
 *   do NOT mount prematurely (and do not run the heartbeat-uptime branch).
 * - `skip`: no valid URL or not a serving phase → fall through to the
 *   heartbeat-uptime branch / no-op.
 */
export function decideEndpointMount(opts: {
  status: RaceSession['status'];
  phase: NtfyEvent['phase'];
  sessionEndpointUrl?: string;
  eventEndpoint?: string;
  existingBaseUrl?: string;
}): { kind: 'mount'; url: string } | { kind: 'veto-queued' } | { kind: 'skip' } {
  const rawUrl = (opts.sessionEndpointUrl || opts.eventEndpoint || opts.existingBaseUrl || '').trim();
  const hasValidHttp = rawUrl.startsWith('http://') || rawUrl.startsWith('https://');
  const isServingPhase = opts.phase === 'ready' || opts.phase === 'serving' || opts.phase === 'heartbeat';
  if (!hasValidHttp || !isServingPhase) return { kind: 'skip' };
  // Kaggle status veto: if Kaggle explicitly remains QUEUED, do not mount the READY endpoint prematurely.
  if (opts.status === 'QUEUED') return { kind: 'veto-queued' };
  // Strip trailing slashes so `https://host/` does not become `https://host//v1`.
  const trimmed = rawUrl.replace(/\/+$/, '');
  const url = trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
  return { kind: 'mount', url };
}

export class InstanceManager {
  private sessions: Map<string, RaceSession> = new Map();
  private listeners: Map<string, NtfyListener> = new Map();
  // accountId -> ntfy topic currently watched by the native foreground-service
  // monitor (survives the app going to background; the WebView JS loop does not)
  private nativeTopics: Map<string, string> = new Map();
  private endpoints: Map<string, LiveEndpoint> = new Map();
  private pollInterval: any = null;
  private onUpdateCb: InstanceUpdateCallback;

  constructor(onUpdate: InstanceUpdateCallback) {
    this.onUpdateCb = onUpdate;
  }

  public getSession(accountId: string): RaceSession | undefined {
    return this.sessions.get(accountId);
  }

  public getAllSessions(): RaceSession[] {
    return Array.from(this.sessions.values());
  }

  public getAllEndpoints(): LiveEndpoint[] {
    const eps: LiveEndpoint[] = [];
    const seen = new Set<string>();
    for (const ep of this.endpoints.values()) {
      if (ep.status === 'READY' && ep.baseUrl && ep.baseUrl.startsWith('http')) {
        eps.push(ep);
        if (ep.accountId) seen.add(ep.accountId);
        seen.add(ep.baseUrl);
      }
    }
    for (const s of this.sessions.values()) {
      if (s.endpoint && s.endpoint.status === 'READY' && s.endpoint.baseUrl?.startsWith('http') && !seen.has(s.account)) {
        eps.push(s.endpoint);
        seen.add(s.account);
      }
    }
    return eps;
  }

  public getActiveEndpoint(): LiveEndpoint | undefined {
    return this.getAllEndpoints()[0];
  }

  // ponytail: unified launcher for both single instance and multi-key race mode
  private async launchAccount(
    account: KaggleAccount,
    config: LaunchConfig,
    raceGroupId?: string
  ): Promise<void> {
    if (!account.token) throw new Error(`Falta el token API de la cuenta [${account.name || account.username}]`);
    await this.stopAccount(account.id);

    // Per-account random topic + API key (crypto-secure, see services/random.ts)
    const topic = randomTopicId();
    const apiKey = randomApiKey(config.model === 'glm53-flash' ? 'glm-' : 'sk-');
    const prep = prepareKernel(config.model, config, topic, apiKey);
    const api = new KaggleApi(account.token, account.username);

    const prefix = raceGroupId ? '[Race] ' : '';
    const session: RaceSession = {
      account: account.id,
      accountName: account.name || account.username,
      username: account.username,
      token: account.token,
      kernel: `${account.username}/${prep.slug}`,
      slug: prep.slug,
      topic: prep.topic,
      apiKey: prep.apiKey,
      endpointUrl: undefined,
      model: config.model,
      status: 'QUEUED',
      seenBoot: false,
      done: false,
      raceGroupId,
      events: [
        {
          time: new Date().toLocaleTimeString(),
          phase: 'pushing',
          text: `${prefix}Enviando ${config.model} a [${account.username || account.name}]...`,
        },
      ],
    };

    this.sessions.set(account.id, session);
    this.notify();

    const ntfy = new NtfyListener(topic, (ev, text) => {
      this.handleNtfyEvent(session, ev, text, config.model);
    });
    ntfy.start(4000);
    this.listeners.set(account.id, ntfy);

    try {
      const pushRes = await api.pushKernel(prep.slug, prep.code, prep.datasets);
      if (pushRes.error && (pushRes.error.includes('running') || pushRes.error.includes('Conflict') || pushRes.error.includes('adopted') || pushRes.error.includes('already'))) {
        session.events.push({
          time: new Date().toLocaleTimeString(),
          phase: 'sync',
          text: `${prefix}Se detectó una TPU activa; adoptando y sincronizando la configuración...`,
        });
        const remoteKernel = await api.getKernel(prep.slug);
        if (remoteKernel?.ntfyTopic) {
          session.topic = remoteKernel.ntfyTopic;
          if (remoteKernel.apiKey) session.apiKey = remoteKernel.apiKey;
          ntfy.stop();
          const realNtfy = new NtfyListener(remoteKernel.ntfyTopic, (ev, text) => {
            this.handleNtfyEvent(session, ev, text, config.model);
          });
          realNtfy.start(3000);
          this.listeners.set(account.id, realNtfy);
        }
      } else {
        session.events.push({
          time: new Date().toLocaleTimeString(),
          phase: 'queued',
          text: `${prefix}Envío correcto; esperando en cola una TPU disponible...`,
        });
      }
    } catch (err: any) {
      session.status = 'ERROR';
      session.done = true;
      session.events.push({
        time: new Date().toLocaleTimeString(),
        phase: 'failed',
        text: `Falló el envío: ${err.message || String(err)}`,
      });
    }

    // Hand the final topic (adopted kernels use the remote topic) to the
    // native foreground-service monitor so the user is notified on
    // ready/serving even with the app backgrounded.
    if (!session.done && session.topic) {
      this.startNativeMonitor(account.id, session.topic);
    }

    this.notify();
    this.ensurePolling();
  }

  /** Hand a topic to the native background monitor (fire-and-forget). */
  private startNativeMonitor(accountId: string, topic: string): void {
    this.stopNativeMonitor(accountId);
    this.nativeTopics.set(accountId, topic);
    startQueueMonitoring(topic).catch(() => {});
  }

  /** Withdraw a topic from the native background monitor. */
  private stopNativeMonitor(accountId: string): void {
    const topic = this.nativeTopics.get(accountId);
    if (topic !== undefined) {
      this.nativeTopics.delete(accountId);
      stopQueueMonitoring(topic).catch(() => {});
    }
  }

  public async startSingle(account: KaggleAccount, config: LaunchConfig): Promise<void> {
    return this.launchAccount(account, config);
  }

  public async startRace(accounts: KaggleAccount[], config: LaunchConfig): Promise<void> {
    const valid = accounts.filter((a) => a.isValid && a.token);
    if (valid.length < 2) throw new Error('La carrera requiere al menos 2 cuentas de Kaggle válidas');
    const groupId = 'race-' + Date.now();
    await Promise.all(valid.map((acc) => this.launchAccount(acc, config, groupId)));
  }

  private handleNtfyEvent(session: RaceSession, ev: NtfyEvent, text: string, modelName: string) {
    session.seenBoot = true;
    session.lastEvent = ev;
    session.events.push({ time: new Date().toLocaleTimeString(), phase: ev.phase, text });

    // Capture tunnel URL allocation event
    if (ev.endpoint) {
      const u = ev.endpoint.trim();
      if (u.startsWith('http://') || u.startsWith('https://')) {
        session.endpointUrl = u.endsWith('/v1') ? u : `${u}/v1`;
      }
    }

    // Mount ready endpoint only when receiving authentic serving / ready / heartbeat while not queued
    // Note: tunnel-url phase only records URL; never mark READY prematurely (aligning with launch.py)
    const verdict = decideEndpointMount({
      status: session.status,
      phase: ev.phase,
      sessionEndpointUrl: session.endpointUrl,
      eventEndpoint: ev.endpoint,
      existingBaseUrl: session.endpoint?.baseUrl,
    });

    if (verdict.kind === 'mount') {
      const cleanBaseUrl = verdict.url;
      session.endpointUrl = cleanBaseUrl;
      const uptime = typeof ev.up_min === 'number' ? ev.up_min : (ev.up_min ? parseInt(String(ev.up_min), 10) : session.endpoint?.uptimeMinutes);
      const ep: LiveEndpoint = {
        accountId: session.account,
        accountName: session.accountName,
        baseUrl: cleanBaseUrl,
        apiKey: ev.api_key || session.apiKey,
        model: getServedModelName(ev.model || session.model || modelName),
        status: 'READY',
        tokensPerSec: ev.decode_tok_s ? String(ev.decode_tok_s) : session.endpoint?.tokensPerSec,
        uptimeMinutes: uptime,
      };
      if (ev.api_key) session.apiKey = ev.api_key;
      session.endpoint = ep;
      this.endpoints.set(session.account, ep);
      if (session.status !== 'RUNNING' && session.status !== 'WINNER') {
        session.status = 'RUNNING';
        session.done = false;
      }
      if (session.raceGroupId && (ev.phase === 'ready' || ev.phase === 'serving')) {
        this.checkRaceWinner(session);
      }
      // Endpoint is live — the native background monitor has done its job.
      if (ev.phase === 'ready' || ev.phase === 'serving') {
        this.stopNativeMonitor(session.account);
      }
    } else if (ev.phase === 'heartbeat' && verdict.kind === 'skip') {
      const ep = this.endpoints.get(session.account);
      if (ep) {
        ep.uptimeMinutes = typeof ev.up_min === 'number' ? ev.up_min : parseInt(String(ev.up_min || 0), 10);
      }
    }

    if (ev.phase === 'failed' || ev.phase === 'stopped' || ev.phase === 'auto-shutdown') {
      session.done = true;
      session.status = ev.phase === 'auto-shutdown' ? 'COMPLETE' : 'ERROR';
      const ep = this.endpoints.get(session.account);
      if (ep) ep.status = 'OFFLINE';
      this.endpoints.delete(session.account);
      this.stopNativeMonitor(session.account);
    }
    this.notify();
  }

  private checkRaceWinner(winner: RaceSession) {
    if (!winner.raceGroupId) return;
    const gid = winner.raceGroupId;
    winner.status = 'WINNER';

    for (const [accId, rival] of this.sessions.entries()) {
      if (rival.raceGroupId === gid && rival.account !== winner.account && !rival.done) {
        rival.done = true;
        rival.status = 'CANCELLED';
        rival.endpoint = undefined;
        this.endpoints.delete(rival.account);
        rival.events.push({
          time: new Date().toLocaleTimeString(),
          phase: 'cancelled',
          text: `🏁 [${winner.username || winner.accountName}] won the TPU race! Automatically cancelled [${rival.username || rival.accountName}] to protect quota.`,
        });
        const l = this.listeners.get(accId);
        if (l) { l.stop(); this.listeners.delete(accId); }
        this.stopNativeMonitor(accId);
        new KaggleApi(rival.token, rival.username).cancelKernel(rival.slug).catch(() => {});
      }
    }
  }

  private ensurePolling() {
    if (!this.pollInterval) {
      this.pollInterval = setInterval(() => this.monitorLoop(), 12000);
    }
  }

  private async monitorLoop() {
    const active = Array.from(this.sessions.values()).filter((s) => !s.done);
    if (active.length === 0) {
      if (this.pollInterval) { clearInterval(this.pollInterval); this.pollInterval = null; }
      return;
    }

    for (const session of active) {
      try {
        const st = await new KaggleApi(session.token, session.username).getStatus(session.slug);
        if (st.status === 'UNKNOWN') {
          // UNKNOWN is not a terminal state. API/network ambiguity must never
          // make us stop tracking a kernel that may still be consuming TPU quota.
          const last = session.events[session.events.length - 1];
          if (!last || last.phase !== 'sync-unknown') {
            const detail = st.httpStatus
              ? `HTTP ${st.httpStatus}`
              : (st.isNetworkError ? 'problema de red' : (st.detail || 'respuesta no reconocida'));
            session.events.push({
              time: new Date().toLocaleTimeString(),
              phase: 'sync-unknown',
              text: `No se pudo confirmar el estado en Kaggle (${detail}). Se mantiene el seguimiento y no se da la sesión por terminada.`,
            });
          }
          if (session.status === 'IDLE') session.status = 'UNKNOWN';
          session.done = false;
          continue;
        }
        if (st.status === 'RUNNING' && session.status !== 'RUNNING' && session.status !== 'WINNER') {
          session.status = 'RUNNING';
          session.events.push({
            time: new Date().toLocaleTimeString(),
            phase: 'running',
            text: `[${session.username || session.accountName}] TPU asignada; preparando el entorno...`,
          });
          if (session.raceGroupId) this.checkRaceWinner(session);
        } else if (st.status === 'QUEUED') {
          if (session.status !== 'QUEUED') {
            session.status = 'QUEUED';
            // While queued, clear any false ready endpoint
            session.endpoint = undefined;
            this.endpoints.delete(session.account);
            session.events.push({
              time: new Date().toLocaleTimeString(),
              phase: 'queued',
              text: `[${session.username || session.accountName}] Kaggle: Waiting in queue for TPU v5e-8 slot...`,
            });
          }
        } else if (st.status === 'ERROR' || st.status === 'CANCELLED' || st.status === 'COMPLETE') {
          session.status = st.status;
          session.done = true;
          session.endpoint = undefined;
          this.endpoints.delete(session.account);
          session.events.push({
            time: new Date().toLocaleTimeString(),
            phase: 'stopped',
            text: `Instance stopped: ${st.status} ${st.failureMessage || ''}`,
          });
          const ep = this.endpoints.get(session.account);
          if (ep) ep.status = 'OFFLINE';
        }
      } catch {}
    }
    this.notify();
  }

  public setCustomEndpoint(accountId: string, customUrl: string) {
    const session = this.sessions.get(accountId);
    if (!session) return;
    const ep: LiveEndpoint = {
      accountId: session.account,
      accountName: session.accountName,
      model: session.model || 'qwen38-27b',
      baseUrl: customUrl.trim(),
      apiKey: session.apiKey || 'sk-custom',
      status: 'READY',
    };
    session.endpoint = ep;
    session.status = 'RUNNING';
    session.done = false;
    this.endpoints.set(session.account, ep);
    this.notify();
  }

  public restoreSessions(saved: RaceSession[]) {
    for (const s of saved) {
      this.sessions.set(s.account, s);
      if (s.topic && !s.done) {
        if (this.listeners.has(s.account)) {
          this.listeners.get(s.account)?.stop();
        }
        const ntfy = new NtfyListener(s.topic, (ev, text) => {
          this.handleNtfyEvent(s, ev, text, s.model || 'qwen38-27b');
        });
        ntfy.start(4000);
        this.listeners.set(s.account, ntfy);
      }
      if (s.endpoint && s.endpoint.status === 'READY' && s.endpoint.baseUrl?.startsWith('http')) {
        this.endpoints.set(s.account, s.endpoint);
      }
    }
    if (saved.some((s) => !s.done)) this.ensurePolling();
    this.notify();
  }

  public async refreshAccount(account: KaggleAccount): Promise<void> {
    if (!account.token) return;

    let session = this.sessions.get(account.id);
    const api = new KaggleApi(account.token, account.username || 'user');
    const slugs = Array.from(new Set([session?.slug, 'qwen38-tpu-serve', 'glm53-tpu-serve'].filter(Boolean) as string[]));

    let foundActive = false;
    let sawUnknown = false;
    let sessionTerminalStatus: RaceSession['status'] | null = null;
    let sessionTerminalMessage = '';

    for (const slug of slugs) {
      try {
        const st = await api.getStatus(slug);
        if (st.status === 'UNKNOWN') {
          sawUnknown = true;
          continue;
        }

        // Only adopt when Kaggle explicitly returns RUNNING or QUEUED
        if (st.status === 'RUNNING' || st.status === 'QUEUED') {
          foundActive = true;
          const kernelInfo = await api.getKernel(slug);
          const topic = kernelInfo?.ntfyTopic || session?.topic;
          const apiKey = kernelInfo?.apiKey || session?.apiKey || '';

          let currentSession = session;
          if (!currentSession) {
            currentSession = {
              account: account.id,
              accountName: account.name || account.username,
              username: account.username,
              token: account.token,
              kernel: `${account.username}/${slug}`,
              slug,
              topic: topic || '',
              apiKey,
              model: slug.includes('glm') ? 'glm53-flash' : 'qwen38-27b',
              status: st.status,
              seenBoot: st.status === 'RUNNING',
              done: false,
              events: [],
            };
            this.sessions.set(account.id, currentSession);
            session = currentSession;
          }
          currentSession.slug = slug;
          currentSession.status = st.status;
          currentSession.done = false;
          if (st.status === 'QUEUED') {
            currentSession.endpoint = undefined;
            this.endpoints.delete(account.id);
          }

          if (topic) currentSession.topic = topic;
          if (apiKey) currentSession.apiKey = apiKey;
          if (st.status === 'RUNNING') currentSession.seenBoot = true;

          const statusText =
            st.status === 'QUEUED' ? 'Queued (QUEUED)' : '🟢 TPU Instance Running';

          currentSession.events.push({
            time: new Date().toLocaleTimeString(),
            phase: 'sync',
            text: `Synced cloud status [${slug}]: ${statusText}`,
          });

          if (currentSession.topic) {
            if (this.listeners.has(account.id)) {
              this.listeners.get(account.id)?.stop();
            }
            const sess = currentSession;
            const ntfy = new NtfyListener(sess.topic, (ev, text) => {
              this.handleNtfyEvent(sess, ev, text, sess.model || 'qwen38-27b');
            });
            ntfy.start(4000);
            this.listeners.set(account.id, ntfy);
            ntfy.pollOnce().catch(() => {});
          }

          this.ensurePolling();
          this.notify();
          return;
        } else if (st.status === 'CANCELLED' || st.status === 'COMPLETE' || st.status === 'ERROR') {
          // Only a positive terminal state for the session's own slug may end it.
          // Other checked slugs are merely fallback candidates.
          if (session && slug === session.slug) {
            sessionTerminalStatus = st.status;
            sessionTerminalMessage = st.failureMessage || '';
          }
        }
      } catch {}
    }

    if (!foundActive) {
      if (session && sessionTerminalStatus) {
        session.status = sessionTerminalStatus;
        session.done = true;
        session.endpoint = undefined;
        this.endpoints.delete(account.id);
        const listener = this.listeners.get(account.id);
        if (listener) {
          listener.stop();
          this.listeners.delete(account.id);
        }
        session.events.push({
          time: new Date().toLocaleTimeString(),
          phase: 'stopped',
          text: `Kaggle confirmó fin de la sesión: ${sessionTerminalStatus}${sessionTerminalMessage ? ` — ${sessionTerminalMessage}` : ''}`,
        });
      } else if (session && sawUnknown && !session.done) {
        // Preserve the last confirmed state. Losing one status probe must not
        // detach the controller from a potentially live TPU.
        if (session.status === 'IDLE') session.status = 'UNKNOWN';
        session.done = false;
        const last = session.events[session.events.length - 1];
        if (!last || last.phase !== 'sync-unknown') {
          session.events.push({
            time: new Date().toLocaleTimeString(),
            phase: 'sync-unknown',
            text: 'Estado de Kaggle sin confirmar. Se conserva la sesión y el monitoreo para evitar perder una TPU que siga consumiendo cuota.',
          });
        }
        this.ensurePolling();
      } else if (session && !session.done) {
        session.status = 'UNKNOWN';
        session.done = false;
        session.events.push({
          time: new Date().toLocaleTimeString(),
          phase: 'sync-unknown',
          text: 'No hubo una confirmación terminal de Kaggle. La sesión permanece bajo seguimiento.',
        });
        this.ensurePolling();
      } else if (!session) {
        this.sessions.set(account.id, {
          account: account.id,
          accountName: account.name || account.username,
          username: account.username,
          token: account.token,
          kernel: `${account.username}/qwen38-tpu-serve`,
          slug: 'qwen38-tpu-serve',
          topic: '',
          apiKey: '',
          status: 'IDLE',
          seenBoot: false,
          done: true,
          events: [{
            time: new Date().toLocaleTimeString(),
            phase: sawUnknown ? 'sync-unknown' : 'sync',
            text: sawUnknown
              ? 'No se pudo confirmar una sesión activa en Kaggle; no se afirma que esté detenida.'
              : 'Sincronización completa; no hay una sesión conocida activa ni en cola para esta cuenta',
          }],
        });
      }
    }

    if (Array.from(this.sessions.values()).every((s) => s.done)) {
      if (this.pollInterval) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
      }
    }

    this.notify();
  }

  public async stopAccount(accountId: string): Promise<void> {
    const listener = this.listeners.get(accountId);
    if (listener) { listener.stop(); this.listeners.delete(accountId); }
    this.stopNativeMonitor(accountId);

    const session = this.sessions.get(accountId);
    if (session && !session.done) {
      session.status = 'CANCELLED';
      session.done = true;
      session.events.push({ time: new Date().toLocaleTimeString(), phase: 'stopped', text: 'User terminated task' });
      new KaggleApi(session.token, session.username).cancelKernel(session.slug).catch(() => {});
    }

    const ep = this.endpoints.get(accountId);
    if (ep) { ep.status = 'OFFLINE'; this.endpoints.delete(accountId); }
    this.notify();
  }

  public async stop(): Promise<void> {
    if (this.pollInterval) { clearInterval(this.pollInterval); this.pollInterval = null; }
    for (const [accId] of this.sessions) await this.stopAccount(accId);
  }

  private notify() {
    this.onUpdateCb(this.getAllSessions(), this.getActiveEndpoint(), this.getAllEndpoints());
  }
}

export { InstanceManager as RaceManager };
