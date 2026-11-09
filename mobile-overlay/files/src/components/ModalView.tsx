import React, { useEffect, useRef, useState } from 'react';
import { ModalApi, validateModalConnection } from '../services/modal';
import type { ModalConnection, ModalProfile, ModalSession } from '../services/modal';
import { secretStore } from '../services/secureStore';
import './ModalView.css';

interface ModalViewProps {
  onStatusChange: (status: string) => void;
}

type Message = { role: 'user' | 'assistant'; content: string };
const CONNECTION_KEY = 'ktl_modal_connection';
const labels: Record<ModalSession['status'], string> = {
  STARTING: 'Iniciando', READY: 'Lista', STOPPING: 'Deteniendo', STOPPED: 'Detenida',
  ERROR: 'Error del controlador', UNKNOWN: 'Estado sin confirmar',
};
const dateLabel = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

export const ModalView: React.FC<ModalViewProps> = ({ onStatusChange }) => {
  const [url, setUrl] = useState('');
  const [token, setToken] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [connected, setConnected] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [profiles, setProfiles] = useState<ModalProfile[]>([]);
  const [profileId, setProfileId] = useState('');
  const [session, setSession] = useState<ModalSession | null>(null);
  const [maxMinutes, setMaxMinutes] = useState('30');
  const [idleMinutes, setIdleMinutes] = useState('5');
  const [hourly, setHourly] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatBusy, setChatBusy] = useState(false);
  const api = useRef<ModalApi | null>(null);
  const connection = useRef<ModalConnection | null>(null);
  const working = useRef(false);
  const generation = useRef(0);
  const mounted = useRef(false);
  const statusCallback = useRef(onStatusChange);
  statusCallback.current = onStatusChange;

  useEffect(() => {
    mounted.current = true;
    let cancelled = false;
    secretStore.get(CONNECTION_KEY).then((raw) => {
      if (cancelled || !raw) return;
      const saved = JSON.parse(raw) as ModalConnection;
      if (typeof saved.url !== 'string' || typeof saved.token !== 'string') throw new Error();
      setUrl(saved.url);
      setToken(saved.token);
    }).catch(() => {
      if (!cancelled) setError('No se pudo leer la conexión guardada. Introduce los datos de nuevo.');
    }).finally(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; mounted.current = false; generation.current += 1; };
  }, []);

  const stateLabel = !connected && !session ? 'Sin conectar'
    : !confirmed || session?.status === 'UNKNOWN'
      ? 'Estado sin confirmar' : session ? labels[session.status] : 'Sin sesión activa';
  useEffect(() => { statusCallback.current(stateLabel); }, [stateLabel]);

  const fail = (message: string) => {
    setConfirmed(false);
    setNotice('');
    setError(message);
  };

  const sync = async () => {
    if (!api.current || working.current) return;
    working.current = true;
    const current = generation.current;
    try {
      const next = await api.current.session();
      if (!mounted.current || current !== generation.current) return;
      setSession(next);
      setConfirmed(true);
      setError('');
    } catch {
      if (mounted.current && current === generation.current) {
        fail('No se pudo confirmar el estado. La sesión anterior podría seguir generando costes. Sincroniza antes de iniciar otra.');
      }
    } finally { working.current = false; }
  };

  useEffect(() => {
    if (!connected) return;
    const timer = window.setInterval(() => { void sync(); }, 10000);
    return () => window.clearInterval(timer);
  }, [connected]);

  const connect = async () => {
    if (working.current) return;
    const nextConnection = { url: url.trim().replace(/\/+$/, ''), token: token.trim() };
    const validation = validateModalConnection(nextConnection);
    if (validation) { setError(validation); return; }
    // An unresolved controller must not disappear behind a different connection.
    if (connection.current && (!confirmed || (session && session.status !== 'STOPPED')) &&
        nextConnection.url !== connection.current.url) {
      setError('Reconecta al mismo controlador y confirma que la sesión está detenida antes de cambiar la URL.');
      return;
    }
    working.current = true;
    setBusy(true);
    setConfirmed(false);
    setError('');
    setNotice('');
    setProfiles([]);
    setProfileId('');
    if (connection.current?.url !== nextConnection.url) { setMessages([]); setLogs([]); }
    const current = ++generation.current;
    try {
      await secretStore.set(CONNECTION_KEY, JSON.stringify(nextConnection));
      const client = new ModalApi(nextConnection);
      api.current = client;
      connection.current = nextConnection;
      setConnected(true);
      const next = await client.session();
      if (!mounted.current || current !== generation.current) return;
      setSession(next);
      setConfirmed(true);
      const available = await client.profiles();
      if (!mounted.current || current !== generation.current) return;
      setProfiles(available);
      setProfileId((previous) => available.some((p) => p.id === previous) ? previous : available[0]?.id || '');
      setNotice('Conexión guardada y estado sincronizado.');
    } catch {
      if (mounted.current && current === generation.current) fail('No se pudo guardar o conectar al controlador. Comprueba la URL HTTPS, el token y el despliegue.');
    } finally {
      working.current = false;
      if (mounted.current) setBusy(false);
    }
  };

  const selected = profiles.find((p) => p.id === profileId);
  const max = Number(maxMinutes);
  const idle = Number(idleMinutes);
  const validLimits = Number.isInteger(max) && max >= 5 && max <= 120 &&
    Number.isInteger(idle) && idle >= 2 && idle <= 30 && idle <= max;
  const price = Number(hourly);
  const validPrice = hourly.trim() !== '' && Number.isFinite(price) && price >= 0;
  const canStart = connected && confirmed && (!session || session.status === 'STOPPED') && !!selected && validLimits && !busy && !chatBusy;

  const control = async (action: 'start' | 'stop') => {
    if (!api.current || working.current || (action === 'start' && !canStart)) return;
    const question = action === 'start'
      ? `¿Iniciar una sesión de pago?\n${selected?.name}: GPU ${selected?.gpu}, CPU ${selected?.cpu}, RAM ${selected?.memory_mib} MiB.\nMáximo ${max} min; inactividad ${idle} min.\n${validPrice ? `Estimación GPU: USD ${(price * max / 60).toFixed(2)} para el máximo indicado.` : 'Sin estimación de precio GPU.'}\nCPU, memoria y almacenamiento no incluidos. No es una factura.`
      : '¿Solicitar la parada de la sesión de este controlador? Se comprobará la respuesta del servidor.';
    if (!window.confirm(question)) return;
    working.current = true;
    setBusy(true);
    setError('');
    setNotice('');
    const current = generation.current;
    try {
      const next = action === 'start'
        ? await api.current.start({ profile_id: profileId, max_minutes: max, idle_minutes: idle, request_id: crypto.randomUUID() })
        : await api.current.stop();
      if (!mounted.current || current !== generation.current) return;
      setSession(next);
      setConfirmed(true);
      if (action === 'start') {
        setMessages([]);
        setLogs([]);
        setNotice('Solicitud recibida. Consulta el estado de la sesión.');
      } else {
        setNotice(!next || next.status === 'STOPPED' ? 'El servidor confirma que no hay sesión activa.' : 'Parada solicitada; aún no está confirmada como detenida.');
      }
    } catch {
      if (mounted.current && current === generation.current) fail(`No se pudo confirmar ${action === 'start' ? 'el inicio' : 'la parada'}. La sesión podría seguir activa. Sincroniza y comprueba el estado.`);
    } finally { working.current = false; if (mounted.current) setBusy(false); }
  };

  const refreshLogs = async () => {
    if (!api.current || busy) return;
    const current = generation.current;
    try {
      const next = await api.current.logs();
      if (mounted.current && current === generation.current) setLogs(next);
    } catch {
      if (mounted.current && current === generation.current) setError('No se pudieron obtener los registros.');
    }
  };

  const send = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!api.current || !confirmed || session?.status !== 'READY' || !prompt.trim() || chatBusy || busy) return;
    const next: Message[] = [...messages, { role: 'user', content: prompt.trim() }];
    const current = generation.current;
    setChatBusy(true);
    setError('');
    try {
      const answer = await api.current.chat(next);
      if (!mounted.current || current !== generation.current) return;
      setMessages([...next, { role: 'assistant', content: answer }]);
      setPrompt('');
    } catch {
      if (mounted.current && current === generation.current) setError('No se pudo enviar el mensaje. Tu texto se conserva para reintentar.');
    } finally { if (mounted.current) setChatBusy(false); }
  };

  return (
    <div className="modal-view">
      <section className="glass-card" aria-labelledby="modal-title">
        <h2 id="modal-title" className="card-title">Modal · Control de recursos</h2>
        <p className="modal-help">Despliega primero el backend <strong>modal-control</strong> y conecta su URL HTTPS y token de acceso. No introduzcas credenciales del proveedor Modal en el APK.</p>
        <p className="modal-help">Solo se administra la sesión de este controlador, no todos los recursos de tu workspace. Cerrar o desconectar el APK no detiene recursos.</p>
        <form onSubmit={(event) => { event.preventDefault(); void connect(); }}>
          <label className="modal-field">URL HTTPS del controlador
            <input className="form-input" type="url" placeholder="https://tu-controlador.modal.run" value={url} onChange={(e) => setUrl(e.target.value)} disabled={!loaded || busy || chatBusy} required autoCapitalize="none" autoCorrect="off" />
          </label>
          <label className="modal-field">Token del controlador
            <input className="form-input" type="password" value={token} onChange={(e) => setToken(e.target.value)} disabled={!loaded || busy || chatBusy} required autoComplete="off" />
          </label>
          <div className="modal-actions">
            <button className="btn btn-primary" type="submit" disabled={!loaded || busy || chatBusy}>{busy ? 'Procesando…' : 'Guardar y conectar'}</button>
            <button className="btn btn-secondary" type="button" disabled={!connected || busy || chatBusy} onClick={() => {
              generation.current += 1; api.current = null; setConnected(false); setConfirmed(false); setNotice('Desconectado. La sesión no se ha detenido.'); setError('');
            }}>Desconectar</button>
          </div>
        </form>
      </section>
      <section className="glass-card" aria-labelledby="modal-state">
        <h3 id="modal-state" className="card-title">Sesión del controlador</h3>
        <p className={`modal-status ${confirmed && session?.status !== 'UNKNOWN' ? '' : 'modal-uncertain'}`} role="status">{stateLabel}</p>
        {connection.current && <p className="modal-help">Controlador: {connection.current.url}</p>}
        {session && <dl className="modal-details">
          <dt>Sesión</dt><dd>{session.id}</dd>
          <dt>Último estado recibido</dt><dd>{labels[session.status]}</dd>
          <dt>Modelo / perfil</dt><dd>{session.model} / {session.profile_id}</dd>
          <dt>Recursos</dt><dd>GPU {session.gpu} · CPU {session.cpu} · RAM {session.memory_mib} MiB</dd>
          <dt>Creada</dt><dd>{dateLabel(session.created_at)}</dd>
          <dt>Expira</dt><dd>{dateLabel(session.expires_at)}</dd>
          <dt>Límites</dt><dd>{session.max_minutes} min máximo · {session.idle_minutes} min sin uso</dd>
          {session.detail && <><dt>Detalle</dt><dd>{session.detail}</dd></>}
        </dl>}
        <p className="modal-help">Sincronización cada 10 segundos mientras la conexión esté activa. Consultar el estado no cuenta como uso del modelo.</p>
        <div className="modal-actions">
          <button type="button" className="btn btn-secondary" disabled={!connected || busy} onClick={() => void sync()}>Sincronizar estado</button>
          <button type="button" className="btn btn-danger" disabled={!connected || busy || (confirmed && (!session || session.status === 'STOPPED'))} onClick={() => void control('stop')}>Detener sesión</button>
        </div>
      </section>
      {error && <p className="modal-feedback modal-error" role="alert">{error}</p>}
      {notice && <p className="modal-feedback" role="status">{notice}</p>}
      <section className="glass-card" aria-labelledby="modal-launch">
        <h3 id="modal-launch" className="card-title">Preparar sesión de pago</h3>
        <label className="modal-field">Perfil permitido por el controlador
          <select className="form-select" value={profileId} onChange={(e) => setProfileId(e.target.value)} disabled={!connected || busy}>
            {!profiles.length && <option value="">Conecta para cargar perfiles</option>}
            {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </label>
        {selected && <div className="modal-profile"><strong>{selected.model}</strong><p>GPU {selected.gpu} · CPU {selected.cpu} · RAM {selected.memory_mib} MiB</p><p className="modal-help">{selected.description}</p></div>}
        <div className="modal-grid">
          <label className="modal-field">Máximo (5–120 min)<input className="form-input" type="number" min="5" max="120" step="1" value={maxMinutes} onChange={(e) => setMaxMinutes(e.target.value)} disabled={busy} /></label>
          <label className="modal-field">Inactividad (2–30 min)<input className="form-input" type="number" min="2" max="30" step="1" value={idleMinutes} onChange={(e) => setIdleMinutes(e.target.value)} disabled={busy} /></label>
        </div>
        {!validLimits && <p className="modal-error">Usa minutos enteros dentro de los límites. La inactividad no puede superar el máximo.</p>}
        <label className="modal-field">Precio GPU por hora (USD, introducido por ti; opcional)
          <input className="form-input" type="number" inputMode="decimal" min="0" step="any" placeholder="Sin precio definido" value={hourly} onChange={(e) => setHourly(e.target.value)} />
        </label>
        <p className="modal-help">{validPrice && validLimits ? `Estimación GPU para ${max} min: USD ${(price * max / 60).toFixed(2)}.` : 'Sin estimación GPU: introduce una tarifa válida.'} No es una factura ni un límite de gasto. Excluye CPU, memoria y almacenamiento.</p>
        <button type="button" className="btn btn-primary" disabled={!canStart} onClick={() => void control('start')}>Confirmar inicio de pago</button>
        <p className="modal-help">Solo se permite iniciar tras confirmar que no hay una sesión activa. Nunca se inicia automáticamente.</p>
      </section>
      <section className="glass-card" aria-labelledby="modal-logs">
        <h3 id="modal-logs" className="card-title">Registros</h3>
        <div className="modal-actions">
          <button type="button" className="btn btn-secondary" aria-expanded={showLogs} aria-controls="modal-log-output" onClick={() => { setShowLogs(!showLogs); if (!showLogs) void refreshLogs(); }}>{showLogs ? 'Ocultar registros' : 'Mostrar registros'}</button>
          {showLogs && <button type="button" className="btn btn-secondary" disabled={!connected || busy} onClick={() => void refreshLogs()}>Actualizar registros</button>}
        </div>
        {showLogs && <pre id="modal-log-output" className="modal-log">{logs.length ? logs.join('\n') : 'Sin registros disponibles.'}</pre>}
      </section>
      <section className="glass-card" aria-labelledby="modal-chat">
        <h3 id="modal-chat" className="card-title">Chat Modal</h3>
        <p className="modal-help">Disponible solo con sesión lista y estado confirmado. Enviar mensajes sí utiliza el modelo.</p>
        <div className="modal-messages" aria-live="polite">
          {messages.map((message, index) => <div className={`modal-message modal-message-${message.role}`} key={index}><strong>{message.role === 'user' ? 'Tú' : 'Modal'}</strong><p>{message.content}</p></div>)}
        </div>
        <form onSubmit={(event) => void send(event)}>
          <label className="modal-field">Mensaje<textarea className="form-input" rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={chatBusy} placeholder="Escribe tu mensaje…" /></label>
          <button className="btn btn-primary" type="submit" disabled={!connected || !confirmed || session?.status !== 'READY' || !prompt.trim() || chatBusy || busy}>{chatBusy ? 'Esperando respuesta…' : 'Enviar a Modal'}</button>
        </form>
      </section>
    </div>
  );
};
