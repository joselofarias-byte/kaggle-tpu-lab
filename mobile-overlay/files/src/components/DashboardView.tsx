import React, { useState } from 'react';
import {
  KaggleAccount,
  LaunchConfig,
  RaceSession,
  LiveEndpoint,
  ModelId,
  ReasoningEffort,
} from '../services/types';
import {
  Cpu,
  Zap,
  StopCircle,
  Play,
  Share2,
  MessageSquare,
  Clock,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Layers,
  CheckSquare,
  Square,
  Copy,
  Check,
} from 'lucide-react';
import { copyToClipboard, copyEndpointBundle } from '../services/clipboard';

interface DashboardViewProps {
  accounts: KaggleAccount[];
  config: LaunchConfig;
  sessions: RaceSession[];
  endpoints: LiveEndpoint[];
  activeEndpoint?: LiveEndpoint;
  isLaunching: boolean;
  onUpdateConfig: (config: LaunchConfig) => void;
  onStartSingle: (account: KaggleAccount) => void;
  onStartRace: (selectedAccounts: KaggleAccount[]) => void;
  onStopAccount: (accountId: string) => void;
  onStopAll: () => void;
  onRefreshAccount: (account: KaggleAccount) => void;
  onGoToChat: (endpoint?: LiveEndpoint) => void;
  onOpenExport: (endpoint?: LiveEndpoint) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  accounts,
  config,
  sessions,
  endpoints,
  activeEndpoint,
  isLaunching,
  onUpdateConfig,
  onStartSingle,
  onStartRace,
  onStopAccount,
  onStopAll,
  onRefreshAccount,
  onGoToChat,
  onOpenExport,
}) => {
  const [viewMode, setViewMode] = useState<'independent' | 'race'>('independent');
  const [selectedRaceIds, setSelectedRaceIds] = useState<string[]>(() =>
    accounts.filter((a) => a.isValid && a.token).slice(0, 2).map((a) => a.id)
  );
  const [expandedTimelines, setExpandedTimelines] = useState<Record<string, boolean>>({});
  const [copiedText, setCopiadoText] = useState<string | null>(null);
  const [refreshingAccounts, setRefreshingAccounts] = useState<Record<string, boolean>>({});

  const toggleTimeline = (id: string) => {
    setExpandedTimelines((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleCopy = async (text: string, id?: string) => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopiadoText(id || text);
      setTimeout(() => setCopiadoText(null), 2000);
    }
  };

  const handleCopyBundle = async (endpoint: LiveEndpoint, id: string) => {
    const success = await copyEndpointBundle(endpoint, 'pair');
    if (success) {
      setCopiadoText(id);
      setTimeout(() => setCopiadoText(null), 2000);
    }
  };

  const handleRefresh = async (acc: KaggleAccount) => {
    setRefreshingAccounts((prev) => ({ ...prev, [acc.id]: true }));
    try {
      await onRefreshAccount(acc);
    } finally {
      setTimeout(() => {
        setRefreshingAccounts((prev) => ({ ...prev, [acc.id]: false }));
      }, 600);
    }
  };

  const toggleRaceSelect = (id: string) => {
    setSelectedRaceIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const getSessionForAccount = (accId: string) => {
    return sessions.find((s) => s.account === accId);
  };

  const anyEn ejecución = sessions.some((s) => !s.done);
  const isRaceActive = sessions.some((s) => !s.done && !!s.raceGroupId);

  // ponytail: static lookup table for status badges
  const BADGES: Record<string, { label: string; color: string; bg: string }> = {
    WINNER: { label: '🏆 Ganadora', color: 'var(--accent-emerald)', bg: 'rgba(16, 185, 129, 0.2)' },
    RUNNING: { label: 'En ejecución', color: 'var(--accent-cyan)', bg: 'rgba(6, 182, 212, 0.2)' },
    QUEUED: { label: 'En cola', color: 'var(--accent-amber)', bg: 'rgba(245, 158, 11, 0.2)' },
    CANCELLED: { label: 'Cancelada', color: 'var(--text-muted)', bg: 'rgba(100, 116, 139, 0.15)' },
    COMPLETE: { label: 'Finalizada', color: 'var(--accent-blue)', bg: 'rgba(59, 130, 246, 0.2)' },
    ERROR: { label: 'Error', color: 'var(--accent-rose)', bg: 'rgba(244, 63, 94, 0.2)' },
  };

  const getStatusBadge = (session?: RaceSession, isValid?: boolean) => {
    if (!session || session.status === 'IDLE') {
      return {
        label: isValid ? 'Lista' : 'Sin verificar',
        color: isValid ? 'var(--text-muted)' : 'var(--accent-rose)',
        bg: 'rgba(148, 163, 184, 0.1)',
      };
    }
    return BADGES[session.status] || BADGES.ERROR;
  };

  // ponytail: reusable timeline snippet
  const renderTimeline = (events: { time: string; text: string }[], limit: number = 10) => (
    <div className="timeline" style={{ marginTop: '12px' }}>
      {events.slice(-limit).reverse().map((ev, i) => (
        <div className="timeline-item" key={i}>
          <span className="timeline-time">{ev.time}</span>
          <span className="timeline-text">{ev.text}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ paddingBottom: '30px' }}>
      {/* Ready endpoints quick banner */}
      {endpoints.length > 0 && (
        <div
          className="glass-card"
          style={{
            background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.15))',
            borderColor: 'var(--accent-emerald)',
            marginBottom: '14px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-emerald)', fontWeight: 700 }}>
              <CheckCircle size={18} />
              <span>
                {endpoints.length === 1
                  ? `Endpoint TPU activo (${endpoints[0].accountName || 'Ready'})`
                  : `${endpoints.length} instancias TPU listas`}
              </span>
            </div>
            {activeEndpoint?.tokensPerSec && (
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {activeEndpoint.tokensPerSec} tok/s
              </span>
            )}
          </div>
          <div
            style={{
              fontSize: '12px',
              color: 'var(--text-secondary)',
              marginBottom: '10px',
              fontFamily: 'var(--font-mono)',
              wordBreak: 'break-all',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '8px',
            }}
          >
            <span>{activeEndpoint?.baseUrl || endpoints[0]?.baseUrl}</span>
            <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
              <button
                className="btn btn-secondary btn-sm"
                style={{ padding: '2px 6px', fontSize: '11px' }}
                title="Copy Base URL"
                onClick={() => handleCopy((activeEndpoint || endpoints[0]).baseUrl, 'top-url')}
              >
                {copiedText === 'top-url' ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                <span style={{ fontSize: '10px', marginLeft: '3px' }}>{copiedText === 'top-url' ? 'Copiado' : 'URL'}</span>
              </button>
              <button
                className="btn btn-secondary btn-sm"
                style={{ padding: '2px 6px', fontSize: '11px' }}
                title="Copiar clave API"
                onClick={() => handleCopy((activeEndpoint || endpoints[0]).apiKey, 'top-key')}
              >
                {copiedText === 'top-key' ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                <span style={{ fontSize: '10px', marginLeft: '3px' }}>{copiedText === 'top-key' ? 'Copiado' : 'Key'}</span>
              </button>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button
              className="btn btn-secondary btn-sm"
              style={{ display: 'flex', alignItems: 'center', gap: '5px' }}
              onClick={() => handleCopyBundle(activeEndpoint || endpoints[0], 'top-bundle')}
            >
              {copiedText === 'top-bundle' ? <Check size={14} color="var(--accent-emerald)" /> : <Copy size={14} />}
              <span>{copiedText === 'top-bundle' ? 'Copiado URL+Key' : 'Copiar URL + clave'}</span>
            </button>
            <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={() => onGoToChat(activeEndpoint)}>
              <MessageSquare size={14} />
              <span>Abrir chat</span>
            </button>
            <button className="btn btn-secondary btn-sm" onClick={() => onOpenExport(activeEndpoint)}>
              <Share2 size={14} />
              <span>Exportar configuración</span>
            </button>
          </div>
        </div>
      )}

      {/* Mode selection tabs */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '8px',
          padding: '0 4px',
          marginBottom: '14px',
        }}
      >
        <button
          className={`btn ${viewMode === 'independent' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
          style={{ padding: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
          onClick={() => setViewMode('independent')}
        >
          <Layers size={15} />
          <span>Monitor individual</span>
        </button>

        <button
          className={`btn ${viewMode === 'race' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
          style={{ padding: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
          onClick={() => setViewMode('race')}
        >
          <Zap size={15} />
          <span>⚡ Carrera multicuenta</span>
        </button>
      </div>

      {/* Model Selection and Parameters */}
      <div className="glass-card">
        <div className="card-title">
          <Cpu size={16} color="var(--accent-cyan)" />
          <span>Modelo y parámetros (TPU v5e-8)</span>
        </div>

        <div className="pill-grid">
          <div
            className={`pill-option ${config.model === 'qwen38-27b' ? 'active' : ''}`}
            onClick={() => onUpdateConfig({ ...config, model: 'qwen38-27b', reasoningEffort: config.reasoningEffort === 'low' ? 'xhigh' : config.reasoningEffort })}
          >
            <div className="pill-title">Qwen3.8-27B</div>
            <div className="pill-desc">bf16 unquantized · 262k context · vLLM-TPU · ~130 tok/s</div>
          </div>

          <div
            className={`pill-option ${config.model === 'glm53-flash' ? 'active' : ''}`}
            onClick={() => onUpdateConfig({ ...config, model: 'glm53-flash', reasoningEffort: config.reasoningEffort === 'xhigh' ? 'low' : config.reasoningEffort })}
          >
            <div className="pill-title">GLM-5.3-Flash</div>
            <div className="pill-desc">320B MoE · 3-bit/int8 · Custom JAX · ~64 tok/s</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '10px' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Esfuerzo de razonamiento</label>
            <select
              className="form-select"
              value={config.reasoningEffort}
              onChange={(e) =>
                onUpdateConfig({ ...config, reasoningEffort: e.target.value as ReasoningEffort })
              }
            >
              <option value="xhigh">xhigh (extremo · predeterminado Qwen)</option>
              <option value="high">high (profundo)</option>
              <option value="medium">medium (moderado)</option>
              <option value="low">low (rápido · predeterminado GLM)</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Modo de inicio rápido</label>
            <select
              className="form-select"
              value={config.fastStart ? 'true' : 'false'}
              onChange={(e) => onUpdateConfig({ ...config, fastStart: e.target.value === 'true' })}
            >
              <option value="false">Compilación estándar (más segura)</option>
              <option value="true">Omitir calentamiento (~4 min más rápido)</option>
            </select>
          </div>
        </div>
      </div>

      {/* ======================= Mode 1: Independent Key Management ======================= */}
      {viewMode === 'independent' && (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 6px',
              marginBottom: '8px',
            }}
          >
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Estado de cuentas ({accounts.length} cuentas)
            </span>
            {anyEn ejecución && (
              <button
                className="btn btn-danger btn-sm"
                style={{ padding: '4px 10px', fontSize: '11px' }}
                onClick={onStopAll}
              >
                <StopCircle size={12} />
                <span>Detener todas</span>
              </button>
            )}
          </div>

          {accounts.length === 0 ? (
            <div className="glass-card" style={{ textAlign: 'center', padding: '30px' }}>
              <AlertTriangle size={32} color="var(--accent-amber)" style={{ margin: '0 auto 10px' }} />
              <div style={{ fontWeight: 600, marginBottom: '6px' }}>No hay cuentas de Kaggle configuradas</div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Abrí Ajustes y agregá tu token API de Kaggle.
              </div>
            </div>
          ) : (
            accounts.map((acc) => {
              const session = getSessionForAccount(acc.id);
              const isSessionActive = session && !session.done;
              const isReady = session?.endpoint?.status === 'READY';
              const badge = getStatusBadge(session, acc.isValid);
              const isExpanded = expandedTimelines[acc.id] || false;

              return (
                <div
                  key={acc.id}
                  className="glass-card"
                  style={{
                    borderColor: isReady
                      ? 'var(--accent-emerald)'
                      : isSessionActive
                      ? 'var(--accent-cyan)'
                      : 'var(--border-color)',
                    transition: 'border-color 0.2s ease',
                  }}
                >
                  {/* Card Header */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '10px',
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span>{acc.name || 'Cuenta sin nombre'}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>
                          (@{acc.username || 'Sin verificar'})
                        </span>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: '4px',
                          color: badge.color,
                          backgroundColor: badge.bg,
                        }}
                      >
                        {badge.label}
                      </span>
                    </div>
                  </div>

                  {/* Status details & latest log */}
                  {session && session.events.length > 0 ? (
                    <div
                      style={{
                        background: 'rgba(15, 23, 42, 0.6)',
                        padding: '10px 12px',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '12px',
                        marginBottom: '10px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '4px' }}>
                        <span>Última actualización ({session.model || config.model})</span>
                        <span>{session.events[session.events.length - 1].time}</span>
                      </div>
                      <div style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                        {session.events[session.events.length - 1].text}
                      </div>

                      {/* Ready endpoint quick copy and entry */}
                      {session.endpoint && session.endpoint.status === 'READY' ? (
                        <div
                          style={{
                            marginTop: '8px',
                            paddingTop: '8px',
                            borderTop: '1px solid var(--border-color)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            flexWrap: 'wrap',
                            gap: '6px',
                          }}
                        >
                          <div style={{ fontSize: '11px', color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
                            {session.endpoint.tokensPerSec ? `${session.endpoint.tokensPerSec} tok/s · ` : ''}
                            En línea y lista
                          </div>
                          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ padding: '3px 8px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '3px' }}
                              title="Copy both BaseURL and API Key"
                              onClick={() => handleCopyBundle(session.endpoint!, `card-bundle-${acc.id}`)}
                            >
                              {copiedText === `card-bundle-${acc.id}` ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                              <span>{copiedText === `card-bundle-${acc.id}` ? 'Copiado' : 'Copiar URL+clave'}</span>
                            </button>
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ padding: '3px 8px', fontSize: '11px' }}
                              onClick={() => handleCopy(session.endpoint!.baseUrl, `card-url-${acc.id}`)}
                            >
                              {copiedText === `card-url-${acc.id}` ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                              <span>{copiedText === `card-url-${acc.id}` ? 'Copiado' : 'Copiar URL'}</span>
                            </button>
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ padding: '3px 8px', fontSize: '11px' }}
                              onClick={() => handleCopy(session.endpoint!.apiKey, `card-key-${acc.id}`)}
                            >
                              {copiedText === `card-key-${acc.id}` ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                              <span>{copiedText === `card-key-${acc.id}` ? 'Copiado' : 'Copiar clave'}</span>
                            </button>
                            <button
                              className="btn btn-primary btn-sm"
                              style={{ padding: '3px 8px', fontSize: '11px' }}
                              onClick={() => onGoToChat(session.endpoint)}
                            >
                              <MessageSquare size={12} />
                              <span>Chat</span>
                            </button>
                          </div>
                        </div>
                      ) : session.apiKey ? (
                        <div
                          style={{
                            marginTop: '8px',
                            paddingTop: '8px',
                            borderTop: '1px solid var(--border-color)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            flexWrap: 'wrap',
                            gap: '6px',
                          }}
                        >
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            Key: {session.apiKey.substring(0, 8)}...{session.apiKey.substring(session.apiKey.length - 4)}
                          </div>
                          <div style={{ display: 'flex', gap: '6px' }}>
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ padding: '3px 8px', fontSize: '11px' }}
                              onClick={() => handleCopy(session.apiKey, `card-key-${acc.id}`)}
                            >
                              {copiedText === `card-key-${acc.id}` ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                              <span>{copiedText === `card-key-${acc.id}` ? 'Copiado' : 'Copiar clave API'}</span>
                            </button>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                      {acc.isValid ? 'La instancia está lista para iniciar' : 'Cuenta sin verificar. Probala y guardala en Ajustes.'}
                    </div>
                  )}

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {isSessionActive ? (
                      <button
                        className="btn btn-danger btn-sm"
                        style={{ flex: 1 }}
                        onClick={() => onStopAccount(acc.id)}
                      >
                        <StopCircle size={14} />
                        <span>Detener instancia</span>
                      </button>
                    ) : (
                      <button
                        className="btn btn-primary btn-sm"
                        style={{ flex: 1 }}
                        disabled={!acc.token || isLaunching}
                        onClick={() => onStartSingle(acc)}
                      >
                        <Play size={14} />
                        <span>Iniciar TPU</span>
                      </button>
                    )}

                    <button
                      className="btn btn-secondary btn-sm"
                      title="Sincronizar estado"
                      onClick={() => handleRefresh(acc)}
                      disabled={refreshingAccounts[acc.id]}
                    >
                      <RefreshCw size={14} className={refreshingAccounts[acc.id] ? 'spin-icon' : ''} />
                      <span>{refreshingAccounts[acc.id] ? 'Sincronizando...' : 'Sincronizar'}</span>
                    </button>

                    {session && session.events.length > 0 && (
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => toggleTimeline(acc.id)}
                        title="Expandir/contraer registro"
                      >
                        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        <span>Registros ({session.events.length})</span>
                      </button>
                    )}
                  </div>

                  {/* Expanded timeline */}
                  {isExpanded && session && session.events.length > 0 && renderTimeline(session.events, 10)}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* ======================= Mode 2: Multi-Account Preemptive Race Mode ======================= */}
      {viewMode === 'race' && (
        <div>
          {/* Race description banner */}
          <div
            className="glass-card"
            style={{
              background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(6, 182, 212, 0.1))',
              borderColor: 'var(--accent-amber)',
              marginBottom: '14px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-amber)', fontWeight: 700, marginBottom: '6px' }}>
              <Zap size={18} />
              <span>Carrera preventiva entre cuentas</span>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Las cuentas seleccionadas entrarán en cola a la vez para conseguir una TPU.
              <strong> En cuanto una quede lista, las demás colas se cancelan automáticamente</strong>,
              para cuidar la cuota semanal de TPU.
            </div>
          </div>

          {/* Account checklist */}
          <div className="glass-card" style={{ marginBottom: '14px' }}>
            <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Seleccionar cuentas para la carrera ({selectedRaceIds.length} seleccionadas)</span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Elegí al menos 2</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {accounts.map((acc) => {
                const isSelected = selectedRaceIds.includes(acc.id);
                const session = getSessionForAccount(acc.id);
                const badge = getStatusBadge(session, acc.isValid);

                return (
                  <div
                    key={acc.id}
                    onClick={() => !isRaceActive && toggleRaceSelect(acc.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-sm)',
                      background: isSelected ? 'rgba(59, 130, 246, 0.12)' : 'rgba(15, 23, 42, 0.5)',
                      border: isSelected ? '1px solid var(--accent-blue)' : '1px solid var(--border-color)',
                      cursor: isRaceActive ? 'not-allowed' : 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      {isSelected ? (
                        <CheckSquare size={18} color="var(--accent-cyan)" />
                      ) : (
                        <Square size={18} color="var(--text-muted)" />
                      )}
                      <div>
                        <div style={{ fontWeight: 600, fontSize: '13px' }}>{acc.name}</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          @{acc.username || 'usuario no configurado'} · {acc.isValid ? 'Verified' : 'Sin verificar'}
                        </div>
                      </div>
                    </div>

                    <div>
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: '4px',
                          color: badge.color,
                          backgroundColor: badge.bg,
                        }}
                      >
                        {badge.label}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Race action button */}
          <div style={{ padding: '0 4px', marginBottom: '14px' }}>
            {isRaceActive ? (
              <button className="btn btn-danger" style={{ width: '100%', padding: '14px' }} onClick={onStopAll}>
                <StopCircle size={18} />
                <span>Detener carrera</span>
              </button>
            ) : (
              <button
                className="btn btn-primary"
                style={{ width: '100%', padding: '14px' }}
                disabled={selectedRaceIds.length < 2 || isLaunching}
                onClick={() => {
                  const selectedAccs = accounts.filter((a) => selectedRaceIds.includes(a.id));
                  onStartRace(selectedAccs);
                }}
              >
                <Zap size={18} />
                <span>
                  {selectedRaceIds.length < 2
                    ? 'Elegí al menos 2 valid accounts to start race'
                    : `⚡ Iniciar carrera con ${selectedRaceIds.length} cuentas`}
                </span>
              </button>
            )}
          </div>

          {/* Race timeline */}
          {sessions.some((s) => !!s.raceGroupId) && (
            <div className="glass-card">
              <div className="card-title">
                <Clock size={16} color="var(--accent-amber)" />
                <span>Cronología y estado de la carrera</span>
              </div>

              {renderTimeline(
                sessions.filter((s) => !!s.raceGroupId).flatMap((s) => s.events),
                12
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
