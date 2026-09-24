import React, { useState, useEffect, useRef } from 'react';
import { KaggleAccount, LaunchConfig, RaceSession, LiveEndpoint } from './services/types';
import { InstanceManager } from './services/racer';
import { DashboardView } from './components/DashboardView';
import { ChatView } from './components/ChatView';
import { SettingsView } from './components/SettingsView';
import { ExportModal } from './components/ExportModal';
import { Cpu, LayoutDashboard, MessageSquare, Settings } from 'lucide-react';
import { clearKaggleCookies } from './services/kaggle';
import { Preferences } from '@capacitor/preferences';
import { secretStore } from './services/secureStore';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'chat' | 'settings'>('dashboard');
  const [accounts, setAccounts] = useState<KaggleAccount[]>([
    {
      id: 'acc-main',
      name: 'Cuenta principal',
      token: '',
      username: '',
      isValid: false,
      isChecking: false,
    },
  ]);

  const [config, setConfig] = useState<LaunchConfig>({
    model: 'qwen38-27b',
    reasoningEffort: 'xhigh',
    maxModelLen: 262144,
    maxNumSeqs: 4,
    mtpTokens: 3,
    streams: 4,
    keepaliveMin: 480,
    textOnly: false,
    fastStart: false,
  });

  const [sessions, setSessions] = useState<RaceSession[]>([]);
  const [endpoints, setEndpoints] = useState<LiveEndpoint[]>([]);
  const [liveEndpoint, setLiveEndpoint] = useState<LiveEndpoint | undefined>(undefined);
  const [selectedExportEndpoint, setSelectedExportEndpoint] = useState<LiveEndpoint | undefined>(undefined);
  const [isLaunching, setIsLaunching] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);

  const instanceManagerRef = useRef<InstanceManager | null>(null);

  useEffect(() => {
    // Clear stale WebView cookies that might cause api.kaggle.com 401 Unauthenticated
    clearKaggleCookies().catch(() => {});

    // Initialize multi-key instance manager
    instanceManagerRef.current = new InstanceManager((newSessions, activeEp, allEndpoints) => {
      setSessions(newSessions);
      setEndpoints(allEndpoints || []);
      if (activeEp && activeEp.status === 'READY' && activeEp.baseUrl?.startsWith('http')) {
        setLiveEndpoint(activeEp);
      } else if (allEndpoints && allEndpoints.length > 0) {
        setLiveEndpoint(allEndpoints[0]);
      } else {
        setLiveEndpoint(undefined);
      }

      // Las sesiones contienen API keys del endpoint: almacenamiento protegido.
      secretStore.set('ktl_sessions', JSON.stringify(newSessions)).catch(() => {});
    });

    // Read stored config, migrate legacy account data and restore sessions
    (async () => {
      try {
        await secretStore.migrateLegacy().catch(() => {});
        let accList: any = null;
        const accRaw = await secretStore.get('ktl_accounts');
        if (accRaw) accList = JSON.parse(accRaw);

        const normalizeAccountName = (name: any, index: number): string => {
          if (!name || typeof name !== 'string') {
            return index === 0 ? 'Cuenta principal' : `Cuenta ${index + 1}`;
          }
          const trimmed = name.trim();
          if (trimmed === '主账号' || trimmed === '账号 1' || trimmed === '账号1') {
            return 'Cuenta principal';
          }
          if (trimmed === '账号 2' || trimmed === '账号2') {
            return 'Cuenta 2';
          }
          const match = trimmed.match(/^账号\s*(\d+)$/);
          if (match) {
            return `Cuenta ${match[1]}`;
          }
          return trimmed;
        };

        let finalAccounts: KaggleAccount[] = accounts;
        if (Array.isArray(accList) && accList.length > 0) {
          finalAccounts = accList.map((a: any, i: number) => ({
            id: a.id || `acc-${i + 1}-${Date.now()}`,
            name: normalizeAccountName(a.name, i),
            token: a.token || '',
            username: a.username || '',
            isValid: Boolean(a.isValid),
            isChecking: false,
            error: a.error,
          }));
          setAccounts(finalAccounts);
          // Guardar la versión normalizada en almacenamiento protegido.
          secretStore.set('ktl_accounts', JSON.stringify(finalAccounts)).catch(() => {});
        }

        const cfgData = await Preferences.get({ key: 'ktl_config' });
        if (cfgData.value) {
          setConfig(JSON.parse(cfgData.value));
        } else {
          const raw = localStorage.getItem('ktl_config');
          if (raw) setConfig(JSON.parse(raw));
        }

        // Las sesiones incluyen API keys: leerlas desde almacenamiento protegido.
        let savedSessions: RaceSession[] = [];
        try {
          const sessRaw = await secretStore.get('ktl_sessions');
          if (sessRaw) savedSessions = JSON.parse(sessRaw);
        } catch {}

        if (Array.isArray(savedSessions) && savedSessions.length > 0) {
          const activeAccIds = new Set(finalAccounts.map((a) => a.id));
          savedSessions = savedSessions
            .filter((s) => activeAccIds.has(s.account))
            .map((s, idx) => ({
              ...s,
              accountName: normalizeAccountName(s.accountName, idx),
            }));
          if (instanceManagerRef.current) {
            instanceManagerRef.current.restoreSessions(savedSessions);
          }
          secretStore.set('ktl_sessions', JSON.stringify(savedSessions)).catch(() => {});
        }

        // Silently probe cloud status for all accounts on boot
        if (Array.isArray(finalAccounts)) {
          for (const acc of finalAccounts) {
            if (acc.token && instanceManagerRef.current) {
              instanceManagerRef.current.refreshAccount(acc).catch(() => {});
            }
          }
        }
      } catch {
        // Use defaults
      }
    })();
  }, []);

  const handleStartSingle = async (account: KaggleAccount) => {
    if (!instanceManagerRef.current) return;
    setIsLaunching(true);
    try {
      await instanceManagerRef.current.startSingle(account, config);
    } catch (err: any) {
      alert(err.message || String(err));
    } finally {
      setIsLaunching(false);
    }
  };

  const handleStartRace = async (selectedAccounts: KaggleAccount[]) => {
    if (!instanceManagerRef.current) return;
    setIsLaunching(true);
    try {
      await instanceManagerRef.current.startRace(selectedAccounts, config);
    } catch (err: any) {
      alert(err.message || String(err));
    } finally {
      setIsLaunching(false);
    }
  };

  const handleStopAccount = async (accountId: string) => {
    if (!instanceManagerRef.current) return;
    await instanceManagerRef.current.stopAccount(accountId);
  };

  const handleStopAll = async () => {
    if (!instanceManagerRef.current) return;
    await instanceManagerRef.current.stop();
  };

  const handleRefreshAccount = async (account: KaggleAccount) => {
    if (!instanceManagerRef.current) return;
    await instanceManagerRef.current.refreshAccount(account);
  };

  const handleGoToChat = (endpoint?: LiveEndpoint) => {
    if (endpoint) {
      setLiveEndpoint(endpoint);
    }
    setActiveTab('chat');
  };

  const handleOpenExport = (endpoint?: LiveEndpoint) => {
    setSelectedExportEndpoint(endpoint || liveEndpoint || endpoints[0]);
    setShowExportModal(true);
  };

  const getStatusInfo = () => {
    if (endpoints.length > 0) {
      return { text: endpoints.length === 1 ? 'TPU lista' : `${endpoints.length} TPU listas`, cls: 'ready' };
    }
    const active = sessions.filter((s) => !s.done);
    if (active.some((s) => s.status === 'QUEUED') && !active.some((s) => s.status === 'RUNNING' || s.status === 'WINNER')) {
      return { text: 'En cola', cls: 'running' };
    }
    if (active.some((s) => s.endpoint?.status === 'OFFLINE')) {
      return { text: 'Sin conexión', cls: 'offline' };
    }
    if (active.length > 0) {
      return { text: 'Iniciando', cls: 'running' };
    }
    return { text: 'Sin conexión', cls: 'offline' };
  };

  const statusInfo = getStatusInfo();

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-badge">
          <div className="brand-logo">
            <Cpu size={20} />
          </div>
          <div>
            <div className="brand-title">Kaggle TPU Lab</div>
            <div className="brand-sub">Controlador TPU v5e-8</div>
          </div>
        </div>

        <div className="header-status">
          <span className={`status-dot ${statusInfo.cls}`}></span>
          <span>{statusInfo.text}</span>
        </div>
      </header>

      {/* Main Tab Content */}
      <main className={`app-content ${activeTab === 'chat' ? 'app-content-chat' : ''}`}>
        {activeTab === 'dashboard' && (
          <DashboardView
            accounts={accounts}
            config={config}
            sessions={sessions}
            endpoints={endpoints}
            activeEndpoint={liveEndpoint}
            isLaunching={isLaunching}
            onUpdateConfig={setConfig}
            onStartSingle={handleStartSingle}
            onStartRace={handleStartRace}
            onStopAccount={handleStopAccount}
            onStopAll={handleStopAll}
            onRefreshAccount={handleRefreshAccount}
            onGoToChat={handleGoToChat}
            onOpenExport={handleOpenExport}
          />
        )}

        {activeTab === 'chat' && (
          <ChatView
            endpoints={endpoints}
            liveEndpoint={liveEndpoint}
            onSelectEndpoint={setLiveEndpoint}
            onGoToDashboard={() => setActiveTab('dashboard')}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsView
            accounts={accounts}
            config={config}
            onUpdateAccounts={setAccounts}
            onUpdateConfig={setConfig}
          />
        )}
      </main>

      {/* Export Modal */}
      {showExportModal && selectedExportEndpoint && (
        <ExportModal endpoint={selectedExportEndpoint} onClose={() => setShowExportModal(false)} />
      )}

      {/* Bottom Nav TabBar */}
      <nav className="tab-bar">
        <div
          className={`tab-item ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <LayoutDashboard size={20} />
          <span>Panel</span>
        </div>

        <div
          className={`tab-item ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          <MessageSquare size={20} />
          <span>Chat</span>
        </div>

        <div
          className={`tab-item ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          <Settings size={20} />
          <span>Ajustes</span>
        </div>
      </nav>
    </div>
  );
};
