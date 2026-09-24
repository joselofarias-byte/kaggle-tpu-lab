import React, { useState, useEffect } from 'react';
import { KaggleAccount, LaunchConfig } from '../services/types';
import { KaggleApi } from '../services/kaggle';
import {
  UserCheck,
  Shield,
  CheckCircle,
  AlertCircle,
  Save,
  Sliders,
  Plus,
  Trash2,
  Key,
  Eye,
  EyeOff,
  Bell,
  RefreshCw,
} from 'lucide-react';
import { Preferences } from '@capacitor/preferences';
import { Capacitor } from '@capacitor/core';
import { secretStore } from '../services/secureStore';
import {
  isBatteryOptimizationIgnored,
  requestBatteryOptimizationExemption,
} from '../services/queueMonitor';

interface SettingsViewProps {
  accounts: KaggleAccount[];
  config: LaunchConfig;
  onUpdateAccounts: (accounts: KaggleAccount[]) => void;
  onUpdateConfig: (config: LaunchConfig) => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  accounts,
  config,
  onUpdateAccounts,
  onUpdateConfig,
}) => {
  const [localAccounts, setLocalAccounts] = useState<KaggleAccount[]>(() => {
    if (!accounts || accounts.length === 0) {
      return [
        {
          id: 'acc-' + Date.now(),
          name: 'Cuenta principal',
          token: '',
          username: '',
          isValid: false,
          isChecking: false,
        },
      ];
    }
    return accounts;
  });

  const [localConfig, setLocalConfig] = useState<LaunchConfig>(config);
  const [showTokens, setMostrarTokens] = useState<Record<string, boolean>>({});
  const [saveSuccess, setSaveSuccess] = useState(false);
  // null = checking / unavailable on web
  const [batteryIgnored, setBatteryIgnored] = useState<boolean | null>(null);

  const refreshBatteryStatus = async () => {
    if (!Capacitor.isNativePlatform()) {
      setBatteryIgnored(null);
      return;
    }
    try {
      setBatteryIgnored(await isBatteryOptimizationIgnored());
    } catch {
      setBatteryIgnored(null);
    }
  };

  useEffect(() => {
    refreshBatteryStatus();
  }, []);

  const handleRequestBatteryExemption = async () => {
    try {
      await requestBatteryOptimizationExemption();
    } catch {
      // System dialog unavailable — user can enable it manually.
    }
    // Re-check when the user comes back (dialog result is not delivered).
    setTimeout(refreshBatteryStatus, 1500);
  };

  const toggleMostrarToken = (id: string) => {
    setMostrarTokens((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleAccountChange = (id: string, field: keyof KaggleAccount, value: string) => {
    setLocalAccounts((prev) =>
      prev.map((acc) =>
        acc.id === id
          ? { ...acc, [field]: value, isValid: field === 'token' ? false : acc.isValid, error: undefined }
          : acc
      )
    );
  };

  const handleAddAccount = () => {
    const newAcc: KaggleAccount = {
      id: 'acc-' + Date.now() + '-' + Math.floor(Math.random() * 1000),
      name: `Cuenta ${localAccounts.length + 1}`,
      token: '',
      username: '',
      isValid: false,
      isChecking: false,
    };
    const updated = [...localAccounts, newAcc];
    setLocalAccounts(updated);
    onUpdateAccounts(updated);
  };

  const handleRemoveAccount = (id: string) => {
    if (localAccounts.length <= 1) {
      alert('Debe quedar configurada al menos una cuenta.');
      return;
    }
    const updated = localAccounts.filter((a) => a.id !== id);
    setLocalAccounts(updated);
    onUpdateAccounts(updated);
  };

  const verifyAccount = async (id: string) => {
    const target = localAccounts.find((a) => a.id === id);
    if (!target || !target.token.trim()) return;

    setLocalAccounts((prev) =>
      prev.map((acc) => (acc.id === id ? { ...acc, isChecking: true, error: undefined } : acc))
    );

    const api = new KaggleApi(target.token, target.username);
    const res = await api.verifyAuth();

    setLocalAccounts((prev) => {
      const next = prev.map((acc) => {
        if (acc.id !== id) return acc;
        return {
          ...acc,
          isChecking: false,
          isValid: res.isValid,
          username: res.isValid && res.username ? res.username : acc.username,
          error: res.isValid ? undefined : res.error || 'Falló la verificación de autenticación',
        };
      });
      onUpdateAccounts(next);
      return next;
    });
  };

  const handleSaveAll = async () => {
    onUpdateAccounts(localAccounts);
    onUpdateConfig(localConfig);
    await secretStore.set('ktl_accounts', JSON.stringify(localAccounts));
    try {
      await Preferences.set({ key: 'ktl_config', value: JSON.stringify(localConfig) });
    } catch {
      localStorage.setItem('ktl_config', JSON.stringify(localConfig));
    }
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2000);
  };

  return (
    <div style={{ paddingBottom: '30px' }}>
      {/* Top title and add button */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 4px',
          marginBottom: '12px',
        }}
      >
        <div>
          <div style={{ fontWeight: 700, fontSize: '15px' }}>Cuentas de Kaggle</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Configurá una o varias cuentas con autenticación y control independientes.
          </div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={handleAddAccount}>
          <Plus size={14} />
          <span>Agregar cuenta</span>
        </button>
      </div>

      {/* Account cards */}
      {localAccounts.map((acc, index) => (
        <div className="glass-card" key={acc.id} style={{ position: 'relative' }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Key size={16} color="var(--accent-cyan)" />
            <span style={{ fontWeight: 600 }}>{acc.name || `Account #${index + 1}`}</span>

            {acc.isValid ? (
              <span
                style={{
                  marginLeft: 'auto',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  color: 'var(--accent-emerald)',
                  fontSize: '11px',
                  fontWeight: 600,
                }}
              >
                <CheckCircle size={13} /> Verificada
              </span>
            ) : acc.error ? (
              <span
                style={{
                  marginLeft: 'auto',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  color: 'var(--accent-rose)',
                  fontSize: '11px',
                }}
              >
                <AlertCircle size={13} /> Falló acceso
              </span>
            ) : (
              <span
                style={{
                  marginLeft: 'auto',
                  color: 'var(--text-muted)',
                  fontSize: '11px',
                }}
              >
                Sin verificar
              </span>
            )}

            {localAccounts.length > 1 && (
              <button
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: '4px',
                  marginLeft: '6px',
                  display: 'flex',
                  alignItems: 'center',
                }}
                title="Eliminar esta cuenta"
                onClick={() => handleRemoveAccount(acc.id)}
              >
                <Trash2 size={14} />
              </button>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div className="form-group">
              <label className="form-label">Nombre de la cuenta</label>
              <input
                className="form-input"
                placeholder="Ej.: Cuenta principal / Respaldo 2"
                value={acc.name}
                onChange={(e) => handleAccountChange(acc.id, 'name', e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Usuario de Kaggle</label>
              <input
                className="form-input"
                placeholder="e.g. your_username"
                value={acc.username}
                onChange={(e) => handleAccountChange(acc.id, 'username', e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label">Token / clave API de Kaggle</label>
              <button
                type="button"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: '11px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  marginBottom: '4px',
                }}
                onClick={() => toggleMostrarToken(acc.id)}
              >
                {showTokens[acc.id] ? <EyeOff size={12} /> : <Eye size={12} />}
                <span>{showTokens[acc.id] ? 'Ocultar' : 'Mostrar'}</span>
              </button>
            </div>
            <input
              className="form-input"
              type={showTokens[acc.id] ? 'text' : 'password'}
              placeholder="KGAT_... or API Key (from kaggle.json)"
              value={acc.token}
              onChange={(e) => handleAccountChange(acc.id, 'token', e.target.value)}
            />
          </div>

          {acc.error && (
            <div
              style={{
                color: 'var(--accent-rose)',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                marginBottom: '10px',
              }}
            >
              <AlertCircle size={13} />
              <span>{acc.error}</span>
            </div>
          )}

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => verifyAccount(acc.id)}
              disabled={acc.isChecking || !acc.token.trim()}
              style={{ flex: 1 }}
            >
              <UserCheck size={14} />
              <span>{acc.isChecking ? 'Verificando credenciales...' : 'Probar y verificar credenciales'}</span>
            </button>
          </div>
        </div>
      ))}

      <div className="glass-card" style={{ borderColor: 'rgba(16, 185, 129, 0.35)' }}>
        <div className="card-title">
          <Shield size={16} color="var(--accent-emerald)" />
          <span>Protección de credenciales</span>
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          Los tokens y las claves de sesión se cifran con Android Keystore. Para evitar pérdidas de datos,
          se conserva temporalmente un ancla privada hasta comprobar que el cifrado sobrevivió a un reinicio.
        </div>
      </div>

      {/* Defaults */}
      <div className="glass-card">
        <div className="card-title">
          <Sliders size={16} color="var(--accent-indigo)" />
          <span>Política de inicio</span>
        </div>

        <div className="form-group">
          <label className="form-label">Tiempo máximo activo (minutos; 480 = 8 horas)</label>
          <input
            className="form-input"
            type="number"
            value={localConfig.keepaliveMin}
            onChange={(e) =>
              setLocalConfig({
                ...localConfig,
                keepaliveMin: parseInt(e.target.value, 10) || 480,
              })
            }
          />
        </div>
      </div>

      {/* Background notifications */}
      <div className="glass-card">
        <div className="card-title">
          <Bell size={16} color="var(--accent-cyan)" />
          <span>Notificaciones en segundo plano</span>
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '10px', lineHeight: 1.5 }}>
          Cuando la app queda en segundo plano, un servicio sigue la cola de TPU y avisa cuando está lista.
          La optimización de batería puede suspenderlo; conviene excluir la app.
        </div>
        {!Capacitor.isNativePlatform() ? (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Los ajustes de batería sólo están disponibles en Android.
          </div>
        ) : batteryIgnored === null ? (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Comprobando optimización de batería…</div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            {batteryIgnored ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-emerald)', fontSize: '12px', fontWeight: 600 }}>
                <CheckCircle size={13} /> Excluida — las alertas en segundo plano funcionarán
              </span>
            ) : (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-rose)', fontSize: '12px', fontWeight: 600 }}>
                <AlertCircle size={13} /> No excluida — Android puede suspender las alertas
              </span>
            )}
            <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
              {!batteryIgnored && (
                <button className="btn btn-secondary btn-sm" onClick={handleRequestBatteryExemption}>
                  <Shield size={14} />
                  <span>Solicitar exclusión</span>
                </button>
              )}
              <button className="btn btn-secondary btn-sm" onClick={refreshBatteryStatus} title="Refresh status">
                <RefreshCw size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Save Button */}
      <div style={{ padding: '0 14px' }}>
        <button
          className="btn btn-primary"
          style={{ width: '100%', padding: '12px' }}
          onClick={handleSaveAll}
        >
          <Save size={16} />
          <span>{saveSuccess ? '✓ Configuración guardada' : 'Guardar configuración'}</span>
        </button>
      </div>
    </div>
  );
};
