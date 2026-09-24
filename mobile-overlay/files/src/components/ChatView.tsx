import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, LiveEndpoint, getServedModelName } from '../services/types';
import { ChatService } from '../services/chat';
import { copyToClipboard, copyEndpointBundle } from '../services/clipboard';
import {
  Send,
  Square,
  Trash2,
  Brain,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Cpu,
  SlidersHorizontal,
  ArrowDown,
  Sparkles,
  RefreshCw,
  X,
  Globe,
  Key,
  Code2,
  ChevronUp,
} from 'lucide-react';

interface ChatViewProps {
  endpoints?: LiveEndpoint[];
  liveEndpoint?: LiveEndpoint;
  onSelectEndpoint?: (ep: LiveEndpoint) => void;
  onGoToDashboard: () => void;
}

interface MessageTelemetry {
  durationSec: number;
  tokensPerSec?: string;
  charCount: number;
}

export const ChatView: React.FC<ChatViewProps> = ({
  endpoints = [],
  liveEndpoint,
  onSelectEndpoint,
  onGoToDashboard,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-1',
      role: 'assistant',
      content:
        '¡Hola! Soy un modelo ejecutándose en una TPU v5e-8 de Kaggle. Cuando la instancia esté lista, podés usarme para código, análisis y razonamiento.',
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showThinking, setShowThinking] = useState<Record<string, boolean>>({});
  const [copiedCode, setCopiadoCode] = useState<string | null>(null);
  const [copiedKey, setCopiadoKey] = useState<string | null>(null);
  const [showEndpointDrawer, setShowEndpointDrawer] = useState(false);
  const [manualUrl, setManualUrl] = useState('');
  const [customBaseUrl, setCustomBaseUrl] = useState('');

  // Expand/collapse long code blocks
  const [expandedCodeBlocks, setExpandedCodeBlocks] = useState<Record<string, boolean>>({});

  // Mobile scroll & stick-to-bottom state
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [hasNewUnread, setHasNewUnread] = useState(false);

  // Deep thinking timer & telemetry
  const [thinkingTimer, setThinkingTimer] = useState<number>(0);
  const [messageTelemetry, setMessageTelemetry] = useState<Record<string, MessageTelemetry>>({});

  const messagesListRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatServiceRef = useRef<ChatService>(new ChatService());
  const generationStartTimeRef = useRef<number>(0);
  const thinkingTimerIntervalRef = useRef<any>(null);

  // Pick ready online endpoint with valid http address
  const baseEp =
    liveEndpoint && liveEndpoint.status === 'READY' && liveEndpoint.baseUrl?.startsWith('http')
      ? liveEndpoint
      : endpoints.find((e) => e.status === 'READY' && e.baseUrl?.startsWith('http')) || liveEndpoint || endpoints[0];

  const currentEndpoint: LiveEndpoint | undefined = customBaseUrl
    ? {
        accountId: baseEp?.accountId || 'custom',
        accountName: 'Endpoint personalizado',
        baseUrl: customBaseUrl.endsWith('/v1') ? customBaseUrl : `${customBaseUrl}/v1`,
        apiKey: baseEp?.apiKey || 'sk-custom',
        model: getServedModelName(baseEp?.model || 'qwen3.8-27b'),
        status: 'READY',
      }
    : baseEp
    ? {
        ...baseEp,
        model: getServedModelName(baseEp.model),
      }
    : undefined;

  const isEndpointReady = Boolean(
    currentEndpoint && currentEndpoint.status === 'READY' && currentEndpoint.baseUrl?.startsWith('http')
  );

  // Monitor scroll position
  const handleScroll = useCallback(() => {
    if (!messagesListRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesListRef.current;
    const distanceToBottom = scrollHeight - scrollTop - clientHeight;
    const atBottom = distanceToBottom < 80;
    setIsAtBottom(atBottom);
    if (atBottom) {
      setHasNewUnread(false);
    }
  }, []);

  const scrollToBottom = useCallback((smooth = true) => {
    if (!messagesEndRef.current) return;
    messagesEndRef.current.scrollIntoView({
      behavior: smooth ? 'smooth' : 'auto',
      block: 'end',
    });
    setIsAtBottom(true);
    setHasNewUnread(false);
  }, []);

  useEffect(() => {
    if (isAtBottom) {
      scrollToBottom(false);
    } else if (isGenerating) {
      setHasNewUnread(true);
    }
  }, [messages, isGenerating, isAtBottom, scrollToBottom]);

  // Thinking timer (seconds)
  useEffect(() => {
    if (isGenerating) {
      setThinkingTimer(0);
      thinkingTimerIntervalRef.current = setInterval(() => {
        setThinkingTimer((t) => t + 1);
      }, 1000);
    } else {
      if (thinkingTimerIntervalRef.current) {
        clearInterval(thinkingTimerIntervalRef.current);
        thinkingTimerIntervalRef.current = null;
      }
    }
    return () => {
      if (thinkingTimerIntervalRef.current) {
        clearInterval(thinkingTimerIntervalRef.current);
      }
    };
  }, [isGenerating]);

  const adjustTextareaHeight = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 110)}px`;
  };

  const handleSendPrompt = async (promptText: string) => {
    if (!promptText.trim() || isGenerating) return;
    if (!isEndpointReady || !currentEndpoint) {
      setShowEndpointDrawer(true);
      return;
    }

    const userMsg: ChatMessage = {
      id: 'msg-' + Date.now(),
      role: 'user',
      content: promptText.trim(),
      timestamp: Date.now(),
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    setIsGenerating(true);
    generationStartTimeRef.current = Date.now();

    const assistantMsgId = 'msg-ast-' + (Date.now() + 1);
    const initialAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoningContent: '',
      timestamp: Date.now(),
      isStreaming: true,
    };

    setMessages([...newMessages, initialAssistantMsg]);
    setShowThinking((prev) => ({ ...prev, [assistantMsgId]: true }));
    scrollToBottom(true);

    let charReceived = 0;
    let accumulatedText = '';
    let accumulatedReasoning = '';

    await chatServiceRef.current.sendMessage(
      currentEndpoint,
      newMessages,
      '',
      (delta, reasoningDelta) => {
        if (delta) {
          charReceived += delta.length;
          accumulatedText += delta;
        }
        if (reasoningDelta) {
          accumulatedReasoning += reasoningDelta;
        }

        let displayContent = accumulatedText;
        let displayReasoning = accumulatedReasoning;

        // Handle <think> tags embedded inside standard content stream
        if (displayContent.includes('<think>')) {
          if (displayContent.includes('</think>')) {
            const thinkStart = displayContent.indexOf('<think>');
            const thinkEnd = displayContent.indexOf('</think>');
            const embedded = displayContent.slice(thinkStart + 7, thinkEnd).trim();
            displayReasoning = (displayReasoning ? displayReasoning + '\n' : '') + embedded;
            displayContent = (displayContent.slice(0, thinkStart) + displayContent.slice(thinkEnd + 8)).trimStart();
          } else {
            const thinkStart = displayContent.indexOf('<think>');
            const streamingThink = displayContent.slice(thinkStart + 7);
            displayReasoning = (displayReasoning ? displayReasoning + '\n' : '') + streamingThink;
            displayContent = displayContent.slice(0, thinkStart);
          }
        }

        setMessages((prev) =>
          prev.map((m) => {
            if (m.id !== assistantMsgId) return m;
            return {
              ...m,
              content: displayContent,
              reasoningContent: displayReasoning,
            };
          })
        );
      },
      (err) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: m.content + `\n\n[Error: ${err.message}]`, isStreaming: false }
              : m
          )
        );
        setIsGenerating(false);
      },
      () => {
        const durationSec = Math.max(0.1, (Date.now() - generationStartTimeRef.current) / 1000);
        const estTokens = Math.round(charReceived / 2.2);
        const tokPerSec = (estTokens / durationSec).toFixed(1);

        setMessageTelemetry((prev) => ({
          ...prev,
          [assistantMsgId]: {
            durationSec: parseFloat(durationSec.toFixed(1)),
            tokensPerSec: currentEndpoint.tokensPerSec ? String(currentEndpoint.tokensPerSec) : tokPerSec,
            charCount: charReceived,
          },
        }));

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, isStreaming: false } : m))
        );
        setIsGenerating(false);
      }
    );
  };

  const handleSend = () => {
    handleSendPrompt(input);
  };

  const handleStop = () => {
    chatServiceRef.current.stop();
    setIsGenerating(false);
  };

  const clearChat = () => {
    setMessages([
      {
        id: 'init-1',
        role: 'assistant',
        content: 'Sesión reiniciada. Podés escribir o elegir una sugerencia.',
        timestamp: Date.now(),
      },
    ]);
  };

  const copySnippet = async (code: string, id: string) => {
    const success = await copyToClipboard(code);
    if (success) {
      setCopiadoCode(id);
      setTimeout(() => setCopiadoCode(null), 2000);
    }
  };

  const copyBundleConfig = async (format: 'pair' | 'env' | 'curl', key: string) => {
    if (!currentEndpoint) return;
    const success = await copyEndpointBundle(currentEndpoint, format);
    if (success) {
      setCopiadoKey(key);
      setTimeout(() => setCopiadoKey(null), 2000);
    }
  };

  // Lightweight markdown rendering
  const renderFormattedContent = (content: string, msgId: string) => {
    let mainContent = content;
    let embeddedThink = '';
    const match = content.match(/<think>([\s\S]*?)<\/think>/);
    if (match) {
      embeddedThink = match[1];
      mainContent = content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
    }

    const parts = mainContent.split(/(```[\s\S]*?```)/g);

    return (
      <>
        {embeddedThink && (
          <div className="think-accordion">
            <div className="think-accordion-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Brain size={13} />
                <span>Razonamiento</span>
              </div>
            </div>
            <div className="think-accordion-content">{embeddedThink}</div>
          </div>
        )}

        {parts.map((part, idx) => {
          if (part.startsWith('```') && part.endsWith('```')) {
            const raw = part.slice(3, -3);
            const firstLineBreak = raw.indexOf('\n');
            const lang = firstLineBreak !== -1 ? raw.slice(0, firstLineBreak).trim() : '';
            const code = firstLineBreak !== -1 ? raw.slice(firstLineBreak + 1) : raw;
            const blockId = `${msgId}-code-${idx}`;
            const lineCount = code.split('\n').length;
            const isLong = lineCount > 22;
            const isExpanded = Boolean(expandedCodeBlocks[blockId]);

            return (
              <div key={idx} className="code-block-card">
                <div className="code-block-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Code2 size={12} color="var(--accent-cyan)" />
                    <span>{lang || 'text'}</span>
                    <span style={{ opacity: 0.6 }}>· {lineCount} líneas</span>
                  </div>
                  <button
                    className="icon-btn"
                    style={{ padding: '2px 6px', fontSize: '11px', gap: '4px' }}
                    onClick={() => copySnippet(code, blockId)}
                  >
                    {copiedCode === blockId ? (
                      <>
                        <Check size={12} color="var(--accent-emerald)" />
                        <span style={{ color: 'var(--accent-emerald)' }}>Copiado</span>
                      </>
                    ) : (
                      <>
                        <Copy size={12} />
                        <span>Copiar</span>
                      </>
                    )}
                  </button>
                </div>

                <div
                  style={{
                    position: 'relative',
                    maxHeight: isLong && !isExpanded ? '280px' : 'none',
                    overflow: 'hidden',
                  }}
                >
                  <pre className="code-block-pre">
                    <code>{code}</code>
                  </pre>
                  {isLong && !isExpanded && (
                    <div
                      style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: '60px',
                        background: 'linear-gradient(to top, #080c16 30%, transparent)',
                        display: 'flex',
                        alignItems: 'flex-end',
                        justifyContent: 'center',
                        paddingBottom: '6px',
                      }}
                    >
                      <button
                        className="btn btn-secondary btn-sm"
                        style={{ fontSize: '11px', padding: '3px 10px', background: 'rgba(15, 23, 42, 0.9)' }}
                        onClick={() => setExpandedCodeBlocks((prev) => ({ ...prev, [blockId]: true }))}
                      >
                        <ChevronDown size={12} />
                        <span>Expandir {lineCount - 15} líneas</span>
                      </button>
                    </div>
                  )}
                </div>

                {isLong && isExpanded && (
                  <div style={{ display: 'flex', justifyContent: 'center', padding: '4px', background: '#0c1322' }}>
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ fontSize: '10px', padding: '2px 8px' }}
                      onClick={() => setExpandedCodeBlocks((prev) => ({ ...prev, [blockId]: false }))}
                    >
                      <ChevronUp size={11} />
                      <span>Contraer código</span>
                    </button>
                  </div>
                )}
              </div>
            );
          }

          return (
            <div key={idx} style={{ whiteSpace: 'pre-wrap', lineHeight: '1.65' }}>
              {part.split('\n').map((line, lineIdx) => {
                const lineElements: React.ReactNode[] = [];
                const regex = /(`[^`]+`|\*\*[^*]+\*\*)/g;
                let lastIndex = 0;
                let m;

                while ((m = regex.exec(line)) !== null) {
                  if (m.index > lastIndex) {
                    lineElements.push(line.substring(lastIndex, m.index));
                  }
                  const token = m[0];
                  if (token.startsWith('`') && token.endsWith('`')) {
                    lineElements.push(
                      <code key={m.index} className="inline-code">
                        {token.slice(1, -1)}
                      </code>
                    );
                  } else if (token.startsWith('**') && token.endsWith('**')) {
                    lineElements.push(
                      <strong key={m.index} style={{ color: '#fff', fontWeight: 600 }}>
                        {token.slice(2, -2)}
                      </strong>
                    );
                  }
                  lastIndex = regex.lastIndex;
                }
                if (lastIndex < line.length) {
                  lineElements.push(line.substring(lastIndex));
                }

                if (line.startsWith('### ')) {
                  return (
                    <div key={lineIdx} style={{ fontSize: '14px', fontWeight: 700, margin: '8px 0 4px', color: 'var(--accent-cyan)' }}>
                      {lineElements.slice(1)}
                    </div>
                  );
                }
                if (line.startsWith('## ')) {
                  return (
                    <div key={lineIdx} style={{ fontSize: '15px', fontWeight: 700, margin: '10px 0 6px', color: '#fff' }}>
                      {lineElements.slice(1)}
                    </div>
                  );
                }

                return (
                  <React.Fragment key={lineIdx}>
                    {lineElements}
                    {lineIdx < part.split('\n').length - 1 && '\n'}
                  </React.Fragment>
                );
              })}
            </div>
          );
        })}
      </>
    );
  };

  // Quick Prompt presets
  const quickPrompts = [
    { label: '⚡ Benchmark TPU', prompt: 'Please provide a Python script to asynchronously benchmark the throughput (tokens/s) of this TPU endpoint.' },
    { label: '💻 Cliente vLLM', prompt: 'Please provide a minimal working example using the standard OpenAI Python SDK for streaming completions.' },
    { label: '🧠 Arquitectura', prompt: 'Please explain how TPU v5e-8 Matrix Multiply Units (MXU) and vLLM scheduling optimize memory under long context windows.' },
  ];

  return (
    <div className="chat-container">
      {/* Top Status Capsule Bar */}
      <div className="chat-topbar">
        <div
          className="chat-endpoint-pill"
          title="View & configure TPU endpoint details"
          onClick={() => setShowEndpointDrawer(true)}
        >
          <span
            className={`status-dot ${isEndpointReady ? 'ready' : currentEndpoint ? 'running' : 'offline'}`}
          />
          <Cpu size={14} color={isEndpointReady ? 'var(--accent-emerald)' : 'var(--accent-cyan)'} />
          <span style={{ fontWeight: 600, color: '#fff' }}>
            {currentEndpoint?.model || 'Sin TPU conectada'}
          </span>
          {currentEndpoint?.tokensPerSec && (
            <span style={{ color: 'var(--accent-emerald)', fontSize: '10.5px' }}>
              · {currentEndpoint.tokensPerSec} tok/s
            </span>
          )}
          <ChevronDown size={12} style={{ opacity: 0.6 }} />
        </div>

        <div className="chat-topbar-actions">
          <button
            className="icon-btn"
            title="Borrar conversación"
            onClick={clearChat}
          >
            <Trash2 size={16} />
          </button>
          <button
            className="icon-btn"
            title="Configurar endpoint"
            onClick={() => setShowEndpointDrawer(true)}
          >
            <SlidersHorizontal size={16} />
          </button>
        </div>
      </div>

      {/* Offline floating warning banner */}
      {!isEndpointReady && (
        <div
          style={{
            margin: '8px 12px 0',
            padding: '8px 12px',
            background: 'rgba(245, 158, 11, 0.14)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: 'var(--radius-sm)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '11.5px',
          }}
        >
          <span style={{ color: 'var(--accent-amber)' }}>⚠️ No hay una TPU v5e-8 lista conectada</span>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button className="btn btn-secondary btn-sm" style={{ padding: '3px 8px', fontSize: '11px' }} onClick={() => setShowEndpointDrawer(true)}>
              Configurar endpoint
            </button>
            <button className="btn btn-primary btn-sm" style={{ padding: '3px 8px', fontSize: '11px' }} onClick={onGoToDashboard}>
              Iniciar instancia
            </button>
          </div>
        </div>
      )}

      {/* Messages list */}
      <div className="messages-list" ref={messagesListRef} onScroll={handleScroll}>
        {messages.map((m, mIdx) => {
          const isAssistant = m.role === 'assistant';
          const telemetry = messageTelemetry[m.id];
          const isLastAssistant = isAssistant && mIdx === messages.length - 1;

          return (
            <div key={m.id} className={`msg-bubble ${m.role === 'user' ? 'msg-user' : 'msg-assistant'}`}>
              {/* Thinking accordion */}
              {isAssistant && m.reasoningContent && (
                <div className="think-accordion">
                  <div
                    className="think-accordion-header"
                    onClick={() => setShowThinking((prev) => ({ ...prev, [m.id]: !prev[m.id] }))}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Brain size={13} className={m.isStreaming && !m.content ? 'spin-icon' : ''} />
                      <span>
                        {m.isStreaming && !m.content
                          ? `Pensando (${thinkingTimer}s)...`
                          : 'Razonamiento (finalizado)'}
                      </span>
                    </div>
                    {showThinking[m.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </div>

                  {showThinking[m.id] && (
                    <div className="think-accordion-content">{m.reasoningContent}</div>
                  )}
                </div>
              )}

              {/* Main content */}
              {renderFormattedContent(m.content, m.id)}

              {/* Streaming cursor */}
              {m.isStreaming && (
                <span
                  style={{
                    display: 'inline-block',
                    width: '6px',
                    height: '14px',
                    marginLeft: '4px',
                    verticalAlign: '-2px',
                    background: 'var(--accent-cyan)',
                    boxShadow: '0 0 8px var(--accent-cyan)',
                    animation: 'pulse 0.8s infinite',
                  }}
                />
              )}

              {/* Telemetry and actions */}
              {isAssistant && !m.isStreaming && m.content && (
                <div className="msg-footer">
                  <div className="msg-footer-telemetry">
                    {telemetry && (
                      <>
                        <span>⏱️ {telemetry.durationSec}s</span>
                        <span>⚡ {telemetry.tokensPerSec} tok/s</span>
                      </>
                    )}
                  </div>

                  <div className="msg-footer-actions">
                    <button
                      className="icon-btn"
                      style={{ padding: '2px 5px', fontSize: '10.5px', gap: '3px' }}
                      title="Copiar respuesta completa"
                      onClick={() => copySnippet(m.content, m.id)}
                    >
                      {copiedCode === m.id ? (
                        <Check size={11} color="var(--accent-emerald)" />
                      ) : (
                        <Copy size={11} />
                      )}
                      <span>{copiedCode === m.id ? 'Copiado' : 'Copiar'}</span>
                    </button>

                    {isLastAssistant && !isGenerating && (
                      <button
                        className="icon-btn"
                        style={{ padding: '2px 5px', fontSize: '10.5px', gap: '3px' }}
                        title="Regenerar respuesta"
                        onClick={() => {
                          const lastUser = [...messages].reverse().find((x) => x.role === 'user');
                          if (lastUser) handleSendPrompt(lastUser.content);
                        }}
                      >
                        <RefreshCw size={11} />
                        <span>Reintentar</span>
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Floating scroll to bottom button */}
      {!isAtBottom && (
        <button className="scroll-bottom-btn" onClick={() => scrollToBottom(true)}>
          <ArrowDown size={13} />
          <span>Ir al final {hasNewUnread && '· contenido nuevo'}</span>
        </button>
      )}

      {/* Quick Prompts */}
      {messages.length <= 1 && !isGenerating && (
        <div className="quick-chips-container">
          {quickPrompts.map((item, idx) => (
            <div
              key={idx}
              className="quick-chip"
              onClick={() => handleSendPrompt(item.prompt)}
            >
              <Sparkles size={11} color="var(--accent-cyan)" />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Input bar */}
      <div className="chat-input-bar">
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          rows={1}
          placeholder={
            isEndpointReady
              ? 'Escribí tu mensaje... (Shift+Enter para nueva línea)'
              : 'Endpoint TPU sin conexión. Tocá arriba para configurar.'
          }
          value={input}
          disabled={!isEndpointReady}
          onChange={(e) => {
            setInput(e.target.value);
            adjustTextareaHeight();
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />

        {isGenerating ? (
          <button
            className="btn btn-danger"
            style={{ padding: '9px 13px', borderRadius: 'var(--radius-md)' }}
            title="Detener generación"
            onClick={handleStop}
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            className="btn btn-primary"
            style={{ padding: '9px 13px', borderRadius: 'var(--radius-md)' }}
            disabled={!input.trim() || !isEndpointReady}
            title="Enviar mensaje"
            onClick={handleSend}
          >
            <Send size={16} />
          </button>
        )}
      </div>

      {/* Bottom Sheet Drawer */}
      {showEndpointDrawer && (
        <div className="drawer-backdrop" onClick={() => setShowEndpointDrawer(false)}>
          <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-handle" />

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 700 }}>
                <Cpu size={16} color="var(--accent-cyan)" />
                <span>Detalles y ajustes del endpoint TPU</span>
              </div>
              <button className="icon-btn" onClick={() => setShowEndpointDrawer(false)}>
                <X size={16} />
              </button>
            </div>

            {/* Instance selector (if multiple endpoints exist) */}
            {endpoints.length > 1 && (
              <div className="form-group">
                <label className="form-label">Cambiar TPU activa:</label>
                <select
                  className="form-select"
                  value={currentEndpoint?.baseUrl || ''}
                  onChange={(e) => {
                    const target = endpoints.find((ep) => ep.baseUrl === e.target.value);
                    if (target && onSelectEndpoint) onSelectEndpoint(target);
                  }}
                >
                  {endpoints.map((ep, i) => (
                    <option key={i} value={ep.baseUrl}>
                      {ep.accountName || `Endpoint ${i + 1}`} ({ep.model}) {ep.tokensPerSec ? `· ${ep.tokensPerSec} tok/s` : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Quick bundle copy */}
            {currentEndpoint && (
              <div
                style={{
                  background: 'rgba(30, 41, 59, 0.6)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '10px 12px',
                  marginBottom: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Modelo servido:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                    {currentEndpoint.model}
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                  <button
                    className="btn btn-primary btn-sm"
                    style={{ padding: '6px 8px', fontSize: '11px', gap: '4px' }}
                    onClick={() => copyBundleConfig('pair', 'pair')}
                  >
                    {copiedKey === 'pair' ? <Check size={12} color="#fff" /> : <Copy size={12} />}
                    <span>{copiedKey === 'pair' ? 'Copiado' : 'Copiar URL+clave'}</span>
                  </button>

                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '6px 8px', fontSize: '11px', gap: '4px' }}
                    onClick={() => copyBundleConfig('env', 'env')}
                  >
                    {copiedKey === 'env' ? <Check size={12} color="var(--accent-emerald)" /> : <Code2 size={12} />}
                    <span>{copiedKey === 'env' ? 'Copiado' : 'Copiar formato .env'}</span>
                  </button>
                </div>
              </div>
            )}

            {/* Base URL */}
            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Globe size={12} color="var(--accent-blue)" />
                <span>Base URL</span>
              </label>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input
                  className="form-input"
                  readOnly
                  value={currentEndpoint?.baseUrl || 'Sin conexión'}
                  style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                />
                {currentEndpoint?.baseUrl && (
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '4px 8px' }}
                    onClick={() => copySnippet(currentEndpoint.baseUrl, 'url')}
                  >
                    {copiedCode === 'url' ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                  </button>
                )}
              </div>
            </div>

            {/* API Key */}
            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Key size={12} color="var(--accent-amber)" />
                <span>Clave API (token)</span>
              </label>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input
                  className="form-input"
                  readOnly
                  value={currentEndpoint?.apiKey || 'Sin asignar'}
                  style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                />
                {currentEndpoint?.apiKey && (
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '4px 8px' }}
                    onClick={() => copySnippet(currentEndpoint.apiKey, 'key')}
                  >
                    {copiedCode === 'key' ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                  </button>
                )}
              </div>
            </div>

            {/* Custom Reverse Proxy / Cloudflare Tunnel URL */}
            <div
              style={{
                marginTop: '12px',
                padding: '10px',
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: 'var(--radius-sm)',
                border: '1px dashed var(--border-color)',
              }}
            >
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                URL base personalizada (proxy / túnel propio):
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input
                  className="form-input"
                  style={{ padding: '4px 8px', fontSize: '11px', flex: 1 }}
                  placeholder="https://xxxx.trycloudflare.com/v1"
                  value={manualUrl}
                  onChange={(e) => setManualUrl(e.target.value)}
                />
                <button
                  className="btn btn-primary btn-sm"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => {
                    if (manualUrl.trim()) {
                      setCustomBaseUrl(manualUrl.trim());
                      setShowEndpointDrawer(false);
                    }
                  }}
                >
                  Aplicar
                </button>
                {customBaseUrl && (
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ padding: '4px 8px', fontSize: '11px' }}
                    onClick={() => {
                      setCustomBaseUrl('');
                      setManualUrl('');
                    }}
                  >
                    Restablecer
                  </button>
                )}
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '14px' }}>
              <button className="btn btn-secondary btn-sm" onClick={() => setShowEndpointDrawer(false)}>
                Listo
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
