export type ModelId = 'qwen38-27b' | 'glm53-flash';

export type ReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh';

/**
 * Normalizes served model name to resolve 404s when vLLM served-model-name includes decimals while recipe name does not.
 */
export function getServedModelName(recipeOrModel?: string): string {
  if (!recipeOrModel) return 'qwen3.8-27b';
  const m = recipeOrModel.toLowerCase();
  if (m.includes('qwen3') || m.includes('qwen38') || m.includes('qwen3.8')) return 'qwen3.8-27b';
  if (m.includes('glm') || m.includes('glm53') || m.includes('glm-5.3')) return 'glm-5.3-flash';
  return recipeOrModel;
}

export interface KaggleAccount {
  id: string;
  name: string;
  token: string;
  username: string;
  isValid: boolean;
  isChecking: boolean;
  error?: string;
  selectedForRace?: boolean;
}

export interface LaunchConfig {
  model: ModelId;
  reasoningEffort: ReasoningEffort;
  maxModelLen: number;
  maxNumSeqs: number;
  mtpTokens: number;
  streams: number;
  keepaliveMin: number;
  textOnly: boolean;
  fastStart: boolean;
}

export interface NtfyEvent {
  event_version?: number;
  phase: string;
  state?: 'starting' | 'ready' | 'error' | 'stopped' | string;
  message_es?: string;
  error_code?: string;
  hint_es?: string;
  recoverable?: boolean;
  topic?: string;
  endpoint?: string;
  api_key?: string;
  model?: string;
  /** Optional identity. Older clients ignore these fields. */
  model_id?: string;
  display_name?: string;
  backend?: string;
  accelerator?: string;
  minutes?: number | string;
  elapsed_s?: number;
  startup_secs?: number;
  decode_tok_s?: number | string;
  sanity?: string;
  cause?: string;
  hint?: string;
  tail?: string;
  step?: string;
  what?: string;
  secs?: number;
  up_min?: number | string;
  rawTime?: number;
}

export interface RaceSession {
  account: string; // accountId
  accountName?: string;
  username: string;
  token: string;
  kernel: string;
  slug: string;
  topic: string;
  apiKey: string;
  endpointUrl?: string;
  model?: ModelId;
  status: 'IDLE' | 'QUEUED' | 'RUNNING' | 'WINNER' | 'CANCELLED' | 'COMPLETE' | 'ERROR';
  seenBoot: boolean;
  done: boolean;
  raceGroupId?: string;
  endpoint?: LiveEndpoint;
  lastEvent?: NtfyEvent;
  events: { time: string; text: string; phase: string }[];
}

export interface LiveEndpoint {
  accountId?: string;
  accountName?: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  /** Present when the kernel event names the running profile. UI may ignore them. */
  modelId?: string;
  displayName?: string;
  backend?: string;
  accelerator?: string;
  status: 'CONNECTING' | 'READY' | 'ERROR' | 'OFFLINE';
  uptimeMinutes?: number;
  tokensPerSec?: string;
  keepaliveMinutes?: number;
  expireTime?: number;
  /** Epoch ms when this URL was first mounted. Probes wait before marking it offline. */
  readySince?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoningContent?: string;
  timestamp: number;
  isStreaming?: boolean;
}
