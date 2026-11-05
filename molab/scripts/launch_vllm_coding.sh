#!/usr/bin/env bash
set -euo pipefail
# JolufaSoft Molab: OpenAI API with streaming and tool calling via vLLM.
# Execute in Molab GPU terminal. Python >=3.10 and CUDA GPU required.
# Qwen3 models with Hermes tool-call format are supported by recent vLLM.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
PORT="${MOLAB_PORT:-18080}"
CONTEXT="${MOLAB_CONTEXT:-8192}"
GPU_UTIL="${MOLAB_GPU_UTIL:-0.85}"
: "${MOLAB_API_KEY:?Set a private MOLAB_API_KEY (>=24 characters)}"
if (( ${#MOLAB_API_KEY} < 24 )); then echo "API key too short" >&2; exit 1; fi
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU unavailable"
p=torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}; VRAM: {p.total_memory/2**30:.1f} GiB")
PY
if ! python3 -c 'import vllm' >/dev/null 2>&1; then
  python3 -m pip install 'vllm>=0.11,<0.14'
fi
echo "Starting vLLM on loopback only. Never expose without authentication and TLS."
exec python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL_ID" --served-model-name "$MODEL_ID" \
  --api-key "$MOLAB_API_KEY" \
  --dtype auto --max-model-len "$CONTEXT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code false
