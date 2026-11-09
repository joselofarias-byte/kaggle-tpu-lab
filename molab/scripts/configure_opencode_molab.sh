#!/usr/bin/env bash
set -euo pipefail
# Compatibility wrapper: endpoint must be authorized by its hosting platform.
: "${MOLAB_URL:?Set an authorized endpoint (never a Molab tunnel)}"
: "${MODEL_ID:?Set the actual served model ID}"
: "${MOLAB_CONTEXT:?Set the server's configured context length}"
exec python3 "$(dirname "$0")/configure_opencode.py" \
  --url "$MOLAB_URL" --model "$MODEL_ID" --context "$MOLAB_CONTEXT" \
  --output "${MOLAB_OUTPUT:-2048}" "$@"
