#!/usr/bin/env bash
set -euo pipefail
# Safe smoke-test of Coding Lab. No remote actions.
ROOT="$HOME/jolufasoft-coding-lab"
echo "=== IDE ==="
if docker ps --format '{{.Names}}' | grep -qx 'jolufasoft-code-server'; then
  curl -fsSI --max-time 10 http://127.0.0.1:18443/ | head -n 1
else
  echo "FAIL: IDE container not running"; exit 1
fi
echo "=== WORKSPACE ==="
test -d "$ROOT/workspace"
test -s "$ROOT/config/password"
echo "PASS workspace and credentials"
echo "=== GPU API ==="
if [[ -n "${MOLAB_URL:-}" && -n "${MOLAB_API_KEY:-}" && -n "${MODEL_ID:-}" ]]; then
  python3 "$ROOT/scripts/smoke_test_coding_api.py"
else
  echo "SKIP: MOLAB_URL, MOLAB_API_KEY, MODEL_ID not all provided"
fi
