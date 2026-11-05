#!/usr/bin/env bash
set -euo pipefail
# Configure OpenCode on Oracle A1 to use a securely exposed Molab inference endpoint.
# Usage: MOLAB_URL=https://your-private-https-endpoint MOLAB_API_KEY=... MODEL_ID=... bash configure_opencode_molab.sh
: "${MOLAB_URL:?Set MOLAB_URL to a trusted HTTPS endpoint with /v1 available}"
: "${MOLAB_API_KEY:?Set MOLAB_API_KEY}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
case "$MOLAB_URL" in https://*) ;; *) echo "HTTPS required" >&2; exit 1;; esac
command -v opencode >/dev/null || { echo "OpenCode not installed" >&2; exit 1; }
command -v python3 >/dev/null || { echo "Python3 required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
BASE="${MOLAB_URL%/}"
BASE="${BASE%/v1}"
curl -fsS --connect-timeout 10 --max-time 30 -H "Authorization: Bearer $MOLAB_API_KEY" "$BASE/health" >/dev/null
export BASE MODEL_ID MOLAB_API_KEY
python3 - <<'PY'
import json, os, pathlib
p=pathlib.Path.home()/".config/opencode/opencode.json"
p.parent.mkdir(parents=True, exist_ok=True)
data=json.loads(p.read_text()) if p.exists() else {}
providers=data.setdefault("provider",{})
providers["jolufasoft-molab"]={
  "npm":"@ai-sdk/openai-compatible",
  "name":"JolufaSoft Molab GPU",
  "options":{"baseURL":os.environ["BASE"]+"/v1", "apiKey":"{env:MOLAB_API_KEY}"},
  "models":{os.environ["MODEL_ID"]:{
    "name":"Molab GPU (inference only, no tool calls)",
    "limit":{"context":32768,"output":4096}
  }}
}
if p.exists():
    backup=p.with_suffix(".json.bak")
    backup.write_text(p.read_text())
p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
os.chmod(p,0o600)
print("Configured:",p)
print("Model:",os.environ["MODEL_ID"])
print("WARNING: This inference endpoint has no tool calling or streaming; it is not yet suitable as an autonomous OpenCode agent.")
PY
