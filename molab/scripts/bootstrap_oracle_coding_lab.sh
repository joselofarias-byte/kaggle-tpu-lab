#!/usr/bin/env bash
set -euo pipefail
# JolufaSoft Coding Lab: execute locally on Oracle Lab (Ubuntu 24.04)
# Downloads private GitHub files using authenticated gh; no external machine access required.
REPO="joselofarias-byte/kaggle-tpu-lab"
REF="feature/molab-llm-lab"
DIR="$HOME/jolufasoft-coding-lab"
mkdir -p "$DIR/scripts" "$DIR/workspace"
umask 077
fetch() {
  local file="$1"
  gh api -H 'Accept: application/vnd.github.raw+json' "repos/$REPO/contents/molab/scripts/$file?ref=$REF" > "$DIR/scripts/$file"
  test -s "$DIR/scripts/$file"
}
command -v gh >/dev/null || { echo "gh CLI required"; exit 1; }
gh auth status >/dev/null
fetch install_coding_ide_oracle.sh
fetch configure_opencode_molab.sh
fetch configure_opencode.py
fetch smoke_test_coding_api.py
fetch launch_vllm_coding.sh
bash "$DIR/scripts/install_coding_ide_oracle.sh"
echo
echo "=== CODING LAB FILES ==="
ls -lh "$DIR/scripts"
echo "=== CONTAINER ==="
docker ps --filter name=jolufasoft-code-server --format '{{.Names}} {{.Status}} {{.Ports}}'
echo
echo "Use an authorized existing VPS access path; no Termux keepalive is required by inference scripts."
echo "Browser: http://127.0.0.1:18443"
echo "Password on Oracle: $DIR/config/password"
echo "Molab forbids exposing compute externally. Use remote clients only with an authorized provider."
