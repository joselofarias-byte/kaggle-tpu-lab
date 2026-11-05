#!/usr/bin/env bash
set -euo pipefail
# JolufaSoft Coding Lab - ejecutar en Oracle Ubuntu 24.04 como usuario normal
ROOT="${HOME}/jolufasoft-coding-lab"
PORT="${CODE_SERVER_PORT:-18443}"
mkdir -p "$ROOT"/{workspace,config,data}
command -v docker >/dev/null || { echo "Falta Docker. Instalalo con: sudo apt-get update && sudo apt-get install -y docker.io"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker no disponible para este usuario. Ejecutar: sudo usermod -aG docker \"$USER\"; newgrp docker"; exit 1; }
if [[ ! -s "$ROOT/config/password" ]]; then
  umask 077
  openssl rand -base64 30 | tr -d '\n' > "$ROOT/config/password"
fi
PASSWORD="$(cat "$ROOT/config/password")"
docker rm -f jolufasoft-code-server >/dev/null 2>&1 || true
docker run -d --name jolufasoft-code-server --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -p "127.0.0.1:${PORT}:8080" \
  -e "PASSWORD=$PASSWORD" \
  -e "DEFAULT_WORKSPACE=/home/coder/project" \
  -v "$ROOT/workspace:/home/coder/project" \
  -v "$ROOT/config:/home/coder/.config" \
  -v "$ROOT/data:/home/coder/.local/share" \
  codercom/code-server:latest
echo "IDE activo en 127.0.0.1:$PORT (solo loopback)"
echo "Clave: $ROOT/config/password"
echo "Desde Termux: ssh -N -L 18443:127.0.0.1:$PORT oracle-lab"
echo "Abrir: http://127.0.0.1:18443"
echo "NOTA: El contenedor no tiene por defecto OpenCode ni toolchains Android."
