#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
REPO=joselofarias-byte/kaggle-tpu-lab
BRANCH=hoplite/kalymna-ddcfd6fb
# Native Termux needs only GitHub CLI and OpenSSL, not the Modal Python SDK.
if ! command -v gh >/dev/null || ! command -v openssl >/dev/null; then
  command -v pkg >/dev/null || { echo 'Instalá gh y openssl antes de continuar.' >&2; exit 1; }
  pkg install -y gh openssl
fi
gh auth status >/dev/null 2>&1 || gh auth login --web --git-protocol https
STATE="$HOME/.config/kaggle-modal-actions"
[[ ! -L "$HOME/.config" && ! -L "$STATE" ]] || { echo 'Ruta privada no válida.' >&2; exit 1; }
mkdir -p "$STATE"
chmod 700 "$STATE"
PASSWORD="$STATE/bundle-password"
[[ ! -L "$PASSWORD" ]] || exit 1
if [[ ! -e "$PASSWORD" ]]; then
  ( set -o noclobber; openssl rand -base64 48 > "$PASSWORD" )
fi
[[ -f "$PASSWORD" && -O "$PASSWORD" ]] || exit 1
chmod 600 "$PASSWORD"
[[ $(wc -c < "$PASSWORD") -ge 16 ]] || { echo 'Contraseña local inválida; no se reemplazó.' >&2; exit 1; }
# Secret values travel over stdin, never as command-line arguments.
trap 'unset VALUE' EXIT
for KEY in MODAL_TOKEN_ID MODAL_TOKEN_SECRET; do
  read -r -s -p "$KEY (workspace Modal correcto): " VALUE
  printf '\n'
  [[ -n "$VALUE" ]] || { echo 'La credencial no puede estar vacía.' >&2; exit 1; }
  printf '%s' "$VALUE" | gh secret set "$KEY" --repo "$REPO"
  unset VALUE
done
gh secret set CONTROLLER_BUNDLE_PASSWORD --repo "$REPO" < "$PASSWORD"
printf 'Contraseña del bundle guardada en %s (0600).\n' "$PASSWORD"
printf 'No se modifica CONTROLLER_TOKEN: se reutiliza el Secret de Modal existente.\n'
printf 'Desplegar puede generar cargos de build y CPU/RAM; no inicia GPU.\n'
read -r -p 'Escribí DESPLEGAR para autorizar esos cargos: ' ACCEPT
[[ "$ACCEPT" == DESPLEGAR ]] || { echo 'Cancelado; no se lanzó ningún workflow.'; exit 0; }
gh workflow run mobile-android-es.yml --repo "$REPO" --ref "$BRANCH" \
  -f modal_deploy=true -f validate_only=false -f acknowledge_cost=true
printf 'Solicitado despliegue desde %s. No se solicitó inferencia.\n' "$BRANCH"
gh run list --repo "$REPO" --workflow mobile-android-es.yml --branch "$BRANCH" --limit 3
