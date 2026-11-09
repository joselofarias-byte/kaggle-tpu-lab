#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
REPO=joselofarias-byte/kaggle-tpu-lab
BRANCH=main
for DEP in gh openssl python; do
  if ! command -v "$DEP" >/dev/null; then
    command -v pkg >/dev/null || { echo "Instalá $DEP antes de continuar." >&2; exit 1; }
    pkg install -y "$DEP"
    command -v "$DEP" >/dev/null || exit 1
  fi
done
gh auth status >/dev/null 2>&1 || gh auth login --web --git-protocol https
STATE="$HOME/.config/kaggle-modal-actions"
[[ ! -L "$HOME/.config" && ! -L "$STATE" ]] || { echo 'Ruta privada no válida.' >&2; exit 1; }
mkdir -p "$STATE"
chmod 700 "$STATE"
PASSWORD="$STATE/bundle-password"
[[ ! -L "$PASSWORD" ]] || exit 1
secret_exists() {
  local NAMES
  NAMES=$(gh secret list --repo "$REPO" --json name --jq '.[].name') || return 2
  while IFS= read -r NAME; do
    [[ "$NAME" != "$1" ]] || return 0
  done <<< "$NAMES"
  return 1
}
# A failed listing must never be mistaken for an absent secret.
has_secret() {
  local STATUS=0
  secret_exists "$1" || STATUS=$?
  [[ "$STATUS" -le 1 ]] || { echo 'No se pudieron consultar los nombres de Secrets.' >&2; exit 2; }
  return "$STATUS"
}
if has_secret CONTROLLER_BUNDLE_PASSWORD; then
  if [[ ! -f "$PASSWORD" ]]; then
    echo "Existe CONTROLLER_BUNDLE_PASSWORD en GitHub, pero falta $PASSWORD." >&2
    echo 'Recuperá la contraseña ORIGINAL desde tu copia privada o gestor de contraseñas y guardala allí (0600).' >&2
    echo 'GitHub no permite leer su valor. Sin la original no se puede continuar; no se generó ni sobrescribió nada.' >&2
    exit 1
  fi
elif [[ ! -e "$PASSWORD" ]]; then
  ( set -o noclobber; openssl rand -base64 48 > "$PASSWORD" )
fi
[[ -f "$PASSWORD" && -O "$PASSWORD" ]] || exit 1
chmod 600 "$PASSWORD"
[[ $(wc -c < "$PASSWORD") -ge 16 ]] || { echo 'Contraseña local inválida; no se reemplazó.' >&2; exit 1; }
TMP=$(mktemp -d "$STATE/.preflight.XXXXXX")
trap 'unset VALUE; rm -rf -- "$TMP"' EXIT
for KEY in MODAL_TOKEN_ID MODAL_TOKEN_SECRET; do
  if has_secret "$KEY"; then
    printf '%s ya existe en GitHub; se conserva.\n' "$KEY"
  else
    read -r -s -p "$KEY (token del workspace Modal deseado): " VALUE
    printf '\n'
    [[ -n "$VALUE" ]] || { echo 'La credencial no puede estar vacía.' >&2; exit 1; }
    if ! has_secret "$KEY"; then
      printf '%s' "$VALUE" | gh secret set "$KEY" --repo "$REPO"
    fi
    unset VALUE
  fi
done
if ! has_secret CONTROLLER_BUNDLE_PASSWORD; then
  python -c 'import pathlib,sys; sys.stdout.write(pathlib.Path(sys.argv[1]).read_text().rstrip("\r\n"))' "$PASSWORD" | gh secret set CONTROLLER_BUNDLE_PASSWORD --repo "$REPO"
fi
printf 'Secrets existentes conservados; contraseña local: %s (0600).\n' "$PASSWORD"
REQUEST=$(openssl rand -hex 16)
# This workflow only queries metadata and proves password correspondence.
gh workflow run modal-workspace.yml --repo "$REPO" --ref "$BRANCH" -f request_id="$REQUEST"
RUN=''
for ((ATTEMPT=0; ATTEMPT<30; ATTEMPT++)); do
  RUN=$(gh run list --repo "$REPO" --workflow modal-workspace.yml --branch "$BRANCH" \
    --event workflow_dispatch --limit 100 --json databaseId,displayTitle \
    --jq ".[] | select(.displayTitle == \"modal-workspace-$REQUEST\") | .databaseId")
  [[ -z "$RUN" ]] || break
  sleep 2
done
[[ "$RUN" =~ ^[0-9]+$ ]] || { echo 'No se identificó una única validación; no se desplegó.' >&2; exit 1; }
gh run watch "$RUN" --repo "$REPO" --exit-status --interval 5
gh run download "$RUN" --repo "$REPO" --name modal-workspace --dir "$TMP"
python - "$PASSWORD" "$TMP/workspace.json" "$REQUEST" > "$TMP/destination" <<'PY'
import hashlib, hmac, json, pathlib, re, sys
password = pathlib.Path(sys.argv[1]).read_text()
record = json.loads(pathlib.Path(sys.argv[2]).read_text())
proofs = [hmac.new(value.encode(), sys.argv[3].encode(), hashlib.sha256).hexdigest()
          for value in (password, password.rstrip('\r\n'))]
if record.get('request_id') != sys.argv[3] or not any(hmac.compare_digest(proof, record.get('password_proof', '')) for proof in proofs):
    sys.exit('La contraseña local no corresponde al Secret GitHub: restaurá la ORIGINAL desde tu respaldo privado. No se sobrescribió ningún secreto existente.')
workspaces = record.get('workspaces')
if not isinstance(workspaces, list) or len(workspaces) != 1 or not isinstance(workspaces[0], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', workspaces[0]):
    sys.exit('Destino no válido; no se desplegó.')
print(workspaces[0])
print('existente; se conservará sin modificar' if record.get('controller_secret_exists') else 'ausente; el primer despliegue creará uno nuevo')
PY
{ IFS= read -r WORKSPACE; IFS= read -r SECRET_STATE; } < "$TMP/destination"
printf '\nWorkspaces disponibles para el token almacenado en GitHub:\n  1) %s\n' "$WORKSPACE"
printf 'Un token Modal sólo accede a su workspace, no a todos los de la cuenta.\n'
printf 'Secret android-modal-control: %s.\n' "$SECRET_STATE"
read -r -p 'Elegí 1 para validar este destino (otra entrada cancela): ' CHOICE
[[ "$CHOICE" == 1 ]] || { echo 'Cancelado sin despliegue.'; exit 0; }
printf 'Destino validado: %s. No se modifica CONTROLLER_TOKEN.\n' "$WORKSPACE"
printf 'Desplegar puede generar cargos de build y CPU/RAM; no inicia GPU.\n'
read -r -p 'Escribí DESPLEGAR para autorizar esos cargos: ' ACCEPT
[[ "$ACCEPT" == DESPLEGAR ]] || { echo 'Cancelado; no se lanzó despliegue.'; exit 0; }
gh workflow run modal-deploy.yml --repo "$REPO" --ref "$BRANCH" \
  -f validate_only=false -f acknowledge_cost=true -f expected_workspace="$WORKSPACE"
printf 'Solicitado despliegue desde %s al workspace %s. No se solicitó inferencia.\n' "$BRANCH" "$WORKSPACE"
gh run list --repo "$REPO" --workflow modal-deploy.yml --branch "$BRANCH" --limit 3
