# Modal desde Termux y GitHub Actions

## Operación única desde Termux

Copiar y ejecutar `termux-deploy.sh` con Bash en el almacenamiento privado de
Termux, no en `/sdcard`. Instala `gh` y OpenSSL si faltan, abre la autenticación
web de GitHub si hace falta y solicita las dos credenciales Modal **sin eco**.
La cuenta GitHub debe poder administrar Secrets de Actions y despachar workflows.
No necesita Debian, Python, SSH ni el SDK de Modal en Android.

El script configura `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` y
`CONTROLLER_BUNDLE_PASSWORD` mediante stdin de `gh secret set`. Genera esta última
con OpenSSL una sola vez y la conserva en
`~/.config/kaggle-modal-actions/bundle-password` (archivo 0600, directorio 0700).
Reutiliza ese archivo al repetirlo: respaldarlo de forma privada. No ejecutarlo
con trazado de shell ni compartir los archivos privados. Si se perdió la contraseña,
los bundles anteriores no se pueden descifrar con una contraseña nueva.

Antes de lanzar despliegue exige escribir `DESPLEGAR` para aceptar posibles cargos
no-GPU. Nunca solicita inferencia. Revisa el enlace de Actions y su resultado;
una solicitud de ejecución no prueba que el despliegue haya terminado.

## Persistencia del token

`android-modal-control` es el Secret de Modal que contiene `CONTROLLER_TOKEN`.
Si existe se recupera mediante una función Modal efímera **sólo CPU**, inyectando
ese Secret y retornando su valor por el canal autenticado del SDK al proceso.
No se imprime ni se actualiza el Secret; un token explícito diferente causa un
error. Si la lectura falla se aborta: no se genera un reemplazo.
Si no existe, se genera criptográficamente y se crea sin sobrescribir.
No borrar ese Secret ni cambiar el workspace entre despliegues: perderlo puede
impedir conservar la conexión de la APK. No redesplegar con una sesión activa.

No hace falta copiar `CONTROLLER_TOKEN` a GitHub ni descifrar un artefacto para
reintroducirlo como Secret. Tanto despliegue como inferencia recuperan el Secret
existente cuando falta la variable. Un Secret GitHub antiguo y distinto debe
eliminarse/corregirse explícitamente; nunca se rota automáticamente en Modal.
La recuperación usa CPU/RAM y puede generar cargos, pero no crea GPU.

El bundle cifrado sigue disponible durante un día para **configurar una APK por
primera vez** o recuperar la conexión. Una APK ya configurada no necesita volver
a introducir el token. La app no incorpora credenciales de Modal ni el token en
la compilación: URL y token se guardan mediante sus campos y almacenamiento seguro.
Para una primera configuración puede descifrarse localmente con
`python operations.py decrypt connection.enc` y la contraseña guardada; el JSON
se escribe fuera del repo con permisos privados. No subirlo ni publicarlo.

## Workflows desde la rama del PR

GitHub exige registrar workflows manuales en la rama predeterminada. Los dos
archivos nuevos no están registrados allí todavía. Por eso hay dos entradas
**existentes y ya registradas**, cuyas definiciones se ejecutan desde `--ref`:

- `.github/workflows/mobile-android-es.yml`: con `modal_deploy=true` llama al
  reutilizable `.github/workflows/modal-deploy.yml` de la **misma revisión**.
  No compila Android en ese modo.
- `.github/workflows/python-syntax.yml`: despacho manual independiente de prueba
  llama a `.github/workflows/modal-inference.yml` de la misma revisión.
  Los eventos habituales siguen comprobando sintaxis.

Todos estos archivos están en `.github/workflows/`. Para comprobar ambas rutas
sin credenciales, recursos Modal ni inferencia:

```sh
REPO=joselofarias-byte/kaggle-tpu-lab
BRANCH=hoplite/kalymna-ddcfd6fb
gh workflow run mobile-android-es.yml -R "$REPO" --ref "$BRANCH" \
  -f modal_deploy=true -f validate_only=true
gh workflow run python-syntax.yml -R "$REPO" --ref "$BRANCH" \
  -f validate_only=true
```

Después de integrar en main también podrán despacharse los nuevos workflows
directamente, siempre seleccionando la rama/revisión correcta.

## Inferencia manual independiente: NO ejecutada

Sólo cuando se autorice explícitamente GPU y los demás cargos:

```sh
gh workflow run python-syntax.yml -R joselofarias-byte/kaggle-tpu-lab \
  --ref hoplite/kalymna-ddcfd6fb \
  -f validate_only=false -f accept_cost=true \
  -f controller_url=https://TU-CONTROLADOR.modal.run \
  -f profile=qwen-small -f gpu_hourly_usd=TARIFA_VERIFICADA -f max_minutes=10
```

Usa los Secrets Modal anteriores. Perfil `qwen-small`: T4/2 CPU/8 GiB;
`qwen-7b`: L40S/4 CPU/16 GiB. La tarifa introducida es sólo GPU USD/h de
Sandbox verificada en la cuenta, no presupuesto total. El límite absoluto de
5–20 minutos incluye descarga y arranque. Hay parada en `finally` y reintento
`always()`, además del timeout del proveedor. Una parada no confirmada no es éxito:
revisar el panel Modal. No se promete coste cero ni que toda cancelación ejecute
la limpieza. [Precios oficiales](https://modal.com/pricing).

## Comprobaciones y límites

```sh
cd modal-control
python -m pip install -r requirements-test.txt
python -m pytest -q
python -m compileall -q .
bash -n termux-deploy.sh
```

Las **106 pruebas anteriores** eran unitarias sin nube: autenticación y contratos
HTTP, perfiles/límites, estados y concurrencia, recuperación, errores y parada,
watchdog/timeout, redacción, adaptación al SDK, cifrado y archivos privados,
validación del despliegue y diario/limpieza de inferencia. SDK, red y sandboxes
están simulados: no prueban despliegue, credenciales reales ni facturación.
Las pruebas adicionales cubren persistencia del token y cableado de workflows.

Continúan pendientes Termux real, lectura remota del Secret, creación de imágenes,
despliegue/HTTPS real, conexión de la APK, disponibilidad y carga GPU, apagado real
por inactividad/timeout/cancelación y consumo facturado. Ninguno se deduce de las
pruebas locales ni de despachar los recorridos de validación sin recursos.
