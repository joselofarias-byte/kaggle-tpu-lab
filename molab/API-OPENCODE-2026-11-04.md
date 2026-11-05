# API autenticada de Molab y OpenCode — 4 de noviembre de 2026

PR #12 mergeado por autorización del usuario en main: dc32b4a8c0d40a942ed58e3b7f99f29f3998e053.

## Ejecutado en Molab

El sandbox conserva el GGUF Q4_K_M MTP y los tres binarios CUDA compilados. Se inició llama-server con el mismo modelo, alias `jolufasoft-davidau-mtp`, contexto 32768, una secuencia, todas las capas GPU y decodificación MTP de hasta tres tokens. El contexto se configuró y cargó; no se hizo una prueba de capacidad de 32768 tokens.

El servidor escucha exclusivamente en loopback. La clave se generó con secrets.token_hex(32), en archivo privado con modo 0600 bajo un directorio 0700. Se pasa por --api-key-file; no aparece en los argumentos ni se registra en este repositorio. Web UI deshabilitada; no se habilitaron herramientas integradas ni SSH.

Pruebas reales:
- GET /v1/models sin clave: **401**.
- GET /v1/models con clave incorrecta: **401**.
- GET /v1/models autenticado: enumera el alias.
- Chat autenticado: respuesta `API_MTP_OK`, tiempo HTTP **0.367 s**.
- Chat en streaming SSE: **6 chunks**, texto `API_MTP_OK` y marcador **[DONE]** recibidos.
- OpenCode **1.19.5**, instalado desde npm en un directorio aislado del experimento.
- OpenCode invocó **read** y leyó correctamente un archivo sintético, luego respondió **JOLUFASOFT_MTP_OPENCODE_OK**.
- Tiempo total de OpenCode: **4.800 s**; proceso exit **0**; una llamada real de herramienta aprobada.

El primer intento de OpenCode buscó una ruta relativa bajo /marimo y recibió File not found. La prueba reproducible se corrigió para proporcionar la ruta absoluta del fixture. Se conservan ambos logs; el primer intento no se presenta como exitoso. Sólo read estaba permitido; el resto de herramientas estaba denegado.

## Archivos

- davidau_mtp_service.py: start/status/test/stop, servidor autenticado y prueba SSE.
- davidau_opencode_smoke.py: prueba real de OpenCode contra el proveedor.
- opencode.molab.json: configuración sin secretos, mediante MOLAB_BASE_URL y MOLAB_API_KEY.
- davidau_mtp_tailnet.py: preparación de Tailscale en userspace con acciones separadas, sin instalar, iniciar sesión ni publicar al importarlo.
- evidence/molab-api-20261104/: logs y respuestas exportadas del notebook, con una allowlist explícita que excluye .private. Antes de exportar se comprobó que no contuvieran la clave generada.

El service.py descargado y ejecutado corresponde al commit ceee649c98fe653fc9d0811a09501edf77a51813; el smoke test ejecutado corresponde a 62e47d575d51aa18dbd0a305299a48231d7a7c66. El script de Tailscale sólo está preparado; no se ejecutó ni se afirma que la ruta externa funcione.

## Acceso desde HONOR / Muse / otro OpenCode

**Autorización recibida el 5/11/2026.** Se instaló Tailscale 1.104.5 desde el archivo oficial, verificando su SHA-256, y se inició tailscaled en modo userspace dentro de Molab. El daemon quedó en **NeedsLogin**: los dominios de Tailscale resuelven DNS, pero la conexión HTTPS a `controlplane.tailscale.com/key?v=148` agotó el tiempo de conexión. La comprobación con curl devolvió exit 28, HTTP 000 y 5,001 s. No se generó una AuthURL ni se incorporó un dispositivo a la red. No se configuró Serve/Funnel ni se abrió un endpoint público.

El siguiente paso depende de que el sandbox pueda conectarse al plano de control de Tailscale. No requiere repetir la autorización de instalación, arranque o vinculación; esas autorizaciones ya fueron dadas. No se debe declarar conectividad desde HONOR/Muse hasta completar login y una prueba desde un segundo cliente.

Además, Molab recreó el sandbox y eliminó el GGUF y los binarios. El archivo Q4_K_M MTP se volvió a descargar; se corrigió el preparador para respaldar directorios incompletos de llama.cpp que sobreviven sin .git, en lugar de fallar al clonar o borrar archivos. La restauración se lanzó en una sesión de proceso independiente para que una interrupción de diagnóstico del notebook no detenga la compilación. El estado final de esa restauración se registra por separado en la evidencia del 5/11.

Cloudflare Quick Tunnels se descartó para OpenCode directo porque su documentación indica que no soporta SSE. La API local sí se verificó con SSE. No se inició un túnel alternativo sin confirmación.

Muse Android no se abrió ni configuró en esta ejecución; se preparó el servidor compatible con OpenAI que puede consumir. No se afirma prueba real desde el teléfono.

## Reproducción en Molab

Desde la raíz del repositorio, con el runtime y modelo preparados:

```bash
python molab/davidau_mtp_service.py start
python molab/davidau_mtp_service.py test
npm install --prefix davidau-gguf-lab/opencode-runtime opencode-ai@1.19.5
python molab/davidau_opencode_smoke.py
```

Para detenerlo:
```bash
python molab/davidau_mtp_service.py stop
```

El servidor requiere que el sandbox Molab siga activo y no es un servicio 24/7 garantizado. La clave y estado son locales al sandbox; no se promete persistencia entre recreaciones.

Fuentes oficiales:
- https://opencode.ai/docs/providers/
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
- https://tailscale.com/docs/features/tailscale-serve
- https://tailscale.com/docs/concepts/userspace-networking


## Estado operativo posterior — 5/11/2026

Molab mostró «Your sandbox stopped responding and was reclaimed» durante la recompilación de restauración. Se pulsó Relaunch una vez. La restauración no terminó y la API del sandbox reclamado se detuvo. La causa del cierre no fue confirmada; no se atribuye a OOM, cuota ni inactividad sin evidencia.

La instalación de Tailscale 1.104.5 y el arranque userspace sí se ejecutaron antes del cierre. No se alcanzó el login ni se vinculó el dispositivo. No se debe interpretar la validación histórica de API/OpenCode del 4/11 como disponibilidad actual. Evidencia resumida del incidente: evidence/molab-tailnet-20261105/setup-result.json.
