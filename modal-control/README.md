# Controlador Modal para la APK

Backend independiente: FastAPI sobre HTTPS de Modal, con **un único sandbox GPU
administrado a la vez**. No se ha desplegado ni se han creado recursos de pago
como parte de la implementación/pruebas.

## Despliegue listo para Termux y Actions

La guía completa con comandos, entrega cifrada de conexión y prueba GPU manual
está en [DEPLOYMENT.md](DEPLOYMENT.md). Los workflows son
`modal-deploy.yml` y `modal-inference.yml`; ambos requieren ejecución explícita.
Desplegar/verificar no crea una GPU, pero construye imágenes y ejecuta CPU/RAM
del controlador: no se garantiza coste cero.

## Seguridad y límites

- Sólo dos perfiles finitos: `qwen-small` (Qwen2.5 0.5B, T4, CPU 2,
  RAM 8192 MiB) y `qwen-7b` (Qwen2.5 7B, L40S, CPU 4, RAM 16384 MiB).
- Todas las rutas requieren `Authorization: Bearer <CONTROLLER_TOKEN>`.
  Token aleatorio de al menos 32 caracteres, en un **Modal Secret** y en el
  `secureStore` de la APK. Nunca empaquetar credenciales de Modal, tokens de
  proveedor, ni el token del controlador en el APK/repositorio/logs.
- El endpoint del controlador es HTTPS. El túnel HTTPS del worker es privado
  a nivel de aplicación: requiere otro token aleatorio por sesión. No se devuelve
  su URL ni su token al teléfono. vLLM sólo escucha en `127.0.0.1`.
- `POST /api/session` es la **única** operación que crea GPU. No hay autoinicio
  por chat, sondeo, registros, reinicio o tarea programada.
- El sandbox recibe `timeout=max_minutes*60` en su creación (5–120 minutos).
  Modal hace cumplir este límite aunque se cierre la APK/controlador.
- Un watchdog **dentro del sandbox** termina su proceso principal después de
  2–30 minutos sin actividad de inferencia. Estado, salud y registros no reinician
  ese reloj. El reloj se arma una sola vez cuando un bucle interno detecta
  que vLLM está listo, nunca por un sondeo del cliente. Durante descarga/carga
  no vence la inactividad, pero el timeout absoluto sigue corriendo.
  Una inferencia en curso no es inactividad, pero nunca evita el timeout absoluto.
- vLLM está fijado a `vllm/vllm-openai:v0.8.5`, contexto 4096, una secuencia,
  float16 y 85% de memoria GPU. CPU y RAM tienen límite duro igual al perfil.
  Se conserva el Python 3.12 de esa imagen;
  no se instala un Python separado que oculte vLLM.
- El estado persistente guarda intención, UUID de petición, nombre, identificador
  del sandbox y token interno antes de crear el recurso. El nombre fijo y único
  dentro de la app impide dos sandboxes simultáneos, incluso si se pierde una
  respuesta de creación. Antes de crear se comprueba también el nombre reservado.
  El controlador tiene un contenedor y ocho entradas simultáneas; serializa
  las mutaciones con un lock local, liberado durante inferencia para permitir
  consultas y parada sin esperar el chat. Sólo se admite una inferencia en
  curso; chats adicionales devuelven 429 sin ocupar slots hasta terminar.
- Detener utiliza únicamente el identificador persistido del sandbox administrado.
  Para recuperar una creación ambigua sólo busca el nombre reservado en esta
  app; nunca enumera ni detiene otras apps/sandboxes.
- Un error de transporte queda `UNKNOWN`, conserva la identidad e impide otro
  inicio. Una respuesta de `terminate` sin comprobación no equivale a `STOPPED`.
  Se requiere que `poll()` confirme la finalización.
- Los logs tienen un máximo de 100 líneas de 2000 caracteres, se redactan
  tokens/URLs y vLLM no registra peticiones. No enviar secretos en el chat.

## Instalación y despliegue (manual y con autorización de gasto)

No ejecutar estos pasos sin una cuenta Modal propia y autorización para pagar
cómputo. Las pruebas locales de abajo **no** requieren credenciales.

```sh
cd modal-control
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
modal setup
```

1. En el panel de Modal crear un Secret llamado `android-modal-control` con la
   clave `CONTROLLER_TOKEN`. Generar localmente un valor con
   `python -c 'import secrets; print(secrets.token_urlsafe(48))'` y copiarlo sólo
   al Secret y al almacenamiento seguro de la APK. No compartirlo en capturas.
2. Ejecutar `modal deploy deploy.py` desde este directorio. Creará la app
   `android-modal-control`, un Dict persistente y las imágenes. El controlador
   no reserva GPU en despliegue. Controlador y sandbox comparten imagen para que
   Modal construya y registre la imagen antes de recibir peticiones; sólo el
   sandbox solicita GPU. El controlador limita CPU a 0.25 y RAM a 512 MiB.
   El endpoint HTTPS publicado es la URL base
   para la APK, sin sufijos `/api/...`.
3. Configurar URL base HTTPS y token en la APK; consultar perfiles/estado antes
   de iniciar. Para cada intención nueva de inicio generar un UUID y conservar
   **ese mismo UUID y parámetros en reintentos**. Una petición nueva mientras
   hay sesión activa o incierta devuelve 409. Un UUID ya completado nunca relanza.
4. Iniciar explícitamente un perfil, esperar `READY`, chatear y detener. Comprobar
   `STOPPED` antes de asumir que no queda GPU en ejecución. El cierre del teléfono
   no es una orden de detener: actúan el watchdog y el límite absoluto.
5. Para rotar el token, actualizar el Secret, redesplegar y actualizar la APK.

Usar esta app/nombre/Dict exclusivamente para este controlador. No lanzar otro
controlador con un Dict distinto y el mismo nombre de sandbox. No borrar ni
editar el Dict durante una sesión; contiene la identidad necesaria para parar.
No hacer despliegues solapados mientras se cambia el código de control.
El Dict conserva los UUID históricos para mantener idempotencia tras reinicios;
no es un almacén de conversaciones ni de credenciales del proveedor.

Si una creación nunca llega a confirmarse y la búsqueda por nombre no encuentra
nada, se mantiene `UNKNOWN` incluso después del vencimiento estimado. Es una
restricción intencional: no se inventa confirmación de parada ni se crea otra GPU.
El operador debe comprobar en Modal el sandbox de la app y el nombre exacto
`android-owned-model-v1`; si existe, detener **sólo ese recurso** desde el panel.
Después de confirmar que ya no existe ningún arranque pendiente, puede retirar
la entrada `current` del Dict para permitir un nuevo inicio. No borrar las
entradas `request:<UUID>`: preservan la protección contra reintentos viejos.
No hay endpoint para operar sobre identificadores arbitrarios.

## CORS

Orígenes explícitos permitidos: `http://localhost`, `https://localhost`,
`capacitor://localhost` y Vite local `http://localhost:5173`. Sin comodines ni
credenciales CORS/cookies; el bearer sigue siendo obligatorio en cada ruta.
Otros hosts/puertos de desarrollo requieren una modificación explícita.

## Contrato HTTP

Todas las claves JSON son `snake_case` y todas las rutas están autenticadas.

| Método/ruta | Entrada | Respuesta |
| --- | --- | --- |
| `GET /api/profiles` | — | `{profiles:[{id,name,model,gpu,cpu,memory_mib,description}]}` |
| `GET /api/session` | — | `{session:null}` o `{session:Session}` |
| `POST /api/session` | `{profile_id,max_minutes,idle_minutes,request_id}` | `{session:Session}` |
| `DELETE /api/session` | — | `{session:Session\|null}` |
| `GET /api/logs` | — | `{lines:string[]}` |
| `POST /api/chat` | `{messages:[{role,content}],max_tokens?}` | `{choices:[{message:{content}}]}` |

`Session`: `id, profile_id, model, gpu, cpu, memory_mib, status, created_at,
expires_at, idle_minutes, max_minutes, detail?`. Timestamps UTC ISO8601.
Estados: `STARTING`, `READY`, `STOPPING`, `STOPPED`, `ERROR`, `UNKNOWN`.
`expires_at` es una estimación conservadora desde la intención de creación;
el timeout real lo impone Modal desde la creación del sandbox. No es una
medición de facturación ni del instante exacto en que se asignó GPU.

`idle_minutes` no puede superar `max_minutes`. Roles: `user|assistant|system`. `max_tokens` por defecto 256, entero 1–1024;
64 mensajes como máximo, 16000 caracteres por mensaje y 32000 por conversación.
No hay streaming. Errores: 401 autenticación, 409 conflicto/no listo, 422 entrada
inválida, 429 inferencia ya en curso y 502 inferencia no disponible. No se exponen excepciones del proveedor.
No hay desviaciones de las rutas/campos requeridos; se añaden límites de tamaño
al chat y rechazo de campos extra por seguridad.

## Coste: estimación, no facturación

Las GPU, CPU y RAM del sandbox cuestan mientras se asignan, incluidos descarga,
arranque y carga. También puede haber coste de CPU/RAM del controlador,
construcción de imágenes, transferencia u otros cargos de Modal. No se monta
volumen de modelos: cada nuevo sandbox puede volver a descargar pesos.

Para un presupuesto orientativo usar las tarifas actuales de la cuenta/región:
`segundos asignados × (tarifa_GPU/s + CPU × tarifa_CPU/s + GiB_RAM × tarifa_RAM/s)`.
El máximo configurado acota la duración de **cada** sandbox, no el gasto mensual,
ni prohíbe que el usuario inicie sesiones posteriores. No se inventan tarifas ni
se presenta el tiempo restante como factura. Revisar también los límites de
presupuesto y el panel de uso de Modal. Los registros locales no prueban ahorro
ni consumo real.

## Pruebas sin nube ni GPU

```sh
cd modal-control
python3 -m venv /tmp/modal-control-tests
/tmp/modal-control-tests/bin/pip install -r requirements-test.txt
/tmp/modal-control-tests/bin/python -m pytest -q
/tmp/modal-control-tests/bin/python -m compileall -q .
```

Las pruebas usan backend y SDK falsos, no llaman a Modal. La versión pública Modal 1.4.2 y las firmas
`Sandbox.terminate(wait=True)` e `Image.entrypoint([])` se verificaron localmente. Cubren autenticación en
las seis rutas, perfiles, límites, entradas inválidas, reintentos/concurrencia,
recuperación tras reinicio/creación ambigua, fallo de parada/sondeo, no autoinicio,
CORS explícito, consultas/parada durante chat, watchdog desde readiness,
chat, redacción, watchdog, timeout absoluto y argumentos del adaptador Modal.
Quedan pendientes el build real de imagen, arranque GPU, HTTPS remoto, tiempo de
carga, cierre real por inactividad/timeout e integración con la APK: requieren
credenciales y autorización de gasto. No se han ejecutado como parte del trabajo.

Referencias: [Sandbox SDK](https://modal.com/docs/sdk/py/latest/Sandbox.md),
[Sandbox guide](https://modal.com/docs/guide/sandbox),
[proxy auth](https://modal.com/docs/guide/webhook-proxy-auth.md),
[vLLM Dockerfile v0.8.5](https://github.com/vllm-project/vllm/blob/v0.8.5/docker/Dockerfile).
Se utiliza bearer propio en FastAPI, no `requires_proxy_auth=True`: activar esa
opción exigiría además credenciales proxy de Modal y cambiaría el contrato APK.
