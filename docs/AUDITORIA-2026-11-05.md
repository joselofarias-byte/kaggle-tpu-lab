# Auditoría técnica — Kaggle TPU Lab — 2026-11-05

## Alcance y procedencia

Inspección estática del árbol seleccionado en `feature/molab-llm-lab`, inicialmente
`cbb347269710e59013f65ff420344960bd44668d`, sus referencias locales, notebooks,
launchers, motores, scripts y los dos workflows presentes en ese commit. Los
cambios concurrentes de otros agentes **no forman parte de esta instantánea**.
Los hallazgos sobre código existente se refieren a ese commit, aunque otro cambio
posterior los corrija. No se inició sesión en Kaggle/Molab ni se asignó hardware.
No se ejecutaron modelos, benchmarks ni túneles durante esta auditoría.

Referencias locales consultadas:

- `origin/main`: `dc32b4a8c0d40a942ed58e3b7f99f29f3998e053`, mensaje de merge
  «Molab: DavidAU Q4_K_M MTP compilado con CUDA, 81 tok/s y herramientas verificadas (#12)».
  Es evidencia histórica de un PR, no consulta del estado vivo de GitHub.
- `origin/feature/molab-authenticated-api`: `2422ed3`, cierre de servidor/sandbox
  registrado; no equivale a medir el estado actual de una cuenta.
- `origin/audit/upstream-2026-11-04`: `7d61687`, comparación upstream.
- `origin/cursor/experimental-vllm-029-3ac3`: `231a1ec`, experimento vllm-tpu 0.29.0.
- Existen además refs locales de perfiles, auditoría, Android y reparaciones de
  heartbeats, cola, estado TPU y recuperación de túnel. No fueron fusionadas.
- No hay refs `refs/pull/*` en esta copia. El listado completo de PR abiertos,
  reviews, issues y ejecuciones/URLs de Actions **queda pendiente de consulta
  autorizada del proveedor por el agente principal**. No se infiere de refs locales.

## Conclusión operativa

Hay dos rutas distintas: Qwen BF16 con vLLM-TPU y GLM con motor JAX propio en
Kaggle; GGUF con llama.cpp/CUDA y notebooks Transformers en Molab. No existe
justificación para sustituir automáticamente el modelo DavidAU por GPT-OSS.
GGUF/CUDA no es ejecutable directamente sobre TPU: el lector GGUF del motor GLM
es una implementación especializada para esa arquitectura, no un backend TPU
genérico de llama.cpp ni prueba de soporte DavidAU.

**Molab no debe presentarse como un hosting de API remota.** La política oficial
actual considera abuso conectar proxies remotos o exponer el cómputo Molab al
tráfico exterior, y advierte que el control remoto, por ejemplo SSH, puede causar
terminación sin aviso. Una API autenticada en loopback para cómputo interactivo
no concede permiso para un túnel público. No se propone eludir esta restricción.
OpenCode remoto desde PC/VPS contra Molab queda bloqueado por política salvo una
vía expresamente autorizada; no es un fallo corregible instalando cloudflared.

## Rutas TPU: topología y reproducibilidad

| Ruta | Implementación observada | Límites de la evidencia |
|---|---|---|
| `qwen38-27b/kernel/serve_qwen38.py` | `vllm-tpu==0.28.0`, CPU torch mediante uv, TP=8, caché XLA y datasets; parche rollback MTP | Los preflight llaman JAX en subproceso. Ver ocho dispositivos TPU no identifica por sí solo v5e, coords, HBM ni sharding válido. Algunas respuestas inconclusas permiten continuar. |
| `glm53-flash/kernel/serve_glm53.py` y `engine/glm53/` | JAX/Pallas, expertos cuantizados, malla obtenida de `jax.devices()`, `libtpu==0.0.42.*` | El wildcard no es un lock exacto. JAX/PyTorch y dependencias transitivas no quedan fijados globalmente. Preflight también permite casos inconclusos. |
| Notebooks `.ipynb` de Qwen/GLM | Empaquetado de scripts/motor, cachés y datasets compartidos | Son superficies duplicadas: validar sincronización con `tools/pack_notebook.py`; ejecutar el notebook no se demostró aquí. |

Los comentarios y reportes del código anuncian «TPU v5e-8», tiempos de arranque,
262144 tokens y mejoras MTP. Son configuración o evidencia previa, **no mediciones
de esta ejecución**. Qwen incluso escribe una etiqueta fija de acelerador en su
reporte; no usarla como detección. Probar identidad, número de procesos, coords,
HBM, carga y límites de contexto en la sesión real antes de anunciar capacidad.

Qwen usa `--exclude-newer` si el manifest aporta fecha de construcción; mejora la
resolución temporal, pero no reemplaza lockfiles/hashes y se pierde en el fallback
pip. `uv` se instala sin pin; el fallback usa extra-index de CPU, sin la misma
selección explícita de backend. GLM instala `transformers>=5.16`, torch CPU y
otras bibliotecas sin lock. La compatibilidad JAX/libtpu/Pallas y PyTorch-XLA debe
medirse por ruta; no hay aquí prueba de una ruta de inferencia PyTorch-XLA para
DavidAU. La rama experimental 0.29 no demuestra estabilidad de la rama activa.

Los paths `/tmp/venv`, `/tmp/xla_cache` y `/tmp/hf` son efímeros; la persistencia
real TPU depende de datasets/cachés restaurados y del manifest compatible.
`keepalive_min=480` es un límite de vida, **no detección de inactividad**. El
comentario de límite Kaggle de nueve horas no se verificó contra una cuenta
actual; disponibilidad, verificaciones y cuota efectiva quedan pendientes.

## Molab: versiones, modelo y memoria

`molab/jolufasoft_llm_lab.py` y `molab/gpu_smoke_test.py` requieren Python >=3.11,
marimo >=0.17, torch >=2.8 y Transformers **4.57.6**; el primero añade kernels
>=0.9,<0.12. `__generated_with` registra 0.17.0/0.17.8, no fija el runtime.
El smoke test sí consulta nombre/VRAM CUDA reales; no asegura una GPU particular.
Los notebooks Transformers y los scripts GGUF usan backends/formatos distintos:
no asumir que el selector del notebook carga automáticamente el GGUF DavidAU.

`molab/davidau_mtp_prepare.py` fija modelo/revisión/hash, llama.cpp
`cd959bae99ff4bed22bef2604f45241c09bbb8c8`, CMake 4.4.4, Ninja 1.13.2,
nvcc/CRT/runtime/NVVM/NVRTC 13.4.92, cuBLAS 13.8.0.4 y CCCL 13.3.4.3.1.
Es una receta del experimento, no garantía de compatibilidad con cualquier GPU.
Adaptar arquitectura CUDA al dispositivo; no generalizar `120`/`120a` a T4/P100.
La versión CUDA máxima indicada por nvidia-smi **no es** el toolkit nvcc instalado
ni `torch.version.cuda`; registrar las tres por separado.

### Evidencia histórica conservada (no reejecutada)

Fuentes: `molab/RESULTADOS-DAVIDAU-MTP-2026-11-04.md` y
`molab/evidence/davidau-mtp-20261104/mtp-evidence/{summary,status,model}.json`,
`devices.txt`, metadatos y logs. El reporte explica fecha local 4/11 y UTC 5/11.

- Revisión HF `b9bce7e859c9b5047aca884e5bff76f8a0443de1`.
- MTP Q4_K_M: archivo único de 18.498.573.856 bytes; SHA-256
  `bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c`.
  GGUF v3, arquitectura `qwen35`, 65 bloques, una capa nextn según reporte.
- `devices.txt`: RTX PRO 6000 Blackwell Server Edition, 97251 MiB enumerables.
- `summary.json`: sin MTP 54,522 tok/s y 18692 MiB pico; con draft-MTP
  81,028 tok/s y 19748 MiB pico. Tres generaciones de 85 tokens por modo;
  carga 3,527/3,514 s y autenticación reportada activa en ambos.
- Existen round trips de herramientas en `tool-execution-*.json` y estado
  `finished=true`, `errors=[]`. No prueba calidad general de programación,
  contexto máximo, estabilidad prolongada ni acceso externo autorizado.
- El experimento anterior Q8_0 registra 29.787.699.808 bytes en
  `molab/evidence/davidau-20261104/model.json`; no confundirlo con Q4 MTP.

Estos tamaños no son requisitos de RAM/VRAM: añadir KV cache, contexto,
concurrencia, buffers y offload. Un pico histórico no es un mínimo universal.
El inventario HF **actual**, todas sus cuantizaciones y shards, y contexto útil
máximo quedan pendientes; un archivo único comprobado no demuestra que todas
las variantes sean monolíticas. Tampoco cabe extrapolar velocidad a GPU menor.

### Persistencia y límites oficiales

Consultados el 2026-11-05:

1. [Guía Molab](https://docs.marimo.io/guides/molab/): configuración por defecto
   4 CPU/32 GB, GPU opcional anunciada RTX Pro 6000/96 GB; sesiones hasta doce
   horas e inactividad superior a noventa minutos. Son condiciones documentadas,
   no hardware ni cuota medidos para la cuenta actual. La guía ofrece `marimo pair`
   para colaboración con agentes: no lo confundir con exponer un servidor LLM.
2. [Restricciones](https://marimo.io/pages/molab/restrictions): tráfico exterior,
   proxies, control remoto y servicios ajenos al cómputo interactivo; también
   prohíbe revender/proxyar/reexponer endpoints AI de Molab. La página no especifica
   una cuota numérica de GPU ni autoriza túneles con API key.
3. [Persistencia, 26/08/2026](https://marimo.io/blog/seamless-storage-in-molab):
   notebooks nuevos desde esa fecha conservan datos subidos/creados mediante el
   explorador de archivos; `mo.persistent_cache` tiene límites de tamaño/TTL
   modificables. Un path HF o carpeta en `/marimo` por sí solo no garantiza
   conservación de 18–30 GB entre sesiones. Validar restauración antes de prometer
   que no se repetirá la descarga o compilación.

## API, lanzamiento y seguridad (instantánea inicial)

- `molab/scripts/molab_inference_api.py` exige clave >=24 caracteres y comparación
  constante; autentica health/models/chat, pero en `cbb3472` devuelve 400 para
  streaming y herramientas. No describir ese bridge histórico como SSE completo.
  Otros agentes pueden corregirlo; evaluar sus pruebas en el diff integrado.
- Los servidores llama.cpp de experimentos cuentan con evidencia de autenticación
  y herramientas, separada del bridge Transformers. Que un modelo genere XML o
  JSON no demuestra tool calls estructurados ni un round trip exitoso.
- `launch.py` genera claves aleatorias pero las inyecta en el código enviado,
  guarda `api_key` en `~/.kaggle-tpu-lab.json` con `write_text` sin modo explícito,
  y las imprime en status. Revisar permisos 0600, kernel privado y separación por
  cuenta; no publicar ese estado ni salidas de consola sin redactar.
- Qwen y GLM imprimen claves y las incluyen en eventos `ready` enviados por ntfy
  cuando hay topic. Un topic no equivale a autenticación; evitar distribuir claves
  en canales/eventos sin control de acceso. No se reproducen valores aquí.
- GLM genera clave por defecto, pero `_auth` admite acceso si `API_KEY` resulta
  vacía y usa comparación ordinaria. `/health` se sirve antes de autenticar.
  Auditar todos los endpoints y evitar depender de un proxy para autenticar.
- Los scripts TPU incluyen cloudflared. No reutilizar esa decisión en Molab;
  también verificar políticas Kaggle antes de publicar un endpoint. Autenticación
  API no sustituye autorización de plataforma ni garantiza TLS extremo a extremo.
- Los comandos de apagado del launcher y las guardas temporales no prueban cierre
  de recursos: verificar proceso y estado del proveedor. No mantener actividad
  artificial para eludir cuotas/inactividad ni prometer encendido remoto sin API.

## Workflows de la instantánea

- `.github/workflows/python-syntax.yml`: Python 3.12; compila únicamente launcher y
  dos servidores TPU. No cubre notebooks Molab, tests CPU, SSE, auth, lint ni motores.
  `py_compile` no valida imports, versiones, semántica ni ejecución en acelerador.
- `.github/workflows/mobile-android-es.yml`: upstream móvil fijado a SHA, Node
  22.12.0 (nombre del paso dice 20), Python 3.11, JDK21, npm ci, tests, build,
  Capacitor, Gradle y APK/checksum como artefactos. `contents: read`; no modelos
  enormes como artefactos ni asignación GPU/TPU. Actions principales usan tags,
  no SHA. No es un workflow de validación del laboratorio de inferencia.
- Workflows adicionales en otras refs no son configuración efectiva de esta
  instantánea. El nuevo CI concurrente deberá verificarse aparte; no hay enlace
  ni resultado CI vivo confirmado por esta auditoría.

## Diagnóstico nuevo, seguro y reproducible

```sh
python3 tools/probe_accelerators.py > accelerator-report.json
python3 tools/probe_accelerators.py --probe-tpu > accelerator-report-tpu.json
python3 -m unittest discover -s tools/tests -p 'test_probe_accelerators.py' -v
```

El modo por defecto no importa JAX/PyTorch ni inicializa TPU. Obtiene versiones
por `importlib.metadata`, RAM de `/proc/meminfo`, GPU real mediante nvidia-smi,
CUDA máxima del driver, nvcc y llama-server si están en PATH. Ausencia de paquetes
o ejecutables se expresa con null/estado, no se instala nada. `--probe-tpu`
inicializa **solo mediante consentimiento explícito** el backend TPU de JAX en
un subproceso de hasta 60 segundos, con count, process_count y coordenadas cuando
el dispositivo las ofrece. No reserva hardware, no hace inferencia ni benchmark.

No vuelca variables de entorno, hostname, UUID/serial GPU, stderr de comandos,
credenciales ni rutas de cuenta. La RAM reportada es la vista del host, no una
garantía de asignación del contenedor; la VRAM libre es una muestra instantánea.
`null` de versión de un binario también puede significar salida no reconocida.
El timeout acota la comprobación pero no demuestra que una TPU ausente sea un
fallo de cuota o de hardware.

Verificación local de este cambio: diez pruebas unitarias CPU con mocks pasan;
CLI real en sandbox informa Python 3.12.3, x86_64, 16 GiB RAM visible, sin
nvidia-smi ni versiones instaladas de JAX/torch/llama; TPU `not_probed`. Esto no
certifica ausencia física de TPU ni hardware de otra plataforma.

## Matriz de aceptación pendiente

| Validación | Estado / requisito |
|---|---|
| Diagnóstico sin inicialización TPU y JSON CPU | Probado localmente, diez tests simulados y CLI real |
| Topología TPU, HBM, sharding y carga Qwen/JAX | Pendiente sesión Kaggle autorizada con TPU real |
| Compatibilidad PyTorch-XLA | No demostrada; no ruta equivalente validada aquí |
| GPU/CUDA y compilación en hardware distinto | Pendiente dispositivo real; no asumir Blackwell |
| Inventario HF completo, shards e integridad | Hash/revisión histórica Q4; inventario vivo pendiente |
| DavidAU, MTP y tool calling | Evidencia histórica local; repetir con contexto/template/config registrados |
| SSE/auth/modelos/chat integrados | Verificar pruebas del agente principal; no probado en hardware por esta auditoría |
| OpenCode local dentro del entorno permitido | Pendiente interacción autorizada y config sin secretos |
| Molab API remota desde PC/VPS | No prometida: contradice restricción oficial de ingreso exterior |
| TTFT, prefill/decode, RAM/VRAM y contexto máximo | Solo cifras históricas acotadas; falta campaña real comparable |
| Persistencia tras reinicio y otra cuenta propia | Pendiente prueba de restauración; credenciales separadas |
| Start/status/stop e inactividad efectiva | Inspección estática; verificar proceso y apagado real por plataforma |
| Issues/PR, publicación, links CI y resultado vivo | Agente principal con acceso autorizado; no inferidos de logs |

No se fusionaron ramas, no se pagó infraestructura, no se descargaron pesos y no
se modificaron experimentos. Esta auditoría no acredita que todo el criterio de
aceptación esté cumplido: distingue implementaciones existentes, pruebas locales,
evidencia histórica y bloqueos de política/hardware.

## Actualización de integración del agente principal

Consulta autenticada de GitHub el 5/11/2026: 18 ramas remotas, PR #12 integrado;
PR #14 abierto en `feature/molab-authenticated-api`, sin incorporarlo ni sobrescribirlo.
Issues está deshabilitado. Se inspeccionaron los listados de PR abiertos/cerrados,
refs y workflows; no se afirma revisión línea por línea de cada rama experimental.
Los PR #3/#4/#5/#10/#11/#13 permanecen separados; no se integra su trabajo a ciegas.
CI previa sólo comprobaba sintaxis de `launch.py` y dos kernels TPU; el nuevo workflow
`lab-tests.yml` agrega tests CPU/API, lint, notebooks y shells, sin solicitar GPU/TPU.

El inventario vivo HF se consultó y guardó en
`docs/evidence/davidau-files-2026-11-05.json`: revisión `b9bce7e8…`, 20 GGUF de pesos,
3 proyectores, sin shards actuales. Se registraron tamaños y hashes LFS declarados;
no se descargaron los pesos ni se verificó su contenido localmente en esta ejecución.
`docs/evidence/local-probe-2026-11-05.json` registra el entorno de prueba CPU:
Python 3.12.3, sin nvidia-smi/CUDA, PyTorch, JAX, libtpu ni llama-server instalados.
No se inicializó TPU. La RAM informada de host no garantiza memoria de contenedor.

Se agregan pruebas con backends CPU para API/SSE, validación de tools en el cliente,
configuración OpenCode, gestión de procesos y selección/shards GGUF. No sustituyen
carga o generación real. Los notebooks marimo existentes pasan validación estática;
no se reclama ejecución interactiva GPU ni evidencia visual nueva. El criterio de
aceptación completo permanece **parcial**, por hardware/acceso ausente y por la
restricción de Molab para clientes externos. No hay benchmarks nuevos de modelos.

Validación final local: **88 tests CPU aprobados**, una advertencia de deprecación
Starlette/AnyIO; 65 fuentes Python compiladas, JSON de notebooks válido, `marimo
check` en ambos notebooks Molab sin incidencias, lint selectivo y sintaxis shell
correctos. Resumen legible por máquina: `docs/evidence/cpu-validation-2026-11-05.json`.
La suite incluye detener un proceso CPU propio real mediante pidfd; las respuestas
LLM son dobles de prueba, no inferencia. No se ejecutó la compilación CUDA.
