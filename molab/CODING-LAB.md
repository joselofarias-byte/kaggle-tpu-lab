# LLM Lab: API autenticada y clientes

## Límite obligatorio de Molab

[Molab prohíbe](https://marimo.io/pages/molab/restrictions) exponer cómputo a tráfico
externo, proxies, SSH y trabajos no interactivos. **No ejecutar túneles ni convertir
Molab en un backend remoto de OpenCode.** Su integración oficial con agentes
[marimo pair](https://docs.marimo.io/guides/molab/) controla notebooks, no ofrece un
permiso para reexponer inferencia. Usar Molab para pruebas interactivas en loopback.
Para OpenCode externo usar Kaggle sólo cuando sus condiciones y acceso lo permitan,
o una máquina propia ya autorizada; no se contrata infraestructura.

## Instalación local reproducible (CPU, sin descargar modelos)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r molab/requirements-test.txt
.venv/bin/python -m pytest molab/tests tools/tests -q
.venv/bin/python tools/probe_accelerators.py
```

Los notebooks `jolufasoft_llm_lab.py` y `gpu_smoke_test.py` conservan la ruta
Transformers de pruebas interactivas; **no cargan GGUF**. No pegar el repositorio
DavidAU GGUF en `AutoModelForCausalLM`: usar la CLI llama.cpp descrita abajo.
La UI histórica no implica que una GPU esté disponible hoy. No se cambia a GPT-OSS.

## Ruta prioritaria DavidAU GGUF

El catálogo consultado se guarda en
[`docs/evidence/davidau-files-2026-11-05.json`](../docs/evidence/davidau-files-2026-11-05.json).
Hay 20 pesos GGUF (IQ2/3/4, Q4/5/6/8, variantes MTP/LOW-MTP) y 3 proyectores,
ningún shard en esa revisión. La CLI admite futuros juegos completos de shards.
Q4_K_M MTP pesa 18.499 GB decimales (17.23 GiB), **no es su requisito total de VRAM**.
Hay que sumar KV cache, buffers, runtime y margen de seguridad. Contexto máximo
anunciado no equivale a contexto utilizable: medir cada configuración.

En una sesión GPU interactiva autorizada, con toolkit CUDA y CMake ya disponibles:

```bash
bash tools/build_llama_cuda.sh "$HOME/.cache/kaggle-tpu-lab"
.venv/bin/python -m pip install 'huggingface-hub==0.36.0'
.venv/bin/python molab/lab.py start \
  --server "$HOME/.cache/kaggle-tpu-lab/llama.cpp/build-cuda/bin/llama-server" \
  --filename Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf \
  --context 4096
.venv/bin/python molab/lab.py status
.venv/bin/python molab/lab.py benchmark --tokens 128
.venv/bin/python molab/lab.py stop
```

Usar `.venv/bin/python` si `python` no pertenece a ese entorno. El build fija el
commit histórico `cd959bae…`, usa arquitecturas nativas y el toolkit del host:
no instala drivers ni fuerza Blackwell/CUDA 13.4. Un driver compatible por sí solo
no garantiza compilación; `nvcc`, CMake y librerías CUDA son necesarios. El build
CUDA y la carga real quedan pendientes si no hay GPU. Usar otro directorio de
build si cambia el toolkit/hardware.

Sin `--filename` se selecciona una cuantización del **mismo** repositorio que pase
la admisión conservadora de RAM/VRAM; esto no garantiza que el KV cache quepa.
No hay fallback silencioso a CPU ni a otro modelo. Se pueden comparar
`--batch-size`, `--ubatch-size`, `--flash-attn` y `--context`; los parámetros se
registran en los reportes. `--idle-seconds` (600 por defecto) pide al servidor
suspender/descargar el modelo cuando no hay solicitudes: no apaga el sandbox,
no garantiza ahorrar cuota y debe verificarse en el binario/hardware real. Carga fallida debe probarse
con menos contexto o una cuantización explícita menor. Se reutiliza el caché HF;
`--cache-dir` permite elegir almacenamiento persistente **realmente disponible**.
Molab no garantiza retener descargas tras Shutdown: su política de almacenamiento
cambió el 26/8/2026. No prometer recuperación instantánea.

El servidor escucha en loopback, autentica con un archivo privado `api.key` y
usa el ID servido `molab-lab`. Estado y resultados privados quedan en
`~/.local/state/molab-lab`. Separar cuentas con `--state-dir` y cachés propios;
no copiar claves entre cuentas. `stop` detiene el proceso administrado, **no apaga
el notebook ni libera la asignación**. Usar Shutdown en Molab y comprobar que no
quede cómputo activo. No se implementa encendido remoto no autorizado.

## API y OpenCode

`molab/scripts/molab_inference_api.py` es una alternativa Transformers experimental
con API key, `/v1/models`, chat y SSE. No entiende tool calling: lo rechaza
explícitamente y por eso **no es un agente autónomo de programación**.
La ruta llama-server usa su template GGUF con `--jinja`; herramientas/MTP requieren
pruebas por modelo y binario. No ejecutar código arbitrario propuesto por el modelo.

Para comprobar un servidor local interactivo, establecer la clave de forma privada
(sin imprimirla, registrarla ni incluirla en historial), `MODEL_ID=molab-lab` y
`MOLAB_URL=http://127.0.0.1:8080`. Ejecutar:

```bash
python molab/scripts/smoke_test_coding_api.py
```

Exige 401 sin clave y con clave errónea, modelo listado, SSE con contenido,
`finish_reason` y `[DONE]`, tool call válida, suma en dispatcher allowlist y
respuesta final que consume 42. Sin esos gates no declarar compatibilidad.

En un **cliente con endpoint autorizado** (local o VPS), generar configuración:

```bash
python molab/scripts/configure_opencode.py \
  --url https://endpoint-autorizado.example/v1 \
  --model molab-lab --context 4096 --output 1024
```

No es una URL real ni un endpoint desplegado. El script conserva otros providers,
no sobreescribe el mismo provider y escribe atómicamente con permisos 0600.
La clave sólo se referencia como `{env:MOLAB_API_KEY}`; definirla privadamente en el
entorno del cliente. Seleccionar `jolufasoft-lab/molab-lab` en OpenCode.
[Formato oficial](https://opencode.ai/docs/providers/).
Ni la configuración generada ni tests con dobles de CPU prueban OpenCode real.

## TPU y resultados

Usar `python tools/probe_accelerators.py --probe-tpu` dentro de una sesión TPU para
registrar dispositivos/topología real. La ruta existente es `launch.py serve`,
`status`, `stop` con credenciales Kaggle propias. No pasar GGUF CUDA a vLLM-TPU.
Conservar vllm-tpu 0.28.0 y su parche hasta validar otra versión; PRs experimentales
no se integran automáticamente. No se inició una sesión Kaggle en esta auditoría.

Los benchmarks de noviembre 4 y del PR #14 son **históricos**, no nueva evidencia.
Esta entrega registra tests de CPU y el catálogo HF; no inventa tokens/s de hardware.
Comparaciones MTP, KV cache, batches y contexto necesitan ejecuciones GPU/TPU reales.
Ver [auditoría](../docs/AUDITORIA-2026-11-05.md) y estado de CI en el PR.
