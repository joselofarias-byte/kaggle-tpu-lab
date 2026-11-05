# DavidAU Q4_K_M MTP en Molab — 4 de noviembre de 2026

Pruebas ejecutadas desde el navegador de Work, con la sesión autenticada de Molab y su GPU RTX PRO 6000 Blackwell Server Edition. Horas de los JSON en UTC: 5 de noviembre, 01:39; fecha local de Montevideo: 4 de noviembre.

## Modelo e integridad

- Repositorio: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF`.
- Revisión: `b9bce7e859c9b5047aca884e5bff76f8a0443de1`.
- Archivo real: `Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`.
- Tamaño: **18.498.573.856 bytes** (17,23 GiB).
- SHA-256 calculado leyendo el archivo completo y comparado con LFS de Hugging Face:
  `bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c`.
- Descarga más cálculo SHA: 60,262 s.
- GGUF v3; arquitectura `qwen35`; file_type 15; 65 bloques y `nextn_predict_layers=1`. El lector encontró tensores MTP en `blk.64.nextn.*`.
- No se sustituyó por Q8_0 ni por la variante Q4 sin MTP. Ambas mediciones siguientes usan exactamente el mismo archivo MTP; cambia la activación del decodificador especulativo.

## Entorno y compilación real

Molab anuncia 4 CPU, 32 GiB RAM y RTX Pro 6000. El driver es 595.71.05; nvidia-smi muestra compatibilidad CUDA 13.2 y 97.887 MiB de memoria física. El runtime enumera 97.251 MiB utilizables. PyTorch de la sesión es 2.11.0+cu130 y su CUDA 13.0 es independiente del toolkit usado para compilar.

Se completó la compilación de llama.cpp `cd959bae99ff4bed22bef2604f45241c09bbb8c8` (v0.9.0, build local identificado como b1-cd959ba):

```
-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=120
-DCUDAToolkit_ROOT=/tmp/uv-venv/lib/python3.13/site-packages/nvidia/cu13
-DLLAMA_CURL=OFF
-DLLAMA_BUILD_TESTS=OFF
cmake --build llama.cpp/build-cu134 -j 4 --target llama-cli llama-bench llama-server
```

CMake seleccionó 120a para Blackwell. Los tres binarios respondieron a --help; llama-server --list-devices detectó CUDA0. llama-cli generó español y llama-bench ejecutó tres repeticiones. Estas mediciones usan los binarios compilados en Molab, no el binario oficial precompilado del experimento anterior.

Toolchain consistente: nvcc/CRT/runtime/NVVM/NVRTC 13.4.92, cuBLAS 13.8.0.4, CCCL 13.3.4.3.1, CMake 4.4.4 y Ninja 1.13.2. Se mantienen juntos los headers y bibliotecas del entorno virtual; enlaces .so y lib64 preparados para FindCUDAToolkit.

El intento anterior mezclaba toolkit del sistema y entorno virtual, carecía de CCCL y terminó interrumpido. Se reanudó la compilación consistente desde sus objetos, hasta enlazar los tres ejecutables. El log documenta avisos no fatales por NCCL/OpenSSL ausentes, un fallback de descarga de recursos UI y AMX no disponible. CUDA e inferencia funcionaron.

Almacenamiento: df -h reporta un filesystem virtual de 8,0E con 169G usados; no permite deducir la cuota real de Molab. La descarga y lectura íntegra del GGUF sí se completaron. Molab advierte que los archivos del sandbox requieren persistent cache para conservarse entre sesiones; no se afirma persistencia del modelo.

## GPU, chat y rendimiento medido

Prueba explícita con log verbose:

```
load_tensors: offloaded 66/66 layers to GPU
load_tensors: CUDA0 model buffer size = 16949.10 MiB
common_speculative_init_result: creating MTP draft context against the target model
spec: adding speculative implementation 'draft-mtp'
```

Quedan buffers de host, incluidos 682,03 MiB de CPU_Mapped, aun con todas las capas descargadas a GPU.

Servidor: contexto 8192, una secuencia, Flash Attention on, Jinja del GGUF, reasoning off, temperatura 0 y seed 0. MTP usa `--spec-type draft-mtp --spec-draft-n-max 3`. Cada fila es la media de tres respuestas de 85 tokens al mismo prompt en español.

| Activación | Decode servidor | Tiempo HTTP medio | Carga hasta health | Pico total GPU muestreado |
|---|---:|---:|---:|---:|
| MTP apagado | 54,522 tok/s | 1,791 s | 3,527 s | 18.692 MiB / 18,254 GiB |
| MTP activo | 81,028 tok/s | 1,317 s | 3,514 s | 19.748 MiB / 19,285 GiB |

MTP aumentó el decode **48.6%** en este prompt; redujo el tiempo HTTP medio aproximadamente 26,4%. No es una garantía para otras tareas. Las respuestas fueron iguales en ambas configuraciones. En cada generación española MTP propuso 102 tokens y aceptó 50 (49,0%).

VRAM: muestreo de uso total de la GPU cada 200 ms durante las peticiones. Incluye otros procesos; no es un pico exacto de asignación ni una medición de todo el arranque. Primera petición sin cache; segunda y tercera reutilizan 32 tokens de prompt. El tiempo HTTP excluye carga. Los tok/s son el campo predicted_per_second del servidor, que calcula el tiempo de decode sobre pasos de generación.

Benchmark independiente **sin decodificación MTP**, con el mismo GGUF, `-p 512 -n 128 -r 3 -ngl 99`:
- Prefill: **3816,115 ± 70,482 tok/s**.
- Generación: **57,513 ± 0,013 tok/s**.
- KV f16, batch 2048, ubatch 512; threads automáticos 20. Molab anuncia 4 CPU, mientras el proceso detecta 20; no se asume una asignación de 20 CPU dedicadas.
- No comparar directamente este prefill 512 con el benchmark anterior de Q8_0 que usó 128.

La plantilla de chat embebida se ejercitó tanto en conversación en español como en un ciclo assistant/tool/assistant; /props se guardó en evidencia.

## Herramientas: sintaxis y ejecución separadas

Ambos modos devolvieron una llamada OpenAI estructurada:

```json
{"name":"sumar","arguments":"{\"a\":17,\"b\":25}"}
```

1. **Soporte sintáctico:** finish_reason=tool_calls, nombre sumar y argumentos JSON válidos; tool_choice=required. No se evaluó selección autónoma con tool_choice=auto.
2. **Ejecución real:** un dispatcher Python con allowlist comprobó el nombre, las claves y que ambos argumentos fueran enteros; ejecutó a+b y obtuvo **42**.
3. Se devolvió 42 como mensaje role=tool con el tool_call_id correspondiente.
4. El modelo respondió **«El resultado es cuarenta y dos.»**.

El modelo produce la petición; Python ejecuta la herramienta. No se habilitaron herramientas de shell ni agentes integrados. En la llamada estructurada, MTP aceptó 27/30 tokens propuestos (90%); esto no representa la aceptación general.

## Seguridad y alcance

Los servidores de prueba escucharon sólo en loopback y exigieron una clave aleatoria efímera vía LLAMA_API_KEY. Se verificó rechazo de /v1/models sin autenticación en ambos modos. La clave no se imprime ni se guarda. Los procesos terminaron al finalizar las pruebas. No se abrió un endpoint público ni túnel.

Se probaron carga, CLI, CUDA, benchmark, español, plantilla embebida, MTP efectivo y un ciclo real de suma. No se probaron visión, contexto largo, robustez de código, integración externa con Muse/OpenCode ni políticas de rechazo.

## Archivos y reproducción

- `davidau_mtp_prepare.py`: fija revisión/tamaño/SHA del GGUF, instala el toolkit coherente, compila y llama al validador.
- `davidau_mtp_validate.py`: pipeline de inferencia y herramientas.
- `evidence/davidau-mtp-20261104/`: 31 archivos exportados del notebook, incluidos scripts exactos ejecutados, logs de build, log verbose de GPU, JSON completos y plantilla /props.

Desde la raíz del repositorio en un entorno Molab CUDA:
```bash
uv run molab/davidau_mtp_prepare.py
```
El preparador consolidado se comprobó por sintaxis; la compilación, descarga y validador que reproduce sí se ejecutaron en Molab, según los scripts y logs exactos conservados en evidence. No se presenta una segunda ejecución completa desde un sandbox limpio.

Notebook: https://molab.marimo.io/notebooks/nb_iVBLiVKuNhwokM43qyfp1H
PR existente de esta rama: https://github.com/joselofarias-byte/kaggle-tpu-lab/pull/12
Rama de trabajo: feature/molab-llm-lab. Main sin modificar.
