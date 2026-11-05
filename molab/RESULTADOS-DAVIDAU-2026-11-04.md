# DavidAU 27B Q8_0 en Molab — 2026-11-04

Modelo: DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF.
Revision b9bce7e859c9b5047aca884e5bff76f8a0443de1. Archivo regular Q8_0, sin MTP: Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-Q8_0.gguf (29,787,699,808 bytes). Descarga real: 69.49 s.
GGUF architecture qwen35; 26,895,998,464 parámetros.

## Entorno y runtime

Molab autenticado; NVIDIA RTX PRO 6000 Blackwell Server Edition. nvidia-smi: 97,887 MiB totales; PyTorch: 94.97 GiB accesibles. CUDA 13.0, PyTorch 2.11.0+cu130, Python 3.13.11.
Binario CUDA oficial llama.cpp 0.9.0-dev, build 12110, commit cd959bae99ff4bed22bef2604f45241c09bbb8c8. Instalador oficial https://llama.app/install.sh con SKIP_INSTALL=1, archivos aislados en el directorio del experimento.
API OpenAI compatible escuchando solamente en 127.0.0.1:18992 durante cada prueba. Contexto 8192, batch de servidor 1, todas las capas GPU, flash attention, Jinja, reasoning off, temperatura 0, seed 0.

## Ejecuciones reales

| Prueba | Tokens generados | Tiempo HTTP | Decode servidor | VRAM total máxima muestreada |
|---|---:|---:|---:|---:|
| Español, definición de API REST | 47 | 1.435 s | 37.20 tok/s | 29,280 MiB |
| Código factorial iterativo | 53 | 1.602 s | 37.30 tok/s | 29,286 MiB |

La VRAM es el uso total de GPU muestreado cada 250 ms durante la petición; incluye otros procesos. No representa un pico exacto del asignador de tensores. 29,286 MiB = 28.60 GiB.
Tiempo HTTP excluye arranque/carga del servidor. El log registra carga del modelo de aproximadamente 4.48 s.

Benchmark ejecutado: llama bench -ngl 99 -p 128 -n 64 -r 3 -o json.
- Prefill 128: media 2313.70 tok/s, desviación 92.05.
- Generación 64: media 38.5397 tok/s, desviación 0.0327.
- Fecha del benchmark: 2026-11-04T23:05:45Z.
- Threads automáticos 20; el recurso Molab anuncia 4 CPU. KV f16 y flash auto en bench.
No comparar estas cifras directamente con tiempos end-to-end de Transformers del informe anterior.

El código generado se analizó con AST antes de ejecutarlo, restringiendo builtins a range y ValueError. Pasaron factorial(0)=1, (1)=1, (5)=120, (10)=3628800 y ValueError para -1. Evidencia cruda en evidence/davidau-20261104/.

## Errores y solución

La compilación inicial encontró bibliotecas CUDA sin symlinks, CCCL ausente y mezcla de versiones nvcc/headers. Se preparó un toolkit consistente; la compilación avanzó pero se detuvo al encontrar y validar el binario CUDA oficial de la misma revisión. El runner final usa ese binario y no requiere compilar ni modificar el toolkit del host.
GPT-OSS queda fuera del objetivo por indicación del usuario.

## Reproducción en Molab/Linux CUDA

Desde la raíz del repositorio:
```bash
uv run molab/davidau_gguf_lab.py prepare
uv run molab/davidau_gguf_lab.py test
uv run molab/davidau_gguf_lab.py bench
uv run molab/davidau_gguf_lab.py serve
```
prepare fija revisión del modelo y comprueba revisión del runtime y detección CUDA. El runner coincide byte a byte con el ejecutado en Molab.
serve queda en primer plano en loopback; test termina su propio servidor al finalizar. El servidor no queda activo tras estas pruebas.

## Integración y límites

La API interna /v1/chat/completions fue ejecutada y validada. Conexión desde Termux, Muse u OpenCode externos: pendiente, no se estableció túnel ni endpoint público.
La función oficial marimo pair / Connect your agent conecta un agente al notebook; por sí sola no publica el modelo como proveedor de OpenCode. No se afirma que la integración externa funcione.
No se probaron MTP, visión, herramientas, contexto largo ni comportamiento de rechazos. Las afirmaciones comerciales del autor no fueron validadas.
Notebook: https://molab.marimo.io/notebooks/nb_iVBLiVKuNhwokM43qyfp1H
