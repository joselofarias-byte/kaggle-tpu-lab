# Segunda cuenta: DavidAU Q4_K_M MTP — 5/11/2026

Ejecutado mediante el navegador autenticado de Work en el notebook:
https://molab.marimo.io/notebooks/nb_X6tbqeajU6M39rYq5N1fwn

## Resultados medidos
GPU: RTX PRO 6000 Blackwell Server Edition, 97887 MiB. Driver 595.71.05.
llama.cpp: cd959bae99ff4bed22bef2604f45241c09bbb8c8, GGML_CUDA=ON, arquitectura 120.
Los tres binarios llama-cli, llama-server y llama-bench se compilaron y ejecutaron.
GGUF: 18498573856 bytes; descarga y hash: 79.0668 segundos.
SHA-256: bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c

| Prueba | Sin MTP | MTP |
|---|---:|---:|
| Generación, promedio de 3 chats de 85 tokens | 53.48 tok/s | 78.11 tok/s |
| Carga del servidor | 3.527 s | 4.017 s |
| Pico de memoria GPU total observado | 17981 MiB | 19037 MiB |

Chats medidos con contexto 8192. El aumento medio con MTP fue 46.0%.
llama-bench, tres repeticiones: pp512 3745.55 tok/s; tg128 56.72 tok/s.
Estos benchmarks no se confunden con la latencia HTTP ni con OpenCode.

Herramientas: llamada estructurada sumar(17,25), ejecutada por dispatcher Python con allowlist; resultado 42 y respuesta final en español en ambos modos. No se atribuye ejecución de herramientas al modelo por sí solo.
API loopback autenticada: sin clave 401, clave incorrecta 401, SSE API_MTP_OK y DONE. Chat HTTP 0.3897 s.
OpenCode 1.19.5: una llamada read real completada y marcador JOLUFASOFT_MTP_OPENCODE_OK; exit 0; 5.7887 s.
Pruebas MTP terminadas sin errores. El ejecutor de sesión devolvió passed=true y server_stop.stopped=true.
Comprobación posterior del servicio: running=false. GPU memory.used: 0 MiB.

## Reproducción y ahorro
Notebook limpio añadido a la segunda cuenta como espejo de GitHub, sin arrancar otro sandbox:
https://molab.marimo.io/github/joselofarias-byte/kaggle-tpu-lab/blob/feature/molab-authenticated-api/molab/davidau_mtp_lab.py
marimo check terminó con exit 0, sin errores ni advertencias.
El ejecutor davidau_mtp_session.py limita tiempos y detiene llama-server en finally.
Eso no detiene la asignación de Molab: al finalizar se usa Shutdown notebook y se verifica Running 0 por separado.
No hay Tailscale ni exposición externa en esta ejecución.
No hay copia persistente de GGUF ni de los binarios: una recreación exige volver a descargarlos y compilar. No se promete recuperación instantánea.

## Evidencia
results.json contiene 22 archivos exportados mediante allowlist, con comprobación de exclusión de la clave privada antes de exportar.
Los resultados son de esta sesión y no reutilizan las mediciones del 4/11.
