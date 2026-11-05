# Molab — pruebas reales, 4 de noviembre de 2026

Notebook: https://molab.marimo.io/notebooks/nb_iVBLiVKuNhwokM43qyfp1H

## Entorno observado

- Sesión autenticada; notebook importado desde feature/molab-llm-lab.
- Compute: 4 CPU, 32 GiB RAM, RTX Pro 6000 (Blackwell).
- CUDA: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GiB de VRAM accesible (95.0 GiB redondeados).
- Python 3.13.11; PyTorch 2.11.0+cu130; CUDA runtime 13.0; Transformers 4.57.6; marimo 0.25.2.
- kernels 0.11.7 y huggingface-hub 0.36.2 tras resolución de dependencias.
- No se contrataron servicios ni se modificó main.

## Inferencia ejecutada en el notebook

| Modelo | Carga | Tokens nuevos | Generación | Tokens/s | Pico VRAM |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B, BF16 | 13.37 s | 96 | 1.964 s | 48.89 | 1.133 GiB |
| Qwen/Qwen3-8B, BF16 | 37.44 s | 96 | 1.688 s | 56.87 | 15.285 GiB |
| Qwen/Qwen3-32B, BF16 | 148.88 s | 96 | 6.474 s | 14.83 | 61.064 GiB |

Medición principal: una ejecución por modelo, batch 1, generación greedy, enable_thinking=False y sincronización CUDA antes y después. El tiempo incluye prefill y generación; no es una medida aislada de decodificación ni un benchmark con repeticiones. La carga incluye descarga e inicialización. Prompt: “Explicá en español qué ventajas tiene ejecutar un LLM localmente.” Las respuestas se vieron en español, truncadas a 96 tokens. El modelo 0.6B y el 8B se probaron mediante celdas corregidas del laboratorio; no se afirma que el archivo gpu_smoke_test.py haya sido ejecutado independientemente.

## Errores encontrados y correcciones

1. Marimo bloqueaba dos celdas por definición global duplicada de start. Se usan temporizadores privados por celda.
2. torch_dtype produjo aviso de obsolescencia; se cambia a dtype en la rama.
3. La instalación sin restricciones de kernels seleccionó 0.18.1 y subió huggingface-hub a 2.4.0, incompatible con Transformers 4.57.6. Se resolvió con kernels>=0.9,<0.12 y huggingface-hub>=0.34,<1.
4. Transformers ya importado conservaba kernels como ausente tras instalarlo. El primer intento de GPT-OSS anunció fallback BF16; se interrumpió. Se reinició el kernel y se verificó kernels=True antes del segundo intento.
5. La cifra de espacio libre reportada por shutil.disk_usage en el montaje remoto fue absurda; no se utiliza para afirmar capacidad de disco.
6. El aviso sobre temperature/top_p/top_k durante greedy no impidió la ejecución de Qwen.

Una segunda inferencia de Qwen3-32B respondió por completo a “Respondé en español en una oración: ¿qué es CUDA?”: 46 tokens, 2.797 s, 16.44 tok/s y pico 61.051 GiB. Texto observado: “CUDA es una plataforma de computación paralela y un entorno de desarrollo creado por NVIDIA para aprovechar el poder de los procesadores gráficos (GPU) en la ejecución de cálculos intensivos.”

Después de la corrección se volvió a pulsar Cargar con Qwen3-32B ya residente: no se descargaron ni se recargaron shards, y la VRAM permaneció aproximadamente en 61.03 GiB. La respuesta de la prueba principal se vio en español, truncada al límite de 96 tokens.

Qwen3-32B: carga completa de 17 shards, 61.024 GiB de VRAM asignada, cuantizado=False. La descarga tardó aproximadamente 138 s; la carga completa, 148.88 s.

## Intento GPT-OSS-120B (cancelado por indicación del usuario)

Segundo intento: pesos descargados (15/15), carga completada en 169.28 s; VRAM asignada 60.79 GiB. Primer intento interrumpido por fallback BF16. La recarga posterior falló por OOM (intento de 508 MiB, 405.56 MiB libres; 94.57 GiB usados por el proceso). No se obtuvo inferencia válida en ese intento. Se sustituyó la carga global por un objeto que conserva una sola instancia y reutiliza cargas repetidas; se reinició el kernel para repetir en limpio.

El usuario indicó que GPT-OSS no le interesa porque ya lo tiene disponible. Se detuvo su prueba y se cambió el objetivo a Qwen/Qwen3-32B. No se atribuye una inferencia exitosa a GPT-OSS.

## Integración oficial

Documentación: https://docs.marimo.io/guides/generate_with_ai/marimo_pair/

Molab ofrece Actions → Connect your agent → OpenCode. Muestra:
- Instalación: npx skills add marimo-team/marimo-pair
- Comando específico de la sesión: opencode --prompt "$(uvx marimo@latest pair prompt --url URL_DEL_ENTORNO --with-token --opencode)"
- Token para pegar de forma local cuando se solicite.

El comando concreto y el token deben copiarse desde el diálogo de la sesión vigente. No se guardan credenciales en GitHub. El procedimiento está documentado para OpenCode; no se ejecutó en el teléfono del usuario. En Termux, uv/Python requieren una instalación compatible (puede usarse Debian/PRoot ya disponible). Muse puede conducir el agente a través de su integración con comandos de Termux si esta funciona; no se verificó una conexión nativa de Muse a Molab.

marimo pair permite editar y ejecutar el notebook, no convierte automáticamente el modelo cargado en un proveedor OpenAI para OpenCode o Muse. No se abrió API pública ni túnel.

## Verificación del código

Se comprobó sintaxis Python de ambos archivos corregidos con py_compile. Una prueba local con un asignador simulado confirmó que dos cargas idénticas reutilizan una instancia y que cambiar de modelo elimina la instancia anterior antes de asignar la nueva; esta prueba no usa GPU. La validación de GPU e inferencia corresponde a las celdas realmente ejecutadas en Molab; no se presenta la comprobación sintáctica como prueba de CUDA.
