# Modelos, perfiles y etiquetas

Esta página explica las palabras que aparecen al elegir un modelo. No cambia el comportamiento de la TPU. El lanzador sigue arrancando Qwen3.8-27B en la TPU si no pedís otra cosa.

```bash
python launch.py models
python launch.py model-info qwen38-27b-gpu
python launch.py serve                      # TPU, receta Qwen de siempre
python launch.py serve --accelerator gpu    # el perfil GPU de Qwen, dos T4
```

Un perfil es un archivo JSON en `models/profiles/`. Describe de dónde sale el modelo y con qué motor se sirve. Agregar otro modelo compatible es agregar un JSON, no un `if` nuevo en Python. Eso no significa que todos los JSON se puedan lanzar: los candidatos están marcados y el lanzador se niega a subirlos.

## Qué es cada cosa

**Modelo base.** Los pesos publicados por quien entrenó el modelo (por ejemplo `Qwen/Qwen3.8-27B`). Ahí están el tamaño, la licencia y el comportamiento de partida, incluidos los rechazos que el equipo de origen dejó.

**Fine-tune.** Un entrenamiento adicional encima del base. Cambia pesos con datos nuevos. Puede especializar el modelo en código, en un idioma o en un estilo. No es lo mismo que borrar rechazos, y no garantiza que el modelo deje de negarse.

**Abliteración.** Un edit de pesos, no un entrenamiento. La idea publicada por quienes la hacen es debilitar las direcciones internas asociadas al rechazo. El modelo no aprende hechos nuevos. El propio método no promete cero rechazos: un publicador puede medir "menos rechazos en este conjunto" y seguir viendo algunos.

**GGUF.** Un archivo de pesos para llama.cpp (y programas compatibles). No es otro modelo: es un empaquetado. Trae, cuando el autor lo puso, la plantilla de chat y el tokenizador. Sirve en GPU con llama.cpp. No es el formato bf16 que usa la receta TPU de Qwen.

**Cuantización.** Guardar esos pesos con menos bits (`Q4_K_M`, `Q5_K_M`, `Q8_0`, …). El archivo pesa menos y cabe en menos VRAM. También pierde precisión. `Q4_K_M` es el compromiso que ya usa la ruta GPU de Qwen. Un `Q8_0` de 27B no cabe con holgura en dos T4 de 16 GB. El nombre de la cuantización no dice nada sobre filtros ni rechazos.

**Backend.** El programa que hace la inferencia.

- `vllm-tpu`: la receta Qwen de producción. Sigue en vllm-tpu 0.28.0.
- `jax`: el motor propio de GLM-5.3-Flash. No es vLLM.
- `llama.cpp`: el motor GPU de esta rama. Un solo programa sirve cualquier perfil GGUF marcado como servible.
- `vllm-gpu`: previsto en el esquema, sin motor todavía.

**Acelerador.** Dónde corre: `tpu` (v5e-8) o `gpu` (en Kaggle, dos Tesla T4). No se pueden mezclar a gusto. llama.cpp no arranca en la TPU con este lanzador, y el perfil TPU de Qwen no se manda a la GPU.

## Etiquetas: uncensored, abliterated, unfiltered

Esas palabras son metadatos del publicador. El lanzador las muestra y no las usa para elegir código, flags ni una "ruta sin filtros".

No significan:

- que el modelo responda cualquier cosa
- que no queden rechazos
- que sea legal o seguro usar la salida
- que el archivo ya haya funcionado en una TPU o en las dos T4 de Kaggle

Un GGUF puede traer además una plantilla de chat que agrega instrucciones propias. Eso es un prompt metido en el archivo, distinto de un edit de pesos. Puede cambiar lo que ves aunque los pesos sean parecidos al base, y puede pisar o combinarse con el mensaje de sistema que mande tu app.

Si una ficha dice "90 % menos rechazos", ese número es del publicador, en su conjunto de prueba. Este repositorio no lo reprodujo.

## Qué sí está servible hoy

| Pedido | Qué pasa |
|---|---|
| `serve` o `serve --model qwen38-27b` | La receta TPU de siempre |
| `serve --model glm53-flash` | La receta TPU de GLM |
| `serve --accelerator gpu` o `--model qwen38-27b-gpu --accelerator gpu` | Qwen UD-Q4_K_M con llama.cpp, localhost, dos T4 |
| un id marcado candidato | `model-info` lo explica; `serve` se niega |

`trust_remote_code` no se enciende solo. Un perfil que lo necesite tiene que decirlo y traer el riesgo por escrito. Ningún perfil de esta rama lo pide. Un GGUF no lo necesita.

Los candidatos concretos, con repo, tamaño y licencia tal como figuran en Hugging Face, están en [GPU_CANDIDATES.md](GPU_CANDIDATES.md).
