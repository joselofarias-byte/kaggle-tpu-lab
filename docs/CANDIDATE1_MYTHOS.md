# Candidato 1 — checklist

`medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic`, servido en papel desde el GGUF de mradermacher. Revisión de metadatos el 2026-09-25 con la API de Hugging Face (`models` + `paths-info` + `config.json` + `generation_config.json` + README). No se bajó ningún peso. La punta no se movió respecto del paquete de investigación del mismo día.

Los tres perfiles están en `models/profiles/qwen38-mythos-27b-q4k{s,m}.json` y `qwen38-mythos-27b-q5ks.json`. `launchable` es false. El default del lanzador sigue siendo Qwen.

## Respuestas

1. **Qué es.** Un Qwen3.8-27B con plantilla de agente (Mythos) encima de la cadena OBLITERATUS. El objetivo externo de esta rama es el GGUF de la comunidad, no el safetensors del autor.
2. **Repo del autor y pin.** `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` @ `528121d7b0b85885a658dfe67e3643b8a4f9337e`. Sigue siendo la punta.
3. **Licencia.** La ficha declara `apache-2.0`. No es un dictamen legal.
4. **Cadena.** `base_model` de la ficha: `OBLITERATUS/Qwen3.8-27B-OBLITERATED` y `Qwen/Qwen3.8-27B`. El GGUF apunta a medismera como `base_model`.
5. **Arquitectura.** `config.json`: `model_type` `qwen3_5`, `Qwen3_5ForConditionalGeneration`. 64 capas, `full_attention_interval` 4, `layer_types` alterna atención lineal y atención plena. Hay claves MTP (`mtp_num_hidden_layers`). El metadato GGUF de mradermacher dice `architecture` `qwen35`.
6. **Parámetros.** Safetensors BF16 total `27781427952` (~27.78B). La ficha habla de un audit de 27,360,914,016 parámetros de lenguaje y de un tamaño de tarjeta ~28B. Se copian como dichos del autor.
7. **Contexto.** `max_position_embeddings` 262144. El metadato GGUF también dice 262144. La ficha de despliegue usa 131072 en un ejemplo SGLang. En dos T4 el perfil lanza a **8192**.
8. **Pesos del autor.** Safetensors en `main`. La ficha nombra ramas `fp8` y `awq`. No son el archivo de esta prueba.
9. **Tokenizador.** En el repo del autor (`tokenizer.json`, `tokenizer_config.json`). `generation_config.json` usa ids de bos/pad `248044` y eos `248046`, coherente con un vocabulario de ese orden. El perfil GGUF dice `tokenizer_source: embedded-in-gguf`.
10. **Plantilla.** El README habla de una plantilla Mythos de ~9.4 KB con razonamiento. El perfil no copia el Jinja. `chat_template` queda `embedded-in-gguf`.
11. **Herramientas.** El ejemplo SGLang de la ficha usa `--tool-call-parser qwen3_coder` y `--reasoning-parser qwen3`. Eso es XML estilo Hermes (`<tool_call><function=…>`), no una medición en llama.cpp. No apareció en el tramo leído un `--tool-call-parser hermes` de vLLM. No se afirma que el tool calling funcione.
12. **Generación.** `generation_config.json` en el pin: temperature 0.65, top_k 20, top_p 0.95, repetition_penalty 1.15, presence_penalty 0.3, max_new_tokens 16384. El motor GPU actual no aplica ese archivo.
13. **trust_remote_code.** Las guías del autor lo encienden (SGLang y `AutoModelForImageTextToText`). Los tres perfiles lo dejan en **false**. Un GGUF no lo necesita.
14. **Rechazos.** La ficha dice 0.00 % en 30 prompts propios (inglés y árabe) y 30/30 de cumplimiento. Es un dicho del autor. No se reprodujo.
15. **GGUF del autor.** El README dice que el repo se queda en safetensors para evitar bugs de kernel GGUF. No hay que bajarse un GGUF de medismera. El archivo a considerar es el de mradermacher.
16. **Repo GGUF y pin.** `mradermacher/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-GGUF` @ `01a19fb59c4130c1ae51b614eccc50dd62de4b02`. Punta confirmada el 2026-09-25. Licencia de la ficha: apache-2.0. Metadato GGUF: `qwen35`, contexto 262144.
17. **Variante i1.** Existe `mradermacher/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-i1-GGUF` @ `eee7c1a5b34280252b44b1b16996d8edd2f76149`, con un `i1-Q4_K_M`. No es el primer archivo. No hay perfil para esa variante.
18. **No confundir.** Hay GGUF de OBLITERATUS sin Mythos. Estos perfiles nombran el archivo Mythos completo.
19. **Tabla de archivos** (oid LFS = campo `sha256`; **UNVERIFIED_UNTIL_FIRST_DOWNLOAD_HASH**; no hay `SHA256SUMS` en la lista de archivos de la API):

| Perfil | Quant | Archivo | Bytes | oid LFS |
|---|---|---|---:|---|
| `qwen38-mythos-27b-q4ks` | Q4_K_S | `Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic.Q4_K_S.gguf` | 15825300672 | `0f146e0c6b1ab09f48f3f9cca8a423362a8573d1cd747eaaccc22c7f6dd07f50` |
| `qwen38-mythos-27b-q4km` | Q4_K_M | `…Q4_K_M.gguf` | 16810716352 | `3cc24a3e431930401b446d9abb52d4e1fa4add4ec19df19b5b5b0c1dbc22da4b` |
| `qwen38-mythos-27b-q5ks` | Q5_K_S | `…Q5_K_S.gguf` | 18971684032 | `8145d2b7cce80ef444d2a3ca9b463043fe04cc9daddef6d0e12383e5fe506049` |

20. **Por qué importa, y qué no.** Es el primer objetivo externo serio de la misma clase ~27B Q4 que el Unsloth que ya describe el perfil GPU, con plantilla de herramientas distinta. Sirve para ver si un JSON nuevo entra en el catálogo sin un `if` de Python. No demuestra que corra, que llame herramientas ni que deje de rechazar. `alignment_style: obliterated` y las etiquetas no entran en el argv.

## Comandos

```bash
python launch.py models
python launch.py model-info qwen38-mythos-27b-q4km
python launch.py serve --model qwen38-mythos-27b-q4km --accelerator gpu
```

El tercero debe negarse mientras `launchable` sea false. `python launch.py serve` sin argumentos sigue siendo Qwen en TPU.
