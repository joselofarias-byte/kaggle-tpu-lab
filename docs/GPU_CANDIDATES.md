# Candidatos GGUF para dos T4 (sin probar)

El objetivo externo principal es el candidato 1 (Mythos), más abajo. Sigue sin ser el default y sin ser servible. Los cuatro repos que ya estaban documentados se conservan después de esa sección.

**Ninguno se descargó ni se cargó.** El lanzador los lista y se niega a servirlos (`launchable: false`). Los SHA-256 son oids LFS de `paths-info`, no un hash recalculado en esta máquina. Mythos se releyó el 2026-09-25; el resto, el 2026-09-24.

El binario fijado sigue siendo el de la ruta Qwen: `ai-dock/llama.cpp-cuda` v0.4.0, SHA-256 `7a229ac0…`. Que ese binario abra Qwen3.8-27B UD-Q4 no demuestra que abra estos otros GGUF (plantilla, cabeza MTP, arquitectura `qwen3` frente a `qwen35`).

Hardware que este repo ya asume, sin una sesión nueva: dos Tesla T4 (16 GB cada una, 32 GB sumados, sin NVLink) y al menos 20 GiB libres en el scratch antes de bajar el archivo. Fuentes públicas describen la sesión T4 x2 de Kaggle con unos 32 GB de RAM de host. No se volvió a medir aquí.

"Cabe en VRAM" significa que el archivo es claramente menor que 32 GB y, partido por capas, menor que ~16 GB por GPU, dejando sitio para KV y buffers. No es una medición.

La línea de base sigue siendo `qwen38-27b-gpu`: Unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16464440224 bytes, contexto de lanzamiento 32768, ya marcado servible en el catálogo y todavía sin una sesión nueva de esta rama. Mythos no lo reemplaza.

## Candidato 1 — Qwen3.8 Mythos Agentic (objetivo externo)

| | |
|---|---|
| Perfiles | `qwen38-mythos-27b-q4ks`, `qwen38-mythos-27b-q4km` (primera prueba), `qwen38-mythos-27b-q5ks` |
| Nombres | Qwen3.8 27B Mythos Agentic — Q4_K_S / Q4_K_M / Q5_K_S |
| Repo GGUF | `mradermacher/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-GGUF` |
| Revisión | `01a19fb59c4130c1ae51b614eccc50dd62de4b02` (punta el 2026-09-25; no hay URL `latest`) |
| Padre | `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` @ `528121d7b0b85885a658dfe67e3643b8a4f9337e` |
| Cadena | `OBLITERATUS/Qwen3.8-27B-OBLITERATED` y `Qwen/Qwen3.8-27B` |
| Licencia en las fichas | `apache-2.0` |
| Arquitectura | `qwen35` en el GGUF; `qwen3_5` / `Qwen3_5ForConditionalGeneration` en el config del autor |
| Contexto | 262144 de arquitectura. **Lanzamiento 8192** en los tres perfiles. El Qwen de serie sigue en 32768 |
| Plantilla | embebida en el GGUF. No se copió el Jinja. `trust_remote_code` false |
| i1 | `…-Mythos-Class-Agentic-i1-GGUF` @ `eee7c1a5b34280252b44b1b16996d8edd2f76149`. Documentado, sin perfil y sin ser la primera prueba |
| Checklist | [CANDIDATE1_MYTHOS.md](CANDIDATE1_MYTHOS.md) |

| Perfil | Archivo | Bytes | oid LFS (no rehasheado) | Dos T4, solo por tamaño |
|---|---|---:|---|---|
| `qwen38-mythos-27b-q4ks` | `Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic.Q4_K_S.gguf` | 15825300672 | `0f146e0c6b1ab09f48f3f9cca8a423362a8573d1cd747eaaccc22c7f6dd07f50` | Plausible (~14.7 GiB). Orden práctico de ensayo, no de calidad |
| `qwen38-mythos-27b-q4km` | `….Q4_K_M.gguf` | 16810716352 | `3cc24a3e431930401b446d9abb52d4e1fa4add4ec19df19b5b5b0c1dbc22da4b` | Plausible, misma clase que el UD-Q4 de Unsloth (~15.7 GiB). **Primera prueba** |
| `qwen38-mythos-27b-q5ks` | `….Q5_K_S.gguf` | 18971684032 | `8145d2b7cce80ef444d2a3ca9b463043fe04cc9daddef6d0e12383e5fe506049` | Marginal (~17.7 GiB). Puede pedir offload |

Los tres dicen `UNVERIFIED_UNTIL_FIRST_DOWNLOAD_HASH`. El repo GGUF no trae `SHA256SUMS`. Split de capas `1,1`, `n_gpu_layers` all, `mtp_tokens` 0, host `127.0.0.1`. Estimación burda de VRAM total (pesos + overhead, sin medir): Q4_K_S ~18–24 GB, Q4_K_M ~19–26 GB, Q5_K_S ~22–28 GB. KV de un híbrido no se midió. 128k y 256k no entran en el plan.

### Etapas, solo después de que el Qwen GPU de serie responda

- **A.** Contexto 8192. Generación corta. Las dos T4 en uso, sin caída rara a CPU. `GET /v1/models` en localhost.
- **B.** 16384. Estabilidad, VRAM por GPU, tok/s.
- **C.** 32768. Igual, más estabilidad de contexto largo.
- Recién ahí se habla de 64k o 128k. El perfil no sube el tope solo.

### Batería de herramientas (no afirmar "agentic" antes)

1. Una función simple.
2. Varias herramientas.
3. Argumentos mal formados.
4. Continuar después del resultado.
5. Un bucle de varios pasos.
6. La misma herramienta repetida.
7. Conversación larga con herramientas.

La ficha del autor, en SGLang, pasa `--tool-call-parser qwen3_coder` y `--reasoning-parser qwen3`, y enciende `trust_remote_code`. Eso no se copia. El 0 % de rechazos es su batería de 30 prompts.

### Cuándo se rechaza como perfil soportado

Si el binario fijado no carga el GGUF, si el SHA-256 del disco no coincide con el oid, si no usa las dos T4 sin un offload malo, si falla la batería de herramientas, si la plantilla rompe clientes OpenAI, si reaparecen los bugs de kernel GGUF que el autor quiso evitar, o si hay OOM en la etapa A a 8192.

`alignment_style: obliterated` y las etiquetas `uncensored`, `agentic`, `tool-calling`, `obliterated` no cambian el argv.

## 1. Blackfrost Qwen3.8-27B abliterated

| | |
|---|---|
| Perfil | `blackfrost-qwen38-27b-abliterated-gpu` |
| Repo GGUF | `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF` |
| Revisión | `994bb4e69663ec880a4d9a61604e6debc3a49b9a` |
| Base | `Qwen/Qwen3.8-27B` (la ficha dice Apache-2.0) |
| Padre de pesos | `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16` |
| Parámetros | clase 27B, según el nombre y la ficha |
| Arquitectura GGUF | `qwen35` (id de llama.cpp; el producto se llama Qwen3.8) |
| Contexto | 262144 en el metadato GGUF. El perfil propone lanzar 32768, igual que el Qwen ya servido |
| Plantilla | embebida. Agrega un prompt de sistema del publicador. No se copió a este repo |
| Etiquetas | `abliterated`. El publicador dice que tocó la superficie de rechazo en los pesos, sin fine-tune ni pruning, y que lo probó en una B200 |

Cuantizaciones consultadas (bytes reales del LFS, no el redondeo de la ficha):

| Archivo | Bytes | ¿Dos T4, solo por tamaño? |
|---|---:|---|
| `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` | 16810716384 | Plausible. Misma clase que el UD-Q4 de Unsloth (16464440224). SHA-256 `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509`, igual al `SHA256SUMS.txt` del repo |
| `Q4_K_S` | 15825300704 | También plausible, sin prueba |
| `Q5_K_M` | 19535703264 | Justo: el archivo ya son ~18.2 GiB |
| `Q6_K` | 22431001824 | Poco realista con KV y buffers |
| `Q8_0` | 29047086304 | No cabe con holgura en 32 GB |

El Q4_K_M es el único archivo nombrado en el perfil. Compatibilidad pendiente: cabeza MTP embebida (el motor la deja en 0), proyector de visión aparte (el motor no lo pasa), plantilla que no es la de Unsloth.

¿Menos restringido de verdad? El publicador lo afirma como edit de pesos y, además, como prompt. Sin una corrida, no se puede separar marketing, prompt y pesos. No es una promesa de cero rechazos.

## 2. bartowski / huihui Qwen3-14B abliterated

| | |
|---|---|
| Perfil | `bartowski-qwen3-14b-abliterated-gpu` |
| Repo GGUF | `bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF` |
| Revisión | `623c0f3fc42a4699d4583fb16e022942c003d1b7` |
| Base | `Qwen/Qwen3-14B` |
| Padre | `huihui-ai/Qwen3-14B-abliterated` |
| Parámetros | 14768307200 en el safetensors del padre |
| Licencia en la ficha | `apache-2.0`, con enlace a la licencia de `Qwen/Qwen3-14B` |
| Arquitectura GGUF | `qwen3` |
| Contexto en el metadato | 40960. El perfil no pasa de 32768 |
| Plantilla | la de Qwen3 que trae el GGUF (herramientas en XML). No se reescribió |
| Cuantizador | bartowski, imatrix, llama.cpp b5284 según la ficha |

| Archivo | Bytes | Nota |
|---|---:|---|
| `huihui-ai_Qwen3-14B-abliterated-Q4_K_M.gguf` | 9001749568 | ~8.38 GiB. SHA-256 `d76889059a3bfab30bc565012a0184827ff2bdc10197f6babc24541b98451dbe` |
| `Q5_K_M` | 10514566208 | Sigue holgado de peso |
| `Q6_K` | 12121933888 | Sigue holgado de peso |
| `Q8_0` | 15698530368 | Parecido al Q4 de 27B; posible y sin prueba |

Dos T4 alcanzan de sobra para el Q4 si el binario lo carga. Una sola T4 también podría, por tamaño. El motor de Kaggle igual exige dos T4: no se relajó esa condición. Qwen3-14B es atención densa: el KV crece más que en el híbrido 3.8. Por eso el contexto de lanzamiento propuesto queda en 32768.

¿Menos restringido? huihui llama al padre abliteración (no un fine-tune de conocimiento) y publica un aviso de que el filtrado está reducido y de que no hay garantía de seguridad por defecto. La ficha no dice cero rechazos. La etiqueta `base_model:finetune` de Hugging Face no demuestra que haya habido un entrenamiento.

## 3. RootMonsteR Qwen3-14B abliterated (Heretic)

| | |
|---|---|
| Perfil | `rootmonster-qwen3-14b-abliterated-gpu` |
| Repo GGUF | `RootMonsteR/Qwen3-14B-Abliterated-GGUF` |
| Revisión | `aad7bb258333fa83991acef39e31a97677eb711b` |
| Base | `Qwen/Qwen3-14B` commit `40c0698` según la ficha fuente |
| Padre | `RootMonsteR/Qwen3-14B-Abliterated` |
| Licencia en la ficha | `apache-2.0`, enlace a `Qwen/Qwen3-14B` |
| Método que declara el autor | Heretic v1.3.0, trial 33 de 200, ablación direccional. Pesos tocados: `attn.o_proj` y `mlp.down_proj`. No es un dataset de fine-tune |
| Arquitectura GGUF | `qwen3` |
| Plantilla | Qwen3 embebida, con tool calling |

El autor reporta, y este repo no lo repitió: KL 0.0333 contra el base en `mlabonne/harmless_alpaca`, y 10/100 rechazos contra 99/100 del base en 100 prompts de `mlabonne/harmful_behaviors`. Diez de cien sigue siendo rechazo. La ficha fuente lo dice así.

Conflicto de contexto, sin resolver: la ficha fuente dice 32768 nativo y 131072 con YaRN; el metadato GGUF de este repo dice `context_length` 40960. El perfil usa 32768 como tope de lanzamiento y 40960 como tope de arquitectura declarado por el GGUF. No se usa 131072.

| Archivo | Bytes | SHA-256 (LFS = `SHA256SUMS`) |
|---|---:|---|
| `qwen3-14b-abliterated-Q4_K_M.gguf` | 9001753792 | `c74b5bcfcf7d4c9386075cde43fd7a4580c602b46b90ad01d7fc58b696748bbb` |
| `qwen3-14b-abliterated-Q5_K_M.gguf` | 10514570432 | `dd502f4f28ec24c24d014ebc67caf39c7f2fae5e537eb818bdb54a25fe7b3703` |

Los dos caben en peso en dos T4. El Q5 es el que el autor recomienda para herramientas. Ninguno está probado con el binario fijado. Cuantización del publicador, sin imatrix, según su ficha.

## 4. mradermacher / huihui Qwen3.5-27B abliterated

| | |
|---|---|
| Perfil | `mradermacher-qwen35-27b-abliterated-gpu` |
| Repo GGUF | `mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF` |
| Revisión | `dcc4a772054979c455894098c9c0074b361a350b` |
| Base | `Qwen/Qwen3.5-27B` |
| Padre | `huihui-ai/Huihui-Qwen3.5-27B-abliterated` |
| Licencia en la ficha | `apache-2.0`, enlace a la licencia de `Qwen/Qwen3.5-27B` |
| Arquitectura GGUF | `qwen35` |
| Contexto metadato | 262144. Tope de lanzamiento propuesto: 32768 |
| Plantilla | Qwen3.5 embebida, con tokens de imagen y video. El motor no descarga el `mmproj` |

No es Qwen3.8. Es la generación 3.5, cuantizada por mradermacher. El repo no trae `SHA256SUMS`; el digest es solo el oid LFS.

| Archivo | Bytes | SHA-256 LFS | Tamaño frente a dos T4 |
|---|---:|---|---|
| `Huihui-Qwen3.5-27B-abliterated.Q4_K_M.gguf` | 16540272704 | `5a2321c22682c7109907d6a4c0293b9c6349bc239d4e7b9aaa1e14b98add4484` | Plausible, misma clase que el Q4 de Qwen3.8 |
| `Q3_K_M` | 13289646144 | `93a545f80dd7793cc31e7e69d741f850fd5ace6c4a2f2175ed1a23b817cbbe02` | Más holgado, más pérdida |
| `Q5_K_M` | 19399608384 | `fb813be07190dddb76762248451903213b680cbb105841935810954a5eac6e0a` | Justo |
| `Q6_K` | 22082529344 | `553b1157cecc55ff7b9a46201bc20cb6d57168c45f3e1d88d89c3d562543cacb` | Poco realista |
| `Q8_0` | 28595763264 | `3e1a8f8332f62818adaeff1c77c3bb83c8677aeb1da816fcb7051a20e4728007` | No cabe con holgura |

¿Menos restringido? La ficha del cuantizador hereda las etiquetas `abliterated` y `uncensored` del padre. No publica una evaluación de rechazos de este archivo. Tratarlo como "sin censura" sería marketing.

## Qué haría falta para marcar uno como servible

1. Sesión real de Kaggle con GPU T4 x2, sin cambiar el pin de cloudflared ni el bind a `127.0.0.1`.
2. Descargar solo el archivo elegido y comprobar que el SHA-256 de disco coincide con el oid de arriba.
3. Ver que `llama-server` de este binario carga el GGUF, pasa `GET /v1/models` y un chat corto.
4. Anotar VRAM por GPU, si el KV a 32768 entra, y si la plantilla embebida pisa el system prompt.
5. Recién ahí poner `launchable` en true y `experimental` en false. Hasta ese momento el perfil es documentación.
