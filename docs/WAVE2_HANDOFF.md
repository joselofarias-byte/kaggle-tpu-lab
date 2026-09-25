# Wave 2 handoff

## Objective

Turn the launcher from a Qwen-only GPU script into a model-profile launcher. A new compatible GGUF should be a JSON file, not a new Python branch. Labels such as uncensored, abliterated, and unfiltered are metadata. They do not select a different engine and they are not a promise of zero refusals.

Production TPU serving is unchanged: vllm-tpu stays 0.28.0, the Qwen and GLM kernels are the same recipes, and the experimental vllm-tpu bump (PR #4, branch `cursor/experimental-vllm-029-3ac3`) is not in this branch. No Kaggle TPU or GPU session was started. No weights were downloaded.

## Architecture chosen

Data in `models/profiles/*.json`, validation in `models/profile.py`, one shared GPU engine in `models/engines/llama_cpp_gpu.py`.

A profile has id, display name, backend, accelerator, source repo and revision, GGUF filename, sha256, architecture, quantization, launch context and architecture context, VRAM/RAM hints, GPU count and type (or TPU chips), `tpu_compatible`, launch args, served model name, chat template / tokenizer source, health check, notes, warnings, experimental, launchable, and optional tags. `trust_remote_code` defaults off and is invalid unless the profile also carries `trust_remote_code_risk`.

Allowed pairs: `llama.cpp`+`gpu`, `vllm-gpu`+`gpu` (described, not implemented), `vllm-tpu`+`tpu`, `jax`+`tpu`. Anything else fails validation. A launchable llama.cpp profile must be two Tesla T4s, bind `127.0.0.1`, pin a lowercase SHA-256 and a byte size, and use `GET /v1/models`. `vllm-tpu` profiles must say version 0.28.0. The string for the experimental runtime is rejected if a profile tries to set it.

`launch.py serve` resolves a profile before it talks to Kaggle.

- TPU ids and the old names `qwen38-27b` and `glm53-flash` map back to the existing kernel recipes. The pushed TPU source is still `serve_qwen38.py` / `serve_glm53.py`.
- GPU uses the shared engine. The launcher embeds the validated profile JSON and the session overrides (`api_key`, ntfy topic, keepalive, clamped context). cloudflared and the llama.cpp tarball stay as constants in the engine, so a profile cannot retarget them.
- `python launch.py models` and `model-info <id>` only read the catalog.
- Candidates are listed and `serve` refuses them. There is no flag that forces an untested profile.

The old file `qwen38-27b/gpu/serve_qwen38_gpu.py` still runs standalone for Qwen. Its pins match `qwen38-27b-gpu.json`. `launch.py serve --accelerator gpu` no longer pushes that file; it pushes the shared engine with the same pins, the same slug `qwen38-t4x2-serve`, localhost, and context cap 32768. `--max-model-len` still defaults to 262144 and is clamped to the profile cap, which is what Wave 1 did with `min(requested, 32768)`.

GPU events keep `event_version` 1, Spanish `message_es`, and `api_key` only on `ready`. They add optional `model_id`, `display_name`, `backend`, and `accelerator`. The PHASE log redacts the key. Tar extraction rejects `..`, absolute paths, symlinks, and hardlinks. Android copies the optional fields onto the live endpoint and otherwise leaves the UI alone. TPU kernels were not edited, so their event bytes stay as in Wave 1.

## Files added

- `models/__init__.py`
- `models/profile.py`
- `models/gpu_launch.py`
- `models/engines/__init__.py`
- `models/engines/llama_cpp_gpu.py`
- `models/profiles/qwen38-27b-gpu.json`
- `models/profiles/qwen38-27b-tpu.json`
- `models/profiles/glm53-flash-tpu.json`
- `models/profiles/blackfrost-qwen38-27b-abliterated-gpu.json`
- `models/profiles/bartowski-qwen3-14b-abliterated-gpu.json`
- `models/profiles/rootmonster-qwen3-14b-abliterated-gpu.json`
- `models/profiles/mradermacher-qwen35-27b-abliterated-gpu.json`
- `tests/test_wave2_profiles.py`
- `docs/MODELOS_ES.md`
- `docs/GPU_CANDIDATES.md`
- `docs/WAVE2_HANDOFF.md`

## Files changed

- `launch.py` — `models`, `model-info`, profile selection, shared GPU kernel
- `qwen38-27b/gpu/serve_qwen38_gpu.py` — symlink/hardlink rejection, note that the launcher uses the profile engine
- `mobile-overlay/files/src/services/types.ts` — optional identity fields
- `mobile-overlay/files/src/services/racer.ts` — copy those fields when an event has them
- `mobile-overlay/files/src/services/ntfy.ts` — notification prefers `display_name` when present
- `mobile-overlay/files/src/services/ntfy.test.ts` — identity fields survive parsing
- `docs/ANDROID_ES.md` — documents the optional fields, version stays 1
- `README.md` — points at the new commands
- `.github/workflows/python-syntax.yml` — compiles the new modules and runs the Wave 2 tests

Not changed: `qwen38-27b/kernel/serve_qwen38.py`, `glm53-flash/kernel/serve_glm53.py`, notebooks, the MTP 0.28 patch.

## Branch, commit, PR

- Base: `cursor/fork-audit-wave1-3ac3` (PR #3, head `75d25c8fbfa36e29c60800963f1281b33837cc2e`). That branch was not modified.
- Branch: `cursor/model-profiles-wave2`
- Implementation commit: `26e340460efa7dff0d35cb369824089d57745f42`
- PR: draft https://github.com/joselofarias-byte/kaggle-tpu-lab/pull/5 stacked on the Wave 1 branch, not on `main`.
- PR #4 (`cursor/experimental-vllm-029-3ac3`) was not touched.

## Tests and results

No Kaggle accelerator.

- `python3 -m unittest tests.test_wave1_hardening tests.test_wave2_profiles` — 37 tests, OK (15 Wave 1 + 22 Wave 2).
- `python3 tools/sync_notebook.py --check` — both notebooks in sync.
- `python3 -m py_compile` on `launch.py`, the new model modules, both TPU kernels, the Qwen GPU script, and the notebook tools — OK.
- `python3 launch.py models` and `model-info qwen38-27b` — the alias prints both the GPU and TPU ids instead of guessing.
- Mobile unit tests: pinned `afterexam/kaggle-tpu-lab-mobile` @ `253a8dd`, overlay copied on top, `npm ci` and `npm test` (vitest). 8 files, 44 tests, passed. The new case checks that `model_id`, `display_name`, `backend`, and `accelerator` survive `pollOnce` next to `event_version` and `message_es`.
- Android APK was not built here. No Android SDK on the machine. The unit tests are the part this change affects.

Wave 2 tests cover schema and missing fields, bad SHA-256, unsupported backend/accelerator pairs, context cap versus architecture context, the Wave 1 clamp to 32768, GPU count and type, SHA-256 required only when launchable, `trust_remote_code` without a written risk, rejection of a vllm-tpu profile that is not 0.28.0, Qwen alias selection, candidate refusal, tags not changing the llama-server argv, CLI list/info, TPU recipe mapping, pin equality with `serve_qwen38_gpu.py`, rendered kernel host and metadata, event identity plus key redaction, tar slip and link rejection, and absence of the experimental runtime tokens from profiles, the GPU engine, and `launch.py`.

## Candidate models researched

Documented in `docs/GPU_CANDIDATES.md`. Digests are Hugging Face LFS oids from `paths-info` on 2026-09-24, not a local re-hash. None were downloaded. None are launchable.

1. `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF` @ `994bb4e6…`. Base `Qwen/Qwen3.8-27B`. GGUF architecture id `qwen35`, context metadata 262144. Q4_K_M is 16810716384 bytes, SHA-256 `5d53637a…` (matches the repo's `SHA256SUMS.txt`). Card says apache-2.0. Publisher describes a weight-level refusal edit plus an embedded system prompt. Tested by them on a B200, not on a T4.
2. `bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF` @ `623c0f3f…`. Quant of `huihui-ai/Qwen3-14B-abliterated`, base `Qwen/Qwen3-14B`, 14768307200 parameters on the parent safetensors. Card says apache-2.0. Q4_K_M is 9001749568 bytes. Architecture `qwen3`, GGUF context 40960.
3. `RootMonsteR/Qwen3-14B-Abliterated-GGUF` @ `aad7bb25…`. Heretic v1.3.0 directional ablation, trial 33, of `Qwen/Qwen3-14B`. Publisher reports KL 0.0333 and 10/100 refusals versus 99/100 on their set. Not reproduced here. Q4_K_M is 9001753792 bytes; SHA-256 matches `SHA256SUMS`. Source card says 32768 native and 131072 with YaRN; this GGUF's metadata says 40960. The profile keeps the launch cap at 32768.
4. `mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF` @ `dcc4a772…`. Static quant of huihui's Qwen3.5-27B abliteration, not Qwen3.8. Card says apache-2.0. Q4_K_M is 16540272704 bytes. No `SHA256SUMS` in the repo; digest is the LFS oid only. No refusal eval on the quantizer's card. `mmproj` files exist and this engine does not pass them.

## Estimated hardware

Kaggle dual T4, as this repo already treats it: two 16 GB Tesla T4s (32 GB combined, no NVLink) and at least 20 GiB free scratch. Public writeups also cite about 32 GB host RAM. Not re-measured.

| Quant | Approx file | Dual T4 on size alone |
|---|---|---|
| Qwen3.8 UD-Q4 (already the served profile) | 16464440224 bytes (~15.3 GiB) | The existing recipe's target. Still not re-run here |
| 27B-class Q4_K_M (Blackfrost, mradermacher Qwen3.5) | ~15.4–15.7 GiB | Plausible, same class, untested |
| 27B Q5_K_M | ~18 GiB | Tight |
| 27B Q6_K / Q8_0 | ~21–27 GiB | Not a realistic dual-T4 choice |
| Qwen3-14B Q4_K_M / Q5_K_M | ~8.4 / ~9.8 GiB | Weights fit easily. Dense attention means a larger KV than Qwen3.8. Keep 32768 until measured. The engine still requires two T4s even if one might hold the file |

## Unresolved risks before TPU-generic support

- TPU profiles are labels over the old kernels. They do not drive vLLM or JAX from JSON. A generic TPU engine would have to absorb datasets, the MTP patch, XLA cache, and the GLM package embedding without changing measured behavior.
- `vllm-gpu` is a legal pair in the schema and has no engine. Do not mark a profile launchable for it.
- The pinned llama.cpp build is ai-dock v0.4.0. Qwen3 versus the `qwen35` GGUF architecture id, embedded MTP heads, and foreign chat templates are unverified on that binary.
- Blackfrost's template adds the publisher's own system prompt. Compliance you observe may be the prompt, not the ablation.
- SHA-256 values for candidates were read from LFS metadata. The first real run must hash the file on disk. The Qwen GPU digest was checked against the same API and matched the Wave 1 pin; the blob itself was still not downloaded.
- Context numbers disagree across cards and GGUF metadata (RootMonsteR). The profile stores both and launches the smaller cap.
- Optional Android fields are not on the TPU event path. A phone watching a TPU session still infers the model the way it did in Wave 1.
- Kaggle scratch and the 20 GiB pre-check can fail a 16 GB download even when VRAM would have been enough.

## What still needs a physical Kaggle test

- The shared engine on two T4s with the existing Qwen Q4 file: process starts, checksum matches, `GET /v1/models` returns 200, the ready event has `model_id` and the key, other phases do not, the tunnel stays on localhost plus cloudflared.
- Any candidate above, one quant at a time, only after that Qwen path is confirmed.
- Refusal behavior. Do not treat a tag as a result.
- TPU Qwen and GLM regression. This branch does not change those kernels, but a stack merge still deserves one session before `main`.

## Recommended next step

On a real Kaggle GPU T4 x2 session, run only `python launch.py serve --accelerator gpu` (the Qwen profile). Confirm the on-disk SHA-256, a short chat, and the ntfy envelope. Do not flip any candidate to `launchable` in that same change. Leave PR #4 alone. TPU-generic profiles are a later wave, after this GPU engine has one successful Qwen run.

Mythos (`qwen38-mythos-27b-q4km`) is the first external profile to try after that Qwen run, still at 8192 context, still only if the on-disk hash matches. It is not the default.

## Candidate #1 — Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic

First serious external target for the generic GPU profile path. Same ~27B Q4 class as the Unsloth baseline, with a different chat template and a community GGUF. It is how we check that a new JSON shows up in `models` / `model-info` and is refused by `serve` without a Python branch. It is not the default. `launchable` stays false. No weight was downloaded. Re-checked 2026-09-25: both pins below were still the Hugging Face tips.

### Why it is interesting

The author repo is an agent-oriented Qwen3.8-27B (Mythos template, tool XML, reasoning) on the OBLITERATUS chain. The GGUF we would actually run is a third-party quant of that repo, same architecture id (`qwen35`) as the Blackfrost note and the same rough file size as the served UD-Q4. Tags and `alignment_style` are metadata. They do not change argv, downloads, or isolation.

### Exact sources and pins

| Role | Repo | Revision |
|---|---|---|
| GGUF we pin | `mradermacher/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-GGUF` | `01a19fb59c4130c1ae51b614eccc50dd62de4b02` |
| Weight parent | `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` | `528121d7b0b85885a658dfe67e3643b8a4f9337e` |
| OBLITERATUS parent | `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | not separately pinned |
| Base | `Qwen/Qwen3.8-27B` | not separately pinned |
| imatrix sibling, not first | `mradermacher/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-i1-GGUF` | `eee7c1a5b34280252b44b1b16996d8edd2f76149` |

Both GGUF and medismera cards declare `apache-2.0`. Author `config.json`: `Qwen3_5ForConditionalGeneration`, `model_type` `qwen3_5`, 64 layers, `full_attention_interval` 4, `max_position_embeddings` 262144, MTP keys present, BF16 safetensors total `27781427952`. GGUF metadata: architecture `qwen35`, context 262144. No `SHA256SUMS` file on the GGUF repo. Do not use a mutable `latest` URL. Do not use the author's own tree for GGUF: the card says that repo standardizes on safetensors to avoid GGUF kernel bugs.

Profiles: `qwen38-mythos-27b-q4ks`, `qwen38-mythos-27b-q4km`, `qwen38-mythos-27b-q5ks`. Spanish display names `Qwen3.8 27B Mythos Agentic — Q4_K_S`, `— Q4_K_M`, `— Q5_K_S`. `trust_remote_code` false. `experimental` true. `alignment_style` `obliterated`. `artifact_hash_status` `UNVERIFIED_UNTIL_FIRST_DOWNLOAD_HASH`. Tags: `uncensored`, `agentic`, `tool-calling`, `obliterated`. Launch context **8192** (baseline Qwen GPU stays 32768). Architecture context 262144. Host `127.0.0.1`, layer split `1,1`, `n_gpu_layers` all, `mtp_tokens` 0, chat template and tokenizer `embedded-in-gguf`.

### GGUF options

| Profile | File | Bytes | LFS oid used as sha256 |
|---|---|---:|---|
| `qwen38-mythos-27b-q4ks` | `Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic.Q4_K_S.gguf` | 15825300672 | `0f146e0c6b1ab09f48f3f9cca8a423362a8573d1cd747eaaccc22c7f6dd07f50` |
| `qwen38-mythos-27b-q4km` | `Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic.Q4_K_M.gguf` | 16810716352 | `3cc24a3e431930401b446d9abb52d4e1fa4add4ec19df19b5b5b0c1dbc22da4b` |
| `qwen38-mythos-27b-q5ks` | `Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic.Q5_K_S.gguf` | 18971684032 | `8145d2b7cce80ef444d2a3ca9b463043fe04cc9daddef6d0e12383e5fe506049` |

Oids matched `paths-info` on the pinned revision. They are not a hash of a downloaded blob.

### Estimated hardware fit

Dual T4 is still 2×16 GB, 32 GB combined, no NVLink, ≥20 GiB scratch. Unmeasured.

| Quant | File | Rough VRAM (weights + overhead) | Dual T4 | Start context | Practical ceiling before KV pain |
|---|---|---|---|---|---|
| Q4_K_S | ~14.7 GiB | ~18–24 GB | plausible with layer split | 8192 | maybe 16–32k, untested |
| Q4_K_M | ~15.7 GiB | ~19–26 GB | plausible, same class as baseline UD-Q4 (16464440224 bytes) | 8192 | maybe 16–32k, untested |
| Q5_K_S | ~17.7 GiB | ~22–28 GB | marginal; may need CPU offload | 8192 | likely 8–16k |

128k/256k is not a dual-T4 plan. Order for a practical eval, not a quality ranking: Q4_K_S, then Q4_K_M, then Q5_K_S. Recommended first GGUF: **`qwen38-mythos-27b-q4km`**.

### Tool calling

Unverified on llama.cpp. Author SGLang snippet uses `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3`, and sets `trust_remote_code`. Their Transformers snippet uses `AutoModelForImageTextToText` with `trust_remote_code=True`. Our profiles do not. A vLLM hermes parser was not confirmed on the card text that was read. Generation defaults in `generation_config.json` (temperature 0.65, top_k 20, top_p 0.95, repetition_penalty 1.15, presence_penalty 0.3, max_new_tokens 16384) are not applied by the GPU engine.

Do not call it agentic until this battery passes: one function call, several tools, malformed arguments, continuation after a tool result, a multi-step loop, repeated invocation, long conversation plus tools.

### Context strategy

Profile cap is 8192 so Stage A is the launch default. Baseline `qwen38-27b-gpu` is unchanged at 32768. Raise Mythos only after A, then B (16384), then C (32768). 64k/128k only after C.

### Risks and unverified claims

- Author "0.00% refusal" / 30/30 compliance is their own 30-prompt battery. Not reproduced. Not a guarantee.
- Hybrid `qwen35` plus MTP plus an embedded Mythos template may not load on pinned ai-dock llama.cpp v0.4.0.
- Author moved off GGUF citing kernel bugs. This file is someone else's quant of their safetensors.
- Checksum is an LFS oid until the first download is hashed locally.
- i1 imatrix exists and is not the first file.
- Do not confuse with non-Mythos OBLITERATUS GGUFs.

### Physical test plan

Only after the stock Qwen GPU profile has one clean dual-T4 run.

- **Stage A:** `qwen38-mythos-27b-q4km`, 8192 context, short generation, both T4s used, no bad CPU fallback, localhost `/v1/models`.
- **Stage B:** 16384, VRAM, tok/s, stability.
- **Stage C:** 32768, VRAM, tok/s, long-context stability.
- Then the tool battery. Q5_K_S only if A was comfortable. Q4_K_S is the smaller fallback, not the first file.

### Reject criteria

Refuse to mark it supported if the pinned llama.cpp cannot load the GGUF, the on-disk SHA-256 mismatches, both T4s are not used without a bad CPU fallback, the tool battery fails, the chat template breaks OpenAI-compatible clients, the author's GGUF kernel bugs show up, or Stage A OOMs at 8192.

### What a dual T4 still has to prove

Load, hash, split, 8192 generation, template behavior, and tools. None of that is known from metadata.
