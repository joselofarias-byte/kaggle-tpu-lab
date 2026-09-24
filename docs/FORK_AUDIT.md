# Fork audit — kaggle-tpu-lab

Audit date: 2026-09-24. Base of this fork: `joselofarias-byte/kaggle-tpu-lab` `main` at `bc76571` ("Android ES: APK paralelo, estados claros y Keystore durable").

**This branch is the experimental runtime.** It changes the Qwen pin from vllm-tpu 0.28.0 to 0.29.0 and embeds `mtp-rollback-v0290.diff`. Do not merge it to `main`. Validation steps are in `docs/EXPERIMENTAL_VLLM_029.md`. The recommendations below still describe what belongs on production `main`.

This document compares that tree with the upstreams listed in the integration brief. It records what we already have, what is safe to adapt, what stays experimental, and what to skip. No Kaggle session API was called. No fork was merged wholesale.

## Our baseline

`main` is `ARahim3/kaggle-tpu-lab` at `1aa1f08` plus two Android commits:

| Commit | What it added |
|---|---|
| `5efd682` | Versioned ntfy envelope (`event_version`, `state`, `message_es`, `error_code`, `hint_es`, `cause`, `tail`) on the Qwen and GLM kernels, Spanish launcher text |
| `bc76571` | Android overlay: Spanish UI, states En cola / Iniciando / TPU lista / Sin conexión, queue adoption, Keystore with durable fallback, package `com.joselofarias.kaggletpulab`, APK workflow |

Primary upstream `ARahim3/kaggle-tpu-lab` HEAD is still `1aa1f08` (2026-09-15). Nothing newer to take from primary.

Mobile upstream `afterexam/kaggle-tpu-lab-mobile` HEAD is `253a8dd` (2026-09-24), the commit our APK workflow already pins. That commit **reverted** Keystore storage back to plaintext Preferences. Our overlay puts Keystore back. Do not fast-forward the pin onto a future mobile commit without re-checking that revert.

The APK workflow checks out `afterexam` and runs `generate_templates.py`, which prefers `mobile-src/kernels/*.py` over this repo. Those bundled kernels do **not** carry our Spanish event envelope. Wave 1 copies our kernels over that directory before template generation so Android launches and `launch.py` share one source.

## How to read the recommendations

- **Have** — already on our `main`, or a strict subset of it. Do not port again.
- **Wave 1** — adapted into this tree, preserving the Android event contract and the vllm-tpu `0.28.0` production pin.
- **Experimental** — isolated on `cursor/experimental-vllm-029-3ac3`. Not a production baseline until a real Kaggle TPU run measures it.
- **Skip** — do not port. Reason is in the row.

`k0valik/kaggle-tpu-lab` (`dba3bb13`, 2026-09-24) already absorbed the useful hunks from the open PRs it was tracking (qdubois, Kitkitkittt fail-fast, and others). Wave 1 follows that consolidation and does not re-copy the older forks where k0valik is the later source.

## 1. k0valik/kaggle-tpu-lab (highest priority)

HEAD `dba3bb13`. Fork point is our upstream `1aa1f08`, then a wholesale Haz4rdovisk Tauri merge (`af6973a`, ref `Haz4rdovisk@6c0ef49`), then stages A–F (`140ab53` … `308396e`) and a double-pass (`f62b1da`). Plans live in `plans/00-OVERVIEW.md`. The owner recorded that live TPU validation was still deferred.

| Area | Source | Verdict |
|---|---|---|
| vllm-tpu `0.29.0` rebaseline | `140ab53` Stage A, `qwen38-27b/kernel/serve_qwen38.py` pin | **Experimental.** Plugin 0.29.0 was not run on a TPU in that fork. Our production pin stays `0.28.0`. |
| MTP rollback re-port | `qwen38-27b/patches/mtp-rollback-v0290.diff` | **Experimental.** The 0.28.0 patch `mtp-rollback-v0280.diff` stays the production patch. The 0.29 diff does not apply to 0.28. |
| `--enable-prefix-caching` | `server_args()` in the 0.29 kernel | **Experimental.** Comment in k0valik says it depends on upstream #3422, which is a 0.29-era fix, and still "needs live test". |
| Prefix-cache-independent MTP-head fallback | `has_mtp_head()` in `f62b1da` | **Wave 1.** Reads `model.safetensors.index.json`. No runtime bump. Default bf16 checkpoint still has MTP tensors, so the shipped MTP=3 path is unchanged. Missing or unreadable index fails safe to `mtp_tokens=0`. |
| Fail-fast topology | `tpu_topology_ok()` Stage C `0e443be` | **Wave 1, adapted.** We already had `tpu_check()` (`9b0ad44` / `1aa1f08`). Topology is an earlier JSON device list: 8 TPUs pass, a reported non-TPU shape fails immediately with `error_code=tpu-topology` and Spanish `hint_es`. An empty/unparseable probe stays inconclusive and falls through to the existing `tpu_check()` so a missing JAX answer does not false-fail harder than today. |
| Env sanitization | `sanitize_tpu_env()` Stage C, from upstream idea in k0valik PR #8 | **Wave 1.** Drops `TPU_WORKER_HOSTNAMES` and `TPU_WORKER_ADDRS` before libtpu starts. Same helper on the GLM preflight. |
| Internet preflight | `internet_check()` | **Wave 1 on Qwen.** GLM already had this inside `preflight()` (`1aa1f08`). |
| Death diagnostics | `server_died()` Stage C | **Wave 1.** Clean rc=0 is described as an external stop. Unknown crashes name `vllm.log`. The `__delitem__` hint still says **0.28.0**, not 0.29.0. |
| Log-start offset | `_LOG_START` in `launch_server()` / `server_death_report()` | **Wave 1.** Append-mode `vllm.log` no longer attributes the previous run's traceback to this one. |
| Generic Qwen checkpoint, HF download, `build-weights` | Stage B `f4c7b56`, Stage B2 `bcec5be` | **Skip for this wave.** Useful later, but it changes weight resolution, cache bypass, and a new CPU kernel flow. Default dataset path is unchanged and already works. Do not land it in the same PR as the security fixes. |
| Pinned cloudflared + SHA-256 | Stage D `63751fd`. Version `2026.9.1`, digest `03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc` | **Wave 1.** Re-downloaded the official asset on 2026-09-24 and confirmed the digest. Dataset copy is used only when it matches. Mismatch or network failure aborts. No `releases/latest`. |
| vLLM `--host 127.0.0.1` | Stage D `server_args()` | **Wave 1.** Health checks were already loopback. The server socket was not pinned. |
| Secret hygiene / ntfy | Stage D `publish()` drops `api_key` | **Wave 1, adapted.** k0valik drops the key from every ntfy event and reads it from `~/.kaggle-tpu-lab.json`. Android queue adoption (`racer.ts`) still accepts `ev.api_key` for a kernel it did not mint. We keep `api_key` on the **`ready` event only**, strip it from every other phase, and redact it from vLLM/pip logs and from the logged PHASE line. The notebook READY banner still prints the key for the person watching the cell. |
| Atomic `0600` state | `write_state()` Stage D | **Wave 1** in `launch.py`. |
| XLA tar-slip | `safe_tar_members()` `f62b1da` | **Wave 1** before `tar` of the Qwen cache. GLM engine extract rejects absolute paths, `..`, and links. |
| Named tunnel + stable API key secret | Stage D `cloudflare_hostname`, `api_key_secret` | **Skip for this wave.** Opt-in and untested end-to-end (k0valik changelog). It adds Kaggle-secret labels the Android CFG injector does not send. Defaults would be safe, but it is not required for the security baseline. |
| Kernel/notebook single source | `tools/sync_notebook.py` Stage E `0d92566` | **Wave 1, adapted** to our `%%writefile` notebooks and `tools/pack_notebook.py` engine cells. |
| CI | `.github/workflows/check.yml` Stage E | **Wave 1, adapted.** Our workflows stay. Python job compiles, runs `tests/`, and `sync_notebook.py --check`. Android workflow is unchanged except it now templates **our** kernels. |
| Tauri app | `af6973a` merge of Haz4rdovisk | **Skip.** Frozen desktop UI. See section 4. |
| Docs rewrite that marks 0.28 numbers stale | Stage F `308396e` | **Skip on main.** Those numbers are the measured 0.28 baseline. The experimental branch carries the "not re-measured" note instead of rewriting production READMEs. |

## 2. qdubois/kaggle-tpu-lab

HEAD `4aa5a79` (2026-09-07). Two commits on top of the old single-folder tree (before `f3a6b48` split recipes into folders):

| Commit | Content | Verdict |
|---|---|---|
| `28dd300` | `hardening.py`, signed progress, cloudflared pin, localhost, atomic state, secret redaction | **Already absorbed by k0valik Stage D** (`63751fd`, their PR #6). Do not port `hardening.py` or `tools/sync_security.py` as a second implementation. |
| `4aa5a79` | Use the installed `kaggle` executable when it is on `PATH` | **Wave 1** in `launch.py` (`shutil.which("kaggle")`, else `python -m kaggle`). |

Their tree has no GLM recipe, no Android contract, and an older Qwen layout. Merging the fork would drop both.

HMAC-signed progress events were considered by k0valik and rejected (overview: "Confirmed rejections: HMAC protocol"). Signing would break Android's current JSON parse of the ntfy `message`. Not ported.

## 3. Kitkitkittt/kaggle-tpu-lab

HEAD `7d59604` (2026-09-08). Standalone import of an audited TPU+GPU tree. Not a descendant of `1aa1f08` in a way we can rebase.

| Area | Verdict |
|---|---|
| TPU fail-fast, `0600` state, notebook `--check`, preflight cells | **Already in k0valik** (their PR #5) and covered by Wave 1 above. Not copied a second time from this older tree. |
| Dual-T4 GPU path | **Wave 1, separate entrypoint.** `qwen38-27b/gpu/serve_qwen38_gpu.py` plus `launch.py serve --accelerator gpu`. Default remains TPU. The script refuses to start unless `nvidia-smi` shows two T4s, pins the GGUF revision and SHA-256 and the llama.cpp CUDA tarball from Kitkitkittt, and uses the same cloudflared `2026.9.1` digest verified above. vLLM binds were loopback; llama-server is bound to `127.0.0.1` instead of `0.0.0.0`. Events use our envelope so a future Android GPU slug would not see a foreign payload. Android `ModelId` is unchanged; the APK does not grow a GPU button. |
| Notebook `notebook/gpu/qwen38-t4x2-serve.ipynb` | **Skip.** The script kernel is the launcher path. A second generated notebook can land once someone has run the GPU kernel on Kaggle. |
| `machine_shape: NvidiaTeslaT4` | Taken from their `notebook/gpu/kernel-metadata.json`. Public CLI reports disagree on whether that shape is T4 x2 or a single T4. The kernel's dual-T4 preflight is the real guard: a single-GPU session exits before the 15 GiB download. |

GGUF SHA-256 and llama.cpp tarball SHA-256 were **not** re-downloaded here (multi-gigabyte). They are copied from Kitkitkittt `kernel/serve_qwen38_gpu.py` at `7d59604` and should be re-hashed on the first real GPU run.

## 4. Haz4rdovisk/kaggle-tpu-lab

HEAD `6c0ef49` (2026-09-22). Tauri 2 + React companion on top of `1aa1f08`. k0valik merged this tree wholesale and then froze `src/` and `src-tauri/`.

We do not want that app. Our companion is the Android overlay.

Ideas that match the Android contract, and what we did:

| Idea | Where it lives upstream | Adaptation |
|---|---|---|
| Do not treat `tunnel-url` as ready | `src-tauri/src/state/machine.rs`, already true in `racer.ts` `decideEndpointMount` | **Have.** Not reimplemented. |
| `GET {endpoint}/v1/models` with the bearer key, key never logged | `src-tauri/src/kaggle/probe.rs` | **Wave 1.** `mobile-overlay/files/src/services/endpointProbe.ts` and the same helper in `launch.py status`. Three failed probes, and only after the endpoint has been mounted for 45s, move that endpoint to `OFFLINE`. A later success mounts it `READY` again. Header: a live READY endpoint is **TPU lista**; a session whose endpoint went `OFFLINE` is **Sin conexión**, not **Iniciando**. |
| Queue vs running vs ready | tray states in `src-tauri/src/tray.rs` | **Have** as En cola / Iniciando / TPU lista / Sin conexión. |
| Stale-session reattach guard | `2d68d26` | **Have** in adoption rules (`refreshAccount` only adopts `RUNNING` or `QUEUED`). Not replaced. |
| `machine_shape: TpuV5E8` on push | companion launcher hunk, also k0valik | **Wave 1** on the TPU metadata so a push does not depend on the account default. |

Not ported: React UI, Rust process supervisor, tray icons, theme, window size, Portuguese/Spanish desktop copy, `scripts/*.ps1`.

## 5. seppegadeyne/kaggle-tpu-lab

HEAD `6dcc9bb1` (2026-09-22).

| File | What it is | Verdict |
|---|---|---|
| `start.sh` / `stop.sh` | Personal wrappers around `launch.py` plus a Hermes config rewrite | **Skip.** Hard-coded home paths and a one-user agent integration. `launch.py serve` / `stop` already cover the generic case. |
| `overnight_watch.py` | One-shot watcher for a dated GLM run: benchmarks, email to `seppe@fushia.be` via Proton Bridge, wiki append, then `launch.py stop` | **Skip.** The scheduling idea is fine; the script is a personal ops note (`CUTOFF_STR`, mailbox, Hermes paths). Porting it would ship someone else's addresses. |
| Removal of `qwen38-27b/` (`79c40577`) | Their fork is GLM-only | **Skip.** We still ship Qwen. |

GLM kernel itself matches upstream `1aa1f08` plus our Android envelope. No extra kernel hunks to take.

## Event contract after Wave 1

Still version 1. Additive fields only.

`api_key` rules:

- Present on `phase=ready` (Android adoption and the CLI banner).
- Absent on every other phase, including `heartbeat`.
- Redacted in pip/vLLM log streams and in the logged `PHASE` line.

New `failed` steps the UI already understands via `message_es` + `hint_es` + `cause` + `tail`:

| `error_code` / `step` | `message_es` intent |
|---|---|
| `tpu-topology`, `no-tpu` | Sesión sin TPU utilizable |
| `no-internet` | Internet apagado |
| `tunnel-binary` | cloudflared no verificado |
| `cache-extract` | tar de caché rechazado |

`mtp-disabled` is `state=starting` with a Spanish sentence. It is not an error.

## Experimental validation (do not merge to production main)

Branch `cursor/experimental-vllm-029-3ac3` carries only the runtime bump on top of Wave 1:

1. Pin `vllm_tpu_version` to `0.29.0`.
2. Swap in `qwen38-27b/patches/mtp-rollback-v0290.diff` (from k0valik `dba3bb13`) and regenerate `MTP_PATCH_B64`.
3. Pass `--enable-prefix-caching`.

Before anyone treats that branch as the default, run one real Kaggle TPU v5e-8 session (not from this audit) and check:

- `jax.device_count()==8` and the MTP patch applies cleanly (`previously applied` or exit 0).
- Greedy exact-match vs MTP off, same prompt set the 0.28 patch used (the historical claim is 12/12).
- One JSON-mode request with MTP on; if the server dies with `__delitem__`, the 0.28 workaround (`async_scheduling: false`) is still required.
- A second identical request hits prefix cache (log line or latency drop), or the flag is reverted.
- Startup still finishes with the env dataset, and `event_version` / `message_es` still appear on ntfy.
- No API key in non-ready ntfy messages.

Until those notes are filled in with a kernel log, the experimental PR stays draft.

## Deliberately not done

- No merge commits from any fork.
- No edits to a running or queued Kaggle kernel.
- No secrets, tokens, or tunnel credentials committed.
- No Tauri tree, no HMAC event protocol, no DFlash2, no hardcoded third-party abliterated checkpoints.
- No production README rewrite that replaces measured 0.28 tok/s figures with unmeasured 0.29 claims.
