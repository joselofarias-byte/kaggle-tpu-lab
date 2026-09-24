# Wave 1 review handoff

Handoff for a reviewer (or another model) who did not see the working session. Do not merge from this document. Do not call any Kaggle session or kernel API.

## Objective

Deep-review draft PR #3 (`cursor/fork-audit-wave1-3ac3` → `main`) and fix real issues before a human merge. Leave PR #4 and branch `cursor/experimental-vllm-029-3ac3` untouched. Preserve the Android Spanish event contract and the production pin **vllm-tpu 0.28.0**.

Repository: https://github.com/joselofarias-byte/kaggle-tpu-lab

## Review findings

PR #3 is an adaptation, not a fork merge. Production Qwen stays on vllm-tpu `0.28.0` with `patches/mtp-rollback-v0280.diff`. The diff does not add `--enable-prefix-caching` or `mtp-rollback-v0290.diff`.

What already matched the requested security baseline:

- vLLM and llama-server bind to `127.0.0.1`.
- cloudflared is pinned to `2026.9.1` with SHA-256 `03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc`. The unpinned GitHub download URL is not used.
- Launcher state is an atomic replace, mode `0600` (`launch.py` `write_state`).
- `api_key` is kept only on ntfy `phase=ready`. Other phases drop it. Pip and vLLM lines go through `redact`.
- Android still reads `event_version`, `state`, `message_es`, `error_code`, `hint_es`, `cause`, `tail`. Queue adoption still uses `ev.api_key` on the ready event (`racer.ts`).
- `launch.py` `probe_endpoint` sends the bearer token and does not log it. The Android probe does the same. `CapacitorHttp.enabled` is true, so device `fetch` is the native HTTP stack rather than a WebView CORS request.

`launch.py` was read against `main`. No launcher logic bug required a code change. `cmd_status` still prints the API key in the operator's terminal after a ready event. That file is mode `0600` and is not sent to ntfy.

## Issues found

1. **Must-fix, confirmed.** `safe_tar_members()` in `qwen38-27b/kernel/serve_qwen38.py` rejected absolute paths and `..` only. A symlink or hardlink under a safe name was accepted. Python's `data` filter would still allow an in-destination link, and the cache is extracted with system `tar` into `/tmp`.
2. **Related clobber.** A regular member named `cloudflared` (or a link that lands on `/tmp/cloudflared`) would replace the binary that had already been SHA-256 verified. The tunnel then executed whatever was on disk.
3. **GLM extract gap.** `safe_extract_tar_bytes()` already rejected `issym()` / `islnk()`, then called `extractall()` with no `filter`. On Python 3.12 that default still writes links if the pre-check is skipped. Special members (fifo, devices) were not named.
4. **GLM bind.** `ThreadingHTTPServer` listened on `0.0.0.0`. The tunnel and the self-test already use `127.0.0.1`. Qwen and the GPU server were already on localhost.
5. **Secret edges.** GLM `sh()` logged stderr without `redact`, and GLM `redact()` did not strip `token=` assignments. The GPU script logged the full PHASE JSON, so the ready line contained `api_key`, and `publish("failed", error=...)` could forward an exception string that contained the key. Android `formatEvent` dumped unknown phases with `JSON.stringify(ev)`, which would show `api_key` if one were present.
6. **Probe counter.** A newly mounted endpoint kept the previous account's miss count, so one more failure could mark a fresh URL offline. `setCustomEndpoint` did not set `readySince`, so a custom URL skipped the 45s grace.
7. **Ignored tar failure.** A non-zero `tar` exit after a passing member check did not stop the kernel.
8. **GPU archive shape, not a reject-all.** The pinned llama.cpp CUDA tarball (`llama.cpp-v0.4.0-cuda-12.8-amd64.tar.gz`, 153,202,310 bytes) was downloaded during this review. Its SHA-256 matched the pin `7a229ac0357d9d80931e6651a4231ee052157875dea9e6160b4636763b5a4bad`. It contains 14 relative soname symlinks (`libllama.so.0` → `libllama.so.0.4.0` and the same pattern for the other libraries) and zero hardlinks. Rejecting every symlink would make the GPU entrypoint unable to load those libraries.

Not a defect: the notebook READY banner and the ready ntfy event still contain `api_key`. That is the adoption contract. See below.

## Fixes applied

Commit `a8ab5c65b07b080b53e23fb1cbacfbf7947725b0` on `cursor/fork-audit-wave1-3ac3`.

- `safe_tar_members(archive, prefix="xla_cache")` refuses absolute paths, `..`, symlinks, hardlinks, non-regular/non-directory members, and names outside `xla_cache/`. The cache call passes that prefix. A non-zero `tar` exit publishes `cache-extract` and exits.
- Before starting the Qwen tunnel, the on-disk cloudflared is hashed again. A mismatch publishes `tunnel-binary` and exits. A dataset copy is hashed again after the atomic install. GLM refuses to `Popen` cloudflared when the file is missing or the digest does not match, and chmods a matching binary to `0755`.
- GLM extraction rejects symlinks and hardlinks by name, rejects other special members, and calls `extractall(..., filter="data")`. The HTTP server binds to `127.0.0.1`. `sh()` and `redact()` follow the Qwen token rules.
- GPU `safe_extract` refuses hardlinks, absolute symlink targets, `..` in the link, and any link that resolves outside the destination. Relative in-tree symlinks are extracted because of the pinned soname links above. Checksums use `hmac.compare_digest`. PHASE logs and failure strings go through `redact`. The READY banner still prints the key. The ready ntfy payload still includes `api_key`.
- Android display text for an unknown phase omits `api_key`. `nextProbeDecision` is the three-miss rule (still listed on misses 1 and 2, `OFFLINE` and unlisted on the third, `READY` again on a later success). Mounting an endpoint clears the miss counter. Custom endpoints set `readySince`.

## API key contract (intentional)

| Surface | `api_key` |
| --- | --- |
| ntfy event `phase=ready` | Present. Android adoption of a queued kernel the app did not mint reads `ev.api_key`. |
| Every other ntfy phase | Removed before send. |
| Kernel PHASE log line | Redacted, including the ready phase. |
| Notebook READY banner (`log()` of the banner, not the PHASE line) | Printed, so the person watching the cell can copy it. |
| `launch.py status` / the ready panel in `watch()` | Printed on the operator's machine from the local `0600` state file or the ready event. |
| Android event list text | Not printed. The in-memory session still keeps the key for chat and for the probe. |
| Kaggle token / `kaggle.json` | Not written by these kernels and not added to events. |

No new secret was committed. Test fixtures use obvious placeholders (`sk-secret`, `sk-adoptionkey`).

## Files modified

- `qwen38-27b/kernel/serve_qwen38.py`
- `qwen38-27b/notebook/qwen38-tpu-serve.ipynb` (kernel cell regenerated by `tools/sync_notebook.py`)
- `glm53-flash/kernel/serve_glm53.py`
- `glm53-flash/notebook/glm53-tpu-serve.ipynb` (same)
- `qwen38-27b/gpu/serve_qwen38_gpu.py`
- `tests/test_wave1_hardening.py`
- `mobile-overlay/files/src/services/endpointProbe.ts`
- `mobile-overlay/files/src/services/endpointProbe.test.ts`
- `mobile-overlay/files/src/services/racer.ts`
- `mobile-overlay/files/src/services/ntfy.ts`
- `mobile-overlay/files/src/services/ntfy.test.ts`
- `docs/WAVE1_REVIEW_HANDOFF.md` (this file)

`launch.py` was reviewed and not edited.

## Commits, branch, PR status

- Branch: `cursor/fork-audit-wave1-3ac3` (tracks `origin` after the push that accompanies this file).
- Base: `main` at `bc76571` (Android ES overlay on ARahim3 `1aa1f08`).
- Prior Wave 1 commits: `9187df8` audit, `75d25c8` integrations.
- Review fix: `a8ab5c65b07b080b53e23fb1cbacfbf7947725b0`.
- PR: https://github.com/joselofarias-byte/kaggle-tpu-lab/pull/3 — **draft, open, not merged**. Base `main`.
- Experimental: https://github.com/joselofarias-byte/kaggle-tpu-lab/pull/4 — still draft, still based on the Wave 1 branch, still at `231a1ec`. This review did not check out, commit, or push that branch. Do not merge #4 into #3 or into `main`. Doing so would put vllm-tpu 0.29.0 on the pull request that targets `main`.

## Tests executed and results

All commands were run on 2026-09-24 from this checkout. No Kaggle API was called.

```text
python3 -m py_compile launch.py qwen38-27b/kernel/serve_qwen38.py \
  qwen38-27b/gpu/serve_qwen38_gpu.py glm53-flash/kernel/serve_glm53.py \
  tools/sync_notebook.py tools/pack_notebook.py
```

Exit 0.

```text
python3 -m unittest tests.test_wave1_hardening -v
```

`Ran 18 tests in 0.128s` — **OK**. New coverage: Qwen symlink, Qwen hardlink, member outside `xla_cache/`, GLM symlink and hardlink (destination not populated), GPU hardlink, GPU absolute symlink, GPU `../` symlink, GPU relative soname symlink kept, GLM `token=` redaction, absence of `"vllm_tpu_version": "0.29.0"`, `mtp-rollback-v0290`, and `--enable-prefix-caching`, GLM bind `127.0.0.1` and no `0.0.0.0`.

```text
python3 tools/sync_notebook.py
python3 tools/sync_notebook.py --check
```

Both recipes updated, then `qwen38-27b: in sync` and `glm53-flash: in sync`. Exit 0.

Android, matching `.github/workflows/mobile-android-es.yml` through the unit-test and web-build steps. Overlay copied onto a local checkout of `afterexam/kaggle-tpu-lab-mobile` @ `253a8dd`, then our Qwen kernel, GLM kernel, and GLM engine:

```text
python3 generate_templates.py
npm test
npm run build
```

- `generate_templates.py` wrote `templates_data.ts` (266026 bytes). `event_version` and `CLOUDFLARED_SHA256` were present in the copied Qwen kernel.
- `npm test` (vitest 5.0.1): **8 files, 45 tests, all passed** (43 before this review, plus the probe-decision test and the api_key display test).
- `npm run build` (`tsc && vite build`): **exit 0**, built in 1.43s.

Not run here: the Gradle APK (`ANDROID_HOME` is unset on this machine). The GitHub Action `Build Android ES` is the APK path and should run because this PR touches `mobile-overlay/**`. Not run: a Kaggle TPU or GPU session.

## Remaining risks

- System `tar` still performs the XLA extract after the Python member check. The check is what blocks links and names outside `xla_cache/`. A tar Python mis-reads and GNU tar extracts differently is a residual gap. The pre-exec cloudflared rehash is the backstop for that one path.
- An official env-dataset cache that actually contains a symlink, a hardlink, or a member outside `xla_cache/` will now stop the kernel instead of extracting. That is fail-closed. It has not been confirmed against `rahim3/qwen38-tpu-env-v5e8`.
- GPU relative symlinks are allowed only when the target stays inside the extract directory. The policy depends on callers verifying the archive first (`download_checked` does).
- The GGUF (`Qwen3.8-27B-UD-Q4_K_M.gguf`, about 15.3 GiB, pin `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`) was not re-downloaded in this review.
- `api_key` on the public ntfy ready message is still the adoption mechanism. Anyone who can read that topic during the ready event can call the endpoint. The topic id is a random `ktl-` prefix. Do not paste ready events into tickets.
- vLLM and llama-server still receive `--api-key` on their process command line inside the Kaggle VM.
- The probe can show **Sin conexión** after three missed `GET /v1/models` calls while Kaggle still reports RUNNING. A later 200 restores **TPU lista** without relaunch. A custom URL that is not an OpenAI `/v1/models` server will follow the same rule after the 45s grace.
- If `CapacitorHttp.enabled` is turned off, WebView CORS can make every probe fail and flip a live tunnel to Sin conexión.
- PR #4 still sits on top of this branch's history at the time it was opened. Merging it into the Wave 1 branch would add the 0.29.0 pin to PR #3.

## What still needs a real Kaggle TPU or GPU

Do this by hand in the Kaggle UI. Do not point an agent at the kernels API while a queued job must be left alone.

TPU (Qwen, vllm-tpu 0.28.0, this branch):

1. Session has Internet and Accelerator TPU VM v5e-8. Topology log says 8 TPU devices, or the older TPU check passes when JAX is inconclusive.
2. cloudflared log says the verified `2026.9.1` pin. The server arguments include `--host 127.0.0.1`.
3. With the env dataset attached, the cache step logs a restore and does **not** publish `cache-extract`. If it does, the dataset has a member this check refuses; do not force-extract it.
4. Ready ntfy JSON has `event_version`, `state`, `message_es`, and `api_key`. A heartbeat from the same run has no `api_key`.
5. Android header moves En cola → Iniciando → TPU lista, and a stopped tunnel becomes Sin conexión without a second launch.

GPU (separate quota, `launch.py serve --accelerator gpu`), only if a dual-T4 session is acceptable:

1. Preflight sees two T4s and stops before the GGUF download otherwise.
2. llama.cpp extract keeps the soname symlinks and `llama-server` starts on `127.0.0.1`.
3. The same ready / non-ready `api_key` rule as the TPU kernel.

Experimental 0.29.0 validation stays in `docs/EXPERIMENTAL_VLLM_029.md` on PR #4. It is not part of this merge.

## Exact recommended next step

Leave both pull requests as drafts. Read this file and the commit `a8ab5c6` on PR #3. Merge PR #3 only after a human accepts the tar and localhost fixes. Do not merge PR #4, and do not retarget it onto `main`, until the TPU checklist in `docs/EXPERIMENTAL_VLLM_029.md` is filled from a real v5e-8 log.
