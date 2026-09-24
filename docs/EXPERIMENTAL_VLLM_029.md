# Experimental: vllm-tpu 0.29.0

**Not merge-ready for production `main`.** This branch is `cursor/experimental-vllm-029-3ac3`. It sits on top of the Wave 1 security and reliability work and changes only the Qwen runtime baseline.

## What changed

Taken from k0valik `140ab53` (Stage A) and the double-pass note in `f62b1da` / `dba3bb13`:

| Item | File |
|---|---|
| Pin `vllm_tpu_version` `0.28.0` → `0.29.0` | `qwen38-27b/kernel/serve_qwen38.py` |
| MTP rollback re-port | `qwen38-27b/patches/mtp-rollback-v0290.diff`, re-embedded by `qwen38-27b/tools/embed_patch.py` |
| `--enable-prefix-caching` | `server_args()` |

`patches/mtp-rollback-v0280.diff` is still in the tree so the production patch can be compared. The kernel applies the 0.29 blob only.

k0valik recorded that this was checked semantically (the patch dry-runs against a pristine 0.29.0 tree) and **not** on a Kaggle TPU. This audit did not run a kernel either.

## Why it is separate

The 0.28.0 recipe has measured numbers (~130 tok/s, MTP exact-match 12/12 on that patch). 0.29.0 rewrote the GDN files, so the old patch does not apply. Prefix caching depends on an upstream fix that k0valik itself marked "needs live test". Shipping that as the default would replace a measured baseline with an unmeasured one.

Wave 1 behavior that this branch keeps: localhost bind, pinned cloudflared, Spanish event envelope, `api_key` only on `ready`, MTP-head fallback, fail-fast gates.

## How to validate on a real Kaggle TPU v5e-8

Do this from a machine that is allowed to use your Kaggle quota. This repository's cloud agent must not call the Kaggle API.

1. Push **this branch's** Qwen kernel, not `main`. Attach the usual weights and env datasets. The env dataset's compile cache was built for 0.28.0, so expect a cold compile and a log line that the cache will not match.
2. Confirm the session actually has a TPU: the topology event must not be `tpu-topology`, and `import jax; print(jax.device_count())` must print 8.
3. Confirm `PHASE mtp-patch-applied` (or a log line that the patch was previously applied). If you see `mtp-patch-failed`, stop. Do not serve with MTP on.
4. Greedy exact-match: same prompt, `temperature=0`, once with `mtp_tokens=3` and once with `mtp_tokens=0`. The 0.28 claim was 12/12 identical completions. Record the count.
5. Send one JSON-mode / structured-output request with MTP left on. If the server dies with `AttributeError: __delitem__`, the 0.28 workaround still applies (`--no-async-scheduling` or `mtp_tokens=0`).
6. Send the same long prompt twice and look for a prefix-cache hit in `vllm.log` (or a clear latency drop on the second prefill). If there is no hit, drop `--enable-prefix-caching` before any further discussion of merging.
7. On ntfy, confirm `event_version`, `state`, and `message_es` are present, and that `api_key` appears only on `ready`.

Paste the kernel log (with the API key redacted) into the experimental PR before anyone retargets it at `main`.

## Merge rule

Leave this pull request in draft. Do not retarget it onto `main` until the checklist above is filled in from a real TPU session.
