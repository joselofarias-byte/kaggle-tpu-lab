#!/usr/bin/env python3
"""Serve a GGUF profile on Kaggle's two Tesla T4 GPUs with llama.cpp.

``launch.py`` embeds the validated profile and the session overrides.
llama-server binds to 127.0.0.1. cloudflared is the Wave 1 pin.
The script refuses symlinks and hardlinks in the llama.cpp archive.

This file is the shared GPU engine. The Qwen-only script
``qwen38-27b/gpu/serve_qwen38_gpu.py`` remains runnable on its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Same pins as qwen38-27b/gpu/serve_qwen38_gpu.py. A profile cannot replace them.
LLAMA_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/v0.4.0/"
    "llama.cpp-v0.4.0-cuda-12.8-amd64.tar.gz"
)
LLAMA_SHA256 = "7a229ac0357d9d80931e6651a4231ee052157875dea9e6160b4636763b5a4bad"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/download/2026.9.1/"
    "cloudflared-linux-amd64"
)
CLOUDFLARED_SHA256 = "03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc"

PROFILE = None  # __PROFILE_JSON__  (launch.py embeds the validated profile)
CFG = None  # __LAUNCHER_CONFIG__  (launch.py embeds the session overrides)

PORT = 8000
SCRATCH = Path(os.environ.get("LLAMACPP_SCRATCH", os.environ.get("QWEN38_SCRATCH", "/tmp/llama-cpp-gpu")))
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else SCRATCH
BIN_ARCHIVE = SCRATCH / "llama.cpp-v0.4.0-cuda-12.8-amd64.tar.gz"
BIN_DIR = SCRATCH / "llama.cpp-v0.4.0"
CLOUDFLARED = SCRATCH / "cloudflared-2026.9.1"
SERVER_LOG = WORK / "llama-server.log"
TUNNEL_LOG = WORK / "cloudflared.log"
T0 = time.time()


def log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def redact(text, api_key=""):
    """Strip the session key and bearer tokens before a line is logged."""
    text = str(text)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"sk-[A-Za-z0-9]{16,}", "[REDACTED]", text)
    text = re.sub(r"glm-[A-Za-z0-9]{16,}", "[REDACTED]", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(token=)[^\s\"']+", r"\1[REDACTED]", text)
    return text


def public_event_fields(phase, extra):
    """``ready`` keeps ``api_key``. Every other phase drops it."""
    out = dict(extra)
    if phase != "ready":
        out.pop("api_key", None)
    return out


def identity_fields(cfg):
    out = {}
    for key in ("model_id", "display_name", "backend", "accelerator"):
        value = cfg.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    return out


def publish(phase, cfg, **details):
    """Spanish event envelope plus optional model identity. Does not log secrets."""
    details = public_event_fields(phase, details)
    state = "ready" if phase in ("ready", "serving") else (
        "error" if phase == "failed" else (
            "stopped" if phase in ("complete", "auto-shutdown") else "starting"))
    messages = {
        "gpu-ready": "Dos T4 detectadas.",
        "download": "Descargando un archivo fijado por SHA-256.",
        "model-ready": "Pesos verificados.",
        "server-starting": "Iniciando llama.cpp en las GPU.",
        "model-loading": "Cargando el modelo en las GPU.",
        "tunnel-url": "Endpoint público reservado; todavía no está listo.",
        "ready": "GPU lista y servicio de inferencia disponible.",
        "failed": "La instancia GPU falló durante el arranque o la ejecución.",
        "complete": "Tiempo máximo alcanzado; instancia detenida correctamente.",
        "auto-shutdown": "Tiempo máximo alcanzado; instancia detenida correctamente.",
    }
    payload = {
        "event_version": 1,
        "phase": phase,
        "state": state,
        "message_es": messages.get(phase, phase),
        **identity_fields(cfg),
        **details,
    }
    log("PHASE", phase, redact(json.dumps(payload, ensure_ascii=False), cfg.get("api_key") or ""))
    topic = cfg.get("ntfy_topic") or ""
    if not topic:
        return
    try:
        body = {"topic": topic, "title": f"kaggle-tpu-lab {phase}",
                "message": json.dumps(payload, ensure_ascii=False)}
        req = urllib.request.Request(
            "https://ntfy.sh", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json; charset=utf-8"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        log("(ntfy publish failed:", redact(exc, cfg.get("api_key") or ""), ")")


def serve_config(profile, overrides=None):
    """Flat runtime config. Tags and notes are not copied into the process."""
    if not isinstance(profile, dict):
        raise RuntimeError("falta el perfil embebido")
    overrides = dict(overrides or {})
    source = profile["source"]
    launch = profile["launch_args"]
    ctx = int(overrides.get("ctx_size", profile["context_size"]))
    if ctx < 1 or ctx > int(profile["context_size"]):
        raise RuntimeError(f"ctx_size {ctx} fuera del tope del perfil ({profile['context_size']})")
    host = launch.get("host", "127.0.0.1")
    if host != "127.0.0.1":
        raise RuntimeError("llama-server must bind to 127.0.0.1")
    api_key = overrides.get("api_key") or ""
    return {
        "model_repo": source["repository"],
        "model_revision": source["revision"],
        "model_file": profile["model_file"],
        "model_sha256": profile["sha256"],
        "model_size": int(profile["model_size_bytes"]),
        "ctx_size": ctx,
        "parallel": int(launch.get("parallel", 2)),
        "mtp_tokens": int(launch.get("mtp_tokens", 0)),
        "n_gpu_layers": str(launch.get("n_gpu_layers", "all")),
        "split_mode": str(launch.get("split_mode", "layer")),
        "tensor_split": str(launch.get("tensor_split", "1,1")),
        "host": host,
        "port": int(launch.get("port", PORT)),
        "keepalive_min": int(overrides.get("keepalive_min", 480)),
        "api_key": api_key,
        "ntfy_topic": overrides.get("ntfy_topic") or "",
        "served_model_name": profile["served_model_name"],
        "model_id": profile["id"],
        "display_name": profile["display_name"],
        "backend": profile["backend"],
        "accelerator": profile["accelerator"],
        "gpu_count": int(profile["gpu"]["count"]),
        "gpu_type": profile["gpu"]["type"],
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checked(url, target, expected_sha256, cfg):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        log("Checking cached", target.name)
        if file_sha256(target) == expected_sha256:
            log("Reusing verified", target.name)
            return
        target.unlink()
    partial = target.with_suffix(target.suffix + ".part")
    publish("download", cfg, file=target.name)
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "5", "--retry-all-errors", "--continue-at", "-",
        "--output", str(partial), url,
    ]
    subprocess.run(command, check=True)
    actual = file_sha256(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {target.name}: expected {expected_sha256}, got {actual}"
        )
    partial.replace(target)
    log("Downloaded and verified", target.name)


def is_dual_t4(rows):
    return len(rows) == 2 and all("T4" in str(row.get("name", "")).upper() for row in rows)


def gpu_requirement_met(rows, cfg):
    need = int(cfg["gpu_count"])
    token = "T4" if "T4" in str(cfg["gpu_type"]).upper() else str(cfg["gpu_type"]).upper()
    if len(rows) != need:
        return False
    return all(token in str(row.get("name", "")).upper() for row in rows)


def gpu_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"index": fields[0], "name": fields[1], "memory_mib": int(fields[2])})
    return rows


def require_gpus(cfg):
    rows = gpu_inventory()
    log("GPU inventory:", json.dumps(rows))
    if not gpu_requirement_met(rows, cfg):
        raise RuntimeError(
            f"Expected {cfg['gpu_count']} {cfg['gpu_type']} GPUs. "
            "In Kaggle Settings, turn Internet on, choose Accelerator -> GPU T4 x2, "
            "restart the session, and rerun."
        )
    publish("gpu-ready", cfg, devices=rows)
    return rows


def safe_extract(archive, destination):
    """Reject path traversal, symlinks, and hardlinks before extraction."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing linked tar member {member.name}")
            if os.path.isabs(member.name) or ".." in Path(member.name).parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        try:
            bundle.extractall(destination, members=members, filter="data")
        except TypeError:
            bundle.extractall(destination, members=members)


def server_command(server, cfg, model_path):
    if cfg["host"] != "127.0.0.1":
        raise RuntimeError("llama-server must bind to 127.0.0.1")
    command = [
        str(server),
        "--model", str(model_path),
        "--alias", cfg["served_model_name"],
        "--host", "127.0.0.1",
        "--port", str(cfg["port"]),
        "--api-key", cfg["api_key"],
        "--ctx-size", str(cfg["ctx_size"]),
        "--parallel", str(cfg["parallel"]),
        "--n-gpu-layers", cfg["n_gpu_layers"],
        "--split-mode", cfg["split_mode"],
        "--tensor-split", cfg["tensor_split"],
    ]
    if int(cfg["mtp_tokens"]) > 0:
        command += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(cfg["mtp_tokens"]),
            "--spec-draft-p-min", "0.7",
        ]
    return command


def prepare_llama_server(cfg):
    download_checked(LLAMA_URL, BIN_ARCHIVE, LLAMA_SHA256, cfg)
    marker = BIN_DIR / ".verified"
    if not marker.exists():
        if BIN_DIR.exists():
            shutil.rmtree(BIN_DIR)
        safe_extract(BIN_ARCHIVE, BIN_DIR)
        marker.touch()
    servers = list(BIN_DIR.rglob("llama-server"))
    if not servers:
        raise RuntimeError("The verified llama.cpp archive contains no llama-server binary")
    server = servers[0]
    server.chmod(0o755)
    lib_dirs = sorted({str(path.parent) for path in BIN_DIR.rglob("*.so*")})
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
    version = subprocess.run([str(server), "--version"], env=env, capture_output=True, text=True)
    if version.returncode != 0:
        detail = redact(version.stderr or version.stdout, cfg.get("api_key") or "")[-800:]
        raise RuntimeError("llama-server could not start: " + detail)
    log((version.stdout or version.stderr).strip().splitlines()[0])
    return server, env


def prepare_cloudflared(cfg):
    download_checked(CLOUDFLARED_URL, CLOUDFLARED, CLOUDFLARED_SHA256, cfg)
    CLOUDFLARED.chmod(0o755)


def model_url(cfg):
    return (
        f"https://huggingface.co/{cfg['model_repo']}/resolve/"
        f"{cfg['model_revision']}/{cfg['model_file']}"
    )


def download_model(cfg, model_path):
    free_gib = shutil.disk_usage(SCRATCH).free / 1024**3
    log(f"Scratch space available: {free_gib:.1f} GiB")
    if free_gib < 20 and not model_path.exists():
        raise RuntimeError(f"At least 20 GiB of free scratch space is required at {SCRATCH}")
    download_checked(model_url(cfg), model_path, cfg["model_sha256"], cfg)
    if model_path.stat().st_size != cfg["model_size"]:
        raise RuntimeError(
            f"Unexpected model size: expected {cfg['model_size']}, got {model_path.stat().st_size}"
        )
    publish("model-ready", cfg, path=str(model_path), bytes=model_path.stat().st_size)


def tail(path, size=4000):
    try:
        return Path(path).read_text(errors="replace")[-size:]
    except FileNotFoundError:
        return "(log file not created)"


def api_request(cfg, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{cfg['port']}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(process, cfg, timeout=1200):
    started = time.time()
    next_update = 0
    while time.time() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}:\n"
                + redact(tail(SERVER_LOG), cfg.get("api_key") or "")
            )
        try:
            api_request(cfg, "/v1/models", timeout=5)
            return int(time.time() - started)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        elapsed = int(time.time() - started)
        if elapsed >= next_update:
            publish("model-loading", cfg, elapsed_s=elapsed)
            next_update += 60
        time.sleep(5)
    raise RuntimeError(
        "llama-server did not become healthy within 20 minutes:\n"
        + redact(tail(SERVER_LOG), cfg.get("api_key") or "")
    )


def start_tunnel(cfg):
    handle = TUNNEL_LOG.open("w")
    process = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{cfg['port']}",
         "--no-autoupdate", "--protocol", "quic"],
        stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            break
        match = pattern.search(tail(TUNNEL_LOG, 12000))
        if match:
            return process, handle, match.group(0).rstrip("/")
        time.sleep(2)
    return process, handle, None


def stop_process(process, label):
    if process is None or process.poll() is not None:
        return
    log("Stopping", label)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def self_test(cfg):
    body = {
        "model": cfg["served_model_name"],
        "messages": [{"role": "user", "content": "Reply with exactly: GPU ready"}],
        "max_tokens": 64,
        "temperature": 0,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = api_request(cfg, "/v1/chat/completions", body, timeout=180)
    message = response["choices"][0]["message"]
    answer = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not answer:
        raise RuntimeError("The local API self-test returned an empty response")
    publish("self-test", cfg, answer=answer[:100])
    return answer


def active_config():
    if not isinstance(PROFILE, dict):
        raise SystemExit("llama_cpp_gpu.py was pushed without an embedded profile")
    cfg = serve_config(PROFILE, CFG if isinstance(CFG, dict) else {})
    if not cfg["api_key"]:
        cfg["api_key"] = "sk-" + secrets.token_hex(16)
    return cfg


def main():
    cfg = active_config()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    model_path = SCRATCH / cfg["model_file"]
    require_gpus(cfg)
    server_process = tunnel_process = None
    server_handle = tunnel_handle = None
    try:
        publish("setup", cfg)
        server_binary, server_env = prepare_llama_server(cfg)
        prepare_cloudflared(cfg)
        download_model(cfg, model_path)
        publish("server-starting", cfg, ctx_size=cfg["ctx_size"], parallel=cfg["parallel"],
                mtp_tokens=cfg["mtp_tokens"])
        server_handle = SERVER_LOG.open("w")
        server_process = subprocess.Popen(
            server_command(server_binary, cfg, model_path),
            env=server_env, stdout=server_handle, stderr=subprocess.STDOUT, text=True,
        )
        tunnel_process, tunnel_handle, public_url = start_tunnel(cfg)
        if public_url:
            publish("tunnel-url", cfg, endpoint=public_url + "/v1", note="wait for READY")
        else:
            publish("tunnel-failed", cfg, log=redact(tail(TUNNEL_LOG, 1000), cfg["api_key"]))
        startup_seconds = wait_for_server(server_process, cfg)
        self_test(cfg)
        endpoint = public_url + "/v1" if public_url else f"http://127.0.0.1:{cfg['port']}/v1"
        log("#" * 72)
        log("# READY", cfg["display_name"])
        log("# ENDPOINT:", endpoint)
        log("# API KEY :", cfg["api_key"])
        log("# MODEL   :", cfg["served_model_name"])
        log("#" * 72)
        publish(
            "ready", cfg,
            endpoint=endpoint if public_url else None,
            api_key=cfg["api_key"],
            model=cfg["served_model_name"],
            startup_seconds=startup_seconds,
        )
        deadline = time.time() + int(cfg["keepalive_min"]) * 60
        while time.time() < deadline:
            if server_process.poll() is not None:
                raise RuntimeError(
                    "llama-server stopped unexpectedly:\n" + redact(tail(SERVER_LOG), cfg["api_key"])
                )
            time.sleep(15)
        publish("auto-shutdown", cfg, reason="keepalive elapsed")
    except KeyboardInterrupt:
        publish("complete", cfg, reason="notebook interrupted")
    except Exception as error:
        publish("failed", cfg, error=redact(str(error), cfg.get("api_key") or "")[-2000:])
        raise
    finally:
        stop_process(tunnel_process, "Cloudflare tunnel")
        stop_process(server_process, "llama-server")
        if tunnel_handle:
            tunnel_handle.close()
        if server_handle:
            server_handle.close()


if __name__ == "__main__":
    main()
