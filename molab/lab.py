#!/usr/bin/env python3
"""Local Molab llama-server lifecycle (Linux, Python 3.11+).

Examples::
    python molab/lab.py status
    python molab/lab.py start --server /path/to/llama-server
    python molab/lab.py benchmark
    python molab/lab.py stop

Only start needs huggingface-hub (>=0.34,<1). Supply an existing CUDA build;
this CLI neither installs software nor allocates paid resources. The HF cache
is reused; use --cache-dir on persistent storage when available. State defaults
to ~/.local/state/molab-lab (private, outside the repository). API keys stay in
that directory's api.key, never in argv or reports. Use an authorized SSH
loopback forward for remote clients; this command never publishes a tunnel.
Memory admission is an estimate, not a guarantee: KV/runtime overhead depends
on the model. No automatic CPU/model fallback, MTP, or tool support is claimed.
Stop releases this server, not the Molab allocation; platform quotas still apply.
"""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
REVISION = "b9bce7e859c9b5047aca884e5bff76f8a0443de1"
GIB = 2**30
ALIAS = "molab-lab"
SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def hardware():
    """Report observations, not a presumed Molab accelerator configuration."""
    result = {"gpus": [], "ram_available_bytes": None, "cuda_toolkit": None}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["ram_available_bytes"] = int(line.split()[1]) * 1024
        # Respect a tighter container memory limit when exposed by cgroup v2.
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            result["ram_available_bytes"] = min(result["ram_available_bytes"], max(0, int(limit) - used))
    except (OSError, ValueError, TypeError):
        pass
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=10)
        for row in csv.reader(io.StringIO(output)):
            index, name, total, free, driver = (item.strip() for item in row)
            result["gpus"].append({"index": int(index), "name": name,
                                   "total_bytes": int(total) * 2**20,
                                   "free_bytes": int(free) * 2**20, "driver": driver})
    except (OSError, subprocess.SubprocessError, ValueError):
        result["gpu_probe"] = "unavailable"
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True,
                                         stderr=subprocess.DEVNULL, timeout=10)
        match = re.search(r"release ([\d.]+)", output)
        result["cuda_toolkit"] = match[1] if match else None
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def gguf_groups(siblings):
    """Return complete sets only; never load a partial split or unknown size."""
    groups = {}
    for item in siblings:
        name, size = item.rfilename, item.size
        path = PurePosixPath(name)
        if (not name.lower().endswith(".gguf") or path.is_absolute()
                or ".." in path.parts or "\\" in name or not size or size <= 0):
            continue
        match = SPLIT.match(name)
        group = (match[1], int(match[3])) if match else (name, 1)
        index = int(match[2]) if match else 1
        groups.setdefault(group, {})[index] = (name, size)
    complete = []
    for (_, count), parts in groups.items():
        if count > 0 and len(parts) == count and min(parts) == 1 and max(parts) == count:
            files = [parts[i] for i in range(1, count + 1)]
            complete.append({"files": files, "bytes": sum(size for _, size in files)})
    return sorted(complete, key=lambda item: item["bytes"])


def choose_model(groups, detected, filename=None):
    # Single-device full offload only, reserving at least 25% / 2 GiB for KV/runtime.
    gpus = detected["gpus"]
    if not gpus or detected["ram_available_bytes"] is None:
        raise RuntimeError("CUDA GPU and available RAM must be detected; no CPU fallback")
    gpu = max(gpus, key=lambda item: item["free_bytes"])
    budget = min(gpu["free_bytes"] * 0.75, gpu["free_bytes"] - 2 * GIB,
                 detected["ram_available_bytes"] - 2 * GIB)
    candidates = [g for g in groups if not filename or filename in [n for n, _ in g["files"]]]
    if not candidates:
        raise RuntimeError("No complete GGUF set matches the requested filename")
    fitting = [g for g in candidates if g["bytes"] <= budget]
    if not fitting:
        raise RuntimeError("No selected GGUF fits conservative RAM/VRAM budget; no model substitution")
    return max(fitting, key=lambda item: item["bytes"]), gpu


def download_model(args, detected):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise RuntimeError("start requires huggingface-hub>=0.34,<1") from None
    info = HfApi().model_info(REPO, revision=args.revision, files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise RuntimeError("HF did not return an immutable model revision")
    group, gpu = choose_model(gguf_groups(info.siblings), detected, args.filename)
    paths = []
    for filename, size in group["files"]:
        path = Path(hf_hub_download(REPO, filename, revision=info.sha,
                                   cache_dir=args.cache_dir))
        if path.stat().st_size != size:
            raise RuntimeError("Cached GGUF size does not match pinned HF metadata")
        paths.append(path)
    # HF snapshot layout keeps every split beside its first shard.
    if len({path.parent for path in paths}) != 1:
        raise RuntimeError("GGUF shards must share a snapshot directory")
    return paths[0], {"repository": REPO, "revision": info.sha,
                      "files": [name for name, _ in group["files"]],
                      "weight_bytes": group["bytes"], "gpu_index": gpu["index"]}


def private_dir(path):
    path = Path(path).expanduser().absolute()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    stat = path.lstat()
    if path.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise RuntimeError("State directory must be owned by this user with mode 0700")
    return path


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


@contextmanager
def locked(base):
    fd = os.open(base / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def process_identity(pid):
    try:
        # comm may contain spaces or ')'; starttime is field 22.
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def alive(state):
    return bool(state and process_identity(state["pid"]) == state["identity"])


def stop_process(state, timeout=15):
    """Use a pidfd, then recheck identity; never signal a recycled numeric PID."""
    if not alive(state):
        return False
    try:
        fd = os.pidfd_open(state["pid"])
    except ProcessLookupError:
        return False
    try:
        if not alive(state):
            return False
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        if not select.select([fd], [], [], timeout)[0]:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            if not select.select([fd], [], [], 5)[0]:
                raise RuntimeError("Server did not exit after SIGKILL")
        return True
    finally:
        os.close(fd)


def read_state(base):
    try:
        return json.loads((base / "state.json").read_text())
    except FileNotFoundError:
        return None


def api_request(state, key, route, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(state['port'])}{route}", data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    # Never let HTTP proxy configuration forward a local authenticated request.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def read_key(base):
    path = base / "api.key"
    stat = path.lstat()
    if path.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise RuntimeError("API key file must be private (0600)")
    key = path.read_text().strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", key):
        raise RuntimeError("Invalid API key file")
    return key


def server_env(gpu_index):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("LLAMA_") and k not in {
               "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY"}}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def start(args, base):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support is required for safe process management")
    if alive(read_state(base)):
        raise RuntimeError("Managed server already running; stop it explicitly first")
    binary = shutil.which(args.server)
    if not binary:
        raise RuntimeError("Supply an existing CUDA llama-server using --server")
    env = server_env(0)
    help_text = subprocess.check_output([binary, "--help"], text=True, env=env,
                                        stderr=subprocess.STDOUT, timeout=20)
    if "--api-key-file" not in help_text:
        raise RuntimeError("llama-server must support --api-key-file; refusing insecure startup")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    detected = hardware()
    started = time.monotonic()
    model_path, model = download_model(args, detected)
    # A long download may change free memory; admit the exact set again.
    _, gpu = choose_model([{"files": [(str(model_path), model["weight_bytes"])],
                            "bytes": model["weight_bytes"]}], hardware())
    model["gpu_index"] = gpu["index"]
    key_path = base / "api.key"
    if not key_path.exists():
        write_private(key_path, secrets.token_urlsafe(32) + "\n")
    key = read_key(base)
    command = [binary, "--model", str(model_path), "--alias", ALIAS,
               "--host", "127.0.0.1", "--port", str(args.port),
               "--api-key-file", str(key_path), "--ctx-size", str(args.context),
               "--n-gpu-layers", "99", "--parallel", "1", "--jinja"]
    # No server logs: some versions log requests or authentication data.
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True,
                               env=server_env(gpu["index"]))
    state = {"pid": process.pid, "identity": process_identity(process.pid),
             "port": args.port, "model": model, "context": args.context,
             "started_unix": time.time(), "startup_seconds": None}
    try:
        if state["identity"] is None:
            raise RuntimeError("Server exited before process identity was recorded")
        write_private(base / "state.json", json.dumps(state, indent=2))
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("llama-server exited during model loading")
            try:
                with api_request(state, key, "/v1/models", timeout=2) as response:
                    models = json.load(response)
                if ALIAS in [item["id"] for item in models["data"]]:
                    state["startup_seconds"] = time.monotonic() - started
                    write_private(base / "state.json", json.dumps(state, indent=2))
                    return {"running": True, "port": args.port, "model": model,
                            "startup_seconds": state["startup_seconds"]}
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.5)
        raise RuntimeError("Model startup timeout; server stopped")
    except BaseException:
        stop_process(state)
        process.wait(timeout=5)
        (base / "state.json").unlink(missing_ok=True)
        raise


def status(base):
    state = read_state(base)
    result = {"running": alive(state), "ready": False, "hardware": hardware()}
    if result["running"]:
        result.update(port=state["port"], model=state["model"],
                      startup_seconds=state["startup_seconds"])
        try:
            with api_request(state, read_key(base), "/v1/models") as response:
                data = json.load(response)
            result["ready"] = ALIAS in [item["id"] for item in data["data"]]
        except (OSError, ValueError, KeyError):
            pass
    return result


def benchmark(args, base):
    state = read_state(base)
    if not alive(state):
        raise RuntimeError("No managed server running")
    key = read_key(base)
    body = {"model": ALIAS, "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0, "max_tokens": args.tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    started = time.monotonic()
    first, usage, done = None, {}, False
    # Deliberately retain only numeric timing/usage, never prompts or completions.
    with api_request(state, key, "/v1/chat/completions", body, args.timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Server did not return SSE")
        for line in response:
            if time.monotonic() - started > args.timeout:
                raise RuntimeError("Benchmark total timeout exceeded")
            if len(line) > 2**20:
                raise RuntimeError("SSE event exceeds limit")
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError("Server returned an SSE error")
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content")
                                      or delta.get("tool_calls")):
                    first = time.monotonic() - started
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = (event.get("usage") or {}).get(field)
                if type(value) is int and value >= 0:
                    usage[field] = value
    elapsed = time.monotonic() - started
    if not done:
        raise RuntimeError("Truncated SSE stream (no DONE); no benchmark recorded")
    count = usage.get("completion_tokens")
    report = {"schema": 1, "measured_unix": time.time(),
              "model": state["model"], "context": state["context"],
              "startup_seconds": state["startup_seconds"], "wall_seconds": elapsed,
              "ttft_seconds": first, "usage": usage,
              "completion_tokens_per_wall_second": count / elapsed if count is not None else None,
              "note": "End-to-end rate includes prompt processing and network overhead; not decode throughput. "
                      "Missing usage stays null. No GPU peak, MTP, or tool validation inferred."}
    write_private(base / "benchmark.json", json.dumps(report, indent=2))
    lines = ["# Molab benchmark", "", "Real authenticated loopback SSE request.", "",
             f"- Wall seconds: {elapsed:.6f}", f"- TTFT seconds: {first}",
             f"- Completion tokens: {count}",
             f"- End-to-end tokens/s: {report['completion_tokens_per_wall_second']}",
             "", report["note"], ""]
    write_private(base / "benchmark.md", "\n".join(lines))
    return report


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/molab-lab")
    commands = parser.add_subparsers(dest="action", required=True)
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--server", default="llama-server")
    start_parser.add_argument("--revision", default=REVISION)
    start_parser.add_argument("--filename", help="Exact GGUF or shard name; never replaced if it does not fit")
    start_parser.add_argument("--cache-dir", type=Path)
    start_parser.add_argument("--context", type=positive, default=4096)
    start_parser.add_argument("--port", type=positive, default=8080)
    start_parser.add_argument("--startup-timeout", type=positive, default=600)
    commands.add_parser("status")
    commands.add_parser("stop")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--prompt", default="Write a short Python function to add two integers.")
    bench.add_argument("--tokens", type=positive, default=128)
    bench.add_argument("--timeout", type=positive, default=300)
    args = parser.parse_args(argv)
    try:
        base = private_dir(args.state_dir)
        with locked(base):
            if args.action == "start":
                if args.port > 65535:
                    raise RuntimeError("Invalid TCP port")
                result = start(args, base)
            elif args.action == "status":
                result = status(base)
            elif args.action == "stop":
                stopped = stop_process(read_state(base))
                (base / "state.json").unlink(missing_ok=True)
                result = {"stopped": stopped}
            else:
                result = benchmark(args, base)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        # External exception messages may embed credentials, URLs or prompt data.
        print(f"Operation failed ({type(error).__name__}); check configuration, local runtime and hardware.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
