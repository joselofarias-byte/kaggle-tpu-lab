# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.34,<1"]
# ///
"""GGUF experiment for Molab. Run with uv; binds inference to loopback only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

REPO = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
REVISION = "b9bce7e859c9b5047aca884e5bff76f8a0443de1"
FILENAME = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-Q8_0.gguf"
LLAMA_COMMIT = "cd959bae99ff4bed22bef2604f45241c09bbb8c8"
LLAMA_BUILD = "b12110"


def run(args, **kwargs):
    print("RUN", args, flush=True)
    return subprocess.run(args, check=True, **kwargs)


def install_runtime(base):
    binary = base / "official-prebuilt/llama"
    if not binary.is_file():
        with urllib.request.urlopen("https://llama.app/install.sh", timeout=60) as response:
            script = response.read().decode()
        # Keep all installer files inside this experiment, and skip global install.
        script = script.replace("~/.llama-app", str(base / "official-prebuilt"))
        installer = base / "install-official.sh"
        installer.write_text(script)
        env = os.environ.copy()
        env.update(SKIP_INSTALL="1", LLAMA_VERSION=LLAMA_BUILD)
        with (base / "prebuilt-install.log").open("w") as log:
            run(["sh", str(installer)], env=env, stdout=log, stderr=subprocess.STDOUT)
    version = subprocess.check_output([str(binary), "version"], text=True).strip()
    if LLAMA_COMMIT[:9] not in version:
        raise RuntimeError("Unexpected official runtime revision: " + version)
    devices = subprocess.check_output([str(binary), "serve", "--list-devices"], text=True)
    if "CUDA0" not in devices:
        raise RuntimeError("Official binary did not detect CUDA: " + devices)
    return version, devices


def prepare(base):
    from huggingface_hub import hf_hub_download
    base.mkdir(parents=True, exist_ok=True)
    version, devices = install_runtime(base)
    t0 = time.monotonic()
    path = hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=base / "models")
    metadata = {"repository": REPO, "revision": REVISION, "filename": FILENAME, "path": path,
                "bytes": Path(path).stat().st_size, "download_seconds": time.monotonic() - t0,
                "llama_commit": LLAMA_COMMIT, "runtime_version": version, "devices": devices}
    (base / "model.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def model_path(base):
    path = base / "models" / FILENAME
    if not path.is_file():
        raise FileNotFoundError("Run prepare first: " + str(path))
    return path


def server_command(base, port):
    return [str(base / "official-prebuilt/llama"), "serve", "-m", str(model_path(base)),
            "-ngl", "99", "-c", "8192", "-np", "1", "-fa", "on", "--jinja",
            "--reasoning", "off", "--host", "127.0.0.1", "--port", str(port)]


def request(url, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def test(base, port, prompt, tokens):
    env = os.environ.copy()
    url = f"http://127.0.0.1:{port}"
    results = {}
    with (base / "server.log").open("w") as log:
        process = subprocess.Popen(server_command(base, port), env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}. See {base / 'server.log'}")
                try:
                    request(url + "/health", timeout=2)
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError("Server startup exceeded 180 seconds")
            samples = []
            stop_sampling = threading.Event()
            def sample_memory():
                while not stop_sampling.is_set():
                    try:
                        value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                        samples.append(int(value.strip().splitlines()[0]))
                    except (subprocess.SubprocessError, ValueError):
                        pass
                    stop_sampling.wait(0.25)
            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            t0 = time.monotonic()
            try:
                response = request(url + "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
                    "temperature": 0, "seed": 0, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
                })
                wall_seconds = time.monotonic() - t0
            finally:
                stop_sampling.set()
                sampler.join(timeout=5)
            results = {"wall_seconds": wall_seconds, "response": response,
                       "gpu_total_used_peak_mib": max(samples) if samples else None,
                       "gpu_memory_note": "Total GPU usage sampled every 250 ms during the request, including other resident processes; not allocator tensor peak.",
                       "gpu_after": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True).strip()}
            (base / "inference.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(json.dumps(results, indent=2, ensure_ascii=False))
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "test", "serve", "bench"])
    parser.add_argument("--base", type=Path, default=Path("davidau-gguf-lab"))
    parser.add_argument("--port", type=int, default=18992)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Respondé en español en una oración: ¿qué es una API REST?")
    args = parser.parse_args()
    base = args.base.resolve()
    if args.action == "prepare":
        prepare(base)
    elif args.action == "test":
        test(base, args.port, args.prompt, args.tokens)
    elif args.action == "serve":
        run(server_command(base, args.port))
    else:
        command = [str(base / "official-prebuilt/llama"), "bench", "-m", str(model_path(base)),
                   "-ngl", "99", "-p", "128", "-n", "64", "-r", "3", "-o", "json"]
        with (base / "bench.json").open("w") as output:
            run(command, stdout=output)


if __name__ == "__main__":
    main()
