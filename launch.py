#!/usr/bin/env python3
"""
Lanzador de kaggle-tpu-lab: levanta un modelo en una TPU gratuita de Kaggle.

    python launch.py serve                          # Qwen3.8-27B
    python launch.py serve --model glm53-flash      # GLM-5.3-Flash
    python launch.py serve --reasoning-effort medium --mtp 3
    python launch.py status                         # estado + eventos recientes
    python launch.py stop                           # detener la sesión TPU

Requiere Kaggle CLI autenticado: pip install kaggle
Este lanzador usa solamente la biblioteca estándar de Python.
"""
import argparse
import base64
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "qwen38-27b" / "kernel" / "serve_qwen38.py"
STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"

WEIGHTS_DATASET = "rahim3/qwen3-8-27b-bf16"
ENV_DATASET = "rahim3/qwen38-tpu-env-v5e8"   # XLA compile cache + cloudflared + manifest
GLM_DATASETS = ["rahim3/glm53-flash-iq3xxs-1", "rahim3/glm53-flash-iq3xxs-2",
                "rahim3/glm53-flash-fp8-1", "rahim3/glm53-flash-fp8-2", "rahim3/glm53-flash-fp8-3", "rahim3/glm53-flash-fp8-4"]

# One entry per model folder: the kernel script, the default kernel name, and (for our own engine) the package to embed.
MODELS = {
    "qwen38-27b": {"kernel": KERNEL_SRC, "slug": "qwen38-tpu-serve", "model_name": "qwen3.8-27b", "minutes": 22},
    "glm53-flash": {"kernel": HERE / "glm53-flash" / "kernel" / "serve_glm53.py", "slug": "glm53-tpu-serve",
                    "model_name": "glm-5.3-flash", "engine": HERE / "glm53-flash" / "engine" / "glm53", "minutes": 25},
}

# Mensajes breves para las fases publicadas por los kernels.
PHASE_TEXT = {
    "install":            "Preparando el entorno Python con uv (~30 s)...",
    "installed":          "Entorno listo.",
    "mtp-patch-applied":  "Parche de rollback MTP aplicado.",
    "mtp-patch-failed":   "No se pudo aplicar MTP; se desactiva por seguridad.",
    "mtp-disabled":       "Checkpoint sin cabeza MTP; se sirve sin decodificación especulativa.",
    "cache-restored":     None,
    "cache-missing":      "No hay caché XLA; la primera compilación demorará más.",
    "weights-mounted":    "Pesos montados; no hace falta descargarlos.",
    "weights-download":   "Descargando los pesos desde Hugging Face...",
    "weights-downloaded": "Pesos descargados.",
    "server-launch":      "Iniciando el motor de inferencia y cargando el modelo...",
    "loading":            "Cargando los pesos en la TPU...",
    "loaded":             None,
    "warmed":             None,
    "tunnel-url":         None,
    "compiling":          None,
    "serving":            "Servidor saludable.",
    "benchmark":          None,
    "ready":              None,
    "heartbeat":          None,
    "failed":             None,
    "auto-shutdown":      "Tiempo máximo cumplido; sesión detenida correctamente.",
    "stopped":            "El servidor se detuvo de forma inesperada.",
}


def kaggle(*args, capture=True, input=None):
    exe = shutil.which("kaggle")
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True, input=input)
    return r


def say(msg):
    print(time.strftime("[%H:%M] "), msg, flush=True)


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit("Kaggle CLI no funciona o no está autenticado.\n"
                 "Instalalo con `pip install kaggle` y configurá tu token de API.\n"
                 "(Kaggle -> Settings -> Create New Token).\n\n"
                 f"Error:\n{(r.stderr or r.stdout).strip()}")


def kaggle_username(cli_arg):
    if cli_arg:
        return cli_arg
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    sys.exit("No pude detectar tu usuario de Kaggle; pasalo con --user <nombre>.")


def write_state(payload):
    """Atomic replace of the launcher state file, mode 0600."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".kaggle-tpu-lab-", dir=str(STATE_FILE.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, STATE_FILE)
        os.chmod(STATE_FILE, 0o600)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def models_url(endpoint):
    """OpenAI models URL. Accepts a bare origin or an origin that already ends in /v1."""
    root = (endpoint or "").strip().rstrip("/")
    if not root:
        return ""
    base = root if root.endswith("/v1") else root + "/v1"
    return base + "/models"


def probe_endpoint(endpoint, api_key, timeout=8):
    """True when GET /v1/models returns 200. The key is not logged."""
    url = models_url(endpoint)
    if not url or not api_key:
        return False
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def push_ok(result):
    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "successfully pushed" not in out:
        sys.exit(f"Falló el envío:\n{out.strip()}")
    return out


def engine_b64(pkg_dir):
    """The engine package (its .py files) as a base64 tar.gz, embedded into the kernel script."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(pkg_dir.glob("*.py")):
            tf.add(f, arcname=f"{pkg_dir.name}/{f.name}")
    return base64.b64encode(buf.getvalue()).decode()


GPU_KERNEL = HERE / "qwen38-27b" / "gpu" / "serve_qwen38_gpu.py"


def cmd_serve_gpu(args, user):
    """Push the dual-T4 llama.cpp recipe. Does not touch the TPU kernel."""
    if args.model != "qwen38-27b":
        sys.exit("El acelerador gpu solo está implementado para qwen38-27b.")
    slug = args.slug or "qwen38-t4x2-serve"
    topic = "ktl-" + uuid.uuid4().hex[:20]
    api_key = "sk-" + secrets.token_hex(16)
    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "keepalive_min": args.keepalive_min,
        "ctx_size": min(args.max_model_len, 32768),
    }
    src = GPU_KERNEL.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{GPU_KERNEL} is missing the __LAUNCHER_CONFIG__ line")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / GPU_KERNEL.name).write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": GPU_KERNEL.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Enviando kernel {user}/{slug} (GPU T4, cuota distinta de la TPU)...")
        push_ok(kaggle("kernels", "push", "-p", str(td)))
    write_state({"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key,
                 "accelerator": "gpu", "model": "qwen38-27b"})
    say("Enviado. El script exige dos T4; si Kaggle asigna otra GPU, termina antes de descargar el modelo.")
    say("Siguiendo el progreso. Ctrl-C no detiene la sesión.")
    watch(f"{user}/{slug}", topic)


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    if getattr(args, "accelerator", "tpu") == "gpu":
        cmd_serve_gpu(args, user)
        return
    model = MODELS[args.model]
    slug = args.slug or model["slug"]
    topic = "ktl-" + uuid.uuid4().hex[:20]
    api_key = ("glm-" if args.model == "glm53-flash" else "sk-") + secrets.token_hex(16)

    if args.model == "qwen38-27b":
        cfg = {
            "ntfy_topic": topic,
            "api_key": api_key,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "mtp_tokens": args.mtp,
            "reasoning_effort_default": args.reasoning_effort,
            "keepalive_min": args.keepalive_min,
            "weights_dataset": args.weights_dataset,
        }
        if args.no_tools:
            cfg["tool_call_parser"] = ""
        if args.text_only:
            cfg["text_only"] = True
        if args.verbose:
            cfg["verbose"] = True
        if args.fast_start:
            cfg["fast_start"] = True
        if args.no_async_scheduling:
            cfg["async_scheduling"] = False
        datasets = [args.weights_dataset, ENV_DATASET]
    else:
        cfg = {
            "ntfy_topic": topic,
            "api_key": api_key,
            "max_len": args.max_len,
            "streams": args.streams,
            "reasoning_effort_default": args.reasoning_effort if args.reasoning_effort in ("low", "medium", "high") else "low",
            "keepalive_min": args.keepalive_min,
            "vision": not args.text_only,
        }
        if args.serve_dataset:
            cfg["serve_dataset"] = args.serve_dataset
            datasets = GLM_DATASETS[:2] + [args.serve_dataset]           # experts + the serve dataset; no FP8 mounts
        else:
            datasets = GLM_DATASETS

    src = model["kernel"].read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{model['kernel']} is missing the __LAUNCHER_CONFIG__ line")
    if model.get("engine"):
        src, n = re.subn(r'^ENGINE_B64 = ""  # __ENGINE__.*$', f'ENGINE_B64 = "{engine_b64(model["engine"])}"',
                         src, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"{model['kernel']} is missing the __ENGINE__ line")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / model["kernel"].name).write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": model["kernel"].name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "machine_shape": "TpuV5E8",
            "dataset_sources": datasets,
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Enviando kernel {user}/{slug} (TPU v5e-8)...")
        out = push_ok(kaggle("kernels", "push", "-p", str(td)))
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"AVISO: {line.strip()} — el kernel seguirá, pero puede tener que descargar pesos o compilar en frío.")

    write_state({"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key,
                 "accelerator": "tpu", "model": args.model})
    say("Enviado. Kaggle puede demorar unos minutos en asignar la TPU y montar los datos; "
        f"el endpoint suele estar listo ~{model['minutes']} min después de arrancar el kernel.")
    say("Siguiendo el progreso. Ctrl-C es seguro: el servidor sigue activo; "
        "`python launch.py status` reconecta y `python launch.py stop` lo detiene.")
    watch(f"{user}/{slug}", topic)


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            events.append((e["time"], json.loads(e.get("message", "{}"))))
        except Exception:
            continue
    return events


def render_event(ev):
    phase = ev.get("phase", "?")
    mensaje = ev.get("message_es")
    if phase == "compiling":
        if "what" in ev:
            say(f"Compilado {ev['what']} en {ev.get('secs', 0)} s")
        else:
            say(mensaje or
                f"Cargando / compilando... {ev.get('elapsed_s', 0) // 60} min transcurridos")
    elif phase == "loaded":
        say(mensaje or
            f"Pesos cargados en los chips tras {ev.get('minutes', '?')} min "
            f"(HBM {ev.get('hbm_gb', '?')} GB por chip); calentando...")
    elif phase == "warmed":
        say(mensaje or f"Calentamiento terminado en {ev.get('minutes', '?')} min; abriendo el túnel...")
    elif phase == "cache-restored":
        if ev.get("covers_this_config", True):
            say(mensaje or "Caché XLA restaurada para esta configuración; inicio rápido.")
        else:
            say("Caché XLA restaurada, pero no cubre esta configuración; habrá compilación en frío.")
    elif phase == "tunnel-url":
        say(mensaje or f"Endpoint reservado: {ev.get('endpoint')} (todavía no está listo)")
    elif phase == "serving":
        say(mensaje or f"Servidor saludable tras {ev.get('startup_secs', 0) // 60} min.")
    elif phase == "benchmark":
        say(mensaje or
            f"Benchmark rápido: {ev.get('decode_tok_s', '?')} tok/s "
            f"(comprobación: {ev.get('sanity', '')!r})")
    elif phase == "ready":
        print("\n" + "=" * 66)
        print("  ENDPOINT LISTO")
        print(f"  URL base : {ev['endpoint']}")
        print(f"  API key  : {ev['api_key']}")
        print(f"  modelo   : {ev['model']}   (contexto: {ev.get('max_model_len', '?')})")
        print("=" * 66)
        base = ev["endpoint"] if ev["endpoint"].endswith("/v1") else ev["endpoint"] + "/v1"
        print(f"""
Prueba rápida:
  curl {base}/chat/completions -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" -d '{{
      "model": "{ev['model']}",
      "messages": [{{"role": "user", "content": "Hola"}}],
      "chat_template_kwargs": {{"reasoning_effort": "low"}}
    }}'
""")
        say(f"La sesión seguirá sirviendo hasta {ev.get('keepalive_min', '?')} min. "
            "Ctrl-C no la detiene; usá `python launch.py stop`.")
    elif phase == "heartbeat":
        say(mensaje or f"Servicio activo ({ev.get('up_min', '?')} min) — {ev.get('endpoint', '')}")
    elif phase == "stopped":
        say(mensaje or "El servidor se detuvo de forma inesperada.")
        if ev.get("cause"):
            say(f"Causa: {ev['cause']}")
        if ev.get("hint_es"):
            say(f"Sugerencia: {ev['hint_es']}")
        elif ev.get("hint"):
            say(f"Sugerencia: {ev['hint']}")
        if ev.get("tail"):
            print("--- detalle del error ---")
            print(ev["tail"])
    elif phase == "failed":
        say(mensaje or f"FALLO en la etapa {ev.get('step', '?')}.")
        if ev.get("error_code"):
            say(f"Código: {ev['error_code']}")
        if ev.get("step") == "no-tpu":
            say("La sesión arrancó sin una TPU utilizable. Verificá que la cuenta tenga acceso a TPU "
                "y volvé a lanzar la instancia.")
        if ev.get("cause"):
            say(f"Causa: {ev['cause']}")
        if ev.get("hint_es"):
            say(f"Sugerencia: {ev['hint_es']}")
        elif ev.get("hint"):
            say(f"Sugerencia: {ev['hint']}")
        if ev.get("tail"):
            print("--- detalle del error ---")
            print(ev["tail"])
        say("El log completo queda disponible en la página del kernel de Kaggle.")
    else:
        text = mensaje or PHASE_TEXT.get(phase)
        say(text if text else f"{phase} {json.dumps({k: v for k, v in ev.items() if k != 'phase'}, ensure_ascii=False)}")

def watch(kernel, topic):
    since = int(time.time()) - 600
    last_status = None
    seen_boot = False
    try:
        while True:
            for ts, ev in read_events(topic, since):
                since = max(since, ts)
                seen_boot = True
                render_event(ev)
                if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    return
            since = max(since, int(time.time()) - 1) if seen_boot else since
            r = kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            status = m.group(1) if m else "UNKNOWN"
            if status != last_status:
                if status == "QUEUED":
                    say("Kaggle: en cola, esperando una TPU v5e-8...")
                elif status == "RUNNING" and not seen_boot:
                    say("Kaggle: TPU asignada; preparando la VM y montando los datos...")
                elif status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
                    say(f"Kernel finalizado con estado {status}.")
                    return
                last_status = status
            time.sleep(30)
    except KeyboardInterrupt:
        say("Seguimiento desconectado. La sesión sigue activa; usá `python launch.py status` "
            "para reconectar o `python launch.py stop` para detenerla.")

def cmd_build_env(args):
    """Maintainer flow. When the kernel finishes:
        kaggle kernels output <user>/<slug> -p bundle_out
        then create/version the dataset from bundle_out/bundle (see README)."""
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-" + uuid.uuid4().hex[:20]
    cfg = {"build_bundle": True, "ntfy_topic": topic, "weights_dataset": args.weights_dataset}
    src = KERNEL_SRC.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("qwen38-27b/kernel/serve_qwen38.py is missing the __LAUNCHER_CONFIG__ line")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "build_env.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "build_env.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "false", "enable_tpu": "true", "enable_internet": "true",
            "machine_shape": "TpuV5E8",
            "dataset_sources": [args.weights_dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out or r.returncode != 0:
            sys.exit(f"Push failed:\n{out.strip()}")
    write_state({"kernel": f"{user}/{args.slug}", "topic": topic, "api_key": ""})
    say(f"Pushed {user}/{args.slug}. It serves each config once (~1.5 h total) and "
        "leaves xla_cache.tar / cloudflared / manifest.json in its output.")
    watch(f"{user}/{args.slug}", topic)


def load_state():
    if not STATE_FILE.exists():
        sys.exit("No hay estado de lanzamiento guardado; ejecutá `python launch.py serve` primero.")
    return json.loads(STATE_FILE.read_text())


def cmd_status(args):
    st = load_state()
    say(f"Kernel: {st['kernel']}")
    r = kaggle("kernels", "status", st["kernel"])
    say(((r.stdout or "") + (r.stderr or "")).strip())
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    for _, ev in events[-8:]:
        render_event(ev)
    if any(ev.get("phase") == "ready" for _, ev in events):
        say(f"API key: {st['api_key']}")
        endpoint = next((ev.get("endpoint") for _, ev in reversed(events) if ev.get("endpoint")), None)
        if endpoint:
            live = probe_endpoint(endpoint, st.get("api_key", ""))
            say("Sonda /v1/models: TPU lista." if live else
                "Sonda /v1/models: Sin conexión (el evento ready no alcanza para afirmar que el túnel responde).")
    if args.follow:
        watch(st["kernel"], st["topic"])


def cmd_stop(args):
    st = load_state()
    say(f"Eliminando kernel {st['kernel']} (esto detiene la sesión)...")
    p = kaggle("kernels", "delete", st["kernel"], input="yes\n")
    say((p.stdout + p.stderr).strip() or "listo")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="enviar el kernel y seguir su arranque")
    s.add_argument("--model", default="qwen38-27b", choices=sorted(MODELS), help="which recipe (model folder) to serve")
    s.add_argument("--accelerator", default="tpu", choices=["tpu", "gpu"],
                   help="tpu (default, v5e-8) or gpu (Qwen Q4 on dual T4, separate quota)")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default=None, help="kernel name (default: the model's)")
    s.add_argument("--max-len", type=int, default=262144, help="glm53-flash: context capacity (a multiple of 32)")
    s.add_argument("--streams", type=int, default=4, help="glm53-flash: requests decoded together")
    s.add_argument("--serve-dataset", default="rahim3/glm53-flash-serve",
                   help="glm53-flash: dataset with base/ + jax_cache/ (replaces the FP8 mounts, shorter warm-up); '' = FP8 mounts, cold")
    s.add_argument("--max-model-len", type=int, default=262144,
                   help="context length (default: native 262k; use 131072 with "
                        "--max-num-seqs 16 for max multi-stream throughput)")
    s.add_argument("--max-num-seqs", type=int, default=4)
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (0 disables). +34%% decode in our A/B test; made "
                        "lossless by the bundled GDN state-rollback patch "
                        "(verified 12/12 greedy exact-match)")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "high", "medium", "low"],
                   help="server-side default; clients can still override per request "
                        "(qwen38-27b: xhigh | medium | low; glm53-flash: low | medium | high, default low)")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower (Qwen: ~8 min faster start; GLM: ~1 min); image inputs "
                        "then error out")
    s.add_argument("--verbose", action="store_true",
                   help="show every vLLM log line in the kernel log")
    s.add_argument("--no-async-scheduling", action="store_true",
                   help="qwen38-27b: pass --no-async-scheduling to vLLM. Needed when clients use JSON mode / "
                        "structured outputs with MTP on (vllm-tpu 0.28.0 otherwise exits with AttributeError: "
                        "__delitem__); costs some throughput")
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile: endpoint live in ~4 min (with the env "
                        "dataset), common request shapes are warmed right after; an "
                        "unusual request shape stalls ~1 min the first time")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                       "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.set_defaults(fn=cmd_build_env)

    s = sub.add_parser("status", help="mostrar estado actual + eventos recientes")
    s.add_argument("--follow", "-f", action="store_true", help="seguir monitoreando")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("stop", help="detener la sesión TPU")
    s.set_defaults(fn=cmd_stop)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
