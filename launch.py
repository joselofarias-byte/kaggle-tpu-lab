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
import signal
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


def kaggle(*args, capture=True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
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


def engine_b64(pkg_dir):
    """The engine package (its .py files) as a base64 tar.gz, embedded into the kernel script."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(pkg_dir.glob("*.py")):
            tf.add(f, arcname=f"{pkg_dir.name}/{f.name}")
    return base64.b64encode(buf.getvalue()).decode()


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
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
            "dataset_sources": datasets,
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Enviando kernel {user}/{slug} (TPU v5e-8)...")
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Falló el envío:\n{out.strip()}")
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"AVISO: {line.strip()} — el kernel seguirá, pero puede tener que descargar pesos o compilar en frío.")

    STATE_FILE.write_text(json.dumps(
        {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key}))
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
        endpoint = ev.get("endpoint")
        print("\n" + "=" * 66)
        print("  ENDPOINT LISTO" if endpoint else "  MODELO LISTO - TUNEL NO DISPONIBLE")
        print(f"  URL base : {endpoint or 'NO DISPONIBLE'}")
        print(f"  API key  : {ev['api_key']}")
        print(f"  modelo   : {ev['model']}   (contexto: {ev.get('max_model_len', '?')})")
        print("=" * 66)
        if endpoint:
            base = endpoint if endpoint.endswith("/v1") else endpoint + "/v1"
            print(f"""
Prueba rápida:
  KEY="{ev['api_key']}" curl {base}/chat/completions -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" -d '{{
      "model": "{ev['model']}",
      "messages": [{{"role": "user", "content": "Hola"}}],
      "chat_template_kwargs": {{"reasoning_effort": "low"}}
    }}'
""")
        else:
            say("Qwen está sirviendo dentro de Kaggle, pero no hay URL pública. "
                "El launcher no se cerrará por este estado; revisá el evento tunnel-failed.")
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
            "dataset_sources": [args.weights_dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
    STATE_FILE.write_text(json.dumps({"kernel": f"{user}/{args.slug}", "topic": topic,
                                      "api_key": ""}))
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
    if args.follow:
        watch(st["kernel"], st["topic"])


def cmd_stop(args):
    st = load_state()
    say(f"Eliminando kernel {st['kernel']} (esto detiene la sesión TPU)...")
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "delete",
                        st["kernel"]], input="yes\n", capture_output=True, text=True)
    say((p.stdout + p.stderr).strip() or "listo")



def cmd_prompt(args):
    """Enviar un prompt al endpoint activo. El argumento puede ser texto o una ruta."""
    st = load_state()
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    ready = next((ev for _, ev in reversed(events)
                  if ev.get("phase") == "ready" and ev.get("endpoint")), None)
    if not ready:
        sys.exit("No hay un endpoint publico disponible. Ejecuta `python launch.py status` "
                 "y confirma que URL base no sea NO DISPONIBLE.")

    source = args.prompt
    path = Path(source).expanduser()
    if path.is_file():
        prompt = path.read_text()
        say(f"Prompt cargado desde {path} ({len(prompt)} caracteres).")
    else:
        prompt = source

    endpoint = ready["endpoint"].rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    payload = {
        "model": ready.get("model", "qwen3.8-27b"),
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
    }
    req = urllib.request.Request(
        endpoint + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {st['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        sys.exit(f"Fallo enviando el prompt: {e}")

    try:
        message = data["choices"][0]["message"]
        answer = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    except Exception:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if args.show_reasoning and reasoning:
        print("\n--- RAZONAMIENTO ---\n")
        print(reasoning)
    print("\n--- RESPUESTA ---\n")
    print(answer)

    out = Path.home() / "storage" / "downloads" / "QWEN_ULTIMA_RESPUESTA.txt"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(answer)
        say(f"Respuesta guardada en {out}")
    except Exception as e:
        say(f"No pude guardar la respuesta en Descargas: {e}")



def cmd_repo_audit(args):
    """Auditar un repositorio local con Qwen usando herramientas de solo lectura."""
    st = load_state()
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    ready = next((ev for _, ev in reversed(events)
                  if ev.get("phase") == "ready" and ev.get("endpoint")), None)
    if not ready:
        sys.exit("No hay endpoint publico disponible. Inicia Qwen y confirma una URL real "
                 "con `python launch.py status`.")

    root = Path(args.repo).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"No existe el repositorio: {root}")

    excluded_dirs = {
        ".git", "node_modules", "vendor", ".venv", "venv", "__pycache__",
        "build", "dist", "target", ".gradle", ".idea", ".cache", "coverage",
        ".next", ".turbo", ".pytest_cache",
    }
    sensitive_exact = {
        ".env", "kaggle.json", "auth.json", "credentials.json", "secrets.json",
        "id_rsa", "id_ed25519", ".npmrc", ".pypirc",
    }
    binary_suffixes = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
        ".tgz", ".tar", ".7z", ".rar", ".apk", ".aab", ".so", ".dll", ".exe",
        ".bin", ".onnx", ".safetensors", ".gguf", ".pt", ".pth", ".class", ".jar",
        ".keystore", ".jks", ".p12", ".pfx", ".db", ".sqlite", ".sqlite3",
    }

    def relative_path(p):
        try:
            return p.resolve().relative_to(root)
        except Exception:
            return None

    def allowed_file(p):
        rel = relative_path(p)
        if rel is None or not p.is_file():
            return False
        if any(part in excluded_dirs for part in rel.parts):
            return False
        low = p.name.lower()
        if low in sensitive_exact or low.startswith(".env."):
            return False
        if p.suffix.lower() in binary_suffixes:
            return False
        return True

    def repo_files():
        files = []
        for p in root.rglob("*"):
            if allowed_file(p):
                files.append(p)
        files.sort(key=lambda x: str(relative_path(x)).lower())
        return files

    def tool_list_files(arguments):
        pattern = str(arguments.get("pattern", "")).strip().lower()
        rows = []
        for p in repo_files():
            rel = str(relative_path(p))
            if pattern and pattern not in rel.lower():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            rows.append(f"{rel}\t{size} bytes")
            if len(rows) >= 4000:
                rows.append("... listado truncado a 4000 archivos ...")
                break
        return "\n".join(rows) or "(sin coincidencias)"

    def tool_read_file(arguments):
        rel_arg = str(arguments.get("path", "")).strip()
        p = (root / rel_arg).resolve()
        if not allowed_file(p):
            return "DENEGADO: archivo inexistente, binario, fuera del repo o potencialmente sensible."
        start = max(1, int(arguments.get("start_line", 1) or 1))
        end_arg = arguments.get("end_line")
        end = int(end_arg) if end_arg not in (None, "") else start + 999
        end = min(end, start + 1999)
        try:
            text = p.read_text(errors="replace")
        except Exception as e:
            return f"ERROR leyendo {rel_arg}: {e}"
        lines = text.splitlines()
        selected = lines[start - 1:end]
        body = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=start))
        if len(body) > 120000:
            body = body[:120000] + "\n... contenido truncado ..."
        return body or "(archivo vacio o rango sin contenido)"

    def tool_search(arguments):
        query = str(arguments.get("query", ""))
        path_filter = str(arguments.get("path_filter", "")).strip().lower()
        if not query:
            return "ERROR: query vacia."
        regex_mode = bool(arguments.get("regex", False))
        try:
            rx = re.compile(query, re.I) if regex_mode else None
        except re.error as e:
            return f"ERROR regex: {e}"
        hits = []
        for p in repo_files():
            rel = str(relative_path(p))
            if path_filter and path_filter not in rel.lower():
                continue
            try:
                lines = p.read_text(errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                matched = bool(rx.search(line)) if rx else query.lower() in line.lower()
                if matched:
                    snippet = line.strip()
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "..."
                    hits.append(f"{rel}:{i}: {snippet}")
                    if len(hits) >= 250:
                        hits.append("... busqueda truncada a 250 coincidencias ...")
                        return "\n".join(hits)
        return "\n".join(hits) or "(sin coincidencias)"

    def git_readonly(*git_args):
        try:
            p = subprocess.run(
                ["git", "-C", str(root), *git_args],
                capture_output=True, text=True, timeout=30)
            out = (p.stdout or "") + (p.stderr or "")
            return out[:120000] or "(sin salida)"
        except Exception as e:
            return f"ERROR git: {e}"

    tool_handlers = {
        "list_files": tool_list_files,
        "read_file": tool_read_file,
        "search": tool_search,
        "git_status": lambda a: git_readonly("status", "--short", "--branch"),
        "git_diff": lambda a: git_readonly("diff", "--no-ext-diff", "--unified=3"),
        "git_log": lambda a: git_readonly("log", "--oneline", "--decorate", "-n",
                                          str(min(max(int(a.get("count", 20)), 1), 100))),
    }

    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List readable source/text files in the repository. Secrets, binaries and build directories are excluded.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Optional substring to filter paths."}
                    }
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text/source file from the repository with line numbers. Read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search text across readable repository files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path_filter": {"type": "string"},
                        "regex": {"type": "boolean"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Show git status. Read-only.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Show current uncommitted git diff. Read-only.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "Show recent commit summaries. Read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"}
                    }
                },
            },
        },
    ]

    if args.instructions:
        ip = Path(args.instructions).expanduser()
        if not ip.is_file():
            sys.exit(f"No existe el archivo de instrucciones: {ip}")
        objective = ip.read_text(errors="replace")
    else:
        objective = (
            "Realiza una auditoria exhaustiva de este repositorio. Recorre el codigo con las "
            "herramientas disponibles antes de concluir. Busca errores de logica, fallos de "
            "seguridad, problemas de concurrencia/estado, manejo de errores deficiente, "
            "regresiones, codigo muerto, dependencias fragiles y riesgos de mantenimiento. "
            "Distingue hallazgos confirmados de hipotesis. Para cada hallazgo importante cita "
            "archivo y lineas, severidad, impacto y una correccion concreta. No modifiques nada."
        )

    endpoint = ready["endpoint"].rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    model_name = ready.get("model", "qwen3.8-27b")

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un auditor senior de software. Tienes herramientas estrictamente de solo "
                "lectura para inspeccionar un repositorio local. Debes usarlas activamente y no "
                "suponer el contenido de archivos que no hayas leido. No solicites secretos. "
                "No propongas hallazgos sin evidencia concreta. Al terminar entrega un informe "
                "Markdown priorizado, con referencias archivo:linea."
            ),
        },
        {
            "role": "user",
            "content": f"Repositorio local: {root.name}\n\nObjetivo:\n{objective}",
        },
    ]

    def call_model():
        payload = {
            "model": model_name,
            "messages": messages,
            "tools": tool_specs,
            "tool_choice": "auto",
            "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
        }
        req = urllib.request.Request(
            endpoint + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {st['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            return json.loads(r.read().decode())

    say(f"Auditoria Qwen en modo SOLO LECTURA: {root}")
    say("Secretos, binarios y directorios de build quedan excluidos automaticamente.")
    final_text = ""
    for round_no in range(1, args.max_rounds + 1):
        try:
            data = call_model()
        except Exception as e:
            sys.exit(f"Fallo consultando Qwen en ronda {round_no}: {e}")
        try:
            msg = data["choices"][0]["message"]
        except Exception:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            sys.exit("Respuesta inesperada del endpoint.")

        tool_calls = msg.get("tool_calls") or []
        assistant_msg = {
            "role": "assistant",
            "content": msg.get("content") or "",
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            final_text = msg.get("content") or msg.get("reasoning_content") or ""
            break

        say(f"Ronda {round_no}: Qwen solicito {len(tool_calls)} lectura(s)/busqueda(s).")
        for tc in tool_calls:
            fn = (tc.get("function") or {}).get("name", "")
            raw_args = (tc.get("function") or {}).get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                parsed_args = {}
            handler = tool_handlers.get(fn)
            result = handler(parsed_args) if handler else f"ERROR: herramienta desconocida {fn}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": fn,
                "content": result,
            })
    else:
        final_text = (
            "La auditoria alcanzo el limite de rondas antes de una conclusion final. "
            "Aumenta --max-rounds y vuelve a ejecutar."
        )

    out = Path.home() / "storage" / "downloads" / "QWEN_REPO_AUDIT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_text)
    print("\n--- INFORME QWEN ---\n")
    print(final_text)
    say(f"Informe guardado en {out}")


def _kernel_status(kernel):
    r = kaggle("kernels", "status", kernel)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
    return (m.group(1) if m else "UNKNOWN"), out.strip()


def _install_qwen_audit_shortcut():
    """Install a tiny Termux command so future audits are just: qwen-audit."""
    prefix = os.environ.get("PREFIX")
    if not prefix:
        return
    bindir = Path(prefix) / "bin"
    try:
        bindir.mkdir(parents=True, exist_ok=True)
        target = bindir / "qwen-audit"
        script = (
            "#!/data/data/com.termux/files/usr/bin/bash\n"
            'exec python "$HOME/kaggle-tpu-lab/launch.py" auto-audit "$@"\n'
        )
        if not target.exists() or target.read_text(errors="ignore") != script:
            target.write_text(script)
            target.chmod(0o755)
    except Exception as e:
        say(f"AVISO: no pude instalar el comando corto qwen-audit: {e}")


def _pick_audit_instructions(explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            sys.exit(f"No existe el archivo de instrucciones: {p}")
        return p

    downloads = Path.home() / "storage" / "downloads"
    if not downloads.is_dir():
        sys.exit(f"No existe la carpeta de Descargas esperada: {downloads}")

    # Si existe un nombre corto conocido, usarlo.
    for preferred_name in ("QN.md", "QWEN.md", "Gwen.md"):
        matches = [p for p in downloads.iterdir()
                   if p.is_file() and p.name.lower() == preferred_name.lower()]
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)

    # Si no, usar automaticamente el prompt de auditoria Qwen mas reciente.
    candidates = []
    for p in downloads.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(".md") and (
            low.startswith("prompt_qwen_auditoria")
            or ("qwen" in low and "auditoria" in low)
            or ("qwen" in low and "audit" in low)
        ):
            candidates.append(p)

    if candidates:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        say(f"Usare automaticamente el prompt de auditoria mas reciente: {chosen.name}")
        return chosen

    sys.exit(
        "No encontre ningun prompt de auditoria Qwen en ~/storage/downloads/. "
        "Deja ahi un archivo .md cuyo nombre contenga Qwen y auditoria."
    )


def _wait_for_public_endpoint(kernel, topic, start_timeout_s=2100, queue_timeout_s=7200):
    """
    Wait for a usable public endpoint.

    QUEUED time is tracked separately and does not consume the Qwen startup timeout.
    The startup timeout begins only after Kaggle reports RUNNING.
    """
    queue_started = time.time()
    running_started = None
    last_notice = 0
    last_status = None
    last_phase = None

    phase_labels = {
        "install": "PREPARANDO ENTORNO",
        "installed": "ENTORNO LISTO",
        "cache-restored": "CACHE XLA RESTAURADA",
        "weights-mounted": "PESOS MONTADOS",
        "weights-download": "DESCARGANDO PESOS",
        "weights-downloaded": "PESOS DESCARGADOS",
        "server-launch": "INICIANDO QWEN",
        "loading": "CARGANDO QWEN EN TPU",
        "compiling": "COMPILANDO GRAFICOS XLA",
        "warmed": "QWEN CALENTADO",
        "tunnel-url": "TUNEL PUBLICO RESERVADO",
        "tunnel-failed": "FALLO EL TUNEL PUBLICO",
        "serving": "SERVIDOR QWEN SALUDABLE",
        "ready": "QWEN LISTO",
    }

    while True:
        now = time.time()
        status, _ = _kernel_status(kernel)

        if status != last_status:
            if status == "QUEUED":
                say("TPU EN COLA: esperando que Kaggle asigne una v5e-8. Este tiempo no cuenta como arranque.")
            elif status == "RUNNING":
                if running_started is None:
                    running_started = now
                say("TPU ASIGNADA: comienza ahora el tiempo de arranque de Qwen.")
            elif status not in ("UNKNOWN",):
                say(f"Estado Kaggle: {status}")
            last_status = status

        if status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
            return None, f"kernel-{status.lower()}"

        if status == "QUEUED":
            queued_s = now - queue_started
            if queue_timeout_s > 0 and queued_s >= queue_timeout_s:
                return None, "queue-timeout"
            if now - last_notice >= 60:
                say(f"TPU EN COLA... {int(queued_s // 60)} min")
                last_notice = now
            time.sleep(10)
            continue

        if status == "RUNNING" and running_started is None:
            running_started = now
            say("TPU ASIGNADA: comienza ahora el tiempo de arranque de Qwen.")

        events_since = int((running_started or queue_started) - 120)
        events = read_events(topic, events_since)
        for _, ev in reversed(events):
            phase = ev.get("phase")
            if phase == "ready" and ev.get("endpoint"):
                say("ENDPOINT PUBLICO LISTO.")
                return ev, None
            if phase in ("failed", "auto-shutdown", "stopped"):
                return None, f"event-{phase}"

        if events:
            latest = events[-1][1]
            phase = latest.get("phase")
            if phase and phase != last_phase:
                label = phase_labels.get(phase)
                if label:
                    if phase == "tunnel-failed":
                        say(f"{label}: Qwen esta vivo dentro de Kaggle, pero aun no hay URL utilizable.")
                    else:
                        say(label)
                last_phase = phase

        if status == "RUNNING":
            run_s = now - running_started
            if run_s >= start_timeout_s:
                return None, "startup-timeout"
            if now - last_notice >= 60:
                say(f"QWEN ARRANCANDO... {int(run_s // 60)} min desde que Kaggle asigno la TPU")
                last_notice = now
        else:
            if now - last_notice >= 60:
                say(f"Esperando estado RUNNING de Kaggle... estado actual: {status}")
                last_notice = now

        time.sleep(10)


def cmd_auto_audit(args):
    """
    One-touch flow:
      reuse/start Qwen -> wait through Kaggle queue -> start timer on RUNNING -> audit repo -> stop TPU.
    """
    _install_qwen_audit_shortcut()
    instructions = _pick_audit_instructions(args.instructions)
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        sys.exit(f"No existe el repositorio: {repo}")

    check_auth()
    active = False
    st = None
    child = None

    if STATE_FILE.exists():
        try:
            st = load_state()
            status, _ = _kernel_status(st["kernel"])
            active = status in ("QUEUED", "RUNNING")
        except Exception:
            active = False

    if active:
        say(f"Reutilizando la instancia actual: {st['kernel']}")
    else:
        say("No hay una instancia activa. Inicio Qwen automaticamente en modo rapido/texto.")
        cmd = [sys.executable, str(HERE / "launch.py"), "serve", "--fast-start", "--text-only"]
        child = subprocess.Popen(cmd, cwd=str(HERE))
        deadline = time.time() + 120
        old_topic = st.get("topic") if isinstance(st, dict) else None
        while time.time() < deadline:
            if STATE_FILE.exists():
                try:
                    candidate = load_state()
                    if candidate.get("topic") and candidate.get("topic") != old_topic:
                        st = candidate
                        break
                except Exception:
                    pass
            if child.poll() is not None:
                sys.exit("El lanzador termino antes de crear la instancia.")
            time.sleep(2)
        if not st or not st.get("topic"):
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)
            sys.exit("No se pudo obtener el estado de la nueva instancia.")

    say(f"Instrucciones: {instructions.name}")
    say(f"Repositorio: {repo}")
    say("Cuando termine la auditoria, la TPU se apagara automaticamente.")

    try:
        ready, wait_reason = _wait_for_public_endpoint(
            st["kernel"],
            st["topic"],
            start_timeout_s=args.start_timeout,
            queue_timeout_s=args.queue_timeout,
        )
        if not ready:
            if wait_reason == "queue-timeout":
                sys.exit("Kaggle supero el limite de espera en cola. La sesion se cancelara automaticamente.")
            if wait_reason == "startup-timeout":
                sys.exit("Qwen supero el limite de arranque DESPUES de recibir la TPU. La sesion se apagara automaticamente.")
            sys.exit(f"Qwen no obtuvo un endpoint publico utilizable ({wait_reason}). La sesion se apagara automaticamente.")

        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate()

        say(f"Endpoint listo. Comienza la auditoria con {ready.get('model', 'Qwen')}.")
        audit_args = argparse.Namespace(
            repo=str(repo),
            instructions=str(instructions),
            reasoning_effort=args.reasoning_effort,
            max_rounds=args.max_rounds,
            timeout=args.timeout,
        )
        cmd_repo_audit(audit_args)
    finally:
        if not args.keep_running:
            try:
                current = load_state()
                status, _ = _kernel_status(current["kernel"])
                if status in ("QUEUED", "RUNNING"):
                    say("Auditoria finalizada. Apagando la TPU automaticamente...")
                    cmd_stop(argparse.Namespace())
                else:
                    say(f"La instancia ya no esta activa (estado {status}).")
            except Exception as e:
                say(f"AVISO: no pude confirmar/apagar la TPU automaticamente: {e}")
        else:
            say("TPU dejada encendida por --keep-running.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="enviar el kernel y seguir su arranque")
    s.add_argument("--model", default="qwen38-27b", choices=sorted(MODELS), help="which recipe (model folder) to serve")
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

    s = sub.add_parser("prompt", help="enviar texto o un archivo al Qwen activo")
    s.add_argument("prompt", help="texto del prompt o ruta a un archivo de texto")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "high", "medium", "low"])
    s.add_argument("--timeout", type=int, default=900,
                   help="timeout HTTP en segundos")
    s.add_argument("--show-reasoning", action="store_true",
                   help="mostrar tambien reasoning_content si el servidor lo devuelve")
    s.set_defaults(fn=cmd_prompt)

    s = sub.add_parser("audit-repo", help="auditar un repo local con Qwen en modo solo lectura")
    s.add_argument("repo", help="ruta al repositorio local")
    s.add_argument("--instructions", help="archivo de texto/Markdown con instrucciones de auditoria")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "high", "medium", "low"])
    s.add_argument("--max-rounds", type=int, default=30,
                   help="maximo de rondas de lectura/busqueda")
    s.add_argument("--timeout", type=int, default=900,
                   help="timeout por llamada HTTP en segundos")
    s.set_defaults(fn=cmd_repo_audit)

    s = sub.add_parser("auto-audit", help="flujo de un toque: iniciar/reusar Qwen, auditar y apagar TPU")
    s.add_argument("--repo", default=str(Path.home() / "9router-license-test"),
                   help="repositorio a auditar (default: ~/9router-license-test)")
    s.add_argument("--instructions",
                   help="archivo de instrucciones; por defecto autodetecta el prompt Qwen mas reciente en Descargas")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "high", "medium", "low"])
    s.add_argument("--max-rounds", type=int, default=30)
    s.add_argument("--timeout", type=int, default=900,
                   help="timeout por llamada al modelo")
    s.add_argument("--start-timeout", type=int, default=2100,
                   help="maximo de segundos para arrancar Qwen, contados solo desde RUNNING")
    s.add_argument("--queue-timeout", type=int, default=7200,
                   help="maximo de segundos en cola de Kaggle; no consume el timeout de arranque")
    s.add_argument("--keep-running", action="store_true",
                   help="no apagar la TPU al terminar")
    s.set_defaults(fn=cmd_auto_audit)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
