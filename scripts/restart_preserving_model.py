#!/usr/bin/env python3
"""Reanuda Qwen con el parche local y conserva los datos del modelo previo."""
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

def main():
    root = Path.home() / "kaggle-tpu-lab"
    launcher = root / "launch.py"
    if not launcher.is_file() or "--idle-timeout-min" not in launcher.read_text():
        print("Primero aplica el parche idle30. No se inicio otra TPU.")
        return 1
    spec = importlib.util.spec_from_file_location("ktl_restart_launcher", launcher)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    st = launch.load_state()
    kernel = st.get("kernel", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel):
        print("El identificador de kernel no es valido.")
        return 1
    status, _ = launch._kernel_status(kernel)
    print("Estado previo:", status)
    terminal = {"CANCEL_ACKNOWLEDGED", "CANCELACKNOWLEDGED", "CANCELLED", "COMPLETE", "ERROR"}
    if status not in terminal:
        print("Estado activo o sin confirmar: no se lanzara otra TPU.")
        return 1
    # Fetch directly so a network error is not mistaken for absence of activity.
    import urllib.request
    import urllib.parse
    topic = urllib.parse.quote(st.get("topic", ""), safe="")
    if not topic:
        print("Falta el topic de la sesion previa. No se relanza.")
        return 1
    with urllib.request.urlopen("https://ntfy.sh/" + topic + "/json?poll=1&since="
                                + str(int(time.time()) - 86400), timeout=25) as resp:
        rows = resp.read().decode().splitlines()
    for row in rows:
        event = json.loads(row)
        if event.get("event") == "message" and time.time() - int(event["time"]) < 900:
            data = json.loads(event.get("message", "{}"))
            if data.get("phase") not in ("failed", "stopped", "auto-shutdown"):
                print("Hay actividad reciente de la sesion. No se lanza otra TPU.")
                return 1
    recipe = st.get("model_recipe") or "qwen38-27b"
    if recipe != "qwen38-27b":
        print("Este reinicio corresponde a Qwen; no se cambia la receta", recipe)
        return 1
    user, slug = kernel.split("/", 1)
    cmd = [sys.executable, str(launcher), "serve", "--user", user, "--slug", slug,
           "--model", recipe, "--fast-start", "--text-only", "--idle-timeout-min", "30"]
    for field, option in (("hf_model_id", "--hf-model-id"),
                          ("served_model_name", "--served-model-name"),
                          ("weights_dataset", "--weights-dataset")):
        if field in st and st[field] is not None:
            cmd += [option, str(st[field])]
    backup = Path.home() / ".local/state/kaggle-tpu-lab/backups"
    backup.mkdir(parents=True, exist_ok=True)
    saved = backup / ("session-before-restart-" + str(time.time_ns()) + ".json")
    # Session metadata contains the old key; keep the backup private.
    import os
    fd = os.open(saved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(st, out)
    print("Modelo conservado:", st.get("hf_model_id") or "receta Qwen previa")
    print("Se inicia una nueva version del mismo kernel, con apagado a los 30 min sin inferencia.")
    print("El arranque todavia puede demorar; el temporizador comienza despues de READY.")
    return subprocess.run(cmd, cwd=root).returncode

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("Reinicio sin confirmar:", type(exc).__name__, "(sin mostrar credenciales).")
        sys.exit(1)
