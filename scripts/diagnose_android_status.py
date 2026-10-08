#!/usr/bin/env python3
"""Diagnostico no destructivo de una sesion Kaggle TPU.

Solo lee: Kaggle API/CLI y, si existe, un topic ntfy guardado localmente.
Nunca lanza, detiene, cancela ni modifica kernels. No imprime credenciales.
"""
import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"

def load_state():
    try:
        data = json.loads(STATE_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def label(value):
    """Solo texto de estados y mensajes, nunca datos privados."""
    s = str(value or "")
    s = re.sub(r"(?i)\\b(Bearer|Basic)\\s+\\S+", r"\\1 [oculto]", s)
    s = re.sub(r"\\b(?:KGAT_|sk-|glm-)[A-Za-z0-9_-]{6,}", "[secreto oculto]", s)
    s = re.sub(r"https?://\\S+", "[URL oculta]", s)
    return s[:200]

def kaggle_sdk_status(kernel):
    try:
        from kaggle import api
        response = api.kernels_status(kernel)
        value = getattr(response, "status", None)
        if value is None:
            print("SDK Kaggle: devolvio status=None (sin estado util).")
        else:
            name = getattr(value, "name", None) or str(value)
            print("SDK Kaggle:", label(name))
        fail = getattr(response, "failure_message", None)
        if fail:
            print("Fallo (resumido):", label(fail))
        return True
    except Exception as exc:
        print("SDK Kaggle: no disponible (" + type(exc).__name__ + ").")
        return False

def kaggle_cli_status(kernel):
    cmd = [shutil.which("kaggle"), "kernels", "status", kernel] if shutil.which("kaggle") else [sys.executable, "-m", "kaggle", "kernels", "status", kernel]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        print("CLI Kaggle: timeout (40 s).")
        return
    except OSError as exc:
        print("CLI Kaggle: no disponible (" + type(exc).__name__ + ").")
        return
    out = (p.stdout or "") + "\\n" + (p.stderr or "")
    status = re.search(r"KernelWorkerStatus\\.([A-Z_]+)", out)
    if status:
        print("CLI Kaggle:", status.group(1))
    else:
        # Avoid exposing arbitrary API output or credentials.
        print("CLI Kaggle:", "sin estado interpretable" if p.returncode == 0 else "fallo de consulta")
        print("  Codigo de salida:", p.returncode)
        if "401" in out or "Unauthenticated" in out:
            print("  Motivo probable: credencial Kaggle no valida en Termux.")
        elif "403" in out:
            print("  Motivo probable: acceso no autorizado.")
        elif "404" in out:
            print("  Motivo probable: kernel inexistente o inaccesible.")
        elif "No module named kaggle" in out:
            print("  Motivo: modulo kaggle no instalado en este Python.")

def ntfy_status(st, kernel):
    local_kernel = st.get("kernel")
    if local_kernel != kernel:
        print("ntfy: estado local ausente o pertenece a otra sesion; no se consulta.")
        return
    topic = st.get("topic", "")
    if not isinstance(topic, str) or not re.fullmatch(r"ktl-[A-Za-z0-9_-]{12,60}", topic):
        print("ntfy: topic local no disponible.")
        return
    cutoff = int(time.time()) - 86400
    url = "https://ntfy.sh/" + urllib.parse.quote(topic, safe="") + "/json?poll=1&since=" + str(cutoff)
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            lines = response.read(800000).decode("utf-8", errors="replace").splitlines()
    except Exception as exc:
        print("ntfy: consulta no disponible (" + type(exc).__name__ + ").")
        return
    events = []
    for line in lines:
        try:
            obj = json.loads(line)
            if obj.get("event") != "message":
                continue
            ev = json.loads(obj.get("message", "{}"))
            phase = ev.get("phase")
            timestamp = int(obj.get("time", 0))
            if timestamp > 0 and isinstance(phase, str):
                events.append((timestamp, phase))
        except (ValueError, TypeError):
            continue
    if not events:
        print("ntfy: sin eventos recuperables de las ultimas 24 h.")
        return
    events.sort()
    print("Ultimos eventos de la sesion local (sin URL ni claves):")
    for ts, phase in events[-6:]:
        stamp = datetime.datetime.fromtimestamp(ts).astimezone().strftime("%H:%M:%S")
        print(" ", stamp, label(phase), "-", max(0, int(time.time()) - ts), "segundos atras")
    fresh_ready = any(phase in ("ready", "serving", "heartbeat") and time.time() - ts < 900 for ts, phase in events)
    print("Evidencia ntfy de servicio en ultimos 15 min:", "SI" if fresh_ready else "NO")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", help="owner/slug; si falta, usa ~/.kaggle-tpu-lab.json")
    args = parser.parse_args()
    st = load_state()
    kernel = args.kernel or st.get("kernel", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel):
        print("Indica --kernel usuario/qwen38-tpu-serve. No se ejecuto ninguna accion.")
        return 2
    print("=== KAGGLE TPU - DIAGNOSTICO SOLO LECTURA ===")
    print("Kernel:", kernel)
    kaggle_sdk_status(kernel)
    kaggle_cli_status(kernel)
    ntfy_status(st, kernel)
    print("NO se inicio, cancelo ni reinicio ninguna sesion.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
