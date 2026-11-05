#!/usr/bin/env python3
"""Diagnostico de la sesion existente: no inicia, cancela ni elimina kernels."""
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

def main():
    state_file = Path.home() / ".kaggle-tpu-lab.json"
    if not state_file.is_file():
        print("No hay estado local de sesion. No se inicio ninguna TPU.")
        return 1
    st = json.loads(state_file.read_text())
    print("Kernel:", st.get("kernel", "desconocido"))
    cmd = [shutil.which("kaggle")] if shutil.which("kaggle") else [sys.executable, "-m", "kaggle"]
    try:
        r = subprocess.run(cmd + ["kernels", "status", st["kernel"]],
                           capture_output=True, text=True, timeout=40)
        match = re.search(r"KernelWorkerStatus\.(\w+)", (r.stdout or "") + (r.stderr or ""))
        print("Estado Kaggle:", match.group(1) if match else "sin confirmar")
    except subprocess.TimeoutExpired:
        print("Estado Kaggle: consulta sin respuesta.")
    topic = urllib.parse.quote(st.get("topic", ""), safe="")
    if not topic:
        print("Falta el topic de la sesion. No se inicio ninguna TPU.")
        return 1
    url = "https://ntfy.sh/" + topic + "/json?poll=1&since=" + str(int(time.time()) - 86400)
    try:
        with urllib.request.urlopen(url, timeout=25) as resp:
            rows = resp.read().decode().splitlines()
    except Exception as exc:
        print("No se pudo consultar el historial de sesion:", type(exc).__name__)
        return 1
    events = []
    for line in rows:
        try:
            envelope = json.loads(line)
            if envelope.get("event") == "message":
                event = json.loads(envelope.get("message", "{}"))
                events.append((int(envelope["time"]), event))
        except (ValueError, KeyError, TypeError):
            continue
    events.sort(key=lambda item: item[0])
    if not events:
        print("Sin eventos recuperables en las ultimas 24 horas; servicio sin confirmar.")
        return 1
    now = time.time()
    print("Eventos recientes (fechas actuales, sin claves):")
    for ts, ev in events[-8:]:
        when = datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
        print(" ", when, ev.get("phase", "?"), "| hace", max(0, int(now-ts)), "segundos")
    ready = None
    for ts, ev in events:
        if ev.get("phase") in ("failed", "stopped", "auto-shutdown"):
            ready = None
        elif ev.get("phase") == "ready":
            ready = ev
    if not ready:
        print("No hay READY vigente en el historial. Inferencia sin confirmar.")
        return 1
    endpoint = ready.get("endpoint")
    if not endpoint:
        print("READY sin endpoint publico: el modelo pudo arrancar, pero no hay tunel utilizable.")
        return 1
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        print("Endpoint no valido para la comprobacion.")
        return 1
    base = endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    key = ready.get("api_key") or st.get("api_key", "")
    print("Endpoint:", base)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    def request(path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    try:
        models = request("/models")
        ids = [m.get("id") for m in models.get("data", []) if m.get("id")]
        print("API /models: responde. Modelos:", ", ".join(ids) or "(sin modelos)")
        if not ids:
            return 1
        model = ready.get("model")
        if model not in ids:
            model = ids[0]
        answer = request("/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": "Responde solamente OK."}],
            "max_tokens": 64, "stream": False,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        })
        content = (answer.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if content:
            print("Inferencia real confirmada:", str(content)[:160])
            return 0
        print("La API respondio sin contenido final. Inferencia util sin confirmar.")
        return 1
    except urllib.error.HTTPError as exc:
        print("Prueba API: HTTP", exc.code, "(no se muestran credenciales ni cuerpo de error).")
    except Exception as exc:
        print("Prueba API:", type(exc).__name__, "(sin respuesta confirmada).")
    return 1

if __name__ == "__main__":
    sys.exit(main())
