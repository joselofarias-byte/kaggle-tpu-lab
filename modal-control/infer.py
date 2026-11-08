"""Opt-in inference smoke test with an ownership journal and provider-side cleanup."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from uuid import uuid4
from urllib.parse import urlsplit
from decimal import Decimal, InvalidOperation

import httpx
import modal

APP_NAME = "android-modal-control"
STATE_NAME = "android-modal-control-state-v1"
SANDBOX_NAME = "android-owned-model-v1"
PROFILES = {"qwen-small": ("T4", 2, 8192), "qwen-7b": ("L40S", 4, 16384)}


def write_journal(path, record):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(record, stream)


def connection(path=None, url=None):
    if url:
        token = os.environ.get("CONTROLLER_TOKEN", "")
        if not token:
            from operations import read_controller_token
            token = read_controller_token()
        data = {"url": url, "token": token}
    else:
        data = json.loads(Path(path or "~/.config/kaggle-modal/connection.json").expanduser().read_text())
    parsed = urlsplit(data["url"])
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or len(data.get("token", "")) < 32
            or any(c.isspace() for c in data["token"])):
        raise ValueError("Invalid controller connection")
    return data["url"].rstrip("/"), data["token"]


def cleanup(journal_path, attempts=12, pause=time.sleep):
    path = Path(journal_path).expanduser()
    if not path.exists():
        print("No hay intento registrado; no se detiene ningún recurso.")
        return
    journal = json.loads(path.read_text())
    if not journal.get("attempted"):
        return
    store = modal.Dict.from_name(STATE_NAME)
    for _ in range(attempts):
        record = store.get("current")
        if not record or record.get("request_id") != journal["request_id"]:
            previous = store.get("request:" + journal["request_id"])
            if previous and previous["session"]["status"] == "STOPPED":
                print("La sesión de prueba ya está detenida.")
                return
            if previous:
                record = previous
            else:
                # An in-flight creation may not have persisted its intent yet.
                pause(5)
                continue
        sandbox_id = record.get("sandbox_id")
        if not sandbox_id:
            current = store.get("current")
            if not current or current.get("request_id") != journal["request_id"]:
                raise RuntimeError("No se resuelve por nombre una sesión histórica sin identificador")
            try:
                sandbox = modal.Sandbox.from_name(APP_NAME, SANDBOX_NAME)
            except modal.exception.NotFoundError:
                if record["session"]["status"] == "STOPPED":
                    return
                pause(5)
                continue
        else:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        if sandbox.poll() is None:
            sandbox.terminate(wait=True)
        if sandbox.poll() is not None:
            print("Parada del sandbox de prueba confirmada por Modal.")
            return
        pause(5)
    raise RuntimeError("Parada sin confirmar; revisar Modal. El timeout absoluto sigue vigente.")


def run(args, client_factory=httpx.Client, clock=time.monotonic, pause=time.sleep):
    if not args.accept_cost or args.profile not in PROFILES or not 5 <= args.max_minutes <= 20:
        raise ValueError("Confirme el coste y un límite de 5–20 minutos")
    try:
        rate = Decimal(args.gpu_hourly_usd)
    except InvalidOperation:
        raise ValueError("Tarifa inválida") from None
    if not rate.is_finite() or rate <= 0:
        raise ValueError("La tarifa GPU debe ser positiva")
    gpu, cpu, memory = PROFILES[args.profile]
    print(f"GPU {gpu}; CPU {cpu}; RAM {memory} MiB; máximo {args.max_minutes} min.")
    print(f"Tarifa GPU declarada: USD {rate}/h; estimación GPU máxima: USD {rate * args.max_minutes / 60:.4f}.")
    print("No es factura ni presupuesto total: CPU/RAM Sandbox, controlador, builds y otros cargos aparte.")
    url, token = connection(args.connection, args.url)
    deadline = clock() + args.max_minutes * 60
    body = dict(profile_id=args.profile, max_minutes=args.max_minutes, idle_minutes=2, request_id=str(uuid4()))
    with client_factory(base_url=url, headers={"Authorization": "Bearer " + token},
                        follow_redirects=False, timeout=30) as client:
        response = client.get("/api/session")
        response.raise_for_status()
        existing = response.json()["session"]
        if existing is not None and existing.get("status") != "STOPPED":
            raise RuntimeError("Existe una sesión activa o sin confirmar; prueba cancelada")
        write_journal(args.journal, {"request_id": body["request_id"], "attempted": True})
        try:
            response = client.post("/api/session", json=body, timeout=120)
            response.raise_for_status()
            session = response.json()["session"]
            session_id = session["id"]
            while session["status"] != "READY":
                if clock() >= deadline or session["status"] in ("STOPPED", "ERROR"):
                    raise RuntimeError("El modelo no estuvo listo dentro del límite")
                pause(5)
                response = client.get("/api/session")
                response.raise_for_status()
                session = response.json()["session"]
                if not session or session["id"] != session_id:
                    raise RuntimeError("Cambió la identidad de la sesión")
            remaining = deadline - clock()
            if remaining <= 0:
                raise RuntimeError("Límite de prueba agotado")
            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Responde solamente: OK"}], "max_tokens": 16}, timeout=min(120, remaining))
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Respuesta de inferencia inválida")
            print("Inferencia recibida y validada (contenido omitido de los registros).")
        finally:
            cleanup(args.journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("run")
    test.add_argument("--connection")
    test.add_argument("--url")
    test.add_argument("--profile", choices=PROFILES, default="qwen-small")
    test.add_argument("--max-minutes", type=int, default=10)
    test.add_argument("--gpu-hourly-usd", required=True)
    test.add_argument("--accept-cost", action="store_true")
    test.add_argument("--journal", default="~/.config/kaggle-modal/inference.json")
    stop = sub.add_parser("stop")
    stop.add_argument("--journal", default="~/.config/kaggle-modal/inference.json")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.command == "run":
            run(args)
        else:
            cleanup(args.journal)
    except (Exception, KeyboardInterrupt):
        print("Operación fallida o interrumpida. No se muestran respuestas ni credenciales. Si hubo inicio, confirme la parada en Modal; el timeout absoluto es el respaldo.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
