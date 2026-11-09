"""Authenticated, single-session control plane; importing this module allocates nothing."""
from __future__ import annotations

import copy
import hmac
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROFILES = {
    "qwen-small": dict(id="qwen-small", name="Qwen 2.5 · 0.5B", model="Qwen/Qwen2.5-0.5B-Instruct", gpu="T4", cpu=2, memory_mib=8192, description="Modelo pequeño para pruebas con GPU T4."),
    "qwen-7b": dict(id="qwen-7b", name="Qwen 2.5 · 7B", model="Qwen/Qwen2.5-7B-Instruct", gpu="L40S", cpu=4, memory_mib=16384, description="Modelo de 7B en L40S; mayor coste y descarga inicial."),
}
TERMINAL = {"STOPPED", "ERROR"}


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str
    max_minutes: int = Field(ge=5, le=120, strict=True)
    idle_minutes: int = Field(ge=2, le=30, strict=True)
    request_id: UUID

    @model_validator(mode="after")
    def valid_idle_window(self):
        if self.idle_minutes > self.max_minutes:
            raise ValueError("idle_minutes must not exceed max_minutes")
        return self


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=16000)


class Chat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=1024, strict=True)


def redact(line: str, secrets=()) -> str:
    for secret in secrets:
        if secret:
            line = line.replace(secret, "[REDACTED]")
    line = re.sub(r"(?i)(bearer\s+|(?:token|secret|api[_-]?key)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", line)
    line = re.sub(r"https?://\S+", "[URL REDACTED]", line)
    return line[:2000]


class Controller:
    def __init__(self, store, backend, token):
        if not token or len(token) < 32:
            raise ValueError("CONTROLLER_TOKEN must contain at least 32 characters")
        self.store, self.backend, self.token = store, backend, token
        self.lock = threading.RLock()

    def save(self, record):
        self.store["current"] = copy.deepcopy(record)
        self.store["request:" + record["request_id"]] = copy.deepcopy(record)

    @staticmethod
    def public(record):
        return copy.deepcopy(record["session"]) if record else None

    def refresh(self):
        record = self.store.get("current")
        if not record or record["session"]["status"] in TERMINAL:
            return record
        try:
            if not record.get("sandbox_id"):
                found = self.backend.find(record["name"])
                if not found:
                    record["session"].update(status="UNKNOWN", detail="Creación sin confirmar; no se iniciará otro recurso.")
                    self.save(record)
                    return record
                record["sandbox_id"] = found
                self.save(record)
            result = self.backend.poll(record)
            if result is not None:
                record["session"].update(status="STOPPED", detail=f"Sandbox finalizado (código {result}).")
            elif record.get("stop_requested"):
                record["session"].update(status="STOPPING", detail="Terminación pendiente de confirmar.")
            elif self.backend.ready(record):
                record["session"].update(status="READY", detail="Servidor listo.")
            else:
                record["session"].update(status="STARTING", detail="Descargando/cargando modelo; el límite absoluto sigue activo.")
        except Exception:
            record["session"].update(status="UNKNOWN", detail="No se pudo confirmar el estado; se conserva la identidad del recurso.")
        self.save(record)
        return record

    def start(self, body):
        with self.lock:
            if body.profile_id not in PROFILES:
                raise HTTPException(422, "Perfil no permitido")
            key = "request:" + str(body.request_id)
            current = self.store.get("current")
            previous = self.store.get(key)
            if not previous and current and current["request_id"] == str(body.request_id):
                previous = current
            if previous:
                if previous["parameters"] != body.model_dump(mode="json"):
                    raise HTTPException(409, "request_id reutilizado con otros parámetros")
                current = self.refresh()
                return self.public(current if current and current["request_id"] == str(body.request_id) else previous)
            current = self.refresh()
            if current and current["session"]["status"] not in TERMINAL:
                raise HTTPException(409, "Ya existe una sesión activa o sin confirmar")
            profile = PROFILES[body.profile_id]
            now = datetime.now(timezone.utc)
            session = {k: profile[k] for k in ("model", "gpu", "cpu", "memory_mib")}
            session.update(id=str(uuid4()), profile_id=body.profile_id, status="STARTING", created_at=now.isoformat(), expires_at=(now + timedelta(minutes=body.max_minutes)).isoformat(), idle_minutes=body.idle_minutes, max_minutes=body.max_minutes)
            record = dict(session=session, request_id=str(body.request_id), parameters=body.model_dump(mode="json"), name=self.backend.name, gateway_token=uuid4().hex + uuid4().hex, sandbox_id=None)
            # Write intent before the provider call. Ambiguous creates are never retried.
            self.save(record)
            try:
                record["sandbox_id"] = self.backend.create(record)
            except Exception:
                record["session"].update(status="UNKNOWN", detail="Creación sin confirmar. Consulte estado o solicite detener; no repita con otro identificador.")
            self.save(record)
            return self.public(record)

    def stop(self):
        with self.lock:
            record = self.refresh()
            if not record or record["session"]["status"] in TERMINAL:
                return self.public(record)
            record["stop_requested"] = True
            record["session"]["status"] = "STOPPING"
            self.save(record)
            try:
                if not record.get("sandbox_id"):
                    raise RuntimeError("No confirmed handle")
                self.backend.stop(record)
                # A terminate response alone does not prove termination.
                if self.backend.poll(record) is not None:
                    record["session"].update(status="STOPPED", detail="Terminación confirmada.")
            except Exception:
                record["session"].update(status="UNKNOWN", detail="No se pudo confirmar la detención; el límite absoluto permanece activo.")
            self.save(record)
            return self.public(record)


def make_app(store, backend, token):
    controller = Controller(store, backend, token)
    inference_slot = threading.BoundedSemaphore(1)

    def authenticate(authorization: str | None = Header(default=None)):
        supplied = (authorization or "").removeprefix("Bearer ")
        if not authorization or not authorization.startswith("Bearer ") or not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "No autorizado", headers={"WWW-Authenticate": "Bearer"})

    api = FastAPI(title="Modal GPU Control", docs_url=None, redoc_url=None, openapi_url=None, dependencies=[Depends(authenticate)])

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "https://localhost", "capacitor://localhost", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @api.get("/api/profiles")
    def profiles():
        return {"profiles": list(PROFILES.values())}

    @api.get("/api/session")
    def session():
        with controller.lock:
            return {"session": controller.public(controller.refresh())}

    @api.post("/api/session")
    def start(body: Start):
        return {"session": controller.start(body)}

    @api.delete("/api/session")
    def stop():
        return {"session": controller.stop()}

    @api.get("/api/logs")
    def logs():
        with controller.lock:
            record = controller.refresh()
            if not record or not record.get("sandbox_id"):
                return {"lines": []}
            try:
                lines = backend.logs(record)
            except Exception:
                lines = ["Registros temporalmente no disponibles; identidad del recurso conservada."]
            return {"lines": [redact(str(line), (token, record["gateway_token"])) for line in lines[-100:]]}

    @api.post("/api/chat")
    def chat(body: Chat):
        with controller.lock:
            record = controller.refresh()
            if not record or record["session"]["status"] != "READY":
                raise HTTPException(409, "Inicie una sesión y espere a READY; el chat nunca inicia recursos")
        if not inference_slot.acquire(blocking=False):
            raise HTTPException(429, "Ya hay una inferencia en curso")
        try:
            return backend.chat(record, body.model_dump())
        except Exception:
            raise HTTPException(502, "Inferencia no disponible; consulte el estado") from None
        finally:
            inference_slot.release()

    return api
