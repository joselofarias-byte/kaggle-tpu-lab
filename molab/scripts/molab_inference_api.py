#!/usr/bin/env python3
"""Experimental Transformers API; not a GGUF/TPU backend.

Install fastapi==0.115.12 uvicorn==0.34.2 transformers==4.57.6 accelerate torch
safetensors. Set MOLAB_API_KEY (24+ characters) and MODEL_ID; binds to loopback.
GPU libraries and weights are loaded only during the application lifespan.
"""

import asyncio
import json
import math
import os
import queue
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Message content must not be blank")
        return value


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1, max_length=256)
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class ChatTemplateError(ValueError):
    pass


class Backend(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...
    def generate(
        self, req: Completion, emit: Callable[[str], None], cancel: threading.Event
    ) -> Generation: ...


class TransformersBackend:
    def __init__(self, model_id):
        self.model_id = model_id

    def start(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: select GPU in Molab Compute")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        self.model.eval()

    def close(self):
        self.model = None
        self.tokenizer = None

    def generate(self, req, emit, cancel):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        try:
            prompt = self.tokenizer.apply_chat_template(
                [m.model_dump() for m in req.messages], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except Exception as exc:
            raise ChatTemplateError("Model chat template is unavailable or incompatible") from exc
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_input = inputs["input_ids"].shape[-1]
        context = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context, int) and n_input + req.max_tokens > context:
            raise ChatTemplateError("Prompt and max_tokens exceed model context")

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set()

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text and not cancel.is_set():
                    emit(text)

        kwargs = {
            "max_new_tokens": req.max_tokens,
            "do_sample": req.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([Cancelled()]),
            "streamer": Streamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
        }
        if req.temperature > 0:
            kwargs["temperature"] = req.temperature
        if cancel.is_set():
            return Generation("", n_input, 0)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        output = out[0][n_input:]
        return Generation(
            self.tokenizer.decode(output, skip_special_tokens=True), n_input,
            len(output), "length" if len(output) >= req.max_tokens else "stop",
        )


def create_app(
    *, backend: Backend | None = None, api_key: str | None = None,
    model_id: str | None = None, generation_timeout: float = 120.0,
) -> FastAPI:
    """Inject a CPU backend for tests; generation must cooperate with cancellation."""
    model_id = model_id or os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
    key = api_key if api_key is not None else os.environ.get("MOLAB_API_KEY", "")
    if not math.isfinite(generation_timeout) or generation_timeout <= 0:
        raise ValueError("generation_timeout must be finite and positive")
    engine = backend if backend is not None else TransformersBackend(model_id)
    generation_lock = threading.Lock()
    active: set[threading.Event] = set()

    @asynccontextmanager
    async def lifespan(app):
        if len(key) < 24:
            raise RuntimeError("MOLAB_API_KEY must be at least 24 characters")
        await asyncio.to_thread(engine.start)
        try:
            yield
        finally:
            for cancel in tuple(active):
                cancel.set()
            # Never unload weights while a cancelled CUDA operation is still running.
            if not generation_lock.locked():
                await asyncio.to_thread(engine.close)

    app = FastAPI(
        title="JolufaSoft Molab API", lifespan=lifespan,
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {key}"
        if len(key) < 24 or not secrets.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8")
        ):
            return JSONResponse(
                {"error": {"message": "Invalid API token", "type": "authentication_error"}},
                status_code=401, headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"ok": True, "model": model_id}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {"id": model_id, "object": "model", "owned_by": "jolufasoft"}
        ]}

    async def events(req, request):
        pending = queue.Queue(maxsize=32)
        cancel = threading.Event()
        active.add(cancel)
        deadline = time.monotonic() + generation_timeout

        def publish(kind, value):
            while not cancel.is_set():
                try:
                    pending.put((kind, value), timeout=0.05)
                    return
                except queue.Full:
                    continue

        def run():
            acquired = False
            try:
                while not cancel.is_set():
                    acquired = generation_lock.acquire(timeout=0.05)
                    if acquired:
                        break
                if not acquired or cancel.is_set():
                    return
                result = engine.generate(req, lambda text: publish("text", text), cancel)
                publish("result", result)
            except ChatTemplateError as exc:
                publish("error", (400, str(exc)))
            except Exception:
                # Backend exceptions may contain local paths, prompts or credentials.
                publish("error", (500, "Generation failed"))
            finally:
                if acquired:
                    generation_lock.release()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            while True:
                if await request.is_disconnected():
                    return
                if time.monotonic() >= deadline:
                    yield "error", (504, "Generation timed out")
                    return
                try:
                    kind, value = pending.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                yield kind, value
                if kind in {"error", "result"}:
                    return
        finally:
            cancel.set()
            active.discard(cancel)

    @app.post("/v1/chat/completions")
    async def completions(req: Completion, request: Request):
        if {"tools", "tool_choice"} & req.model_fields_set:
            raise HTTPException(400, "Tool calling is unsupported by this experimental bridge")
        if req.model != model_id:
            raise HTTPException(404, "Unknown model")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        source = events(req, request)
        if not req.stream:
            try:
                async for kind, value in source:
                    if kind == "error":
                        raise HTTPException(*value)
                    if kind == "result":
                        return {
                            "id": completion_id, "object": "chat.completion",
                            "created": created, "model": model_id,
                            "choices": [{"index": 0, "message": {
                                "role": "assistant", "content": value.text,
                            }, "finish_reason": value.finish_reason}],
                            "usage": {
                                "prompt_tokens": value.prompt_tokens,
                                "completion_tokens": value.completion_tokens,
                                "total_tokens": value.prompt_tokens + value.completion_tokens,
                            },
                        }
            finally:
                await source.aclose()
            raise HTTPException(499, "Client disconnected")

        def chunk(delta, finish_reason=None):
            return "data: " + json.dumps({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"

        async def stream():
            try:
                yield chunk({"role": "assistant"})
                async for kind, value in source:
                    if kind == "text":
                        yield chunk({"content": value})
                    elif kind == "error":
                        status, message = value
                        yield "data: " + json.dumps({"error": {
                            "message": message, "type": "generation_error", "code": status,
                        }}) + "\n\n"
                    elif kind == "result":
                        yield chunk({}, value.finish_reason)
                yield "data: [DONE]\n\n"
            finally:
                await source.aclose()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ.get("MOLAB_PORT", "18080")),
        log_level="info",
    )
