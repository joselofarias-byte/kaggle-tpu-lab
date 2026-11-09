"""Private gateway and inference-activity watchdog, running inside the GPU sandbox."""
import hmac
import os
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Activity:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.last = None
        self.inflight = 0
        self.lock = threading.Lock()

    def arm(self):
        with self.lock:
            if self.last is None:
                self.last = self.clock()

    def begin(self):
        with self.lock:
            if self.last is None:
                return False
            self.inflight += 1
            self.last = self.clock()
            return True

    def end(self):
        with self.lock:
            self.inflight -= 1
            self.last = self.clock()

    def expired(self, idle_seconds):
        with self.lock:
            return self.last is not None and self.inflight == 0 and self.clock() - self.last >= idle_seconds


def wait_for_readiness(activity, process, client, sleep=time.sleep):
    while process.poll() is None:
        try:
            if client.get("/health").status_code == 200:
                activity.arm()
                return
        except httpx.HTTPError:
            pass
        sleep(1)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=16000)


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=1024, strict=True)


def main():
    token = os.environ["GATEWAY_TOKEN"]
    activity = Activity()
    lines = deque(maxlen=100)
    process = subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", os.environ["MODEL"], "--host", "127.0.0.1", "--port", "8000",
        "--dtype", "half", "--max-model-len", "4096", "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.85", "--disable-log-requests",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def collect():
        for line in process.stdout:
            lines.append(line.rstrip()[:2000])

    def readiness():
        # Only this internal loop arms idle expiry; external polling never touches it.
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=2) as client:
            wait_for_readiness(activity, process, client)

    def watchdog():
        while True:
            time.sleep(1)
            if process.poll() is not None or activity.expired(int(os.environ["IDLE_SECONDS"])):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                os._exit(0)

    threading.Thread(target=collect, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=readiness, daemon=True).start()

    def auth(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization.encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Unauthorized")

    @asynccontextmanager
    async def lifespan(api):
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=180) as client:
            api.state.client = client
            yield

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, dependencies=[Depends(auth)], lifespan=lifespan)

    @api.get("/health")
    async def health():
        try:
            response = await api.state.client.get("/health", timeout=2)
            return {"ready": response.status_code == 200}
        except httpx.HTTPError:
            return {"ready": False}

    @api.get("/logs")
    async def logs():
        return {"lines": list(lines)}

    @api.post("/chat")
    async def chat(body: Completion):
        if sum(len(message.content) for message in body.messages) > 32000:
            raise HTTPException(422, "Conversation too large")
        if not activity.begin():
            raise HTTPException(409, "Model not ready")
        try:
            response = await api.state.client.post("/v1/chat/completions", json={**body.model_dump(), "model": os.environ["MODEL"], "stream": False})
            response.raise_for_status()
            return {"choices": [{"message": {"content": response.json()["choices"][0]["message"]["content"]}}]}
        except (httpx.HTTPError, KeyError, ValueError):
            raise HTTPException(502, "Inference unavailable") from None
        finally:
            activity.end()

    uvicorn.run(api, host="0.0.0.0", port=8080, access_log=False)


if __name__ == "__main__":
    main()
