"""CPU-only tests: fastapi==0.115.12 httpx==0.28.1 pytest==8.3.5."""

import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

PATH = Path(__file__).resolve().parents[1] / "scripts" / "molab_inference_api.py"
SPEC = importlib.util.spec_from_file_location("molab_inference_api", PATH)
api = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = api
SPEC.loader.exec_module(api)
KEY = "test-key-not-a-secret-1234567890"
HEADERS = {"Authorization": f"Bearer {KEY}"}
PAYLOAD = {"model": "fake", "messages": [{"role": "user", "content": "Hello"}]}


class FakeBackend:
    def __init__(self, error=None, stall=False):
        self.started = False
        self.closed = False
        self.cancelled = threading.Event()
        self.error = error
        self.stall = stall
        self.calls = 0

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def generate(self, req, emit, cancel):
        self.calls += 1
        if self.stall:
            cancel.wait(2)
            if cancel.is_set():
                self.cancelled.set()
            return api.Generation("", 0, 0)
        emit("Hello ")
        if self.error:
            raise self.error
        emit("world")
        return api.Generation("Hello world", 3, 2, "length")


def make_app(backend=None, timeout=1):
    return api.create_app(
        backend=backend or FakeBackend(), api_key=KEY,
        model_id="fake", generation_timeout=timeout,
    )


@pytest.fixture
def client():
    with TestClient(make_app()) as test_client:
        yield test_client


def post(client, **updates):
    return client.post("/v1/chat/completions", headers=HEADERS, json=PAYLOAD | updates)


def sse(response):
    return [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]


def test_import_has_no_gpu_dependencies_or_key_requirement():
    result = subprocess.run([
        sys.executable, "-c",
        "import runpy,sys,os; os.environ.pop('MOLAB_API_KEY',None); "
        f"runpy.run_path({str(PATH)!r}); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lifespan_and_key():
    backend = FakeBackend()
    app = make_app(backend)
    assert not backend.started
    with TestClient(app):
        assert backend.started
    assert backend.closed
    with pytest.raises(RuntimeError, match="24 characters"):
        with TestClient(api.create_app(backend=FakeBackend(), api_key="short")):
            pass


@pytest.mark.parametrize("path", ["/health", "/v1/models", "/v1/chat/completions", "/docs", "/openapi.json", "/missing"])
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic invalid"])
def test_auth_all_endpoints(client, path, authorization):
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.request("POST" if path.endswith("completions") else "GET", path, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_models_health_and_disabled_docs(client):
    assert client.get("/v1/models", headers=HEADERS).json()["data"][0]["id"] == "fake"
    assert client.get("/health", headers=HEADERS).json() == {"ok": True, "model": "fake"}
    assert client.get("/docs", headers=HEADERS).status_code == 404
    assert post(client, model="other").status_code == 404


@pytest.mark.parametrize("updates", [
    {"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
    {"messages": [{"role": "user", "content": " "}]},
    {"messages": [{"role": "user", "content": 7}]},
    {"messages": [{"role": "user", "content": "x", "name": "unsupported"}]},
    {"max_tokens": "12"}, {"max_tokens": True}, {"stream": "true"},
    {"model": 42}, {"temperature": -1}, {"unknown": True},
])
def test_strict_validation(client, updates):
    assert post(client, **updates).status_code == 422


@pytest.mark.parametrize("updates", [{"tools": []}, {"tools": None}, {"tool_choice": "none"}, {"tool_choice": None}])
def test_tools_explicitly_rejected(client, updates):
    response = post(client, **updates)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"]


def test_nonstream(client):
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert post(client).json()["id"] != body["id"]


def test_stream(client):
    response = post(client, stream=True)
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = sse(response)
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert [c["choices"][0]["delta"].get("content") for c in chunks[1:-1]] == ["Hello ", "world"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert len({c["id"] for c in chunks}) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("error,status,message", [
    (api.ChatTemplateError("No chat template"), 400, "No chat template"),
    (RuntimeError("secret local path"), 500, "Generation failed"),
])
def test_generation_errors(stream, error, status, message):
    with TestClient(make_app(FakeBackend(error=error))) as client:
        response = post(client, stream=stream)
        assert "secret local path" not in response.text
        if stream:
            assert response.status_code == 200
            frames = sse(response)
            assert frames[-1] == "[DONE]"
            body = json.loads(frames[-2])["error"]
            assert body["code"] == status
            assert body["message"] == message
        else:
            assert response.status_code == status
            assert response.json()["detail"] == message


@pytest.mark.parametrize("stream", [False, True])
def test_timeout_cancels_backend(stream):
    backend = FakeBackend(stall=True)
    with TestClient(make_app(backend, timeout=0.05)) as client:
        start = time.monotonic()
        response = post(client, stream=stream)
        assert time.monotonic() - start < 1
        if stream:
            assert json.loads(sse(response)[-2])["error"]["code"] == 504
        else:
            assert response.status_code == 504
        assert backend.cancelled.wait(1)


def test_disconnect_cancels_backend():
    backend = FakeBackend(stall=True)
    app = make_app(backend)

    async def exercise():
        disconnected = asyncio.Event()
        sent = []

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                # Wait until the worker has started before simulating disconnection.
                for _ in range(100):
                    if backend.calls:
                        break
                    await asyncio.sleep(0.005)
                disconnected.set()

        # Direct ASGI request exposes real disconnects, unlike TestClient buffering.
        payload = json.dumps(PAYLOAD | {"stream": True}).encode()
        first = True

        async def receive_body():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
            "query_string": b"", "root_path": "", "server": ("test", 80),
            "client": ("test", 123), "headers": [
                (b"authorization", f"Bearer {KEY}".encode()),
                (b"content-type", b"application/json"),
            ],
        }
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive_body, send), timeout=1)
        assert any(m["type"] == "http.response.start" for m in sent)

    asyncio.run(exercise())
    assert backend.cancelled.wait(1)


def test_sse_emits_content_before_generation_finishes():
    received = threading.Event()

    class GatedBackend(FakeBackend):
        def generate(self, req, emit, cancel):
            emit("early token")
            if not received.wait(0.5):
                raise RuntimeError("Stream was buffered until generation finished")
            return api.Generation("early token", 1, 1)

    app = make_app(GatedBackend())
    payload = json.dumps(PAYLOAD | {"stream": True}).encode()
    frames = []

    async def exercise():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": payload, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            body = message.get("body", b"")
            frames.append(body)
            if b"early token" in body:
                received.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
            "query_string": b"", "root_path": "", "server": ("test", 80),
            "client": ("test", 123), "headers": [
                (b"authorization", f"Bearer {KEY}".encode()),
                (b"content-type", b"application/json"),
            ],
        }
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), timeout=1)

    asyncio.run(exercise())
    assert received.is_set()
    assert b"generation_error" not in b"".join(frames)
    assert b"[DONE]" in b"".join(frames)
