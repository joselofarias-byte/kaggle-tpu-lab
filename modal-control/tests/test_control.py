import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from control import make_app, redact
from worker import Activity

TOKEN = "test-controller-token-with-at-least-32-characters"


class FakeBackend:
    name = "android-owned-model-v1"

    def __init__(self):
        self.creates = 0
        self.live = False
        self.fail_create = False
        self.fail_stop = False
        self.fail_poll = False
        self.is_ready = True
        self.chats = 0

    def create(self, record):
        self.creates += 1
        self.live = True
        if self.fail_create:
            raise TimeoutError("response lost after allocation")
        return "sb-owned"

    def find(self, name):
        assert name == self.name
        return "sb-owned" if self.live else None

    def poll(self, record):
        assert record["sandbox_id"] == "sb-owned"
        if self.fail_poll:
            raise TimeoutError()
        return None if self.live else 0

    def ready(self, record):
        return self.is_ready

    def stop(self, record):
        assert record["sandbox_id"] == "sb-owned"
        if self.fail_stop:
            raise TimeoutError()
        self.live = False

    def logs(self, record):
        return ["Bearer secret-value", TOKEN, record["gateway_token"], "https://private.modal.run"]

    def chat(self, record, body):
        self.chats += 1
        return {"choices": [{"message": {"content": "Hola"}}]}


@pytest.fixture
def env():
    store, backend = {}, FakeBackend()
    client = TestClient(make_app(store, backend, TOKEN))
    client.headers["Authorization"] = "Bearer " + TOKEN
    return client, store, backend


def payload(**changes):
    return {"profile_id": "qwen-small", "max_minutes": 30, "idle_minutes": 5, "request_id": str(uuid4()), **changes}


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/api/profiles", None), ("GET", "/api/session", None),
    ("POST", "/api/session", payload()), ("DELETE", "/api/session", None),
    ("GET", "/api/logs", None), ("POST", "/api/chat", {"messages": [{"role": "user", "content": "hi"}]}),
])
def test_every_endpoint_requires_auth(env, method, path, body):
    client, _, backend = env
    for auth in ("", "Bearer wrong", "Basic " + TOKEN):
        assert client.request(method, path, json=body, headers={"Authorization": auth}).status_code == 401
    assert backend.creates == 0


@pytest.mark.parametrize("change", [
    {"max_minutes": 4}, {"max_minutes": 121}, {"max_minutes": True}, {"max_minutes": 5.5},
    {"idle_minutes": 1}, {"idle_minutes": 31}, {"idle_minutes": "2"},
    {"request_id": "invalid"}, {"profile_id": "arbitrary-model"}, {"gpu": "H100"},
])
def test_invalid_start_cannot_allocate(env, change):
    client, _, backend = env
    assert client.post("/api/session", json=payload(**change)).status_code == 422
    assert backend.creates == 0


def test_profile_contract(env):
    client, _, backend = env
    profiles = client.get("/api/profiles").json()["profiles"]
    assert [(p["id"], p["gpu"], p["cpu"], p["memory_mib"]) for p in profiles] == [("qwen-small", "T4", 2, 8192), ("qwen-7b", "L40S", 4, 16384)]
    assert backend.creates == 0


def test_idempotent_start_and_conflicting_request(env):
    client, _, backend = env
    body = payload()
    first = client.post("/api/session", json=body).json()["session"]
    repeated = client.post("/api/session", json=body).json()["session"]
    assert repeated["id"] == first["id"]
    assert backend.creates == 1
    assert (datetime.fromisoformat(first["expires_at"]) - datetime.fromisoformat(first["created_at"])).total_seconds() == 1800
    assert client.post("/api/session", json={**body, "max_minutes": 31}).status_code == 409
    assert client.post("/api/session", json=payload()).status_code == 409
    assert backend.creates == 1


def test_concurrent_starts_create_one_resource(env):
    client, _, backend = env
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(lambda _: client.post("/api/session", json=payload()).status_code, range(4)))
    assert sorted(codes) == [200, 409, 409, 409]
    assert backend.creates == 1


def test_transport_failure_recovers_identity_after_restart(env):
    client, store, backend = env
    backend.fail_create = True
    body = payload()
    assert client.post("/api/session", json=body).json()["session"]["status"] == "UNKNOWN"
    restarted = TestClient(make_app(copy.deepcopy(store), backend, TOKEN), headers={"Authorization": "Bearer " + TOKEN})
    assert restarted.get("/api/session").json()["session"]["status"] == "READY"
    assert restarted.post("/api/session", json=body).status_code == 200
    assert restarted.delete("/api/session").json()["session"]["status"] == "STOPPED"
    assert backend.creates == 1


def test_unconfirmed_creation_fails_closed(env):
    client, store, backend = env
    backend.fail_create = True
    client.post("/api/session", json=payload())
    backend.live = False
    assert client.get("/api/session").json()["session"]["status"] == "UNKNOWN"
    assert client.delete("/api/session").json()["session"]["status"] == "UNKNOWN"
    assert client.post("/api/session", json=payload()).status_code == 409
    assert not store["current"]["sandbox_id"]
    assert backend.creates == 1


def test_stop_failure_keeps_handle_and_can_retry(env):
    client, store, backend = env
    client.post("/api/session", json=payload())
    backend.fail_stop = True
    assert client.delete("/api/session").json()["session"]["status"] == "UNKNOWN"
    assert store["current"]["sandbox_id"] == "sb-owned"
    assert client.get("/api/session").json()["session"]["status"] == "STOPPING"
    assert client.post("/api/session", json=payload()).status_code == 409
    backend.fail_stop = False
    assert client.delete("/api/session").json()["session"]["status"] == "STOPPED"


def test_poll_failure_preserves_identity(env):
    client, store, backend = env
    client.post("/api/session", json=payload())
    backend.fail_poll = True
    assert client.get("/api/session").json()["session"]["status"] == "UNKNOWN"
    assert store["current"]["sandbox_id"] == "sb-owned"
    assert client.post("/api/session", json=payload()).status_code == 409


def test_polling_logs_chat_and_stop_never_allocate(env):
    client, _, backend = env
    for _ in range(3):
        assert client.get("/api/session").json() == {"session": None}
        assert client.get("/api/logs").json() == {"lines": []}
        assert client.delete("/api/session").json() == {"session": None}
        assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 409
    assert backend.creates == 0


def test_chat_contract_and_limits(env):
    client, _, backend = env
    client.post("/api/session", json=payload())
    good = {"messages": [{"role": "user", "content": "Hola"}], "max_tokens": 1024}
    assert client.post("/api/chat", json=good).json() == {"choices": [{"message": {"content": "Hola"}}]}
    for invalid in ({**good, "max_tokens": 1025}, {**good, "max_tokens": 0}, {**good, "max_tokens": True}, {**good, "messages": []}, {**good, "messages": [{"role": "tool", "content": "hi"}]}):
        assert client.post("/api/chat", json=invalid).status_code == 422
    assert backend.chats == 1


def test_logs_redact_tokens_and_tunnels(env):
    client, store, _ = env
    client.post("/api/session", json=payload())
    lines = client.get("/api/logs").json()["lines"]
    assert TOKEN not in str(lines)
    assert store["current"]["gateway_token"] not in str(lines)
    assert "https://" not in str(lines)
    assert "secret-value" not in str(lines)
    assert "secret123" not in redact("api_key=secret123")


def test_old_request_never_relaunches_after_stop(env):
    client, _, backend = env
    body = payload()
    original = client.post("/api/session", json=body).json()["session"]
    client.delete("/api/session")
    assert client.post("/api/session", json=body).json()["session"]["status"] == "STOPPED"
    assert backend.creates == 1
    new = client.post("/api/session", json=payload()).json()["session"]
    assert new["id"] != original["id"]
    assert backend.creates == 2
    assert client.post("/api/session", json=body).json()["session"]["status"] == "STOPPED"
    assert backend.creates == 2


def test_watchdog_only_inference_extends_idle():
    now = [0]
    activity = Activity(lambda: now[0])
    activity.arm()
    now[0] = 119
    assert not activity.expired(120)
    now[0] = 120
    assert activity.expired(120)
    activity.begin()
    now[0] = 600
    assert not activity.expired(120)
    activity.end()
    now[0] = 719
    assert not activity.expired(120)
    now[0] = 720
    assert activity.expired(120)


def test_missing_server_secret_fails_closed():
    with pytest.raises(ValueError):
        make_app({}, FakeBackend(), "")


def test_retry_recovers_if_crash_between_current_and_request_writes(env):
    client, store, backend = env
    body = payload()
    first = client.post("/api/session", json=body).json()["session"]
    del store["request:" + body["request_id"]]
    assert client.post("/api/session", json=body).json()["session"]["id"] == first["id"]
    assert backend.creates == 1


def test_terminate_acknowledgement_is_not_confirmation(env):
    client, store, backend = env
    client.post("/api/session", json=payload())
    backend.stop = lambda record: None
    assert client.delete("/api/session").json()["session"]["status"] == "STOPPING"
    assert store["current"]["sandbox_id"] == "sb-owned"
    assert client.post("/api/session", json=payload()).status_code == 409


def test_idle_cannot_exceed_absolute_limit(env):
    client, _, backend = env
    assert client.post("/api/session", json=payload(max_minutes=5, idle_minutes=6)).status_code == 422
    assert backend.creates == 0
    assert client.post("/api/session", json=payload(max_minutes=5, idle_minutes=5)).status_code == 200


@pytest.mark.parametrize("origin", ["http://localhost", "https://localhost", "capacitor://localhost", "http://localhost:5173"])
def test_cors_capacitor_and_explicit_browser_origin(env, origin):
    client, _, _ = env
    response = client.options("/api/session", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-credentials" not in response.headers
    assert client.get("/api/profiles", headers={"Origin": origin, "Authorization": ""}).status_code == 401


def test_cors_denies_unlisted_origins(env):
    client, _, _ = env
    response = client.options("/api/session", headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"})
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_status_and_stop_complete_while_inference_is_blocked(env):
    import threading

    client, _, backend = env
    client.post("/api/session", json=payload())
    entered, release = threading.Event(), threading.Event()

    def blocking_chat(record, body):
        entered.set()
        assert release.wait(5)
        return {"choices": [{"message": {"content": "done"}}]}

    backend.chat = blocking_chat
    with ThreadPoolExecutor(max_workers=3) as pool:
        chat = pool.submit(client.post, "/api/chat", json={"messages": [{"role": "user", "content": "hello"}]})
        try:
            assert entered.wait(2)
            assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "second"}]}).status_code == 429
            status = pool.submit(client.get, "/api/session").result(timeout=2)
            assert status.json()["session"]["status"] == "READY"
            stopped = pool.submit(client.delete, "/api/session").result(timeout=2)
            assert stopped.json()["session"]["status"] == "STOPPED"
            assert not chat.done()
        finally:
            release.set()
        assert chat.result(timeout=2).status_code == 200


def test_watchdog_startup_grace_arms_once_on_internal_readiness():
    now = [0]
    activity = Activity(lambda: now[0])
    now[0] = 3600
    assert not activity.expired(120)
    activity.arm()
    now[0] = 3719
    assert not activity.expired(120)
    activity.arm()
    now[0] = 3720
    assert activity.expired(120)


def test_terminal_sessions_preserve_required_fields(env):
    client, _, _ = env
    first = client.post("/api/session", json=payload()).json()["session"]
    stopped = client.delete("/api/session").json()["session"]
    required = {"id", "profile_id", "model", "gpu", "cpu", "memory_mib", "status", "created_at", "expires_at", "idle_minutes", "max_minutes"}
    assert required <= stopped.keys()
    assert {key: stopped[key] for key in required - {"status"}} == {key: first[key] for key in required - {"status"}}


def test_internal_readiness_loop_arms_after_health_success_only():
    from types import SimpleNamespace

    import httpx
    from worker import wait_for_readiness

    now = [0]
    activity = Activity(lambda: now[0])
    replies = iter([httpx.ConnectError("starting"), 503, 200])
    checks = []

    def get(path):
        assert path == "/health"
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(status_code=reply)

    def sleep(seconds):
        now[0] += 600
        checks.append(activity.expired(120))

    wait_for_readiness(activity, SimpleNamespace(poll=lambda: None), SimpleNamespace(get=get), sleep)
    assert checks == [False, False]
    assert activity.last == 1200
    now[0] = 1320
    assert activity.expired(120)


def test_startup_chat_does_not_arm_watchdog():
    activity = Activity(lambda: 1000)
    assert not activity.begin()
    assert activity.last is None
    assert not activity.expired(120)
