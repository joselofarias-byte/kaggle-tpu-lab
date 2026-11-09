import json
from types import SimpleNamespace

import httpx
import pytest

import infer


def args(tmp_path, **changes):
    values = dict(accept_cost=True, profile="qwen-small", max_minutes=5,
                  gpu_hourly_usd="1", connection=None, url="https://controller.example",
                  journal=str(tmp_path / "journal.json"))
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("changes", [{"accept_cost": False}, {"max_minutes": 121},
                                      {"max_minutes": 4}, {"gpu_hourly_usd": "NaN"},
                                      {"gpu_hourly_usd": "-1"}, {"profile": "arbitrary"}])
def test_cost_validation_precedes_requests(tmp_path, changes):
    with pytest.raises(ValueError):
        infer.run(args(tmp_path, **changes), client_factory=lambda **_: pytest.fail("network"))


def test_success_stops_after_one_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTROLLER_TOKEN", "x" * 48)
    calls, stopped = [], []
    session = {"id": "ours", "status": "READY"}

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"session": None})
        if request.url.path.endswith("chat"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return httpx.Response(200, json={"session": session})

    monkeypatch.setattr(infer, "cleanup", lambda journal: stopped.append(json.loads(open(journal).read())))
    infer.run(args(tmp_path), client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs))
    assert calls == [("GET", "/api/session"), ("POST", "/api/session"), ("POST", "/api/chat")]
    assert len(stopped) == 1
    assert stopped[0]["attempted"]
    assert "token" not in stopped[0]
    assert (tmp_path / "journal.json").stat().st_mode & 0o777 == 0o600


def test_ambiguous_start_still_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTROLLER_TOKEN", "x" * 48)
    stopped = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session": None})
        raise httpx.ReadTimeout("response lost")

    monkeypatch.setattr(infer, "cleanup", stopped.append)
    with pytest.raises(httpx.ReadTimeout):
        infer.run(args(tmp_path), client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs))
    assert stopped == [str(tmp_path / "journal.json")]


def test_existing_session_is_never_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTROLLER_TOKEN", "x" * 48)
    monkeypatch.setattr(infer, "cleanup", lambda _: pytest.fail("stopping someone else"))
    handler = lambda request: httpx.Response(200, json={"session": {"status": "UNKNOWN"}})
    with pytest.raises(RuntimeError):
        infer.run(args(tmp_path), client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs))
    assert not (tmp_path / "journal.json").exists()


def test_provider_cleanup_uses_owned_persisted_handle(tmp_path, monkeypatch):
    path = tmp_path / "journal.json"
    infer.write_journal(path, {"request_id": "ours", "attempted": True})
    store = {"current": {"request_id": "ours", "sandbox_id": "sb-ours", "session": {"status": "READY"}}}
    calls = []
    live = [True]
    sandbox = SimpleNamespace(poll=lambda: None if live[0] else 0,
                              terminate=lambda **_: live.__setitem__(0, False))
    monkeypatch.setattr(infer.modal.Dict, "from_name", lambda _: store)
    monkeypatch.setattr(infer.modal.Sandbox, "from_id", lambda identity: (calls.append(identity), sandbox)[1])
    infer.cleanup(path, pause=lambda _: None)
    assert calls == ["sb-ours"]
    assert live == [False]


def test_cleanup_never_stops_other_request(tmp_path, monkeypatch):
    path = tmp_path / "journal.json"
    infer.write_journal(path, {"request_id": "ours", "attempted": True})
    store = {"current": {"request_id": "theirs", "sandbox_id": "sb-theirs"}}
    monkeypatch.setattr(infer.modal.Dict, "from_name", lambda _: store)
    monkeypatch.setattr(infer.modal.Sandbox, "from_id", lambda _: pytest.fail("wrong handle"))
    with pytest.raises(RuntimeError):
        infer.cleanup(path, attempts=2, pause=lambda _: None)


def test_cleanup_missing_journal_does_not_contact_modal(tmp_path, monkeypatch):
    monkeypatch.setattr(infer.modal.Dict, "from_name", lambda _: pytest.fail("network"))
    infer.cleanup(tmp_path / "absent")


def test_chat_error_still_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTROLLER_TOKEN", "x" * 48)
    stopped = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"session": None})
        if request.url.path.endswith("chat"):
            return httpx.Response(502)
        return httpx.Response(200, json={"session": {"id": "ours", "status": "READY"}})

    monkeypatch.setattr(infer, "cleanup", stopped.append)
    with pytest.raises(httpx.HTTPStatusError):
        infer.run(args(tmp_path), client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs))
    assert stopped == [str(tmp_path / "journal.json")]


def test_model_readiness_timeout_still_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTROLLER_TOKEN", "x" * 48)
    stopped = []
    ticks = iter([0, 301])

    def handler(request):
        return httpx.Response(200, json={"session": None if request.method == "GET" else {"id": "ours", "status": "STARTING"}})

    monkeypatch.setattr(infer, "cleanup", stopped.append)
    with pytest.raises(RuntimeError):
        infer.run(args(tmp_path), clock=lambda: next(ticks), client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs))
    assert stopped == [str(tmp_path / "journal.json")]


def test_historical_ambiguous_creation_cannot_stop_new_sandbox(tmp_path, monkeypatch):
    path = tmp_path / "journal.json"
    infer.write_journal(path, {"request_id": "ours", "attempted": True})
    store = {"current": {"request_id": "theirs"}, "request:ours": {
        "request_id": "ours", "sandbox_id": None, "session": {"status": "UNKNOWN"}}}
    monkeypatch.setattr(infer.modal.Dict, "from_name", lambda _: store)
    monkeypatch.setattr(infer.modal.Sandbox, "from_name", lambda *_: pytest.fail("wrong named sandbox"))
    with pytest.raises(RuntimeError):
        infer.cleanup(path, pause=lambda _: None)


def test_connection_recovers_modal_secret_when_env_absent(monkeypatch):
    import operations
    monkeypatch.delenv("CONTROLLER_TOKEN", raising=False)
    monkeypatch.setattr(operations, "read_controller_token", lambda: "x" * 48)
    assert infer.connection(url="https://example.modal.run")[1] == "x" * 48
