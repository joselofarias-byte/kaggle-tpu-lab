"""SDK adapter contracts, without connecting to Modal or building images."""
from types import SimpleNamespace

import modal
import pytest

import deploy


@pytest.mark.parametrize("profile,gpu,cpu,memory", [("qwen-small", "T4", 2, 8192), ("qwen-7b", "L40S", 4, 16384)])
def test_create_allowlisted_resources_and_hard_timeout(monkeypatch, profile, gpu, cpu, memory):
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(object_id="sb-owned")

    monkeypatch.setattr(modal.Sandbox, "create", create)
    monkeypatch.setattr(deploy.ModalBackend, "find", lambda self, name: None)
    record = {"name": deploy.ModalBackend.name, "session": {"profile_id": profile, "max_minutes": 17, "idle_minutes": 3}, "gateway_token": "private-gateway-only"}
    assert deploy.ModalBackend().create(record) == "sb-owned"
    args, kwargs = calls[0]
    assert args == ("python3", "/opt/worker.py")
    assert (kwargs["gpu"], kwargs["cpu"], kwargs["memory"]) == (gpu, (cpu, cpu), (memory, memory))
    assert kwargs["timeout"] == 17 * 60
    assert kwargs["name"] == deploy.ModalBackend.name
    assert kwargs["encrypted_ports"] == [8080]
    assert kwargs["env"]["IDLE_SECONDS"] == "180"
    assert "secrets" not in kwargs
    assert "CONTROLLER_TOKEN" not in kwargs["env"]


def test_stop_attaches_only_persisted_owned_handle(monkeypatch):
    stopped = []
    monkeypatch.setattr(modal.Sandbox, "from_id", lambda sandbox_id: SimpleNamespace(terminate=lambda **kwargs: stopped.append((sandbox_id, kwargs))))
    deploy.ModalBackend().stop({"sandbox_id": "sb-owned"})
    assert stopped == [("sb-owned", {"wait": True})]


def test_lookup_only_in_dedicated_app_and_notfound_is_not_transport_error(monkeypatch):
    calls = []

    def find(app, name):
        calls.append((app, name))
        raise modal.exception.NotFoundError("absent")

    monkeypatch.setattr(modal.Sandbox, "from_name", find)
    backend = deploy.ModalBackend()
    assert backend.find(backend.name) is None
    assert calls == [(deploy.APP_NAME, backend.name)]
    monkeypatch.setattr(modal.Sandbox, "from_name", lambda *args: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):
        backend.find(backend.name)


def test_worker_image_is_a_deployed_dependency_not_a_runtime_local_build():
    assert any(image is deploy.gpu_image for image in deploy.controller.deps(only_explicit_mounts=True))


def test_discovery_before_create_never_launches_over_existing_name(monkeypatch):
    backend = deploy.ModalBackend()
    monkeypatch.setattr(backend, "find", lambda name: "sb-already-owned")
    monkeypatch.setattr(modal.Sandbox, "create", lambda *args, **kwargs: pytest.fail("must discover before creation"))
    with pytest.raises(RuntimeError):
        backend.create({"name": backend.name})


def test_deployment_has_concurrent_inputs_but_one_container():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(deploy.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "controller")
    decorators = {ast.unparse(node.func): {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg in {"max_inputs", "max_containers"}} for node in function.decorator_list if isinstance(node, ast.Call)}
    assert decorators["modal.concurrent"]["max_inputs"] == 8
    assert decorators["app.function"]["max_containers"] == 1
