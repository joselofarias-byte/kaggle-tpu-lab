"""Deployment command contracts, without remote credentials or resources."""
import json
from types import SimpleNamespace

import httpx
import pytest

import operations

TOKEN = "private-controller-token-" + "x" * 48
PASSWORD = "private-bundle-password-for-tests"
URL = "https://controller.example.test"


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("CONTROLLER_TOKEN", raising=False)
    monkeypatch.delenv("CONTROLLER_BUNDLE_PASSWORD", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-modal-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-modal-secret")
    manager = SimpleNamespace(list=lambda: [], create=lambda *a, **kw: calls.append(("secret", a, kw)))
    monkeypatch.setattr(operations.modal, "Secret", SimpleNamespace(objects=manager))
    monkeypatch.setattr(operations.modal.Dict, "from_name", lambda *a, **kw: SimpleNamespace(get=lambda key: None))
    monkeypatch.setattr(operations.modal.Function, "from_name", lambda *a: SimpleNamespace(get_web_url=lambda: URL))
    monkeypatch.setattr(operations, "read_controller_token", lambda: TOKEN)
    monkeypatch.setattr(operations, "verify_controller", lambda *a: calls.append(("verify", a)))
    monkeypatch.setattr(operations.subprocess, "run", lambda *a, **kw: calls.append(("subprocess", a, kw)) or SimpleNamespace(returncode=0))
    return tmp_path / "connection.json", calls


def test_deploy_generates_scoped_secret_and_private_connection(deployment, monkeypatch, capsys):
    output, calls = deployment
    monkeypatch.setenv("CONTROLLER_BUNDLE_PASSWORD", PASSWORD)
    def unexpected_retrieval():
        pytest.fail("New secrets must not attempt remote token retrieval")
    monkeypatch.setattr(operations, "read_controller_token", unexpected_retrieval)
    sizes = []
    monkeypatch.setattr(operations.secrets, "token_urlsafe", lambda size: sizes.append(size) or TOKEN)
    assert operations.main(["deploy", "--output", str(output)]) == 0
    assert sizes == [48]
    assert json.loads(output.read_text()) == {"url": URL, "token": TOKEN}
    assert output.stat().st_mode & 0o777 == 0o600
    assert calls[0] == ("secret", (operations.APP_NAME, {"CONTROLLER_TOKEN": TOKEN}), {"allow_existing": False})
    run = next(c for c in calls if c[0] == "subprocess")
    assert run[1][0] == [operations.sys.executable, "-m", "modal", "deploy", "deploy.py"]
    assert run[2]["cwd"] == operations.DIRECTORY
    assert "CONTROLLER_TOKEN" not in run[2]["env"]
    assert "CONTROLLER_BUNDLE_PASSWORD" not in run[2]["env"]
    assert run[2]["capture_output"] is True
    assert calls[-1] == ("verify", (URL, TOKEN))
    captured = capsys.readouterr()
    assert captured.out.strip() == URL
    assert TOKEN not in str(captured) and PASSWORD not in str(captured)


def test_existing_secret_without_token_reuses_stored_token(deployment, monkeypatch, capsys):
    output, calls = deployment
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    assert operations.main(["deploy", "--output", str(output)]) == 0
    assert [c[0] for c in calls] == ["verify", "subprocess", "verify"]
    assert json.loads(output.read_text()) == {"url": URL, "token": TOKEN}
    assert TOKEN not in str(capsys.readouterr())


def test_existing_secret_is_not_updated_with_supplied_token(deployment, monkeypatch):
    output, calls = deployment
    monkeypatch.setenv("CONTROLLER_TOKEN", TOKEN)
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    assert operations.main(["deploy", "--output", str(output)]) == 0
    assert [c[0] for c in calls] == ["verify", "subprocess", "verify"]


def test_existing_wrong_token_fails_verification_without_changing_secret(deployment, monkeypatch):
    output, calls = deployment
    monkeypatch.setenv("CONTROLLER_TOKEN", TOKEN)
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    def fail(*args):
        raise operations.OperationError("Authenticated controller profiles check failed.")
    monkeypatch.setattr(operations, "verify_controller", fail)
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert not calls and not output.exists()


def test_actions_encrypted_artifact_and_decrypt(deployment, monkeypatch, capsys):
    output, calls = deployment
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CONTROLLER_BUNDLE_PASSWORD", PASSWORD)
    monkeypatch.setenv("CONTROLLER_TOKEN", TOKEN)
    bundle = output.with_suffix(".enc")
    assert operations.main(["deploy", "--encrypted-output", str(bundle)]) == 0
    assert not output.exists()
    assert TOKEN.encode() not in bundle.read_bytes()
    assert operations.decrypt_connection(bundle.read_bytes(), PASSWORD) == {"url": URL, "token": TOKEN}
    monkeypatch.delenv("GITHUB_ACTIONS")
    assert operations.main(["decrypt", str(bundle), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {"url": URL, "token": TOKEN}
    assert output.stat().st_mode & 0o777 == 0o600
    assert TOKEN not in str(capsys.readouterr())


def test_actions_rejects_plaintext(deployment, monkeypatch):
    output, calls = deployment
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert not output.exists() and not calls


def test_password_required_before_remote_mutation(deployment):
    output, calls = deployment
    assert operations.main(["deploy", "--encrypted-output", str(output)]) == 1
    assert not output.exists() and not calls


def test_deploy_failure_suppresses_provider_output_retains_recovery(deployment, monkeypatch, capsys):
    output, calls = deployment
    monkeypatch.setenv("CONTROLLER_TOKEN", TOKEN)
    monkeypatch.setattr(operations.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=TOKEN, stderr=TOKEN))
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert json.loads(output.read_text()) == {"url": None, "token": TOKEN}
    assert TOKEN not in str(capsys.readouterr())


def test_generic_exceptions_never_print_secret(deployment, monkeypatch, capsys):
    output, calls = deployment
    def fail(*args, **kwargs):
        raise RuntimeError(TOKEN)
    monkeypatch.setattr(operations.modal.Secret.objects, "list", fail)
    assert operations.main(["deploy", "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert TOKEN not in str(captured)
    assert "sensitive exception details suppressed" in captured.err


@pytest.mark.parametrize("url", ["http://example.test", "https://user:password@example.test", "https://example.test/path", "https://example.test/?token=x", "https://example.test/#x", "https://example.test/\n"])
def test_reject_unsafe_urls(url):
    with pytest.raises(operations.OperationError):
        operations.validate_url(url)


def test_verify_only_profiles_authentication_and_never_follows_redirects(monkeypatch, tmp_path):
    requests = []
    def handle(request):
        requests.append(request)
        if "authorization" not in request.headers:
            return httpx.Response(401)
        assert request.headers["authorization"] == "Bearer " + TOKEN
        return httpx.Response(200, json={"profiles": [{"id": "qwen-small"}]})
    real_client = httpx.Client
    settings = []
    def client(**kwargs):
        settings.append(kwargs)
        return real_client(transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr(operations.httpx, "Client", client)
    connection = tmp_path / "connection.json"
    connection.write_text(json.dumps({"url": URL, "token": TOKEN}))
    assert operations.main(["verify", "--connection", str(connection)]) == 0
    assert [(r.method, r.url.path) for r in requests] == [("GET", "/api/profiles"), ("GET", "/api/profiles")]
    assert settings == [{"timeout": 60, "follow_redirects": False, "trust_env": False}]


@pytest.mark.parametrize("anonymous,status,payload", [(302, 200, {}), (200, 200, {}), (401, 302, {}), (401, 401, {}), (401, 200, {"profiles": []}), (401, 200, {"profiles": "wrong"})])
def test_verification_fails_closed(monkeypatch, anonymous, status, payload):
    def handle(request):
        return httpx.Response(status if "authorization" in request.headers else anonymous, json=payload)
    real_client = httpx.Client
    monkeypatch.setattr(operations.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    with pytest.raises(operations.OperationError):
        operations.verify_controller(URL, TOKEN)


def test_bundle_rejects_wrong_password_tampering_and_version():
    connection = {"url": URL, "token": TOKEN}
    encrypted = operations.encrypt_connection(connection, PASSWORD)
    assert operations.encrypt_connection(connection, PASSWORD) != encrypted
    with pytest.raises(operations.OperationError):
        operations.decrypt_connection(encrypted, "wrong-password-value")
    data = json.loads(encrypted)
    data["ciphertext"] = data["ciphertext"][:-5] + "AAAA="
    with pytest.raises(operations.OperationError):
        operations.decrypt_connection(json.dumps(data), PASSWORD)
    data["version"] = 2
    with pytest.raises(operations.OperationError):
        operations.decrypt_connection(json.dumps(data), PASSWORD)


def test_private_write_replaces_symlink_without_exposing_token(tmp_path):
    target = tmp_path / "unrelated"
    target.write_text("unchanged")
    output = tmp_path / "connection.json"
    output.symlink_to(target)
    operations.private_write(output, TOKEN.encode())
    assert target.read_text() == "unchanged"
    assert not output.is_symlink()
    assert output.stat().st_mode & 0o777 == 0o600


def test_no_connections_written_inside_repository():
    with pytest.raises(operations.OperationError):
        operations.private_write(operations.DIRECTORY / "connection.json", TOKEN.encode())


def test_decrypt_prompts_without_echo(deployment, monkeypatch):
    output, calls = deployment
    bundle = output.with_suffix(".enc")
    bundle.write_bytes(operations.encrypt_connection({"url": None, "token": TOKEN}, PASSWORD))
    monkeypatch.setattr(operations.getpass, "getpass", lambda prompt: PASSWORD)
    assert operations.main(["decrypt", str(bundle), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["token"] == TOKEN


def test_actions_cannot_decrypt_to_plaintext(deployment, monkeypatch):
    output, calls = deployment
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert operations.main(["decrypt", str(output.with_suffix(".enc")), "--output", str(output)]) == 1
    assert not output.exists()


@pytest.mark.parametrize("status", ["STARTING", "READY", "UNKNOWN", "STOPPING", "INVALID"])
def test_redeploy_refuses_nonterminal_state(deployment, monkeypatch, status):
    output, calls = deployment
    monkeypatch.setattr(operations.modal.Dict, "from_name", lambda *a, **kw: SimpleNamespace(get=lambda key: {"session": {"status": status}}))
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    def unexpected_retrieval():
        pytest.fail("Session guard must run before token retrieval")
    monkeypatch.setattr(operations, "read_controller_token", unexpected_retrieval)
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert not calls and not output.exists()


@pytest.mark.parametrize("status", ["STOPPED", "ERROR"])
def test_redeploy_accepts_terminal_state(deployment, monkeypatch, status):
    output, calls = deployment
    monkeypatch.setattr(operations.modal.Dict, "from_name", lambda *a, **kw: SimpleNamespace(get=lambda key: {"session": {"status": status}}))
    assert operations.main(["deploy", "--output", str(output)]) == 0


def test_recovery_without_deployed_controller_preserves_existing_secret(deployment, monkeypatch):
    output, calls = deployment
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    def missing_state(*a, **kw):
        assert kw == {"create_if_missing": False}
        raise operations.modal.exception.NotFoundError("absent")
    monkeypatch.setattr(operations.modal.Dict, "from_name", missing_state)
    urls = iter([None, URL])
    def lookup():
        value = next(urls)
        if value is None:
            raise operations.modal.exception.NotFoundError("not deployed")
        return value
    monkeypatch.setattr(operations, "controller_url", lookup)
    assert operations.main(["deploy", "--output", str(output)]) == 0
    assert [c[0] for c in calls] == ["subprocess", "verify"]
    assert json.loads(output.read_text()) == {"url": URL, "token": TOKEN}


def test_state_transport_failure_never_allows_deploy(deployment, monkeypatch):
    output, calls = deployment
    def unavailable(*a, **kw):
        raise TimeoutError()
    monkeypatch.setattr(operations.modal.Dict, "from_name", unavailable)
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert not calls and not output.exists()


@pytest.mark.parametrize("key", ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_actions_missing_modal_secrets_fails_before_sdk(deployment, monkeypatch, capsys, key, value):
    output, calls = deployment
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CONTROLLER_BUNDLE_PASSWORD", PASSWORD)
    if value is None:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, value)
    def unexpected_sdk_call():
        pytest.fail("SDK must not be called before Actions credential validation")
    monkeypatch.setattr(operations.modal.Secret.objects, "list", unexpected_sdk_call)
    assert operations.main(["deploy", "--encrypted-output", str(output)]) == 1
    assert not output.exists() and not calls
    captured = capsys.readouterr()
    assert "requires nonempty MODAL_TOKEN_ID and MODAL_TOKEN_SECRET" in captured.err
    assert PASSWORD not in str(captured)


@pytest.fixture
def token_reader(monkeypatch):
    from contextlib import contextmanager

    calls = []
    image, secret = object(), object()

    class FakeApp:
        def function(self, **kwargs):
            calls.append(("function", kwargs))
            def decorate(function):
                def remote():
                    calls.append(("remote",))
                    return function()
                return SimpleNamespace(remote=remote)
            return decorate

        @contextmanager
        def run(self):
            calls.append(("run",))
            try:
                yield
            finally:
                calls.append(("stop",))

    monkeypatch.setattr(operations.modal, "App", FakeApp)
    monkeypatch.setattr(operations.modal.Image, "debian_slim", lambda **kw: calls.append(("image", kw)) or image)
    monkeypatch.setattr(operations.modal.Secret, "from_name", lambda name: calls.append(("secret", name)) or secret)
    def unexpected_output(*args, **kwargs):
        pytest.fail("Token retrieval must not enable Modal output")
    monkeypatch.setattr(operations.modal, "enable_output", unexpected_output)
    return calls, image, secret


def test_read_controller_token_uses_ephemeral_cpu_only_function(token_reader, monkeypatch, capsys):
    calls, image, secret = token_reader
    monkeypatch.setenv("CONTROLLER_TOKEN", TOKEN)
    assert operations.read_controller_token() == TOKEN
    assert calls == [
        ("image", {"python_version": "3.12"}),
        ("secret", operations.APP_NAME),
        ("function", {"image": image, "secrets": [secret], "cpu": 0.125,
                      "memory": 128, "timeout": 60, "max_containers": 1,
                      "serialized": True, "include_source": False}),
        ("run",), ("remote",), ("stop",),
    ]
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


@pytest.mark.parametrize("value", [None, "", "short", "x" * 32 + "\n", "é" * 32])
def test_read_controller_token_rejects_invalid_stored_value(token_reader, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CONTROLLER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CONTROLLER_TOKEN", value)
    with pytest.raises(operations.OperationError, match="at least 32 ASCII"):
        operations.read_controller_token()
    assert token_reader[0][-1] == ("stop",)


def test_read_controller_token_suppresses_provider_failure(token_reader, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError(TOKEN)
    monkeypatch.setattr(operations.modal.Secret, "from_name", fail)
    with pytest.raises(operations.OperationError, match="Cannot retrieve") as error:
        operations.read_controller_token()
    assert TOKEN not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("supplied", ["different-token-" + "y" * 48, "short"])
def test_existing_secret_rejects_conflicting_supplied_token_before_controller_lookup(deployment, monkeypatch, capsys, supplied):
    output, calls = deployment
    monkeypatch.setenv("CONTROLLER_TOKEN", supplied)
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    def unexpected_lookup():
        pytest.fail("Conflicting token must be rejected even without a deployed controller")
    monkeypatch.setattr(operations, "controller_url", unexpected_lookup)
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert not calls and not output.exists()
    assert TOKEN not in str(capsys.readouterr())


def test_existing_token_retrieval_failure_does_not_generate_or_deploy(deployment, monkeypatch, capsys):
    output, calls = deployment
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    def fail():
        raise RuntimeError(TOKEN)
    def unexpected_generation(*args):
        pytest.fail("Retrieval failure must never fall back to rotating credentials")
    monkeypatch.setattr(operations, "read_controller_token", fail)
    monkeypatch.setattr(operations.secrets, "token_urlsafe", unexpected_generation)
    assert operations.main(["deploy", "--output", str(output)]) == 1
    assert not calls and not output.exists()
    assert TOKEN not in str(capsys.readouterr())


@pytest.mark.parametrize("supplied", [None, ""])
def test_actions_redeploy_encrypts_retrieved_token_without_supplied_token(deployment, monkeypatch, capsys, supplied):
    output, calls = deployment
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CONTROLLER_BUNDLE_PASSWORD", PASSWORD)
    if supplied is not None:
        monkeypatch.setenv("CONTROLLER_TOKEN", supplied)
    monkeypatch.setattr(operations.modal.Secret.objects, "list", lambda: [SimpleNamespace(name=operations.APP_NAME)])
    assert operations.main(["deploy", "--encrypted-output", str(output)]) == 0
    assert operations.decrypt_connection(output.read_bytes(), PASSWORD) == {"url": URL, "token": TOKEN}
    assert output.stat().st_mode & 0o777 == 0o600
    assert [c[0] for c in calls] == ["verify", "subprocess", "verify"]
    assert TOKEN.encode() not in output.read_bytes()
    assert TOKEN not in str(capsys.readouterr())


def test_selected_workspace_mismatch_aborts_before_remote_mutations(deployment, monkeypatch, capsys):
    import workspace_check
    output, calls = deployment
    monkeypatch.setenv('EXPECTED_MODAL_WORKSPACE', 'selected-workspace')
    monkeypatch.setattr(workspace_check, 'lookup_workspace', lambda: 'different-workspace')
    assert operations.main(['deploy', '--output', str(output)]) == 1
    assert not calls and not output.exists()
    assert 'Workspace differs' in capsys.readouterr().err


def test_selected_workspace_match_allows_deploy(deployment, monkeypatch):
    import workspace_check
    output, calls = deployment
    monkeypatch.setenv('EXPECTED_MODAL_WORKSPACE', 'selected-workspace')
    monkeypatch.setattr(workspace_check, 'lookup_workspace', lambda: 'selected-workspace')
    assert operations.main(['deploy', '--output', str(output)]) == 0
    assert output.exists()
