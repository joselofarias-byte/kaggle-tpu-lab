"""Workspace preflight contracts; all Modal interactions are mocked."""

import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import workspace_check


REQUEST_ID = "0123456789abcdef" * 2
PASSWORD = "test-password-never-uploaded"


@pytest.fixture
def lookup(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-token-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-token-secret")
    monkeypatch.setenv("CONTROLLER_BUNDLE_PASSWORD", PASSWORD)
    lookup = AsyncMock(return_value=SimpleNamespace(username="test-workspace"))
    monkeypatch.setattr(workspace_check, "_lookup_workspace", lookup)
    monkeypatch.setattr(workspace_check, "config", SimpleNamespace(get=lambda key: "https://modal.test"))
    manager = SimpleNamespace(list=Mock(return_value=[SimpleNamespace(name="android-modal-control")]))
    monkeypatch.setattr(workspace_check.modal, "Secret", SimpleNamespace(objects=manager))
    return lookup, manager


def test_token_scoped_workspace_and_password_proof(lookup, tmp_path, capsys):
    remote, manager = lookup
    output = tmp_path / "workspace.json"
    assert workspace_check.main(["--request-id", REQUEST_ID, "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {
        "workspaces": ["test-workspace"],
        "controller_secret_exists": True,
        "password_proof": hmac.new(PASSWORD.encode(), REQUEST_ID.encode(), hashlib.sha256).hexdigest(),
        "request_id": REQUEST_ID,
    }
    assert output.stat().st_mode & 0o777 == 0o600
    remote.assert_awaited_once_with("https://modal.test", "test-token-id", "test-token-secret")
    manager.list.assert_called_once_with()
    assert capsys.readouterr() == ("", "")
    assert PASSWORD not in output.read_text()
    assert "test-token-secret" not in output.read_text()


def test_missing_controller_secret_is_reported_without_creating_it(lookup):
    _, manager = lookup
    manager.list.return_value = [SimpleNamespace(name="unrelated")]
    assert workspace_check.check_workspace(REQUEST_ID)["controller_secret_exists"] is False


@pytest.mark.parametrize("request_id", ["", "a" * 31, "a" * 33, "g" * 32, "a" * 32 + "\n", "$(echo unsafe)"])
def test_invalid_request_rejected_before_remote_access(lookup, request_id):
    remote, manager = lookup
    with pytest.raises(ValueError):
        workspace_check.check_workspace(request_id)
    remote.assert_not_called()
    manager.list.assert_not_called()


@pytest.mark.parametrize("key,value", [
    ("MODAL_TOKEN_ID", ""), ("MODAL_TOKEN_SECRET", ""), ("MODAL_TOKEN_ID", "  "),
    ("CONTROLLER_BUNDLE_PASSWORD", ""), ("CONTROLLER_BUNDLE_PASSWORD", "x" * 15),
])
def test_credentials_validated_before_remote_access(lookup, monkeypatch, key, value):
    remote, manager = lookup
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        workspace_check.check_workspace(REQUEST_ID)
    remote.assert_not_called()
    manager.list.assert_not_called()


@pytest.mark.parametrize("failing_call", ["lookup", "list"])
def test_remote_errors_are_redacted(lookup, tmp_path, capsys, failing_call):
    remote, manager = lookup
    (remote if failing_call == "lookup" else manager.list).side_effect = RuntimeError(PASSWORD)
    output = tmp_path / "workspace.json"
    assert workspace_check.main(["--request-id", REQUEST_ID, "--output", str(output)]) == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Workspace check failed; no secrets were changed.\n"


def test_empty_workspace_is_rejected(lookup):
    remote, manager = lookup
    remote.return_value = SimpleNamespace(username="")
    with pytest.raises(ValueError):
        workspace_check.check_workspace(REQUEST_ID)
    manager.list.assert_not_called()


def test_private_write_replaces_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target"
    target.write_text("unchanged")
    output = tmp_path / "workspace.json"
    output.symlink_to(target)
    workspace_check.private_write(output, {"request_id": REQUEST_ID})
    assert target.read_text() == "unchanged"
    assert not output.is_symlink()
    assert output.stat().st_mode & 0o777 == 0o600


def test_password_proof_changes_with_challenge(lookup):
    first = workspace_check.check_workspace(REQUEST_ID)
    second = workspace_check.check_workspace("f" * 32)
    assert first["password_proof"] != second["password_proof"]


def test_write_failure_is_redacted(lookup, tmp_path, capsys):
    output = tmp_path / "missing" / "workspace.json"
    assert workspace_check.main(["--request-id", REQUEST_ID, "--output", str(output)]) == 1
    assert capsys.readouterr().err == "Workspace check failed; no secrets were changed.\n"
