import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "termux-deploy.sh"


@pytest.fixture
def shell_env(tmp_path):
    if not shutil.which("openssl"):
        pytest.skip("OpenSSL is required for the Termux script test")
    home = tmp_path / "home"
    home.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$HOME/calls"
if [[ "$1 $2" == 'secret set' ]]; then
  cat > "$HOME/$3"
fi
''')
    gh.chmod(0o700)
    return dict(os.environ, HOME=str(home), PATH=f"{binary}:{os.environ['PATH']}")


def execute(env, confirmation="DESPLEGAR"):
    return subprocess.run(["bash", str(SCRIPT)], env=env,
                          input=f"dummy-id\ndummy-api-secret\n{confirmation}\n",
                          text=True, capture_output=True)


def test_configures_secrets_privately_and_dispatches_exact_branch(shell_env):
    result = execute(shell_env)
    assert result.returncode == 0, result.stderr
    home = Path(shell_env["HOME"])
    password = home / ".config/kaggle-modal-actions/bundle-password"
    assert stat.S_IMODE(password.stat().st_mode) == 0o600
    assert stat.S_IMODE(password.parent.stat().st_mode) == 0o700
    assert (home / "CONTROLLER_BUNDLE_PASSWORD").read_bytes() == password.read_bytes()
    assert (home / "MODAL_TOKEN_ID").read_text() == "dummy-id"
    assert (home / "MODAL_TOKEN_SECRET").read_text() == "dummy-api-secret"
    calls = (home / "calls").read_text()
    assert 'workflow run mobile-android-es.yml' in calls
    assert '--ref hoplite/kalymna-ddcfd6fb' in calls
    assert '-f modal_deploy=true -f validate_only=false -f acknowledge_cost=true' in calls
    assert 'inference' not in calls and 'secret set CONTROLLER_TOKEN ' not in calls
    for value in ["dummy-id", "dummy-api-secret", password.read_text().strip()]:
        assert value not in result.stdout + result.stderr + calls
    original = password.read_bytes()
    assert execute(shell_env).returncode == 0
    assert password.read_bytes() == original


def test_declining_cost_does_not_dispatch(shell_env):
    assert execute(shell_env, "NO").returncode == 0
    assert 'workflow run' not in (Path(shell_env["HOME"]) / 'calls').read_text()


def test_rejects_symlink_password(shell_env):
    home = Path(shell_env["HOME"])
    state = home / '.config/kaggle-modal-actions'
    state.mkdir(parents=True)
    target = home / 'unrelated'
    target.write_text('unchanged')
    (state / 'bundle-password').symlink_to(target)
    assert execute(shell_env).returncode != 0
    assert target.read_text() == 'unchanged'
    assert 'secret set' not in (home / 'calls').read_text()
