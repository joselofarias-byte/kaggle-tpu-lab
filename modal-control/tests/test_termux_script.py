import json
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
    gh.write_text('''#!/usr/bin/env python3
import hashlib, hmac, json, os, pathlib, sys
args = sys.argv[1:]
home = pathlib.Path.home()
with (home / 'calls').open('a') as stream:
    stream.write(' '.join(args) + '\\n')
def arg(name):
    return args[args.index(name) + 1]
if args[:2] == ['secret', 'list']:
    if os.environ.get('LIST_FAIL'):
        sys.exit(1)
    print('\\n'.join(p.name for p in home.glob('MODAL_*')))
    if (home / 'CONTROLLER_BUNDLE_PASSWORD').exists():
        print('CONTROLLER_BUNDLE_PASSWORD')
elif args[:2] == ['secret', 'set']:
    (home / args[2]).write_text(sys.stdin.read())
elif args[:2] == ['workflow', 'run'] and args[2] == 'modal-workspace.yml':
    (home / 'request').write_text(next(x.split('=', 1)[1] for x in args if x.startswith('request_id=')))
elif args[:2] == ['run', 'list'] and 'modal-workspace.yml' in args:
    print('123')
elif args[:2] == ['run', 'watch'] and os.environ.get('WATCH_FAIL'):
    sys.exit(1)
elif args[:2] == ['run', 'download']:
    request = (home / 'request').read_text()
    password = (home / 'CONTROLLER_BUNDLE_PASSWORD').read_text().rstrip('\\r\\n')
    record = dict(request_id=request, workspaces=['test-workspace'], controller_secret_exists=True,
                  password_proof=hmac.new(password.encode(), request.encode(), hashlib.sha256).hexdigest())
    (pathlib.Path(arg('--dir')) / 'workspace.json').write_text(json.dumps(record))
''')
    gh.chmod(0o700)
    return dict(os.environ, HOME=str(home), PATH=f"{binary}:{os.environ['PATH']}")


def execute(env, answers="dummy-id\ndummy-api-secret\n1\nDESPLEGAR\n"):
    return subprocess.run(["bash", str(SCRIPT)], env=env, input=answers,
                          text=True, capture_output=True)


def seed(env, local=True, matching=True):
    home = Path(env['HOME'])
    for name in ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'CONTROLLER_BUNDLE_PASSWORD']:
        (home / name).write_text('original-password-' + name)
    if local:
        state = home / '.config/kaggle-modal-actions'
        state.mkdir(parents=True)
        value = (home / 'CONTROLLER_BUNDLE_PASSWORD').read_text() if matching else 'wrong-local-password'
        (state / 'bundle-password').write_text(value)
    return home


def test_configures_missing_secrets_privately_validates_workspace_and_dispatches_main(shell_env):
    result = execute(shell_env)
    assert result.returncode == 0, result.stderr
    home = Path(shell_env["HOME"])
    password = home / ".config/kaggle-modal-actions/bundle-password"
    assert stat.S_IMODE(password.stat().st_mode) == 0o600
    assert stat.S_IMODE(password.parent.stat().st_mode) == 0o700
    assert (home / "CONTROLLER_BUNDLE_PASSWORD").read_text() == password.read_text().rstrip("\r\n")
    calls = (home / "calls").read_text()
    assert 'workflow run modal-deploy.yml' in calls
    assert '--ref main' in calls
    assert '-f validate_only=false -f acknowledge_cost=true -f expected_workspace=test-workspace' in calls
    assert 'inference' not in calls and 'secret set CONTROLLER_TOKEN ' not in calls
    assert '1) test-workspace' in result.stdout
    assert 'existente; se conservará' in result.stdout
    for value in ["dummy-id", "dummy-api-secret", password.read_text().strip()]:
        assert value not in result.stdout + result.stderr + calls


def test_existing_secrets_are_never_written_and_local_password_preserved(shell_env):
    home = seed(shell_env)
    original = {p.name: p.read_bytes() for p in home.glob('MODAL_*')}
    password = home / '.config/kaggle-modal-actions/bundle-password'
    before = password.read_bytes()
    result = execute(shell_env, '1\nDESPLEGAR\n')
    assert result.returncode == 0, result.stderr
    assert 'secret set' not in (home / 'calls').read_text()
    assert password.read_bytes() == before
    assert all((home / name).read_bytes() == value for name, value in original.items())


def test_existing_password_without_local_original_aborts_before_writes(shell_env):
    home = seed(shell_env, local=False)
    result = execute(shell_env)
    assert result.returncode != 0
    assert 'ORIGINAL' in result.stderr and 'GitHub no permite leer' in result.stderr
    assert not (home / '.config/kaggle-modal-actions/bundle-password').exists()
    assert 'secret set' not in (home / 'calls').read_text()
    assert 'workflow run' not in (home / 'calls').read_text()


def test_wrong_local_password_aborts_without_overwriting_or_deploying(shell_env):
    home = seed(shell_env, matching=False)
    result = execute(shell_env)
    assert result.returncode != 0
    assert 'no corresponde' in result.stderr
    calls = (home / 'calls').read_text()
    assert 'secret set' not in calls and 'workflow run modal-deploy.yml' not in calls


@pytest.mark.parametrize('answers', ['NO\n', '1\nNO\n'])
def test_declining_destination_or_cost_does_not_deploy(shell_env, answers):
    home = seed(shell_env)
    assert execute(shell_env, answers).returncode == 0
    assert 'workflow run modal-deploy.yml' not in (home / 'calls').read_text()


@pytest.mark.parametrize('failure', ['LIST_FAIL', 'WATCH_FAIL'])
def test_metadata_failure_never_deploys(shell_env, failure):
    home = seed(shell_env)
    shell_env[failure] = '1'
    assert execute(shell_env).returncode != 0
    calls = (home / 'calls').read_text()
    assert 'secret set' not in calls and 'workflow run modal-deploy.yml' not in calls


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
