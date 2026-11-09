import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'build_llama_cuda.sh'


@pytest.fixture
def build_lab(tmp_path):
    source = tmp_path / 'upstream'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / 'tracked').write_text('original\n')
    (source / '.gitignore').write_text('build-cuda/\n')
    subprocess.run(['git', '-C', str(source), 'add', 'tracked', '.gitignore'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    script = tmp_path / 'build.sh'
    script.write_text(SCRIPT.read_text().replace(
        'https://github.com/ggml-org/llama.cpp.git', str(source)
    ).replace('cd959bae99ff4bed22bef2604f45241c09bbb8c8', revision))
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    for name in ('nvcc', 'nvidia-smi'):
        executable = binaries / name
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
    cmake = binaries / 'cmake'
    cmake.write_text('''#!/bin/sh
set -eu
if [ "$1" = --build ]; then
  mkdir -p "$2/bin"
  printf '#!/bin/sh\nexit 0\n' > "$2/bin/llama-server"
  chmod +x "$2/bin/llama-server"
fi
''')
    cmake.chmod(0o755)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'])
    base = tmp_path / 'cache'

    def run():
        return subprocess.run(['bash', str(script), str(base)], env=env,
                              capture_output=True, text=True, timeout=15)

    return run, base / 'llama.cpp', revision


def test_fresh_clone_and_clean_reuse(build_lab):
    run, checkout, revision = build_lab
    for _ in range(2):
        result = run()
        assert result.returncode == 0, result.stderr
        assert (checkout / 'tracked').read_text() == 'original\n'
        assert subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True
        ).strip() == revision


def test_existing_dirty_checkout_is_preserved(build_lab):
    run, checkout, _ = build_lab
    assert run().returncode == 0
    tracked = checkout / 'tracked'
    tracked.write_text('local edits\n')
    result = run()
    assert result.returncode != 0
    assert 'Refusing to overwrite a modified llama.cpp checkout' in result.stderr
    assert tracked.read_text() == 'local edits\n'
