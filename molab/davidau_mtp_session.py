"""Run bounded interactive checks, then stop the loopback server.

This does not shut down the Molab sandbox: use Shutdown notebook in Molab.
No tunnel, external endpoint, or keepalive is started.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=pathlib.Path, default=pathlib.Path('davidau-gguf-lab'))
    parser.add_argument('--skip-prepare', action='store_true')
    args = parser.parse_args()
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    code = pathlib.Path(__file__).resolve().parent
    report = {'started_unix': time.time(), 'steps': [], 'passed': False}

    def run(name, command, timeout):
        start = time.monotonic()
        with (base / ('session-' + name + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code_value = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        report['steps'].append({'name': name, 'exit': code_value,
                                'seconds': time.monotonic() - start})
        if code_value:
            raise RuntimeError(name + ' failed; see session-' + name + '.log')

    spec = importlib.util.spec_from_file_location('local_service', code / 'davidau_mtp_service.py')
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    try:
        run('gpu', ['nvidia-smi'], 15)
        if not args.skip_prepare:
            run('prepare', ['uv', 'run', str(code / 'davidau_mtp_prepare.py'), '--base', str(base)], 3600)
        evidence = base / 'mtp-evidence'
        status = json.loads((evidence / 'status.json').read_text())
        if not status.get('finished') or status.get('errors') or (evidence / 'fatal.json').exists():
            raise RuntimeError('MTP validation contains errors')
        run('api-start', [sys.executable, str(code / 'davidau_mtp_service.py'), 'start', '--base', str(base)], 210)
        run('api-test', [sys.executable, str(code / 'davidau_mtp_service.py'), 'test', '--base', str(base)], 240)
        run('opencode-install', ['npm', 'install', '--prefix', str(base / 'opencode-runtime'), 'opencode-ai@1.19.5'], 300)
        run('opencode-test', [sys.executable, str(code / 'davidau_opencode_smoke.py'), '--base', str(base)], 210)
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
    finally:
        try:
            report['server_stop'] = service.stop(base)
        except Exception as error:
            report['stop_error'] = str(error)
            report['passed'] = False
        report['finished_unix'] = time.time()
        report['sandbox_shutdown_required'] = True
        (base / 'session-result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
