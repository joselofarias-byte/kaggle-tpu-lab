#!/usr/bin/env python3
"""Read-only hardware inventory; TPU backend initialization is strictly opt-in."""
import argparse
import csv
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("jax", "jaxlib", "libtpu", "torch", "torch-xla", "vllm", "vllm-tpu", "transformers", "llama-cpp-python")
TPU_CODE = """
import json
import jax
items = []
for d in jax.devices('tpu'):
    item = {'id': int(d.id), 'process_index': int(d.process_index),
            'platform': d.platform, 'device_kind': d.device_kind}
    for field in ('coords', 'core_on_chip'):
        value = getattr(d, field, None)
        if value is not None:
            item[field] = list(map(int, value)) if field == 'coords' else int(value)
    items.append(item)
print(json.dumps({'status': 'ok', 'device_count': len(items),
                  'process_count': jax.process_count(), 'devices': items}))
"""


def command(args, timeout=10):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def host_memory():
    try:
        values = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('MemTotal', 'MemAvailable'):
                values[key] = int(value.split()[0]) * 1024
        return {'total_bytes': values.get('MemTotal'), 'available_bytes': values.get('MemAvailable'),
                'scope': 'host /proc/meminfo; not a container allocation guarantee'}
    except (OSError, ValueError, IndexError):
        return {'total_bytes': None, 'available_bytes': None, 'scope': 'unavailable'}


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def probe_gpu():
    executable = shutil.which('nvidia-smi')
    if not executable:
        return {'status': 'unavailable', 'devices': []}
    output = command([executable, '--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap',
                      '--format=csv,noheader,nounits'])
    if output is None:
        return {'status': 'query_failed', 'devices': []}
    devices = []
    try:
        for row in csv.reader(output.splitlines()):
            if not row:
                continue
            index, name, total, free, driver, capability = (part.strip() for part in row)
            total, free = float(total), float(free)
            if not all(math.isfinite(n) and n >= 0 for n in (total, free)):
                raise ValueError('invalid memory')
            devices.append({'index': int(index), 'name': name, 'memory_total_bytes': int(total * 1024**2),
                            'memory_free_bytes': int(free * 1024**2), 'driver_version': driver,
                            'compute_capability': capability})
    except (ValueError, OverflowError):
        return {'status': 'invalid_response', 'devices': []}
    header = command([executable]) or ''
    match = re.search(r'CUDA Version:\s*([\d.]+)', header)
    return {'status': 'ok' if devices else 'no_devices', 'devices': devices,
            'cuda_driver_supported_version': match.group(1) if match else None}


def probe_tpu(enabled=False):
    if not enabled:
        return {'status': 'not_probed', 'reason': 'use --probe-tpu to initialize JAX TPU backend'}
    output = command([sys.executable, '-c', TPU_CODE], timeout=60)
    if output is None:
        return {'status': 'unavailable_or_failed', 'reason': 'TPU subprocess failed or timed out'}
    try:
        data = json.loads(output)
        if data.get('status') != 'ok' or not isinstance(data.get('devices'), list):
            raise ValueError('invalid TPU response')
        return data
    except (ValueError, AttributeError):
        return {'status': 'invalid_response'}


def binary_versions():
    result = {}
    for name, args, pattern in (
        ('nvcc', ['--version'], r'release\s+([\d.]+)'),
        ('llama-server', ['--version'], r'(?:version:|build:)\s*([^\r\n]+)'),
    ):
        executable = shutil.which(name)
        output = command([executable, *args]) if executable else None
        match = re.search(pattern, output or '')
        result[name] = match.group(1) if match else None
    return result


def collect(probe_tpu_enabled=False):
    return {'schema_version': 1, 'python': platform.python_version(), 'machine': platform.machine(),
            'ram': host_memory(), 'packages': package_versions(), 'binaries': binary_versions(),
            'gpu': probe_gpu(), 'tpu': probe_tpu(probe_tpu_enabled)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-tpu', action='store_true', help='Initialize JAX TPU backend in a child process (60s timeout)')
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.probe_tpu), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
