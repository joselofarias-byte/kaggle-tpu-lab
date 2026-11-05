import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[1] / 'probe_accelerators.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_default_does_not_initialize_tpu(self):
        with patch.object(probe, 'command') as run:
            self.assertEqual(probe.probe_tpu()['status'], 'not_probed')
        run.assert_not_called()

    def test_no_nvidia_smi(self):
        with patch.object(probe.shutil, 'which', return_value=None):
            self.assertEqual(probe.probe_gpu(), {'status': 'unavailable', 'devices': []})

    def test_real_device_fields(self):
        with patch.object(probe.shutil, 'which', return_value='/usr/bin/nvidia-smi'), patch.object(
            probe, 'command', side_effect=['0, Example GPU, 24576, 20000, 570.1, 8.6\n', 'CUDA Version: 12.8']
        ):
            data = probe.probe_gpu()
        self.assertEqual(data['devices'][0]['memory_total_bytes'], 24576 * 1024**2)
        self.assertEqual(data['cuda_driver_supported_version'], '12.8')

    def test_gpu_failure_and_malformed(self):
        for output, expected in [(None, 'query_failed'), ('bad,csv', 'invalid_response'),
                                 ('0,GPU,nan,20,570,8.6', 'invalid_response'), ('', 'no_devices')]:
            with self.subTest(output=output), patch.object(probe.shutil, 'which', return_value='nvidia-smi'), patch.object(probe, 'command', return_value=output):
                self.assertEqual(probe.probe_gpu()['status'], expected)

    def test_tpu_is_opt_in_and_bounded(self):
        response = {'status': 'ok', 'device_count': 8, 'devices': [{'coords': [0, 0, 0]}]}
        with patch.object(probe, 'command', return_value=json.dumps(response)) as run:
            self.assertEqual(probe.probe_tpu(True), response)
        self.assertEqual(run.call_args.kwargs['timeout'], 60)
        self.assertIn("jax.devices('tpu')", run.call_args.args[0][-1])

    def test_tpu_failure_no_stderr(self):
        for response in (None, 'not json', '[]', '{}'):
            with self.subTest(response=response), patch.object(probe, 'command', return_value=response):
                self.assertNotEqual(probe.probe_tpu(True)['status'], 'ok')

    def test_command_failure_timeout(self):
        with patch.object(probe.subprocess, 'run', side_effect=subprocess.TimeoutExpired('test', 10)):
            self.assertIsNone(probe.command(['test']))
        with patch.object(probe.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'secret')):
            self.assertIsNone(probe.command(['test']))

    def test_memory(self):
        with patch.object(probe.Path, 'read_text', return_value='MemTotal: 1000 kB\nMemAvailable: 500 kB\n'):
            self.assertEqual(probe.host_memory()['available_bytes'], 512000)
        with patch.object(probe.Path, 'read_text', side_effect=OSError):
            self.assertIsNone(probe.host_memory()['total_bytes'])

    def test_package_metadata_only(self):
        with patch.object(probe.importlib.metadata, 'version', side_effect=probe.importlib.metadata.PackageNotFoundError):
            self.assertTrue(all(v is None for v in probe.package_versions().values()))

    def test_cpu_cli_json(self):
        with patch.object(probe.shutil, 'which', return_value=None), patch.object(probe, 'package_versions', return_value={}), contextlib.redirect_stdout(io.StringIO()) as out:
            probe.main([])
        data = json.loads(out.getvalue())
        self.assertEqual(data['tpu']['status'], 'not_probed')
        self.assertEqual(data['gpu']['devices'], [])
        self.assertNotIn('environment', data)


if __name__ == '__main__':
    unittest.main()
