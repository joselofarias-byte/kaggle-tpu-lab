import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('configure', Path(__file__).parents[1] / 'scripts/configure_opencode.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConfigTests(unittest.TestCase):
    def test_urls_and_limits(self):
        for url in ['http://remote.example', 'https://key:secret@example.com', 'https://example.com/?key=secret', 'https://example.com/#secret']:
            with self.assertRaises(ValueError):
                module.provider(url, 'model', 8192, 512)
        for url in ['http://127.0.0.1:18080', 'https://example.com/v1/']:
            entry = module.provider(url, 'model', 8192, 512)
            self.assertTrue(entry['options']['baseURL'].endswith('/v1'))
            self.assertEqual(entry['options']['apiKey'], '{env:MOLAB_API_KEY}')
        with self.assertRaises(ValueError):
            module.provider('https://example.com', 'model', 512, 8192)

    def test_preserve_existing_and_private_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'opencode.json'
            path.write_text('{"theme":"system","provider":{"existing":{}}}')
            entry = module.provider('http://127.0.0.1:18080', 'exact-model', 8192, 1024)
            module.write_config(path, entry)
            data = json.loads(path.read_text())
            self.assertEqual(data['theme'], 'system')
            self.assertIn('existing', data['provider'])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                module.write_config(path, entry)
            self.assertEqual(path.read_bytes(), original)
