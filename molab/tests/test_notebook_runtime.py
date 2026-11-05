"""Exercise notebook load admission without importing PyTorch or marimo."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class NotebookRuntimeTests(unittest.TestCase):
    def runtime(self, bf16_supported=False):
        source = ast.parse((Path(__file__).parents[1] / 'jolufasoft_llm_lab.py').read_text())
        node = next(node for node in ast.walk(source) if isinstance(node, ast.ClassDef) and node.name == 'LLMRuntime')
        module = ast.Module(body=[node], type_ignores=[])
        scope = {'torch': SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(is_bf16_supported=lambda: bf16_supported))}
        exec(compile(module, '<notebook-runtime>', 'exec'), scope)
        return scope['LLMRuntime']()

    def test_gguf_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'llama.cpp'):
            self.runtime().load('DavidAU/model-GGUF', 'bf16')

    def test_unsupported_bf16_rejected_before_loading(self):
        with self.assertRaisesRegex(RuntimeError, 'FP16'):
            self.runtime().load('Qwen/test', 'bf16')

    def test_exact_loaded_model_reused(self):
        runtime = self.runtime(True)
        runtime.model = object()
        runtime.key = ('Qwen/test', 'bf16', False)
        runtime.load('Qwen/test', 'bf16')
        self.assertIsNotNone(runtime.model)
