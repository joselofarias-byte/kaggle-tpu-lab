import ast
import re
import unittest
from pathlib import Path

class TunnelUrlTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "qwen38-27b/kernel/serve_qwen38.py").read_text())
        pattern = next(ast.literal_eval(n.value.args[0]) for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Attribute)
                       and n.value.func.attr == "compile" and n.value.args
                       and isinstance(n.value.args[0], ast.Constant)
                       and "trycloudflare" in str(n.value.args[0].value))
        self.pattern = re.compile(pattern)

    def test_extract_real_quick_tunnel_url_from_log(self):
        url = "https://quiet-field-123.trycloudflare.com"
        match = self.pattern.search("INF | " + url + " |")
        self.assertIsNotNone(match, "real tunnel URL must be detected")
        self.assertEqual(match.group(0), url)

    def test_dots_are_literal(self):
        self.assertIsNone(self.pattern.search("https://quiet-fieldXtrycloudflareYcom"))
