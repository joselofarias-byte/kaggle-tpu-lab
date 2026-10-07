import ast
import re
import unittest
from pathlib import Path

class TunnelUrlTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "qwen38-27b/kernel/serve_qwen38.py").read_text())
        patterns = [ast.literal_eval(n.value.args[0]) for n in ast.walk(tree)
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == "compile" and n.value.args
                    and isinstance(n.value.args[0], ast.Constant)
                    and "trycloudflare" in str(n.value.args[0].value)]
        self.patterns = [re.compile(pattern) for pattern in patterns]
        self.pattern = self.patterns[0]

    def test_extract_real_quick_tunnel_url_from_log(self):
        url = "https://quiet-field-123.trycloudflare.com"
        match = self.pattern.search("INF | " + url + " |")
        self.assertIsNotNone(match, "real tunnel URL must be detected")
        self.assertEqual(match.group(0), url)

    def test_dots_are_literal(self):
        self.assertIsNone(self.pattern.search("https://quiet-fieldXtrycloudflareYcom"))

    def test_every_quick_tunnel_extractor_accepts_real_url(self):
        url = "https://quiet-field-123.trycloudflare.com"
        self.assertGreaterEqual(len(self.patterns), 2)
        for pattern in self.patterns:
            with self.subTest(pattern=pattern.pattern):
                match = pattern.search("INF | " + url + " |")
                self.assertIsNotNone(match)
                self.assertEqual(match.group(0), url)

class TunnelRecoveryRegressionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.kernel = (root / "qwen38-27b/kernel/serve_qwen38.py").read_text()
        self.launcher = (root / "launch.py").read_text()

    def test_retry_happens_after_server_health_and_before_ready(self):
        serving = self.kernel.index('publish("serving", startup_secs=startup)')
        retry = self.kernel.index('publish("tunnel-retry"', serving)
        ready = self.kernel.index('publish("ready"', retry)
        idle_arm = self.kernel.index("idle_activity.arm()", ready)
        self.assertLess(serving, retry)
        self.assertLess(retry, ready)
        self.assertLess(ready, idle_arm)

    def test_failed_recovery_stops_inaccessible_tpu(self):
        self.assertIn('"tunnel_recovery_min": 5', self.kernel)
        self.assertIn('reason="tunnel-unavailable"', self.kernel)
        self.assertIn("stop_server(server)", self.kernel)

    def test_launcher_propagates_idle_and_tunnel_settings(self):
        self.assertIn('"idle_timeout_min": args.idle_timeout_min', self.launcher)
        self.assertIn('"tunnel_recovery_min": args.tunnel_recovery_min', self.launcher)
        self.assertIn('s.add_argument("--tunnel-recovery-min"', self.launcher)

