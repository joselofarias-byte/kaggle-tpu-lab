import ast
import asyncio
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "qwen38-27b/kernel/serve_qwen38.py"

class IdleShutdownTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {"KTL_IDLE_STATE": str(Path(self.temp.name) / "state.json"),
                                      "KTL_IDLE_API_KEY": "test-token"})
        env.start()
        self.addCleanup(env.stop)
        tree = ast.parse(KERNEL.read_text())
        source = next(ast.literal_eval(n.value) for n in tree.body
                      if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "IDLE_ACTIVITY_SOURCE" for t in n.targets))
        self.mod = types.ModuleType("idle_test_module")
        exec(source, self.mod.__dict__)
        self.now = 100.0
        self.mod.time = types.SimpleNamespace(monotonic=lambda: self.now)
        self.mod.reset()

    def scope(self, path="/v1/chat/completions", method="POST", key="test-token"):
        return {"type": "http", "method": method, "path": path,
                "headers": [(b"authorization", ("Bearer " + key).encode())]}

    async def call(self, scope, app=None):
        sent = []
        async def default_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})
        async def receive():
            return {"type": "http.request", "body": b""}
        async def send(message):
            sent.append(message)
        await self.mod.IdleActivityMiddleware(app or default_app)(scope, receive, send)
        return sent

    async def test_not_armed_during_startup(self):
        self.now += 10000
        self.assertFalse(self.mod.claim_idle_shutdown(1800))

    async def test_shutdown_at_thirty_minutes_without_work(self):
        self.mod.arm()
        self.now += 1799
        self.assertFalse(self.mod.claim_idle_shutdown(1800))
        self.now += 1
        self.assertTrue(self.mod.claim_idle_shutdown(1800))

    async def test_polls_and_bad_credentials_do_not_reset_timer(self):
        self.mod.arm()
        self.now += 1790
        await self.call(self.scope("/v1/models", "GET"))
        await self.call(self.scope("/health", "GET"))
        await self.call(self.scope(key="wrong-token"))
        self.now += 10
        self.assertTrue(self.mod.claim_idle_shutdown(1800))

    async def test_generation_resets_timer_after_completion(self):
        self.mod.arm()
        self.now += 1700
        sent = await self.call(self.scope())
        self.assertEqual(sent[-1]["body"], b"ok")
        self.now += 1799
        self.assertFalse(self.mod.claim_idle_shutdown(1800))
        self.now += 1
        self.assertTrue(self.mod.claim_idle_shutdown(1800))

    async def test_active_stream_is_not_interrupted(self):
        self.mod.arm()
        started, finish = asyncio.Event(), asyncio.Event()
        async def stream(scope, receive, send):
            started.set()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"chunk", "more_body": True})
            await finish.wait()
            await send({"type": "http.response.body", "body": b"end", "more_body": False})
        task = asyncio.create_task(self.call(self.scope(), stream))
        await started.wait()
        self.now += 4000
        self.assertFalse(self.mod.claim_idle_shutdown(1800))
        finish.set()
        sent = await task
        self.assertEqual(sent[-1]["body"], b"end")
        self.now += 1799
        self.assertFalse(self.mod.claim_idle_shutdown(1800))
        self.now += 1
        self.assertTrue(self.mod.claim_idle_shutdown(1800))

    async def test_application_error_does_not_leak_active_count(self):
        self.mod.arm()
        async def failing(scope, receive, send):
            raise RuntimeError("upstream failed")
        with self.assertRaises(RuntimeError):
            await self.call(self.scope(), failing)
        self.now += 1800
        self.assertTrue(self.mod.claim_idle_shutdown(1800))

    async def test_request_after_shutdown_claim_is_rejected(self):
        self.mod.arm()
        self.now += 1800
        self.assertTrue(self.mod.claim_idle_shutdown(1800))
        async def forbidden(scope, receive, send):
            self.fail("request must not start after shutdown is claimed")
        sent = await self.call(self.scope(), forbidden)
        self.assertEqual(sent[0]["status"], 503)

class WatchdogCleanupTests(unittest.TestCase):
    def run_watchdog(self, cleanup_failure=False):
        import threading
        tree = ast.parse(KERNEL.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "idle_watchdog")
        events = []
        server = types.SimpleNamespace(poll=lambda: None)
        tunnel = types.SimpleNamespace(
            poll=lambda: None, terminate=lambda: events.append("tunnel-terminate"),
            wait=lambda **kw: events.append("tunnel-wait"), kill=lambda: events.append("tunnel-kill"))
        def stop_server(process):
            events.append("server-stop")
            if cleanup_failure:
                raise RuntimeError("cleanup failed")
        def exit_kernel(code):
            events.append(("exit", code))
            raise SystemExit(code)
        namespace = {
            "CFG": {"idle_timeout_min": 30}, "server": server, "tunnel": tunnel,
            "time": types.SimpleNamespace(sleep=lambda seconds: None),
            "idle_activity": types.SimpleNamespace(claim_idle_shutdown=lambda seconds: True),
            "idle_shutdown_requested": threading.Event(), "log": lambda *args: None,
            "stop_server": stop_server, "publish": lambda phase, **kw: events.append((phase, kw["reason"])),
            "_raw": types.SimpleNamespace(flush=lambda: events.append("flush")),
            "os": types.SimpleNamespace(_exit=exit_kernel),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(KERNEL), "exec"), namespace)
        with self.assertRaises(SystemExit) as exc:
            namespace["idle_watchdog"]()
        self.assertEqual(exc.exception.code, 0)
        self.assertTrue(namespace["idle_shutdown_requested"].is_set())
        self.assertIn(("auto-shutdown", "idle-timeout"), events)
        self.assertEqual(events[-1], ("exit", 0))
        return events

    def test_watchdog_stops_server_and_tunnel_before_kernel_exit(self):
        events = self.run_watchdog()
        self.assertLess(events.index("server-stop"), events.index("tunnel-terminate"))
        self.assertLess(events.index("tunnel-wait"), events.index(("exit", 0)))

    def test_cleanup_error_still_exits_kernel(self):
        self.run_watchdog(cleanup_failure=True)
