"""Offline CPU tests; never download a model or require an accelerator."""
import argparse
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location("molab_lab", Path(__file__).parents[1] / "lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def sibling(name, size):
    return SimpleNamespace(rfilename=name, size=size)


RUNTIME_HELP = "--api-key-file --sleep-idle-seconds --batch-size --ubatch-size --flash-attn"


def detected(free=24 * lab.GIB, ram=64 * lab.GIB):
    return {"gpus": [{"index": 0, "free_bytes": free}], "ram_available_bytes": ram}


class ModelTests(unittest.TestCase):
    def test_split_requires_all_shards_and_sorts_first(self):
        groups = lab.gguf_groups([
            sibling("Q4-00002-of-00002.gguf", 20),
            sibling("Q4-00001-of-00002.gguf", 10),
            sibling("Q8-00001-of-00002.gguf", 60),
            sibling("README.md", 5), sibling("mmproj-F16.gguf", 2),
            sibling("vision-MMProj.gguf", 3), sibling("unknown.gguf", None),
            sibling("../bad.gguf", 10), sibling("/bad.gguf", 10),
            sibling("x-00000-of-00002.gguf", 10),
            sibling("x-00001-of-00002.gguf", 10)])
        self.assertEqual(groups, [{"files": [("Q4-00001-of-00002.gguf", 10),
                                               ("Q4-00002-of-00002.gguf", 20)], "bytes": 30}])

    def test_select_largest_fitting_quant_without_model_substitution(self):
        groups = lab.gguf_groups([sibling("Q4.gguf", 10 * lab.GIB),
                                  sibling("Q8.gguf", 22 * lab.GIB)])
        chosen, _ = lab.choose_model(groups, detected())
        self.assertEqual(chosen["files"][0][0], "Q4.gguf")
        with self.assertRaisesRegex(RuntimeError, "no model substitution"):
            lab.choose_model(groups, detected(), "Q8.gguf")
        with self.assertRaises(RuntimeError):
            lab.choose_model(groups, {"gpus": [], "ram_available_bytes": 10**12})
        with self.assertRaises(RuntimeError):
            lab.choose_model(groups, detected(ram=3 * lab.GIB))
        with self.assertRaises(RuntimeError):
            lab.choose_model(groups, detected(), "nonexistent.gguf")

    def test_hf_download_pins_resolved_sha_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "Q4-00001-of-00002.gguf", root / "Q4-00002-of-00002.gguf"
            first.write_bytes(b"first")
            second.write_bytes(b"last")
            hf = MagicMock()
            hf.HfApi.return_value.model_info.return_value = SimpleNamespace(
                sha="a" * 40, siblings=[sibling(second.name, 4), sibling(first.name, 5)])
            hf.hf_hub_download.side_effect = lambda repo, filename, **kw: str(root / filename)
            args = SimpleNamespace(revision="main", filename=None, cache_dir=root / "cache")
            with patch.dict(sys.modules, {"huggingface_hub": hf}):
                path, metadata = lab.download_model(args, detected())
                self.assertEqual(path, first)
                self.assertEqual(metadata["revision"], "a" * 40)
                self.assertEqual(metadata["repository"], lab.REPO)
                self.assertEqual(hf.hf_hub_download.call_count, 2)
                for call in hf.hf_hub_download.call_args_list:
                    self.assertEqual(call.kwargs, {"revision": "a" * 40, "cache_dir": args.cache_dir})
                second.write_bytes(b"truncated")
                with self.assertRaisesRegex(RuntimeError, "size"):
                    lab.download_model(args, detected())

    def test_probe_reports_absence_not_assumed_gpu(self):
        with patch.object(lab.subprocess, "check_output", side_effect=FileNotFoundError):
            info = lab.hardware()
        self.assertEqual(info["gpus"], [])
        self.assertIsNone(info["cuda_toolkit"])

    def test_probe_parses_real_gpu_and_toolkit_output(self):
        with patch.object(lab.subprocess, "check_output", side_effect=[
                '0, GPU-test, NVIDIA Test GPU, 24576, 18000, 600.1\n', 'Cuda compilation tools, release 13.4, V13.4']):
            info = lab.hardware()
        self.assertEqual(info["gpus"][0]["free_bytes"], 18000 * 2**20)
        self.assertEqual(info["cuda_toolkit"], "13.4")


class SecurityTests(unittest.TestCase):
    def test_unauthenticated_models_must_be_rejected(self):
        response = MagicMock()
        with patch.object(lab, "api_request", return_value=response):
            with self.assertRaisesRegex(lab.LabError, "Unauthenticated"):
                lab.verify_auth({"port": 8080})
        error = lab.urllib.error.HTTPError("http://127.0.0.1", 401, "Unauthorized", {}, None)
        with patch.object(lab, "api_request", side_effect=error) as request:
            lab.verify_auth({"port": 8080})
            self.assertIsNone(request.call_args.args[1])

    def test_idle_runtime_options_require_support(self):
        args = SimpleNamespace(idle_seconds=600, batch_size=512, ubatch_size=128, flash_attn="off")
        with self.assertRaisesRegex(lab.LabError, "sleep-idle"):
            lab.runtime_options(args, "--api-key-file")
        options = lab.runtime_options(args, RUNTIME_HELP)
        self.assertIn("--sleep-idle-seconds", options)
        args.ubatch_size = 1024
        with self.assertRaisesRegex(lab.LabError, "Microbatch"):
            lab.runtime_options(args, RUNTIME_HELP)

    def test_private_dir_and_key_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = lab.private_dir(Path(tmp) / "state")
            key_path = base / "api.key"
            lab.write_private(key_path, "A" * 43)
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(lab.read_key(base), "A" * 43)
            key_path.chmod(0o644)
            with self.assertRaises(RuntimeError):
                lab.read_key(base)
            base.chmod(0o755)
            with self.assertRaises(RuntimeError):
                lab.private_dir(base)

    def test_private_write_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.write_text("keep")
            (root / "link").symlink_to(target)
            with self.assertRaises(OSError):
                lab.write_private(root / "link", "secret")
            self.assertEqual(target.read_text(), "keep")

    def test_env_removes_auth_and_llama_overrides(self):
        with patch.dict(os.environ, {"LLAMA_ARG_API_KEY": "secret", "LLAMA_ARG_HOST": "0.0.0.0",
                                     "OPENAI_API_KEY": "secret", "HF_TOKEN": "secret"}):
            env = lab.server_env(2)
        self.assertNotIn("LLAMA_ARG_API_KEY", env)
        self.assertNotIn("LLAMA_ARG_HOST", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")

    def test_api_auth_uses_header_loopback_and_disables_proxies(self):
        with patch.object(lab.urllib.request, "build_opener") as opener:
            lab.api_request({"port": 8080}, "secret", "/v1/models")
        handler = opener.call_args.args[0]
        self.assertEqual(handler.proxies, {})
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertNotIn("secret", request.full_url)
        self.assertIsInstance(opener.call_args.args[1], lab.NoRedirect)
        self.assertIsNone(opener.call_args.args[1].redirect_request(None, None, 302, "", {}, "https://external.test"))

    def test_failures_do_not_echo_external_exception_data(self):
        with patch.object(lab, "private_dir", side_effect=RuntimeError("secret-token private-url")), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(lab.main(["status"]), 1)
            self.assertNotIn("secret-token", stderr.getvalue())
            self.assertNotIn("private-url", stderr.getvalue())


class LifecycleTests(unittest.TestCase):
    def test_pid_reuse_prevents_signalling(self):
        state = {"pid": 4321, "identity": {"start_ticks": "old", "boot_id": "boot"}}
        with patch.object(lab, "process_identity", return_value={"start_ticks": "new", "boot_id": "boot"}), \
                patch.object(lab.os, "pidfd_open") as pidfd:
            self.assertFalse(lab.stop_process(state))
            pidfd.assert_not_called()
        with patch.object(lab, "process_identity", return_value=None):
            self.assertFalse(lab.alive({"pid": 4321, "identity": None}))

    def test_pid_reuse_between_open_and_signal_is_safe(self):
        state = {"pid": 4321, "identity": {"start_ticks": "old", "boot_id": "boot"}}
        with patch.object(lab, "alive", side_effect=[True, False]), \
                patch.object(lab.os, "pidfd_open", return_value=123), \
                patch.object(lab.os, "close") as close, \
                patch.object(lab.signal, "pidfd_send_signal") as send:
            self.assertFalse(lab.stop_process(state))
            send.assert_not_called()
            close.assert_called_once_with(123)

    def test_stop_escalates_only_via_identity_checked_pidfd(self):
        with patch.object(lab, "alive", return_value=True), \
                patch.object(lab.os, "pidfd_open", return_value=123), \
                patch.object(lab.os, "close"), \
                patch.object(lab.select, "select", side_effect=[([], [], []), ([123], [], [])]), \
                patch.object(lab.signal, "pidfd_send_signal") as send:
            self.assertTrue(lab.stop_process({"pid": 4321}))
            self.assertEqual(send.call_args_list[0].args, (123, signal.SIGTERM))
            self.assertEqual(send.call_args_list[1].args, (123, signal.SIGKILL))

    def test_actual_owned_cpu_process_can_be_stopped(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            state = {"pid": process.pid, "identity": lab.process_identity(process.pid)}
            self.assertTrue(lab.alive(state))
            self.assertTrue(lab.stop_process(state, timeout=2))
            process.wait(timeout=3)
            self.assertFalse(lab.alive(state))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_start_refuses_runtime_without_key_file_support(self):
        args = SimpleNamespace(server="llama-server")
        with patch.object(lab, "read_state", return_value=None), \
                patch.object(lab.shutil, "which", return_value="/bin/llama-server"), \
                patch.object(lab.subprocess, "check_output", return_value="--api-key KEY"), \
                patch.object(lab, "download_model") as download:
            with self.assertRaisesRegex(RuntimeError, "api-key-file"):
                lab.start(args, Path("/unused"))
            download.assert_not_called()

    def test_start_rejects_cpu_build_before_download(self):
        args = SimpleNamespace(server="llama-server", port=8080, idle_seconds=600,
                                   batch_size=512, ubatch_size=128, flash_attn="auto")
        with patch.object(lab, "read_state", return_value=None), \
                patch.object(lab.shutil, "which", return_value="/bin/llama-server"), \
                patch.object(lab.subprocess, "check_output", side_effect=[RUNTIME_HELP, "CPU only"]), \
                patch.object(lab.socket, "socket"), \
                patch.object(lab, "hardware", return_value=detected()), \
                patch.object(lab, "download_model") as download:
            with self.assertRaisesRegex(RuntimeError, "CUDA device"):
                lab.start(args, Path("/unused"))
            download.assert_not_called()

    def test_start_auth_loopback_and_no_log_or_key_argv(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            base = lab.private_dir(Path(tmp) / "state")
            args = SimpleNamespace(server="llama-server", port=8080, context=4096, startup_timeout=2,
                                   idle_seconds=600, batch_size=512, ubatch_size=128, flash_attn="auto")
            metadata = {"weight_bytes": 10, "gpu_index": 0, "repository": lab.REPO,
                        "revision": lab.REVISION, "files": ["model.gguf"]}
            stack.enter_context(patch.object(lab.shutil, "which", return_value="/bin/llama-server"))
            stack.enter_context(patch.object(lab.subprocess, "check_output", side_effect=[RUNTIME_HELP, "CUDA0: Test GPU", "CUDA0: Test GPU"]))
            stack.enter_context(patch.object(lab.socket, "socket"))
            stack.enter_context(patch.object(lab, "hardware", return_value=detected()))
            stack.enter_context(patch.object(lab, "download_model", return_value=(Path("/cache/model.gguf"), metadata)))
            stack.enter_context(patch.object(lab, "process_identity", return_value={"start_ticks": "1", "boot_id": "boot"}))
            popen = stack.enter_context(patch.object(lab.subprocess, "Popen"))
            popen.return_value.pid = 4321
            popen.return_value.poll.return_value = None
            response = MagicMock()
            response.__enter__.return_value = io.StringIO(json.dumps({"data": [{"id": lab.ALIAS}]}))
            stack.enter_context(patch.object(lab, "api_request", side_effect=[response,
                lab.urllib.error.HTTPError("http://127.0.0.1", 401, "Unauthorized", {}, None)]))
            self.assertTrue(lab.start(args, base)["running"])
            key = lab.read_key(base)
            command = popen.call_args.args[0]
            self.assertNotIn(key, " ".join(command))
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertIn("--api-key-file", command)
            self.assertEqual(command[command.index("--sleep-idle-seconds") + 1], "600")
            self.assertEqual(command[command.index("--flash-attn") + 1], "auto")
            self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.DEVNULL)
            self.assertNotIn(key, (base / "state.json").read_text())


class BenchmarkTests(unittest.TestCase):
    def run_benchmark(self, root, events, content_type="text/event-stream"):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Type": content_type}
        response.__iter__.return_value = iter(events)
        args = SimpleNamespace(prompt="private prompt secret", tokens=32, timeout=10)
        state = {"model": {"repository": lab.REPO, "revision": lab.REVISION},
                 "context": 4096, "startup_seconds": 2.5}
        with patch.object(lab, "read_state", return_value=state), \
                patch.object(lab, "alive", return_value=True), \
                patch.object(lab, "read_key", return_value="secret-key"), \
                patch.object(lab, "api_request", return_value=response):
            return lab.benchmark(args, root)

    def test_sse_measures_only_observed_usage_and_sanitizes_reports(self):
        events = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                  b'data: {"choices":[{"delta":{"content":"private answer secret-key"}}]}\n',
                  b'data: {"usage":{"completion_tokens":7,"prompt_tokens":13,"total_tokens":20,"secret":"key"}}\n',
                  b'data: [DONE]\n']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_benchmark(root, events)
            self.assertEqual(result["usage"]["completion_tokens"], 7)
            self.assertGreater(result["ttft_seconds"], 0)
            self.assertGreater(result["completion_tokens_per_wall_second"], 0)
            for filename in ("benchmark.json", "benchmark.md"):
                text = (root / filename).read_text()
                self.assertNotIn("secret", text)
                self.assertNotIn("private", text)
                self.assertEqual((root / filename).stat().st_mode & 0o777, 0o600)

    def test_missing_usage_is_null_not_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_benchmark(Path(tmp), [b'data: [DONE]\n'])
            self.assertIsNone(report["completion_tokens_per_wall_second"])
            self.assertIsNone(report["ttft_seconds"])
            self.assertEqual(report["usage"], {})

    def test_truncated_stream_or_http_json_does_not_record_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Truncated"):
                self.run_benchmark(root, [b'data: {"choices":[]}\n'])
            self.assertFalse((root / "benchmark.json").exists())
            with self.assertRaisesRegex(RuntimeError, "SSE"):
                self.run_benchmark(root, [], "application/json")

    def test_error_event_is_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "SSE error") as error:
                self.run_benchmark(root, [b'data: {"error":"secret key here"}\n'])
            self.assertNotIn("secret", str(error.exception))
            self.assertFalse((root / "benchmark.json").exists())


if __name__ == "__main__":
    unittest.main()
