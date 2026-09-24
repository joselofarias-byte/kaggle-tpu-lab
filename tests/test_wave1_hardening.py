"""Wave 1 checks that do not need Kaggle, a TPU, or the network.

Kernel helpers are loaded from the shipped sources with ast so the tests
follow the real functions rather than a copy.
"""
import ast
import collections
import hashlib
import hmac
import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import launch  # noqa: E402


def load_funcs(path, *names, **extra):
    src = path.read_text()
    tree = ast.parse(src)
    wanted = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = set(names) - set(wanted)
    if missing:
        raise AssertionError(f"missing {missing} in {path}")
    mod = ast.Module(body=list(wanted.values()), type_ignores=[])
    ns = {
        "os": os, "re": re, "sys": sys, "json": json, "hashlib": hashlib, "hmac": hmac,
        "Path": Path, "collections": collections, "CFG": {}, "log": lambda *a: None,
        "publish": lambda *a, **k: None,
    }
    ns.update(extra)
    exec(compile(mod, str(path), "exec"), ns)  # noqa: S102 — test sandbox
    return ns


QWEN = REPO / "qwen38-27b" / "kernel" / "serve_qwen38.py"
GLM = REPO / "glm53-flash" / "kernel" / "serve_glm53.py"
GPU = REPO / "qwen38-27b" / "gpu" / "serve_qwen38_gpu.py"


class DigestAndRedactTest(unittest.TestCase):
    def test_sha256_accepts_and_rejects(self):
        ns = load_funcs(QWEN, "verify_sha256")
        data = b"cloudflared-stand-in" * 100
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(data)
            path = handle.name
        try:
            good = hashlib.sha256(data).hexdigest()
            self.assertTrue(ns["verify_sha256"](path, good))
            with open(path, "r+b") as handle:
                handle.seek(3)
                handle.write(b"Z")
            self.assertFalse(ns["verify_sha256"](path, good))
        finally:
            os.unlink(path)

    def test_redact_strips_key_and_bearer(self):
        ns = load_funcs(QWEN, "redact")
        key = "sk-" + "a" * 20
        ns["CFG"] = {"api_key": key}
        red = ns["redact"](f"Authorization: Bearer {key} token=sekretvalue")
        self.assertNotIn(key, red)
        self.assertNotIn("sekretvalue", red)
        self.assertIn("[REDACTED]", red)

    def test_ready_keeps_api_key_other_phases_drop_it(self):
        ns = load_funcs(QWEN, "public_event_fields")
        self.assertEqual(ns["public_event_fields"]("ready", {"api_key": "sk-x", "endpoint": "https://e"})["api_key"], "sk-x")
        self.assertNotIn("api_key", ns["public_event_fields"]("heartbeat", {"api_key": "sk-x", "up_min": 3}))

    def test_glm_has_the_same_key_rule(self):
        ns = load_funcs(GLM, "public_event_fields")
        self.assertNotIn("api_key", ns["public_event_fields"]("heartbeat", {"api_key": "glm-abc"}))


class StateAndPinsTest(unittest.TestCase):
    def test_state_file_is_atomic_0600(self):
        old = os.umask(0)
        try:
            with tempfile.TemporaryDirectory() as td:
                target = Path(td) / "state.json"
                with mock.patch.object(launch, "STATE_FILE", target):
                    launch.write_state({"api_key": "sk-secret", "topic": "t"})
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
                self.assertEqual(json.loads(target.read_text())["api_key"], "sk-secret")
                self.assertEqual([p.name for p in Path(td).iterdir()], ["state.json"])
        finally:
            os.umask(old)

    def test_cloudflared_pin_is_present_and_not_latest(self):
        for path in (QWEN, GLM, GPU):
            src = path.read_text()
            self.assertNotIn("releases/latest", src, path.name)
            self.assertIn("03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc", src)
        self.assertIn('"vllm_tpu_version": "0.29.0"', QWEN.read_text())
        self.assertIn("--enable-prefix-caching", QWEN.read_text())
        self.assertIn("mtp-rollback-v0290.diff", QWEN.read_text())
        self.assertIn('"--host", "127.0.0.1"', QWEN.read_text())
        self.assertIn('"--host", "127.0.0.1"', GPU.read_text())
        self.assertNotIn('"--host", "0.0.0.0"', GPU.read_text())

    def test_models_url_shapes(self):
        self.assertEqual(launch.models_url("https://q.example/v1"), "https://q.example/v1/models")
        self.assertEqual(launch.models_url("https://g.example"), "https://g.example/v1/models")
        self.assertEqual(launch.models_url("https://g.example/"), "https://g.example/v1/models")


class FailFastTest(unittest.TestCase):
    def test_sanitize_drops_poisoned_env(self):
        logs = []
        ns = load_funcs(QWEN, "sanitize_tpu_env", log=lambda *a: logs.append(" ".join(map(str, a))))
        os.environ["TPU_WORKER_HOSTNAMES"] = "WARNING: could not determine"
        try:
            ns["sanitize_tpu_env"]()
            self.assertNotIn("TPU_WORKER_HOSTNAMES", os.environ)
            self.assertTrue(any("TPU_WORKER_HOSTNAMES" in line for line in logs))
        finally:
            os.environ.pop("TPU_WORKER_HOSTNAMES", None)
            os.environ.pop("TPU_WORKER_ADDRS", None)

    def test_death_report_ignores_bytes_before_offset(self):
        old = b"[vllm] Traceback (most recent call last)\n[vllm] RuntimeError: old boom\n"
        new = b"[vllm] (EngineCore) INFO startup ok\n"
        path = Path(tempfile.mkdtemp()) / "vllm.log"
        path.write_bytes(old + new)
        ns = load_funcs(QWEN, "server_death_report", RAW_LOG=path, _LOG_START=len(old))
        cause, block, _hint = ns["server_death_report"]()
        self.assertNotIn("old boom", cause + block)

    def test_server_died_clean_exit_names_external_stop(self):
        path = Path(tempfile.mkdtemp()) / "vllm.log"
        path.write_bytes(b"")
        logs, events = [], []

        def publish(phase, **kw):
            events.append((phase, kw))

        ns = load_funcs(
            QWEN, "server_died", "server_death_report", "redact",
            RAW_LOG=path, _LOG_START=0,
            log=lambda *a: logs.append(" ".join(map(str, a))),
            publish=publish,
        )
        ns["server_death_report"] = lambda *a, **k: ("", "", "")
        server = type("S", (), {"returncode": 0, "tail": collections.deque(["ok"])})()
        with self.assertRaises(SystemExit):
            ns["server_died"](server, "stopped")
        self.assertTrue(any("Save & Run All" in line for line in logs))
        self.assertEqual(events[0][0], "stopped")
        self.assertNotIn("api_key", events[0][1])

    def test_tar_slip_rejected(self):
        ns = load_funcs(QWEN, "safe_tar_members")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name="../escape.txt")
            payload = b"nope"
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        raw = buf.getvalue()
        path = Path(tempfile.mkdtemp()) / "cache.tar"
        path.write_bytes(raw)
        with self.assertRaises(ValueError):
            ns["safe_tar_members"](str(path))

    def test_mtp_head_false_without_index(self):
        ns = load_funcs(QWEN, "has_mtp_head")
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(ns["has_mtp_head"](td))
            Path(td, "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.layers.0.mlp.down_proj.weight": "model.safetensors"}
            }))
            self.assertFalse(ns["has_mtp_head"](td))
            Path(td, "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"mtp.layers.0.weight": "model.safetensors"}
            }))
            self.assertTrue(ns["has_mtp_head"](td))


class GpuAndContractTest(unittest.TestCase):
    def test_gpu_metadata_shape_helper_and_dual_t4(self):
        ns = load_funcs(GPU, "is_dual_t4")
        self.assertTrue(ns["is_dual_t4"]([
            {"name": "Tesla T4"}, {"name": "Tesla T4"},
        ]))
        self.assertFalse(ns["is_dual_t4"]([{"name": "Tesla T4"}]))
        src = GPU.read_text()
        self.assertIn("CFG = None  # __LAUNCHER_CONFIG__", src)
        self.assertIn("event_version", src)

    def test_event_contract_markers_remain(self):
        for path in (QWEN, GLM):
            src = path.read_text()
            self.assertIn("event_version", src)
            self.assertIn("message_es", src)
            self.assertIn("CFG = None  # __LAUNCHER_CONFIG__", src)
        self.assertIn('ENGINE_B64 = ""  # __ENGINE__', GLM.read_text())

    def test_spanish_messages_cover_new_failures(self):
        ns = load_funcs(QWEN, "_event_message_es")
        self.assertIn("Internet", ns["_event_message_es"]("failed", {"step": "no-internet"}))
        self.assertIn("cloudflared", ns["_event_message_es"]("failed", {"step": "tunnel-binary"}))
        self.assertIn("MTP", ns["_event_message_es"]("mtp-disabled", {}))


if __name__ == "__main__":
    unittest.main()
