"""Wave 2 profile catalog. No Kaggle session, no weight download."""

import ast
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))

import launch  # noqa: E402
from models.engines.llama_cpp_gpu import (  # noqa: E402
    CLOUDFLARED_SHA256,
    LLAMA_SHA256,
    gpu_requirement_met,
    public_event_fields,
    publish,
    redact,
    safe_extract,
    serve_config,
    server_command,
)
from models.gpu_launch import gpu_overrides, render_gpu_kernel  # noqa: E402
from models.profile import (  # noqa: E402
    ProfileError,
    assert_servable,
    check_gpu_inventory,
    clamp_launch_context,
    format_model_list,
    load_catalog,
    resolve_profile,
    validate_profile,
)

GPU_SCRIPT = REPO / "qwen38-27b" / "gpu" / "serve_qwen38_gpu.py"
FORBIDDEN = ("0.29.0", "enable-prefix-caching", "mtp-rollback-v0290")


def _base_profile(**overrides):
    data = {
        "schema_version": 1,
        "id": "example-gpu",
        "display_name": "Example",
        "backend": "llama.cpp",
        "accelerator": "gpu",
        "source": {"repository": "org/example", "revision": "abc123"},
        "model_file": "example-Q4_K_M.gguf",
        "sha256": "a" * 64,
        "model_size_bytes": 1000,
        "architecture": "qwen3",
        "quantization": "Q4_K_M",
        "context_size": 8192,
        "architecture_context": 32768,
        "resources": {
            "min_vram_gb": 8,
            "recommended_vram_gb": 16,
            "min_ram_gb": 8,
            "recommended_ram_gb": 16,
        },
        "gpu": {"count": 2, "type": "Tesla T4", "tensor_parallel": None},
        "tpu_compatible": False,
        "launch_args": {"host": "127.0.0.1", "parallel": 2, "mtp_tokens": 0},
        "served_model_name": "example-q4",
        "chat_template": None,
        "tokenizer_source": None,
        "health_check": {"path": "/v1/models"},
        "notes": [],
        "warnings": [],
        "experimental": False,
        "launchable": True,
        "trust_remote_code": False,
        "tags": [],
    }
    data.update(overrides)
    return data


def _script_defaults():
    tree = ast.parse(GPU_SCRIPT.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("DEFAULTS missing")


class SchemaTest(unittest.TestCase):
    def test_catalog_loads(self):
        catalog = load_catalog()
        self.assertIn("qwen38-27b-gpu", catalog)
        self.assertIn("qwen38-27b-tpu", catalog)
        self.assertIn("glm53-flash-tpu", catalog)
        self.assertGreaterEqual(len(catalog), 7)

    def test_missing_fields(self):
        with self.assertRaises(ProfileError) as caught:
            validate_profile({"id": "x"})
        self.assertIn("faltan campos", str(caught.exception))

    def test_invalid_sha256(self):
        with self.assertRaises(ProfileError):
            validate_profile(_base_profile(sha256="ABC"))
        with self.assertRaises(ProfileError):
            validate_profile(_base_profile(sha256="g" * 64))
        self.assertEqual(validate_profile(_base_profile())["sha256"], "a" * 64)

    def test_sha256_optional_until_launchable(self):
        data = _base_profile(sha256=None, model_size_bytes=None, launchable=False, experimental=True)
        self.assertIsNone(validate_profile(data)["sha256"])
        with self.assertRaises(ProfileError):
            validate_profile(_base_profile(sha256=None))

    def test_unsupported_pairs(self):
        with self.assertRaises(ProfileError) as caught:
            validate_profile(_base_profile(backend="llama.cpp", accelerator="tpu", gpu=None,
                                           tpu={"chips": 8}, launchable=False, experimental=True,
                                           sha256=None, model_size_bytes=None))
        self.assertIn("combinación no admitida", str(caught.exception))
        with self.assertRaises(ProfileError):
            validate_profile(_base_profile(backend="vllm-tpu", accelerator="gpu",
                                           vllm_tpu_version="0.28.0"))
        with self.assertRaises(ProfileError):
            validate_profile(_base_profile(backend="vllm-gpu"))

    def test_context_cannot_exceed_architecture(self):
        with self.assertRaises(ProfileError) as caught:
            validate_profile(_base_profile(context_size=65536, architecture_context=32768))
        self.assertIn("architecture_context", str(caught.exception))

    def test_context_clamp_matches_wave1_gpu_default(self):
        profile = load_catalog()["qwen38-27b-gpu"]
        self.assertEqual(profile["context_size"], 32768)
        self.assertEqual(clamp_launch_context(profile, 262144), 32768)
        self.assertEqual(clamp_launch_context(profile, 4096), 4096)
        with self.assertRaises(ProfileError):
            clamp_launch_context(profile, 0)

    def test_gpu_count(self):
        profile = _base_profile()
        rows = [{"name": "Tesla T4"}, {"name": "Tesla T4"}]
        self.assertTrue(check_gpu_inventory(profile, rows))
        self.assertFalse(check_gpu_inventory(profile, rows[:1]))
        self.assertFalse(check_gpu_inventory(profile, [{"name": "Tesla P100"}, {"name": "Tesla P100"}]))
        with self.assertRaises(ProfileError) as caught:
            validate_profile(_base_profile(gpu={"count": 4, "type": "A100", "tensor_parallel": 4}))
        self.assertIn("dos Tesla T4", str(caught.exception))
        candidate = _base_profile(launchable=False, experimental=True, gpu={"count": 1, "type": "Tesla T4"})
        self.assertEqual(validate_profile(candidate)["gpu"]["count"], 1)
        self.assertFalse(check_gpu_inventory(candidate, rows))

    def test_trust_remote_code_must_surface_risk(self):
        with self.assertRaises(ProfileError) as caught:
            validate_profile(_base_profile(trust_remote_code=True))
        self.assertIn("trust_remote_code_risk", str(caught.exception))
        ok = validate_profile(_base_profile(
            trust_remote_code=True, trust_remote_code_risk="ejecuta código del repositorio"))
        self.assertTrue(ok["trust_remote_code"])

    def test_vllm_029_rejected(self):
        with self.assertRaises(ProfileError):
            validate_profile(_base_profile(
                backend="vllm-tpu", accelerator="tpu", gpu=None,
                tpu={"chips": 8, "tensor_parallel": 8},
                vllm_tpu_version="0.29.0", legacy_recipe="qwen38-27b",
                model_file=None, sha256=None, model_size_bytes=None,
            ))


class SelectionTest(unittest.TestCase):
    def test_qwen_aliases_and_exact_ids(self):
        gpu = resolve_profile("qwen38-27b", "gpu")
        tpu = resolve_profile("qwen38-27b", "tpu")
        self.assertEqual(gpu["id"], "qwen38-27b-gpu")
        self.assertEqual(tpu["id"], "qwen38-27b-tpu")
        self.assertEqual(tpu["vllm_tpu_version"], "0.28.0")
        self.assertEqual(tpu["legacy_recipe"], "qwen38-27b")
        self.assertEqual(resolve_profile("glm53-flash", "tpu")["legacy_recipe"], "glm53-flash")
        with self.assertRaises(ProfileError):
            resolve_profile("glm53-flash", "gpu")
        with self.assertRaises(ProfileError):
            resolve_profile("qwen38-27b-gpu", "tpu")

    def test_candidates_are_listed_and_not_servable(self):
        catalog = load_catalog()
        candidates = [p for p in catalog.values() if not p["launchable"]]
        self.assertGreaterEqual(len(candidates), 4)
        for profile in candidates:
            self.assertTrue(profile["experimental"])
            self.assertIn(profile["tags"][0], ("abliterated", "uncensored", "unfiltered"))
            with self.assertRaises(ProfileError):
                assert_servable(profile)
        listed = format_model_list(catalog)
        self.assertIn("qwen38-27b-gpu", listed)
        self.assertIn("candidato", listed)

    def test_tags_do_not_change_the_server_command(self):
        left = serve_config(_base_profile(tags=[]))
        right = serve_config(_base_profile(tags=["uncensored", "abliterated", "unfiltered"]))
        self.assertEqual(server_command("/bin/llama-server", left, "/m.gguf"),
                         server_command("/bin/llama-server", right, "/m.gguf"))

    def test_cli_list_and_info(self):
        buf = io.StringIO()
        with mock.patch("sys.argv", ["launch.py", "models"]), mock.patch("sys.stdout", buf):
            launch.main()
        self.assertIn("qwen38-27b-gpu", buf.getvalue())
        buf = io.StringIO()
        with mock.patch("sys.argv", ["launch.py", "model-info", "qwen38-27b-gpu"]), mock.patch("sys.stdout", buf):
            launch.main()
        text = buf.getvalue()
        self.assertIn("unsloth/Qwen3.8-27B-GGUF", text)
        self.assertIn("322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482", text)
        with self.assertRaises(SystemExit):
            with mock.patch("sys.argv", ["launch.py", "model-info", "no-such-model"]):
                launch.main()

    def test_prepare_serve_keeps_tpu_recipe_and_blocks_candidates(self):
        args = mock.Mock(model="qwen38-27b", accelerator="tpu")
        self.assertIsNone(launch.prepare_serve_selection(args))
        self.assertEqual(args.model, "qwen38-27b")
        args = mock.Mock(model="qwen38-27b-tpu", accelerator="tpu")
        launch.prepare_serve_selection(args)
        self.assertEqual(args.model, "qwen38-27b")
        args = mock.Mock(model="glm53-flash", accelerator="tpu")
        launch.prepare_serve_selection(args)
        self.assertEqual(args.model, "glm53-flash")
        args = mock.Mock(model="blackfrost-qwen38-27b-abliterated-gpu", accelerator="gpu")
        with self.assertRaises(SystemExit):
            launch.prepare_serve_selection(args)


class QwenGpuCompatibilityTest(unittest.TestCase):
    def test_profile_matches_wave1_script_pins(self):
        defaults = _script_defaults()
        profile = load_catalog()["qwen38-27b-gpu"]
        self.assertEqual(profile["source"]["repository"], defaults["model_repo"])
        self.assertEqual(profile["source"]["revision"], defaults["model_revision"])
        self.assertEqual(profile["model_file"], defaults["model_file"])
        self.assertEqual(profile["sha256"], defaults["model_sha256"])
        self.assertEqual(profile["model_size_bytes"], defaults["model_size"])
        self.assertEqual(profile["served_model_name"], defaults["served_model_name"])
        self.assertEqual(profile["context_size"], defaults["ctx_size"])
        self.assertEqual(LLAMA_SHA256, defaults["llama_sha256"])
        self.assertEqual(CLOUDFLARED_SHA256, defaults["cloudflared_sha256"])

    def test_rendered_kernel_matches_qwen_command(self):
        profile = load_catalog()["qwen38-27b-gpu"]
        overrides = gpu_overrides(profile, api_key="sk-abc", ntfy_topic="ktl-test",
                                  keepalive_min=480, max_model_len=262144)
        self.assertEqual(overrides["ctx_size"], 32768)
        cfg = serve_config(profile, overrides)
        command = server_command("/opt/llama-server", cfg, "/tmp/Qwen3.8-27B-UD-Q4_K_M.gguf")
        self.assertIn("--host", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("0.0.0.0", command)
        self.assertEqual(command[command.index("--ctx-size") + 1], "32768")
        self.assertEqual(command[command.index("--alias") + 1], "qwen3.8-27b-q4")
        self.assertNotIn("--spec-type", command)
        rendered = render_gpu_kernel(profile, overrides)
        self.assertIn(profile["sha256"], rendered)
        self.assertIn(CLOUDFLARED_SHA256, rendered)
        self.assertIn('"--host", "127.0.0.1"', rendered)
        self.assertNotIn("0.0.0.0", rendered)
        self.assertNotIn("releases/latest", rendered)
        meta = launch.gpu_kernel_metadata("someone", profile["kernel_slug"])
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(meta["enable_tpu"], "false")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(profile["kernel_slug"], "qwen38-t4x2-serve")

    def test_gpu_requirement_helper(self):
        cfg = {"gpu_count": 2, "gpu_type": "Tesla T4"}
        self.assertTrue(gpu_requirement_met([{"name": "Tesla T4"}, {"name": "Tesla T4"}], cfg))
        self.assertFalse(gpu_requirement_met([{"name": "Tesla T4"}], cfg))


class EventAndRedactionTest(unittest.TestCase):
    def test_ready_keeps_key_and_identity_other_phases_drop_key(self):
        ready = public_event_fields("ready", {"api_key": "sk-x", "endpoint": "https://e"})
        self.assertEqual(ready["api_key"], "sk-x")
        beat = public_event_fields("heartbeat", {"api_key": "sk-x", "up_min": 3})
        self.assertNotIn("api_key", beat)
        cfg = serve_config(load_catalog()["qwen38-27b-gpu"], {"api_key": "sk-" + "b" * 20, "ctx_size": 32768})
        extra = public_event_fields("heartbeat", {"api_key": cfg["api_key"], "up_min": 1})
        self.assertNotIn("api_key", extra)
        from models.engines.llama_cpp_gpu import identity_fields
        ident = identity_fields(cfg)
        self.assertEqual(ident["model_id"], "qwen38-27b-gpu")
        self.assertEqual(ident["backend"], "llama.cpp")
        self.assertEqual(ident["accelerator"], "gpu")
        self.assertIn("display_name", ident)
        logged = []
        with mock.patch("models.engines.llama_cpp_gpu.log", side_effect=lambda *a: logged.append(" ".join(map(str, a)))):
            publish("ready", cfg, api_key=cfg["api_key"], model=cfg["served_model_name"], endpoint="https://e")
            publish("download", cfg, api_key=cfg["api_key"], file="weights")
        ready_line = logged[0]
        self.assertIn("event_version", ready_line)
        self.assertIn("message_es", ready_line)
        self.assertIn("qwen38-27b-gpu", ready_line)
        self.assertNotIn(cfg["api_key"], ready_line)
        self.assertIn("[REDACTED]", ready_line)
        self.assertNotIn(cfg["api_key"], logged[1])

    def test_redact_strips_bearer(self):
        key = "sk-" + "c" * 20
        red = redact(f"Authorization: Bearer {key} token=sekretvalue", key)
        self.assertNotIn(key, red)
        self.assertNotIn("sekretvalue", red)
        self.assertIn("[REDACTED]", red)

    def test_safe_extract_rejects_links_and_slip(self):
        def archive(member, payload=None):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.addfile(member, io.BytesIO(payload) if payload is not None else None)
            return buf.getvalue()

        slip = tarfile.TarInfo(name="../escape.txt")
        slip.size = 1
        link = tarfile.TarInfo(name="link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/outside"
        hard = tarfile.TarInfo(name="hard")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "other"
        ok = tarfile.TarInfo(name="llama-server")
        ok.size = 1
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "slip.tar.gz"
            path.write_bytes(archive(slip, b"x"))
            with self.assertRaises(RuntimeError):
                safe_extract(path, root / "out")
            for info in (link, hard):
                path = root / (info.name + ".tar.gz")
                path.write_bytes(archive(info))
                with self.assertRaises(RuntimeError):
                    safe_extract(path, root / "out")
            good = root / "ok.tar.gz"
            good.write_bytes(archive(ok, b"x"))
            dest = root / "ok"
            safe_extract(good, dest)
            self.assertEqual((dest / "llama-server").read_bytes(), b"x")


class NoExperimentalLeakTest(unittest.TestCase):
    def test_wave2_tree_does_not_carry_vllm_029(self):
        # profile.py may name 0.29.0 only to reject it. Profiles, the GPU
        # engine, and the launcher must not select that runtime.
        roots = [
            REPO / "models" / "profiles",
            REPO / "models" / "engines",
            REPO / "models" / "gpu_launch.py",
            REPO / "launch.py",
        ]
        blobs = []
        for root in roots:
            if root.is_file():
                blobs.append(root.read_text())
            else:
                for path in root.rglob("*"):
                    if path.suffix in {".py", ".json"}:
                        blobs.append(path.read_text())
        blob = "\n".join(blobs)
        for token in FORBIDDEN:
            self.assertNotIn(token, blob, token)
        self.assertIn('"vllm_tpu_version": "0.28.0"', (REPO / "models" / "profiles" / "qwen38-27b-tpu.json").read_text())
        qwen_kernel = (REPO / "qwen38-27b" / "kernel" / "serve_qwen38.py").read_text()
        self.assertIn('"vllm_tpu_version": "0.28.0"', qwen_kernel)
        self.assertNotIn("mtp-rollback-v0290", qwen_kernel)


if __name__ == "__main__":
    unittest.main()
