"""Render the shared llama.cpp kernel from a validated profile.

The file that Kaggle runs is self-contained. Profile JSON is embedded in the
launcher, after validation. The cloudflared and llama.cpp pins stay in the
engine source so a profile cannot point them somewhere else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from models.profile import ProfileError, assert_servable, clamp_launch_context

ENGINE = Path(__file__).resolve().parent / "engines" / "llama_cpp_gpu.py"
CODE_FILE = "llama_cpp_gpu.py"


def render_gpu_kernel(profile, overrides):
    assert_servable(profile)
    if profile["backend"] != "llama.cpp" or profile["accelerator"] != "gpu":
        raise ProfileError(f"{profile['id']}: este render solo admite llama.cpp en gpu")
    src = ENGINE.read_text()
    payload = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    src, n = re.subn(
        r"^PROFILE = None  # __PROFILE_JSON__.*$",
        "PROFILE = json.loads(" + repr(payload) + ")",
        src,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise ProfileError("llama_cpp_gpu.py no tiene la línea __PROFILE_JSON__")
    src, n = re.subn(
        r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
        "CFG = " + repr(dict(overrides)),
        src,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise ProfileError("llama_cpp_gpu.py no tiene la línea __LAUNCHER_CONFIG__")
    return src


def gpu_overrides(profile, *, api_key, ntfy_topic, keepalive_min, max_model_len):
    return {
        "api_key": api_key,
        "ntfy_topic": ntfy_topic,
        "keepalive_min": int(keepalive_min),
        "ctx_size": clamp_launch_context(profile, max_model_len),
    }


def gpu_kernel_metadata(user, slug):
    return {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
