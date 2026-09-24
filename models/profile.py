"""Load, validate, list, and select model profiles.

Profiles are JSON documents under ``models/profiles/``. Labels such as
``uncensored`` or ``abliterated`` are metadata. They do not select a
different execution path and they are not a promise of zero refusals.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES_DIR = HERE / "profiles"

SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")

BACKENDS = ("llama.cpp", "vllm-gpu", "vllm-tpu", "jax")
ACCELERATORS = ("gpu", "tpu")
# Pairs this repo knows how to describe. Serving is a separate, narrower check.
ALLOWED_PAIRS = {
    ("llama.cpp", "gpu"),
    ("vllm-gpu", "gpu"),
    ("vllm-tpu", "tpu"),
    ("jax", "tpu"),
}
# Production TPU recipe. A profile must not name any other vllm-tpu release.
VLLM_TPU_VERSION = "0.28.0"
IDENTITY_FIELDS = ("model_id", "display_name", "backend", "accelerator")

REQUIRED = (
    "schema_version",
    "id",
    "display_name",
    "backend",
    "accelerator",
    "source",
    "model_file",
    "sha256",
    "architecture",
    "quantization",
    "context_size",
    "architecture_context",
    "resources",
    "tpu_compatible",
    "launch_args",
    "served_model_name",
    "chat_template",
    "tokenizer_source",
    "health_check",
    "notes",
    "warnings",
    "experimental",
    "launchable",
    "trust_remote_code",
)


class ProfileError(ValueError):
    """A profile is missing data, inconsistent, or not servable."""


def _fail(profile_id, message):
    label = profile_id or "<perfil>"
    raise ProfileError(f"{label}: {message}")


def _require_bool(data, key, profile_id):
    if not isinstance(data.get(key), bool):
        _fail(profile_id, f"'{key}' debe ser true o false")
    return data[key]


def _require_int(value, label, profile_id, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(profile_id, f"{label} debe ser un entero >= {minimum}")
    return value


def normalize_sha256(value, profile_id, *, required):
    """Accept a lowercase SHA-256 hex digest. Never invent one."""
    if value is None:
        if required:
            _fail(profile_id, "falta sha256")
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(profile_id, "sha256 debe ser 64 hexadecimales en minúsculas, o null")
    return value


def validate_profile(data):
    """Return a validated profile dict, or raise ProfileError."""
    if not isinstance(data, dict):
        _fail(None, "el perfil no es un objeto JSON")
    profile_id = data.get("id") if isinstance(data.get("id"), str) else None
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        _fail(profile_id, "faltan campos: " + ", ".join(missing))
    if data["schema_version"] != SCHEMA_VERSION:
        _fail(profile_id, f"schema_version debe ser {SCHEMA_VERSION}")
    if not isinstance(data["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,80}", data["id"]):
        _fail(profile_id, "id inválido")
    profile_id = data["id"]
    if not isinstance(data["display_name"], str) or not data["display_name"].strip():
        _fail(profile_id, "display_name vacío")
    backend = data["backend"]
    accelerator = data["accelerator"]
    if backend not in BACKENDS or accelerator not in ACCELERATORS:
        _fail(profile_id, f"backend/acelerador desconocido: {backend}/{accelerator}")
    if (backend, accelerator) not in ALLOWED_PAIRS:
        _fail(profile_id, f"combinación no admitida: {backend} + {accelerator}")

    source = data["source"]
    if not isinstance(source, dict) or not isinstance(source.get("repository"), str) or not source["repository"].strip():
        _fail(profile_id, "source.repository es obligatorio")
    revision = source.get("revision", None)
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        _fail(profile_id, "source.revision debe ser un commit o null")

    for key in ("architecture", "quantization", "served_model_name"):
        if not isinstance(data[key], str) or not data[key].strip():
            _fail(profile_id, f"'{key}' vacío")
    if data["model_file"] is not None and (not isinstance(data["model_file"], str) or not data["model_file"].strip()):
        _fail(profile_id, "model_file debe ser un nombre de archivo o null")

    ctx = _require_int(data["context_size"], "context_size", profile_id)
    arch = _require_int(data["architecture_context"], "architecture_context", profile_id)
    if ctx > arch:
        _fail(profile_id, f"context_size {ctx} supera architecture_context {arch}")

    resources = data["resources"]
    if not isinstance(resources, dict):
        _fail(profile_id, "resources debe ser un objeto")
    for key in ("min_vram_gb", "recommended_vram_gb", "min_ram_gb", "recommended_ram_gb"):
        value = resources.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            _fail(profile_id, f"resources.{key} inválido")
    if resources["min_vram_gb"] > resources["recommended_vram_gb"]:
        _fail(profile_id, "min_vram_gb supera recommended_vram_gb")
    if resources["min_ram_gb"] > resources["recommended_ram_gb"]:
        _fail(profile_id, "min_ram_gb supera recommended_ram_gb")

    _validate_device(data, profile_id)
    _require_bool(data, "tpu_compatible", profile_id)
    if not isinstance(data["launch_args"], dict):
        _fail(profile_id, "launch_args debe ser un objeto")
    if accelerator == "gpu" and backend == "llama.cpp":
        host = data["launch_args"].get("host", "127.0.0.1")
        if host != "127.0.0.1":
            _fail(profile_id, "llama-server solo puede escuchar en 127.0.0.1")
    for key in ("chat_template", "tokenizer_source"):
        if data[key] is not None and not isinstance(data[key], str):
            _fail(profile_id, f"{key} debe ser texto o null")
    health = data["health_check"]
    if not isinstance(health, dict) or not isinstance(health.get("path"), str) or not health["path"].startswith("/"):
        _fail(profile_id, "health_check.path es obligatorio")
    for key in ("notes", "warnings"):
        if not isinstance(data[key], list) or not all(isinstance(item, str) for item in data[key]):
            _fail(profile_id, f"{key} debe ser una lista de textos")
    experimental = _require_bool(data, "experimental", profile_id)
    launchable = _require_bool(data, "launchable", profile_id)
    trust = _require_bool(data, "trust_remote_code", profile_id)
    if trust:
        risk = data.get("trust_remote_code_risk")
        if not isinstance(risk, str) or not risk.strip():
            _fail(profile_id, "trust_remote_code exige trust_remote_code_risk (no se activa en silencio)")
    tags = data.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        _fail(profile_id, "tags debe ser una lista de textos")
    aliases = data.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias for alias in aliases):
        _fail(profile_id, "aliases debe ser una lista de textos")

    version = data.get("vllm_tpu_version")
    if backend == "vllm-tpu":
        if version != VLLM_TPU_VERSION:
            _fail(profile_id, f"vllm-tpu debe ser {VLLM_TPU_VERSION}")
    elif version not in (None,):
        _fail(profile_id, "vllm_tpu_version solo aplica a backend vllm-tpu")
    if version == "0.29.0":
        _fail(profile_id, "vllm-tpu 0.29.0 no entra en esta rama")

    sha_required = bool(launchable and backend == "llama.cpp")
    normalize_sha256(data["sha256"], profile_id, required=sha_required)
    size = data.get("model_size_bytes")
    if size is not None:
        _require_int(size, "model_size_bytes", profile_id, minimum=1)
    if sha_required and size is None:
        _fail(profile_id, "un GGUF servible necesita model_size_bytes")
    if launchable and backend == "llama.cpp":
        if not data["model_file"] or not revision:
            _fail(profile_id, "un GGUF servible necesita model_file y source.revision")
        if health.get("path") != "/v1/models":
            _fail(profile_id, "health_check.path debe ser /v1/models")
        _assert_dual_t4_shape(data)
    if launchable and backend == "vllm-gpu":
        _fail(profile_id, "vllm-gpu todavía no tiene motor en esta rama")
    if launchable and backend in ("vllm-tpu", "jax"):
        recipe = data.get("legacy_recipe")
        if not isinstance(recipe, str) or not recipe.strip():
            _fail(profile_id, "el perfil TPU servible necesita legacy_recipe")
    if experimental and launchable and backend == "llama.cpp":
        _fail(profile_id, "un perfil experimental no puede marcarse servible sin una prueba física")
    return data


def _validate_device(data, profile_id):
    accelerator = data["accelerator"]
    gpu = data.get("gpu", None)
    tpu = data.get("tpu", None)
    if accelerator == "gpu":
        if not isinstance(gpu, dict):
            _fail(profile_id, "accelerator gpu exige el objeto gpu")
        _require_int(gpu.get("count"), "gpu.count", profile_id)
        if not isinstance(gpu.get("type"), str) or not gpu["type"].strip():
            _fail(profile_id, "gpu.type vacío")
        tp = gpu.get("tensor_parallel", None)
        if tp is not None:
            _require_int(tp, "gpu.tensor_parallel", profile_id)
    elif accelerator == "tpu":
        if gpu not in (None,):
            _fail(profile_id, "un perfil TPU no declara gpu")
        if tpu is not None:
            if not isinstance(tpu, dict):
                _fail(profile_id, "tpu debe ser un objeto")
            if "chips" in tpu:
                _require_int(tpu["chips"], "tpu.chips", profile_id)
            if "tensor_parallel" in tpu:
                _require_int(tpu["tensor_parallel"], "tpu.tensor_parallel", profile_id)


def _assert_dual_t4_shape(profile):
    """The only GPU machine this launcher knows how to start on Kaggle."""
    gpu = profile["gpu"]
    if gpu["count"] != 2 or "T4" not in gpu["type"].upper():
        _fail(profile["id"], "el motor llama.cpp de Kaggle solo admite dos Tesla T4")


def check_gpu_inventory(profile, rows):
    """True when the visible GPUs match the profile's count and type substring."""
    gpu = profile.get("gpu") or {}
    need = gpu.get("count")
    kind = str(gpu.get("type") or "")
    if not isinstance(rows, list) or not isinstance(need, int):
        return False
    if len(rows) != need:
        return False
    token = "T4" if "T4" in kind.upper() else kind.upper()
    return all(token in str(row.get("name", "")).upper() for row in rows)


def clamp_launch_context(profile, requested):
    """Wave 1 GPU behavior: never exceed the profile cap.

    ``python launch.py serve --accelerator gpu`` still defaults
    ``--max-model-len`` to 262144. The Qwen GPU profile caps that at 32768.
    """
    cap = profile["context_size"]
    if requested is None:
        return cap
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        _fail(profile["id"], "el contexto pedido debe ser un entero >= 1")
    return min(requested, cap)


def assert_servable(profile):
    """Refuse candidates and unfinished backends. Tags do not change this."""
    if not profile["launchable"]:
        _fail(profile["id"],
              "no es servible: no hay una prueba física en Kaggle y el lanzador no lo va a intentar")
    if profile["experimental"]:
        _fail(profile["id"], "está marcado experimental; esta rama no lo lanza")
    if profile["backend"] == "llama.cpp" and profile["accelerator"] != "gpu":
        _fail(profile["id"], "combinación no admitida: llama.cpp + " + profile["accelerator"])
    if profile["backend"] == "vllm-gpu":
        _fail(profile["id"], "vllm-gpu todavía no tiene motor")
    if profile.get("vllm_tpu_version") not in (None, VLLM_TPU_VERSION):
        _fail(profile["id"], "versión de vllm-tpu no admitida")


def load_catalog(directory=None):
    root = Path(directory) if directory else PROFILES_DIR
    if not root.is_dir():
        raise ProfileError(f"no existe el directorio de perfiles: {root}")
    catalog = {}
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ProfileError(f"{path.name}: JSON inválido ({exc})") from exc
        profile = validate_profile(data)
        if profile["id"] != path.stem:
            _fail(profile["id"], f"el id no coincide con el archivo {path.name}")
        if profile["id"] in catalog:
            _fail(profile["id"], "id duplicado")
        catalog[profile["id"]] = profile
    if not catalog:
        raise ProfileError(f"no hay perfiles en {root}")
    return catalog


def resolve_profile(model, accelerator, catalog=None):
    """Pick one profile for a model id (or legacy alias) and an accelerator."""
    if accelerator not in ACCELERATORS:
        _fail(model, f"acelerador no admitido: {accelerator}")
    catalog = catalog if catalog is not None else load_catalog()
    if model in catalog:
        profile = catalog[model]
        if profile["accelerator"] != accelerator:
            _fail(model, f"el perfil es {profile['accelerator']}, no {accelerator}")
        return profile
    matches = []
    for profile in catalog.values():
        aliases = profile.get("aliases") or []
        if not isinstance(aliases, list):
            _fail(profile["id"], "aliases debe ser una lista")
        if model in aliases and profile["accelerator"] == accelerator:
            matches.append(profile)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(item["id"] for item in matches)
        _fail(model, f"varios perfiles coinciden para {accelerator}: {ids}")
    _fail(model, f"no hay perfil {accelerator} para '{model}'")


def legacy_tpu_recipe(model, catalog=None):
    """Map a TPU profile back to the existing kernel recipe. None if it is not TPU."""
    profile = resolve_profile(model, "tpu", catalog)
    recipe = profile.get("legacy_recipe")
    if not isinstance(recipe, str) or not recipe:
        _fail(profile["id"], "perfil TPU sin legacy_recipe")
    return recipe


def format_model_list(catalog):
    lines = ["Perfiles de modelo (la etiqueta no cambia el motor):", ""]
    for profile in sorted(catalog.values(), key=lambda item: (not item["launchable"], item["accelerator"], item["id"])):
        flag = "servible" if profile["launchable"] else "candidato, sin prueba física"
        tags = ",".join(profile.get("tags") or []) or "-"
        lines.append(
            f"{profile['id']}  [{profile['backend']} / {profile['accelerator']}]  {flag}  tags={tags}"
        )
        lines.append(f"    {profile['display_name']}")
    return "\n".join(lines)


def format_model_info(profile):
    source = profile["source"]
    gpu = profile.get("gpu") or {}
    lines = [
        f"id: {profile['id']}",
        f"nombre: {profile['display_name']}",
        f"backend: {profile['backend']}",
        f"acelerador: {profile['accelerator']}",
        f"repositorio: {source.get('repository')}",
        f"revisión: {source.get('revision')}",
        f"archivo: {profile.get('model_file')}",
        f"sha256: {profile.get('sha256')}",
        f"arquitectura: {profile.get('architecture')}",
        f"cuantización: {profile.get('quantization')}",
        f"contexto de lanzamiento: {profile.get('context_size')}",
        f"contexto de arquitectura: {profile.get('architecture_context')}",
        f"GPU: count={gpu.get('count')} type={gpu.get('type')} tensor_parallel={gpu.get('tensor_parallel')}",
        f"tpu_compatible: {profile.get('tpu_compatible')}",
        f"modelo servido: {profile.get('served_model_name')}",
        f"chat_template: {profile.get('chat_template')}",
        f"experimental: {profile.get('experimental')}",
        f"servible: {profile.get('launchable')}",
        f"trust_remote_code: {profile.get('trust_remote_code')}",
        f"tags: {', '.join(profile.get('tags') or []) or '-'}",
        "notas:",
    ]
    lines.extend(f"- {note}" for note in profile["notes"])
    lines.append("avisos:")
    lines.extend(f"- {note}" for note in profile["warnings"])
    if not profile["launchable"]:
        lines.append("Estado: candidato documentado. No se lanza hasta una prueba física en Kaggle.")
    return "\n".join(lines)
