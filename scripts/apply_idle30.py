#!/usr/bin/env python3
"""Instala idle30 + recuperación de túnel desde un commit fijo; no lanza instancias."""
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "joselofarias-byte/kaggle-tpu-lab"
REF = "98d613a01d8fac26b2a7df3da5c1556b05e2189b"
FILES = ("launch.py", "qwen38-27b/kernel/serve_qwen38.py",
         "scripts/diagnose_session.py", "scripts/restart_preserving_model.py")

def main():
    root = Path.home() / "kaggle-tpu-lab"
    if not (root / "launch.py").is_file():
        print("No encontre ~/kaggle-tpu-lab/launch.py. No se modifico nada.")
        return 1
    if not shutil.which("gh"):
        print("Falta gh (GitHub CLI). No se modifico nada.")
        return 1
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = Path.home() / ".local/state/kaggle-tpu-lab/backups" / stamp
    with tempfile.TemporaryDirectory(prefix="ktl-update-", dir=root) as td:
        stage = Path(td)
        for name in FILES:
            result = subprocess.run(
                ["gh", "api", "-H", "Accept: application/vnd.github.raw+json",
                 "repos/" + REPOSITORY + "/contents/" + name + "?ref=" + REF],
                capture_output=True, text=True, timeout=90)
            if result.returncode:
                print("No se pudo descargar", name, "desde GitHub. No se modifico nada.")
                return 1
            compile(result.stdout, name, "exec")
            candidate = stage / name
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(result.stdout)
        backup.mkdir(parents=True)
        existed = {}
        for name in FILES:
            target = root / name
            existed[name] = target.exists()
            if target.exists():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
        installed = []
        try:
            for name in FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, target)
                installed.append(name)
        except Exception:
            for name in reversed(installed):
                target = root / name
                if existed[name]:
                    shutil.copy2(backup / name, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    (backup / "manifest.json").write_text(json.dumps({"source": REPOSITORY, "ref": REF,
                                                     "files": existed}, indent=2))
    print("Parche instalado: idle configurable + recuperacion automatica del tunel publico.")
    print("Respaldo:", backup)
    print("El kernel ya enviado a Kaggle sigue usando su codigo anterior.")
    print("No se lanzo ni detuvo ninguna instancia. Consultando la sesion existente...")
    return subprocess.run([sys.executable, str(root / "scripts/diagnose_session.py")],
                          cwd=root).returncode

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("Actualizacion incompleta:", type(exc).__name__, "(sin exponer claves).")
        sys.exit(1)
