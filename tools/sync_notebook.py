#!/usr/bin/env python3
"""Refresh generated notebook cells from the kernel sources.

    python tools/sync_notebook.py qwen38-27b
    python tools/sync_notebook.py glm53-flash
    python tools/sync_notebook.py --check

The ``serve_*.py`` file under ``<recipe>/kernel/`` is the source of truth.
Notebooks keep their hand-written markdown. This tool only replaces:

- the ``%%writefile <kernel-name>`` cell, found by its first line
- the GLM engine cell, found by ``os.makedirs("<pkg>"`` and regenerated with
  ``pack_notebook.engine_cell``

``--check`` exits 1 when a managed cell is stale. Exit 2 is a usage error.
Adapted from k0valik ``tools/sync_notebook.py`` (Stage E) to this repo's
``tools/pack_notebook.py``.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
from pack_notebook import engine_cell  # noqa: E402

RECIPES = ("qwen38-27b", "glm53-flash")
WRITEFILE_RE = re.compile(r"^%%writefile\s+(\S+)\s*$")


def first_line(cell):
    src = "".join(cell.get("source", []))
    lines = src.splitlines()
    return lines[0] if lines else ""


def to_lines(text):
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def find_writefile_cell(cells, filename):
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        match = WRITEFILE_RE.match(first_line(cell))
        if match and match.group(1) == filename:
            return i
    return None


def find_engine_cell(cells, pkg):
    marker = f'os.makedirs("{pkg}", exist_ok=True)'
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if marker in "".join(cell.get("source", [])):
            return i
    return None


def sync_recipe(recipe, check=False):
    folder = ROOT / recipe
    notebooks = sorted((folder / "notebook").glob("*.ipynb"))
    kernels = sorted((folder / "kernel").glob("serve_*.py"))
    if len(notebooks) != 1 or len(kernels) != 1:
        return [f"error: {recipe} needs exactly one notebook and one kernel"], 2
    nb_path, kernel = notebooks[0], kernels[0]
    nb = json.loads(nb_path.read_text())
    cells = nb.get("cells", [])
    stale = []

    idx = find_writefile_cell(cells, kernel.name)
    if idx is None:
        return [f"error: {nb_path.name} has no %%writefile {kernel.name} cell"], 2
    wanted = f"%%writefile {kernel.name}\n" + kernel.read_text()
    if not wanted.endswith("\n"):
        wanted += "\n"
    current = "".join(cells[idx].get("source", []))
    if current != wanted:
        stale.append(f"{recipe}: {kernel.name} cell")
        if not check:
            cells[idx]["source"] = to_lines(wanted)
            cells[idx]["execution_count"] = None
            cells[idx]["outputs"] = []

    engine_inits = sorted((folder / "engine").glob("*/__init__.py")) if (folder / "engine").is_dir() else []
    for init in engine_inits:
        pkg = init.parent.name
        eidx = find_engine_cell(cells, pkg)
        if eidx is None:
            return [f"error: {nb_path.name} has no engine cell for {pkg}"], 2
        body = engine_cell(init.parent, pkg)
        if "".join(cells[eidx].get("source", [])) != body:
            stale.append(f"{recipe}: engine {pkg}")
            if not check:
                cells[eidx]["source"] = to_lines(body)
                cells[eidx]["execution_count"] = None
                cells[eidx]["outputs"] = []

    if check:
        return (stale or [f"{recipe}: in sync"]), (1 if stale else 0)
    if not stale:
        return [f"{recipe}: already in sync"], 0
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    nb_path.write_text(text)
    return [f"{recipe}: updated {', '.join(stale)}"], 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", nargs="*", choices=RECIPES)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    recipes = args.recipe or list(RECIPES)
    worst = 0
    for recipe in recipes:
        lines, code = sync_recipe(recipe, check=args.check)
        for line in lines:
            print(line)
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
