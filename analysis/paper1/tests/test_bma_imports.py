# SPDX-License-Identifier: BSD-3-Clause
"""Pin how the evidence runner imports its sibling modules.

``xlike_configs`` (like ``_bma_keys``) uses package-relative imports, so it only
imports as a submodule of its package. The runner is launched as
``python -m analysis.paper1.bma_evidence`` with
``PYTHONPATH=<wt>/src:<wt>/analysis:<wt>``; a flat ``import xlike_configs`` off a
patched ``sys.path`` raises "attempted relative import with no known parent
package" and every X-like evidence cell fails. Pinned here:

- the builders load in a fresh interpreter with exactly the runner's PYTHONPATH,
  through both ``paper1`` and ``analysis.paper1``;
- no sibling module flat-imports a module that itself uses relative imports.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
ROOT = ANALYSIS.parent

XLIKE_KEYS = ["bagpipes_like", "beagle_like", "cigale_like", "dense_basis_like", "prospector_like"]


def _run_import(package_import: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ANALYSIS), str(ROOT)])
    env.setdefault("JAX_PLATFORMS", "cpu")
    code = (
        f"{package_import}; b = bma_evidence._load_xlike_builders(); "
        f"assert sorted(b) == {XLIKE_KEYS!r}, sorted(b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "package_import",
    ["from paper1 import bma_evidence", "from analysis.paper1 import bma_evidence"],
)
def test_runner_pythonpath_loads_xlike_builders(package_import):
    """The X-like builders load under the runner's PYTHONPATH, either package spelling."""
    result = _run_import(package_import)
    assert result.returncode == 0, result.stderr[-2000:]


def _uses_relative_imports(path: Path) -> bool:
    """True if the module has a relative import not guarded by an ImportError fallback."""
    tree = ast.parse(path.read_text())
    guarded = {
        id(n)
        for t in ast.walk(tree)
        if isinstance(t, ast.Try)
        and any(
            isinstance(h.type, ast.Name) and h.type.id in {"ImportError", "ModuleNotFoundError"}
            for h in t.handlers
        )
        for b in t.body
        for n in ast.walk(b)
    }
    return any(
        isinstance(n, ast.ImportFrom) and n.level > 0 and id(n) not in guarded
        for n in ast.walk(tree)
    )


def _flat_imported_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_flat_import_of_a_relative_import_module():
    """No paper1 module flat-imports a sibling that needs its package to import."""
    relative = {p.stem for p in PAPER1.glob("*.py") if _uses_relative_imports(p)}
    offenders = [
        f"{p.name} -> {name}"
        for p in PAPER1.glob("*.py")
        for name in sorted(_flat_imported_names(p) & relative)
        if name != p.stem
    ]
    assert not offenders, offenders
