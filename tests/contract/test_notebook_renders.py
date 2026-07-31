# SPDX-License-Identifier: BSD-3-Clause
"""The published-render guard, and the hand-rolled parser it depends on (#1506).

``tools/check_notebook_renders.py`` runs in the ``lint`` CI job, which installs
ruff and nothing else, so it parses the jupytext ``py:percent`` format with a
stdlib regex rather than importing jupytext. That parser is the guard's single
point of failure: if it silently disagreed with jupytext, the guard would pass on
a drifted render and the whole check would be decorative. These tests pin it
against jupytext itself, which the test environment does have.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "check_notebook_renders.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_notebook_renders", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_notebook_renders"] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


def _jupytext_code_cells(py_path: Path) -> list[str]:
    jupytext = pytest.importorskip("jupytext")
    nb = jupytext.read(str(py_path))
    return [c.source.strip() for c in nb.cells if c.cell_type == "code" and c.source.strip()]


@pytest.mark.parametrize("slug,py_path,_ipynb", tool.published(), ids=lambda v: str(v)[:40])
def test_stdlib_parser_agrees_with_jupytext(slug, py_path, _ipynb):
    """The regex parser must see exactly the code cells jupytext sees.

    A silent disagreement here is the failure mode that would make the guard
    decorative: it would compare the wrong cells and pass on real drift.
    """
    assert py_path.is_file(), f"{slug}: missing source"
    assert tool.parse_percent_code_cells(py_path.read_text(encoding="utf-8")) == (
        _jupytext_code_cells(py_path)
    ), f"{slug}: stdlib percent parser disagrees with jupytext"


def test_guard_passes_on_the_committed_tree():
    """Every published render currently matches its source and kept its figures."""
    proc = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, f"guard failed on the committed tree:\n{proc.stderr}"


def _run_guard_on(tmp_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tmp_root / "tools" / "check_notebook_renders.py")],
        capture_output=True,
        text=True,
        cwd=tmp_root,
    )


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """A minimal copy of the pieces the guard reads, so tests can corrupt it."""
    for rel in ("tools/check_notebook_renders.py", "scripts/sync_spine_notebooks_for_docs.py"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    shutil.copytree(
        ROOT / "notebooks",
        tmp_path / "notebooks",
        ignore=shutil.ignore_patterns("*.ipynb", "figures", "__pycache__"),
    )
    shutil.copytree(ROOT / "docs" / "spine", tmp_path / "docs" / "spine")
    return tmp_path


def test_guard_catches_a_drifted_render(repo_copy: Path):
    """Editing a render's code without its source is exactly the #1506 failure."""
    target = repo_copy / "docs" / "spine" / "00_quickstart.ipynb"
    nb = json.loads(target.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = ["# drifted\n", *(cell["source"] or [])]
            break
    target.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")

    proc = _run_guard_on(repo_copy)
    assert proc.returncode == 1, "guard did not notice a drifted render"
    assert "drifted from its source" in proc.stderr


def test_guard_catches_a_render_that_lost_its_figures(repo_copy: Path):
    """Executing under MPLBACKEND=Agg strips every image/png; the guard must see it."""
    target = repo_copy / "docs" / "spine" / "00_quickstart.ipynb"
    nb = json.loads(target.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        cell["outputs"] = [
            o for o in (cell.get("outputs") or []) if "image/png" not in (o.get("data") or {})
        ]
    target.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")

    proc = _run_guard_on(repo_copy)
    assert proc.returncode == 1, "guard did not notice a render with no figures"
    assert "no image/png output" in proc.stderr


def test_experimental_notebooks_are_checked():
    """The nested renders must be in scope -- their absence is what caused #1506."""
    slugs = {slug for slug, _py, _ipynb in tool.published()}
    assert {"stochastic_sfh_recovery", "multimodel_bma_candels"} <= slugs
    paths = {ipynb for _slug, _py, ipynb in tool.published()}
    assert any("experimental" in str(p) for p in paths)
