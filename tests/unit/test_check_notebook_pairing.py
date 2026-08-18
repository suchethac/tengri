# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_notebook_pairing.py``.

Both polarities: pairings the guard must accept and pairings it must reject.
A guard tested only on the clean tree passes just as well when it is blind.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_notebook_pairing import (
    _mirror_path,
    _percent_code_cells,
)

pytestmark = pytest.mark.contract


def _write_ipynb(path: Path, cells: list[tuple[str, str]], formats: str | None) -> None:
    """Write a minimal notebook of ``(cell_type, source)`` pairs."""
    meta: dict = {}
    if formats is not None:
        meta["jupytext"] = {"formats": formats}

    def _cell(cell_type: str, source: str) -> dict:
        cell: dict = {"cell_type": cell_type, "source": [source], "metadata": {}}
        if cell_type == "code":
            cell |= {"outputs": [], "execution_count": None}
        return cell

    nb = {
        "cells": [_cell(t, s) for t, s in cells],
        "metadata": meta,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


class TestMirrorPath:
    """Resolving the ``.py`` a formats string declares."""

    def test_same_directory_pairing(self, tmp_path):
        nb = tmp_path / "demo.ipynb"
        got, reason = _mirror_path(nb, "ipynb,py:percent")
        assert reason is None
        assert got == tmp_path / "demo.py"

    def test_subdirectory_pairing(self, tmp_path):
        nb = tmp_path / "demo.ipynb"
        got, reason = _mirror_path(nb, "notebook_code//py:percent,ipynb")
        assert reason is None
        assert got == tmp_path / "notebook_code" / "demo.py"

    def test_ipynb_is_not_mistaken_for_a_python_entry(self, tmp_path):
        """`ipynb` contains the substring `py`.

        A membership test rather than a token match makes every pairing look
        like it declares two Python mirrors, and the guard reports the whole
        tree as unparseable — which is how it first behaved.
        """
        got, reason = _mirror_path(tmp_path / "demo.ipynb", "ipynb,py:percent")
        assert reason is None, f"'ipynb' was read as a python entry: {reason}"
        assert got is not None

    def test_a_format_with_no_python_side_is_reported_not_skipped(self, tmp_path):
        got, reason = _mirror_path(tmp_path / "demo.ipynb", "ipynb")
        assert got is None
        assert reason and "no python entry" in reason

    def test_unsupported_relative_prefix_is_reported_not_skipped(self, tmp_path):
        got, reason = _mirror_path(tmp_path / "demo.ipynb", "../elsewhere//py:percent,ipynb")
        assert got is None
        assert reason and "relative prefix" in reason


class TestPercentParsing:
    """Reading code cells out of a jupytext percent mirror."""

    def test_yaml_header_is_not_a_cell(self, tmp_path):
        py = tmp_path / "m.py"
        py.write_text("# ---\n# jupyter:\n#   x: 1\n# ---\n\n# %%\na = 1\n", encoding="utf-8")
        assert _percent_code_cells(py) == ["a = 1"]

    def test_markdown_cells_are_excluded(self, tmp_path):
        py = tmp_path / "m.py"
        py.write_text("# %% [markdown]\n# prose\n\n# %%\nb = 2\n", encoding="utf-8")
        assert _percent_code_cells(py) == ["b = 2"]

    def test_code_cell_metadata_containing_brackets_is_still_code(self, tmp_path):
        """`# %% tags=["imports"]` is a CODE cell.

        The cell type is the bracketed token immediately after the marker.
        Testing for any "[" in the remainder reclassifies this as markdown and
        silently drops a real code cell — the bug that made
        `analysis/hst_proposal` report as drifted when it was in sync.
        """
        py = tmp_path / "m.py"
        py.write_text('# %% tags=["imports"]\nimport os\n', encoding="utf-8")
        assert _percent_code_cells(py) == ["import os"]

    def test_raw_cells_are_excluded(self, tmp_path):
        py = tmp_path / "m.py"
        py.write_text("# %% [raw]\n# raw\n\n# %%\nc = 3\n", encoding="utf-8")
        assert _percent_code_cells(py) == ["c = 3"]


class TestGuardOnRealTrees:
    """End-to-end: the guard must go red on a pairing that does not hold."""

    def _run(self, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "check_notebook_pairing.py")],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    def _repo(self, tmp_path: Path) -> Path:
        root = tmp_path / "repo"
        (root / "notebooks").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        return root

    def test_matching_pair_is_accepted(self, tmp_path):
        root = self._repo(tmp_path)
        _write_ipynb(root / "notebooks" / "d.ipynb", [("code", "a = 1")], "ipynb,py:percent")
        (root / "notebooks" / "d.py").write_text("# %%\na = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        code = _pair_check(root)
        assert code == 0

    def test_drifted_pair_is_rejected(self, tmp_path):
        root = self._repo(tmp_path)
        _write_ipynb(root / "notebooks" / "d.ipynb", [("code", "a = 1")], "ipynb,py:percent")
        (root / "notebooks" / "d.py").write_text("# %%\na = 999\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _pair_check(root) == 1

    def test_missing_mirror_is_rejected(self, tmp_path):
        root = self._repo(tmp_path)
        _write_ipynb(root / "notebooks" / "d.ipynb", [("code", "a = 1")], "ipynb,py:percent")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _pair_check(root) == 1

    def test_unpaired_notebook_is_ignored(self, tmp_path):
        root = self._repo(tmp_path)
        _write_ipynb(root / "notebooks" / "d.ipynb", [("code", "a = 1")], None)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _pair_check(root) == 0

    def test_archive_and_docs_are_out_of_scope(self, tmp_path):
        """Drift under the excluded prefixes must not fail the build."""
        root = self._repo(tmp_path)
        for rel in ("notebooks/archive/a.ipynb", "docs/b.ipynb"):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            _write_ipynb(p, [("code", "a = 1")], "ipynb,py:percent")
            p.with_suffix(".py").write_text("# %%\na = 999\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        assert _pair_check(root) == 0

    def test_the_live_tree_is_clean(self):
        """The guard passes on this repository as committed."""
        proc = self._run(REPO_ROOT)
        assert proc.returncode == 0, proc.stdout + proc.stderr


def _pair_check(root: Path) -> int:
    """Run the guard against a throwaway repo by pointing REPO_ROOT at it."""
    script = (REPO_ROOT / "tools" / "check_notebook_pairing.py").read_text(encoding="utf-8")
    script = script.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]",
        f"REPO_ROOT = Path({str(root)!r})",
    )
    tmp_script = root / "_guard.py"
    tmp_script.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(tmp_script)], cwd=root, capture_output=True, text=True
    )
    return proc.returncode
