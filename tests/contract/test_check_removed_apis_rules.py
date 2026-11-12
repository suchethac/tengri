# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the #3358 rules in tools/check_removed_apis.py.

The rules: ``from tengri import Model`` (and ParamSpec, HierarchicalFitter,
SpectroscopyConfig), ``.precompute_spectroscopy(`` as a model method (the
module-level function stays valid), a free ``sfh_tsnorm_peak_lbt_gyr=`` prior,
and python fences in docs markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import check_removed_apis as guard

pytestmark = pytest.mark.contract


def _reasons(tmp_path: Path, name: str, source: str) -> list[str]:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return [reason for _where, reason in guard.scan_file(path)]


@pytest.mark.parametrize(
    "name", ["Model", "ParamSpec", "HierarchicalFitter", "SpectroscopyConfig"]
)
def test_removed_top_level_import_is_flagged(tmp_path, name):
    found = _reasons(tmp_path, "s.py", f"from tengri import {name}\n")
    assert len(found) == 1
    assert f"`tengri.{name}` was removed" in found[0]


def test_canonical_top_level_import_is_not_flagged(tmp_path):
    src = "from tengri import SEDModel, Parameters, Spectroscopy, PopulationFitter\n"
    assert _reasons(tmp_path, "s.py", src) == []


def test_model_method_precompute_spectroscopy_is_flagged(tmp_path):
    found = _reasons(tmp_path, "s.py", "model.precompute_spectroscopy(wave)\n")
    assert len(found) == 1
    assert "approx=SpectrumPrecomp()" in found[0]


def test_module_level_precompute_spectroscopy_is_not_flagged(tmp_path):
    src = (
        "from tengri.components.stellar.sps import precompute\n"
        "precompute.precompute_spectroscopy(ssp, wave)\n"
    )
    assert _reasons(tmp_path, "s.py", src) == []


def test_bare_imported_precompute_spectroscopy_is_not_flagged(tmp_path):
    src = (
        "from tengri.components.stellar.sps.precompute import precompute_spectroscopy\n"
        "precompute_spectroscopy(ssp, wave)\n"
    )
    assert _reasons(tmp_path, "s.py", src) == []


def test_free_gyr_prior_is_flagged(tmp_path):
    src = "build(sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0))\n"
    found = _reasons(tmp_path, "s.py", src)
    assert len(found) == 1
    assert "Fixed-only alias" in found[0]
    assert "sfh_tsnorm_peak_lbt_frac" in found[0]


def test_free_gyr_sentinel_is_flagged(tmp_path):
    assert _reasons(tmp_path, "s.py", "build(sfh_tsnorm_peak_lbt_gyr=FREE)\n")


@pytest.mark.parametrize(
    "value", ["Fixed(2.0)", "tengri.Fixed(2.0)", "2.0", "5"], ids=["fixed", "attr", "float", "int"]
)
def test_pinned_gyr_value_is_not_flagged(tmp_path, value):
    assert _reasons(tmp_path, "s.py", f"build(sfh_tsnorm_peak_lbt_gyr={value})\n") == []


def test_frac_prior_is_not_flagged(tmp_path):
    src = "build(sfh_tsnorm_peak_lbt_frac=Uniform(0.07, 0.87))\n"
    assert _reasons(tmp_path, "s.py", src) == []


def test_notebook_code_cell_is_scanned(tmp_path):
    import json

    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["from tengri import Model\n"]},
            {"cell_type": "code", "source": ["from tengri import ParamSpec\n"]},
        ]
    }
    path = tmp_path / "n.ipynb"
    path.write_text(json.dumps(nb))
    found = list(guard.scan_file(path))
    assert [where for where, _ in found] == ["cell 2"]


def test_docs_fence_is_scanned_but_prose_is_not(tmp_path):
    text = (
        "Prose may name `from tengri import Model` freely.\n\n"
        "```python\nfrom tengri import HierarchicalFitter\n```\n"
    )
    path = tmp_path / "page.md"
    path.write_text(text)
    found = list(guard.scan_file(path))
    assert len(found) == 1
    assert found[0][0].startswith("fence at line 3")


def test_markdown_is_scanned_only_under_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "bench").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x")
    (tmp_path / "bench" / "b.md").write_text("x")
    names = {p.name for p in guard.iter_files(["docs", "bench"])}
    assert names == {"a.md"}


def test_history_snapshot_is_excluded_from_fence_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    hist = tmp_path / "docs" / "dev" / "history"
    hist.mkdir(parents=True)
    (hist / "old.md").write_text("x")
    (tmp_path / "docs" / "dev" / "live.md").write_text("x")
    names = {p.name for p in guard.iter_files(["docs"])}
    assert names == {"live.md"}


def test_main_exit_code_follows_hits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bad.py").write_text("from tengri import Model\n")
    assert guard.main(["--root", "scripts"]) == 1
    assert "bad.py" in capsys.readouterr().out
    (tmp_path / "scripts" / "bad.py").write_text("from tengri import SEDModel\n")
    assert guard.main(["--root", "scripts"]) == 0
