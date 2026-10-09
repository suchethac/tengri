# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for tools/check_rendered_registry_strings.py (#2396).

The guard reads executed notebook outputs and compares the registry tables they
print with the live registry. These tests drive its line matcher with a small
fake registry, so they pin the matching rules without importing tengri.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
_TOOL = ROOT / "tools" / "check_rendered_registry_strings.py"


def _load_tool():
    name = "check_rendered_registry_strings"
    spec = importlib.util.spec_from_file_location(name, _TOOL)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the defining module through sys.modules at class creation.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_tool()


@pytest.fixture
def registry():
    return guard.LiveRegistry(
        texts=(
            "Fritz et al. 2006, MNRAS, 366, 767",
            "Fritz et al. 2006 smooth-dust torus",
            "5100 Å (510 nm; powerlaw AGN disc)",
            "Tabulated Schreiber+2018 library, AGNfitter-rX packaging; (T_dust, f_PAH)",
        ),
        names=frozenset({"fritz", "agn_grahsp_a_bc"}),
        statuses=frozenset({"production", "experimental", "primary"}),
        parameter_names=frozenset({"agn_fritz_beta"}),
    )


def test_live_menu_row_passes(registry):
    line = "fritz                         production    Fritz et al. 2006, MNRAS, 366, 767"
    assert guard.check_line(line, registry) == []


def test_stale_citation_is_flagged(registry):
    line = "fritz                         production    Fritz et al. 2006, A&A, 470, 221"
    problems = guard.check_line(line, registry)
    assert len(problems) == 1
    assert "A&A, 470, 221" in problems[0]


def test_retired_description_is_flagged(registry):
    line = "agn_grahsp_a_bc             experimental  powerlaw at 3000 nm"
    problems = guard.check_line(line, registry)
    assert problems and "powerlaw at 3000 nm" in problems[0]


def test_truncated_live_text_passes(registry):
    line = "fritz                         production    5100 Å (510 nm; powerlaw AGN..."
    assert guard.check_line(line, registry) == []


def test_removed_name_is_flagged(registry):
    line = "retired_model                 production    Fritz et al. 2006 smooth-dust torus"
    problems = guard.check_line(line, registry)
    assert problems == ["menu row names 'retired_model', which is not in the live registry"]


def test_parameter_row_must_name_a_live_parameter(registry):
    live = "agn_fritz_beta  Fixed(-0.5)  deg  tengri.components.agn._params  PARAMS"
    retired = "agn_fritz_psy  Fritz2006 viewing angle  deg  tengri.components.agn._params  PARAMS"
    assert guard.check_line(live, registry) == []
    problems = guard.check_line(retired, registry)
    assert problems == [
        "parameter row names 'agn_fritz_psy', which is not in the live parameter menu"
    ]


@pytest.mark.parametrize(
    "line",
    [
        "name                          status        citation",
        "  tengri  calzetti, cardelli, conroy2010",
        "fritz  [{'name': 'agn_fritz_beta', 'default': 'Fixed(-0.5)'}]",
        "Result  ok  0.4 min",
        "",
    ],
)
def test_non_registry_lines_are_ignored(registry, line):
    assert guard.check_line(line, registry) == []


def test_check_notebook_reports_cell_and_relative_path(tmp_path, registry):
    import json

    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# text"], "outputs": []},
            {
                "cell_type": "code",
                "source": ["print(1)"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "fritz                         production    "
                            "Fritz et al. 2006, A&A, 470, 221\n"
                        ],
                    }
                ],
            },
        ]
    }
    path = tmp_path / "docs" / "spine" / "03.ipynb"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(nb), encoding="utf-8")
    findings = guard.check_notebook(path, registry, tmp_path)
    assert len(findings) == 1
    assert findings[0].notebook == "docs/spine/03.ipynb"
    assert findings[0].cell == 1


def test_archive_notebooks_are_excluded(tmp_path):
    for rel in ("notebooks/a.ipynb", "notebooks/archive/b.ipynb"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    found = guard.published_notebooks(tmp_path)
    assert [p.name for p in found] == ["a.ipynb"]


def test_current_published_notebooks_match_live_registry():
    """Integration pin: the checked-in renders quote the live registry (needs tengri)."""
    pytest.importorskip("tengri")
    registry = guard.load_live_registry()
    problems = [
        f.render()
        for path in guard.published_notebooks(ROOT)
        for f in guard.check_notebook(path, registry, ROOT)
    ]
    assert problems == []
