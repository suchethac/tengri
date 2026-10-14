# SPDX-License-Identifier: BSD-3-Clause
"""A Paper I figure family must not exit 0 while one of its figures is absent.

``regenerate.py`` takes each family's exit status as its verdict, so a family
that prints ``MISSING`` and exits 0 inflates the count (#2500). These tests pin
the reporting helpers, the refusal path of the CANDELS family, and the per-figure
count that ``regenerate.py`` reports.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FIGURES = _REPO / "paper1_figures"
_NOTEBOOKS = _FIGURES / "notebooks"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_mod():
    sys.path.insert(0, str(_FIGURES))
    try:
        import _run

        yield _run
    finally:
        sys.path.remove(str(_FIGURES))


def test_figure_outcome_reports_refusal_and_nonzero_status(run_mod, tmp_path):
    present = tmp_path / "fig.pdf"
    present.write_bytes(b"%PDF")

    assert run_mod.figure_outcome(0, present) is None
    assert run_mod.figure_outcome(0, tmp_path / "absent.pdf") is not None
    assert run_mod.figure_outcome(2, present) == "exit status 2"
    assert run_mod.figure_outcome("refused: ValueError: no", None) == "refused: ValueError: no"
    assert run_mod.figure_outcome(0, None) is None


def test_report_figures_is_nonzero_when_any_figure_is_missing(run_mod, capsys):
    assert run_mod.report_figures({"a.pdf": None, "b.pdf": None}) == 0
    assert run_mod.report_figures({"a.pdf": None, "b.pdf": "refused: x"}) == run_mod.SKIPPED
    out = capsys.readouterr().out
    assert "FIGURE a.pdf ok" in out
    assert "FIGURE b.pdf missing: refused: x" in out


def test_candels_family_exits_nonzero_when_a_figure_is_refused(run_mod, tmp_path, monkeypatch):
    """Family 02: a refused figure makes the family fail even though the others ran."""
    fits = tmp_path / "fits"
    fits.mkdir()
    (fits / "cell_0.json").write_text("{}")
    monkeypatch.setenv("PAPER1_FITS_DIR", str(fits))
    monkeypatch.chdir(_REPO)

    def fake_run_figure(module_name, argv):
        if module_name == "fig06_code_overlay":
            raise ValueError("cells sample a different SFH family")
        return 0

    monkeypatch.setattr(run_mod, "run_figure", fake_run_figure)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(_NOTEBOOKS / "02_candels_grid.py"), run_name="__main__")
    assert excinfo.value.code == run_mod.SKIPPED


def test_regenerate_counts_each_figure_from_its_family_lines(tmp_path, monkeypatch, capsys):
    """A family that exits 0 but names a missing figure is not counted as produced."""
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    figures = tmp_path / "figures"
    (notebooks / "01_a.py").write_text(
        "print('FIGURE a.pdf ok')\nprint('FIGURE b.pdf missing: refused')\nraise SystemExit(3)\n"
    )
    (notebooks / "02_b.py").write_text("print('FIGURE c.pdf ok')\n")

    regen = _load("regenerate_under_test", _FIGURES / "regenerate.py")
    monkeypatch.setattr(regen, "NOTEBOOKS", notebooks)
    monkeypatch.setattr(regen, "FIGURES", figures)

    status = regen.main([])
    out = capsys.readouterr().out
    assert status == 3
    assert "figures produced: 2, not produced: 1" in out
    assert "01_a.py: b.pdf: refused" in out
