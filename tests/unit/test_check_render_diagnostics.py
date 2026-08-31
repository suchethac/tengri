# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_render_diagnostics.py``.

Both polarities: violations the guard must flag, and safe cases it must accept.
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

from check_render_diagnostics import check_notebook

pytestmark = pytest.mark.unit


def _make_notebook(cells: list[dict]) -> dict:
    """Create a minimal valid .ipynb structure."""
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _code_cell(source: str = "", outputs: list[dict] | None = None) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "source": source,
        "outputs": outputs or [],
        "metadata": {},
    }


def _text_output(text: str) -> dict:
    """Create a text output."""
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text,
    }


def _plain_output(text: str) -> dict:
    """Create a text/plain output."""
    return {
        "output_type": "display_data",
        "data": {"text/plain": text},
    }


# ---------------------------------------------------------------------------
# Synthetic fixture tests
# ---------------------------------------------------------------------------


def test_healthy_render_passes(tmp_path):
    """A clean notebook with no diagnostics passes."""
    nb = _make_notebook(
        [
            _code_cell("print('fit complete')", [_text_output("fit complete\n")]),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, warnings = check_notebook(path)
    assert len(failures) == 0
    assert len(warnings) == 0


def test_dead_fit_warning_fails(tmp_path):
    """DeadFitWarning text causes a failure."""
    nb = _make_notebook(
        [
            _code_cell(
                "fit()",
                [
                    _text_output(
                        "DeadFitWarning: dead fit: parameter(s) alpha have "
                        "1 unique draw in 600 samples (issue #1999)\n"
                    )
                ],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("DeadFitWarning" in f for f in failures)


def test_dead_fit_error_fails(tmp_path):
    """DeadFitError text causes a failure."""
    nb = _make_notebook(
        [
            _code_cell(
                "fit()",
                [_text_output("DeadFitError: dead fit condition detected\n")],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("DeadFitWarning" in f or "DeadFitError" in f for f in failures)


def test_catastrophic_rhat_max_notation_fails(tmp_path):
    """max R̂ = 1.5e14 causes a failure."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(diag)",
                [_text_output("max R̂ = 155806725793603.9062, divergences=0\n")],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("R-hat" in f for f in failures)


def test_catastrophic_rhat_inline_notation_fails(tmp_path):
    """R̂ = 1e14 causes a failure."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(diag)",
                [_text_output("R̂ = 1000000000000.0\n")],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("R-hat" in f for f in failures)


def test_catastrophic_rhat_hyphen_notation_fails(tmp_path):
    """max R-hat 1e13 causes a failure."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(diag)",
                [_text_output("max R-hat 10000000000000.0\n")],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("R-hat" in f for f in failures)


def test_honest_rhat_passes(tmp_path):
    """R-hat 1.271 (honest comparison arm) passes."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(diag)",
                [
                    _text_output(
                        "HMC: 21s   max R-hat 1.271   divergences 0   unique draws 567/600\n"
                    )
                ],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) == 0


def test_nan_in_table_fails(tmp_path):
    """NaN in a parameter summary table row fails."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(summary)",
                [
                    _text_output(
                        "                                 16%        50%        84%\n"
                        "            sfh_dpl_alpha    -0.123     0.456      1.234\n"
                        "            sfh_dpl_beta        nan      1.234      2.345\n"
                    )
                ],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("NaN" in f or "nan" in f for f in failures)


def test_handled_log_evidence_nan_warns_but_passes(tmp_path):
    """log_evidence: nan with fallback warns but doesn't fail."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(evidence)",
                [
                    _text_output(
                        "⚠ Evidence computation truncated. "
                        "log_evidence: nan using NSS weights as fallback.\n"
                    )
                ],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, warnings = check_notebook(path)
    assert len(failures) == 0
    assert len(warnings) > 0
    assert any("log_evidence" in w for w in warnings)


def test_unhandled_log_evidence_nan_fails(tmp_path):
    """log_evidence: nan without fallback fails."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(evidence)",
                [_text_output("log_evidence: nan\n")],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) > 0
    assert any("log_evidence" in f for f in failures)


def test_prose_nan_does_not_fail(tmp_path):
    """NaN in prose text (not in a table) does not fail."""
    nb = _make_notebook(
        [
            _code_cell(
                "print(explanation)",
                [_text_output("The parameter is NaN because the prior was too tight.\n")],
            ),
        ]
    )
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")

    failures, _warnings = check_notebook(path)
    assert len(failures) == 0


# ---------------------------------------------------------------------------
# Known-bad ledger tests
# ---------------------------------------------------------------------------


def _run_guard(root: Path) -> subprocess.CompletedProcess:
    """Run the guard with its ROOT pointed at a throwaway tree."""
    script = (REPO_ROOT / "tools" / "check_render_diagnostics.py").read_text(encoding="utf-8")
    script = script.replace(
        "ROOT = Path(__file__).resolve().parents[1]",
        f"ROOT = Path({str(root)!r})",
    )
    tmp_script = root / "_guard.py"
    tmp_script.write_text(script, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(tmp_script)], cwd=root, capture_output=True, text=True
    )


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A throwaway git repo holding exactly ``files``, all tracked."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_ledgered_path_converts_fail_to_known_bad(tmp_path):
    """A ledgered notebook that would fail shows as KNOWN-BAD and passes the run."""
    bad_nb = json.dumps(
        _make_notebook(
            [
                _code_cell(
                    "fit()",
                    [_text_output("max R̂ = 155806725793603.9062\nDeadFitWarning: dead fit\n")],
                ),
            ]
        )
    )
    proc = _run_guard(
        _make_repo(
            tmp_path,
            {
                "notebooks/05_fitting_photometry.ipynb": bad_nb,
                "notebooks/other.ipynb": json.dumps(
                    _make_notebook([_code_cell("print('ok')", [_text_output("ok\n")])])
                ),
            },
        )
    )
    # Should pass (exit 0) because the ledger entries don't cause failure
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Should mention KNOWN-BAD
    assert "KNOWN-BAD" in proc.stdout + proc.stderr
    # Should mention OK
    assert "OK:" in proc.stdout


def test_removing_ledger_entry_fails_the_run(tmp_path):
    """Mutation check: removing the ledger entry must make a bad render fail."""
    bad_nb = json.dumps(
        _make_notebook(
            [
                _code_cell(
                    "fit()",
                    [_text_output("max R̂ = 1e14\n")],
                ),
            ]
        )
    )
    # Create a guard without the ledger entry for this notebook
    script = (REPO_ROOT / "tools" / "check_render_diagnostics.py").read_text(encoding="utf-8")
    # Remove the ledger entry
    script = script.replace(
        '"notebooks/05_fitting_photometry.ipynb": "#2095",\n',
        "",
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    files = {
        "notebooks/05_fitting_photometry.ipynb": bad_nb,
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)

    # Write and run the modified guard
    tmp_script = root / "_guard.py"
    script = script.replace(
        "ROOT = Path(__file__).resolve().parents[1]",
        f"ROOT = Path({str(root)!r})",
    )
    tmp_script.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(tmp_script)], cwd=root, capture_output=True, text=True
    )

    # Should fail because the ledger entry is gone
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_weakening_rhat_threshold_breaks_synthetic_test(tmp_path):
    """Mutation check: changing R-hat threshold must affect detections."""
    # This test ensures the threshold is actually used
    # Create a notebook with R-hat = 1e14 (above current 2.0 threshold)
    nb_content = json.dumps(
        _make_notebook(
            [
                _code_cell(
                    "print(diag)",
                    [_text_output("max R̂ = 100000000000000.0\n")],
                ),
            ]
        )
    )

    # First, verify it fails with the normal guard (threshold 2.0)
    proc_normal = _run_guard(_make_repo(tmp_path, {"notebooks/test.ipynb": nb_content}))
    assert proc_normal.returncode == 1, proc_normal.stdout

    # Now create a modified guard with weakened threshold (>1e20)
    script = (REPO_ROOT / "tools" / "check_render_diagnostics.py").read_text(encoding="utf-8")
    script = script.replace("value > 2.0", "value > 1e20")

    root = tmp_path / "repo2"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    path = root / "notebooks/test.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nb_content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)

    tmp_script = root / "_guard.py"
    script = script.replace(
        "ROOT = Path(__file__).resolve().parents[1]",
        f"ROOT = Path({str(root)!r})",
    )
    tmp_script.write_text(script, encoding="utf-8")
    proc_weakened = subprocess.run(
        [sys.executable, str(tmp_script)], cwd=root, capture_output=True, text=True
    )

    # Should pass with weakened threshold (because 1e14 < 1e20)
    assert proc_weakened.returncode == 0, proc_weakened.stdout


# ---------------------------------------------------------------------------
# Real-tree test
# ---------------------------------------------------------------------------


def test_stale_ledger_entry_clean_notebook_fails(tmp_path):
    """A ledgered notebook with zero failures must fail (stale entry)."""
    clean_nb = json.dumps(_make_notebook([_code_cell("print('ok')", [_text_output("ok\n")])]))
    proc = _run_guard(
        _make_repo(
            tmp_path,
            {
                "notebooks/05_fitting_photometry.ipynb": clean_nb,
            },
        )
    )
    # Should fail because the ledger entry is stale (notebook is now clean)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "stale ledger entry" in proc.stdout + proc.stderr


def test_ledgered_notebook_with_failures_passes(tmp_path):
    """A ledgered notebook with failures is reported as KNOWN-BAD and passes."""
    bad_nb = json.dumps(
        _make_notebook(
            [
                _code_cell(
                    "fit()",
                    [_text_output("max R̂ = 1e14\nDeadFitWarning: dead fit\n")],
                ),
            ]
        )
    )
    proc = _run_guard(
        _make_repo(
            tmp_path,
            {
                "notebooks/05_fitting_photometry.ipynb": bad_nb,
                "notebooks/other.ipynb": json.dumps(
                    _make_notebook([_code_cell("print('ok')", [_text_output("ok\n")])])
                ),
            },
        )
    )
    # Should pass (exit 0) because the ledger entry has failures
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "KNOWN-BAD" in proc.stdout + proc.stderr


def test_live_tree_passes_with_full_ledger():
    """The guard passes on the real repository with the full ledger present."""
    from check_render_diagnostics import main

    # Run with argv=None to use actual sys.argv behavior
    exit_code = main([])
    assert exit_code == 0, "Guard should pass with the full ledger present"
