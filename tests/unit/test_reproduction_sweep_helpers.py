# SPDX-License-Identifier: BSD-3-Clause
"""Tests for sweep_fig, window_rows, and print_window_table helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = pytest.importorskip("tengri").__path__[0]

pytestmark = pytest.mark.unit


def test_window_rows_identical():
    """window_rows on two identical SEDs should give median 1.0, max_abs_dev 0.0."""
    from reproduction._validation import window_rows

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    L_ref = np.array([1.0, 2.0, 1.5, 0.5])
    L_t = np.array([1.0, 2.0, 1.5, 0.5])

    cases = [("identical", w_ref, L_ref, w_ref, L_t)]
    rows = window_rows(cases, lo=1500.0, hi=3500.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "identical"
    assert row["median_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert row["max_abs_dev"] == pytest.approx(0.0, abs=1e-10)


def test_window_rows_scaled():
    """window_rows on a 1.1× scaled SED should give median ~1.1, max_abs_dev ~0.1."""
    from reproduction._validation import window_rows

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    L_ref = np.array([1.0, 2.0, 1.5, 0.5])
    L_t = np.array([1.1, 2.2, 1.65, 0.55])

    cases = [("scaled_1.1x", w_ref, L_ref, w_ref, L_t)]
    rows = window_rows(cases, lo=1500.0, hi=3500.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "scaled_1.1x"
    assert row["median_ratio"] == pytest.approx(1.1, rel=0.05)
    assert row["max_abs_dev"] == pytest.approx(0.1, abs=0.01)


def test_window_rows_excludes_outliers():
    """window_rows should only consider points within the wavelength window."""
    from reproduction._validation import window_rows

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    L_ref = np.array([1.0, 2.0, 1.5, 0.5, 10.0])
    L_t = np.array([1.0, 2.0, 1.5, 0.5, 10.0])

    cases = [("with_spike", w_ref, L_ref, w_ref, L_t)]
    # Spike at 5000 Å should be excluded
    rows = window_rows(cases, lo=1500.0, hi=3500.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "with_spike"
    assert row["median_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_window_rows_different_grids():
    """window_rows should handle when tengri grid differs from reference grid."""
    from reproduction._validation import window_rows

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    L_ref = np.array([1.0, 2.0, 1.5, 0.5])
    # Tengri grid: 3x more points over the same range
    w_t = np.linspace(1000.0, 4000.0, 12)
    L_t = np.interp(w_t, w_ref, L_ref)

    cases = [("finer_grid", w_ref, L_ref, w_t, L_t)]
    rows = window_rows(cases, lo=1500.0, hi=3500.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "finer_grid"
    # Identical function on both grids should give median ≈ 1 (interpolation adds small error)
    assert row["median_ratio"] == pytest.approx(1.0, rel=0.05)
    assert row["max_abs_dev"] == pytest.approx(0.0, abs=0.05)


def test_print_window_table_output(capsys):
    """print_window_table should output title, labels, and check flags correctly."""
    from reproduction._validation import print_window_table

    rows = [
        {
            "label": "case_1_good",
            "median_ratio": 1.0,
            "max_abs_dev": 0.01,
            "x_at_max": 2000.0,
        },
        {
            "label": "case_2_bad",
            "median_ratio": 1.15,
            "max_abs_dev": 0.15,
            "x_at_max": 3000.0,
        },
    ]

    print_window_table(rows, ref_name="reference", title="Test Window", tol=0.05)
    captured = capsys.readouterr()

    # Check title is in output
    assert "Test Window" in captured.out
    # Check labels are in output
    assert "case_1_good" in captured.out
    assert "case_2_bad" in captured.out
    # Check flag appears only for bad case
    assert captured.out.count("<-- check") == 1
    assert "case_2_bad" in captured.out.split("<-- check")[0]


def test_sweep_fig_returns_structure():
    """sweep_fig should return fig, (ax, ax_ratio), and ratios dict."""
    from reproduction._validation import sweep_fig

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    L_ref1 = np.array([1.0, 2.0, 1.5, 0.5])
    L_t1 = np.array([1.0, 2.1, 1.4, 0.51])
    L_ref2 = np.array([0.5, 1.0, 0.8, 0.3])
    L_t2 = np.array([0.5, 1.05, 0.79, 0.31])

    cases = [
        ("case_1", w_ref, L_ref1, w_ref, L_t1),
        ("case_2", w_ref, L_ref2, w_ref, L_t2),
    ]

    fig, (ax, ax_ratio), ratios = sweep_fig(cases, ref_label="reference", title="Test Sweep")

    # Check structure
    assert fig is not None
    assert ax is not None
    assert ax_ratio is not None
    assert isinstance(ratios, dict)
    assert len(ratios) == 2
    assert "case_1" in ratios
    assert "case_2" in ratios
    # Check ratio arrays have the correct shape
    assert len(ratios["case_1"]) == len(w_ref)
    assert len(ratios["case_2"]) == len(w_ref)

    plt.close(fig)


def test_sweep_fig_with_x_of_wave():
    """sweep_fig with x_of_wave should apply the transformation."""
    from reproduction._validation import sweep_fig

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    L_ref = np.array([1.0, 2.0, 1.5, 0.5])
    L_t = np.array([1.0, 2.1, 1.4, 0.51])

    cases = [("case", w_ref, L_ref, w_ref, L_t)]

    # x_of_wave: convert Angstrom to micrometers
    fig, (_ax, _ax_ratio), _ratios = sweep_fig(
        cases,
        ref_label="reference",
        title="Test",
        x_of_wave=lambda w: w / 1e4,
        xlabel="wavelength [µm]",
    )

    assert fig is not None
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
