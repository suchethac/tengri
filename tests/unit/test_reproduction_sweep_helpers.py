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


def test_window_rows_peak_relative_tails():
    """window_rows with rel_to='peak' should handle tails correctly."""
    from reproduction._validation import window_rows

    # Gaussian SED with tails
    w = np.linspace(0, 10, 201)
    L_ref = np.exp(-0.5 * ((w - 5) / 1.0) ** 2)
    L_t = L_ref + 1e-3

    cases = [("gaussian", w, L_ref, w, L_t)]

    # Pointwise mode: tails will have large deviations
    rows_point = window_rows(cases, lo=0, hi=10, rel_to="point")
    assert len(rows_point) == 1
    assert rows_point[0]["rel_to"] == "point"
    assert rows_point[0]["max_abs_dev"] > 0.05  # tails blow up

    # Peak mode: should give ~1e-3 max deviation
    rows_peak = window_rows(cases, lo=0, hi=10, rel_to="peak")
    assert len(rows_peak) == 1
    assert rows_peak[0]["rel_to"] == "peak"
    assert rows_peak[0]["max_abs_dev"] == pytest.approx(1e-3, rel=1e-6)
    assert rows_peak[0]["median_ratio"] == pytest.approx(1.0, abs=0.01)


def test_window_rows_point_mode_unchanged():
    """window_rows with rel_to='point' should match existing default behavior."""
    from reproduction._validation import window_rows

    w_ref = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    L_ref = np.array([1.0, 2.0, 1.5, 0.5])
    L_t = np.array([1.1, 2.2, 1.65, 0.55])

    cases = [("scaled", w_ref, L_ref, w_ref, L_t)]

    # Default call (should be rel_to="point")
    rows_default = window_rows(cases, lo=1500.0, hi=3500.0)
    # Explicit rel_to="point"
    rows_point = window_rows(cases, lo=1500.0, hi=3500.0, rel_to="point")

    assert len(rows_default) == len(rows_point) == 1
    assert rows_default[0]["median_ratio"] == pytest.approx(rows_point[0]["median_ratio"])
    assert rows_default[0]["max_abs_dev"] == pytest.approx(rows_point[0]["max_abs_dev"])
    assert rows_default[0]["x_at_max"] == pytest.approx(rows_point[0]["x_at_max"])
    assert rows_point[0]["rel_to"] == "point"


def test_window_rows_rejects_unknown_rel_to():
    """window_rows should reject unknown rel_to values."""
    from reproduction._validation import window_rows

    w = np.array([1.0, 2.0, 3.0])
    L = np.array([1.0, 2.0, 1.0])
    cases = [("test", w, L, w, L)]

    with pytest.raises(ValueError, match="rel_to must be"):
        window_rows(cases, lo=0, hi=4, rel_to="unknown")


def test_window_rows_peak_override_scales_deviation():
    """window_rows with peak= should override the window maximum for normalization."""
    from reproduction._validation import window_rows

    w = np.linspace(0, 10, 101)
    L_ref = np.full(101, 0.001)
    L_t = L_ref + 0.002  # difference of 0.002

    cases = [("low_ref", w, L_ref, w, L_t)]

    # Without peak=: window maximum is 0.001, so max_abs_dev = 0.002 / 0.001 = 2.0 (200%)
    rows_no_peak = window_rows(cases, lo=0, hi=10, rel_to="peak")
    assert len(rows_no_peak) == 1
    assert rows_no_peak[0]["max_abs_dev"] == pytest.approx(2.0, rel=1e-9)

    # With peak=1.0: max_abs_dev = 0.002 / 1.0 = 0.002
    rows_with_peak = window_rows(cases, lo=0, hi=10, rel_to="peak", peak=1.0)
    assert len(rows_with_peak) == 1
    assert rows_with_peak[0]["max_abs_dev"] == pytest.approx(0.002, rel=1e-9)

    # With peak=1.0: no point reaches 1% of 1.0 (0.01), so median_ratio is nan
    assert np.isnan(rows_with_peak[0]["median_ratio"])


def test_window_rows_peak_override_rejects_point_mode_and_nonpositive():
    """window_rows should reject peak= with rel_to='point' and reject non-positive peak."""
    from reproduction._validation import window_rows

    w = np.array([1.0, 2.0, 3.0])
    L = np.array([1.0, 2.0, 1.0])
    cases = [("test", w, L, w, L)]

    # peak= with rel_to="point" should raise ValueError
    with pytest.raises(ValueError, match="peak= applies to rel_to='peak' only"):
        window_rows(cases, lo=0, hi=4, rel_to="point", peak=1.0)

    # Non-positive peak should raise ValueError
    with pytest.raises(ValueError, match="peak must be positive"):
        window_rows(cases, lo=0, hi=4, rel_to="peak", peak=0.0)

    with pytest.raises(ValueError, match="peak must be positive"):
        window_rows(cases, lo=0, hi=4, rel_to="peak", peak=-1.0)


def test_print_window_table_x_unit_and_peak_header(capsys):
    """print_window_table should show x_unit and peak-specific headers."""
    from reproduction._validation import print_window_table

    # Point mode rows
    rows_point = [
        {
            "label": "case_point",
            "median_ratio": 1.0,
            "max_abs_dev": 0.01,
            "x_at_max": 2000.0,
            "rel_to": "point",
        }
    ]

    # Peak mode rows
    rows_peak = [
        {
            "label": "case_peak",
            "median_ratio": 1.0,
            "max_abs_dev": 0.001,
            "x_at_max": 5000.0,
            "rel_to": "peak",
        }
    ]

    # Test default x_unit (should be "Å")
    print_window_table(rows_point, ref_name="ref", title="Point Mode")
    captured = capsys.readouterr()
    assert "x at max [Å]" in captured.out
    assert "max |Δ| [%]" in captured.out

    # Test custom x_unit
    print_window_table(rows_point, ref_name="ref", title="Custom Unit", x_unit="yr")
    captured = capsys.readouterr()
    assert "x at max [yr]" in captured.out

    # Test peak mode header
    print_window_table(rows_peak, ref_name="ref", title="Peak Mode")
    captured = capsys.readouterr()
    assert "max |Δ| [% peak]" in captured.out
    assert "x at max [Å]" in captured.out

    # Test x_scale parameter with large values
    rows_gyr = [
        {
            "label": "case_gyr",
            "median_ratio": 1.0,
            "max_abs_dev": 0.01,
            "x_at_max": 4.95e9,  # Large value to be scaled to Gyr
            "rel_to": "point",
        }
    ]
    print_window_table(rows_gyr, ref_name="ref", title="Scaled", x_unit="Gyr", x_scale=1e-9)
    captured = capsys.readouterr()
    assert "x at max [Gyr]" in captured.out
    assert "4.95" in captured.out
    assert "4949758794" not in captured.out  # Should not show unscaled value

    # Test mixed rows raise ValueError
    mixed_rows = rows_point + rows_peak
    with pytest.raises(ValueError, match="same rel_to mode"):
        print_window_table(mixed_rows, ref_name="ref", title="Mixed")


def test_filter_rows_native_preserves_a_narrow_line_on_a_coarse_grid():
    """filter_rows_native should preserve a narrow line when grids differ."""
    from reproduction._validation import (
        BROAD_FILTERS,
        band_average,
        filter_rows,
        filter_rows_native,
        load_filter,
    )

    # Fine grid covering SDSS g filter range (3630-5830 Å, with margin)
    w_fine = np.arange(3500.0, 6000.0, 0.5)
    sigma = 2.0  # Å
    peak_wavelength = 5007.0
    ew = 200.0  # Angstrom equivalent width
    # For a Gaussian: integral = peak * sigma * sqrt(2*pi) = ew
    # So: peak = ew / (sigma * sqrt(2*pi))
    line_peak = ew / (sigma * np.sqrt(2 * np.pi))
    line = line_peak * np.exp(-0.5 * ((w_fine - peak_wavelength) / sigma) ** 2)
    L_fine = 1.0 + line  # Flat continuum + line (tengri side)

    # Coarse grid: same range, much coarser spacing (reference side)
    w_coarse = np.arange(3500.0, 6000.0, 25.0)
    L_coarse = np.ones_like(w_coarse)  # Just continuum, no line on reference

    # Get SDSS g filter (covers 5007 Å)
    g_filter = BROAD_FILTERS[3]  # SDSS g
    assert g_filter[1] == "SDSS g"
    fw, ft = load_filter(g_filter[0])

    # Demonstrate the aliasing problem: when you interpolate L_fine to coarse grid
    # and band-average, you lose the line information
    L_fine_interp_to_coarse = np.interp(w_coarse, w_fine, L_fine)
    band_via_interp = band_average(
        w_coarse, L_fine_interp_to_coarse, fw, ft, weight="photon"
    )
    band_coarse_ref = band_average(w_coarse, L_coarse, fw, ft, weight="photon")

    # Band-average on fine grid (correct result)
    band_on_fine = band_average(w_fine, L_fine, fw, ft, weight="photon")
    band_coarse_ref_direct = band_average(w_coarse, L_coarse, fw, ft, weight="photon")

    # The ratio via interpolation + band-average should be closer to 1.0 than
    # it should be
    if np.isfinite(band_via_interp) and np.isfinite(band_coarse_ref) and band_coarse_ref > 0:
        ratio_via_interp = band_via_interp / band_coarse_ref
        # The ratio should be closer to 1.0 than the true ratio
        if (
            np.isfinite(band_on_fine)
            and np.isfinite(band_coarse_ref_direct)
            and band_coarse_ref_direct > 0
        ):
            ratio_true = band_on_fine / band_coarse_ref_direct
            # Aliasing error: the interpolated ratio should be significantly smaller
            # than the true ratio
            aliasing_underestimation = (ratio_true - ratio_via_interp) / ratio_true
            assert (
                aliasing_underestimation > 0.10
            ), f"Expected >10% underestimation, got {aliasing_underestimation:.1%}"

    # Now test filter_rows_native: should preserve the line and give correct ratio
    rows_native = filter_rows_native(
        w_fine, L_fine, w_coarse, L_coarse, filters=(g_filter,), weight="photon"
    )
    assert len(rows_native) == 1
    label, _lambda_eff_um, _L_t_band, _L_ref_band, ratio_native = rows_native[0]
    assert label == "SDSS g"
    # filter_rows_native should give the same result as band-averaging each on its own grid
    # So it should match band_on_fine / band_coarse_ref_direct
    if np.isfinite(band_on_fine) and np.isfinite(band_coarse_ref_direct):
        expected_ratio = band_on_fine / band_coarse_ref_direct
        assert ratio_native == pytest.approx(expected_ratio, rel=1e-12)
        # And it should be significantly larger than the interpolated ratio
        assert ratio_native > ratio_via_interp * 1.05  # At least 5% larger

    # Also verify that filter_rows (interpolated method) gives a worse result
    rows_bad = filter_rows(
        w_coarse, L_fine_interp_to_coarse, L_coarse, filters=(g_filter,), weight="photon"
    )
    assert len(rows_bad) == 1
    _label, _lambda_eff_um, _L_t, _L_ref, ratio_bad = rows_bad[0]
    # filter_rows result should be significantly smaller than filter_rows_native
    assert ratio_bad < ratio_native * 0.95


def test_filter_rows_native_matches_filter_rows_on_a_shared_grid():
    """filter_rows_native should match filter_rows when both use the same grid."""
    from reproduction._validation import (
        BROAD_FILTERS,
        filter_rows,
        filter_rows_native,
    )

    # Create a smooth spectrum on one grid
    w = np.linspace(1000.0, 20000.0, 500)
    L_t = 1.0 + 0.1 * np.sin(2 * np.pi * w / 5000.0)  # Smooth variation
    L_ref = L_t * 1.05  # Reference is slightly different

    # Use first 5 broad filters
    test_filters = BROAD_FILTERS[:5]

    # Call filter_rows (both on shared grid w)
    rows_shared = filter_rows(w, L_t, L_ref, filters=test_filters, weight="photon")

    # Call filter_rows_native (both on same grid w)
    rows_native = filter_rows_native(w, L_t, w, L_ref, filters=test_filters, weight="photon")

    # Both should return same number of rows
    assert len(rows_shared) == len(rows_native) == len(test_filters)

    # Each row should match to very high precision
    for i, (row_shared, row_native) in enumerate(zip(rows_shared, rows_native)):
        label_s, piv_s, L_t_s, L_ref_s, ratio_s = row_shared
        label_n, piv_n, L_t_n, L_ref_n, ratio_n = row_native

        assert label_s == label_n, f"Row {i}: label mismatch"
        assert piv_s == pytest.approx(piv_n, rel=1e-12)
        assert L_t_s == pytest.approx(L_t_n, rel=1e-12)
        assert L_ref_s == pytest.approx(L_ref_n, rel=1e-12)

        # Ratio match should also be excellent (both NaN or both finite and equal)
        if np.isnan(ratio_s) and np.isnan(ratio_n):
            pass  # Both NaN is fine
        elif np.isfinite(ratio_s) and np.isfinite(ratio_n):
            assert ratio_s == pytest.approx(ratio_n, rel=1e-12)
        else:
            raise AssertionError(f"Row {i}: one ratio is NaN, the other is not")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
