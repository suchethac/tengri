# SPDX-License-Identifier: BSD-3-Clause
"""Tests for filter transmission curve visualization.

Tests Feature 2 — the plotting functions in
``tengri.analysis.plotting.filters``.
"""

import matplotlib
import numpy as np
import pytest

pytestmark = pytest.mark.bounds
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tengri.analysis.plotting.filters import (
    compare_filter_sets,
    plot_filter_coverage,
    plot_filter_curves,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


OPTICAL_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
IR_FILTERS = ["2mass_j", "2mass_h", "2mass_ks"]


# ── plot_filter_curves ────────────────────────────────────────────


class TestPlotFilterCurves:
    """Tests for plot_filter_curves()."""

    def test_returns_axes(self):
        ax = plot_filter_curves(OPTICAL_FILTERS[:3])
        assert isinstance(ax, plt.Axes)

    def test_correct_number_of_lines(self):
        ax = plot_filter_curves(OPTICAL_FILTERS[:3], show_eff_wave=False)
        lines = ax.get_lines()
        assert len(lines) == 3

    def test_with_eff_wave_markers(self):
        """Effective wavelength markers add vertical lines."""
        ax = plot_filter_curves(OPTICAL_FILTERS[:3], show_eff_wave=True)
        lines = ax.get_lines()
        # 3 curves + 3 vertical markers = 6 lines
        assert len(lines) == 6

    def test_normalize_flag(self):
        """Normalized curves should have peak ≤ 1.0."""
        ax = plot_filter_curves(OPTICAL_FILTERS[:2], normalize=True, show_eff_wave=False)
        for line in ax.get_lines():
            ydata = line.get_ydata()
            assert np.max(ydata) <= 1.0 + 1e-10

    def test_no_normalize(self):
        """Without normalize, raw transmission values are used."""
        ax = plot_filter_curves(OPTICAL_FILTERS[:1], normalize=False, show_eff_wave=False)
        # Just verify no error — transmission may or may not exceed 1
        assert len(ax.get_lines()) == 1

    def test_label_filters(self):
        ax = plot_filter_curves(OPTICAL_FILTERS[:3], label_filters=True, show_eff_wave=False)
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert len(texts) == 3

    def test_no_labels(self):
        ax = plot_filter_curves(OPTICAL_FILTERS[:3], label_filters=False, show_eff_wave=False)
        legend = ax.get_legend()
        assert legend is None

    def test_custom_axes(self):
        _fig, ax_in = plt.subplots()
        ax_out = plot_filter_curves(OPTICAL_FILTERS[:2], ax=ax_in, show_eff_wave=False)
        assert ax_out is ax_in

    def test_single_filter(self):
        ax = plot_filter_curves(["sdss_r"], show_eff_wave=False)
        assert len(ax.get_lines()) == 1

    def test_custom_alpha(self):
        ax = plot_filter_curves(OPTICAL_FILTERS[:2], alpha=0.3, show_eff_wave=False)
        for line in ax.get_lines():
            assert line.get_alpha() == pytest.approx(0.3)

    def test_xlabel_set(self):
        ax = plot_filter_curves(OPTICAL_FILTERS[:2])
        assert "Wavelength" in ax.get_xlabel()

    def test_ylabel_changes_with_normalize(self):
        ax_raw = plot_filter_curves(OPTICAL_FILTERS[:1], normalize=False, show_eff_wave=False)
        ax_norm = plot_filter_curves(OPTICAL_FILTERS[:1], normalize=True, show_eff_wave=False)
        assert "Normalized" in ax_norm.get_ylabel()
        assert "Normalized" not in ax_raw.get_ylabel()


# ── plot_filter_coverage ──────────────────────────────────────────


class TestPlotFilterCoverage:
    """Tests for plot_filter_coverage()."""

    def test_returns_axes(self):
        ax = plot_filter_coverage(OPTICAL_FILTERS[:3])
        assert isinstance(ax, plt.Axes)

    def test_log_xscale(self):
        ax = plot_filter_coverage(OPTICAL_FILTERS)
        assert ax.get_xscale() == "log"

    def test_correct_ytick_count(self):
        """One y-tick per filter."""
        ax = plot_filter_coverage(OPTICAL_FILTERS[:3])
        assert len(ax.get_yticklabels()) == 3

    def test_custom_axes(self):
        _fig, ax_in = plt.subplots()
        ax_out = plot_filter_coverage(OPTICAL_FILTERS[:2], ax=ax_in)
        assert ax_out is ax_in

    def test_color_by_facility(self):
        mixed = ["sdss_r", "2mass_j"]
        ax = plot_filter_coverage(mixed, color_by_facility=True)
        legend = ax.get_legend()
        # Two different facilities → legend should have entries
        assert legend is not None

    def test_no_color_by_facility(self):
        ax = plot_filter_coverage(OPTICAL_FILTERS[:3], color_by_facility=False)
        # No facility-based legend expected (all same facility anyway)
        assert isinstance(ax, plt.Axes)

    def test_show_labels_false(self):
        ax = plot_filter_coverage(OPTICAL_FILTERS[:2], show_labels=False)
        # Should still work without text annotations
        assert isinstance(ax, plt.Axes)

    def test_sorted_by_wavelength(self):
        """Filters should be sorted by effective wavelength (y axis)."""
        ax = plot_filter_coverage(OPTICAL_FILTERS)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # sdss_u has shortest wavelength, should be first (top after invert)
        assert labels[0] == "sdss_u"
        assert labels[-1] == "sdss_z"


# ── compare_filter_sets ───────────────────────────────────────────


class TestCompareFilterSets:
    """Tests for compare_filter_sets()."""

    def test_returns_axes(self):
        ax = compare_filter_sets(OPTICAL_FILTERS[:2], IR_FILTERS[:2])
        assert isinstance(ax, plt.Axes)

    def test_legend_present(self):
        ax = compare_filter_sets(OPTICAL_FILTERS[:2], IR_FILTERS[:2], labels=("Optical", "IR"))
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Optical" in texts
        assert "IR" in texts

    def test_custom_axes(self):
        _fig, ax_in = plt.subplots()
        ax_out = compare_filter_sets(["sdss_r"], ["sdss_i"], ax=ax_in)
        assert ax_out is ax_in

    def test_normalize_flag(self):
        """Both sets should be normalized when normalize=True."""
        ax = compare_filter_sets(["sdss_r"], ["sdss_i"], normalize=True)
        # Check that filled regions exist (PolyCollections from fill_between)
        collections = ax.collections
        assert len(collections) >= 2

    def test_no_normalize(self):
        ax = compare_filter_sets(["sdss_r"], ["sdss_i"], normalize=False)
        assert isinstance(ax, plt.Axes)

    def test_xlabel_set(self):
        ax = compare_filter_sets(["sdss_r"], ["sdss_i"])
        assert "Wavelength" in ax.get_xlabel()

    def test_same_filter_both_sets(self):
        """Comparing a filter against itself should work."""
        ax = compare_filter_sets(["sdss_r"], ["sdss_r"])
        assert isinstance(ax, plt.Axes)
