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

    @pytest.mark.parametrize("n_filters", [1, 3, 5])
    def test_returns_axes_and_correct_line_count(self, n_filters):
        """Function returns Axes with one line per filter."""
        filters = OPTICAL_FILTERS[:n_filters]
        ax = plot_filter_curves(filters, show_eff_wave=False)
        assert isinstance(ax, plt.Axes)
        lines = ax.get_lines()
        assert len(lines) == n_filters

    def test_with_eff_wave_markers(self):
        """Effective wavelength markers add vertical lines."""
        ax = plot_filter_curves(OPTICAL_FILTERS[:3], show_eff_wave=True)
        lines = ax.get_lines()
        # 3 curves + 3 vertical markers = 6 lines
        assert len(lines) == 6

    def test_normalize_flag(self):
        """Normalized curves should have peak ≤ 1.0.

        The line count is asserted first because the peak check lives inside
        the loop: had ``plot_filter_curves`` drawn nothing, the loop would run
        zero times and this test would pass while proving nothing. Neighboring
        tests in this class already pin the count; these two did not.
        """
        ax = plot_filter_curves(OPTICAL_FILTERS[:2], normalize=True, show_eff_wave=False)
        lines = ax.get_lines()
        assert len(lines) == 2, f"expected one line per filter, drew {len(lines)}"
        for line in lines:
            ydata = line.get_ydata()
            assert len(ydata) > 0, "line drawn with no data"
            assert np.max(ydata) <= 1.0 + 1e-10

    def test_no_normalize_has_lines(self):
        """Without normalize, raw transmission values are used and returned as lines."""
        ax = plot_filter_curves(OPTICAL_FILTERS[:1], normalize=False, show_eff_wave=False)
        lines = ax.get_lines()
        assert len(lines) == 1
        # Verify line has data (not just exists)
        assert len(lines[0].get_ydata()) > 0

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
        """Providing ax parameter returns the same axes object."""
        _fig, ax_in = plt.subplots()
        ax_out = plot_filter_curves(OPTICAL_FILTERS[:2], ax=ax_in, show_eff_wave=False)
        assert ax_out is ax_in
        # Verify axes has the expected content
        assert len(ax_out.get_lines()) == 2

    def test_custom_alpha(self):
        """Alpha parameter propagates to all line objects.

        Count asserted before the loop for the same reason as
        ``test_normalize_flag``: "alpha reached every line" is vacuously true
        of no lines at all.
        """
        ax = plot_filter_curves(OPTICAL_FILTERS[:2], alpha=0.3, show_eff_wave=False)
        lines = ax.get_lines()
        assert len(lines) == 2, f"expected one line per filter, drew {len(lines)}"
        for line in lines:
            assert line.get_alpha() == pytest.approx(0.3)

    def test_xlabel_set(self):
        """Wavelength label is present on x-axis."""
        ax = plot_filter_curves(OPTICAL_FILTERS[:2])
        assert "Wavelength" in ax.get_xlabel()

    def test_ylabel_changes_with_normalize(self):
        """Ylabel indicates whether normalization was applied."""
        ax_raw = plot_filter_curves(OPTICAL_FILTERS[:1], normalize=False, show_eff_wave=False)
        ax_norm = plot_filter_curves(OPTICAL_FILTERS[:1], normalize=True, show_eff_wave=False)
        assert "Normalized" in ax_norm.get_ylabel()
        assert "Normalized" not in ax_raw.get_ylabel()


# ── plot_filter_coverage ──────────────────────────────────────────


class TestPlotFilterCoverage:
    """Tests for plot_filter_coverage()."""

    def test_returns_axes_with_log_scale(self):
        """Function returns log-scale Axes."""
        ax = plot_filter_coverage(OPTICAL_FILTERS[:3])
        assert isinstance(ax, plt.Axes)
        assert ax.get_xscale() == "log"

    def test_correct_ytick_count(self):
        """One y-tick per filter."""
        ax = plot_filter_coverage(OPTICAL_FILTERS[:3])
        assert len(ax.get_yticklabels()) == 3

    def test_custom_axes(self):
        """Providing ax parameter returns the same axes object with content."""
        _fig, ax_in = plt.subplots()
        ax_out = plot_filter_coverage(OPTICAL_FILTERS[:2], ax=ax_in)
        assert ax_out is ax_in
        # Verify axes has the expected content (at least one collection from barh)
        assert len(ax_out.get_yticklabels()) == 2

    def test_color_by_facility(self):
        """Mixed facility filters produce a legend when color_by_facility=True."""
        mixed = ["sdss_r", "2mass_j"]
        ax = plot_filter_coverage(mixed, color_by_facility=True)
        legend = ax.get_legend()
        # Two different facilities → legend should have entries
        assert legend is not None

    def test_no_color_by_facility_has_ticks(self):
        """Without facility coloring, axis still has correct y-ticks."""
        ax = plot_filter_coverage(OPTICAL_FILTERS[:3], color_by_facility=False)
        assert isinstance(ax, plt.Axes)
        # Verify the plot is populated (same facility, so no legend expected)
        assert len(ax.get_yticklabels()) == 3

    def test_show_labels_false_has_ticks(self):
        """Without labels, axis still has correct y-ticks."""
        ax = plot_filter_coverage(OPTICAL_FILTERS[:2], show_labels=False)
        assert isinstance(ax, plt.Axes)
        assert len(ax.get_yticklabels()) == 2

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

    def test_returns_axes_with_legend(self):
        """Function returns Axes with legend when labels provided."""
        ax = compare_filter_sets(OPTICAL_FILTERS[:2], IR_FILTERS[:2], labels=("Optical", "IR"))
        assert isinstance(ax, plt.Axes)
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Optical" in texts
        assert "IR" in texts

    def test_custom_axes(self):
        """Providing ax parameter returns the same axes object."""
        _fig, ax_in = plt.subplots()
        ax_out = compare_filter_sets(["sdss_r"], ["sdss_i"], ax=ax_in)
        assert ax_out is ax_in
        # Verify axes has expected content
        assert len(ax_out.collections) >= 2

    def test_normalize_flag(self):
        """Both sets should be normalized when normalize=True, producing filled regions."""
        ax = compare_filter_sets(["sdss_r"], ["sdss_i"], normalize=True)
        # Check that filled regions exist (PolyCollections from fill_between)
        collections = ax.collections
        assert len(collections) >= 2

    def test_no_normalize_has_content(self):
        """Without normalization, axes still contains filter comparison data."""
        ax = compare_filter_sets(["sdss_r"], ["sdss_i"], normalize=False)
        assert isinstance(ax, plt.Axes)
        # Verify axes has collections (fill_between regions)
        assert len(ax.collections) >= 2

    def test_xlabel_set(self):
        """Wavelength label is present on x-axis."""
        ax = compare_filter_sets(["sdss_r"], ["sdss_i"])
        assert "Wavelength" in ax.get_xlabel()

    def test_same_filter_both_sets(self):
        """Comparing a filter against itself should produce coincident curves."""
        ax = compare_filter_sets(["sdss_r"], ["sdss_r"])
        assert isinstance(ax, plt.Axes)
        # Verify the plot has the expected structure
        assert len(ax.collections) >= 2
