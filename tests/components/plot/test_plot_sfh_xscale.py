# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the log-t axis option on ``plot_sfh`` (field-SFH recovery notebook)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tengri.analysis.plotting import plot_sfh
from tests._shared_mocks import MockSpec

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _MockModel:
    """Minimal model stub exposing ``spec.stochastic`` and ``predict_sfh``."""

    spec = MockSpec(free_names=[], stochastic=True)

    def __init__(self):
        # Youngest SFH grid point ~1 Myr = 1e-3 Gyr, matching the real grid.
        self._t_gyr = np.logspace(-3.0, np.log10(13.5), 128)

    def predict_sfh(self, params):
        # SFR shape is irrelevant to the axis-scaling logic under test.
        sfr = 5.0 + 2.0 * np.sin(self._t_gyr)
        return {"t_gyr": self._t_gyr, "sfr_full": sfr, "sfr_mean": sfr}


class _MockPosterior:
    def __init__(self, n=50):
        rng = np.random.default_rng(0)
        self.samples = {"sfh_field_psd_sigma": 0.3 + 0.05 * rng.standard_normal(n)}
        self.params = {"sfh_field_psd_sigma": 0.3}


class TestPlotSFHXScale:
    def test_linear_default_is_reversed_present_at_right(self):
        fig, ax = plt.subplots()
        plot_sfh(_MockModel(), _MockPosterior(), ax=ax)  # default xscale="linear"
        assert ax.get_xscale() == "linear"
        lo, hi = ax.get_xlim()
        assert lo > hi  # reversed: high lookback at left, present at right
        plt.close(fig)

    def test_log_axis_is_ascending_and_clamped_off_zero(self):
        fig, ax = plt.subplots()
        plot_sfh(_MockModel(), _MockPosterior(), ax=ax, xscale="log")
        assert ax.get_xscale() == "log"
        left, right = ax.get_xlim()
        assert 0.0 < left < right  # ascending, clamped off zero (log-safe)
        assert left <= 1e-3 + 1e-9  # clamped to the youngest grid point (~1 Myr)
        plt.close(fig)

    def test_invalid_xscale_raises(self):
        with pytest.raises(ValueError, match="xscale"):
            plot_sfh(_MockModel(), _MockPosterior(), xscale="symlog")
