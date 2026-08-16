# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for plot_1d_posterior and plot_calibration (closes #509)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tengri.analysis.plotting import plot_1d_posterior, plot_calibration

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _MockPosterior:
    """Minimal posterior stub exposing the .samples attribute used by the plotters."""

    def __init__(self, samples):
        self.samples = samples


class TestPlot1DPosterior:
    def test_plots_one_param_with_summary(self):
        rng = np.random.default_rng(0)
        post = _MockPosterior({"redshift": rng.normal(2.0, 0.1, size=2000)})
        fig, ax = plt.subplots()
        out_ax = plot_1d_posterior(post, "redshift", ax=ax)
        assert out_ax is ax
        # An annotation should have been added with the median/16-84 summary.
        assert len(ax.texts) >= 1
        plt.close(fig)

    def test_raises_on_missing_param(self):
        post = _MockPosterior({"redshift": np.array([1.0, 2.0])})
        with pytest.raises(KeyError, match="dust_tau_diff"):
            plot_1d_posterior(post, "dust_tau_diff")


class TestPlotCalibration:
    def test_plots_chebyshev_band(self):
        rng = np.random.default_rng(0)
        n_draws = 500
        # Free coefficients: cal_c1, cal_c2, cal_c3
        # (c_0 = 1 is fixed and not a free parameter)
        samples = {
            "cal_c1": 0.05 + 0.01 * rng.standard_normal(n_draws),
            "cal_c2": -0.02 + 0.01 * rng.standard_normal(n_draws),
            "cal_c3": 0.01 + 0.005 * rng.standard_normal(n_draws),
        }
        post = _MockPosterior(samples)
        wave = np.linspace(4000.0, 9000.0, 400)
        fig, ax = plt.subplots()
        plot_calibration(post, ax=ax, wave_aa=wave)
        # Should produce a filled band + median line + reference axhline.
        assert len(ax.collections) >= 1  # band
        plt.close(fig)

    def test_raises_when_no_calibration_samples(self):
        post = _MockPosterior({"redshift": np.array([1.0, 2.0])})
        with pytest.raises(KeyError, match="Chebyshev"):
            plot_calibration(post, wave_aa=np.linspace(4000.0, 9000.0, 100))
