# SPDX-License-Identifier: BSD-3-Clause
"""Plotting helpers for tengri results.

Re-exports the plotting functions and visual constants from
:mod:`tengri.analysis.plotting` under the shorter ``tengri.plot``
namespace, so users can ``from tengri import plot`` rather than reach
into :mod:`analysis`.

These helpers are *not* JIT-compatible (matplotlib has side effects).
They consume :class:`Posterior` / :class:`SEDResult` / numpy arrays.

Examples
--------
>>> from tengri import plot
>>> fig = plot.plot_sed_fit(wave_eff, flux_obs, noise)
>>> plot.setup_style()  # apply tengri rcParams
"""

from __future__ import annotations

from tengri.analysis.plotting import (
    COLORS,
    SDSS_WAVE_EFF,
    SPECTRAL_FEATURES,
    diagnostics_table,
    plot_1d_posterior,
    plot_calibration,
    plot_corner_comparison,
    plot_sed_fit,
    plot_sfh,
    plot_sfh_comparison,
    plot_spectrum_fit,
    safe_corner,
    setup_style,
)

__all__ = [
    "COLORS",
    "SDSS_WAVE_EFF",
    "SPECTRAL_FEATURES",
    "diagnostics_table",
    "plot_1d_posterior",
    "plot_calibration",
    "plot_corner_comparison",
    "plot_sed_fit",
    "plot_sfh",
    "plot_sfh_comparison",
    "plot_spectrum_fit",
    "safe_corner",
    "setup_style",
]
