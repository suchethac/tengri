# SPDX-License-Identifier: BSD-3-Clause
"""Plotting utilities.

Organized by plot type into reusable modules: styles, SFH, SED, corner plots,
and convergence diagnostics. Each module is focused and ~100-400 lines.

Usage::

    from tengri.analysis.plotting import plot_sfh, plot_sed_fit, safe_corner
    from tengri.analysis.plotting import COLORS, SDSS_WAVE_EFF, setup_style
"""

# ═══════════════════════════════════════════════════════════════════
# Style constants and setup
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════
from tengri.analysis.plotting.convergence import (
    CONVERGENCE_THRESHOLDS,
    convergence_check,
    convergence_table,
    diagnostics_table,
    plot_autocorrelation,
    posterior_plot_sfh,
)

# ═══════════════════════════════════════════════════════════════════
# Corner plot utilities
# ═══════════════════════════════════════════════════════════════════
from tengri.analysis.plotting.corner import plot_corner_comparison, safe_corner

# ═══════════════════════════════════════════════════════════════════
# Filter curve visualization
# ═══════════════════════════════════════════════════════════════════
from tengri.analysis.plotting.filters import (
    compare_filter_sets,
    plot_filter_coverage,
    plot_filter_curves,
)

# ═══════════════════════════════════════════════════════════════════
# SED and spectrum plotting
# ═══════════════════════════════════════════════════════════════════
from tengri.analysis.plotting.sed import (
    mock_plot,
    parameter_gallery,
    plot_sed_fit,
    plot_spectrum_fit,
    posterior_plot_sed,
    sfh_sed_comparison,
    sweep_parameter,
)

# ═══════════════════════════════════════════════════════════════════
# SFH plotting
# ═══════════════════════════════════════════════════════════════════
from tengri.analysis.plotting.sfh import add_sfh_inset, plot_sfh, plot_sfh_comparison
from tengri.analysis.plotting.styles import (
    COLORS,
    GALAXY_ANNOTATIONS,
    REFERENCE_STYLE,
    SAMPLER_STYLE,
    SDSS_BAND_COLORS,
    SDSS_BAND_NAMES,
    SDSS_BANDS,
    SDSS_WAVE_EFF,
    SED_XLABEL,
    SED_XLIM,
    SED_XSCALE,
    SED_YLABEL,
    SFH_XLABEL,
    SFH_YLABEL,
    SPECTRAL_FEATURES,
    SWEEP_CMAPS,
    setup_style,
)

__all__ = [
    "COLORS",
    "CONVERGENCE_THRESHOLDS",
    "GALAXY_ANNOTATIONS",
    "REFERENCE_STYLE",
    "SAMPLER_STYLE",
    "SDSS_BANDS",
    "SDSS_BAND_COLORS",
    "SDSS_BAND_NAMES",
    "SDSS_WAVE_EFF",
    "SED_XLABEL",
    "SED_XLIM",
    "SED_XSCALE",
    "SED_YLABEL",
    "SFH_XLABEL",
    "SFH_YLABEL",
    "SPECTRAL_FEATURES",
    "SWEEP_CMAPS",
    "add_sfh_inset",
    "compare_filter_sets",
    "convergence_check",
    "convergence_table",
    "diagnostics_table",
    "mock_plot",
    "parameter_gallery",
    "plot_autocorrelation",
    "plot_corner_comparison",
    "plot_filter_coverage",
    "plot_filter_curves",
    "plot_sed_fit",
    "plot_sfh",
    "plot_sfh_comparison",
    "plot_spectrum_fit",
    "posterior_plot_sed",
    "posterior_plot_sfh",
    "safe_corner",
    "setup_style",
    "sfh_sed_comparison",
    "sweep_parameter",
]
