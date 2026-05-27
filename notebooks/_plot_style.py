"""Notebook-specific plotting setup and convenience shim.

This module re-exports the public plotting API from tengri.analysis.plotting
and provides notebook-specific matplotlib configuration overrides.

The primary API lives in :mod:`tengri.analysis.plotting` — this file is just
a convenience shim for notebook workflows that keeps the namespace familiar.

Usage in notebooks:
    from _plot_style import setup_style, plot_sfh, COLORS
    setup_style()  # call once at start of notebook
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

from tengri.analysis.plotting import (
    COLORS,
    CONVERGENCE_THRESHOLDS,
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
    add_sfh_inset,
    compare_filter_sets,
    convergence_check,
    convergence_table,
    diagnostics_table,
    mock_plot,
    parameter_gallery,
    plot_autocorrelation,
    plot_corner_comparison,
    plot_filter_coverage,
    plot_filter_curves,
    plot_sed_fit,
    plot_sfh,
    plot_sfh_comparison,
    plot_spectrum_fit,
    posterior_plot_sed,
    posterior_plot_sfh,
    safe_corner,
    setup_style as _setup_style_base,
    sfh_sed_comparison,
    sweep_parameter,
)

# Re-export all symbols
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


def setup_style():
    """Configure matplotlib for publication-quality astronomy figures.

    This is a convenience wrapper around :func:`tengri.analysis.plotting.setup_style`
    with optional notebook-specific overrides.

    Call once at the start of a notebook::

        from _plot_style import setup_style
        setup_style()

    See Also
    --------
    tengri.analysis.plotting.setup_style : The underlying function from the package.
    """
    import matplotlib.pyplot as plt

    # Apply base style from the package
    _setup_style_base()

    # Optional: notebook-specific overrides (e.g., larger DPI for screen previews)
    # For now, just use package defaults
