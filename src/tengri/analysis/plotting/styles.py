# SPDX-License-Identifier: BSD-3-Clause
"""Style constants, color palettes, and matplotlib configuration for tengri plotting.

Reusable color schemes (colorblind-safe, print-friendly) and style setup for
publication-quality astronomy figures. Inspired by BAGPIPES (Carnall+2018).
"""

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Color palette: colorblind-safe, print-friendly
# ═══════════════════════════════════════════════════════════════════

#: The shared plotting palette: colorblind-safe and print-friendly. Keys are
#: sampler names (``'map'``, ``'rt'``, ``'nuts'``, …) and photometric bands
#: (``'u'``, ``'g'``, ``'r'``, ``'i'``, ``'z'``); values are hex colors. Use
#: these rather than matplotlib's cycle so figures stay consistent across
#: notebooks, the gallery, and the paper.
COLORS = {
    # Sampler colors (consistent across all notebooks)
    "map": "#888888",  # gray, point estimate
    "rt": "#1f77b4",  # blue, Ray Tracing (exact MCMC)
    # Canonical names
    "vi": "#ff7f0e",  # orange, geoVI (variational)
    "vi_linear": "#9467bd",  # purple, MGVI (linear VI)
    "mcmc_nuts": "#2ca02c",  # green, NUTS (gold standard)
    "mcmc_raytrace": "#1f77b4",  # blue, Ray Tracing (exact MCMC)
    "geovi": "#ff7f0e",  # geoVI (old method name)
    "nuts": "#2ca02c",  # NUTS (old method name)
    "mgvi": "#9467bd",  # MGVI (old method name)
    # Data colors
    "truth": "#1a1a1a",  # near-black, ground truth
    "data": "#333333",  # dark gray; observed data
    "model": "#d62728",  # red, model prediction
    # SFH components
    "sfh_mean": "#1f77b4",  # blue, mean SFH backbone
    "sfh_full": "#ff7f0e",  # orange, full SFH (mean + GP)
    "sfh_gp": "#2ca02c",  # green, GP contribution
    # Band colors (SDSS)
    "u": "#7b3294",
    "g": "#008837",
    "r": "#d73027",
    "i": "#fc8d59",
    "z": "#4575b4",
    # Sequential for progressive reveal
    "seq": ["#d4d4d4", "#a8a8a8", "#1f77b4", "#2ca02c", "#d62728"],
}

#: Matplotlib line and color styles for consistent sampler legends across plots.
#: Keys are sampler names (``'MAP'``, ``'RT'``, ``'VI'``, ``'MCMC_NUTS'``, etc.);
#: values are dicts with ``'color'``, ``'ls'`` (line style), ``'lw'`` (line width),
#: and ``'alpha'`` (opacity). Use these for consistency across the gallery and papers.
SAMPLER_STYLE = {
    "MAP": {"color": COLORS["map"], "ls": "--", "lw": 2.0, "alpha": 1.0},
    "RT": {"color": COLORS["rt"], "ls": "-", "lw": 2.0, "alpha": 1.0},
    # Canonical names
    "VI": {"color": COLORS["vi"], "ls": "-", "lw": 2.0, "alpha": 1.0},
    "VI_Linear": {"color": COLORS["vi_linear"], "ls": "-", "lw": 2.0, "alpha": 1.0},
    "MCMC_NUTS": {"color": COLORS["mcmc_nuts"], "ls": "-", "lw": 2.0, "alpha": 1.0},
    "geoVI": {"color": COLORS["geovi"], "ls": "-", "lw": 2.0, "alpha": 1.0},
    "NUTS": {"color": COLORS["nuts"], "ls": "-", "lw": 2.0, "alpha": 1.0},
    "MGVI": {"color": COLORS["mgvi"], "ls": "-", "lw": 2.0, "alpha": 1.0},
}

#: SDSS effective wavelengths [Angstrom], keyed by band name.
SDSS_BANDS = {"u": 3551, "g": 4686, "r": 6166, "i": 7480, "z": 8932}

#: SDSS effective wavelengths [Angstrom] as an array in ``ugriz`` order: the
#: x-axis for a five-band photometry plot.
SDSS_WAVE_EFF = np.array([3551, 4686, 6166, 7480, 8932])
SDSS_BAND_NAMES = ["u", "g", "r", "i", "z"]
SDSS_BAND_COLORS = [COLORS["u"], COLORS["g"], COLORS["r"], COLORS["i"], COLORS["z"]]


# ═══════════════════════════════════════════════════════════════════
# Style setup
# ═══════════════════════════════════════════════════════════════════


def setup_style(style="tengri"):
    """Configure matplotlib for publication-quality astronomy figures.

    Defaults to a single cohesive style across every tengri notebook and example,
    built on top of the `scienceplots` ``science`` preset: the standard
    Nature/PRL/ApJ-compatible look (serif fonts, thin inward ticks on all
    four sides, frameless legends, no grid).

    Parameters
    ----------
    style : str
        Which composite style to apply. One of:

        - ``"tengri"`` (default); ``scienceplots.science`` + ``no-latex`` +
          tengri-specific tick and font-size overrides. Safe everywhere (no
          system LaTeX required).
        - ``"tengri-nature"``: Nature-journal variant (single-column width,
          sans-serif Arial). Use for slide decks.
        - ``"tengri-minimal"``: same tengri overrides but skip scienceplots
          so the style works when ``scienceplots`` is unavailable. Keeps the
          BAGPIPES-inspired look as a fallback.

    Returns
    -------
    None

    Examples
    --------
    >>> from tengri.analysis.plotting import setup_style
    >>> setup_style()  # default: science + tengri overrides
    >>> setup_style("tengri-minimal")  # fallback if scienceplots absent
    """
    # Composite-style block: layer scienceplots presets first, then tengri
    # overrides. Falls back gracefully if scienceplots is not installed.
    if style != "tengri-minimal":
        try:
            import scienceplots  # noqa: F401  (import registers styles)

            base_styles = ["science", "no-latex"]
            if style == "tengri-nature":
                base_styles.append("nature")
            plt.style.use(base_styles)
        except (ImportError, OSError):
            # scienceplots not installed or style name not found: fall through
            # to the minimal (BAGPIPES-inspired) overrides below.
            pass

    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Font: scienceplots uses Times; we keep DejaVu Serif so plots
            # render identically on systems without Times installed.
            "font.size": 13,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            # Axes labels and ticks: readable in publications and slides.
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            # Axes frame and grid
            "axes.linewidth": 1.0,
            "axes.grid": False,
            # Ticks: inward, all four sides, minor ticks visible
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            # Legend: no frame
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            # Lines: bumped to 2.0 for better visibility in galleries
            # and slide decks; per-call lw= overrides still win.
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "lines.markeredgewidth": 1.0,
            "patch.linewidth": 1.0,
            # Image colormap: perceptually-uniform default. Specific plots
            # can override (e.g. cividis, rocket) but the default must be
            # colorblind-safe and monotonic.
            "image.cmap": "viridis",
        }
    )


# ═══════════════════════════════════════════════════════════════════
# Common spectral features for annotation
# ═══════════════════════════════════════════════════════════════════

#: Rest-frame wavelengths [Angstrom] of the spectral features worth annotating
#: on an SED plot, keyed by a matplotlib-ready mathtext label. Vacuum
#: wavelengths, matching the rest of the package.
SPECTRAL_FEATURES = {
    r"Ly$\alpha$": 1216.0,
    "D4000": 4000.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4340.0,
    r"H$\beta$": 4861.0,
    "[O III]": 5007.0,
    "Mg b": 5175.0,
    "Na D": 5893.0,
    r"H$\alpha$": 6564.61,
}


# ═══════════════════════════════════════════════════════════════════
# Burstiness plane (for NB01)
# ═══════════════════════════════════════════════════════════════════

#: Galaxy type labels for the burstiness plane (sigma-tau grid).
#: Keys are tuples ``(sigma_bin, tau_bin)`` representing position on the grid;
#: values are descriptive galaxy type names. Use for annotating parameter space plots.
GALAXY_ANNOTATIONS = {
    (0, 0): "Dead elliptical",
    (0, 2): "Secular disk",
    (1, 1): "Normal SF galaxy",
    (2, 0): "Extreme dwarf",
    (2, 2): "Post-starburst",
}


# ═══════════════════════════════════════════════════════════════════
# Visual language specification: import from here for consistency
# ═══════════════════════════════════════════════════════════════════

SED_XLIM = (912, 1e7)  # Å, rest-frame
SED_XSCALE = "log"
SED_YLABEL = r"$\lambda F_\lambda$ (normalized at 5500 Å)"
SED_XLABEL = r"Rest-frame wavelength (Å)"

SFH_XLABEL = "Lookback time (Gyr)"
SFH_YLABEL = r"SFR (M$_\odot$ yr$^{-1}$)"

#: Matplotlib colormaps for parameter sweep plots, keyed by sweep parameter name.
#: Maps ``'dust'``, ``'agn'``, ``'sfh'``, ``'nebular'``, ``'radio'``, ``'redshift'``,
#: ``'metallicity'``, ``'stellar_age'`` to colormap names. Defaults are
#: perceptually-uniform and colorblind-safe; see ``SWEEP_VMIN`` and ``SWEEP_VMAX``
#: for value clamping.
SWEEP_CMAPS = {
    # Default to viridis everywhere: perceptually-uniform, colorblind-safe,
    # and the bright-yellow tail is suppressed by the SWEEP_VMAX clamp below
    # so curves stay readable on light backgrounds.
    "dust": "viridis",
    "agn": "viridis",
    "sfh": "viridis",
    "nebular": "viridis",
    "radio": "viridis",
    "redshift": "viridis",
    "metallicity": "cividis",  # colorblind-safe alternative for technical figures
    "stellar_age": "cividis",
}

# Most callers index a sweep colormap from 0 to 1; the upper end of viridis
# (>~0.85) is bright yellow which washes out on print. Helpers should clamp
# to this range when picking sweep colors.
SWEEP_VMIN = 0.0
SWEEP_VMAX = 0.85

#: Matplotlib style dict for reference curves in comparison plots.
#: Contains ``'color'`` (mid-gray), ``'lw'`` (line width), ``'zorder'`` (stacking),
#: and ``'label'`` for use with :func:`matplotlib.pyplot.plot` and legend.
REFERENCE_STYLE = dict(color="0.55", lw=2.0, zorder=0, label="reference")
