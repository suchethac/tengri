"""Style constants, color palettes, and matplotlib configuration for tengri plotting.

Reusable color schemes (colorblind-safe, print-friendly) and style setup for
publication-quality astronomy figures. Inspired by BAGPIPES (Carnall+2018).
"""

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Color palette — colorblind-safe, print-friendly
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    # Sampler colors (consistent across all notebooks)
    "map": "#888888",  # grey — point estimate
    "rt": "#1f77b4",  # blue — Ray Tracing (exact MCMC)
    # Canonical names
    "vi": "#ff7f0e",  # orange — geoVI (variational)
    "vi_linear": "#9467bd",  # purple — MGVI (linear VI)
    "mcmc_nuts": "#2ca02c",  # green — NUTS (gold standard)
    "mcmc_raytrace": "#1f77b4",  # blue — Ray Tracing (exact MCMC)
    # Legacy names (deprecated but still supported)
    "geovi": "#ff7f0e",  # orange — geoVI (variational)
    "nuts": "#2ca02c",  # green — NUTS (gold standard)
    "mgvi": "#9467bd",  # purple — MGVI (linear VI)
    # Data colors
    "truth": "#1a1a1a",  # near-black — ground truth
    "data": "#333333",  # dark grey — observed data
    "model": "#d62728",  # red — model prediction
    # SFH components
    "sfh_mean": "#1f77b4",  # blue — mean SFH backbone
    "sfh_full": "#ff7f0e",  # orange — full SFH (mean + GP)
    "sfh_gp": "#2ca02c",  # green — GP contribution
    # Band colors (SDSS)
    "u": "#7b3294",
    "g": "#008837",
    "r": "#d73027",
    "i": "#fc8d59",
    "z": "#4575b4",
    # Sequential for progressive reveal
    "seq": ["#d4d4d4", "#a8a8a8", "#1f77b4", "#2ca02c", "#d62728"],
}

# Named sampler styles for consistent legends
SAMPLER_STYLE = {
    "MAP": {"color": COLORS["map"], "ls": "--", "lw": 1.5, "alpha": 1.0},
    "RT": {"color": COLORS["rt"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    # Canonical names
    "VI": {"color": COLORS["vi"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "VI_Linear": {"color": COLORS["vi_linear"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "MCMC_NUTS": {"color": COLORS["mcmc_nuts"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    # Legacy names (deprecated but still supported)
    "geoVI": {"color": COLORS["geovi"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "NUTS": {"color": COLORS["nuts"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "MGVI": {"color": COLORS["mgvi"], "ls": "-", "lw": 1.5, "alpha": 1.0},
}

# SDSS effective wavelengths (Angstrom)
SDSS_BANDS = {"u": 3551, "g": 4686, "r": 6166, "i": 7480, "z": 8932}
SDSS_WAVE_EFF = np.array([3551, 4686, 6166, 7480, 8932])
SDSS_BAND_NAMES = ["u", "g", "r", "i", "z"]
SDSS_BAND_COLORS = [COLORS["u"], COLORS["g"], COLORS["r"], COLORS["i"], COLORS["z"]]


# ═══════════════════════════════════════════════════════════════════
# Style setup
# ═══════════════════════════════════════════════════════════════════


def setup_style():
    """Configure matplotlib for publication-quality astronomy figures.

    Follows BAGPIPES (Carnall+2018) styling closely:
    - Large axis labels (18pt) and tick labels (14pt)
    - Thick lines (2pt data, 1.5pt axes)
    - Inward ticks on all four sides
    - No frame on legends
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Font — BAGPIPES uses Helvetica/sans-serif; we use DejaVu Serif
            # for journal compatibility without requiring LaTeX installation
            "font.size": 14,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            # Axes labels — BAGPIPES: 18pt labels, 14pt ticks
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            # Axes frame — BAGPIPES: 1.5pt
            "axes.linewidth": 1.5,
            "axes.grid": False,
            # Ticks — BAGPIPES: inward, all four sides
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            # Legend — BAGPIPES: no frame
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            # Lines — BAGPIPES: 2pt
            "lines.linewidth": 2.0,
        }
    )


# ═══════════════════════════════════════════════════════════════════
# Common spectral features for annotation
# ═══════════════════════════════════════════════════════════════════

SPECTRAL_FEATURES = {
    r"Ly$\alpha$": 1216.0,
    "D4000": 4000.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4340.0,
    r"H$\beta$": 4861.0,
    "[O III]": 5007.0,
    "Mg b": 5175.0,
    "Na D": 5893.0,
    r"H$\alpha$": 6563.0,
}


# ═══════════════════════════════════════════════════════════════════
# Burstiness plane (for NB01)
# ═══════════════════════════════════════════════════════════════════

# Galaxy type annotations for the sigma-tau grid
GALAXY_ANNOTATIONS = {
    (0, 0): "Dead elliptical",
    (0, 2): "Secular disk",
    (1, 1): "Normal SF galaxy",
    (2, 0): "Extreme dwarf",
    (2, 2): "Post-starburst",
}


# ═══════════════════════════════════════════════════════════════════
# Visual language specification — import from here for consistency
# ═══════════════════════════════════════════════════════════════════

SED_XLIM = (912, 1e7)  # Å, rest-frame
SED_XSCALE = "log"
SED_YLABEL = r"$\lambda F_\lambda$ (normalized at 5500 Å)"
SED_XLABEL = r"Rest-frame wavelength (Å)"

SFH_XLABEL = "Lookback time (Gyr)"
SFH_YLABEL = r"SFR (M$_\odot$ yr$^{-1}$)"

SWEEP_CMAPS = {
    "dust": "YlOrRd",  # yellow→red for reddening
    "agn": "PuRd",  # purple→red for AGN dominance
    "sfh": "Blues",  # light→dark for SFH variation
    "nebular": "Greens",  # for ionization
    "radio": "cool",  # blue→purple for radio
    "redshift": "plasma",  # for redshift sweeps
}

REFERENCE_STYLE = dict(color="0.75", lw=1.5, zorder=0, label="reference")
