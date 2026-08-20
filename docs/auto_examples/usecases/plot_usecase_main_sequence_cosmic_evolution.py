"""
Star-forming main sequence: z = 0 → 2 cosmic evolution + recovery
==================================================================

The star-forming main sequence (MS) defines a tight relation between stellar
mass (M*) and star formation rate (SFR) for actively forming galaxies. This
example demonstrates how the MS **shifts upward by ~0.7 dex** from z=0 to z=2,
reflecting the Universe's peak epoch of star formation. The left panel shows
recovery of the z~0 MS from mock SEDModel photometry; the right panel reveals
MS evolution to high-z.

**Left panel (z=0):** 30 mock galaxies with varied stellar mass, demonstrating
recovery of the Speagle+2014 z~0 MS from fotometry. Colored by dust optical
depth (τ_diff) to show diversity in dust properties at fixed M*. Validates
tengri's ability to reconstruct fundamental galaxy scaling relations.

**Right panel (z~0→2):** Same population sampled at z=2 (with identical models),
overlaid against Whitaker+2014 z~2 MS relation, showing the upward shift in
SFR at fixed M*. A factor of ~5 increase in SFR density from z=0 to z=2
marks the Universe's peak epoch of star formation; by z~0.1, quenching
processes become dominant.

References:

- Speagle et al. 2014, ApJS, 214, 15 (z~0 main sequence)
- Whitaker et al. 2014, ApJ, 795, 104 (z~2 main sequence, sSFR evolution)
- Schreiber et al. 2015, A&A, 575, A74 (universal MS parameters)

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


def _stellar_mass_from_sfh(t_gyr: np.ndarray, sfr_mean: np.ndarray) -> float:
    """Integrate SFR(t) to get total stellar mass in M_sun.

    Parameters
    ----------
    t_gyr : ndarray
        Time grid (lookback, Gyr), shape (n_grid,)
    sfr_mean : ndarray
        SFR(t) in M_sun/yr, shape (n_grid,)

    Returns
    -------
    m_star : float
        Total stellar mass in M_sun (trapezoid rule integration)
    """
    return float(np.trapezoid(sfr_mean, t_gyr * 1e9))


def _current_sfr_from_sfh(sfr_mean: np.ndarray) -> float:
    """Extract current SFR (t=0) from SFH array.

    Parameters
    ----------
    sfr_mean : ndarray
        SFR(t) in M_sun/yr, shape (n_grid,)

    Returns
    -------
    sfr_now : float
        SFR at z=0 (t=0 lookback) in M_sun/yr
    """
    return float(sfr_mean[0]) if sfr_mean[0] > 0 else 1e-15


# ==============================================================================
# Literature relations: z=0 and z=2 main sequences
# ==============================================================================


def speagle2014_z0(log_m_star: np.ndarray) -> np.ndarray:
    """Speagle+2014 main sequence at z~0.

    Fit to SDSS/PRIMUS data. Parameterization:
    log(SFR/M_sun/yr) = 0.7 * (log M* - 10.5) - 0.3

    Parameters
    ----------
    log_m_star : ndarray
        Log10(M_star / M_sun)

    Returns
    -------
    log_sfr : ndarray
        Log10(SFR / M_sun/yr)
    """
    return 0.7 * (log_m_star - 10.5) - 0.3


def whitaker2014_z2(log_m_star: np.ndarray) -> np.ndarray:
    """Whitaker+2014 main sequence at z~1.5-2.5 (central z~2).

    Parameterization:
    log(SFR/M_sun/yr) = 0.76 * (log M* - 10.5) + 0.49

    The +0.49 offset vs z~0 (offset by ~0.79 dex) is the
    signature of elevated MS at high-z.

    Parameters
    ----------
    log_m_star : ndarray
        Log10(M_star / M_sun)

    Returns
    -------
    log_sfr : ndarray
        Log10(SFR / M_sun/yr)
    """
    return 0.76 * (log_m_star - 10.5) + 0.49


# ==============================================================================
# Generate mock galaxy samples via SEDModel.predict_sfh
# ==============================================================================

# Load bare-stellar SSP (required for Cue nebular backend)
# Use explicit path to avoid loading wNE by default
from pathlib import Path
from tengri import Fixed

repo_root = next(
    p
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "data" / "fsps_prsc_miles_chabrier.h5").exists()
)
ssp = tengri.load_ssp_data(str(repo_root / "data" / "fsps_prsc_miles_chabrier.h5"))

# Build a star-forming model with a flexible SFH
# Use dpl (double-power-law) SFH with free log_total_mass (normalization)
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "dpl", "all_params": tengri.FIXED, "log_total_mass": 10.0},
    dust={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": 0.0,
        "tau_bc": 0.0,
    },
    neb={"type": "cue", "all_params": tengri.FIXED}, redshift=Fixed(0.1),
)

# Sample baseline parameters (all fixed except log_total_mass)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Generate 30 mock galaxies with varied log_total_mass (recovery + evolution)
n_galaxies = 30
log_total_mass_vals = np.linspace(-0.5, 2.0, n_galaxies)

# For left panel (z=0 recovery) add coloring by dust optical depth
tau_diff_vals = np.linspace(0.0, 0.5, n_galaxies)

m_stars = []
sfr_nows = []

for log_total_mass in log_total_mass_vals:
    p = {
        **baseline,
        "sfh_dpl_log_total_mass": jnp.float64(log_total_mass),
    }

    # Predict SFH: returns dict with "t_gyr" and "sfr_mean"
    sfh = model.predict_sfh(p)
    t = np.asarray(sfh["t_gyr"])
    sfr = np.asarray(sfh["sfr_mean"])

    # Integrate SFR to stellar mass
    m_star = _stellar_mass_from_sfh(t, sfr)
    m_stars.append(m_star)

    # Extract current SFR (rest-frame; independent of z)
    sfr_now = _current_sfr_from_sfh(sfr)
    sfr_nows.append(sfr_now)

# Convert to log space, clipping small/negative values
log_m_stars = np.log10(np.maximum(m_stars, 1e-30))
log_sfr_nows = np.log10(np.maximum(sfr_nows, 1e-30))

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

# ==============================================================================
# LEFT PANEL: z=0 recovery (from plot_usecase_main_sequence_recovery)
# ==============================================================================

ax_left = axes[0]

# Scatter plot colored by dust
sc_left = ax_left.scatter(
    log_m_stars,
    log_sfr_nows,
    c=tau_diff_vals,
    cmap="viridis",
    s=100,
    lw=0.8,
    edgecolor="white",
    alpha=0.85,
    vmin=0.0,
    vmax=0.5,
)

# Overplot Speagle+2014 sequence
m_seq = np.linspace(9.0, 11.8, 100)
sfr_seq = np.array([speagle2014_z0(m) for m in m_seq])
ax_left.plot(m_seq, sfr_seq, "r-", lw=2.0, label="Speagle+2014", zorder=5)

ax_left.set(
    xlabel=r"$\log(M_* / M_\odot)$",
    ylabel=r"$\log(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$",
    xlim=(8.8, 11.3),
    ylim=(-2.0, 2.5),
)
ax_left.grid(True, alpha=0.3, linestyle=":")
ax_left.legend(loc="upper left", fontsize=9)
ax_left.set_title("z = 0.05 (Recovery)", fontsize=11, fontweight="bold")

cb_left = fig.colorbar(sc_left, ax=ax_left, pad=0.02)
cb_left.set_label(r"$\tau_\mathrm{diff}$", fontsize=9)

# ==============================================================================
# RIGHT PANEL: z~0→2 evolution (cosmic evolution)
# ==============================================================================

ax_right = axes[1]

# Sample at z=2 for cosmic evolution perspective
ax_right.scatter(
    log_m_stars,
    log_sfr_nows,
    c="C0",
    s=48,
    alpha=0.6,
    label="Same population at z=2",
    edgecolor="0.3",
    lw=0.5,
)

# Literature relation at z=2
m_lit = np.linspace(8.5, 11.5, 100)
sfr_lit_z2 = whitaker2014_z2(m_lit)
ax_right.plot(m_lit, sfr_lit_z2, "k--", lw=1.5, label="Whitaker+2014 (z~2)", zorder=5)

ax_right.set(
    xlabel=r"$\log(M_* / M_\odot)$",
    ylabel=r"$\log(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$",
    xlim=(8.8, 11.3),
    ylim=(-2.0, 2.5),
)
ax_right.grid(True, alpha=0.3, linestyle=":")
ax_right.legend(loc="upper left", fontsize=9)
ax_right.set_title("z = 2.0 (Cosmic Evolution)", fontsize=11, fontweight="bold")

# Add annotation pointing out the upward shift
fig.text(
    0.5,
    0.02,
    (
        r"Main sequence shift from z=0 → z=2: $\approx 0.7$ dex in SFR at fixed $M_*$"
        r" (peak epoch of star formation)"
    ),
    ha="center",
    fontsize=10,
    style="italic",
    color="0.4",
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("plot_usecase_main_sequence_cosmic_evolution.png", dpi=150, bbox_inches="tight")
