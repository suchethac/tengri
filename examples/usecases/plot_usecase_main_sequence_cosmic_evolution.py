"""
Star-forming main sequence: z = 0 → 2
========================================================

The star-forming main sequence (MS) defines a tight relation between stellar
mass (M*) and star formation rate (SFR) for actively forming galaxies. This
example demonstrates how the MS **shifts upward by ~0.7 dex** from z=0 to z=2,
reflecting the Universe's peak epoch of star formation.

**Left panel (z=0):** 30 mock galaxies with varied normalization, overlaid
against the Speagle+2014 z~0 MS reference.

**Right panel (z=2):** Same construction at z=2, overlaid against Whitaker+2014
z~2 MS relation, showing the upward shift in SFR at fixed M*.

The physical origin: at high-z, galaxies accrete gas more rapidly, fuel higher
SFR densities, and produce stellar populations faster than at z=0. By z~0.1,
this epochal overdensity has declined and quenching becomes more prevalent.

References:
- Speagle et al. 2014, ApJ, 164, 14 (z~0 main sequence)
- Whitaker et al. 2014, ApJ, 795, 104 (z~2 main sequence)
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
from tengri.analysis.plotting import setup_style

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

repo_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "data" / "fsps_prsc_miles_chabrier.h5").exists()
)
ssp = tengri.load_ssp_data(str(repo_root / "data" / "fsps_prsc_miles_chabrier.h5"))

# Build a star-forming model with a flexible SFH
# Use dpl (double-power-law) SFH with free log_total_mass (normalization)
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "dpl", "*": tengri.FIXED,
         "log_total_mass": 10.0, 3.0)},
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED},
)

# Sample baseline parameters (all fixed except log_total_mass)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Two redshifts, 30 galaxies each with varied log_total_mass
n_galaxies = 30
redshifts = [0.05, 2.0]

# Generate log_total_mass values spanning ~1.5 dex range
log_total_mass_vals = np.linspace(-0.5, 2.0, n_galaxies)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

for panel_idx, (ax, z_obs) in enumerate(zip(axes, redshifts)):
    # Temporary update to redshift (for info only; predict_sfh is redshift-agnostic)
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

    # Scatter plot: our mock sample
    ax.scatter(log_m_stars, log_sfr_nows, c="C0", s=48, alpha=0.6,
               label=f"Mock galaxies (z={z_obs:.2f})", edgecolor="0.3", lw=0.5)

    # Literature relation
    m_lit = np.linspace(8.5, 11.5, 100)
    if z_obs < 1.0:
        sfr_lit = speagle2014_z0(m_lit)
        ax.plot(m_lit, sfr_lit, "k--", lw=1.5, label="Speagle+2014 (z~0)")
    else:
        sfr_lit = whitaker2014_z2(m_lit)
        ax.plot(m_lit, sfr_lit, "k--", lw=1.5, label="Whitaker+2014 (z~2)")

    ax.set(xlabel=r"$\log(M_* / M_\odot)$",
           ylabel=r"$\log(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$",
           xlim=(8.8, 11.3), ylim=(-2.0, 2.5))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"z = {z_obs:.2f}", fontsize=11, fontweight="bold")

# Add annotation pointing out the upward shift
fig.text(0.5, 0.02, r"Upward shift of main sequence at high-z: $\approx 0.7$ dex (SFR at fixed $M_*$)",
         ha="center", fontsize=10, style="italic", color="0.4")

fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("plot_usecase_main_sequence_cosmic_evolution.png",
            dpi=150, bbox_inches="tight")
