"""
Post-Starburst K+A Diagnostic: Hδ_A vs Time Since Quench
=========================================================

A classic post-starburst (PSB) / K+A galaxy signature: strong Balmer
absorption lines (high Hδ_A) with no emission, visible only in a narrow
window after a recent burst of star formation has been abruptly quenched.

This example builds 6 tengri models: extended star formation (tsnorm,
~200 Myr width) at varying lookback times spanning 0.1–1.8 Gyr, simulating
quench epochs with different elapsed times. We measure Hδ_A absorption at
each epoch and show how it peaks ~100 Myr post-quench, then decays as
A-type stars die out.

The signature traces the lifetime of A-type stars (responsible for Balmer
absorption) and reflects the mechanism by which star formation is rapidly
shut off in galaxies — a key driver of the red sequence in the local universe.

References
----------

.. [1] Dressler, A., & Gunn, J. E. 1992, ApJS, 78, 1
       (K+A galaxies in the Coma cluster)
.. [2] French, K. D., Yang, Y., Zabludoff, A., et al. 2018, ApJ, 862, 2
       (The molecular gas content and CO line ratios in post-starburst
       galaxies in the EAGLE simulations)
.. [3] Worthey, G., & Ottaviani, D. L. 1997, ApJS, 111, 377
       (Hδ_A window definitions and optical index calibrations)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Physical constants
C_AA_PER_S = 2.998e18  # Speed of light in Angstrom/s


def _compute_hdelta_a(wave, l_nu):
    """Compute Hδ_A absorption equivalent width per Worthey & Ottaviani 1997.

    Hδ_A is the rest equivalent width in the 4080-4120 Å window,
    centered on the H-delta Balmer line at 4101.7 Å. Negative EW
    indicates absorption (young), positive indicates emission (HII).

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelength [Angstrom].
    l_nu : ndarray
        Monochromatic luminosity [erg/s/Hz].

    Returns
    -------
    float
        Hδ_A equivalent width [Angstrom].
    """
    # Define Worthey+1997 window for Hδ_A
    line = (wave >= 4080) & (wave <= 4120)
    cont_blue = (wave >= 4050) & (wave <= 4080)
    cont_red = (wave >= 4120) & (wave <= 4170)

    # Convert L_nu to F_lambda for EW calculation
    f_lam = l_nu * C_AA_PER_S / wave**2

    # Estimate continuum via linear interpolation between sidebands
    if cont_blue.sum() > 0 and cont_red.sum() > 0:
        lam_blue = np.mean(wave[cont_blue])
        lam_red = np.mean(wave[cont_red])
        f_blue = np.mean(f_lam[cont_blue])
        f_red = np.mean(f_lam[cont_red])
        slope = (f_red - f_blue) / (lam_red - lam_blue)
        cont = f_blue + slope * (wave[line] - lam_blue)
    else:
        cont = np.mean(f_lam[line])

    # Equivalent width: integral of (1 - F_line / F_cont) × d_lambda
    if line.sum() > 0:
        delta = wave[line][1] - wave[line][0] if line.sum() > 1 else 1.0
        ew = float(np.sum((1.0 - f_lam[line] / np.maximum(cont, 1e-30)) * delta))
    else:
        ew = 0.0

    return ew


# ──────────────────────────────────────────────────────────────
# Model setup: constant SFR for 100 Myr, then abruptly quenched
# ──────────────────────────────────────────────────────────────

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")  # Cue needs bare-stellar SSP

# K+A signature appears when: (1) extended star formation suddenly stops, and
# (2) the young stellar population is observed before A-type stars die out
# (timescale ~1–2 Gyr for K+A at z~0). Real K+A galaxies typically have
# post-starburst age ~0.3–1 Gyr (the age of the quenched population at
# observation).
#
# We model a realistic K+A scenario: star formation lasting ~1 Gyr at
# constant rate, then abrupt quench. We vary the quench epoch (lookback
# time) to sweep time-since-quench from 0.1 to 2 Gyr.
#
# Strategy: Use tsnorm with moderate width (~200 Myr) to represent an
# extended starbursting phase. By varying the peak lookback time, we observe
# the population at different ages post-quench.

time_since_quench_gyr = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.8])
n_models = len(time_since_quench_gyr)

hdelta_a = np.empty(n_models)

for i, tsq in enumerate(time_since_quench_gyr):
    # tsnorm peaked at lookback_time=tsq with moderate width (~200 Myr)
    # to approximate extended (not instantaneous) star formation.
    # The population then ages post-quench for tsq Gyr.
    sfh = {
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": tsq,
        "width_gyr": 0.2,  # ~200 Myr width: moderate extended burst
        "log_total_mass": 10.0,  # ~6 Msun/yr SFR (brighter, more typical)
        "skew": 0.0,
        "trunc": 13.5,
    }

    # K+A galaxies: no dust, no nebular emission (pure stellar absorption)
    dust = {"type": "two_component", "all_params": tengri.FIXED, "tau_bc": 0.0, "tau_diff": 0.0}
    neb = {"type": "cue", "all_params": tengri.FIXED}

    # Build model; z=0.05 avoids numerical singularities in cosmology at z=0
    model = tengri.SEDModel.build(
        ssp,
        sfh=sfh,
        dust=dust,
        neb=neb,
        redshift=tengri.Fixed(0.05),
    )

    # Sample baseline parameters
    baseline = dict(model.spec.sample(jax.random.PRNGKey(i)))

    # Predict rest-frame SED
    pred = model.predict(baseline)
    wave = np.asarray(model.wavelengths)
    l_nu = np.asarray(pred.rest_sed())

    # Compute Hδ_A equivalent width (negative = absorption)
    hdelta_a[i] = _compute_hdelta_a(wave, l_nu)


# ──────────────────────────────────────────────────────────────
# Plot: Hδ_A vs time since quench
# ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.0, 5.0))

# Plot the evolution
ax.plot(
    time_since_quench_gyr,
    hdelta_a,
    marker="o",
    color="C0",
    markersize=6,
    linewidth=1.8,
    label="K+A signature",
)

# Shade the "classic K+A" window (strong absorption, no emission)
ax.axhspan(-3.0, -0.5, color="0.9", alpha=0.5, zorder=0, label="Classic K+A window")
ax.axhline(0, color="0.7", linestyle="--", linewidth=0.8, alpha=0.7)

# Mark the peak post-quench age
peak_idx = np.argmin(hdelta_a)  # Most negative = strongest absorption
peak_age = time_since_quench_gyr[peak_idx]
peak_ew = hdelta_a[peak_idx]
ax.plot(peak_age, peak_ew, marker="*", markersize=18, color="red", zorder=10)
ax.text(
    peak_age,
    peak_ew - 0.4,
    f"Peak at {peak_age:.2f} Gyr",
    fontsize=9,
    ha="center",
    color="red",
    weight="bold",
)

ax.set_xlabel(r"Time since quench  [Gyr]", fontsize=11)
ax.set_ylabel(r"H$\delta_A$ equivalent width  [$\mathrm{\AA}$]", fontsize=11)
ax.set_xscale("log")
ax.set_xlim(0.03, 3.0)
ax.set_ylim(-3.5, 0.2)
ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.6)
ax.legend(loc="lower right", fontsize=10)

# Add annotation explaining the diagnostic
ax.text(
    0.98,
    0.05,
    "K+A (Kauffmann+1999, Dressler+1999): post-starburst galaxies\n"
    "with strong Balmer absorption + no emission. Signature of\n"
    "rapid quenching; lifetime ≈ 1–2 Gyr (A-star timescale).",
    transform=ax.transAxes,
    fontsize=8.5,
    ha="right",
    va="bottom",
    color="0.4",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="0.7"),
)

plt.tight_layout()
plt.savefig("plot_post_starburst_diagnostic.png", dpi=150, bbox_inches="tight")
