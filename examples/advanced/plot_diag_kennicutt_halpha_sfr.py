"""
Hα-to-SFR calibration against Kennicutt (1998)
===============================================

Diagnostic: the Hα luminosity traces the ionizing photon rate from young stars,
which correlates with the instantaneous SFR. Kennicutt (1998, ApJ 498 541, Eq. 2)
calibrated this relationship for Salpeter IMF; for Chabrier IMF (used by tengri),
the coefficient is 4.97e-42: SFR / (Msun/yr) = 4.97e-42 × L(Hα) / (erg/s).

This script builds a young, dust-free model with constant SFR over the last ~10 Myr,
varies the SFR value across a grid, and compares the implied Hα→SFR coefficient
to the canonical Kennicutt+Chabrier value. A few-percent agreement validates that
tengri's Cue nebular emulator correctly maps ionizing photon rates to Hα luminosity.
"""

import os
import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Kennicutt (1998) + Chabrier IMF correction
KENNICUTT_CHABRIER_COEFF = 4.97e-42  # SFR / L_Ha [Msun/yr / erg/s]

# Load bare-stellar SSP (Cue requirement)
model = tengri.SEDModel.build(
    tengri.load_ssp("fsps_prsc_miles_chabrier"),
    sfh={"type": "dpl", "*": tengri.FIXED, "alpha": 5.0, "beta": 2.0,
         "tau_gyr": 1.0, "log_peak_sfr": tengri.FREE},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "neb_logU": -2.5,
         "neb_fesc": 0.0, "neb_fesc_lya": 0.0},
    redshift=tengri.Fixed(0.0),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Sweep log_peak_sfr to vary the instantaneous SFR at recent epochs
log_peak_vals = np.linspace(-1.0, 2.0, 8)  # peak SFR ~ 0.1 to 100 Msun/yr
sfr_10myr_vals = []
halpha_lum_vals = []
implied_coeff_vals = []

for log_peak in log_peak_vals:
    params = {**baseline, "sfh_dpl_log_peak_sfr": jnp.float64(log_peak)}

    # Get the actual 10 Myr SFR (what Hα traces)
    sfh_q = model.predict_sfh_quantities(params)
    sfr_10myr = float(sfh_q.sfr_10myr)

    # Get Hα luminosity from nebular component
    lines = model.predict_emission_lines(params)
    halpha_lum = float(lines.halpha)

    # Implied coefficient: SFR / L_Ha
    if halpha_lum > 0:
        implied_coeff = sfr_10myr / halpha_lum
        implied_coeff_vals.append(implied_coeff)
    else:
        implied_coeff_vals.append(np.nan)

    sfr_10myr_vals.append(sfr_10myr)
    halpha_lum_vals.append(halpha_lum)

sfr_10myr_vals = np.array(sfr_10myr_vals)
halpha_lum_vals = np.array(halpha_lum_vals)
implied_coeff_vals = np.array(implied_coeff_vals)

# Plot: implied coefficient vs SFR
fig, ax = plt.subplots(figsize=(6.5, 4.2))
mask = ~np.isnan(implied_coeff_vals)
ax.semilogy(sfr_10myr_vals[mask], implied_coeff_vals[mask], "o-", lw=1.5, ms=6, color="C0")
ax.axhline(KENNICUTT_CHABRIER_COEFF, color="red", linestyle="--", lw=2.0,
           label=r"Kennicutt+Chabrier: $4.97 \times 10^{-42}$")
ax.set_xlabel(r"SFR$_{10\mathrm{Myr}}$ [M$_{\odot}$ yr$^{-1}$]")
ax.set_ylabel(r"Implied coeff: SFR / $L_{\mathrm{H}\alpha}$ [M$_{\odot}$ yr$^{-1}$ erg$^{-1}$ s]")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
outpath = os.path.join(os.path.dirname(__file__), "plot_diag_kennicutt_halpha_sfr.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
