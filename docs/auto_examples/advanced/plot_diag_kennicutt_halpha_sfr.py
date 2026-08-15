"""
Hα-to-SFR calibration against Kennicutt (1998)
===============================================

Diagnostic: the Hα luminosity traces the ionizing photon rate from young stars,
which correlates with the instantaneous SFR. Kennicutt (1998, ApJ 498 541, Eq. 2)
calibrated this relationship for Salpeter IMF; for Chabrier IMF (used by tengri),
the coefficient is 4.97e-42: SFR / (M☉/yr) = 4.97e-42 × L(Hα) / (erg/s).

This script builds a young, dust-free model with constant SFR over the last ~10 Myr,
varies the SFR value across a grid, and compares the implied Hα→SFR coefficient
to the canonical Kennicutt+Chabrier value. A few-percent agreement validates that
tengri's Cue nebular emulator correctly maps ionizing photon rates to Hα luminosity.
"""

import warnings

import jax
import jax.numpy as jnp
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
    sfh={
        "type": "const",
        "all_params": tengri.FIXED,
        "start_gyr": 0.01,  # constant SFR over the last 10 Myr (what Ha traces)
        "end_gyr": 0.0,
        "log_total_mass": tengri.FREE,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={
        "type": "cue",
        "all_params": tengri.FIXED,
        "neb_logU": -2.5,
        "neb_fesc": 0.0,
        "neb_fesc_lya": 0.0,
    },
    redshift=tengri.Fixed(0.0),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Sweep the constant SFR over [0.1, 100] Msun/yr. Over a 10 Myr window the total
# mass formed is SFR x 1e7 yr, so log_total_mass = log_sfr + 7.
log_sfr_vals = np.linspace(-1.0, 2.0, 8)  # SFR ~ 0.1 to 100 Msun/yr
log_total_mass_vals = log_sfr_vals + 7.0
sfr_10myr_vals = []
halpha_lum_vals = []
implied_coeff_vals = []

for log_total_mass in log_total_mass_vals:
    params = {**baseline, "sfh_const_log_total_mass": jnp.float64(log_total_mass)}

    # Get the actual 10 Myr SFR (what Hα traces)
    sfh_q = model.predict_properties(params, names=("sfr_10myr",))
    sfr_10myr = float(sfh_q["sfr_10myr"])

    # Get Hα luminosity from nebular component
    halpha_lum = float(model.predict(params).halpha)

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
ax.axhline(
    KENNICUTT_CHABRIER_COEFF,
    color="red",
    linestyle="--",
    lw=2.0,
    label=r"Kennicutt+Chabrier: $4.97 \times 10^{-42}$",
)
ax.set_xlabel(r"SFR$_{10\mathrm{Myr}}$ [M$_{\odot}$ yr$^{-1}$]")
ax.set_ylabel(r"Implied coeff: SFR / $L_{\mathrm{H}\alpha}$ [M$_{\odot}$ yr$^{-1}$ erg$^{-1}$ s]")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_diag_kennicutt_halpha_sfr.png", dpi=150, bbox_inches="tight")
