"""
AB Magnitude Zero-point Consistency Check
==========================================

Validates that AB magnitude zero-point definitions are consistent across filters.
Compares photometry converted to magnitude via the formula m_AB = -2.5 log10(F_ν)
- 48.6 against tengri's built-in magnitude conversion. The AB magnitude system
requires this relationship to hold across all filters—any deviation signals a
zero-point calibration issue.

Reference: Fukugita et al. 1996, AJ, 111, 1748 (AB magnitude system).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# SSP for baseline model
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Filter set: UV → NIR coverage
band_names = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
]

obs = tengri.Observation(photometry=tengri.Photometry.from_names(band_names))

# Build a simple model (fixed dust, fixed SFH)
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": tengri.FIXED},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.3, "tau_bc": 0.2},
    redshift=tengri.Fixed(0.05),
)

# Fiducial parameters
params = {
    "sfh_tsnorm_log_peak_sfr": 0.5,
    "sfh_tsnorm_peak_lbt_gyr": 3.5,
    "sfh_tsnorm_width_gyr": 1.5,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 5.0,
    "dust_slope": -0.7,
    "redshift": 0.05,
}

# Get photometry (observer-frame flux densities in erg/s/cm²/Hz)
phot_fluxes = np.asarray(model.predict_photometry(params))

# Convert to magnitudes two ways:
# Method 1: Manual conversion using standard formula
mag_manual = -2.5 * np.log10(np.maximum(phot_fluxes, 1e-30)) - 48.6

# Method 2: Use tengri's built-in converter
mag_tengri = np.asarray(model.predict_magnitudes(params))

# Residuals: manual − tengri (should be near zero if zero-point is correct)
residuals = mag_manual - mag_tengri

# Effective wavelengths for plotting (center of filter bandpass in obs frame)
wave_eff = np.array([np.mean(np.asarray(f.wave)) for f in obs.photometry.filters])

# --- Plotting ---
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(10, 4), sharey=False, gridspec_kw={"width_ratios": [1, 1]}
)

# LEFT: AB magnitudes vs filter wavelength
ax_left.scatter(wave_eff / 1e4, mag_tengri, s=80, alpha=0.7, color="C0")
ax_left.set_xscale("log")
ax_left.set_xlabel(r"$\lambda_{\rm eff}$ [$\mu$m]")
ax_left.set_ylabel(r"$m_{\rm AB}$ [mag]")
ax_left.grid(True, alpha=0.3)

# RIGHT: Residuals (manual − tengri in millimagnitudes)
colors = ["C0" if np.abs(r) < 0.01 else "C3" for r in residuals]
ax_right.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
ax_right.scatter(wave_eff / 1e4, residuals * 1000, s=80, alpha=0.7, color=colors)
ax_right.set_xscale("log")
ax_right.set_xlabel(r"$\lambda_{\rm eff}$ [$\mu$m]")
ax_right.set_ylabel(r"$\Delta m$ [mmag] (manual − tengri)")
ax_right.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_diag_ab_mag_zero_point.png", dpi=150, bbox_inches="tight")
