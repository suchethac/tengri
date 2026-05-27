"""
Cosmological Distance Validation: tengri vs Astropy
====================================================

Compares tengri's Planck18 cosmology implementation (DSPS-backed, Ω_m = 0.315,
h = 0.674) against astropy.cosmology.Planck18 (which uses slightly different
parameter values) across z = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]. Validates
luminosity distance d_L(z), comoving distance d_C(z), age(z), and comoving
volume element consistency. Residuals should be stable across z and <1% due
to underlying parameter differences rather than numerical bugs. Tengri's
PLANCK18 parameters (Om0=0.315, h=0.674) match Planck 2018 published values.

Reference: Planck Collaboration 2018, A&A 641, A6.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Redshift grid for comparison
z_vals = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])

# Tengri cosmology (default: DSPS-backed Planck18)
d_L_tengri = np.array([tengri.cosmology.luminosity_distance_mpc(z) for z in z_vals])
d_C_tengri = np.array([tengri.cosmology.comoving_distance_mpc(z) for z in z_vals])
age_tengri = np.array([tengri.cosmology.age_at_z(z) for z in z_vals])
dV_tengri = np.array([tengri.cosmology.comoving_volume_element(z) for z in z_vals])

# Astropy Planck18 (reference implementation)
d_L_apy = np.array([Planck18.luminosity_distance(z).value for z in z_vals])
d_C_apy = np.array([Planck18.comoving_distance(z).value for z in z_vals])
age_apy = np.array([Planck18.age(z).value for z in z_vals])
dV_apy = np.array([Planck18.differential_comoving_volume(z).value for z in z_vals])

# Compute relative residuals: (tengri - astropy) / astropy
rel_err_dL = (d_L_tengri - d_L_apy) / np.abs(d_L_apy)
rel_err_dC = (d_C_tengri - d_C_apy) / np.abs(d_C_apy)
rel_err_age = (age_tengri - age_apy) / np.abs(age_apy)
rel_err_dV = (dV_tengri - dV_apy) / np.abs(dV_apy)

# Check for large deviations
max_err_dL = np.max(np.abs(rel_err_dL))
max_err_dC = np.max(np.abs(rel_err_dC))
max_err_age = np.max(np.abs(rel_err_age))
max_err_dV = np.max(np.abs(rel_err_dV))

print(f"Max relative error d_L: {max_err_dL:.2e} ({max_err_dL * 100:.3f}%)")
print(f"Max relative error d_C: {max_err_dC:.2e} ({max_err_dC * 100:.3f}%)")
print(f"Max relative error age: {max_err_age:.2e} ({max_err_age * 100:.3f}%)")
print(f"Max relative error dV/dz/dΩ: {max_err_dV:.2e} ({max_err_dV * 100:.3f}%)")

# Note: Residuals are driven by parameter differences (astropy's Planck18 differs
# slightly from Planck 2018 published values), not numerical bugs.
# Astropy: Om0≈0.310, H0≈67.7; Tengri (DSPS): Om0=0.315, H0=67.4

# --- Plotting ---
fig, ((ax_dl, ax_dc), (ax_age, ax_dv)) = plt.subplots(
    2, 2, figsize=(11, 8), gridspec_kw={"hspace": 0.35, "wspace": 0.3}
)

# Luminosity distance curves (should overlap)
ax_dl.loglog(z_vals, d_L_tengri, "o-", label="tengri (DSPS)", lw=1.5, ms=6)
ax_dl.loglog(z_vals, d_L_apy, "s--", label="astropy.Planck18", lw=1.5, ms=5, alpha=0.8)
ax_dl.set_xlabel(r"$z$")
ax_dl.set_ylabel(r"$d_L$ [Mpc]")
ax_dl.legend(frameon=False, fontsize=8)
ax_dl.grid(True, alpha=0.3, which="both")

# d_L relative residual
ax_dc.semilogx(z_vals, rel_err_dL * 1e6, "o-", color="C0", lw=1.5, ms=6)
ax_dc.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
ax_dc.set_xlabel(r"$z$")
ax_dc.set_ylabel(r"$\Delta d_L / d_L$ [ppm]")
ax_dc.grid(True, alpha=0.3)

# Comoving distance curves
ax_age.loglog(z_vals, d_C_tengri, "o-", label=r"$d_C$ (tengri)", lw=1.5, ms=6)
ax_age.loglog(z_vals, d_C_apy, "s--", label=r"$d_C$ (astropy)", lw=1.5, ms=5, alpha=0.8)
ax_age.set_xlabel(r"$z$")
ax_age.set_ylabel(r"$d_C$ [Mpc]")
ax_age.legend(frameon=False, fontsize=8)
ax_age.grid(True, alpha=0.3, which="both")

# Age and comoving volume residuals
ax_dv.semilogx(z_vals, rel_err_age * 1e6, "o-", label="age(z)", color="C1", lw=1.5, ms=6)
ax_dv.semilogx(
    z_vals, rel_err_dV * 1e6, "s-", label=r"$dV_c/dz/d\Omega$", color="C2", lw=1.5, ms=6
)
ax_dv.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
ax_dv.set_xlabel(r"$z$")
ax_dv.set_ylabel(r"Relative error [ppm]")
ax_dv.legend(frameon=False, fontsize=8)
ax_dv.grid(True, alpha=0.3)

fig.suptitle("Planck18 Consistency: DSPS (tengri) vs Astropy", fontsize=11, y=0.995)
plt.savefig("plot_diag_cosmology_vs_astropy.png", dpi=150, bbox_inches="tight")
