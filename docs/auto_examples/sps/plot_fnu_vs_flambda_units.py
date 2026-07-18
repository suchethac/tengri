"""
SED Conventions: F_λ vs F_ν vs νF_ν
===================================

The same stellar population SED looks different depending on the units
chosen for visualization. a single galaxy SED in three
complementary representations on a 3-panel grid:

1. **Left panel (F_λ vs λ):** Flux per unit wavelength. Peak appears
   at shorter wavelengths due to density of states weighting.

2. **Middle panel (F_ν vs ν):** Flux per unit frequency. Peak position
   differs from F_λ because the Jacobian of the wavelength–frequency
   transformation (dλ/dν ∝ λ²) redistributes power.

3. **Right panel (νF_ν vs λ):** Luminosity-weighted representation
   (rest-frame λ on x-axis). Peak position differs again; this is the
   conventional choice for SEDs because it equalizes the visual weight
   of each photon.

**Key insight:** The peak position of an SED is *not* a physical property
— it depends entirely on the choice of units. Always cite your
convention when presenting results.

**Galaxy model:** Truncated skew-normal SFH with peak lookback time = 2 Gyr,
width = 0.5 Gyr. Default DSPS SSP (no metallicity variation or dust).
z = 0.05 (rest-frame SED only; no cosmological redshift applied).

References:
- Hogg, D. W., Blanton, M. R., et al. 2002, AJ, 123, 1147 (K-corrections
  and SED unit conventions)
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
C_AA_PER_S = 2.998e18  # speed of light in Angstrom/s

# Build a simple galaxy: truncated skew-normal SFH, peak at 2 Gyr lookback
model = tengri.SEDModel.build(
    tengri.load_ssp(),
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": 2.0,
        "width_gyr": 0.5,
    },
    redshift=tengri.Fixed(0.05),
)

# Sample from the model (all parameters are fixed, so deterministic)
params = dict(model.spec.sample(jax.random.PRNGKey(0)))
pred = model.predict(params)

# Extract rest-frame wavelength [Angstrom] and SED [erg/s/Hz]
lambda_rest = np.asarray(model.wavelengths)
sed_fnu = np.asarray(pred.rest_sed())

# Compute the three representations
# 1. F_λ = F_ν * (c/λ²) [normalized for comparison]
f_lambda = sed_fnu * (C_AA_PER_S / lambda_rest**2)

# 2. F_ν as-is
f_nu = sed_fnu

# 3. νF_ν (luminosity-weighted; often called "lambda F lambda" in wavelength space)
nu = C_AA_PER_S / lambda_rest
nu_f_nu = nu * sed_fnu

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

# Panel 1: F_λ vs λ
ax = axes[0]
ax.loglog(lambda_rest, f_lambda, "C0-", lw=1.5)
peak_idx_flambda = np.argmax(f_lambda)
ax.plot(
    lambda_rest[peak_idx_flambda],
    f_lambda[peak_idx_flambda],
    "o",
    color="C0",
    markersize=5,
    markeredgewidth=0,
)
ax.set_xlabel(r"Wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]")
ax.grid(True, alpha=0.25, which="both")

# Panel 2: F_ν vs ν
ax = axes[1]
nu_rest = C_AA_PER_S / lambda_rest
ax.loglog(nu_rest, f_nu, "C1-", lw=1.5)
peak_idx_fnu = np.argmax(f_nu)
ax.plot(
    nu_rest[peak_idx_fnu],
    f_nu[peak_idx_fnu],
    "o",
    color="C1",
    markersize=5,
    markeredgewidth=0,
)
ax.set_xlabel(r"Frequency $\nu$ [Hz]")
ax.set_ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.grid(True, alpha=0.25, which="both")

# Panel 3: νF_ν vs λ (luminosity-weighted in rest-frame wavelength space)
ax = axes[2]
ax.loglog(lambda_rest, nu_f_nu, "C2-", lw=1.5)
peak_idx_nufnu = np.argmax(nu_f_nu)
ax.plot(
    lambda_rest[peak_idx_nufnu],
    nu_f_nu[peak_idx_nufnu],
    "o",
    color="C2",
    markersize=5,
    markeredgewidth=0,
)
ax.set_xlabel(r"Wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu F_\nu$ [erg s$^{-1}$ cm$^{-2}$]")
ax.grid(True, alpha=0.25, which="both")

fig.tight_layout()
plt.savefig("plot_fnu_vs_flambda_units.png", dpi=150, bbox_inches="tight")
