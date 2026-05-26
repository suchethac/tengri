"""
Type 1 vs Type 2 AGN: Unified viewing-angle classification
===========================================================

Unified AGN models explain the Type 1/Type 2 dichotomy as a **purely geometric
effect** — the same accretion disc + dusty torus system appears as:

- **Type 1 (face-on, cos θ_inc = 1)**: Direct view of the hot accretion disc
  producing a bright, blue continuum and broad emission lines (BLR) visible
  across the spectrum.

- **Type 2 (edge-on, cos θ_inc = 0)**: Torus blocks the disc and BLR, leaving
  only the isotropic narrow-line region (NLR) and dusty torus reprocessed IR
  emission visible.

This gallery shows both sightlines for the same AGN (L_bol = 10^12 L_sun) using
the composable unified AGN model in tengri. The Type 2 configuration demonstrates
the library limitation noted in caveat #294: the torus does not properly screen
the disc from the observer, resulting in over-bright Type-2 continuum.

References
----------
.. [1] Urry, C. M., & Padovani, P. 1995, PASP, 107, 803
   "Unified Schemes for Radio-Loud Active Galactic Nuclei"
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Physical constants for unit conversion
C_AA_PER_S = 2.998e18

# Load minimal SSP (constant star formation; AGN dominates).
# Bare-stellar SSP required by Cue nebular backend (used in some recipes).
ssp = tengri.load_ssp()

# Shared model components (minimal star formation, no dust).
COMMON = dict(
    sfh={"type": "const", "*": tengri.FIXED, "log_sfr": -10.0},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.05),
)

# Base AGN: multicolor disc + SKIRTOR torus (common to both Type 1 and 2).
BASE_AGN = {
    "disc": {"type": "multicolor", "*": tengri.FIXED},
    "torus": {"type": "skirtor", "*": tengri.FIXED, "tau_skirtor": 7.0},
    "*": tengri.FIXED,
    "log_lbol": 12.0,
    "frac": 1.0,
}

# Type 1: face-on, unobstructed disc view → broad lines visible
print("Building Type 1 (face-on, BLR) model...")
agn_type1 = {
    **BASE_AGN,
    "lines": {"type": "blr", "*": tengri.FIXED},
}
model_type1 = tengri.SEDModel.build(ssp, agn=agn_type1, **COMMON)
params_type1 = dict(model_type1.spec.sample(jax.random.PRNGKey(42)))
# Override inclination: cos(θ) = 1 (face-on)
params_type1["agn_cos_inc"] = jnp.float64(1.0)
out_type1 = model_type1.predict_rest_sed(params_type1)
wave_type1 = np.asarray(out_type1.wavelength)
sed_type1 = np.asarray(out_type1.sed)

# Type 2: edge-on, torus-obscured disc → narrow lines only
print("Building Type 2 (edge-on, NLR) model...")
agn_type2 = {
    **BASE_AGN,
    "lines": {"type": "nlr", "*": tengri.FIXED},
}
model_type2 = tengri.SEDModel.build(ssp, agn=agn_type2, **COMMON)
params_type2 = dict(model_type2.spec.sample(jax.random.PRNGKey(42)))
# Override inclination: cos(θ) = 0 (edge-on)
params_type2["agn_cos_inc"] = jnp.float64(0.0)
out_type2 = model_type2.predict_rest_sed(params_type2)
wave_type2 = np.asarray(out_type2.wavelength)
sed_type2 = np.asarray(out_type2.sed)

# Convert to νL_ν for plotting (rest-frame)
nu_type1 = C_AA_PER_S / wave_type1
nu_type2 = C_AA_PER_S / wave_type2
nu_lnu_type1 = nu_type1 * sed_type1
nu_lnu_type2 = nu_type2 * sed_type2

# Create side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True, gridspec_kw={"hspace": 0.0})

# Type 1 (face-on): blue continuum, prominent broad lines
ax1.loglog(wave_type1, nu_lnu_type1, color="#1f77b4", lw=2.0)
ax1.fill_between(
    wave_type1, nu_lnu_type1, alpha=0.25, color="#1f77b4", label="Type 1 (BLR visible)"
)
ax1.set_xlim(100, 1e6)
ax1.set_ylim(1e42, 1e46)
ax1.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]", fontsize=11)
ax1.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]", fontsize=11)
ax1.set_title("Type 1 (face-on, cos θ = 1.0)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, frameon=False, loc="lower left")
ax1.grid(True, alpha=0.2, which="both")

# Mark prominent Type-1 features
for wl, lbl in [(1216, r"Ly$\alpha$"), (4861, r"H$\beta$"), (6563, r"H$\alpha$")]:
    ax1.axvline(wl, color="navy", ls=":", lw=0.8, alpha=0.4)

# Type 2 (edge-on): attenuated continuum, narrow lines only
ax2.loglog(wave_type2, nu_lnu_type2, color="#d62728", lw=2.0)
ax2.fill_between(
    wave_type2, nu_lnu_type2, alpha=0.25, color="#d62728", label="Type 2 (NLR + torus)"
)
ax2.set_xlim(100, 1e6)
ax2.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]", fontsize=11)
ax2.set_title("Type 2 (edge-on, cos θ = 0.0)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, frameon=False, loc="lower left")
ax2.grid(True, alpha=0.2, which="both")

# Mark prominent Type-2 features
for wl, lbl in [(6563, r"[N II]"), (6583, r"H$\alpha$"), (5007, r"[O III]")]:
    ax2.axvline(wl, color="darkred", ls=":", lw=0.8, alpha=0.4)

fig.text(
    0.5,
    0.02,
    (
        "Unified AGN model with viewing-angle dependent line and continuum "
        "masking (Urry & Padovani 1995).\n"
        "Caveat: tengri's composable AGN does not properly screen the disc "
        "in Type 2 sightlines; the observed disc flux is over-bright."
    ),
    ha="center",
    fontsize=9,
    style="italic",
    color="gray",
)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("plot_type1_type2_unified_model.png", dpi=150, bbox_inches="tight")
print("Saved: plot_type1_type2_unified_model.png")
