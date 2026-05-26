"""
Gas-phase metallicity effect on nebular continuum
==================================================

Nebular free-free, free-bound, and two-photon emission respond to gas-phase
metallicity (``logZ_gas``) through changes in metal cooling efficiency and
ionization balance. This example demonstrates the metallicity sensitivity of
the nebular continuum at fixed ionization parameter.

Single rest-frame νLν trace (1000–10000 Å) across four gas metallicities
(logZ_gas = −1.5, −0.5, 0.0, +0.3), all at fixed ``logU = -2``. The
metallicity effect is strongest in the UV continuum (Lyα → HeII Balmer), where
metal opacity and recombination cooling compete.

References:
- Osterbrock & Ferland 2006, *Astrophysics of Gaseous Nebulae and Active
  Galactic Nuclei*, 2nd ed., University Science Books
- Li, Leja & Speagle 2023, ApJ, 956, 23 (Cue nebular model)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load bare-stellar SSP (Cue requirement)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Build model with Cue nebular, fixed SFH + dust, varying logZ_gas
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.2,
        "beta": 2.0,
        "tau_gyr": 0.5,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.05, "tau_bc": 0.1},
    neb={
        "type": "cue",
        "*": tengri.FIXED,
        "logU": -2.0,  # Fixed ionization parameter
        "logZ_gas": tengri.Uniform(-2.0, 0.5),
    },
    redshift=tengri.Fixed(0.0),
)

# Baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Metallicity sweep
logz_values = np.array([-1.5, -0.5, 0.0, 0.3])
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
labels = [f"$Z_{{\\rm gas}}/Z_\\odot = 10^{{{z:.1f}}}$" for z in logz_values]

fig, ax = plt.subplots(figsize=(8.0, 5.5))

for logz, color, label in zip(logz_values, colors, labels):
    params = {**baseline, "neb_logZ_gas": jnp.float64(logz)}
    pred = model.predict_rest_sed(params)

    wave = np.asarray(pred.wavelength)
    sed = np.asarray(pred.sed)

    # Convert L_nu [erg/s/Hz] to nu*L_nu [erg/s]
    nu = 2.998e18 / wave  # Hz
    nu_l_nu = nu * sed

    # Plot only 1000-10000 Å
    mask = (wave >= 1000) & (wave <= 1e4)
    ax.loglog(
        wave[mask],
        nu_l_nu[mask],
        color=color,
        lw=2.0,
        label=label,
        alpha=0.85,
    )

ax.set_xlim(900, 1.1e4)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]", fontsize=12)
ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]", fontsize=12)
ax.legend(fontsize=10, frameon=True, loc="lower left")
ax.grid(True, alpha=0.25, which="both")

fig.tight_layout()
plt.savefig("plot_gas_z_continuum_effect.png", dpi=150, bbox_inches="tight")
