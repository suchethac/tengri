"""
Gas-phase metallicity effect on nebular continuum
==================================================

Nebular free-free, free-bound, and two-photon emission respond to gas-phase
metallicity (``logZ_gas``) through changes in metal cooling efficiency and
ionization balance. metallicity sensitivity of
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
        # NB: short-form keys inside the `neb` group (full `neb_*` is silently ignored)
        "logU": -2.0,
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

# Compute SEDs at each metallicity AND a reference baseline (lowest Z) so we
# can visualise the *nebular* response — the stellar continuum dominates by
# factors of 10–100 at these wavelengths, so plotting νLν directly hides
# everything. We show the fractional residual against the lowest-Z model.
seds = []
for logz in logz_values:
    params = {**baseline, "neb_logZ_gas": jnp.float64(logz)}
    pred = model.predict_rest_sed(params)
    seds.append(np.asarray(pred.sed))
wave = np.asarray(pred.wavelength)
ref = seds[0]  # logz = -1.5 baseline

mask = (wave >= 1000) & (wave <= 1e4)

fig, ax = plt.subplots(figsize=(8.0, 5.5))
for sed, color, label in zip(seds, colors, labels):
    delta = (sed - ref) / ref
    ax.semilogx(
        wave[mask],
        100.0 * delta[mask],
        color=color,
        lw=2.0,
        label=label,
        alpha=0.9,
    )

ax.axhline(0.0, color="0.4", lw=0.6, ls=":")
ax.set_xlim(1000, 1e4)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]", fontsize=12)
ax.set_ylabel(
    r"$\Delta L_\nu / L_\nu(Z_{\rm gas}=10^{-1.5}Z_\odot)$  [%]", fontsize=12
)
ax.set_title(
    "Nebular continuum sensitivity to gas-phase metallicity (residual vs. lowest-Z)",
    fontsize=11,
)
ax.legend(fontsize=10, frameon=True, loc="best")
ax.grid(True, alpha=0.25, which="both")

fig.tight_layout()
plt.savefig("plot_gas_z_continuum_effect.png", dpi=150, bbox_inches="tight")
