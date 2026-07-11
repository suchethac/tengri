"""
Dust attenuation laws across the galaxy zoo
=============================================

Public dust attenuation laws applied to the *same* intrinsic SED at the
*same* V-band optical depth (τ_V = 1.0), illustrating how dust geometry
and grain-size composition vary across the local universe.

Overlays all production-status attenuation laws, highlighting the diversity:
Milky Way extinction curves with 2175 Å bump (Cardelli, Draine, Hensley–Draine),
starburst laws with flattened UV slopes (Calzetti, Salim), SMC steep extinction
without bump, and flexible empirical models (Kriek & Conroy, Noll). Key features:
SMC shows steepest UV attenuation due to small grain size; Cardelli/Draine laws
retain the graphite bump; Calzetti/Salim flatten the UV; Kriek & Conroy and Noll
span both regimes via adjustable parameters.
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

laws_table = tengri.list_dust_laws()
LAWS = [(entry["name"], entry["short_doc"]) for entry in laws_table]

# Fixed SFH and stellar population
SFH = {
    "type": "tsnorm",
    "*": tengri.FIXED,
    "peak_lbt_gyr": 2.0,
    "width_gyr": 1.0,
    "log_total_mass": 10.0,
    "skew": 0.0,
    "trunc": 13.0,
}

# Load default SSP (bare stellar, compatible with all nebular backends)
ssp = tengri.load_ssp()

fig, ax = plt.subplots(figsize=(8.0, 5.2))

# Intrinsic (unreddened) SED as reference
ref_model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust={
        "type": "two_component",
        "law_diff": "calzetti",
        "*": tengri.FIXED,
        "tau_diff": 0.0,
        "tau_bc": 0.0,
    },
    redshift=tengri.Fixed(0.05),
)
p_ref = dict(ref_model.spec.sample(jax.random.PRNGKey(0)))
sed_ref = np.asarray(ref_model.predict_rest_sed(p_ref).sed)
wave = np.asarray(ref_model.predict_rest_sed(p_ref).wavelength)
C_AA_PER_S = 2.998e18
nu = C_AA_PER_S / wave
ax.loglog(wave, nu * sed_ref, color="0.05", lw=2.0, label="intrinsic", zorder=10, ls="--")

cmap = plt.get_cmap("tab20c")
n_laws = len(LAWS)
colors = [cmap(i / max(n_laws - 1, 1)) for i in range(n_laws)]

# Plot reddened SED for each law at fixed tau_V = 1.0
for (law, label), color in zip(LAWS, colors):
    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust={
            "type": "two_component",
            "law_diff": law,
            "*": tengri.FIXED,
            "tau_diff": 1.0,
            "tau_bc": 0.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    sed = np.asarray(model.predict_rest_sed(p).sed)
    ax.loglog(wave, nu * sed, color=color, lw=1.4, label=label.split("(")[0].strip(), alpha=0.8)

ax.axvline(2175, color="0.55", lw=0.5, ls=":", alpha=0.7)
ax.axvline(5500, color="0.7", lw=0.4, ls=":", alpha=0.5)

ax.set(
    xlim=(900, 3e4),
    ylim=(1e40, 8e43),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)

fig.tight_layout()
plt.savefig("plot_galactic_zoo_dust_laws.png", dpi=150, bbox_inches="tight")
