"""
Alpha-element enhancement suppresses iron absorption features
=============================================================

The [α/Fe] abundance ratio encodes the chemical enrichment history: rapid
enrichment by core-collapse supernovae before Type Ia SNe begin leads to high
[α/Fe]. In the SED, enhanced alpha-elements suppress iron absorption lines in
the optical (especially around 4000–5000 Å) because the higher abundance of
alpha elements shifts the line-blanketing opacity. We sweep [α/Fe] on a quiescent
passively evolving galaxy where iron features dominate the continuum absorption.

Reference: Thomas et al. 2003, MNRAS, 339, 897 (alpha-element effects).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 8.0,
        "log_total_mass": 10.0,
    },
    dust={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.0,
        "tau_diff": 0.1,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

met_alpha_fe_values = np.linspace(-0.2, 0.6, 7)
norm = mpl.colors.Normalize(vmin=met_alpha_fe_values.min(), vmax=met_alpha_fe_values.max())
cmap = plt.get_cmap("magma")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for met_alpha_fe in met_alpha_fe_values:
    params = {**baseline, "met_alpha_fe": jnp.float64(met_alpha_fe)}
    out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=cmap(norm(met_alpha_fe)), lw=1.4)

ax.set_xlim(3000, 1e4)
# Tighten y-limits to optical/optical-IR region with small margin
ax.set_ylim(1e40, 2e44)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

for wl in [4000, 5200, 8662]:
    ax.axvline(wl, color="gray", ls=":", lw=0.6, alpha=0.3)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"[$\alpha$/Fe]")

fig.tight_layout()
plt.savefig("plot_alpha_fe_sweep.png", dpi=150, bbox_inches="tight")
