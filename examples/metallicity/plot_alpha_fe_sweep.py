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

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 8.0,
        "log_peak_sfr": 0.5,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
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
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(met_alpha_fe)), lw=1.4)

ax.set_xlim(3000, 1e4)
ax.set_ylim(1e40, 1e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

# Mark iron-dominated regions
for wl, label in [(4000, "4000 Å break"), (5200, r"Mg $b$"), (8662, r"Ca II triplet")]:
    ax.axvline(wl, color="grey", ls=":", lw=0.6, alpha=0.3)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"[$\alpha$/Fe]")

fig.tight_layout()
fig.savefig("plot_alpha_fe_sweep.png", dpi=150, bbox_inches="tight")
