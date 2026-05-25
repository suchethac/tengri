"""
SKIRTOR torus: viewing angle tunes IR profile shape
====================================================

The torus inclination angle determines how much cold dust emission we
observe. Face-on (high ``cos_inc``) views show a smooth thermal bump;
edge-on (low ``cos_inc``) views expose more reprocessed mid-infrared
flux and can show silicate absorption features.
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
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_total_mass": 10.0,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED, "tau_skirtor": 7.0},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "log_lbol": 11.0,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

cos_inc_values = np.array([0.95, 0.75, 0.5, 0.25, 0.05])
norm = mpl.colors.Normalize(vmin=cos_inc_values.min(), vmax=cos_inc_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for cos_inc in cos_inc_values:
    params = {**baseline, "agn_cos_inc": jnp.float64(cos_inc)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(cos_inc)), lw=1.4)

ax.set_xlim(100, 1e6)
ax.set_ylim(1e40, 1e45)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\cos \theta_{\rm torus}$")

fig.tight_layout()
plt.savefig("plot_agn_cos_inc_sweep.png", dpi=150, bbox_inches="tight")
