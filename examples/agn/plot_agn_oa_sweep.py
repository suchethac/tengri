"""
SKIRTOR torus: opening angle controls exposed disc fraction
============================================================

The torus opening angle (``oa_skirtor``) sets how much of the central
disc is visible. A narrower torus (smaller opening angle) hides the disc
and relies on reprocessed torus emission; a more open torus exposes the
hot disc continuum and shifts the SED blueward.
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
        "torus": {"type": "skirtor", "*": tengri.FIXED, "oa": tengri.Uniform(20, 60)},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "log_lbol": 11.0,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

oa_values = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
norm = mpl.colors.Normalize(vmin=oa_values.min(), vmax=oa_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for oa in oa_values:
    params = {**baseline, "agn_oa_skirtor": jnp.float64(oa)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(oa)), lw=1.4)

ax.set_xlim(100, 1e6)
ax.set_ylim(1e40, 1e45)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Opening angle $\theta_{\rm oa}$ [°]")

fig.tight_layout()
plt.savefig("plot_agn_oa_sweep.png", dpi=150, bbox_inches="tight")
