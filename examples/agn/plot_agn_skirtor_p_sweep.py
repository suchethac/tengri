"""
SKIRTOR torus: radial density profile tunes IR emission peak
=============================================================

The SKIRTOR clumpy torus has a radial dust-density profile with power-law
index ``p``. Steeper profiles (higher p) concentrate more dust closer to the
disc, reducing the mid-IR peak temperature and shifting flux toward the far-IR.
Flatter profiles distribute dust more uniformly and hotter on average.
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_peak_sfr": 0.5,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED, "p": tengri.Uniform(0, 1.5)},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "log_lbol": 11.0,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

p_values = np.array([0.0, 0.5, 1.0, 1.5])
norm = mpl.colors.Normalize(vmin=p_values.min(), vmax=p_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for p in p_values:
    params = {**baseline, "agn_p_skirtor": jnp.float64(p)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(p)), lw=1.4)

ax.set_xlim(100, 1e6)
ax.set_ylim(1e40, 1e45)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Radial power index $p$")

fig.tight_layout()
plt.savefig("plot_agn_skirtor_p_sweep.png", dpi=150, bbox_inches="tight")
