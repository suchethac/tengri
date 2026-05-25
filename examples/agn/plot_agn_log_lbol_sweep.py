"""
QSOgen disc: bolometric luminosity controls overall flux
=========================================================

The disc continuum normalisation tracks bolometric luminosity directly;
the disc temperature shifts more subtly with the implied accretion rate.
Varying ``agn_log_lbol`` from 10 to 14 (in log10 L_sun) sweeps four orders of
magnitude in disc luminosity, comparable to typical Seyfert through bright-QSO
regimes. The spectral shape (slope, peak position) remains nearly fixed.
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
        "log_peak_sfr": 0.5,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

log_lbol_values = np.linspace(10.0, 14.0, 7)  # log10(L_bol / L_sun)
norm = mpl.colors.Normalize(vmin=log_lbol_values.min(), vmax=log_lbol_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for log_lbol in log_lbol_values:
    params = {**baseline, "agn_log_lbol": jnp.float64(log_lbol)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(log_lbol)), lw=1.4)

ax.set_xlim(100, 1e6)
ax.set_ylim(1e41, 1e48)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log L_{\mathrm{bol}}/L_\odot$")

fig.tight_layout()
plt.savefig("plot_agn_log_lbol_sweep.png", dpi=150, bbox_inches="tight")
