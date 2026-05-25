"""
Post-peak quenching slope β shapes stellar age distribution
===========================================================

The falling slope β of a double-power-law SFH controls quenching after the peak.
Large β means rapid quenching and an old stellar population; small β means
a gentle tail and more mixed ages. We vary β across its prior range with every
other parameter fixed.
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

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "beta": tengri.Uniform(0.3, 10.0),
        "alpha": 1.5,
        "tau_gyr": 3.0,
        "log_peak_sfr": 1.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

beta_values = np.linspace(0.3, 10.0, 7)
norm = mpl.colors.Normalize(vmin=beta_values.min(), vmax=beta_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for beta in beta_values:
    params = {**baseline, "sfh_dpl_beta": jnp.float64(beta)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(beta)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"DPL falling slope $\beta$")

fig.tight_layout()
plt.savefig("plot_dpl_beta_sweep.png", dpi=150, bbox_inches="tight")
