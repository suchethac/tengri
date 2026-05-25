"""
Delayed-exponential timescale τ controls decay after peak SFR
=============================================================

The timescale τ of a delayed-exponential SFH sets how quickly star formation
falls after its peak: short τ means rapid decline and old stars, long τ means
a sustained tail and younger mean age. We vary τ across the prior range with
every other parameter fixed.
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
        "type": "dexp",
        "*": tengri.FIXED,
        "tau_gyr": tengri.Uniform(0.1, 10.0),
        "log_peak_sfr": 1.0,
        "start_gyr": 10.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

tau_values = np.linspace(0.5, 10.0, 7)
norm = mpl.colors.Normalize(vmin=tau_values.min(), vmax=tau_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for tau in tau_values:
    params = {**baseline, "sfh_dexp_tau_gyr": jnp.float64(tau)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(tau)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Timescale $\tau$ [Gyr]")

fig.tight_layout()
plt.savefig("plot_dexp_tau_sweep.png", dpi=150, bbox_inches="tight")
