"""
Early-time SFH slope α shapes the UV continuum
==============================================

The rising slope ``α`` of a double-power-law star formation history controls
how rapidly the galaxy assembled its mass before the peak. Larger α means a
more abrupt onset of star formation, leaving a younger O/B-star population
at the time of observation and a steeper rest-frame UV slope. We vary α
across the prior range with every other parameter fixed.

Reference: Behroozi et al. 2013, ApJ, 770, 57 (functional form, Eq. 1).
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
        "alpha": tengri.Uniform(0.3, 6.0),
        "beta": 2.5,
        "tau_gyr": 1.5,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

alpha_values = np.linspace(0.5, 6.0, 7)
norm = mpl.colors.Normalize(vmin=alpha_values.min(), vmax=alpha_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for alpha in alpha_values:
    params = {**baseline, "sfh_dpl_alpha": jnp.float64(alpha)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(alpha)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"DPL rising slope $\alpha$")

fig.tight_layout()
plt.savefig("plot_dpl_alpha_sweep.png", dpi=150, bbox_inches="tight")
