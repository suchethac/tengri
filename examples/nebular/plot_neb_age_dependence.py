"""
Nebular emission fades with stellar population age
====================================================

Ionizing photon production declines rapidly with stellar population age
(~t^-1). We show how nebular line strength evolves from young (50 Myr)
to old (5 Gyr) populations.
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

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": tengri.Uniform(0.05, 5.0),
        "log_peak_sfr": 1.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    neb={"type": "cue", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

age_values = np.array([0.05, 0.1, 0.3, 0.7, 1.5, 3.0, 5.0])
norm = mpl.colors.LogNorm(vmin=age_values.min(), vmax=age_values.max())
cmap = plt.get_cmap("plasma")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for age in age_values:
    params = {**baseline, "sfh_dpl_tau_gyr": jnp.float64(age)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.semilogy(wave, nu_l_nu, color=cmap(norm(age)), lw=1.4)

ax.set_xlim(4000, 7500)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Peak age [Gyr]")

fig.tight_layout()
fig.savefig("plot_neb_age_dependence.png", dpi=150, bbox_inches="tight")
