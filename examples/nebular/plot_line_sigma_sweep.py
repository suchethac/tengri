"""
Emission line broadening traces gas kinematics
================================================

Emission line velocity dispersion broadens lines from a few km/s (narrow,
kinematically resolved) to hundreds of km/s (unresolved at typical
spectroscopic resolution). We show the [OIII] region broadened across
the dynamical range.
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

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": 0.3,
        "log_peak_sfr": 1.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "neb_eline_sigma": tengri.Uniform(50, 800)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

sigma_values = np.array([50, 100, 200, 400, 800])
norm = mpl.colors.LogNorm(vmin=sigma_values.min(), vmax=sigma_values.max())
cmap = plt.get_cmap("Blues")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for sigma in sigma_values:
    params = {**baseline, "neb_eline_sigma": jnp.float64(sigma)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.semilogy(wave, nu_l_nu, color=cmap(norm(sigma)), lw=1.4)

ax.set_xlim(4700, 5200)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\sigma$ [km/s]")

fig.tight_layout()
plt.savefig("plot_line_sigma_sweep.png", dpi=150, bbox_inches="tight")
