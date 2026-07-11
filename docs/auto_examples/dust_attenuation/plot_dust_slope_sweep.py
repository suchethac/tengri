"""
Dust attenuation curve slope controls UV vs optical hardness
==============================================================

The power-law slope ``δ`` steepens (negative) or flattens (positive) UV
attenuation relative to the optical, controlling whether dust absorbs more
or less light at short wavelengths. We vary δ with elevated τ_bc and τ_diff
to make slope effects visible (low dust opacities wash out the continuum slope).

Reference: Conroy et al. 2009, ApJ, 699, 626 (power-law attenuation model).
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
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 1.5,
        "log_total_mass": 10.0,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_bc": 1.0,
        "tau_diff": 0.5,
        "slope": tengri.Uniform(-1.5, 0.5),
    },
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

slope_values = np.array([-1.5, -0.9, -0.3, 0.3])
norm = mpl.colors.Normalize(vmin=slope_values.min(), vmax=slope_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for slope in slope_values:
    params = {**baseline, "dust_slope": jnp.float64(slope)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(slope)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Attenuation curve slope $\delta$")

fig.tight_layout()
plt.savefig("plot_dust_slope_sweep.png", dpi=150, bbox_inches="tight")
