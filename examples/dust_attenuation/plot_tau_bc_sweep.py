"""
Birth cloud dust suppresses young-stellar UV and nebular emission
==================================================================

Birth-cloud dust optical depth ``τ_bc`` attenuates only the youngest stellar
light (age < ~10 Myr), controlling nebular emission from embedded HII regions.
τ_bc effects are clearest on young star-forming populations; we use a 500 Myr
starburst and vary τ_bc across the prior range.

Reference: Charlot & Fall 2000, ApJ, 539, 718 (two-component dust model).
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
        "tau_gyr": 0.5,
        "log_peak_sfr": 1.0,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_bc": tengri.Uniform(0.0, 4.0),
        "tau_diff": 0.1,
    },
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

tau_bc_values = np.linspace(0.0, 4.0, 7)
norm = mpl.colors.Normalize(vmin=tau_bc_values.min(), vmax=tau_bc_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for tau_bc in tau_bc_values:
    params = {**baseline, "dust_tau_bc": jnp.float64(tau_bc)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(tau_bc)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Birth cloud optical depth $\tau_{\rm bc}$")

fig.tight_layout()
plt.savefig("plot_tau_bc_sweep.png", dpi=150, bbox_inches="tight")
