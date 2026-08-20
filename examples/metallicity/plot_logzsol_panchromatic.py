"""
Metallicity shapes panchromatic SED with dust emission
======================================================

Stellar metallicity affects the stellar continuum shape and overall energy
balance. Dust emission responds to absorbed stellar photons: metal-poor hot
stars emit bluer light with less IR-absorbed energy, while metal-rich cooler
stars are less bright in the UV but more absorbed in the optical/NIR. We sweep
stellar metallicity on a young star-forming galaxy at z = 0.2 with dust
attenuation and thermal emission from warm dust.

Reference: Conroy 2013 (stellar), Silva et al. 1998 (dust emission).
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
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 1.0,
        "log_total_mass": 10.0,
    },
    dust={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 1.0,
        "tau_diff": 0.5,
        "emission": {
            "type": "modified_blackbody",
            "all_params": tengri.FIXED,
            "T": 30.0,
            "beta_ir": 1.8,
        },
    },
    redshift=tengri.Fixed(0.2),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

met_logzsol_values = np.array([-1.5, -0.7, 0.0, 0.5])
norm = mpl.colors.Normalize(vmin=met_logzsol_values.min(), vmax=met_logzsol_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for met_logzsol in met_logzsol_values:
    params = {**baseline, "met_logzsol": jnp.float64(met_logzsol)}
    out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=cmap(norm(met_logzsol)), lw=1.4)

ax.set_xlim(900, 3e5)
ax.set_ylim(1e40, 1e44)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

for wl in [912, 6563, 1e4, 1e5]:
    ax.axvline(wl, color="gray", ls=":", lw=0.6, alpha=0.3)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Stellar metallicity $\log Z_\star/Z_\odot$")

fig.tight_layout()
plt.savefig("plot_logzsol_panchromatic.png", dpi=150, bbox_inches="tight")
