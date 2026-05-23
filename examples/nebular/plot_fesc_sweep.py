"""
Ionizing photon escape suppresses nebular emission
===================================================

Escape fraction ``f_esc`` sets what fraction of ionizing photons reach the
ISM. Higher ``f_esc`` suppresses all nebular emission lines since fewer
photons remain to ionize gas.
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
        "tau_gyr": 0.3,
        "log_peak_sfr": 1.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "neb_fesc": tengri.Uniform(0.0, 1.0)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

fesc_values = np.linspace(0.0, 1.0, 6)
norm = mpl.colors.Normalize(vmin=fesc_values.min(), vmax=fesc_values.max())
cmap = plt.get_cmap("Purples")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for fesc in fesc_values:
    params = {**baseline, "neb_fesc": jnp.float64(fesc)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(fesc)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$f_{\mathrm{esc}}$")

fig.tight_layout()
fig.savefig("plot_fesc_sweep.png", dpi=150, bbox_inches="tight")
