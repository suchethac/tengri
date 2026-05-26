"""
Gas metallicity shifts optical emission line ratios
===================================================

Gas metallicity controls [NII]/Hα and [OIII]/Hβ ratios, the primary
optical metallicity diagnostics. We vary nebular metallicity across
the abundance range.
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
    neb={"type": "cue", "*": tengri.FIXED, "logZ_gas": tengri.Uniform(-1.5, 0.3)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

logz_values = np.linspace(-1.5, 0.3, 7)
norm = mpl.colors.Normalize(vmin=logz_values.min(), vmax=logz_values.max())
cmap = plt.get_cmap("YlGn")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for logz in logz_values:
    params = {**baseline, "neb_logZ_gas": jnp.float64(logz)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.semilogy(wave, nu_l_nu, color=cmap(norm(logz)), lw=1.4)

ax.set_xlim(4000, 7500)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log(Z/Z_\odot)$")

fig.tight_layout()
plt.savefig("plot_logz_gas_sweep.png", dpi=150, bbox_inches="tight")
