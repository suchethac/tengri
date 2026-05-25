"""
Log-normal peak lookback time shifts stellar age and SED morphology
===================================================================

The peak lookback time of a log-normal SFH controls when most stars formed,
shifting the age structure and dramatically affecting UV slope, 4000 Å break
strength, and NIR luminosity. We vary the peak time across its prior range with
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
        "type": "lnorm",
        "*": tengri.FIXED,
        "peak_lbt_gyr": tengri.Uniform(1.0, 11.0),
        "log_total_mass": 10.0,
        "width_gyr": 0.3,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

peak_values = np.linspace(1.0, 11.0, 7)
norm = mpl.colors.Normalize(vmin=peak_values.min(), vmax=peak_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for peak in peak_values:
    params = {**baseline, "sfh_lnorm_peak_lbt_gyr": jnp.float64(peak)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave  # Å/s -> Hz
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(peak)), lw=1.4)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Peak lookback time [Gyr]")

fig.tight_layout()
plt.savefig("plot_lnorm_peak_sweep.png", dpi=150, bbox_inches="tight")
