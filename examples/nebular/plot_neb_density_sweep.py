"""
Nebular density affects recombination and cooling
==================================================

Nebular gas density controls ionization balance and recombination rates,
affecting emission line strengths. Higher density increases cooling efficiency,
shifting line ratios through recombination rate changes.
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
        "all_params": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": 0.3,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    neb={"type": "cue", "all_params": tengri.FIXED, "gas_logn": tengri.Uniform(0.0, 3.0)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

nh_values = np.logspace(0, 3, 7)
norm = mpl.colors.LogNorm(vmin=nh_values.min(), vmax=nh_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Collect data to compute data-driven ylim
all_curve_data = []
for nh in nh_values:
    params = {**baseline, "gas_logn": jnp.float64(np.log10(nh))}
    out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.rest_sed())
    ax.semilogy(wave, nu_l_nu, color=cmap(norm(nh)), lw=1.4)

    # Track values in the plotted window for ylim
    mask = (wave >= 4000) & (wave <= 7500)
    all_curve_data.append(nu_l_nu[mask])

# Set ylim to focus on the continuum and lines
all_vals = np.concatenate(all_curve_data)
y_median = np.median(all_vals)
y_max = np.max(all_vals)
y_min_auto = y_median / 30.0
y_max_auto = y_max * 2.0
ax.set_ylim(y_min_auto, y_max_auto)

ax.set_xlim(4000, 7500)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$n_H$ [cm$^{-3}$]")

fig.tight_layout()
plt.savefig("plot_neb_density_sweep.png", dpi=150, bbox_inches="tight")
