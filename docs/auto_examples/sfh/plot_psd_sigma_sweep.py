"""
PSD amplitude σ controls burst magnitude in stochastic SFHs
===========================================================

The amplitude σ of the power spectral density sets how dramatically star
formation fluctuates around the smooth trend: σ ≈ 0 means nearly constant SFR,
large σ produces dramatic bursts that leave imprints in UV slope, optical
colors, and stellar masses. We vary σ across its prior range with the timescale
τ fixed.
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

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "field_psd",
        "*": tengri.FIXED,
        "mean": "tsnorm",
        "tsnorm_log_peak_sfr": 1.0,
        "tsnorm_peak_lbt_gyr": 3.0,
        "tsnorm_width_gyr": 2.0,
        "tsnorm_skew": 0.3,
        "tsnorm_trunc": 2.0,
        "psd_sigma": tengri.Uniform(0.1, 3.5),
        "psd_tau_myr": 100.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

sigma_values = np.array([0.1, 0.5, 1.0, 2.0, 3.5])
norm = mpl.colors.Normalize(vmin=sigma_values.min(), vmax=sigma_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
key_base = jax.random.PRNGKey(0)
for i, sigma in enumerate(sigma_values):
    for k in range(3):
        params = {**baseline, "sfh_field_psd_sigma": jnp.float64(sigma)}
        key = jax.random.fold_in(key_base, i * 10 + k)
        # Sample from the stochastic field
        out = model.predict_rest_sed(params, key=key)
        wave = np.asarray(out.wavelength)
        nu = 2.998e18 / wave  # Å/s -> Hz
        nu_l_nu = nu * np.asarray(out.sed)
        ax.loglog(wave, nu_l_nu, color=cmap(norm(sigma)), lw=0.8, alpha=0.6)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"PSD amplitude $\sigma$")

fig.tight_layout()
fig.savefig("plot_psd_sigma_sweep.png", dpi=150, bbox_inches="tight")
