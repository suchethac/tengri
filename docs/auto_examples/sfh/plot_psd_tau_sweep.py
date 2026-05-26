"""
PSD timescale τ controls burst duration in stochastic SFHs
==========================================================

The damping timescale τ (in Myr) of the power spectral density governs how
long star-formation bursts persist. Short τ means rapid flickering; long τ
means sustained episodes that leave their imprint on the SED. We vary τ
across the prior range with the burst amplitude σ fixed.
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
        "type": "field_psd",
        "*": tengri.FIXED,
        "mean": "tsnorm",
        "tsnorm_log_total_mass": 1.0,
        "tsnorm_peak_lbt_gyr": 3.0,
        "tsnorm_width_gyr": 2.0,
        "tsnorm_skew": 0.3,
        "tsnorm_trunc": 2.0,
        "psd_sigma": 1.0,
        "psd_tau_myr": tengri.Uniform(30, 3000),
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

tau_values = np.array([30, 100, 300, 1000, 3000])
norm = mpl.colors.Normalize(vmin=tau_values.min(), vmax=tau_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
key_base = jax.random.PRNGKey(42)
for i, tau in enumerate(tau_values):
    for k in range(3):
        params = {**baseline, "sfh_field_psd_tau_myr": jnp.float64(tau)}
        key = jax.random.fold_in(key_base, i * 10 + k)
        # Sample from the stochastic field
        out = model.predict_rest_sed(params, key=key)
        wave = np.asarray(out.wavelength)
        nu = 2.998e18 / wave  # Å/s -> Hz
        nu_l_nu = nu * np.asarray(out.sed)
        ax.loglog(wave, nu_l_nu, color=cmap(norm(tau)), lw=0.8, alpha=0.6)

ax.set_xlim(800, 3e4)
ax.set_ylim(1e40, 5e43)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"PSD timescale $\tau$ [Myr]")

fig.tight_layout()
plt.savefig("plot_psd_tau_sweep.png", dpi=150, bbox_inches="tight")
