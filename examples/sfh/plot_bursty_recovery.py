"""
SFH Recovery Across Four Burstiness Regimes
============================================

Four burstiness regimes — Smooth, Moderate, Bursty, Extreme — each defined by
the PSD amplitude σ and correlation time τ. Forward-model SFH draws show the
range of histories each regime produces before inference.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_bursty_recovery_001.png
   :alt: plot_bursty_recovery
   :class: sphx-glr-single-img

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fixed,
    Observation,
    Parameters,
    SEDModel,
    Spectroscopy,
    Uniform,
    load_ssp,
    setup_style,
)

setup_style()


ssp = load_ssp()
wave_obs = jnp.linspace(3800.0, 9200.0, 200)
obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.2),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(3.0),
    sfh_tsnorm_skew=Fixed(0.3),
    sfh_tsnorm_trunc=Fixed(2.0),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.0),
    mean_sfh_type=["tsnorm", "field"],
)
model = SEDModel(spec, ssp, observation=obs)

REGIMES = [
    {"label": "Smooth", "sigma": 0.3, "tau": 100.0, "color": "#1f77b4"},
    {"label": "Moderate", "sigma": 1.0, "tau": 50.0, "color": "#ff7f0e"},
    {"label": "Bursty", "sigma": 2.0, "tau": 20.0, "color": "#2ca02c"},
    {"label": "Extreme", "sigma": 3.0, "tau": 5.0, "color": "#d62728"},
]

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
axes_flat = axes.flatten()

for ax, reg in zip(axes_flat, REGIMES):
    key = jax.random.PRNGKey(42)
    params = {
        **spec.sample(key),
        "sfh_field_psd_sigma": jnp.array(reg["sigma"]),
        "sfh_field_psd_tau_myr": jnp.array(reg["tau"]),
    }
    sfh = model.predict_sfh(params)
    t_gyr = np.array(sfh["t_gyr"])
    sfr_full = np.array(sfh["sfr_full"])
    sfr_mean = np.array(sfh["sfr_mean"])
    ax.fill_between(t_gyr, 0, sfr_full, alpha=0.4, color=reg["color"])
    ax.plot(t_gyr, sfr_full, color=reg["color"], lw=1.2)
    ax.plot(t_gyr, sfr_mean, color="k", lw=0.8, ls="--", alpha=0.5, label="Mean SFH")
    ax.set_title(f"{reg['label']}: σ={reg['sigma']}, τ={reg['tau']:.0f} Myr")
    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot$ yr$^{-1}$]")
    ax.set_xlim(0, 13)
    ax.set_ylim(bottom=0)

fig.suptitle("Four Burstiness Regimes: IFT PSD Prior Draws", fontsize=12, y=1.02)
fig.tight_layout()
plt.savefig("plot_bursty_recovery.png", dpi=150, bbox_inches="tight")
plt.show()
