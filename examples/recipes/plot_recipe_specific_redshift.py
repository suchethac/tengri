"""
Redshift constraint: spectroscopy vs photometry alone
======================================================

Two fits on the same mock data: one with redshift fixed (spectroscopic
known, free SFH/dust/met), one with redshift free (photometric only).
The fixed-z fit converges to truth; free-z is degenerate with dust and SFH,
showing why spectroscopy breaks the age-dust-redshift degeneracies
that plague photometry-only fitting.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
TRUE_REDSHIFT = 0.15
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# Generate mock data at known redshift
key = jax.random.PRNGKey(42)
true_params = {
    "sfh_tsnorm_log_total_mass": 1.2,
    "sfh_tsnorm_peak_lbt_gyr": 2.5,
    "sfh_tsnorm_width_gyr": 1.8,
    "sfh_tsnorm_skew": 0.1,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.2,
    "dust_tau_bc": 0.1,
    "dust_tau_diff": 0.25,
    "dust_slope": -0.7,
    "redshift": TRUE_REDSHIFT,
}
model_template = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        **{k: v for k, v in true_params.items() if k.startswith("sfh_")},
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        **{k: v for k, v in true_params.items() if k.startswith("dust_")},
    },
    redshift=tengri.Fixed(TRUE_REDSHIFT),
)
mock = model_template.mock(true_params, snr=25.0, key=key)

# Fit 1: Redshift FIXED (spectroscopy known)
model_fixed = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FREE},
    dust={"type": "two_component", "all_params": tengri.FREE},
    redshift=tengri.Fixed(TRUE_REDSHIFT),
)
forward_fixed = tengri.ForwardModel.build(sed=model_fixed, observation=obs)
posterior_fixed = forward_fixed.fit(
    mock.flux_obs, mock.noise, method="map", optimizer="adam", n_steps=200, verbose=False
)

# Fit 2: Redshift FREE (photometry only)
model_free = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FREE},
    dust={"type": "two_component", "all_params": tengri.FREE},
    redshift=tengri.Uniform(0.0, 0.5),
)
forward_free = tengri.ForwardModel.build(sed=model_free, observation=obs)
posterior_free = forward_free.fit(
    mock.flux_obs, mock.noise, method="map", optimizer="adam", n_steps=200, verbose=False
)

# Plot: SFH comparison
sfh_fixed = model_fixed.predict_sfh(posterior_fixed.params)
sfh_free = model_free.predict_sfh(posterior_free.params)

fig, (ax_fixed, ax_free) = plt.subplots(1, 2, figsize=(10.0, 3.5), sharey=True)

t_gyr = np.array(sfh_fixed["t_gyr"])
mask = t_gyr < 2.0

ax_fixed.plot(
    t_gyr[mask], np.array(sfh_fixed["sfr_mean"])[mask], color="C0", lw=1.4, label="MAP SFR"
)
ax_fixed.set_xlabel("Lookback time [Gyr]")
ax_fixed.set_ylabel(r"SFR [$M_\odot$/yr]")
ax_fixed.set_ylim(bottom=0)
ax_fixed.legend(frameon=False, fontsize=8)
ax_text = ax_fixed.text(
    0.05, 0.95, "Redshift fixed", transform=ax_fixed.transAxes, fontsize=9, verticalalignment="top"
)

t_gyr_free = np.array(sfh_free["t_gyr"])
mask_free = t_gyr_free < 2.0
ax_free.plot(
    t_gyr_free[mask_free],
    np.array(sfh_free["sfr_mean"])[mask_free],
    color="C3",
    lw=1.4,
    label="MAP SFR",
)
ax_free.set_xlabel("Lookback time [Gyr]")
ax_free.set_ylim(bottom=0)
ax_free.legend(frameon=False, fontsize=8)
ax_text = ax_free.text(
    0.05, 0.95, "Redshift free", transform=ax_free.transAxes, fontsize=9, verticalalignment="top"
)

fig.tight_layout()
plt.savefig("plot_recipe_specific_redshift.png", dpi=150, bbox_inches="tight")
