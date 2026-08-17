"""
Model misspecification: post-starburst galaxies reveal wrong SFH
================================================================

A post-starburst galaxy shows a recent burst followed by quenching.
When fit with a smooth exponential (incorrect), the fit biases the
recovered SFH.

References: Cid Fernandes+2005; Conroy+2013.
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
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

key = jax.random.PRNGKey(42)
truth_params = {
    "sfh_tsnorm_log_total_mass": 1.5,
    "sfh_tsnorm_peak_lbt_gyr": 0.5,
    "sfh_tsnorm_width_gyr": 0.2,
    "sfh_tsnorm_skew": 0.8,
    "sfh_tsnorm_trunc": 1.5,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.2,
    "dust_tau_diff": 0.1,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

model_truth = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FIXED},
    dust={"type": "two_component", "all_params": tengri.FIXED},
    redshift=tengri.Fixed(0.1),
)
mock = model_truth.mock(truth_params, snr=20.0, key=key)

model_correct = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_total_mass": 10.0,
        "peak_lbt_gyr": tengri.Uniform(0.2, 2.0),
        "width_gyr": tengri.Uniform(0.1, 1.0),
        "skew": tengri.Uniform(-0.5, 2.0),
        "trunc": tengri.Uniform(0.5, 5.0),
        "logzsol": tengri.Fixed(-0.1),
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.2,
        "tau_diff": 0.1,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.1),
)
forward_correct = tengri.ForwardModel.build(sed=model_correct, observation=obs)
post_correct = forward_correct.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)

model_wrong = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "dpl",
        "log_total_mass": 10.0,
        "tau_gyr": tengri.Uniform(0.5, 10.0),
        "alpha": tengri.Fixed(1.0),
        "beta": tengri.Fixed(0.1),
        "logzsol": tengri.Fixed(-0.1),
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.2,
        "tau_diff": 0.1,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.1),
)
forward_wrong = tengri.ForwardModel.build(sed=model_wrong, observation=obs)
post_wrong = forward_wrong.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)

sfh_true = model_truth.predict_sfh(truth_params)
sfh_correct = model_correct.predict_sfh(post_correct.params)
sfh_wrong = model_wrong.predict_sfh(post_wrong.params)

fig, ax = plt.subplots(figsize=(9, 5))

t_gyr_true = np.array(sfh_true["t_gyr"])
mask = t_gyr_true < 2.0

t_gyr_correct = np.array(sfh_correct["t_gyr"])
mask_c = t_gyr_correct < 2.0

t_gyr_wrong = np.array(sfh_wrong["t_gyr"])
mask_w = t_gyr_wrong < 2.0

ax.plot(
    t_gyr_true[mask],
    np.array(sfh_true["sfr_mean"])[mask],
    "k-",
    lw=2.5,
    label="Truth (burst + quench)",
)
ax.plot(
    t_gyr_correct[mask_c],
    np.array(sfh_correct["sfr_mean"])[mask_c],
    "--",
    color="C0",
    lw=2.0,
    label="tsnorm fit (correct model)",
)
ax.plot(
    t_gyr_wrong[mask_w],
    np.array(sfh_wrong["sfr_mean"])[mask_w],
    "--",
    color="C3",
    lw=2.0,
    label="DPL fit (wrong model)",
)

ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
ax.text(0.5, ax.get_ylim()[1] * 0.9, "Quench epoch", fontsize=9, color="gray")

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(bottom=0)

fig.tight_layout()
plt.savefig("plot_workflow_post_starburst.png", dpi=150, bbox_inches="tight")
