"""
Recovering a truncated-skew-normal SFH from SDSS photometry via MAP
===================================================================

Reference: Conroy+2013.
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

jax.config.update("jax_enable_x64", True)

ssp = tengri.load_ssp()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_total_mass": tengri.Uniform(9.0, 11.0),
        "peak_lbt_gyr": tengri.Uniform(0.5, 12.0),
        "width_gyr": tengri.Uniform(0.3, 5.0),
        "skew": tengri.Uniform(-1.0, 1.5),
        "trunc": tengri.Uniform(1.0, 10.0),
        "logzsol": tengri.Uniform(-2.0, 0.2),
    },
    dust={
        "type": "two_component",
        "tau_bc": tengri.Uniform(0.0, 2.0),
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": tengri.Fixed(-0.7),
    },
    redshift=tengri.Fixed(0.1),
)

key = jax.random.PRNGKey(42)
truth_params = {
    "sfh_tsnorm_peak_lbt_gyr": 2.5,
    "sfh_tsnorm_width_gyr": 1.5,
    "sfh_tsnorm_log_total_mass": 10.0,
    "sfh_tsnorm_skew": 0.2,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.6,
    "dust_tau_diff": 0.4,
    "dust_slope": -0.7,
    "redshift": 0.1,
}
mock = model.mock(truth_params, snr=20.0, key=key)

forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior_map = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)

sfh_truth = model.predict_sfh(truth_params)
sfh_map = model.predict_sfh(posterior_map.params)

fig_sfh, ax_sfh = plt.subplots(figsize=(9, 5))

t_gyr_truth = np.array(sfh_truth["t_gyr"])
mask = t_gyr_truth < 5.0

ax_sfh.plot(
    t_gyr_truth[mask],
    np.array(sfh_truth["sfr_mean"])[mask],
    "k-",
    lw=2.5,
    label="Truth",
)
ax_sfh.plot(
    np.array(sfh_map["t_gyr"])[mask],
    np.array(sfh_map["sfr_mean"])[mask],
    "--",
    color="C0",
    lw=2.0,
    label="MAP fit",
)

ax_sfh.set_xlabel("Lookback time [Gyr]")
ax_sfh.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
ax_sfh.legend(frameon=False)
ax_sfh.set_ylim(bottom=0)

fig_sfh.tight_layout()
fig_sfh.savefig("plot_workflow_method_comparison.png", dpi=150, bbox_inches="tight")
