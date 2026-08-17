"""
MAP fit recovery: star-formation history from mock photometry
=============================================================

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

ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FREE},
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.05),
)

key = jax.random.PRNGKey(7)
truth = dict(model.spec.sample(key))
truth.update(
    sfh_tsnorm_peak_lbt_gyr=3.0,
    sfh_tsnorm_width_gyr=2.0,
    sfh_tsnorm_log_total_mass=10.0,
    sfh_tsnorm_skew=0.3,
    sfh_tsnorm_trunc=10.0,
    dust_tau_diff=0.3,
)
mock = model.mock(truth, snr=20.0, key=key)

forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)

fig, ax = plt.subplots(figsize=(7, 4.5))

# SFH truth vs MAP
sfh_true = model.predict_sfh(truth)
sfh_fit = model.predict_sfh(posterior.params)
t_gyr = np.array(sfh_true["t_gyr"])
mask = t_gyr < 5.0
ax.plot(t_gyr[mask], np.array(sfh_true["sfr_mean"])[mask], "k-", lw=1.5, label="Truth")
ax.plot(t_gyr[mask], np.array(sfh_fit["sfr_mean"])[mask], "C3--", lw=1.2, label="MAP")
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_convergence.png", dpi=150, bbox_inches="tight")
