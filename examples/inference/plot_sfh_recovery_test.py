"""
SFH recovery with MAP: double power-law against mock photometry
===============================================================

Reference: Conroy+2013.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
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
    sfh={"type": "dpl", "all_params": tengri.FREE},
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.05),
)

# Define true parameters: dpl model with alpha=2.5, beta=1.2, tau_gyr=8.0
key = jax.random.PRNGKey(42)
truth = dict(model.spec.sample(key))
truth.update(
    sfh_dpl_alpha=2.5,
    sfh_dpl_beta=1.2,
    sfh_dpl_tau_gyr=8.0,
    sfh_dpl_log_total_mass=10.0,
    dust_tau_diff=0.3,
)

# Mock photometry at S/N = 20
mock = model.mock(truth, snr=20.0, key=key)

# Fit with MAP
forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)
fit_params = posterior.params

# Get photometric predictions and residuals
flux_truth = np.asarray(mock.flux_true)
flux_fit = np.asarray(model.predict_photometry(fit_params))
flux_obs = np.asarray(mock.flux_obs)
noise = np.asarray(mock.noise)
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

# Get SFH predictions over a time grid
sfh_true_dict = model.predict_sfh(truth, n_linear=100)
sfh_fit_dict = model.predict_sfh(fit_params, n_linear=100)
t_lookback = sfh_true_dict["t_gyr"]
sfh_true = sfh_true_dict["sfr_mean"]
sfh_fit = sfh_fit_dict["sfr_mean"]

# Set up figure: top panel SFH, bottom panel photometric residuals
fig, (ax_sfh, ax_res) = plt.subplots(
    2,
    1,
    figsize=(8.0, 5.8),
    sharex=False,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.25},
)

# Top: SFH comparison
ax_sfh.plot(t_lookback, sfh_true, color="0.55", lw=1.2, label="True SFH")
ax_sfh.plot(t_lookback, sfh_fit, color="C3", lw=1.2, alpha=0.8, label="MAP SFH")
ax_sfh.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax_sfh.set_xlabel(r"Lookback time [Gyr]")
ax_sfh.legend(frameon=False, fontsize=9, loc="upper left")
ax_sfh.set_xlim(13.5, 0.0)
ax_sfh.grid(True, alpha=0.2, linestyle=":")

# Bottom: photometric residuals
residual = (flux_fit - flux_obs) / noise
ax_res.axhline(0.0, color="0.5", lw=0.6)
ax_res.axhspan(-1.0, 1.0, color="0.85", alpha=0.6, lw=0)
ax_res.plot(wave_eff, residual, "o", color="C3", ms=5)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_ylabel(r"$(F_{\rm fit} - F_{\rm obs})/\sigma$")
ax_res.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")

plt.savefig("plot_sfh_recovery_test.png", dpi=150, bbox_inches="tight")
