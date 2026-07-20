"""
Recovering a z=3 galaxy from JWST NIRCam photometry
====================================================

A higher-redshift counterpart to the SDSS quickstart fit. We mock JWST
NIRCam wide-band photometry of a star-forming galaxy at z=3 (S/N=15),
run a MAP fit, and show the recovered SED + per-band residuals. NIRCam
samples the rest-frame UV-optical at this redshift, so the SFH and dust
attenuation are the dominant levers.

Reference: Conroy 2013, ARA&A, 51, 393.
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

BANDS = ["jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w", "jwst_f356w", "jwst_f444w"]

obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))
model = tengri.SEDModel.build(
    tengri.load_ssp(),
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FREE},
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(3.0),
)

key = jax.random.PRNGKey(7)
truth = dict(model.spec.sample(key))
truth.update(
    sfh_tsnorm_peak_lbt_gyr=1.5,
    sfh_tsnorm_width_gyr=0.7,
    sfh_tsnorm_log_total_mass=1.4,
    sfh_tsnorm_skew=0.2,
    sfh_tsnorm_trunc=2.0,
    dust_tau_diff=0.5,
)
mock = model.mock(truth, snr=15.0, key=key)

forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs, mock.noise, method="map", optimizer="adam", n_steps=300, verbose=False
)
fit_params = posterior.params

flux_truth = np.asarray(mock.flux_true)
flux_fit = np.asarray(model.predict_photometry(fit_params))
flux_obs = np.asarray(mock.flux_obs)
noise = np.asarray(mock.noise)
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

fig, (ax_sed, ax_res) = plt.subplots(
    2,
    1,
    figsize=(7.2, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)
ax_sed.errorbar(
    wave_eff,
    flux_obs,
    yerr=noise,
    fmt="o",
    color="k",
    ms=5,
    capsize=2,
    label="NIRCam mock (S/N = 15)",
)
ax_sed.plot(wave_eff, flux_truth, "s", color="0.2", ms=6, mfc="none", mew=1.2, label="Truth")
ax_sed.plot(wave_eff, flux_fit, "^", color="C3", ms=6, mfc="none", mew=1.2, label="MAP")
ax_sed.set_yscale("log")
ax_sed.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=8, loc="lower right")

residual = (flux_fit - flux_obs) / noise
ax_res.axhline(0.0, color="0.5", lw=0.6)
ax_res.axhspan(-1.0, 1.0, color="0.85", alpha=0.6, lw=0)
ax_res.plot(wave_eff, residual, "o", color="C3", ms=5)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_ylabel(r"$(F_{\rm fit} - F_{\rm obs})/\sigma$")
ax_res.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")

plt.savefig("plot_photometric_fit.png", dpi=150, bbox_inches="tight")
