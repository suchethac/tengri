"""
Recovering stellar mass from 5-band SDSS photometry
====================================================

MAP returns a point estimate; nothing here estimates uncertainty. Six free
parameters, which is the validated ceiling for ``method="mcmc_nuts"``;
``method="laplace"`` is the cheaper route to intervals, from the Hessian at the
MAP. ``vi`` and ``mcmc_raytrace`` target D ≳ 20. See the method-selection page
for the full decision table.

Reference: Calzetti+2000 (attenuation law).
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

BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

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
    redshift=tengri.Fixed(0.05),
)

key = jax.random.PRNGKey(42)
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
fit_params = posterior.params

# Print fit diagnostics
residual_phot = (np.asarray(model.predict_photometry(fit_params)) -
                 np.asarray(mock.flux_obs)) / np.asarray(mock.noise)
chi2 = np.sum(residual_phot**2)
n_bands = len(mock.flux_obs)
print("MAP fit diagnostics:")
print(f"  χ²: {chi2:.2f}  |  n_bands: {n_bands}  |  max|residual|/σ: {np.max(np.abs(residual_phot)):.2f}")
print("SFH parameter recovery:")
print(f"  log M* [M☉]:     truth {truth['sfh_tsnorm_log_total_mass']:.3f}  fit {fit_params['sfh_tsnorm_log_total_mass']:.3f}")
print(f"  peak age [Gyr]:  truth {truth['sfh_tsnorm_peak_lbt_gyr']:.3f}  fit {fit_params['sfh_tsnorm_peak_lbt_gyr']:.3f}")
print(f"  τ_diff [mag]:    truth {truth['dust_tau_diff']:.3f}  fit {fit_params['dust_tau_diff']:.3f}")

flux_truth = np.asarray(mock.flux_true)
flux_fit = np.asarray(model.predict_photometry(fit_params))
flux_obs = np.asarray(mock.flux_obs)
noise = np.asarray(mock.noise)
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

sed_truth = model.predict(truth)
sed_fit = model.predict(fit_params)
wave_rest = np.asarray(model.wavelengths)
z = 0.05
wave_obs = wave_rest * (1.0 + z)


# Anchor the L_nu SED to the observed flux scale by matching the r-band
# integral: keeps the shape but lands the curve on the data points so a
# reader can read the SED off the same axis as the photometry.
def _band_anchor(sed):
    idx = np.argmin(np.abs(wave_obs - wave_eff[2]))
    return sed[idx]


scale_truth = flux_truth[2] / _band_anchor(np.asarray(sed_truth.rest_sed()))
fnu_truth = scale_truth * np.asarray(sed_truth.rest_sed())
fnu_fit = scale_truth * np.asarray(sed_fit.rest_sed())

fig, (ax_sed, ax_res) = plt.subplots(
    2,
    1,
    figsize=(7.0, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

vis = (wave_obs > 2.5e3) & (wave_obs < 1.2e4)
ax_sed.plot(wave_obs[vis], fnu_truth[vis], color="0.55", lw=0.8, label="Truth SED")
ax_sed.plot(wave_obs[vis], fnu_fit[vis], color="C3", lw=0.8, alpha=0.8, label="MAP SED")
ax_sed.errorbar(
    wave_eff,
    flux_obs,
    yerr=noise,
    fmt="o",
    color="k",
    ms=5,
    capsize=2,
    label="SDSS mock (S/N = 20)",
)
ax_sed.plot(wave_eff, flux_truth, "s", color="0.2", ms=6, mfc="none", mew=1.2, label="Truth bands")
ax_sed.plot(wave_eff, flux_fit, "^", color="C3", ms=6, mfc="none", mew=1.2, label="MAP bands")
ax_sed.set_yscale("log")
ax_sed.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

residual = (flux_fit - flux_obs) / noise
ax_res.axhline(0.0, color="0.5", lw=0.6)
ax_res.axhspan(-1.0, 1.0, color="0.85", alpha=0.6, lw=0)
ax_res.plot(wave_eff, residual, "o", color="C3", ms=5)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_ylabel(r"$(F_{\rm fit} - F_{\rm obs})/\sigma$")
ax_res.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")

plt.savefig("plot_first_fit.png", dpi=150, bbox_inches="tight")
