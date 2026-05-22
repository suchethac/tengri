"""
Workflow: Dust Attenuation Uncertainty via Posterior Resampling
===============================================================

Demonstrates quantifying observational uncertainties through posterior
predictive resampling. A galaxy is fit with NUTS, then the posterior
is resampled 200 times to generate a posterior predictive SED ensemble.
Shows the SED with 1σ and 2σ confidence envelopes. This workflow
illustrates how to propagate Bayesian posterior uncertainty into
derived predictions for robust error budgets.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_workflow_dust_mc_resampling_001.png
   :alt: plot_workflow_dust_mc_resampling
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fitter,
    Fixed,
    ForwardModel,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    data_path,
    load_ssp,
)
from tengri.plot import setup_style

setup_style()

jax.config.update("jax_enable_x64", True)


# --- SSP data ---


ssp = load_ssp()


bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(bands, cache_dir=str(data_path("filters"))))

# --- Model with dust as a free parameter ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.5),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),  # Free
    dust_tau_diff=Uniform(0.0, 1.5),  # Free
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)

model = SEDModel(spec, ssp, observation=obs)

# --- Generate mock photometry ---
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 1.5
true_params["sfh_tsnorm_log_peak_sfr"] = 0.8
true_params["dust_tau_bc"] = 0.5
true_params["dust_tau_diff"] = 0.3
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit with NUTS (generates posterior samples) ---
forward = ForwardModel.build(sed=model, observation=obs)
fitter = Fitter(forward, data=mock.flux_obs, noise=mock.noise)
# Use MAP first to warm-start NUTS
fitter.run("map", optimizer="adam", n_steps=300, verbose=False)
fitter.compile(verbose=False)
# Run NUTS with modest settings for speed
posterior = fitter.run(
    "mcmc_nuts",
    n_warmup=100,
    n_samples=200,
    verbose=False,
)

# --- Posterior predictive resampling ---
# Draw 200 samples from posterior and predict photometry for each
n_resample = 200
posterior_samples = posterior.samples  # dict mapping param name -> array of samples
posterior_photometry = []

# Determine number of samples available
n_samples = len(next(iter(posterior_samples.values()))) if posterior_samples else 0

key_pred = jax.random.PRNGKey(999)
for i in range(min(n_resample, n_samples)):
    # Extract i-th sample from each parameter in the posterior dict
    params_i = {
        param_name: float(sample_array[i])
        for param_name, sample_array in posterior_samples.items()
    }
    phot_i = model.predict_photometry(params_i)
    posterior_photometry.append(np.array(phot_i))

posterior_photometry = np.array(posterior_photometry)

# --- Compute envelopes ---
phot_median = np.median(posterior_photometry, axis=0)
phot_p16 = np.percentile(posterior_photometry, 16, axis=0)
phot_p84 = np.percentile(posterior_photometry, 84, axis=0)
phot_p2_5 = np.percentile(posterior_photometry, 2.5, axis=0)
phot_p97_5 = np.percentile(posterior_photometry, 97.5, axis=0)

# --- Plot: posterior predictive SED with envelopes ---
fig, ax = plt.subplots(figsize=(9, 5))

wave_eff = np.array([3551, 4686, 6166, 7480, 8932])
band_labels = ["u", "g", "r", "i", "z"]

# 2σ (95%) envelope
ax.fill_between(
    wave_eff,
    phot_p2_5,
    phot_p97_5,
    color="C0",
    alpha=0.2,
    label="2σ (95% credible)",
)
# 1σ (68%) envelope
ax.fill_between(
    wave_eff,
    phot_p16,
    phot_p84,
    color="C0",
    alpha=0.4,
    label="1σ (68% credible)",
)

# Median posterior
ax.plot(wave_eff, phot_median, "C0-", lw=2.5, label="Posterior median", marker="o", ms=8)

# Observed data
ax.errorbar(
    wave_eff,
    np.array(mock.flux_obs),
    yerr=np.array(mock.noise),
    fmt="o",
    color="k",
    ms=7,
    capsize=3,
    label="Observed (SNR=20)",
    zorder=5,
)

# Truth
ax.scatter(
    wave_eff,
    np.array(mock.flux_true),
    marker="s",
    s=80,
    facecolors="none",
    edgecolors="red",
    lw=1.5,
    label="Truth",
    zorder=4,
)

ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=12)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]", fontsize=12)
ax.set_title(
    "Posterior Predictive SED: Dust Uncertainty Quantification",
    fontsize=12,
    fontweight="bold",
)
ax.legend(fontsize=10, frameon=False, loc="upper right")
ax.set_xticks(wave_eff)
ax.set_xticklabels(band_labels)

fig.tight_layout()
plt.savefig("plot_workflow_dust_mc_resampling.png", dpi=150, bbox_inches="tight")
plt.show()
