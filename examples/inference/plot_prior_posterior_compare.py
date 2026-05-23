"""
Prior vs posterior: data-driven parameter constraints
=====================================================

Bayesian inference refines broad priors (uniform distributions) into
narrow posteriors using observed data. This script shows three key
parameters (age of peak SFR, stellar metallicity, dust optical depth)
as priors (dashed gray lines) and posteriors (blue histograms) after
fitting mock 5-band photometry. The red vertical line marks the injected
truth value.

Reference: Conroy 2013, ARA&A, 51, 393 (SED fitting overview).
"""

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
    sfh={"type": "tsnorm", "*": tengri.FREE},
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.1),
)

key = jax.random.PRNGKey(99)
truth = dict(model.spec.sample(key))
truth.update(
    sfh_tsnorm_peak_lbt_gyr=3.0,
    sfh_tsnorm_width_gyr=2.0,
    sfh_tsnorm_log_peak_sfr=1.0,
    sfh_tsnorm_skew=0.3,
    sfh_tsnorm_trunc=10.0,
    met_logzsol=-0.2,
    dust_tau_diff=0.5,
)
mock = model.mock(truth, snr=20.0, key=key)

forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs, mock.noise, method="native_vi_nonlinear", n_iterations=500, n_samples=3,
    verbose=False,
)

selected_params = [
    "sfh_tsnorm_peak_lbt_gyr",
    "met_logzsol",
    "dust_tau_diff",
]
param_labels = [
    r"Age of peak SFR [Gyr]",
    r"log(Z/Z$_\odot$)",
    r"$\tau_{\rm diff}$",
]
prior_ranges = [
    (0.5, 12.0),
    (-2.0, 0.2),
    (0.0, 1.5),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for i, (param_name, param_label, (prior_min, prior_max)) in enumerate(
    zip(selected_params, param_labels, prior_ranges)
):
    ax = axes[i]
    samples = np.array(posterior.samples[param_name])

    prior_density = 1.0 / (prior_max - prior_min)
    ax.plot(
        [prior_min, prior_max],
        [prior_density, prior_density],
        color="gray",
        lw=2.0,
        ls="--",
        label="Prior (uniform)",
    )

    ax.hist(
        samples,
        bins=30,
        color="C0",
        alpha=0.6,
        density=True,
        label="Posterior",
        edgecolor="C0",
        linewidth=1.0,
    )

    truth_val = float(truth[param_name])
    ax.axvline(truth_val, color="red", lw=2.0, ls="-", alpha=0.7, label="Truth")

    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("plot_prior_posterior_compare.png", dpi=150, bbox_inches="tight")
