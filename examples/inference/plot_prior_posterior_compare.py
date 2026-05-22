"""
Prior vs Posterior: Parameter Constraints from Inference
=========================================================

A Bayesian fit uses prior distributions over parameters and refines them
using observed data to obtain posteriors. This script shows how priors
(dashed lines) and posteriors (histograms) differ for key physical
parameters (stellar mass age, metallicity, dust optical depth) after
fitting mock photometry.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_prior_posterior_compare_001.png
   :alt: plot_prior_posterior_compare
   :class: sphx-glr-single-img

"""

from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Already has Path imported
from tengri import (
    Fitter,
    Fixed,
    ForwardModel,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp,
    setup_style,
)

setup_style()

ssp = load_ssp()
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

# --- Define priors (will be visualized as "prior" distributions) ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)

model = SEDModel(spec, ssp, observation=obs)

# --- Generate mock data with known truth ---
key = jax.random.PRNGKey(99)
true_params = spec.sample(key)
# Override to ensure interesting star-forming galaxy
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 2.0
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
true_params["met_logzsol"] = -0.2
true_params["dust_tau_diff"] = 0.5
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit ---
forward = ForwardModel.build(sed=model, observation=obs)
fitter = Fitter(forward, mock.flux_obs, mock.noise, data_type="photometry")
fitter.run("map", n_steps=200, verbose=False)
fitter.compile(verbose=False)
posterior = fitter.run(
    "vi",
    n_iterations=8,
    n_samples=3,
    n_posterior_samples=1000,
    verbose=False,
)

# --- Extract posterior samples and priors ---
posterior_samples = posterior.samples
param_names = list(posterior_samples.keys())

# Select key 3 parameters for comparison
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

# Prior ranges (from Uniform definitions in spec)
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

    if param_name in posterior_samples:
        samples = np.array(posterior_samples[param_name])

        # Plot prior as uniform horizontal line over the prior support.
        prior_density = 1.0 / (prior_max - prior_min)
        ax.plot(
            [prior_min, prior_max],
            [prior_density, prior_density],
            color="gray",
            lw=2.0,
            ls="--",
            label="Prior (uniform)",
        )

        # Plot posterior histogram
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

        # Mark truth value
        if param_name in true_params:
            truth_val = true_params[param_name]
            ax.axvline(truth_val, color="red", lw=2.0, ls="-", alpha=0.7, label="Truth")

        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel("Probability density", fontsize=10)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "Prior vs Posterior: Parameter Constraints from Mock Photometry Fit",
    fontsize=13,
)
fig.tight_layout()
# Save to script directory
script_dir = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
plt.savefig(str(script_dir / "plot_prior_posterior_compare.png"), dpi=150, bbox_inches="tight")
plt.close()
