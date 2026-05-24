"""
Posterior corner plot from MCMC: double-power-law SFH
=====================================================

Demonstrates MCMC parameter estimation and posterior covariance structure
after fitting mock 5-band SDSS photometry with a double-power-law (dpl) star
formation history. The corner plot visualizes all 1-D marginalized posteriors
and 2-D joint distributions, with blue lines marking the injected truth values.

The model has four free parameters in the SFH: alpha (rise), beta (decline),
tau_gyr (timescale), and log_peak_sfr (normalization). The posterior reveals
parameter degeneracies between SFH shape and dust attenuation.

Reference: Foreman-Mackey 2016, corner.py (https://arxiv.org/abs/1606.02919);
Conroy 2013, ARA&A, 51, 393 (SED fitting overview).
"""

import warnings

import corner
import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# --- Build the model with a double-power-law SFH ---
ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "dpl", "*": tengri.FREE},
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.05),
)

# --- Define true parameters and mock photometry at S/N=20 ---
key = jax.random.PRNGKey(42)
truth = dict(model.spec.sample(key))
truth.update(
    sfh_dpl_alpha=2.5,
    sfh_dpl_beta=1.2,
    sfh_dpl_tau_gyr=8.0,
    sfh_dpl_log_peak_sfr=1.1,
    dust_tau_diff=0.3,
)
mock = model.mock(truth, snr=20.0, key=key)

# --- Fit with MCMC (NUTS warmup + sampling) ---
forward = tengri.ForwardModel.build(sed=model, observation=obs)

# Run NUTS MCMC with small iteration count for demo (warmup=500, samples=500 ~ 90s)
posterior = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="nuts",
    warmup=500,
    samples=500,
    verbose=False,
)

# Extract samples and parameter names for the corner plot
samples_dict = posterior.samples
param_names = list(samples_dict.keys())
samples_array = np.array([samples_dict[p] for p in param_names]).T
truths = [float(truth[p]) for p in param_names]

# --- Create corner plot ---
fig = corner.corner(
    samples_array,
    labels=param_names,
    truths=truths,
    color="C0",
    hist_kwargs={"density": True},
    show_titles=False,
)

plt.savefig("plot_posterior_corner_dpl.png", dpi=150, bbox_inches="tight")
