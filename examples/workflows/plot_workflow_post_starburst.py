"""
Workflow: Post-Starburst (E+A) Galaxy Diagnosis
================================================

Demonstrates identifying post-starburst galaxies through model comparison.
A post-starburst has a truncated SFH with a recent burst followed by
quenching. When fit with a smooth tau model (incorrect), the fit
poorly recovers the truth. This shows how model misspecification
can bias SFH inference and why flexible models matter for
interpreting star formation histories.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_workflow_post_starburst_001.png
   :alt: plot_workflow_post_starburst
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    data_path,
    load_ssp,
)
from tengri.analysis.plotting import setup_style

setup_style()


# --- SSP data ---


ssp = load_ssp()


bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(bands, cache_dir=str(data_path("filters"))))

# --- True model: burst + quench (E+A) ---
# Recent starburst ~500 Myr ago, then sharp quench
spec_true = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.5),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),  # Burst 500 Myr ago
    sfh_tsnorm_width_gyr=Fixed(0.2),  # Sharp truncation
    sfh_tsnorm_skew=Fixed(0.8),  # Recent star formation peak
    sfh_tsnorm_trunc=Fixed(1.5),  # Rapid quench after
    met_logzsol=Fixed(-0.1),
    dust_tau_bc=Fixed(0.2),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)

model_true = SEDModel(spec_true, ssp, observation=obs)

# Generate mock photometry from post-starburst model
key = jax.random.PRNGKey(42)
true_params = {
    "sfh_tsnorm_log_peak_sfr": 1.5,
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
mock = model_true.mock(true_params, snr=20.0, key=key)

# --- Fit 1: correct model (tsnorm) ---
spec_correct = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(0.5, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.2, 2.0),
    sfh_tsnorm_width_gyr=Uniform(0.1, 1.0),
    sfh_tsnorm_skew=Uniform(-0.5, 2.0),
    sfh_tsnorm_trunc=Uniform(0.5, 5.0),
    met_logzsol=Fixed(-0.1),
    dust_tau_bc=Fixed(0.2),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model_correct = SEDModel(spec_correct, ssp, observation=obs)
fitter_c = Fitter(model_correct, data=mock.flux_obs, noise=mock.noise)
post_correct = fitter_c.run("map", optimizer="adam", n_steps=400, verbose=False)

# --- Fit 2: wrong model (delayed-tau, smooth) ---
spec_wrong = Parameters(
    sfh_dpl_log_peak_sfr=Uniform(0.5, 2.5),
    sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(0.1),
    met_logzsol=Fixed(-0.1),
    dust_tau_bc=Fixed(0.2),
    dust_tau_diff=Fixed(0.1),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)
model_wrong = SEDModel(spec_wrong, ssp, observation=obs)
fitter_w = Fitter(model_wrong, data=mock.flux_obs, noise=mock.noise)
post_wrong = fitter_w.run("map", optimizer="adam", n_steps=400, verbose=False)

# --- Plot: SFH comparison ---
sfh_true = model_true.predict_sfh(true_params)
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
    label="delayed-tau fit (wrong model)",
)

ax.axvline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
ax.text(0.5, ax.get_ylim()[1] * 0.9, "Quench epoch", fontsize=9, color="grey")

ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel("SFR [Msun/yr]", fontsize=12)
ax.set_title(
    "Post-Starburst (E+A) Galaxy: Model Misspecification Bias",
    fontsize=12,
    fontweight="bold",
)
ax.legend(fontsize=10, frameon=False, loc="upper right")
ax.set_ylim(bottom=0)

fig.tight_layout()
plt.savefig("plot_workflow_post_starburst.png", dpi=150, bbox_inches="tight")
plt.show()
