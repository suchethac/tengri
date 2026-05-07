"""
Fix Redshift to a Known Value
==============================

How do I fit a spectrum when redshift is known from spectroscopy? This recipe
shows how fixing redshift with Fixed() constrains other parameters more tightly
compared to letting it vary.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)
from tengri.analysis.plotting import setup_style

setup_style()


def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

# Known spectroscopic redshift
TRUE_REDSHIFT = 0.15

# --- Generate mock data at known redshift ---
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(bands))

true_spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Fixed(1.2),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.5),
    sfh_tsnorm_width_gyr=Fixed(1.8),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.1),
    dust_tau_diff=Fixed(0.25),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(TRUE_REDSHIFT),
    mean_sfh_type="tsnorm",
)
model_true = SEDModel(true_spec, ssp, observation=obs)

key = jax.random.PRNGKey(42)
true_params_dict = {
    "sfh_tsnorm_log_peak_sfr": 1.2,
    "sfh_tsnorm_peak_lbt_gyr": 2.5,
    "sfh_tsnorm_width_gyr": 1.8,
    "sfh_tsnorm_skew": 0.1,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.2,
    "dust_tau_bc": 0.1,
    "dust_tau_diff": 0.25,
    "dust_slope": -0.7,
    "redshift": TRUE_REDSHIFT,
}
mock = model_true.mock(true_params_dict, snr=25.0, key=key)

# --- Fit 1: Redshift FIXED (spectroscopy known) ---
spec_fixed_z = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(TRUE_REDSHIFT),  # ← Fixed from spectroscopy
    mean_sfh_type="tsnorm",
)
model_fixed = SEDModel(spec_fixed_z, ssp, observation=obs)
fitter_fixed = Fitter(model_fixed, data=mock.flux_obs, noise=mock.noise)
fitter_fixed.run("map", optimizer="adam", n_steps=200, verbose=False)
posterior_fixed = fitter_fixed.run(
    "vi",
    n_iterations=8,
    n_samples=3,
    n_posterior_samples=2000,
    verbose=False,
)

# --- Fit 2: Redshift FREE (photometry only) ---
spec_free_z = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Uniform(0.0, 0.5),  # ← Free to vary
    mean_sfh_type="tsnorm",
)
model_free = SEDModel(spec_free_z, ssp, observation=obs)
fitter_free = Fitter(model_free, data=mock.flux_obs, noise=mock.noise)
fitter_free.run("map", optimizer="adam", n_steps=200, verbose=False)
posterior_free = fitter_free.run(
    "vi",
    n_iterations=8,
    n_samples=3,
    n_posterior_samples=2000,
    verbose=False,
)

# --- Plot corner: Fixed vs Free redshift ---
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, wspace=0.3)
ax_fixed = fig.add_subplot(gs[0])
ax_free = fig.add_subplot(gs[1])

# Extract a subset of parameters for legibility
params_to_plot = [
    "sfh_tsnorm_log_peak_sfr",
    "sfh_tsnorm_peak_lbt_gyr",
    "met_logzsol",
    "dust_tau_diff",
]


def corner_subset(posterior, params, ax, title):
    """Simple 2D scatter projection onto first 2 params."""
    p1, p2 = params[0], params[1]
    if posterior.samples and p1 in posterior.samples and p2 in posterior.samples:
        ax.scatter(
            posterior.samples[p1][:500],
            posterior.samples[p2][:500],
            alpha=0.4,
            s=20,
            color="C0",
        )
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(title)


corner_subset(posterior_fixed, params_to_plot, ax_fixed, "Fixed redshift (spec known)")
corner_subset(posterior_free, params_to_plot, ax_free, "Free redshift (photo only)")

fig.suptitle("Impact of Redshift Prior: Fixed vs Free", fontsize=12, y=1.02)
plt.savefig("plot_recipe_specific_redshift.png", dpi=150, bbox_inches="tight")
plt.show()
