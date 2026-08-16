"""Starter script: fit one galaxy with tengri, end-to-end.

Copy this into your analysis folder and edit the marked blocks. It is meant
to be LIFTED — everything a typical astronomer needs lives in one file.

Usage
-----
    python scripts/starter_fit.py

What it does
------------
1. Loads an SSP grid.
2. Loads observed photometry from hard-coded arrays (replace with your own).
3. Defines a smooth 7-D SFH model with sensible priors.
4. Runs NUTS (gold-standard posterior sampler).
5. Prints a posterior summary and saves a corner + SED-fit PNG.

To adapt
--------
- Swap the SSP path for the grid you downloaded.
- Replace ``FILTER_NAMES``, ``flux_obs``, ``flux_noise``, ``redshift`` with your data.
- Narrow/broaden priors based on what you know about the galaxy.
- Switch the inference method: ``map`` (point estimate, fast),
  ``laplace`` (approximate posterior, fast), ``pathfinder`` (approximate,
  good NUTS initializer), ``mcmc_nuts`` (gold-standard), ``mcmc_raytrace``
  (exact MCMC, scales past D = 30), ``evidence`` (Bayesian evidence).
"""

import os

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Gaussian,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)
from tengri.analysis.plotting import (
    plot_sed_fit,
    safe_corner,
    setup_style,
)

setup_style()


# ──────────────────────────────────────────────────────────────────────
# 1. Observed data  (EDIT THIS BLOCK)
# ──────────────────────────────────────────────────────────────────────

FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

# flux in [erg/s/cm^2/Hz], redshift dimensionless
flux_obs = np.array([1.0e-28, 2.0e-28, 3.0e-28, 2.5e-28, 2.0e-28])
flux_noise = np.array([1.0e-29] * 5)
redshift = 0.1


# ──────────────────────────────────────────────────────────────────────
# 2. SSP grid  (EDIT PATH)
# ──────────────────────────────────────────────────────────────────────

SSP_PATH = os.environ.get(
    "TENGRI_SSP",
    "data/ssp_fsps_v3.2.h5",
)
ssp = load_ssp_data(SSP_PATH)


# ──────────────────────────────────────────────────────────────────────
# 3. Model + priors
# ──────────────────────────────────────────────────────────────────────

obs = Observation(photometry=Photometry.from_names(FILTER_NAMES))

spec = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1.0, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5.0),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0.0, 4.0),
    redshift=redshift,
)

model = SEDModel(spec, ssp, observation=obs)


# ──────────────────────────────────────────────────────────────────────
# 4. Fit
# ──────────────────────────────────────────────────────────────────────

fitter = Fitter(model, flux_obs, flux_noise)
result = fitter.run("mcmc_nuts", n_warmup=500, n_samples=1000)

print(result.summary_table())


# ──────────────────────────────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────────────────────────────

os.makedirs("starter_outputs", exist_ok=True)

fig_corner, _ = safe_corner(result, truths=None)
fig_corner.savefig("starter_outputs/corner.png", dpi=200, bbox_inches="tight")
plt.close(fig_corner)

fig_sed, _ = plot_sed_fit(model, result, flux_obs, flux_noise)
fig_sed.savefig("starter_outputs/sed_fit.png", dpi=200, bbox_inches="tight")
plt.close(fig_sed)

print("\nSaved: starter_outputs/corner.png, starter_outputs/sed_fit.png")
