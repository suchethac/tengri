"""
Age-metallicity color degeneracy in SDSS colors
==================================================

Young, metal-rich and old, metal-poor stellar populations can produce
similar colors — a fundamental degeneracy in stellar population
inference. This example builds a 2D grid of single-burst SSP-like models
varying age (log10(t/Gyr) = -2 to 1.1) and metallicity (log10(Z/Zsun) = -2 to 0.4),
then plots three SDSS broadband colors (u − r, g − r, NUV − r) as
pcolormesh grids to visualize the degeneracy.

Each (age, Z) point uses a narrow Gaussian-like SFH (tsnorm)
centered at the appropriate lookback time, fixed dust extinction,
and redshift z = 0.05 to avoid numerical issues at z = 0.
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

# Age and metallicity grids
LOG10_AGES_GYR = np.linspace(-2.0, 1.1, 13)  # log10(t/Gyr)
AGES_GYR = 10.0**LOG10_AGES_GYR
MET_LOGZSOL = np.linspace(-2.0, 0.4, 10)

# Filter setup: NUV, u, g, r
BANDS = ["galex_nuv", "sdss_u", "sdss_g", "sdss_r"]
COLORS_TO_PLOT = [
    ("u - r", 1, 3),
    ("g - r", 2, 3),
    ("NUV - r", 0, 3),
]

obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))

# Build model: single SSP-like burst, no dust, varying age and Z
ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": 1.0,  # Will override per grid point
        "width_gyr": 0.05,
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.05),
)

# Sample baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Pre-allocate grids for each color
color_grids = {name: np.empty((len(AGES_GYR), len(MET_LOGZSOL))) for name, _, _ in COLORS_TO_PLOT}

# Loop over age and metallicity
for i, age_gyr in enumerate(AGES_GYR):
    # Ensure age is within valid range (0.03 Gyr to universe age)
    age_clamped = np.clip(age_gyr, 0.03, 13.0)
    for j, met_logzsol in enumerate(MET_LOGZSOL):
        # Build parameter dict for this (age, Z)
        p = {
            **baseline,
            "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age_clamped),
            "met_logzsol": jnp.float64(met_logzsol),
        }
        # Predict photometry and compute colors
        flux = np.asarray(model.predict_photometry(p))
        # Protect against log of zero by ensuring positive flux
        flux = np.where(flux > 0, flux, 1e-30)
        for name, idx_1, idx_2 in COLORS_TO_PLOT:
            mag_1 = -2.5 * np.log10(flux[idx_1])
            mag_2 = -2.5 * np.log10(flux[idx_2])
            color_grids[name][i, j] = mag_1 - mag_2

# Create 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, (color_name, _, _) in zip(axes, COLORS_TO_PLOT):
    color_data = color_grids[color_name]

    # pcolormesh: X = metallicity, Y = age
    im = ax.pcolormesh(
        MET_LOGZSOL,
        LOG10_AGES_GYR,
        color_data,
        cmap="viridis",
        shading="auto",
    )

    ax.set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
    if ax is axes[0]:
        ax.set_ylabel(r"$\log_{10}(t/\mathrm{Gyr})$")

    # Colorbar for each panel
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{color_name} [mag]", fontsize=9)

fig.tight_layout()
plt.savefig("plot_age_metallicity_color_grid.png", dpi=150, bbox_inches="tight")
