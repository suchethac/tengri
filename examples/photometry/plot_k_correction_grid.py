"""
K-corrections as a function of redshift for different SED types
================================================================

How do K-corrections vary with redshift for different galaxy populations?
K-corrections quantify the shift in filter response as galaxies move
to higher redshifts: K(z) = −2.5 log₁₀[(1+z) × F_ν(z) / F_ν(0)] for a
fixed rest-frame filter. We compute K(z) for the SDSS r-band across
four galaxy types — young star-forming, old star-forming, red-sequence
elliptical, and post-starburst — from z = 0.01 to z = 2.0. This
illustrates why stellar mass measurements require careful K-corrections
at high redshift and why colour-matched template sets dominate
photometric redshift algorithms.

Reference: Hogg (1999) on K-correction formalism; Blanton & Roweis (2007)
on SDSS template K-corrections.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_k_correction_grid_001.png
   :alt: plot_k_correction_grid
   :class: sphx-glr-single-img

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

# sphinx_gallery_thumbnail_number = 1

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

import tengri
from tengri.analysis.plotting import setup_style

setup_style()


def _find_filters():
    """Find filter cache directory in standard locations."""
    for p in [
        Path("data/filters"),
        Path("../data/filters"),
        Path("../../data/filters"),
        Path("../../../data/filters"),
    ]:
        if p.exists():
            return str(p)
    return "data/filters"


# Load SSP
ssp = tengri.load_ssp()

# Locate filter cache
filter_dir = _find_filters()

# Observation: SDSS r-band only
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_r"], cache_dir=filter_dir)
)


def _build_galaxy(sfh_config, dust_config, label):
    """Build a model for one galaxy type."""
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh=sfh_config,
        dust=dust_config,
        redshift=tengri.FIXED,
    )


# Define four SED types
GALAXY_TYPES = [
    (
        "Young star-forming",
        {
            "type": "tsnorm",
            "*": tengri.FIXED,
            "log_peak_sfr": 1.0,  # 10 Msun/yr
            "peak_lbt_gyr": 0.2,  # age ≈ 0.2 Gyr
            "width_gyr": 0.15,
            "skew": 0.0,
            "trunc": 13.0,
        },
        {
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.4,  # moderate dust
            "tau_bc": 0.5,
            "slope": -0.7,
        },
        "#1f77b4",  # blue
    ),
    (
        "Old star-forming",
        {
            "type": "tsnorm",
            "*": tengri.FIXED,
            "log_peak_sfr": 1.0,
            "peak_lbt_gyr": 5.0,  # age ≈ 5 Gyr
            "width_gyr": 1.0,
            "skew": 0.0,
            "trunc": 13.0,
        },
        {
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.15,  # lighter dust
            "tau_bc": 0.2,
            "slope": -0.7,
        },
        "#ff7f0e",  # orange
    ),
    (
        "Red-sequence elliptical",
        {
            "type": "tsnorm",
            "*": tengri.FIXED,
            "log_peak_sfr": 2.0,  # 100 Msun/yr
            "peak_lbt_gyr": 10.0,  # age ≈ 10 Gyr (old starburst)
            "width_gyr": 0.5,
            "skew": 0.0,
            "trunc": 13.0,
        },
        {
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.05,  # minimal dust
            "tau_bc": 0.0,
            "slope": -0.7,
        },
        "#2ca02c",  # green
    ),
    (
        "Post-starburst",
        {
            "type": "tsnorm",
            "*": tengri.FIXED,
            "log_peak_sfr": 2.0,
            "peak_lbt_gyr": 1.5,  # intermediate age
            "width_gyr": 0.3,
            "skew": 0.0,
            "trunc": 13.0,
        },
        {
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.2,
            "tau_bc": 0.3,
            "slope": -0.7,
        },
        "#d62728",  # red
    ),
]

# Redshift grid
z_grid = np.logspace(np.log10(0.01), np.log10(2.0), 40)

fig, ax = plt.subplots(figsize=(7.5, 5.0))

for galaxy_label, sfh_cfg, dust_cfg, color in GALAXY_TYPES:
    # Build model at z = 0 reference
    model_z0 = _build_galaxy(sfh_cfg, dust_cfg, galaxy_label)
    baseline_params = dict(model_z0.spec.sample(jax.random.PRNGKey(0)))

    # Predict rest-frame photometry at z=0
    f_rest = np.asarray(model_z0.predict_photometry(baseline_params))[0]

    # K-correction as function of redshift
    k_corr = np.empty_like(z_grid)

    for i, z in enumerate(z_grid):
        # Build model at redshift z
        model_z = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh=sfh_cfg,
            dust=dust_cfg,
            redshift=tengri.Fixed(float(z)),
        )

        # Predict observed-frame photometry
        params = {**baseline_params, "redshift": float(z)}
        f_obs = np.asarray(model_z.predict_photometry(params))[0]

        # K(z) = -2.5 * log10[(1+z) * F_obs / F_rest]
        k_corr[i] = -2.5 * np.log10((1.0 + z) * f_obs / f_rest)

    ax.semilogx(z_grid, k_corr, color=color, lw=1.8, label=galaxy_label, zorder=3)

ax.set_xlabel(r"Redshift $z$", fontsize=11)
ax.set_ylabel(r"$K(z)$ [mag]", fontsize=11)
ax.set_xlim(0.008, 2.5)
ax.legend(frameon=False, fontsize=10, loc="best")
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()

# Save to script directory
script_dir = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
plt.savefig(
    str(script_dir / "plot_k_correction_grid.png"), dpi=150, bbox_inches="tight"
)
plt.close()
