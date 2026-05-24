"""
Fisher ellipses and parameter degeneracies
===========================================

The Fisher Information Matrix quantifies which parameter combinations are
well-constrained by data, and which are degenerate. Tengri's differentiable
forward model makes computing the exact Fisher matrix trivial — just apply
``jax.hessian`` to the likelihood.

This plot shows the classic age-dust degeneracy in galaxy SED fitting:
at fixed stellar population mass, older stars + more dust = same SED as
younger stars + less dust. The principal axes of the Fisher ellipse
(eigenvectors of the Hessian) reveal the least-constrained and
most-constrained linear combinations of parameters.

Three datasets (optical only, optical+NIR, panchromatic) show how
adding longer-wavelength data breaks the degeneracy: the ellipse becomes
rounder and smaller as constraints tighten.
"""

import os
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import recipes, FIXED
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*wNE.*")

# Load a star-forming galaxy SED; fit multiple filter sets to show
# how degeneracies break with multiwavelength data.

BARE = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Common fiducial parameters for fair comparison across filter sets
FIDUCIAL_PARAMS = {
    "sfh_dpl_alpha": 1.5,
    "sfh_dpl_beta": 2.0,
    "sfh_dpl_tau_gyr": 1.0,
    "sfh_dpl_log_peak_sfr": 0.5,
    "dust_calzetti_tau_bc": 0.3,
    "dust_calzetti_tau_diffuse": 0.1,
    "redshift": 0.1,
    "neb_logZ_gas": 0.0,
}

# Three observation configs: optical only, optical+NIR, panchromatic
OBSERVATIONS = [
    {
        "name": "Optical",
        "filters": ["SLOAN_SDSS_u", "SLOAN_SDSS_r", "GALEX_GALEX_NUV"],
        "color": "#1f77b4",
    },
    {
        "name": "Optical + NIR",
        "filters": [
            "SLOAN_SDSS_u",
            "SLOAN_SDSS_r",
            "GALEX_GALEX_NUV",
            "2MASS_2MASS_J",
        ],
        "color": "#2ca02c",
    },
    {
        "name": "Panchromatic",
        "filters": [
            "SLOAN_SDSS_u",
            "SLOAN_SDSS_r",
            "GALEX_GALEX_NUV",
            "2MASS_2MASS_J",
            "Spitzer_IRAC_I1",
        ],
        "color": "#d62728",
    },
]

fig, ax = plt.subplots(figsize=(8.5, 6.5))

for obs_config in OBSERVATIONS:
    filters = obs_config["filters"]

    # Build observation: use a subset of filters
    observation = tengri.Observation(
        photometry=tengri.Photometry.from_names(filters)
    )

    # Create a minimal model with just the degeneracy of interest
    recipe = recipes.star_forming_photometry()
    model = tengri.SEDModel.build(
        ssp_data=BARE,
        observation=observation,
        sfh=recipe["sfh"],
        dust=recipe["dust"],
        neb=recipe.get("neb", {"type": "cue", "*": FIXED}),
        redshift=recipe.get("redshift", tengri.Fixed(0.1)),
    )

    # Mock observation: generate synthetic data at the fiducial params
    params_fiducial = dict(FIDUCIAL_PARAMS)
    params_fiducial["neb_logZ_gas"] = 0.0
    params_fiducial["redshift"] = 0.1

    # Predict SED and add noise
    out = model.predict_photometry(params_fiducial)
    flux_fiducial = np.asarray(out)
    snr = 50.0  # S/N per filter
    noise = flux_fiducial / snr
    noise_inv = 1.0 / noise**2

    # ─ Build a likelihood and compute the Hessian at the fiducial point
    # This is the Killer Feature™: full autodiff through the SED forward model.

    def neg_log_likelihood(params_dict):
        """Negative log likelihood at parameter point."""
        out = model.predict_photometry(params_dict)
        flux_model = jnp.asarray(out)
        chi2 = jnp.sum((flux_model - flux_fiducial) ** 2 * noise_inv)
        return 0.5 * chi2

    # Extract the free parameters (log_peak_sfr and tau_bc are the
    # degeneracy pair; others held at fiducial for simplicity)
    free_names = ["sfh_dpl_log_peak_sfr", "dust_calzetti_tau_bc"]

    def likelihood_flat(flat_params):
        """Likelihood as a function of a flat parameter vector."""
        p = dict(params_fiducial)
        p["sfh_dpl_log_peak_sfr"] = flat_params[0]
        p["dust_calzetti_tau_bc"] = flat_params[1]
        return neg_log_likelihood(p)

    # Fiducial point in flat space
    flat_fiducial = jnp.array(
        [
            params_fiducial["sfh_dpl_log_peak_sfr"],
            params_fiducial["dust_calzetti_tau_bc"],
        ]
    )

    # Compute the Hessian (Fisher Information Matrix) at the fiducial
    hessian_fn = jax.hessian(likelihood_flat)
    fisher = hessian_fn(flat_fiducial)

    # Eigendecompose the Fisher matrix
    eigenvals, eigenvecs = jnp.linalg.eigh(fisher)

    # Compute 1-sigma error ellipse (inverse of Fisher gives covariance)
    cov = jnp.linalg.inv(fisher)
    marginal_errors = jnp.sqrt(jnp.diag(cov))

    # Plot the ellipse in the parameter plane
    angle_rad = jnp.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
    angle_deg = float(jnp.degrees(angle_rad))

    # Confidence level: 1-sigma → scale = 1; 2-sigma → scale = 2
    scale = 2.0  # 2-sigma ellipse (68% → 95%)
    width = 2.0 * scale * jnp.sqrt(1.0 / eigenvals[0])
    height = 2.0 * scale * jnp.sqrt(1.0 / eigenvals[1])

    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        xy=(
            float(flat_fiducial[0]),
            float(flat_fiducial[1]),
        ),
        width=float(width),
        height=float(height),
        angle=angle_deg,
        facecolor="none",
        edgecolor=obs_config["color"],
        linewidth=2.0,
        label=obs_config["name"],
    )
    ax.add_patch(ellipse)

ax.set_xlim(-0.5, 2.0)
ax.set_ylim(-0.1, 0.8)
ax.set_xlabel(r"$\log_{10}(\dot{M}_* / M_\odot \, \mathrm{yr}^{-1})$", fontsize=10)
ax.set_ylabel(r"Dust optical depth (Calzetti) $\tau_\mathrm{BC}$", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3, linestyle="--")

ax.text(
    0.05,
    0.95,
    "Fisher Information Ellipses (2σ)",
    transform=ax.transAxes,
    fontsize=11,
    weight="bold",
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

fig.tight_layout()
output_path = os.path.join(
    os.path.dirname(__file__), "plot_gradient_degeneracy_direction.png"
)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
