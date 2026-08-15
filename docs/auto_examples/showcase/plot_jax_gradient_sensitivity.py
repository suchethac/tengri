"""
Automatic differentiation: parameter sensitivities via jax.grad
===============================================================

Compute logarithmic sensitivities ∂(log F) / ∂(log θ) for each photometric
band. Finite-difference methods (∂F/∂θ ≈ [F(θ+δ) − F(θ−δ)] / (2δ)) are slow
and fragile; JAX autodiff computes exact sensitivities via one forward and
reverse pass per parameter.

References: Bradbury+2018 (JAX); Hearin+2023 (DSPS).
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

# Build a baseline model: star-forming galaxy at z=0 with 5 SDSS filters
ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)

# Construct a realistic model with free parameters
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "dpl",
        "all_params": tengri.FREE,
        "alpha": tengri.Fixed(0.5),  # Fix early slope; vary peak SFR, beta, tau
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "all_params": tengri.FREE,  # Allow dust parameters to vary
    },
    redshift=tengri.Fixed(0.0),
)

# Sample a baseline set of parameters
key = jax.random.PRNGKey(42)
baseline_params = dict(model.spec.sample(key))

# Four parameters of interest: log_total_mass, metallicity, dust tau, age
# We'll compute ∂(log F_ν) / ∂(log θ) for each band and parameter
param_names = [
    "sfh_dpl_log_total_mass",
    "met_logzsol",
    "dust_tau_diff",
    "sfh_dpl_tau_gyr",
]
param_labels = [
    r"$\log_{10}({\rm SFR}_{\rm peak})$ [M$_\odot$/yr]",
    r"$\log_{10}(Z/Z_\odot)$",
    r"$\tau_{\rm diff}$ [optical depth]",
    r"$\tau_{\rm age}$ [Gyr]",
]

band_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


# Function to compute log-log sensitivity: ∂(log F) / ∂(log θ)
def compute_log_sensitivities(params):
    """
    Compute logarithmic elasticity for each (band, parameter) pair.
    Uses jax.grad to differentiate log(photometric flux) w.r.t. log(parameter).

    Returns:
      sensitivity_matrix : ndarray, shape (n_bands, n_params)
        Normalized sensitivities (dimensionless elasticity)
    """
    sensitivities = []

    for param_key in param_names:
        param_val = baseline_params[param_key]

        # Convert params dict to a JAX-compatible form for autodiff
        # Create a function that accepts a single parameter value and returns flux
        def flux_array_for_param(param_value, param_key=param_key):
            """Predict photometric flux for a given parameter value (scalar JAX array)."""
            test_params = dict(baseline_params)
            # Ensure the parameter is stored as a JAX array
            test_params[param_key] = jnp.asarray(param_value)
            # predict_photometry returns a JAX array directly
            fluxes = model.predict_photometry(test_params)
            return jnp.asarray(fluxes)

        # Compute gradient of each band's log-flux w.r.t. log parameter
        # Ensure the parameter value is positive before taking log
        param_val_float = float(param_val)
        if param_val_float <= 0:
            # Skip negative or zero parameters (e.g., logzsol can be negative)
            # Use absolute value and adjust sign
            log_param_baseline = np.log(np.abs(param_val_float) + 1e-6)
        else:
            log_param_baseline = np.log(param_val_float)

        # For each band, the sensitivity is the gradient of log F w.r.t. log θ
        band_sensitivities = []
        for band_idx in range(len(band_names)):

            def log_flux_band(log_param, band_idx=band_idx):
                """Log-flux of a single band as function of log parameter."""
                param = jnp.exp(log_param)
                fluxes = flux_array_for_param(param)
                return jnp.log(fluxes[band_idx])

            grad_fn = jax.grad(log_flux_band)
            sens = float(grad_fn(jnp.asarray(log_param_baseline)))
            band_sensitivities.append(sens)

        sensitivities.append(band_sensitivities)

    return np.array(sensitivities).T  # shape (n_bands, n_params)


# Compute sensitivities
sensitivity_matrix = compute_log_sensitivities(baseline_params)

# Normalize each column to [-1, 1] for visibility
sensitivity_normalized = sensitivity_matrix.copy()
for col in range(sensitivity_normalized.shape[1]):
    col_max = np.max(np.abs(sensitivity_normalized[:, col]))
    if col_max > 0:
        sensitivity_normalized[:, col] /= col_max

# Plot heatmap
fig, ax = plt.subplots(figsize=(6.5, 4.0))

im = ax.imshow(
    sensitivity_normalized,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
)

# Tick labels
ax.set_xticks(np.arange(len(param_labels)))
ax.set_yticks(np.arange(len(band_names)))
ax.set_xticklabels(param_labels, fontsize=9)
ax.set_yticklabels(band_names, fontsize=9)

# Rotate x labels for readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Add colorbar
cbar = fig.colorbar(im, ax=ax, label=r"$\partial \log F / \partial \log \theta$ (normalized)")

# Title and labels
ax.set_ylabel("Photometric Band", fontsize=10)

fig.tight_layout()
plt.savefig("plot_jax_gradient_sensitivity.png", dpi=150, bbox_inches="tight")
