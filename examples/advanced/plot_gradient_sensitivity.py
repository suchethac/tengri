"""
Gradient Sensitivity Heatmap
=============================

Computes the Jacobian d(flux)/d(theta) of the forward model and displays
it as a heatmap showing which photometric bands are sensitive to which
physical parameters.
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fixed,
    Model,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
    setup_style,
)

setup_style()

# --- Data ---
SSP_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)
if not os.path.exists(SSP_PATH):
    sys.exit("SSP data not found — skipping")

ssp = load_ssp_data(SSP_PATH)
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
filters = load_filter_set(bands)

# --- Model ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = Model(spec, ssp, filters=filters)

# --- Fiducial point ---
fiducial = {
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.5,
    "dust_slope": -0.7,
    "redshift": 0.1,
}
free_names = spec.free_params
fixed_vals = spec.get_fixed_values()


def photometry_from_array(param_array):
    """Map flat array to photometric fluxes."""
    params = dict(fixed_vals)
    for i, name in enumerate(free_names):
        params[name] = param_array[i]
    return model.predict_photometry(params)


param_array = jnp.array([fiducial[n] for n in free_names])
jacobian = jax.jacobian(photometry_from_array)(param_array)  # (n_bands, n_params)

# --- Figure: heatmap ---
J = np.array(jacobian)
# Normalize each column (parameter) to unit max for visual clarity
J_norm = J / (np.abs(J).max(axis=0, keepdims=True) + 1e-30)

fig, ax = plt.subplots(figsize=(8, 4))
short_names = [n.replace("sfh_tsnorm_", "").replace("_", " ") for n in free_names]
im = ax.imshow(J_norm, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(free_names)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(bands)))
ax.set_yticklabels([b.replace("sdss_", "") for b in bands])
ax.set_xlabel("Parameter")
ax.set_ylabel("Band")
ax.set_title(r"Normalized Jacobian $\partial f_{\rm band} / \partial \theta$")
fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized sensitivity")
fig.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "gradient_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.show()
