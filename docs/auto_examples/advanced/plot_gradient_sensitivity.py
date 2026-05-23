"""
Gradient Sensitivity Heatmap
=============================

Computes the Jacobian d(flux)/d(theta) of the forward model and displays
it as a heatmap showing which photometric bands are sensitive to which
physical parameters. Each column shows normalized sensitivity to one
parameter; dark blue/red indicates strong dependence.

Reference: Automatic differentiation via JAX enables exact gradients
for SED model validation and Fisher analysis (Conroy 2013, ARA&A, 51, 393).
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": tengri.FREE},
    met={"type": "fixed"},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.5, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)

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
free_names = model.spec.free_params


def photometry_from_array(param_array):
    """Map flat array to photometric fluxes."""
    params = {n: v for n, v in zip(free_names, param_array)}
    return model.predict_photometry(params)


param_array = jnp.array([fiducial[n] for n in free_names])
jacobian = jax.jacobian(photometry_from_array)(param_array)

J = np.array(jacobian)
J_norm = J / (np.abs(J).max(axis=0, keepdims=True) + 1e-30)

fig, ax = plt.subplots(figsize=(8, 4))
short_names = [n.replace("sfh_tsnorm_", "").replace("_", " ") for n in free_names]
im = ax.imshow(J_norm, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(free_names)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(bands)))
ax.set_yticklabels([b.replace("sdss_", "") for b in bands])
ax.set_ylim(-0.5, len(bands) - 0.5)
ax.set_xlabel("Parameter")
ax.set_ylabel("Band")
fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized sensitivity")
fig.tight_layout()
fig.savefig("plot_gradient_sensitivity.png", dpi=150, bbox_inches="tight")
