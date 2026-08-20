"""
Autodiff gradients vs. finite-difference derivatives: diagnostic verification
==============================================================================

tengri is a differentiable JAX package. Every model gradient ∂L/∂θ computed via
`jax.grad()` should numerically match a central finite-difference approximation.
This diagnostic builds a star-forming model with several free parameters,
defines a chi-squared loss, and compares autodiff vs FD gradients for each
parameter. A mismatch (>1e-3) indicates a non-differentiable operation.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
jax.config.update("jax_enable_x64", True)

# Build model with multiple free parameters (DPL SFH, two-component dust) at a
# free redshift. Nebular is held fixed: every Cue parameter carries a Fixed
# registry default, so the ``FREE`` this line used to request never freed any of
# them — the gradient check has always run with nebular pinned, and now says so.
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]),
)
model = tengri.SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": "dpl", "all_params": tengri.FREE},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "all_params": tengri.FREE,
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    neb={"type": "cue", "all_params": tengri.FIXED},
    redshift=tengri.FREE,
)

# Generate synthetic observed photometry
key = jax.random.PRNGKey(42)
p_true = dict(model.spec.sample(key))
photometry_true = model.predict_photometry(p_true)
noise = jnp.ones_like(photometry_true) * 0.05 * photometry_true
photometry_obs = photometry_true + jax.random.normal(key, photometry_true.shape) * noise

# Use off-target parameters to create non-vanishing gradients
p_ref = dict(model.spec.sample(jax.random.PRNGKey(99)))
p_ref["sfh_dpl_log_total_mass"] += 2.0
p_ref["dust_tau_bc"] = jnp.clip(p_ref["dust_tau_bc"] + 1.5, 0, 2.0)
p_ref["met_logzsol"] = jnp.clip(p_ref["met_logzsol"] - 1.0, -2.0, 0.2)


# Objective: reduced chi-squared
def objective(params_dict):
    model_phot = model.predict_photometry(params_dict)
    residual = (model_phot - photometry_obs) / (noise + 1e-12)
    return jnp.sum(residual**2) / len(residual)


# Compute autodiff gradients
grad_autodiff = jax.grad(objective)(p_ref)
free_param_names = sorted(model.spec.free_params)

# Central finite-difference: eps ~ 1e-3 for float64 (avoids round-off)
eps = 1e-3
grad_fd_dict = {}
for param_name in free_param_names:
    p_plus = {**p_ref, param_name: p_ref[param_name] + eps}
    p_minus = {**p_ref, param_name: p_ref[param_name] - eps}
    grad_fd = (objective(p_plus) - objective(p_minus)) / (2.0 * eps)
    grad_fd_dict[param_name] = float(grad_fd)

# Compute relative errors
rel_errors = {}
worst_param, worst_error = None, 0.0
for param_name in free_param_names:
    g_auto, g_fd = float(grad_autodiff[param_name]), grad_fd_dict[param_name]
    rel_err = abs(g_auto - g_fd) / (abs(g_fd) + 1e-12)
    rel_errors[param_name] = rel_err
    if rel_err > worst_error:
        worst_error, worst_param = rel_err, param_name

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
log_rel_errors = np.array([np.log10(rel_errors[p]) for p in free_param_names])
colors = ["C1" if lre > np.log10(1e-3) else "C0" for lre in log_rel_errors]
ax.bar(range(len(free_param_names)), log_rel_errors, color=colors, width=0.6)
ax.axhline(np.log10(1e-3), color="red", ls="--", lw=1.5, label=r"$10^{-3}$ threshold")
ax.set_xticks(range(len(free_param_names)))
ax.set_xticklabels(free_param_names, rotation=45, ha="right")
ax.set_ylabel(r"$\log_{10}$(relative error)")
ax.set_xlabel("Parameter")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
plt.savefig("plot_diag_gradient_finite_difference.png", dpi=150, bbox_inches="tight")
