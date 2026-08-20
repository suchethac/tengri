"""
Age-Dust-Metallicity Degeneracy: Fisher Analysis
=================================================

The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS
5-band photometry alone cannot separately constrain age, dust, and
metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of
2–5×, quantifying the information gain from multiwavelength coverage.

Fisher Information Matrix is computed directly via JAX differentiation on the
public prediction surface: F = J^T C^{-1} J, where J is the Jacobian of model
predictions w.r.t. parameters and C^{-1} is the inverse noise covariance.

Reference: Fisher Information Matrix in parameter estimation; see
Conroy 2013 (ARA&A, 51, 393) for SED fitting context.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

# No cache_dir: tengri resolves each curve across its own data directories, so
# an example works from any working directory. This used to guess with a
# four-deep relative walk ending in a bare "data/filters" -- the literal #1486
# removed from the library for creating a stray cache beside whatever directory
# the caller started in. Its first candidate also selected the ten-curve copy
# that #1857 deleted, in preference to the 249 in data/filters/.
FILTER_SETS = {
    "SDSS (5)": ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    "+ NIR (8)": [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "2mass_j",
        "2mass_h",
        "2mass_ks",
    ],
    "+ MIR (10)": [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "2mass_j",
        "2mass_h",
        "2mass_ks",
        "wise_w1",
        "wise_w2",
    ],
}

fisher_params = ["met_logzsol", "dust_tau_bc", "dust_tau_diff"]
PARAM_LABELS = [r"$\log(Z/Z_\odot)$", r"$\tau_{\rm bc}$", r"$\tau_{\rm diff}$"]
COLORS_BAR = ["#4477AA", "#EE6677", "#228833"]

key = jax.random.PRNGKey(42)
true_params = {
    "sfh_tsnorm_log_total_mass": 10.5,
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.8,
    "dust_tau_diff": 0.4,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

sigmas = {}
first_failure: Exception | None = None
for fname, filters in FILTER_SETS.items():
    try:
        obs = tengri.Observation(photometry=tengri.Photometry.from_names(filters))
        mdl = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": tengri.FIXED},
            dust_attenuation={"law": "power_law", "type": "two_component", "all_params": tengri.FIXED},
            redshift=tengri.Fixed(0.1),
        )
        phot = jnp.abs(mdl.predict_photometry(true_params))
        noise = phot / 20.0

        # Compute Fisher Information Matrix using JAX jacobian.
        # F = J^T C^{-1} J, where J is the Jacobian and C^{-1} is inverse noise covariance.
        def predict_fn(params_dict, _mdl=mdl):
            """Return photometry for a parameter dictionary."""
            return _mdl.predict_photometry(params_dict)

        # Build an array of the free parameters in the order they appear in fisher_params.
        param_values = []
        for pname in fisher_params:
            param_values.append(true_params[pname])
        param_array = jnp.array(param_values)

        def forward_free_params(free_array):
            """Forward model taking only free parameters; reconstruct full params dict."""
            params_dict = true_params.copy()
            for i, pname in enumerate(fisher_params):
                params_dict[pname] = free_array[i]
            return predict_fn(params_dict)

        # Compute Jacobian of predictions w.r.t. free parameters.
        jac = jax.jacobian(forward_free_params)(param_array)  # shape: (n_bands, n_params)

        # Inverse noise covariance (diagonal).
        noise_var = noise**2
        c_inv_diag = 1.0 / noise_var

        # Fisher = J^T C^{-1} J
        fim = jnp.dot(jac.T, jac * c_inv_diag[:, jnp.newaxis])  # (n_params, n_params)

        # Cramér-Rao bound: sqrt(diag(F^{-1}))
        try:
            fim_inv = jnp.linalg.inv(fim)
            errs = np.array(np.sqrt(np.maximum(np.diag(fim_inv), 0)))
        except np.linalg.LinAlgError:
            # Singular matrix; use pseudoinverse as fallback.
            fim_inv = jnp.linalg.pinv(fim)
            errs = np.array(np.sqrt(np.maximum(np.diag(fim_inv), 0)))

        errs = np.where(np.isfinite(errs) & (errs > 0), errs, 5.0)
        sigmas[fname] = np.minimum(errs, 5.0)
    except Exception as e:
        if first_failure is None:
            first_failure = e
        print(f"[{fname}] skipped: {e}")

# This guard was already here and is the pattern the other examples now follow.
# It only gains the cause: "check filter availability" was a guess at why, and
# when the reason was something else (a dust-law KeyError, say) it sent the
# reader looking in the wrong place.
if not sigmas:
    raise RuntimeError(
        f"Fisher computation failed for every filter set. First failure: "
        f"{type(first_failure).__name__}: {first_failure}"
    ) from first_failure

x = np.arange(len(fisher_params))
width = 0.22
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (fname, sigma_arr) in enumerate(sigmas.items()):
    ax.bar(x + (i - 1) * width, sigma_arr, width, label=fname, color=COLORS_BAR[i], alpha=0.85)

ax.set_yscale("log")
ax.set_ylim(1e-3, 1e1)
ax.set_xticks(x)
ax.set_xticklabels(PARAM_LABELS, fontsize=10)
ax.set_ylabel(r"Cramér-Rao $1\sigma$ bound (log scale)")
ax.legend(fontsize=10, frameon=False)
fig.tight_layout()
plt.savefig("plot_fisher_degeneracy.png", dpi=150, bbox_inches="tight")
