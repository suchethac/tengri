"""
JAX JIT Compilation: Eager vs Compiled Numerical Equivalence
=============================================================

Verifies that JIT-compiled predictions are bit-identical to eager-mode evaluations.
For ``predict_photometry`` and ``predict(params).lines``, we sample random parameter
sets and compare max relative difference between eager and JIT outputs. A value < 1e-10
confirms no spurious numerical divergence; > 1e-10 suggests platform-dependent
floating-point behavior.

Reference: JAX JIT compilation is semantically transparent and should not alter
floating-point results beyond ~1e-14 unit roundoff (IEEE 754 double precision).
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

key = jax.random.PRNGKey(42)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# Build a model with free SFH, fixed dust/metallicity/redshift
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FREE},
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.5, "tau_bc": 0.3},
    neb={"type": "cue", "all_params": tengri.FIXED},
    redshift=tengri.Fixed(0.1),
)

# Get baseline parameters (all fixed except SFH)
baseline = {
    "sfh_tsnorm_log_total_mass": 11.0,
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": 0.0,
    "met_alpha_fe": 0.0,
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.5,
    "dust_slope": -0.7,
    "dust_Rv": 3.1,
    "dust_bump_strength": 0.0,
    "dust_delta": 0.0,
    "dust_f_obscuration": 0.0,
    "redshift": 0.1,
}

free_names = model.spec.free_params
n_samples = 20

# ============================================================================
# Predict and compare photometry
# ============================================================================
f_phot = model.predict_photometry
f_phot_jit = jax.jit(f_phot)

diffs_phot = []
for _ in range(n_samples):
    key, subkey = jax.random.split(key)
    # Sample random SFH parameters
    param_vals = jax.random.uniform(subkey, shape=(len(free_names),))
    params = dict(baseline)
    params["sfh_tsnorm_log_total_mass"] = 9.0 + 4.0 * float(param_vals[0])
    params["sfh_tsnorm_peak_lbt_gyr"] = 0.1 + 11.9 * float(param_vals[1])
    params["sfh_tsnorm_width_gyr"] = 0.01 + 4.99 * float(param_vals[2])
    params["sfh_tsnorm_skew"] = -2.0 + 4.0 * float(param_vals[3])
    params["sfh_tsnorm_trunc"] = 1.0 + 12.8 * float(param_vals[4])

    phot_eager = f_phot(params)
    phot_jit = f_phot_jit(params)

    safe_eager = jnp.where(jnp.abs(phot_eager) > 1e-40, phot_eager, 1.0)
    rel_diff = jnp.abs(phot_jit - phot_eager) / jnp.abs(safe_eager)
    max_rel_diff = float(jnp.max(rel_diff))
    diffs_phot.append(max_rel_diff)

# ============================================================================
# Predict and compare emission lines (Cue model)
# ============================================================================
diffs_lines = []
_line_failure: Exception | None = None
# The emission lines are catalog properties, and ``predict_properties`` is the
# single JIT/vmap-safe surface for them (NAMING_CONTRACT §4b.5). Stack them into
# one array so the eager-vs-jit comparison below is a plain elementwise diff.
_LINE_NAMES = ("halpha", "hbeta", "oiii_5007", "nii_6584")
has_cue = all(name in model.available_properties for name in _LINE_NAMES)
if has_cue:

    def f_lines(p):
        q = model.predict_properties(p, names=_LINE_NAMES)
        return jnp.stack([jnp.asarray(q[n]) for n in _LINE_NAMES])

    f_lines_jit = jax.jit(f_lines)

    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        param_vals = jax.random.uniform(subkey, shape=(len(free_names),))
        params = dict(baseline)
        params["sfh_tsnorm_log_total_mass"] = 9.0 + 4.0 * float(param_vals[0])
        params["sfh_tsnorm_peak_lbt_gyr"] = 0.1 + 11.9 * float(param_vals[1])
        params["sfh_tsnorm_width_gyr"] = 0.01 + 4.99 * float(param_vals[2])
        params["sfh_tsnorm_skew"] = -2.0 + 4.0 * float(param_vals[3])
        params["sfh_tsnorm_trunc"] = 1.0 + 12.8 * float(param_vals[4])

        try:
            lines_eager = f_lines(params)
            lines_jit = f_lines_jit(params)

            # Handle dict or array output
            if isinstance(lines_eager, dict):
                fluxes_eager = lines_eager["flux"]
                fluxes_jit = lines_jit["flux"]
            else:
                fluxes_eager = jnp.asarray(lines_eager)
                fluxes_jit = jnp.asarray(lines_jit)

            safe_eager = jnp.where(jnp.abs(fluxes_eager) > 1e-40, fluxes_eager, 1.0)
            rel_diff = jnp.abs(fluxes_jit - fluxes_eager) / jnp.abs(safe_eager)
            max_rel_diff = float(jnp.max(rel_diff))
            diffs_lines.append(max_rel_diff)
        except Exception as e:
            if _line_failure is None:
                _line_failure = e

# `diffs_lines` empty is read below as "this build has no Cue lines" and the
# figure quietly drops to one panel. That reading is only true when `has_cue` is
# False. With Cue present and every sample failing, the same empty list hides a
# broken JIT path -- and this example exists precisely to detect broken JIT
# paths, so failing open here defeats its purpose.
if has_cue and not diffs_lines:
    raise RuntimeError(
        f"Cue lines are available but all {n_samples} eager-vs-jit comparisons "
        f"failed, so the line panel would be silently dropped. First failure: "
        f"{type(_line_failure).__name__}: {_line_failure}"
    ) from _line_failure

# ============================================================================
# Plot histograms
# ============================================================================
fig, axes = plt.subplots(1, 2 if diffs_lines else 1, figsize=(8.5 if diffs_lines else 5, 3.5))

if diffs_lines:
    axes_list = [axes[0], axes[1]]
else:
    axes_list = [axes]

# Photometry
ax = axes_list[0]
ax.hist(
    np.log10(np.array(diffs_phot) + 1e-20),
    bins=8,
    color="#EE6677",
    edgecolor="black",
    alpha=0.7,
)
ax.axvline(np.log10(1e-10), color="red", linestyle="--", linewidth=1.5, label="1e-10 threshold")
ax.set_xlabel(r"$\log_{10}(\max \Delta_{\rm rel})$")
ax.set_ylabel("Count")
ax.legend(fontsize=8, frameon=False)
ax.grid(axis="y", alpha=0.3)

# Emission lines (if available)
if diffs_lines:
    ax = axes_list[1]
    ax.hist(
        np.log10(np.array(diffs_lines) + 1e-20),
        bins=8,
        color="#228833",
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(
        np.log10(1e-10), color="red", linestyle="--", linewidth=1.5, label="1e-10 threshold"
    )
    ax.set_xlabel(r"$\log_{10}(\max \Delta_{\rm rel})$")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
plt.savefig("plot_diag_jit_concrete_identity.png", dpi=150, bbox_inches="tight")
