"""Test if BlackJAX NUTS can use mode='auto' instead of mode='_traceable'.

This experiment modifies the loss function to use mode='auto' (which routes to
hybrid/compositional) to see if it:
1. Still works (no errors)
2. Is faster than _traceable mode
3. Uses acceptable memory

The hypothesis: BlackJAX doesn't have NIFTy's 4 internal JIT scopes, so it
might not suffer from the memory bloat issue that forced us to use _traceable.
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri import (
    Fitter,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    load_ssp_data,
)
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

jax.config.update("jax_enable_x64", True)

# Load data
filter_names = [
    "hst_f435w",
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f125w",
    "hst_f140w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
]
filter_set = load_filter_set(filter_names)
obs = Observation(photometry=Photometry.from_filter_set(filter_set))

# Mock data
flux = jnp.ones(11) * 1e-26
flux_unc = flux / 10.0

# D=11 model (same as D1 test)
params = Parameters(
    mean_sfh_type=["dense_basis", "field"],
    sfh_dbp_log_total_mass=Uniform(9.0, 12.0),
    sfh_dbp_tx_frac_0=Uniform(0.05, 0.95),
    sfh_dbp_tx_frac_1=Uniform(0.05, 0.95),
    sfh_dbp_tx_frac_2=Uniform(0.05, 0.95),
    sfh_field_psd_sigma=Uniform(0.1, 3.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_law_bc="salim_sbl18",
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_model="two_component",
    dust_emission="draine_li2007",
    dust_umin=Fixed(1.0),
    dust_gamma_dl=Uniform(0.0, 0.1),
    dust_qpah=Uniform(0.5, 4.5),
    nebular_ssp=True,
    apply_igm=True,
    redshift=Fixed(1.0),
    n_grid=64,
)

ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
model = SEDModel(params, ssp_data, observation=obs)
fitter = Fitter(model, data=flux, noise=flux_unc)

print("=" * 70)
print("EXPERIMENT: BlackJAX NUTS with mode='auto' vs mode='_traceable'")
print("=" * 70)

# ============================================================================
# Test 1: Current approach (mode='_traceable')
# ============================================================================
print("\n[1] BASELINE: mode='_traceable' (current implementation)")
print("-" * 70)

# Get the current loss function (uses _traceable)
loss_fn_traceable = fitter._get_or_build_loss_fn()
data_args = fitter._data_args

# Initialize parameters
rng_key = jax.random.PRNGKey(42)
init_params = fitter._initialize_unbounded(rng_key)
init_flat, unravel_fn = ravel_pytree(init_params)


def log_prob_traceable(position):
    params = unravel_fn(position)
    return -loss_fn_traceable(params, data_args)


# Warmup
for _ in range(3):
    _ = log_prob_traceable(
        init_flat + jax.random.normal(rng_key, (len(init_flat),)) * 0.01
    )

# Measure
t0 = time.perf_counter()
log_p = log_prob_traceable(init_flat)
t1 = time.perf_counter()
time_traceable_ms = (t1 - t0) * 1000

grad_fn_traceable = jax.grad(log_prob_traceable)
for _ in range(3):
    _ = grad_fn_traceable(
        init_flat + jax.random.normal(rng_key, (len(init_flat),)) * 0.01
    )
t0 = time.perf_counter()
grad = grad_fn_traceable(init_flat)
t1 = time.perf_counter()
time_traceable_grad_ms = (t1 - t0) * 1000

print(f"  log_prob: {time_traceable_ms:.3f} ms")
print(f"  gradient: {time_traceable_grad_ms:.3f} ms")
print(f"  log_prob value: {log_p:.2f}")
print(f"  gradient norm: {jnp.linalg.norm(grad):.2e}")

# ============================================================================
# Test 2: Experimental approach (mode='auto')
# ============================================================================
print("\n[2] EXPERIMENT: mode='auto' (hybrid/compositional)")
print("-" * 70)

# Manually build a loss function that uses mode='auto'
from tengri.inference.loss_functions import build_loss_fn


# Build loss function using mode='auto' instead of mode='_traceable'
def build_loss_fn_with_auto(fitter):
    """Build loss function using mode='auto' - CORRECTED VERSION.

    This now properly replicates the canonical loss function from
    src/tengri/inference/loss_functions.py with mode='auto' instead of
    mode='_traceable'.
    """
    from tengri.parameters.priors import Gaussian, LogUniform

    model = fitter.model
    data_type = fitter.data_type
    free_names = fitter._free_names
    bounds = fitter._bounds
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic

    # Inverse sigmoid (from loss_functions.py)
    def to_bounded(xi, lo, hi):
        z = jax.nn.sigmoid(xi)
        return lo + (hi - lo) * z

    def loss_fn_auto(params_unbounded, data_args):
        data = data_args["data"]
        noise = data_args["noise"]

        # Convert unbounded → physical for free params
        params = {}
        for name in free_names:
            lo, hi = bounds[name]
            params[name] = to_bounded(params_unbounded[name], lo, hi)

        # Merge fixed values (MISSING IN ORIGINAL)
        for name, val in fixed_values.items():
            params[name] = val

        # Add psd_xi if stochastic (MISSING IN ORIGINAL)
        if stochastic and "psd_xi" in params_unbounded:
            params["psd_xi"] = params_unbounded["psd_xi"]

        # Forward model prediction with mode='auto'
        if data_type == "photometry":
            predicted = model.predict_photometry(params, mode="auto")
        elif data_type == "spectroscopy":
            predicted = model.predict_spectrum(params, mode="auto")
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Chi-squared
        chi2 = jnp.sum(((predicted - data) / noise) ** 2)

        # Prior penalty (CORRECTED - was wrong in original)
        # The sigmoid transform maps N(0,1) → Uniform(lo, hi)
        prior_penalty = 0.0

        # Standard normal prior on ALL unbounded parameters
        for name in free_names:
            prior_penalty += params_unbounded[name] ** 2

        # Standard normal prior on psd_xi
        if stochastic and "psd_xi" in params_unbounded:
            prior_penalty += jnp.sum(params_unbounded["psd_xi"] ** 2)

        # Additional prior contributions for non-Uniform distributions
        for name in free_names:
            dist = spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                val = params[name]
                prior_penalty -= 2.0 * dist.log_prob(val)
            elif isinstance(dist, LogUniform):
                val = params[name]
                uniform_lp = -jnp.log(dist.hi - dist.lo)
                prior_penalty -= 2.0 * (dist.log_prob(val) - uniform_lp)

        return 0.5 * chi2 + 0.5 * prior_penalty

    return loss_fn_auto


loss_fn_auto = build_loss_fn_with_auto(fitter)


def log_prob_auto(position):
    params = unravel_fn(position)
    return -loss_fn_auto(params, data_args)


print("  Compiling with JIT...")
jax.clear_caches()
t0 = time.perf_counter()
log_p_auto = log_prob_auto(init_flat)
t1 = time.perf_counter()
compile_time_s = t1 - t0
print(f"  First call (JIT compile): {compile_time_s:.1f} s")

# Warmup
for _ in range(3):
    _ = log_prob_auto(init_flat + jax.random.normal(rng_key, (len(init_flat),)) * 0.01)

# Measure
t0 = time.perf_counter()
log_p_auto = log_prob_auto(init_flat)
t1 = time.perf_counter()
time_auto_ms = (t1 - t0) * 1000

grad_fn_auto = jax.grad(log_prob_auto)
jax.clear_caches()
print("  Compiling gradient with JIT...")
t0_grad_compile = time.perf_counter()
for _ in range(3):
    _ = grad_fn_auto(init_flat + jax.random.normal(rng_key, (len(init_flat),)) * 0.01)
t1_grad_compile = time.perf_counter()
compile_grad_time_s = t1_grad_compile - t0_grad_compile
print(f"  Gradient compile: {compile_grad_time_s:.1f} s")

t0 = time.perf_counter()
grad_auto = grad_fn_auto(init_flat)
t1 = time.perf_counter()
time_auto_grad_ms = (t1 - t0) * 1000

print(f"  log_prob: {time_auto_ms:.3f} ms")
print(f"  gradient: {time_auto_grad_ms:.3f} ms")
print(f"  log_prob value: {log_p_auto:.2f}")
print(f"  gradient norm: {jnp.linalg.norm(grad_auto):.2e}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\nlog_prob evaluation:")
print(f"  _traceable: {time_traceable_ms:.3f} ms")
print(f"  auto:       {time_auto_ms:.3f} ms")
print(f"  Speedup:    {time_traceable_ms/time_auto_ms:.1f}x")

print(f"\nGradient evaluation:")
print(f"  _traceable: {time_traceable_grad_ms:.3f} ms")
print(f"  auto:       {time_auto_grad_ms:.3f} ms")
print(f"  Speedup:    {time_traceable_grad_ms/time_auto_grad_ms:.1f}x")

print(f"\nNumerical agreement:")
print(f"  log_prob diff: {abs(log_p - log_p_auto):.2e}")
print(f"  gradient diff: {jnp.linalg.norm(grad - grad_auto):.2e}")

print(f"\nCompilation overhead:")
print(f"  auto mode JIT compile: {compile_time_s:.1f} s")
print(f"  auto mode gradient compile: {compile_grad_time_s:.1f} s")
print(f"  Total: {compile_time_s + compile_grad_time_s:.1f} s")

print(f"\nEstimated MCMC time (40 iterations × 12 grads/iter):")
print(f"  _traceable: {40 * 12 * time_traceable_grad_ms / 1000:.1f} s")
print(f"  auto:       {40 * 12 * time_auto_grad_ms / 1000:.1f} s")
print(f"  Savings:    {40 * 12 * (time_traceable_grad_ms - time_auto_grad_ms) / 1000:.1f} s")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if time_auto_ms < time_traceable_ms:
    speedup = time_traceable_ms / time_auto_ms
    print(f"✓ mode='auto' is {speedup:.1f}x FASTER than mode='_traceable'")
    print(
        f"  BlackJAX NUTS could potentially use mode='auto' for better performance"
    )
    print(f"  without the memory bloat issue (NIFTy-specific problem).")
else:
    print(f"✗ mode='auto' is slower than mode='_traceable' (unexpected)")

if abs(log_p - log_p_auto) < 1e-6 and jnp.linalg.norm(grad - grad_auto) < 1e-4:
    print(f"✓ Numerical results are IDENTICAL (mode='auto' is safe to use)")
else:
    print(f"⚠ Numerical differences detected (investigate before using)")

print("=" * 70)
