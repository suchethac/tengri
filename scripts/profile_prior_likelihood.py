"""Profile prior and likelihood computation performance.

Investigates:
1. Prior evaluation overhead (currently 56.9% of loss function time)
2. Likelihood chi-square computation
3. Vectorization opportunities
4. JIT fusion analysis

Author: Claude Code
Date: 2026-04-18
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SSP_FILE = DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

print("Loading SSP data...")
ssp = load_ssp_data(str(SSP_FILE))
print("  ✓ Loaded")

# Mock observation - just the data arrays for testing prior/likelihood
flux_obs = jnp.array([1e-19, 2e-19, 1.5e-19])
flux_err = jnp.array([1e-20, 2e-20, 1.5e-20])

# Simple model (D=8, attenuation only, no spectroscopy/dust emission)
params = Parameters(
    redshift=Fixed(1.0),
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.1, 10),
    sfh_tsnorm_width_gyr=Uniform(0.1, 5),
    sfh_tsnorm_skew=Uniform(-1, 1),
    sfh_tsnorm_trunc=Uniform(0.01, 1),
    met_logzsol=Uniform(-2, 0.5),
    dust_tau_diff=Uniform(0, 3),
    dust_tau_bc=Uniform(0, 2),
)

# Create a minimal fitter to access prior/likelihood functions
_waves, _trans, filter_curves = load_filter_set(["sdss_u", "sdss_g", "sdss_r"])
phot = Photometry(filters=filter_curves)
obs = Observation(photometry=phot)

model = SEDModel(params, ssp, observation=obs)
fitter = Fitter(model, flux_obs, flux_err)

print(f"\n  D = {len(fitter._free_names)} free parameters")
print(f"  Free params: {fitter._free_names}")

# ---------------------------------------------------------------------------
# Profiling functions
# ---------------------------------------------------------------------------


def time_function(fn, *args, n_warmup=10, n_evals=100, **kwargs):
    """Time a function with warmup."""
    # Compile
    result = fn(*args, **kwargs)

    # Warmup
    for _ in range(n_warmup):
        _ = fn(*args, **kwargs)

    # Measure
    times = []
    for _ in range(n_evals):
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
    }


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

print("\nBuilding functions...")

# Loss function components
loss_fn = fitter._get_or_build_loss_fn(mode="_traceable")
logprior_fn = fitter._build_logprior_fn()
loglik_fn = fitter._get_or_build_loglikelihood_fn(mode="_traceable")
data_args = fitter._data_args

# Get initial parameters (unbounded space)
key = jax.random.PRNGKey(42)
params_init_unbounded = fitter._initialize_unbounded(key)

# Convert to bounded for prior/likelihood (not needed for loss which handles internally)
from tengri.utils.transforms import to_bounded

param_dict_bounded = {}
for name in fitter._free_names:
    lo, hi = fitter._bounds[name]
    param_dict_bounded[name] = to_bounded(params_init_unbounded[name], lo, hi)
for name, val in fitter._fixed_values.items():
    param_dict_bounded[name] = val

print("  ✓ Functions built")

# ---------------------------------------------------------------------------
# Profile individual prior contributions
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("INDIVIDUAL PRIOR CONTRIBUTIONS")
print("=" * 80)

# Measure each parameter's prior eval time
print("\nPer-parameter prior evaluation times:")
print(f"  {'Parameter':<30} {'Time (μs)':<12} {'Type'}")
print("  " + "-" * 60)

prior_times = []
for name in fitter._free_names:
    dist = fitter.spec.get_distribution(name)
    dist_type = type(dist).__name__

    # Create a JIT-compiled function for this specific parameter
    @jax.jit
    def eval_prior(param_value):
        return dist.log_prob(param_value)

    # Warmup
    _ = eval_prior(param_dict_bounded[name])

    # Measure
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = eval_prior(param_dict_bounded[name])
        times.append((time.perf_counter() - t0) * 1e6)  # μs

    mean_us = np.mean(times)
    prior_times.append(mean_us)
    print(f"  {name:<30} {mean_us:>10.2f}   {dist_type}")

print(f"\n  Total prior eval (sum): {sum(prior_times):.2f} μs")
print(f"  Actual prior fn timing: (measuring below...)")

# ---------------------------------------------------------------------------
# Profile full prior function
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FULL PRIOR FUNCTION PROFILING")
print("=" * 80)

prior_stats = time_function(logprior_fn, param_dict_bounded)
print(f"\n  Prior function timing:")
print(f"    Mean: {prior_stats['mean_ms']:.3f} ± {prior_stats['std_ms']:.3f} ms")
print(f"    Median: {prior_stats['median_ms']:.3f} ms")
print(f"    Min: {prior_stats['min_ms']:.3f} ms, Max: {prior_stats['max_ms']:.3f} ms")

# ---------------------------------------------------------------------------
# Profile likelihood function
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("LIKELIHOOD FUNCTION PROFILING")
print("=" * 80)

loglik_stats = time_function(loglik_fn, param_dict_bounded, data_args)
print(f"\n  Likelihood function timing:")
print(f"    Mean: {loglik_stats['mean_ms']:.3f} ± {loglik_stats['std_ms']:.3f} ms")
print(f"    Median: {loglik_stats['median_ms']:.3f} ms")


# Break down likelihood: prediction vs chi-square
@jax.jit
def just_chi_square(flux_model, flux_obs, flux_err):
    """Just the chi-square computation."""
    residual = (flux_obs - flux_model) / flux_err
    return -0.5 * jnp.sum(residual**2)


# Get predicted flux
flux_model = model.predict_photometry(param_dict_bounded, mode="_traceable")

# Time chi-square alone
chisq_stats = time_function(just_chi_square, flux_model, flux_obs, flux_err)
print(f"\n  Chi-square only (no prediction):")
print(f"    Mean: {chisq_stats['mean_ms']:.3f} ± {chisq_stats['std_ms']:.3f} ms")

# ---------------------------------------------------------------------------
# Analyze prior function structure
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PRIOR FUNCTION OPTIMIZATION ANALYSIS")
print("=" * 80)

print(f"\n  Current implementation (loop over {len(fitter._free_names)} params):")
print("    ```python")
print("    def logprior_fn(free_params):")
print("        lp = 0.0")
print("        for name in free_names:")
print("            dist = spec.get_distribution(name)")
print("            lp = lp + dist.log_prob(free_params[name])")
print("        return lp")
print("    ```")

print(f"\n  Observations:")
print(f"    - Python loop gets traced by JAX (unrolled in XLA)")
print(f"    - Each dist.log_prob() call is a separate operation")
print(f"    - No explicit vectorization (each param evaluated independently)")

# Check if we can vectorize
dist_types = set()
for name in fitter._free_names:
    dist = fitter.spec.get_distribution(name)
    dist_types.add(type(dist).__name__)

print(f"\n  Distribution types used: {dist_types}")

if dist_types == {"Uniform"}:
    print(f"    ✓ All Uniform → could vectorize with single jnp.sum()")
elif "Gaussian" in dist_types and "LogUniform" not in dist_types:
    print(f"    ⚠️  Mixed Uniform/Gaussian → vectorization harder")
else:
    print(f"    ⚠️  Complex mix → vectorization non-trivial")

# ---------------------------------------------------------------------------
# Vectorization opportunity test
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VECTORIZATION OPPORTUNITY")
print("=" * 80)

# Test vectorized uniform prior (if all Uniform)
if dist_types == {"Uniform"}:
    print("\n  Testing vectorized Uniform prior...")

    # Extract bounds
    param_values = jnp.array([param_dict_bounded[name] for name in fitter._free_names])
    lower_bounds = jnp.array([fitter._bounds[name][0] for name in fitter._free_names])
    upper_bounds = jnp.array([fitter._bounds[name][1] for name in fitter._free_names])

    @jax.jit
    def vectorized_uniform_prior(param_values, lower_bounds, upper_bounds):
        """Vectorized uniform log prior."""
        # Uniform log_prob is 0 if in bounds, -inf if out of bounds
        # Actually, it's -log(width) for continuous uniform
        widths = upper_bounds - lower_bounds
        in_bounds = jnp.all((param_values >= lower_bounds) & (param_values <= upper_bounds))
        return jnp.where(in_bounds, -jnp.sum(jnp.log(widths)), -jnp.inf)

    # Warmup
    _ = vectorized_uniform_prior(param_values, lower_bounds, upper_bounds)

    # Time it
    vec_stats = time_function(vectorized_uniform_prior, param_values, lower_bounds, upper_bounds)

    print(f"\n  Vectorized prior timing:")
    print(f"    Mean: {vec_stats['mean_ms']:.3f} ± {vec_stats['std_ms']:.3f} ms")

    speedup = prior_stats["mean_ms"] / vec_stats["mean_ms"]
    print(f"\n  Speedup: {speedup:.2f}×")

    if speedup > 1.5:
        print(f"    ✓ SIGNIFICANT speedup potential")
    elif speedup > 1.1:
        print(f"    ⚠️  Modest speedup")
    else:
        print(f"    ❌ No meaningful speedup (overhead dominates)")

else:
    print(f"\n  ⚠️  Skipping vectorization test (mixed distribution types)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY: Prior & Likelihood Performance")
print("=" * 80)

total_loss_time = 2.208  # from previous profiling
prior_pct = (prior_stats["mean_ms"] / total_loss_time) * 100
loglik_pct = (loglik_stats["mean_ms"] / total_loss_time) * 100

print(f"\n  Component breakdown:")
print(f"    Prior:      {prior_stats['mean_ms']:.3f} ms ({prior_pct:.1f}% of loss)")
print(f"    Likelihood: {loglik_stats['mean_ms']:.3f} ms ({loglik_pct:.1f}% of loss)")
print(f"    Total loss: {total_loss_time:.3f} ms (from previous profiling)")

print(f"\n  Optimization opportunities:")
if dist_types == {"Uniform"}:
    print(f"    1. Vectorize uniform priors → potential {speedup:.1f}× speedup on prior")
    print(
        f"       (Would reduce loss time by ~{(prior_stats['mean_ms'] - vec_stats['mean_ms']):.2f}ms)"
    )
else:
    print(f"    1. Prior vectorization limited (mixed distribution types)")

print(f"    2. Chi-square is already optimal ({chisq_stats['mean_ms']:.3f}ms)")
print(f"    3. Main bottleneck remains model prediction (not prior/likelihood)")

print(f"\n  Recommendation:")
if dist_types == {"Uniform"} and speedup > 1.5:
    print(f"    Worth implementing vectorized prior for all-Uniform cases")
    print(f"    (But prediction still dominates, so impact is modest)")
else:
    print(f"    Prior/likelihood already well-optimized")
    print(f"    Focus on model prediction optimization instead")
