"""Diagnose JAX cache instability and runtime variance.

Investigates the 96-136% variance observed in MAP inference runs.
Tests hypotheses:
1. GC pauses causing slowdowns
2. Cache invalidation on some runs
3. XLA recompilation despite cache hits
4. Thread scheduling variance
"""

import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

# Force CPU and enable compilation logging
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_log_compiles", True)  # Log XLA compilations

# SSP and filter setup
SSP_PATH = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
FILTER_NAMES = [
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

print("Loading SSP data...")
ssp_data = load_ssp_data(str(SSP_PATH))

# Create mock data
filter_data = load_filter_set(FILTER_NAMES)
obs = Observation(photometry=Photometry.from_filter_set(filter_data))

rng = np.random.default_rng(42)
n_bands = len(FILTER_NAMES)
log_flux = rng.uniform(np.log10(0.1), np.log10(100.0), size=n_bands)
flux_cgs = (10.0**log_flux * 1e-3) * 1e-26
noise_cgs = flux_cgs / 10.0
flux_obs = jnp.array(flux_cgs + rng.normal(0, noise_cgs))
noise = jnp.array(noise_cgs)

# Test model (simple to isolate variance)
params = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_law_bc="salim_sbl18",
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    redshift=Fixed(1.0),
)

model = SEDModel(params, ssp_data, observation=obs)
fitter = Fitter(model, data=flux_obs, noise=noise)

print("=" * 60)
print("CACHE INSTABILITY DIAGNOSTIC")
print("=" * 60)

# Test 1: Baseline (no GC control)
print("\n1. Baseline (no intervention)")
print("-" * 60)
jax.clear_caches()
rng_key = jax.random.PRNGKey(42)

# First call (JIT)
t0 = time.perf_counter()
_ = fitter.run("map", key=rng_key)
t_jit = time.perf_counter() - t0
print(f"  JIT: {t_jit:.2f}s")

# Warmup runs
times_baseline = []
for i in range(10):
    key_i = jax.random.fold_in(rng_key, i + 100)
    t0 = time.perf_counter()
    _ = fitter.run("map", key=key_i)
    t = time.perf_counter() - t0
    times_baseline.append(t)
    print(f"  Run {i + 1}: {t:.3f}s")

mean_baseline = np.mean(times_baseline)
std_baseline = np.std(times_baseline)
print(f"  Mean: {mean_baseline:.3f}s ± {std_baseline:.3f}s")
print(f"  Variance: {std_baseline / mean_baseline:.1%}")

# Test 2: With explicit GC before each run
print("\n2. Explicit GC before each run")
print("-" * 60)
jax.clear_caches()

# First call (JIT)
t0 = time.perf_counter()
_ = fitter.run("map", key=rng_key)
t_jit = time.perf_counter() - t0
print(f"  JIT: {t_jit:.2f}s")

# Warmup runs with GC
times_gc = []
for i in range(10):
    gc.collect()  # Force GC before each run
    key_i = jax.random.fold_in(rng_key, i + 200)
    t0 = time.perf_counter()
    _ = fitter.run("map", key=key_i)
    t = time.perf_counter() - t0
    times_gc.append(t)
    print(f"  Run {i + 1}: {t:.3f}s (post-GC)")

mean_gc = np.mean(times_gc)
std_gc = np.std(times_gc)
print(f"  Mean: {mean_gc:.3f}s ± {std_gc:.3f}s")
print(f"  Variance: {std_gc / mean_gc:.1%}")

# Test 3: Reuse same key (eliminate RNG variance)
print("\n3. Same key every run (eliminate RNG variance)")
print("-" * 60)
jax.clear_caches()

# First call (JIT)
fixed_key = jax.random.PRNGKey(999)
t0 = time.perf_counter()
_ = fitter.run("map", key=fixed_key)
t_jit = time.perf_counter() - t0
print(f"  JIT: {t_jit:.2f}s")

# Warmup runs with fixed key
times_fixed = []
for i in range(10):
    t0 = time.perf_counter()
    _ = fitter.run("map", key=fixed_key)
    t = time.perf_counter() - t0
    times_fixed.append(t)
    print(f"  Run {i + 1}: {t:.3f}s (same key)")

mean_fixed = np.mean(times_fixed)
std_fixed = np.std(times_fixed)
print(f"  Mean: {mean_fixed:.3f}s ± {std_fixed:.3f}s")
print(f"  Variance: {std_fixed / mean_fixed:.1%}")

# Test 4: Check for cache directory corruption
print("\n4. Cache directory check")
print("-" * 60)
cache_dir = Path.home() / ".cache" / "tengri_jax_cache"
if cache_dir.exists():
    cache_files = list(cache_dir.glob("**/*"))
    total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
    print(f"  Cache directory: {cache_dir}")
    print(f"  Files: {len([f for f in cache_files if f.is_file()])}")
    print(f"  Total size: {total_size / (1024**2):.1f} MB")
else:
    print(f"  Cache directory not found: {cache_dir}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(
    f"Baseline:      {mean_baseline:.3f}s ± {std_baseline:.3f}s ({std_baseline / mean_baseline:.1%} variance)"
)
print(f"With GC:       {mean_gc:.3f}s ± {std_gc:.3f}s ({std_gc / mean_gc:.1%} variance)")
print(
    f"Fixed key:     {mean_fixed:.3f}s ± {std_fixed:.3f}s ({std_fixed / mean_fixed:.1%} variance)"
)

# Determine root cause
if std_gc / mean_gc < std_baseline / mean_baseline * 0.5:
    print("\n✓ GC pauses are the primary cause of variance")
    print("  Recommendation: Call gc.collect() before critical inference runs")
elif std_fixed / mean_fixed < std_baseline / mean_baseline * 0.5:
    print("\n✓ RNG key generation causes variance")
    print("  Recommendation: This variance is expected and benign")
else:
    print("\n⚠️ Variance persists across all tests")
    print("  Root cause unclear - may be OS scheduling or XLA runtime")
    print("  Recommendation: Accept 10-20% variance as normal for JAX inference")

# Outlier analysis
outliers_baseline = [t for t in times_baseline if t > mean_baseline + 2 * std_baseline]
outliers_gc = [t for t in times_gc if t > mean_gc + 2 * std_gc]
outliers_fixed = [t for t in times_fixed if t > mean_fixed + 2 * std_fixed]

print(f"\nOutliers (>2σ):")
print(f"  Baseline: {len(outliers_baseline)}/10 runs")
print(f"  With GC:  {len(outliers_gc)}/10 runs")
print(f"  Fixed key: {len(outliers_fixed)}/10 runs")

if outliers_baseline:
    print(f"\n  Baseline outliers: {[f'{t:.3f}s' for t in outliers_baseline]}")
if outliers_gc:
    print(f"  GC outliers: {[f'{t:.3f}s' for t in outliers_gc]}")
if outliers_fixed:
    print(f"  Fixed key outliers: {[f'{t:.3f}s' for t in outliers_fixed]}")
