"""Quick profiling to check JIT cache and component breakdown.

Runs simplified versions of tests to verify:
1. JAX cache is working (no recompilation)
2. Which components dominate runtime
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

# Force CPU
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

# SSP and filter setup (matching test fixtures)
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

# Load SSP data
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

print("=" * 60)
print("JIT CACHE VERIFICATION")
print("=" * 60)

# Test 1: Simple model (stellar + dust)
print("\n1. Simple model (stellar + dust)")
params_simple = Parameters(
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

model_simple = SEDModel(params_simple, ssp_data, observation=obs)
fitter_simple = Fitter(model_simple, data=flux_obs, noise=noise)

# Clear cache for cold start
jax.clear_caches()

# First call (JIT compilation)
rng_key = jax.random.PRNGKey(42)
t0 = time.perf_counter()
_ = fitter_simple.run("map", key=rng_key)
t_jit = time.perf_counter() - t0
print(f"  First call (JIT): {t_jit:.2f}s")

# Subsequent calls (should NOT recompile)
times = []
for i in range(5):
    key_i = jax.random.fold_in(rng_key, i + 100)
    t0 = time.perf_counter()
    _ = fitter_simple.run("map", key=key_i)
    t = time.perf_counter() - t0
    times.append(t)
    print(f"  Call {i + 2}: {t:.3f}s")

mean_t = np.mean(times)
std_t = np.std(times)
print(f"  Mean runtime: {mean_t:.3f}s ± {std_t:.3f}s")
print(f"  Variance: {std_t / mean_t:.1%}")

if std_t / mean_t > 0.1:
    print("  ⚠️  WARNING: High variance - possible recompilation!")
else:
    print("  ✓ Cache working correctly")

print("\n" + "=" * 60)
print("COMPONENT COMPARISON")
print("=" * 60)

# Test 2: Stochastic model
print("\n2. Stochastic model (dense_basis + field)")
params_stoch = Parameters(
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
    nebular_ssp=True,
    apply_igm=True,
    redshift=Fixed(1.0),
    n_grid=64,
)

model_stoch = SEDModel(params_stoch, ssp_data, observation=obs)
fitter_stoch = Fitter(model_stoch, data=flux_obs, noise=noise)

# Clear cache
jax.clear_caches()

# First call
t0 = time.perf_counter()
_ = fitter_stoch.run("map", key=rng_key)
t_jit_stoch = time.perf_counter() - t0
print(f"  First call (JIT): {t_jit_stoch:.2f}s")

# Subsequent calls
times_stoch = []
for i in range(3):
    key_i = jax.random.fold_in(rng_key, i + 200)
    t0 = time.perf_counter()
    _ = fitter_stoch.run("map", key=key_i)
    t = time.perf_counter() - t0
    times_stoch.append(t)
    print(f"  Call {i + 2}: {t:.3f}s")

mean_t_stoch = np.mean(times_stoch)
print(f"  Mean runtime: {mean_t_stoch:.3f}s")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Simple model:")
print(f"  JIT: {t_jit:.2f}s")
print(f"  Runtime: {mean_t:.3f}s")
print(f"  Speedup after JIT: {t_jit / mean_t:.1f}x")
print(f"\nStochastic model:")
print(f"  JIT: {t_jit_stoch:.2f}s")
print(f"  Runtime: {mean_t_stoch:.3f}s")
print(f"  Speedup after JIT: {t_jit_stoch / mean_t_stoch:.1f}x")
print(f"\nStochastic overhead:")
print(f"  JIT: {t_jit_stoch / t_jit:.1f}x slower")
print(f"  Runtime: {mean_t_stoch / mean_t:.1f}x slower")
