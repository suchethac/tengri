"""Diagnose why different RNG keys cause 5-8× MAP runtime variance.

Root cause investigation:
1. Do different initializations explore different regions of parameter space?
2. Do some trajectories hit numerically unstable regions (NaN/Inf checks)?
3. Does the loss landscape have pathological conditioning?
4. Is ADAM's adaptive learning rate causing divergent behavior?
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

# Test model
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
print("MAP INITIALIZATION SENSITIVITY ANALYSIS")
print("=" * 60)

# Run many trials with different keys to find fast vs slow cases
print("\n1. Screening run: 50 different RNG keys")
print("-" * 60)

jax.clear_caches()
base_key = jax.random.PRNGKey(42)

# First JIT call
_ = fitter.run("map", key=base_key)

# Screening
runtimes = []
keys_list = []
for i in range(50):
    key_i = jax.random.fold_in(base_key, abs(hash(f"trial_{i}")) % (2**31))
    t0 = time.perf_counter()
    _ = fitter.run("map", key=key_i)
    t = time.perf_counter() - t0
    runtimes.append(t)
    keys_list.append(key_i)

    if i % 10 == 9:
        print(
            f"  Trials {i - 8}-{i + 1}: {np.mean(runtimes[-10:]):.3f}s ± {np.std(runtimes[-10:]):.3f}s"
        )

runtimes = np.array(runtimes)
mean_runtime = np.mean(runtimes)
std_runtime = np.std(runtimes)

print(f"\nOverall:")
print(f"  Mean: {mean_runtime:.3f}s ± {std_runtime:.3f}s")
print(f"  Min: {np.min(runtimes):.3f}s")
print(f"  Max: {np.max(runtimes):.3f}s")
print(f"  Variance: {std_runtime / mean_runtime:.1%}")

# Find fast and slow cases
threshold_fast = np.percentile(runtimes, 10)
threshold_slow = np.percentile(runtimes, 90)

fast_idx = np.where(runtimes <= threshold_fast)[0]
slow_idx = np.where(runtimes >= threshold_slow)[0]

print(f"\n  Fast cases (p10 ≤ {threshold_fast:.3f}s): {len(fast_idx)} trials")
print(f"  Slow cases (p90 ≥ {threshold_slow:.3f}s): {len(slow_idx)} trials")

# Detailed comparison: 3 fast vs 3 slow
print("\n2. Detailed comparison: 3 fast vs 3 slow")
print("-" * 60)

# Compare 3 fast and 3 slow cases
# Just run fitter.run("map") and check final loss value
from tengri.inference.loss_functions import build_loss_fn

loss_fn = build_loss_fn(fitter)
data_args = {"data": flux_obs, "noise": noise}

print("\nFast cases:")
fast_results = []
for i, idx in enumerate(fast_idx[:3]):
    key = keys_list[idx]
    t0 = time.perf_counter()
    result = fitter.run("map", key=key)
    t = time.perf_counter() - t0

    # Get final loss: convert physical params to unbounded, then call loss_fn
    params_unbounded = fitter._unbounded_from_posterior(result)
    final_loss = loss_fn(params_unbounded, data_args)

    print(f"  Fast {i + 1}:")
    print(f"    Runtime: {t:.3f}s (screening: {runtimes[idx]:.3f}s)")
    print(f"    Final loss: {float(final_loss):.1f}")
    fast_results.append(
        {
            "runtime": t,
            "loss": float(final_loss),
            "params": result.params,
        }
    )

print("\nSlow cases:")
slow_results = []
for i, idx in enumerate(slow_idx[:3]):
    key = keys_list[idx]
    t0 = time.perf_counter()
    result = fitter.run("map", key=key)
    t = time.perf_counter() - t0

    # Get final loss: convert physical params to unbounded, then call loss_fn
    params_unbounded = fitter._unbounded_from_posterior(result)
    final_loss = loss_fn(params_unbounded, data_args)

    print(f"  Slow {i + 1}:")
    print(f"    Runtime: {t:.3f}s (screening: {runtimes[idx]:.3f}s)")
    print(f"    Final loss: {float(final_loss):.1f}")
    slow_results.append(
        {
            "runtime": t,
            "loss": float(final_loss),
            "params": result.params,
        }
    )

# Compare final losses
fast_losses = [r["loss"] for r in fast_results]
slow_losses = [r["loss"] for r in slow_results]

print(f"\nLoss comparison:")
print(f"  Fast cases: {np.mean(fast_losses):.1f} ± {np.std(fast_losses):.1f}")
print(f"  Slow cases: {np.mean(slow_losses):.1f} ± {np.std(slow_losses):.1f}")

if np.abs(np.mean(fast_losses) - np.mean(slow_losses)) > 5.0:
    print("  ⚠️ Significant difference in final loss!")
    print("     Slow cases may be stuck in poor local minima")
else:
    print("  ✓ Similar final losses")
    print("     Variance is runtime-only, not optimization quality")

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

# Hypothesis testing
if np.mean(runtimes[slow_idx]) > 2 * np.mean(runtimes[fast_idx]):
    print("✓ Confirmed: Slow cases are 2x+ slower than fast cases")
    print("  → This is NOT normal optimizer variance")
    print("  → Likely causes:")
    print("    1. Bad initializations explore poor regions of parameter space")
    print("    2. Numerical instability (check for NaN/Inf in loss)")
    print("    3. JIT cache misses due to different code paths")
else:
    print("✗ Slow cases are not significantly slower")
    print("  → Variance is likely due to OS scheduling or CPU throttling")

# Check if slow cases were reproducible
print("\nReproducibility check:")
slow_screening_times = [runtimes[idx] for idx in slow_idx[:3]]
slow_rerun_times = [r["runtime"] for r in slow_results]
for i, (t_screen, t_rerun) in enumerate(zip(slow_screening_times, slow_rerun_times)):
    ratio = t_screen / t_rerun
    print(f"  Slow case {i + 1}: {t_screen:.3f}s → {t_rerun:.3f}s ({ratio:.1f}× change)")

if all(t_rerun < 0.5 for t_rerun in slow_rerun_times):
    print("\n⚠️  CRITICAL: 'Slow' cases became fast on re-run!")
    print("  → Variance is NOT deterministic (RNG key is not the cause)")
    print("  → Likely: OS scheduling, CPU throttling, background processes")
    print("  → Implication: Users can just re-run if MAP is slow")
else:
    print("\n✓ 'Slow' cases remained slow on re-run")
    print("  → Variance is deterministic (RNG key IS the cause)")
    print("  → Implication: Some initializations are genuinely bad")

# Recommendation
print("\nRecommendation:")
if all(t_rerun < 0.5 for t_rerun in slow_rerun_times):
    print("  1. Document that MAP runtime has high variance (up to 50× slowdown)")
    print("  2. Tell users: if MAP is slow, just re-run with the same settings")
    print("  3. Consider implementing retry logic: run MAP 3× and take fastest")
    print("  4. Investigate system-level causes (GC, CPU frequency scaling)")
else:
    print("  1. Use seeded RNG keys for reproducibility")
    print("  2. Run MAP 3-5 times with different seeds, take best result")
    print("  3. Consider warm-starting MAP from prior mean instead of random draw")
    print("  4. Monitor loss trajectory for NaN/Inf (indicates numerical issues)")
