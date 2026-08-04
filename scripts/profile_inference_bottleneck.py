#!/usr/bin/env python3
"""Profile inference bottlenecks to understand why it's slow.

Investigates:
1. Is DL07 preintegration active?
2. What is the actual per-step loss function time?
3. How much overhead is MCMC vs loss evaluation?
4. Where is the 73s coming from in test_a2?
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import SEDModel, Fitter, Parameters, Observation, Photometry
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform


def profile_loss_function(fitter, n_evals=100):
    """Profile raw loss function evaluation time."""
    print("\n" + "=" * 80)
    print("LOSS FUNCTION PROFILING")
    print("=" * 80)

    # Get loss function
    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args

    # Initialize params
    key = jax.random.PRNGKey(42)
    params = fitter._initialize_unbounded(key)

    # Warmup (compile)
    print("\n  Compiling loss function...")
    t0 = time.perf_counter()
    loss_val = loss_fn(params, data_args)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.3f}s")
    print(f"  Initial loss: {loss_val:.4f}")

    # Warmup (additional calls to stabilize JIT)
    print(f"\n  Warmup (10 calls)...")
    for _ in range(10):
        _ = loss_fn(params, data_args)

    # Measure steady-state evaluation time
    print(f"\n  Measuring steady-state loss evaluation time ({n_evals} calls)...")
    times = []
    for _ in range(n_evals):
        t0 = time.perf_counter()
        _ = loss_fn(params, data_args)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    mean_time_ms = np.mean(times_arr) * 1000
    std_time_ms = np.std(times_arr) * 1000

    print(f"\n  Steady-state loss function timing:")
    print(f"    Mean: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
    print(f"    Min: {np.min(times_arr) * 1000:.3f} ms")
    print(f"    Max: {np.max(times_arr) * 1000:.3f} ms")
    print(f"    Median: {np.median(times_arr) * 1000:.3f} ms")

    return {
        "compile_time_s": compile_time,
        "mean_eval_ms": mean_time_ms,
        "std_eval_ms": std_time_ms,
    }


def check_preintegration_status(model):
    """Check if DL07 preintegration is active."""
    print("\n" + "=" * 80)
    print("PREINTEGRATION STATUS")
    print("=" * 80)

    has_precomputed = hasattr(model, "_precomputed")
    print(f"\n  model._precomputed exists: {has_precomputed}")

    if has_precomputed:
        has_dust_lookup = model._precomputed.dust_ir_lookup is not None
        print(f"  model._precomputed.dust_ir_lookup exists: {has_dust_lookup}")

        if has_dust_lookup:
            print("\n  ✓ DL07 preintegration is ACTIVE")
            # dust_ir_lookup may be a JIT function or array
            lookup_type = type(model._precomputed.dust_ir_lookup).__name__
            print(f"  Lookup type: {lookup_type}")
            return True
        else:
            print("\n  ❌ DL07 preintegration is DISABLED")
            print("  Falling back to full-wavelength integration")
            return False
    else:
        print("\n  ❌ No precomputed data")
        return False


def estimate_mcmc_overhead(n_steps, loss_eval_ms, total_time_s):
    """Estimate MCMC overhead beyond loss evaluation."""
    pure_loss_time_s = (n_steps * loss_eval_ms) / 1000
    overhead_s = total_time_s - pure_loss_time_s
    overhead_pct = (overhead_s / total_time_s) * 100

    print("\n" + "=" * 80)
    print("MCMC OVERHEAD ANALYSIS")
    print("=" * 80)

    print(f"\n  Total MCMC steps: {n_steps}")
    print(f"  Loss eval time: {loss_eval_ms:.3f} ms/step")
    print(f"  Pure loss time: {pure_loss_time_s:.2f}s")
    print(f"  Total time: {total_time_s:.2f}s")
    print(f"  Overhead: {overhead_s:.2f}s ({overhead_pct:.1f}%)")

    print("\n  Overhead breakdown (typical):")
    print("    - Gradient computation: ~2-3× loss eval time")
    print("    - NUTS tree building: ~10-20% of total")
    print("    - Parameter transforms: ~5-10% of total")
    print("    - Acceptance/rejection: ~5% of total")

    return {
        "pure_loss_s": pure_loss_time_s,
        "overhead_s": overhead_s,
        "overhead_pct": overhead_pct,
    }


def main():
    print("=" * 80)
    print("INFERENCE BOTTLENECK PROFILING")
    print("=" * 80)

    # Load SSP data
    print("\nLoading SSP data...")
    ssp_path = (
        Path(__file__).parent.parent
        / "data"
        / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    )
    if not ssp_path.exists():
        print(f"  ❌ SSP data not found at {ssp_path}")
        return

    ssp_data = load_ssp_data(str(ssp_path))
    print(f"  ✓ Loaded {ssp_path.name}")

    # Create test_a2 model (DL07 case)
    print("\n" + "=" * 80)
    print("TEST A2: FIR-Constrained Fit with DL07")
    print("=" * 80)

    params = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=Uniform(8.5, 11.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_emission="draine_li2007",
        dust_umin=Fixed(1.0),
        dust_qpah=Uniform(0.5, 4.5),
        dust_gamma_dl=Uniform(0.0, 0.2),
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
    )

    filter_names = [
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
        "herschel_160",
        "herschel_250",
    ]
    filters = load_filter_set(filter_names)
    observation = Observation(photometry=Photometry.from_filter_set(filters))

    print("\nCreating SEDModel...")
    model = SEDModel(params, ssp_data, observation=observation)

    # Check preintegration status
    preint_active = check_preintegration_status(model)

    # Mock data
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9, 3.5, 2.8])
    flux_unc_mjy = flux_mjy * 0.15
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    fitter = Fitter(model, flux_cgs, flux_unc_cgs)
    D = len(fitter._free_names)
    print(f"\n  D = {D} free parameters")

    # Profile loss function
    loss_timing = profile_loss_function(fitter, n_evals=100)

    # Estimate MCMC overhead for 500 steps (250 warmup + 250 samples)
    n_mcmc_steps = 500
    observed_total_time = 73.93  # From test_a2 result

    overhead = estimate_mcmc_overhead(
        n_mcmc_steps,
        loss_timing["mean_eval_ms"],
        observed_total_time,
    )

    # Compare to expected performance
    print("\n" + "=" * 80)
    print("EXPECTED VS ACTUAL PERFORMANCE")
    print("=" * 80)

    expected_loss_ms = 12.6  # From dust-preintegration.md

    print(f"\n  Documentation says:")
    print(f"    Expected loss eval: ~{expected_loss_ms} ms (with preintegration)")
    print(f"    DL07 preintegration speedup: 16.3x (667μs → 41μs)")

    print(f"\n  Actual measurement:")
    print(f"    Loss eval: {loss_timing['mean_eval_ms']:.3f} ms")
    print(f"    Preintegration active: {preint_active}")

    if preint_active:
        if loss_timing["mean_eval_ms"] > 2 * expected_loss_ms:
            print(f"\n  ❌ SLOW despite preintegration")
            print(
                f"     {loss_timing['mean_eval_ms'] / expected_loss_ms:.1f}x slower than expected"
            )
            print("\n  Possible causes:")
            print("    1. Preintegration not working correctly")
            print("    2. Additional components not in baseline (nebular, IGM)")
            print("    3. Filter convolution overhead")
            print("    4. Parameter transforms overhead")
        else:
            print(f"\n  ✓ Performance roughly matches expectation")
    else:
        print(f"\n  ❌ Preintegration is disabled")
        print("  This explains the slow performance")

        expected_slowdown = 16.3
        print(f"\n  Expected slowdown without preintegration: {expected_slowdown}x")
        print(f"  Expected loss eval: ~{expected_loss_ms * expected_slowdown:.1f} ms")

    # MCMC time budget
    print("\n" + "=" * 80)
    print("MCMC TIME BUDGET (500 steps)")
    print("=" * 80)

    print(f"\n  If loss eval is {loss_timing['mean_eval_ms']:.3f} ms:")
    print(f"    Pure loss: {n_mcmc_steps * loss_timing['mean_eval_ms'] / 1000:.1f}s")
    print(
        f"    + Gradients (~2x loss): {2 * n_mcmc_steps * loss_timing['mean_eval_ms'] / 1000:.1f}s"
    )
    print(
        f"    + NUTS overhead (~20%): {0.2 * 3 * n_mcmc_steps * loss_timing['mean_eval_ms'] / 1000:.1f}s"
    )
    print(f"    ≈ Total: {3.6 * n_mcmc_steps * loss_timing['mean_eval_ms'] / 1000:.1f}s")

    print(f"\n  Observed total: {observed_total_time:.1f}s")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if not preint_active:
        print("\n  🔧 Fix #1: Enable DL07 preintegration")
        print("     Check why it's disabled:")
        print("       - Are DL07 templates in data/?")
        print("       - Is redshift Fixed?")
        print("       - Are there exceptions preventing precompute?")
        print(f"     Expected speedup: {16.3:.1f}x → {observed_total_time / 16.3:.1f}s total")

    if loss_timing["mean_eval_ms"] > expected_loss_ms:
        print("\n  🔧 Fix #2: Optimize loss function")
        print("     Current bottlenecks:")
        print("       - Filter convolution")
        print("       - Parameter transforms")
        print("       - Gradient overhead")

    if overhead["overhead_pct"] > 60:
        print("\n  🔧 Fix #3: Reduce MCMC overhead")
        print("     Try:")
        print("       - Fewer warmup steps (250 → 100)")
        print("       - Use MAP init point")
        print("       - Switch to VI for exploration")


if __name__ == "__main__":
    main()
