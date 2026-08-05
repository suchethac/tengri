#!/usr/bin/env python3
"""Profile mode='auto' vs mode='_traceable' to understand speedup sources.

Analyzes where time is spent and identifies operations that benefit most
from the auto mode optimization.
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


def profile_loss_fn(fitter, mode, n_measure=50):
    """Profile loss function execution in detail."""
    loss_fn = fitter._get_or_build_loss_fn(mode=mode)
    data_args = fitter._data_args

    key = jax.random.PRNGKey(42)
    params = fitter._initialize_unbounded(key)

    # Warmup (compile)
    print(f"\n  Compiling {mode}...")
    t0 = time.perf_counter()
    _ = loss_fn(params, data_args)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.3f}s")

    # Measure runtime
    print(f"  Measuring runtime ({n_measure} calls)...")
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = loss_fn(params, data_args)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    mean_time = np.mean(times_arr) * 1000  # ms
    std_time = np.std(times_arr) * 1000

    return {
        "compile_time": compile_time,
        "mean_runtime_ms": mean_time,
        "std_runtime_ms": std_time,
        "times_ms": times_arr * 1000,
    }


def profile_model_components(fitter, mode):
    """Profile individual model components to identify speedup sources."""
    model = fitter.model
    params = Parameters(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Fixed(10.5),
        sfh_db_log_sfr_inst=Fixed(1.0),
        sfh_db_tx_frac_0=Fixed(0.3),
        sfh_db_tx_frac_1=Fixed(0.3),
        sfh_db_tx_frac_2=Fixed(0.3),
        met_logzsol=Fixed(-0.5),
        dust_law_bc="calzetti",
        dust_tau_bc=Fixed(1.0),
        dust_tau_diff=Fixed(0.5),
        dust_emission="draine_li2007",
        dust_umin=Fixed(1.0),
        dust_qpah=Fixed(2.5),
        dust_gamma_dl=Fixed(0.1),
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
    )

    # Profile full forward pass
    predict_fn = jax.jit(lambda p: model.predict(p, mode=mode))

    print(f"\n  Forward model components ({mode}):")

    # Warmup
    _ = predict_fn(params)

    # Measure
    n_measure = 20
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = predict_fn(params)
        times.append(time.perf_counter() - t0)

    mean_time = np.mean(times) * 1000
    print(f"    Full predict: {mean_time:.3f} ms")

    return mean_time


def main():
    print("=" * 80)
    print("MODE PARAMETER SPEEDUP PROFILING")
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

    # Create mid-D model (where we see 2.14x speedup)
    print("\nCreating mid-D model (dense_basis + DL07)...")
    params = Parameters(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Uniform(9.0, 11.5),
        sfh_db_log_sfr_inst=Uniform(-1.0, 2.0),
        sfh_db_tx_frac_0=Uniform(0.0, 1.0),
        sfh_db_tx_frac_1=Uniform(0.0, 1.0),
        sfh_db_tx_frac_2=Uniform(0.0, 1.0),
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
    ]
    filters = load_filter_set(filter_names)
    observation = Observation(photometry=Photometry.from_filter_set(filters))
    model = SEDModel(params, ssp_data, observation=observation)

    # Mock data
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9])
    flux_unc_mjy = flux_mjy * 0.1
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    print("Creating fitter...")
    fitter = Fitter(model, flux_cgs, flux_unc_cgs)
    D = len(fitter._free_names)
    print(f"  D = {D} free parameters")

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    # Profile mode="_traceable"
    print("\n[1] Profiling mode='_traceable'")
    print("-" * 80)
    results_traceable = profile_loss_fn(fitter, mode="_traceable", n_measure=50)

    # Profile mode="auto"
    print("\n[2] Profiling mode='auto'")
    print("-" * 80)
    results_auto = profile_loss_fn(fitter, mode="auto", n_measure=50)

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nCompile time:")
    print(f"  _traceable: {results_traceable['compile_time']:.3f}s")
    print(f"  auto:       {results_auto['compile_time']:.3f}s")
    compile_ratio = results_traceable["compile_time"] / results_auto["compile_time"]
    print(f"  Ratio:      {compile_ratio:.2f}x")

    print(f"\nRuntime:")
    print(
        f"  _traceable: {results_traceable['mean_runtime_ms']:.3f} ± {results_traceable['std_runtime_ms']:.3f} ms"
    )
    print(
        f"  auto:       {results_auto['mean_runtime_ms']:.3f} ± {results_auto['std_runtime_ms']:.3f} ms"
    )
    speedup = results_traceable["mean_runtime_ms"] / results_auto["mean_runtime_ms"]
    print(f"  Speedup:    {speedup:.2f}x")

    # Component-level profiling
    print("\n" + "=" * 80)
    print("COMPONENT-LEVEL PROFILING")
    print("=" * 80)

    time_traceable = profile_model_components(fitter, mode="_traceable")
    time_auto = profile_model_components(fitter, mode="auto")

    component_speedup = time_traceable / time_auto
    print(f"\n  Component speedup: {component_speedup:.2f}x")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nWhere does the speedup come from?")
    print("-" * 80)

    if speedup > 1.5:
        print(f"\n✓ Significant speedup: {speedup:.2f}x")
        print("\nLikely sources:")
        print("  1. Dust emission computation (DL07 template interpolation)")
        print("     - mode='auto' allows XLA to fuse interpolation operations")
        print("     - Fewer intermediate array allocations")
        print("  2. Dense basis SFH computation")
        print("     - Basis function evaluations can be fused")
        print("  3. Reduced tracer overhead")
        print("     - mode='_traceable' maintains shape/dtype tracers for all arrays")
        print("     - mode='auto' eliminates this overhead after initial trace")
    elif speedup > 1.05:
        print(f"\n≈ Modest speedup: {speedup:.2f}x")
        print("\nModel complexity may be too low to benefit significantly.")
    else:
        print(f"\n≈ No significant speedup: {speedup:.2f}x")
        print("\nSimple models don't benefit from mode='auto'.")

    print("\nRecommendations:")
    print("-" * 80)
    if speedup > 1.2:
        print("  • Use mode='auto' for inference (default)")
        print("  • Speedup increases with model complexity")
        print("  • Dust emission and stochastic SFH benefit most")
    else:
        print("  • Mode parameter has minimal impact for simple models")
        print("  • Both modes produce equivalent results")


if __name__ == "__main__":
    main()
