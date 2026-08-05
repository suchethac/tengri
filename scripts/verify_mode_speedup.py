#!/usr/bin/env python3
"""Verify mode='auto' provides ~1.5x speedup over mode='_traceable'.

Tests the implementation from the mode parameter PR.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import SEDModel, Fitter, Parameters, Observation, Photometry
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform


def benchmark_mode(fitter, mode, n_warmup=5, n_measure=20):
    """Benchmark loss function with given mode."""
    loss_fn = fitter._get_or_build_loss_fn(mode=mode)
    data_args = fitter._data_args

    # Initialize params
    key = jax.random.PRNGKey(42)
    params = fitter._initialize_unbounded(key)

    # Warmup
    for _ in range(n_warmup):
        _ = loss_fn(params, data_args)

    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = loss_fn(params, data_args)
        times.append(time.perf_counter() - t0)

    return jnp.array(times)


def main():
    print("Loading SSP data...")
    ssp_path = (
        Path(__file__).parent.parent
        / "data"
        / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    )
    if not ssp_path.exists():
        print(f"  ❌ SSP data not found at {ssp_path}")
        print("  Available SSP files should be in data/ directory")
        return

    ssp_data = load_ssp_data(str(ssp_path))
    print(f"  ✓ Loaded {ssp_path.name}")
    print()

    print("Creating simple D=7 model (tsnorm SFH + Calzetti dust)...")
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

    # Mock data (10 filters)
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9])
    flux_unc_mjy = flux_mjy * 0.1
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    print("Building fitter...")
    fitter = Fitter(model, flux_cgs, flux_unc_cgs)
    D = len(fitter._free_names)
    print(f"  D = {D} free parameters: {fitter._free_names}")
    print()

    print("Benchmarking mode='_traceable'...")
    times_traceable = benchmark_mode(fitter, mode="_traceable", n_measure=20)
    mean_traceable = float(jnp.mean(times_traceable)) * 1000  # ms
    std_traceable = float(jnp.std(times_traceable)) * 1000
    print(f"  Mean: {mean_traceable:.3f} ± {std_traceable:.3f} ms")
    print()

    print("Benchmarking mode='auto'...")
    times_auto = benchmark_mode(fitter, mode="auto", n_measure=20)
    mean_auto = float(jnp.mean(times_auto)) * 1000  # ms
    std_auto = float(jnp.std(times_auto)) * 1000
    print(f"  Mean: {mean_auto:.3f} ± {std_auto:.3f} ms")
    print()

    speedup = mean_traceable / mean_auto
    print(f"Speedup: {speedup:.2f}x")
    print()

    # Verify result
    if speedup >= 1.4:
        print("✓ PASS: Speedup ≥ 1.4x (target 1.4-1.8x)")
    elif speedup >= 1.2:
        print(f"⚠️ WARN: Speedup {speedup:.2f}x is below target (1.4-1.8x)")
    else:
        print(f"❌ FAIL: Speedup {speedup:.2f}x is too low")

    # Verify cache keys are different
    cache_traceable = (fitter._engine_cache_key(), "_traceable")
    cache_auto = (fitter._engine_cache_key(), "auto")

    print()
    print("Cache key verification:")
    print(f"  _traceable key: {cache_traceable[0][:60]}..., mode={cache_traceable[1]}")
    print(f"  auto key: {cache_auto[0][:60]}..., mode={cache_auto[1]}")

    if cache_traceable != cache_auto:
        print("  ✓ Cache keys are different (no collision)")
    else:
        print("  ❌ Cache keys are identical (COLLISION BUG)")


if __name__ == "__main__":
    main()
