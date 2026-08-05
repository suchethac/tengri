#!/usr/bin/env python3
"""Profile NUTS overhead to understand why 500 steps takes 73s.

The loss function is fast (2.3ms), so where is the time going?
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


def profile_gradient_computation(fitter, n_evals=50):
    """Profile gradient computation time."""
    print("\n" + "=" * 80)
    print("GRADIENT COMPUTATION PROFILING")
    print("=" * 80)

    # Get loss function
    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args

    # Initialize params
    key = jax.random.PRNGKey(42)
    params = fitter._initialize_unbounded(key)

    # Create value_and_grad function
    value_and_grad_fn = jax.jit(jax.value_and_grad(lambda p: loss_fn(p, data_args)))

    # Warmup
    print("\n  Compiling value_and_grad...")
    t0 = time.perf_counter()
    loss_val, grad = value_and_grad_fn(params)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.3f}s")
    print(f"  Loss: {loss_val:.4f}")
    print(
        f"  Grad norm: {jnp.linalg.norm(jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(grad)])):.4f}"
    )

    # Additional warmup
    for _ in range(10):
        _ = value_and_grad_fn(params)

    # Measure steady-state
    print(f"\n  Measuring steady-state gradient time ({n_evals} calls)...")
    times = []
    for _ in range(n_evals):
        t0 = time.perf_counter()
        _ = value_and_grad_fn(params)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    mean_time_ms = np.mean(times_arr) * 1000
    std_time_ms = np.std(times_arr) * 1000

    print(f"\n  Steady-state gradient timing:")
    print(f"    Mean: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
    print(f"    Min: {np.min(times_arr) * 1000:.3f} ms")
    print(f"    Max: {np.max(times_arr) * 1000:.3f} ms")
    print(f"    Median: {np.median(times_arr) * 1000:.3f} ms")

    return mean_time_ms


def profile_nuts_step(fitter, n_steps=50):
    """Profile a single NUTS step."""
    print("\n" + "=" * 80)
    print("NUTS STEP PROFILING")
    print("=" * 80)

    from numpyro.infer import NUTS
    from numpyro import sample, distributions as dist

    # Create NUTS kernel
    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args

    def potential_fn(params):
        return loss_fn(params, data_args)

    # Initialize
    key = jax.random.PRNGKey(42)
    init_params = fitter._initialize_unbounded(key)

    # Create NUTS kernel
    nuts_kernel = NUTS(potential_fn=potential_fn)

    print("\n  Initializing NUTS kernel...")
    t0 = time.perf_counter()
    nuts_state = nuts_kernel.init(key, init_params)
    init_time = time.perf_counter() - t0
    print(f"  Kernel init time: {init_time:.3f}s")

    # Warmup step
    print("\n  Compiling NUTS step...")
    t0 = time.perf_counter()
    key, subkey = jax.random.split(key)
    nuts_state = nuts_kernel.sample(nuts_state, subkey)
    compile_time = time.perf_counter() - t0
    print(f"  Compile time: {compile_time:.3f}s")

    # Additional warmup
    for _ in range(10):
        key, subkey = jax.random.split(key)
        nuts_state = nuts_kernel.sample(nuts_state, subkey)

    # Measure steady-state
    print(f"\n  Measuring steady-state NUTS step time ({n_steps} steps)...")
    times = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        nuts_state = nuts_kernel.sample(nuts_state, subkey)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    mean_time_ms = np.mean(times_arr) * 1000
    std_time_ms = np.std(times_arr) * 1000

    print(f"\n  Steady-state NUTS step timing:")
    print(f"    Mean: {mean_time_ms:.3f} ± {std_time_ms:.3f} ms")
    print(f"    Min: {np.min(times_arr) * 1000:.3f} ms")
    print(f"    Max: {np.max(times_arr) * 1000:.3f} ms")
    print(f"    Median: {np.median(times_arr) * 1000:.3f} ms")

    # Estimate 500 steps
    estimated_500_s = (mean_time_ms * 500) / 1000
    print(f"\n  Estimated 500 steps: {estimated_500_s:.1f}s")

    return mean_time_ms


def main():
    print("=" * 80)
    print("NUTS OVERHEAD DEEP DIVE")
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

    # Create test_a2 model
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
    model = SEDModel(params, ssp_data, observation=observation)

    # Mock data
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9, 3.5, 2.8])
    flux_unc_mjy = flux_mjy * 0.15
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    fitter = Fitter(model, flux_cgs, flux_unc_cgs)
    D = len(fitter._free_names)
    print(f"\n  D = {D} free parameters")

    # Profile gradient computation
    grad_time_ms = profile_gradient_computation(fitter, n_evals=50)

    # Profile NUTS steps
    try:
        nuts_time_ms = profile_nuts_step(fitter, n_steps=50)
    except Exception as e:
        print(f"\n  ❌ NUTS profiling failed: {e}")
        nuts_time_ms = None

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print(f"\n  Gradient computation: {grad_time_ms:.3f} ms/call")
    if nuts_time_ms is not None:
        print(f"  Full NUTS step: {nuts_time_ms:.3f} ms/step")
        print(
            f"  NUTS overhead: {nuts_time_ms - grad_time_ms:.3f} ms/step ({(nuts_time_ms / grad_time_ms - 1) * 100:.1f}%)"
        )

        print(f"\n  Estimated 500 NUTS steps: {(nuts_time_ms * 500) / 1000:.1f}s")
        print(f"  Observed test_a2: 73.9s")
        print(f"  Discrepancy: {73.9 - (nuts_time_ms * 500) / 1000:.1f}s")

    print("\n  Possible explanations for remaining discrepancy:")
    print("    1. Warmup steps have additional overhead (adaptation)")
    print("    2. Tree building depth varies (not constant per step)")
    print("    3. Parameter transforms (bounded ↔ unbounded)")
    print("    4. NumPyro diagnostic computation")


if __name__ == "__main__":
    main()
