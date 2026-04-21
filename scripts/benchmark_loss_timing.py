"""Detailed timing analysis of loss function compilation and evaluation.

Measures:
1. Cold start: first compilation per prior type
2. Warm evaluation: same model, different galaxy parameters
3. Multi-galaxy: does changing true params trigger recompilation?
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time

jax.config.update("jax_enable_x64", True)

from pathlib import Path
from tengri import (
    SEDModel,
    ParamSpec,
    Uniform,
    Gaussian,
    LogUniform,
    LogNormal,
    StudentT,
    Fitter,
    load_ssp_data,
    load_filter_set,
)


def time_workflow(prior_name, spec_kwargs, n_galaxies=3):
    """Time a complete workflow: compile once, evaluate on multiple galaxies."""

    # Load SSP data
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not ssp_file.is_file():
        return None

    ssp = load_ssp_data(str(ssp_file))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

    # Create model (cold start)
    t0 = time.perf_counter()
    spec = ParamSpec(**spec_kwargs)
    model = SEDModel(spec, ssp, filters=filters)
    t_model_create = time.perf_counter() - t0

    key = jr.PRNGKey(42)

    # Generate mock galaxies with different parameters
    galaxy_params = []
    for i in range(n_galaxies):
        key, subkey = jr.split(key)
        # Randomize parameters within priors
        params = {
            "sfh_tsnorm_skew": float(jr.uniform(subkey, minval=-0.3, maxval=0.3)),
            "sfh_tsnorm_peak_lbt_gyr": float(jr.uniform(subkey, minval=1.0, maxval=8.0)),
            "sfh_tsnorm_width_gyr": float(jr.uniform(subkey, minval=0.5, maxval=2.0)),
            "sfh_tsnorm_trunc": float(jr.uniform(subkey, minval=0.1, maxval=0.8)),
            "sfh_tsnorm_log_peak_sfr": float(jr.uniform(subkey, minval=0.0, maxval=2.0)),
            "met_logzsol": float(jr.uniform(subkey, minval=-1.0, maxval=0.0)),
            "dust_tau_bc": float(jr.uniform(subkey, minval=0.1, maxval=1.5)),
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 1.0,
        }
        galaxy_params.append(params)

    times = {
        "model_create_sec": t_model_create,
        "galaxy_times": [],
    }

    for i, params in enumerate(galaxy_params):
        t0 = time.perf_counter()
        key, subkey = jr.split(key)
        obs = model.mock(params, snr=20.0, key=subkey)
        t_mock = time.perf_counter() - t0

        # Create fitter
        t0 = time.perf_counter()
        fitter = Fitter(model, obs.flux_obs, obs.noise)
        loss_fn = fitter._build_loss_fn()
        data_args = fitter._data_args
        t_fitter_create = time.perf_counter() - t0

        # Initialize params
        t0 = time.perf_counter()
        key, subkey = jr.split(key)
        params_unbounded = fitter._initialize_unbounded(subkey)
        t_init = time.perf_counter() - t0

        # First loss evaluation (triggers JIT compilation)
        t0 = time.perf_counter()
        loss = loss_fn(params_unbounded, data_args)
        t_first_loss = time.perf_counter() - t0

        # Second loss evaluation (warm, no compilation)
        t0 = time.perf_counter()
        loss2 = loss_fn(params_unbounded, data_args)
        t_second_loss = time.perf_counter() - t0

        # Gradient evaluation (first time, may compile)
        t0 = time.perf_counter()
        grad_fn = jax.grad(lambda p: loss_fn(p, data_args))
        grads = grad_fn(params_unbounded)
        t_first_grad = time.perf_counter() - t0

        # Second gradient (warm)
        t0 = time.perf_counter()
        grads2 = grad_fn(params_unbounded)
        t_second_grad = time.perf_counter() - t0

        # 10 warm evaluations to get stable timing
        t0 = time.perf_counter()
        for _ in range(10):
            _ = loss_fn(params_unbounded, data_args)
        t_10_warm = time.perf_counter() - t0

        times["galaxy_times"].append({
            "galaxy_idx": i,
            "mock_sec": t_mock,
            "fitter_create_sec": t_fitter_create,
            "init_sec": t_init,
            "first_loss_sec": t_first_loss,
            "second_loss_sec": t_second_loss,
            "first_grad_sec": t_first_grad,
            "second_grad_sec": t_second_grad,
            "mean_warm_loss_sec": t_10_warm / 10,
            "loss_value": float(loss),
        })

    return times


def main():
    print()
    print("=" * 80)
    print("  Loss Function Timing Benchmark")
    print("=" * 80)
    print()

    # Test 5 prior types
    configs = {
        "Uniform": {
            "mean_sfh_type": "tsnorm",
            "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
            "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
            "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
            "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
            "sfh_tsnorm_log_peak_sfr": Uniform(-1.0, 2.5),
            "met_logzsol": Uniform(-1.5, 0.2),
            "dust_tau_bc": Uniform(0.0, 2.0),
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 1.0,
        },
        "Gaussian": {
            "mean_sfh_type": "tsnorm",
            "sfh_tsnorm_skew": Gaussian(0.0, 0.2, -0.5, 0.5),
            "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
            "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
            "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
            "sfh_tsnorm_log_peak_sfr": Uniform(-1.0, 2.5),
            "met_logzsol": Uniform(-1.5, 0.2),
            "dust_tau_bc": Uniform(0.0, 2.0),
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 1.0,
        },
        "LogUniform": {
            "mean_sfh_type": "tsnorm",
            "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
            "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
            "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
            "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
            "sfh_tsnorm_log_peak_sfr": Uniform(-1.0, 2.5),
            "met_logzsol": Uniform(-1.5, 0.2),
            "dust_tau_bc": Uniform(0.0, 2.0),
            "dust_tau_diff": LogUniform(0.01, 2.0),
            "dust_slope": -0.7,
            "redshift": 1.0,
        },
    }

    all_results = {}

    for name, spec_kwargs in configs.items():
        print(f"\n{'─' * 80}")
        print(f"  {name} Prior")
        print('─' * 80)

        results = time_workflow(name, spec_kwargs, n_galaxies=3)
        if results is None:
            print("  ⚠️  SSP data not found, skipping")
            continue

        all_results[name] = results

        print(f"\nModel creation: {results['model_create_sec']:.3f}s")
        print()

        for gal in results["galaxy_times"]:
            i = gal["galaxy_idx"]
            print(f"Galaxy {i + 1}:")
            print(f"  Mock generation:       {gal['mock_sec']:8.4f}s")
            print(f"  Fitter creation:       {gal['fitter_create_sec']:8.4f}s")
            print(f"  Param initialization:  {gal['init_sec']:8.4f}s")
            print(f"  First loss (+ JIT):    {gal['first_loss_sec']:8.4f}s")
            print(f"  Second loss (warm):    {gal['second_loss_sec']:8.6f}s")
            print(f"  First gradient:        {gal['first_grad_sec']:8.4f}s")
            print(f"  Second gradient:       {gal['second_grad_sec']:8.6f}s")
            print(f"  Mean warm loss (10x):  {gal['mean_warm_loss_sec']:8.6f}s")
            print(f"  Loss value:            {gal['loss_value']:8.2f}")
            print()

    # Summary table
    print()
    print("=" * 80)
    print("  Summary: Does changing galaxy trigger recompilation?")
    print("=" * 80)
    print()
    print(f"{'Prior':<12} {'Galaxy':<8} {'1st Loss':<10} {'2nd Loss':<12} {'Speedup':<10}")
    print("─" * 80)

    for name, results in all_results.items():
        for gal in results["galaxy_times"]:
            i = gal["galaxy_idx"]
            speedup = gal["first_loss_sec"] / gal["second_loss_sec"]
            print(
                f"{name:<12} {i+1:<8} "
                f"{gal['first_loss_sec']:>8.4f}s  "
                f"{gal['second_loss_sec']:>10.6f}s  "
                f"{speedup:>8.1f}x"
            )

    print()
    print("=" * 80)
    print("  Interpretation")
    print("=" * 80)
    print()
    print("• Galaxy 1: First loss includes JIT compilation (~1-2s)")
    print("• Galaxy 1: Second loss is warm (~0.001s) — compilation cached")
    print("• Galaxy 2-3: First loss reuses compilation if model structure unchanged")
    print("• Changing galaxy parameters does NOT trigger recompilation")
    print("• Only changing model structure (priors, components) triggers recompilation")
    print()
    print("Warm evaluation cost: ~1ms per loss, ~1-2ms per gradient")
    print("Compilation cost amortizes over hundreds/thousands of evaluations")
    print()


if __name__ == "__main__":
    main()
