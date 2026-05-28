"""Comprehensive inference engine benchmark across model complexities.

Tests realistic science cases:
- Simple optical fit (D~7): map, laplace, mcmc_nuts, vi, nss
- Mid-complexity (D~12): map, mcmc_nuts, vi, nss
- High-D stochastic (D~20): map, vi, nss

Measures:
- Total inference time (separating JIT warmup from solver)
- Posterior quality (if applicable)
- Memory usage
- Convergence diagnostics

The benchmark reuses the same SEDModel/Fitter across all methods within a
scenario so that cached loss/gradient functions are shared — matching
how a user would compare methods on the same galaxy.
"""

import gc
import time
import tracemalloc

import jax
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from pathlib import Path

from tengri import (
    Fitter,
    Gaussian,
    LogUniform,
    Parameters,
    SEDModel,
    Uniform,
    load_filter_set,
    load_ssp_data,
)


def build_scenario_fitter(spec_kwargs, true_params, snr=20.0, key=None):
    """Build a SEDModel + Fitter for one scenario (reused across methods)."""
    if key is None:
        key = jr.PRNGKey(42)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not ssp_file.is_file():
        return None

    ssp = load_ssp_data(str(ssp_file))
    filters = load_filter_set(
        [
            "sdss_u",
            "sdss_g",
            "sdss_r",
            "sdss_i",
            "sdss_z",
            "2mass_j",
            "2mass_h",
            "2mass_ks",
        ]
    )

    spec = Parameters(**spec_kwargs)
    model = SEDModel(spec, ssp, filters=filters)

    key, subkey = jr.split(key)
    obs = model.mock(true_params, snr=snr, key=subkey)
    fitter = Fitter(model, obs.flux_obs, obs.noise)
    return fitter


def run_inference_method(
    scenario_name,
    fitter,
    method,
    method_kwargs,
    key,
):
    """Run one inference method on a pre-built fitter and measure performance."""
    n_free = len(fitter.spec.free_params)

    tracemalloc.start()
    peak_mem_before = tracemalloc.get_traced_memory()[1] / 1024**2

    t0 = time.perf_counter()
    try:
        posterior = fitter.run(method=method, key=key, **method_kwargs)
        t_inference = time.perf_counter() - t0
        success = True
        error_msg = None
    except Exception as e:
        t_inference = time.perf_counter() - t0
        success = False
        error_msg = str(e)[:100]
        posterior = None

    peak_mem_after = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()
    peak_mem_delta = peak_mem_after - peak_mem_before

    diagnostics = {}
    if success and posterior is not None:
        if method == "map":
            if hasattr(posterior, "diagnostics") and "final_loss" in posterior.diagnostics:
                diagnostics["loss_final"] = float(posterior.diagnostics["final_loss"])
                if "converged" in posterior.diagnostics:
                    diagnostics["converged"] = posterior.diagnostics["converged"]
                if "n_steps" in posterior.diagnostics:
                    diagnostics["n_steps"] = int(posterior.diagnostics["n_steps"])

        elif method.startswith("mcmc_"):
            if hasattr(posterior, "diagnostics"):
                diag = posterior.diagnostics
                diagnostics["n_divergent"] = int(diag.get("n_divergent", 0))
                diagnostics["n_samples"] = int(diag.get("n_samples", 0))

        elif method == "nss":
            if hasattr(posterior, "diagnostics"):
                diag = posterior.diagnostics
                diagnostics["n_live"] = int(diag.get("n_live", 0))
                diagnostics["log_z"] = float(diag.get("log_z", 0.0))
                if hasattr(posterior, "samples") and isinstance(posterior.samples, dict):
                    first_key = next(iter(posterior.samples))
                    diagnostics["n_samples"] = int(posterior.samples[first_key].shape[0])

        elif method in ["vi", "vi_linear"] and hasattr(posterior, "diagnostics"):
            diag = posterior.diagnostics
            diagnostics["final_kl"] = float(diag.get("final_kl", 0.0))
            diagnostics["n_iterations"] = int(diag.get("n_iterations", 0))

    # CRITICAL: Delete posterior to free memory (samples arrays can be huge)
    # For VI with 2000 samples × D=73: ~85MB per posterior object
    del posterior
    gc.collect()  # Force garbage collection to release memory immediately

    display_method = method
    if method == "map" and "optimizer" in method_kwargs:
        display_method = f"map/{method_kwargs['optimizer']}"

    return {
        "scenario": scenario_name,
        "method": display_method,
        "n_free": n_free,
        "success": success,
        "error": error_msg,
        "inference_sec": t_inference,
        "memory_mb": peak_mem_delta,
        "diagnostics": diagnostics,
    }


def main():
    print()
    print("=" * 90)
    print("  Comprehensive Inference Engine Benchmark")
    print("  (shared SEDModel/Fitter per scenario — cached loss/grad functions)")
    print("=" * 90)
    print()

    scenarios = []

    # ===== SCENARIO A1: Simple optical fit (D=7) =====
    scenarios.append(
        {
            "name": "A1_optical_simple",
            "spec_kwargs": {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
                "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
                "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
                "sfh_tsnorm_log_total_mass": Uniform(8.0, 12.0),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_tsnorm_skew": 0.1,
                "sfh_tsnorm_peak_lbt_gyr": 3.0,
                "sfh_tsnorm_width_gyr": 1.2,
                "sfh_tsnorm_trunc": 0.3,
                "sfh_tsnorm_log_total_mass": 11.0,
                "met_logzsol": -0.3,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "methods": [
                ("map", {}),
                ("map", {"optimizer": "lbfgs"}),
                ("laplace", {}),
                ("pathfinder", {"n_samples": 500}),
                ("mcmc_nuts", {"n_warmup": 200, "n_samples": 300}),
                ("mcmc_hmc", {"n_warmup": 100, "n_samples": 200, "n_leapfrog_steps": 10}),
                ("mcmc_dynamic_hmc", {"n_warmup": 100, "n_samples": 200}),
                ("mcmc_ghmc", {"n_warmup": 100, "n_samples": 200}),
                ("mcmc_mclmc", {"n_warmup": 200, "n_samples": 300}),
                ("mcmc_adjusted_mclmc", {"n_warmup": 200, "n_samples": 300}),
                ("vi", {"n_iterations": 3, "n_samples": 8}),
                ("nss", {"n_live": 200, "max_iterations": 2000}),
            ],
        }
    )

    # ===== SCENARIO A2: Mid-complexity with metallicity (D=8) =====
    scenarios.append(
        {
            "name": "A2_optical_met",
            "spec_kwargs": {
                "mean_sfh_type": "dpl",
                "sfh_dpl_alpha": Uniform(0.1, 3.0),
                "sfh_dpl_beta": Uniform(0.1, 3.0),
                "sfh_dpl_tau_gyr": Uniform(0.1, 10.0),
                "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
                "met_logzsol": Gaussian(-0.3, 0.3, -1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": LogUniform(0.01, 2.0),
                "dust_slope": Uniform(-1.5, 0.5),
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_dpl_alpha": 1.0,
                "sfh_dpl_beta": 0.5,
                "sfh_dpl_tau_gyr": 2.0,
                "sfh_dpl_log_total_mass": 11.2,
                "met_logzsol": -0.2,
                "dust_tau_bc": 0.6,
                "dust_tau_diff": 0.4,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "methods": [
                ("map", {}),
                ("map", {"optimizer": "lbfgs"}),
                ("mcmc_nuts", {"n_warmup": 200, "n_samples": 300}),
                ("vi", {"n_iterations": 3, "n_samples": 8}),
                ("nss", {"n_live": 200, "max_iterations": 2000}),
            ],
        }
    )

    # ===== SCENARIO A3: Stochastic SFH (D~9) =====
    scenarios.append(
        {
            "name": "A3_stochastic_sfh",
            "spec_kwargs": {
                "mean_sfh_type": ["dense_basis", "field"],
                "sfh_dbp_log_total_mass": Uniform(9.0, 11.5),
                "sfh_dbp_tx_frac_0": Uniform(0.0, 1.0),
                "sfh_dbp_tx_frac_1": Uniform(0.0, 1.0),
                "sfh_dbp_tx_frac_2": Uniform(0.0, 1.0),
                "sfh_field_psd_type": "power_law",
                "sfh_field_psd_sigma": LogUniform(0.01, 1.0),
                "sfh_field_psd_tau_myr": LogUniform(10.0, 1000.0),
                "met_logzsol": Uniform(-1.5, 0.2),
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": LogUniform(0.01, 2.0),
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "true_params": {
                "sfh_dbp_log_total_mass": 10.2,
                "sfh_dbp_tx_frac_0": 0.3,
                "sfh_dbp_tx_frac_1": 0.4,
                "sfh_dbp_tx_frac_2": 0.3,
                "sfh_field_psd_sigma": 0.3,
                "sfh_field_psd_tau_myr": 100.0,
                "met_logzsol": -0.3,
                "dust_tau_bc": 0.5,
                "dust_tau_diff": 0.3,
                "dust_slope": -0.7,
                "redshift": 1.0,
            },
            "methods": [
                ("map", {}),
                ("map", {"optimizer": "lbfgs"}),
                ("vi", {"n_iterations": 3, "n_samples": 8}),
                ("nss", {"n_live": 200, "max_iterations": 2000}),
            ],
        }
    )

    # Run all scenarios
    all_results = []
    key = jr.PRNGKey(42)

    for scenario in scenarios:
        print(f"\n{'=' * 90}")
        print(f"  {scenario['name'].upper()}")
        print(f"{'=' * 90}\n")

        # Build model + fitter ONCE per scenario
        print("  Building model + fitter...", end=" ", flush=True)
        t0 = time.perf_counter()
        fitter = build_scenario_fitter(
            scenario["spec_kwargs"],
            scenario["true_params"],
        )
        t_build = time.perf_counter() - t0
        if fitter is None:
            print("SSP data not found, skipping")
            continue
        print(f"done ({t_build:.1f}s, D={len(fitter.spec.free_params)})")

        for method, method_kwargs in scenario["methods"]:
            opt = method_kwargs.get("optimizer", "")
            label = f"{method}/{opt}" if opt else method
            print(f"  Running {label}...", end=" ", flush=True)

            key, subkey = jr.split(key)
            result = run_inference_method(
                scenario["name"],
                fitter,
                method,
                method_kwargs,
                key=subkey,
            )
            all_results.append(result)

            if result["success"]:
                print(f"done {result['inference_sec']:.1f}s")
            else:
                print(f"FAILED: {result['error']}")

        # Clear JAX compilation cache and force GC between scenarios
        # to prevent XLA executable accumulation (each method compiles different paths)
        print("  Clearing JAX cache and running GC...", end=" ", flush=True)
        jax.clear_caches()
        gc.collect()
        print("done")

    # ===== CACHE PROFILING: cold → cached → new-fitter =====
    print(f"\n{'=' * 90}")
    print("  CACHE PROFILING (A1 MCMC methods)")
    print(f"{'=' * 90}\n")

    a1_fitter = build_scenario_fitter(
        scenarios[0]["spec_kwargs"],
        scenarios[0]["true_params"],
    )
    mcmc_methods_a1 = [m for m in scenarios[0]["methods"] if m[0].startswith("mcmc_")]

    # Cold run (JIT compile + adaptation)
    print("  --- Cold (JIT + adaptation) ---")
    for method, method_kwargs in mcmc_methods_a1:
        print(f"  {method}...", end=" ", flush=True)
        key, subkey = jr.split(key)
        r = run_inference_method("A1_cold", a1_fitter, method, method_kwargs, key=subkey)
        all_results.append(r)
        if r["success"]:
            print(f"{r['inference_sec']:.2f}s")
        else:
            print(f"FAIL: {r['error']}")

    # Cached run (adaptation skipped, JIT warm)
    print("  --- Cached (skip adaptation, JIT warm) ---")
    for method, method_kwargs in mcmc_methods_a1:
        print(f"  {method}...", end=" ", flush=True)
        key, subkey = jr.split(key)
        r = run_inference_method("A1_cached", a1_fitter, method, method_kwargs, key=subkey)
        all_results.append(r)
        if r["success"]:
            print(f"{r['inference_sec']:.2f}s")
        else:
            print(f"FAIL: {r['error']}")

    # New fitter, DIFFERENT galaxy, same SEDModel, REVERSED engine order
    print("  --- New fitter, different galaxy, reversed engine order ---")
    model_shared = a1_fitter.model
    different_params = {
        "sfh_tsnorm_skew": -0.3,
        "sfh_tsnorm_peak_lbt_gyr": 7.0,
        "sfh_tsnorm_width_gyr": 0.5,
        "sfh_tsnorm_trunc": 0.8,
        "sfh_tsnorm_log_total_mass": 10.2,
        "met_logzsol": -1.0,
        "dust_tau_bc": 1.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 1.0,
    }
    obs2 = model_shared.mock(different_params, snr=15.0, key=jr.PRNGKey(777))
    a1_fitter2 = Fitter(model_shared, obs2.flux_obs, obs2.noise)
    for method, method_kwargs in reversed(mcmc_methods_a1):
        print(f"  {method}...", end=" ", flush=True)
        key, subkey = jr.split(key)
        r = run_inference_method("A1_new_gal", a1_fitter2, method, method_kwargs, key=subkey)
        all_results.append(r)
        if r["success"]:
            print(f"{r['inference_sec']:.2f}s")
        else:
            print(f"FAIL: {r['error']}")

    # Summary table
    print()
    print("=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print()
    print(f"{'Scenario':<20} {'Method':<22} {'D':<4} {'Status':<8} {'Time':<10} {'Memory':<10}")
    print("-" * 90)

    for r in all_results:
        status = "OK" if r["success"] else "FAIL"
        time_str = f"{r['inference_sec']:.2f}s" if r["success"] else "-"
        mem_str = f"{r['memory_mb']:.0f} MB" if r["success"] else "-"

        print(
            f"{r['scenario']:<20} "
            f"{r['method']:<22} "
            f"{r['n_free']:<4} "
            f"{status:<8} "
            f"{time_str:<10} "
            f"{mem_str:<10}"
        )

    # Detailed diagnostics
    print()
    print("=" * 90)
    print("  DETAILED DIAGNOSTICS")
    print("=" * 90)

    for r in all_results:
        if r["success"] and r["diagnostics"]:
            print(f"\n{r['scenario']} / {r['method']}:")
            for k, val in r["diagnostics"].items():
                print(f"  {k}: {val}")

    # Performance insights
    print()
    print("=" * 90)
    print("  KEY INSIGHTS")
    print("=" * 90)
    print()

    by_scenario = {}
    for r in all_results:
        if r["success"]:
            if r["scenario"] not in by_scenario:
                by_scenario[r["scenario"]] = []
            by_scenario[r["scenario"]].append(r)

    for scenario_name, results in by_scenario.items():
        print(f"\n{scenario_name}:")
        results_sorted = sorted(results, key=lambda x: x["inference_sec"])
        fastest = results_sorted[0]
        print(f"  Fastest: {fastest['method']} ({fastest['inference_sec']:.1f}s)")
        print(f"  D = {fastest['n_free']} free parameters")

        if len(results_sorted) > 1:
            slowest = results_sorted[-1]
            speedup = slowest["inference_sec"] / fastest["inference_sec"]
            print(
                f"  Slowest: {slowest['method']} "
                f"({slowest['inference_sec']:.1f}s, "
                f"{speedup:.1f}x slower)"
            )

    print()
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
