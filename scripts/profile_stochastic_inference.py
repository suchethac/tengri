"""Profile stochastic SFH inference to identify runtime bottlenecks.

Measures:
1. JIT compilation time (first call)
2. Inference runtime (post-JIT)
3. Component-by-component breakdown
4. Cache verification (no recompilation on subsequent runs)
"""

import time
import tracemalloc
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

# Force CPU (Metal unreliable)
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


def load_fixtures():
    """Load SSP data, filters, and create mock observation."""
    # SSP data (matching test fixture)
    ssp_path = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not ssp_path.exists():
        raise FileNotFoundError(f"SSP data not found: {ssp_path}")

    ssp_data = load_ssp_data(str(ssp_path))

    # Filters (HST + VISTA + IRAC) - matching test fixture
    filter_names = [
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

    filter_data = load_filter_set(filter_names)

    # Mock data at z=1 (matching test fixture approach)
    n_bands = len(filter_names)
    rng = np.random.default_rng(42)

    # Synthetic flux distribution: log-uniform from 0.1 to 100 μJy
    log_flux = rng.uniform(np.log10(0.1), np.log10(100.0), size=n_bands)
    flux_mjy = 10.0**log_flux * 1e-3  # Convert μJy → mJy

    # Convert to erg/s/cm²/Hz (IAU convention)
    # 1 mJy = 1e-26 erg/s/cm²/Hz
    flux_cgs = flux_mjy * 1e-26

    # SNR ~ 10 per band
    noise_cgs = flux_cgs / 10.0

    # Add Gaussian noise
    flux_obs_cgs = flux_cgs + rng.normal(0, noise_cgs)

    mock_data = {
        "flux": jnp.array(flux_obs_cgs),
        "flux_unc": jnp.array(noise_cgs),
        "redshift": 1.0,
    }

    obs = Observation(photometry=Photometry.from_filter_set(filter_data))

    return ssp_data, obs, mock_data


def create_stochastic_params(redshift):
    """Create stochastic SFH parameter spec matching A4 test."""
    return Parameters(
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
        redshift=Fixed(redshift),
        n_grid=64,
    )


def create_simple_params(redshift):
    """Create simple parametric SFH for comparison."""
    return Parameters(
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
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(redshift),
    )


def measure_compilation_and_runtime(fitter, method, rng_key, n_warmup=5, **kwargs):
    """Measure JIT compilation time and post-JIT runtime."""
    print(f"\n{'=' * 60}")
    print(f"Method: {method}")
    print(f"Parameters: {kwargs}")
    print(f"{'=' * 60}")

    # Clear JAX cache to measure cold-start compilation
    print("Clearing JAX cache...")
    jax.clear_caches()

    # Measure JIT compilation time (first call)
    print("First call (JIT compilation + execution)...")
    tracemalloc.start()
    t_start_jit = time.perf_counter()
    posterior_jit = fitter.run(method, key=rng_key, **kwargs)
    t_jit = time.perf_counter() - t_start_jit
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  JIT time: {t_jit:.1f}s")
    print(f"  Peak RAM: {peak_ram / (1024**2):.2f} MB")

    # Measure post-JIT runtime (should NOT recompile)
    print(f"\nWarmup runs (n={n_warmup}, should be fast - no recompilation)...")
    runtimes = []
    for i in range(n_warmup):
        key_i = jax.random.fold_in(rng_key, abs(hash(f"warmup_{i}")) % (2**31))
        t_start = time.perf_counter()
        _ = fitter.run(method, key=key_i, **kwargs)
        t_runtime = time.perf_counter() - t_start
        runtimes.append(t_runtime)
        print(f"  Run {i + 1}: {t_runtime:.3f}s")

    mean_runtime = float(np.mean(runtimes))
    std_runtime = float(np.std(runtimes))

    print(f"\nSummary:")
    print(f"  JIT (first call): {t_jit:.1f}s")
    print(f"  Runtime (mean ± std): {mean_runtime:.3f}s ± {std_runtime:.3f}s")
    print(f"  Runtime/JIT ratio: {mean_runtime / t_jit:.2%}")

    # Check if runtimes are similar (cache working)
    if std_runtime / mean_runtime > 0.1:
        print(
            f"  ⚠️  WARNING: High variance in runtime ({std_runtime / mean_runtime:.1%}) - possible recompilation!"
        )
    else:
        print(f"  ✓ Cache working (variance {std_runtime / mean_runtime:.1%})")

    return {
        "jit_sec": t_jit,
        "mean_runtime_sec": mean_runtime,
        "std_runtime_sec": std_runtime,
        "peak_ram_mb": peak_ram / (1024**2),
    }


def profile_component_breakdown(ssp_data, obs, mock_data):
    """Profile component-by-component to find bottlenecks."""
    print("\n" + "=" * 60)
    print("COMPONENT BREAKDOWN PROFILING")
    print("=" * 60)

    redshift = mock_data["redshift"]

    # Test configurations with increasing complexity
    configs = [
        (
            "Stellar only",
            {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 12.0),
                "sfh_tsnorm_width_gyr": Uniform(0.2, 5.0),
                "sfh_tsnorm_skew": Uniform(-1.0, 1.0),
                "sfh_tsnorm_trunc": Uniform(1.0, 10.0),
                "met_logzsol": Uniform(-2.0, 0.2),
                "redshift": Fixed(redshift),
            },
        ),
        (
            "Stellar + Dust",
            {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 12.0),
                "sfh_tsnorm_width_gyr": Uniform(0.2, 5.0),
                "sfh_tsnorm_skew": Uniform(-1.0, 1.0),
                "sfh_tsnorm_trunc": Uniform(1.0, 10.0),
                "met_logzsol": Uniform(-2.0, 0.2),
                "dust_law_bc": "salim_sbl18",
                "dust_tau_bc": Uniform(0.0, 3.0),
                "dust_tau_diff": Uniform(0.0, 2.0),
                "redshift": Fixed(redshift),
            },
        ),
        (
            "Stellar + Dust + Nebular",
            {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 12.0),
                "sfh_tsnorm_width_gyr": Uniform(0.2, 5.0),
                "sfh_tsnorm_skew": Uniform(-1.0, 1.0),
                "sfh_tsnorm_trunc": Uniform(1.0, 10.0),
                "met_logzsol": Uniform(-2.0, 0.2),
                "dust_law_bc": "salim_sbl18",
                "dust_tau_bc": Uniform(0.0, 3.0),
                "dust_tau_diff": Uniform(0.0, 2.0),
                "nebular_ssp": True,
                "redshift": Fixed(redshift),
            },
        ),
        (
            "Full (+ IGM)",
            {
                "mean_sfh_type": "tsnorm",
                "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
                "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 12.0),
                "sfh_tsnorm_width_gyr": Uniform(0.2, 5.0),
                "sfh_tsnorm_skew": Uniform(-1.0, 1.0),
                "sfh_tsnorm_trunc": Uniform(1.0, 10.0),
                "met_logzsol": Uniform(-2.0, 0.2),
                "dust_law_bc": "salim_sbl18",
                "dust_tau_bc": Uniform(0.0, 3.0),
                "dust_tau_diff": Uniform(0.0, 2.0),
                "nebular_ssp": True,
                "apply_igm": True,
                "redshift": Fixed(redshift),
            },
        ),
    ]

    results = []
    for name, param_dict in configs:
        print(f"\n{name}:")
        print(f"  D = {len([k for k, v in param_dict.items() if isinstance(v, (Uniform,))])}")

        params = Parameters(**param_dict)
        model = SEDModel(params, ssp_data, observation=obs)
        fitter = Fitter(model, data=mock_data["flux"], noise=mock_data["flux_unc"])

        rng_key = jax.random.PRNGKey(42)

        # Just measure MAP (fastest inference method)
        result = measure_compilation_and_runtime(fitter, "map", rng_key, n_warmup=3)
        result["name"] = name
        results.append(result)

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPONENT BREAKDOWN SUMMARY")
    print("=" * 60)
    print(f"{'Component':<25} {'JIT (s)':<10} {'Runtime (ms)':<15} {'RAM (MB)':<10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<25} {r['jit_sec']:>8.1f}  {r['mean_runtime_sec'] * 1000:>12.1f}  {r['peak_ram_mb']:>8.1f}"
        )

    return results


def main():
    """Run comprehensive profiling."""
    print("Loading fixtures...")
    ssp_data, obs, mock_data = load_fixtures()

    # 1. Component breakdown (simple models)
    component_results = profile_component_breakdown(ssp_data, obs, mock_data)

    # 2. Stochastic vs simple comparison
    print("\n" + "=" * 60)
    print("STOCHASTIC VS SIMPLE SFH COMPARISON")
    print("=" * 60)

    for name, param_fn in [
        ("Simple parametric (tsnorm)", create_simple_params),
        ("Stochastic (dense_basis+field)", create_stochastic_params),
    ]:
        print(f"\n{name}:")
        params = param_fn(mock_data["redshift"])
        print(f"  Free params: {len(params.free_params)}")
        print(f"  n_grid: {params.get('n_grid', 'N/A')}")

        model = SEDModel(params, ssp_data, observation=obs)
        fitter = Fitter(model, data=mock_data["flux"], noise=mock_data["flux_unc"])

        rng_key = jax.random.PRNGKey(42)

        # Use VI for stochastic, MAP for simple (to match test)
        if "stochastic" in name.lower():
            method = "vi"
            kwargs = {"n_iterations": 10, "n_samples_per_iteration": 4}
        else:
            method = "map"
            kwargs = {}

        result = measure_compilation_and_runtime(fitter, method, rng_key, n_warmup=3, **kwargs)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
