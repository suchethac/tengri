"""Focused compile-once verification.

Tests that a second Fitter with DIFFERENT galaxy data and REVERSED
engine order reuses all cached compilation and adaptation.
"""

import time
import warnings

import jax
import jax.random as jr

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

from pathlib import Path

from tengri import (
    Fitter,
    Parameters,
    SEDModel,
    Uniform,
    load_filter_set,
    load_ssp_data,
)


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not ssp_file.is_file():
        print("SSP file not found, skipping")
        return

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

    spec_kwargs = {
        "mean_sfh_type": "tsnorm",
        "sfh_tsnorm_skew": Uniform(-0.5, 0.5),
        "sfh_tsnorm_peak_lbt_gyr": Uniform(0.5, 10.0),
        "sfh_tsnorm_width_gyr": Uniform(0.1, 3.0),
        "sfh_tsnorm_trunc": Uniform(0.01, 1.0),
        "sfh_tsnorm_log_total_mass": Uniform(7.0, 12.5),
        "met_logzsol": Uniform(-1.5, 0.2),
        "dust_tau_bc": Uniform(0.0, 2.0),
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 1.0,
    }

    # Galaxy A: young, dusty, high SFR
    params_a = {
        "sfh_tsnorm_skew": 0.1,
        "sfh_tsnorm_peak_lbt_gyr": 3.0,
        "sfh_tsnorm_width_gyr": 1.2,
        "sfh_tsnorm_trunc": 0.3,
        "sfh_tsnorm_log_total_mass": 1.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 1.0,
    }

    # Galaxy B: old, metal-poor, heavy dust — completely different SED
    params_b = {
        "sfh_tsnorm_skew": -0.4,
        "sfh_tsnorm_peak_lbt_gyr": 8.0,
        "sfh_tsnorm_width_gyr": 0.3,
        "sfh_tsnorm_trunc": 0.9,
        "sfh_tsnorm_log_total_mass": -0.5,
        "met_logzsol": -1.2,
        "dust_tau_bc": 1.8,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 1.0,
    }

    spec = Parameters(**spec_kwargs)
    model = SEDModel(spec, ssp, filters=filters)

    # Build fitters via mock observations — use Fitter constructor directly
    obs_a = model.mock(params_a, snr=20.0, key=jr.PRNGKey(42))
    fitter_a = Fitter(model, obs_a.flux_obs, obs_a.noise)

    obs_b = model.mock(params_b, snr=12.0, key=jr.PRNGKey(999))
    fitter_b = Fitter(model, obs_b.flux_obs, obs_b.noise)

    key = jr.PRNGKey(0)

    methods = [
        ("mcmc_nuts", {"n_warmup": 200, "n_samples": 300}),
        ("mcmc_hmc", {"n_warmup": 100, "n_samples": 200, "n_leapfrog_steps": 10}),
        ("mcmc_dynamic_hmc", {"n_warmup": 100, "n_samples": 200}),
        ("mcmc_ghmc", {"n_warmup": 100, "n_samples": 200}),
        ("mcmc_mclmc", {"n_warmup": 200, "n_samples": 300}),
        ("mcmc_adjusted_mclmc", {"n_warmup": 200, "n_samples": 300}),
    ]

    print("=" * 70)
    print("  Galaxy A — cold start (JIT + adaptation)")
    print("=" * 70)

    times_a = {}
    for method, kwargs in methods:
        key, subkey = jr.split(key)
        t0 = time.perf_counter()
        fitter_a.run(method=method, key=subkey, verbose=False, **kwargs)
        dt = time.perf_counter() - t0
        times_a[method] = dt
        print(f"  {method:<25} {dt:.2f}s")

    print()
    print("=" * 70)
    print("  Galaxy B — different data, REVERSED order (compile-once)")
    print("=" * 70)

    times_b = {}
    for method, kwargs in reversed(methods):
        key, subkey = jr.split(key)
        t0 = time.perf_counter()
        fitter_b.run(method=method, key=subkey, verbose=False, **kwargs)
        dt = time.perf_counter() - t0
        times_b[method] = dt
        print(f"  {method:<25} {dt:.2f}s")

    print()
    print("=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'Method':<25} {'Galaxy A (cold)':<18} {'Galaxy B (warm)':<18} {'Speedup':<10}")
    print("-" * 70)
    for method, _ in methods:
        ta = times_a[method]
        tb = times_b[method]
        speedup = ta / tb if tb > 0 else float("inf")
        print(f"  {method:<25} {ta:>8.2f}s       {tb:>8.2f}s       {speedup:>6.1f}x")

    all_ok = all(times_b[m] < max(times_a[m] * 0.5, 3.0) for m, _ in methods)
    print()
    if all_ok:
        print("  PASS: All methods show compile-once speedup")
    else:
        print("  WARN: Some methods did not show expected speedup")


if __name__ == "__main__":
    main()
