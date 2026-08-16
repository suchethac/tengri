"""Diagnose whether repeated inference calls get JAX JIT cache hits.

Tests the hypothesis: do NUTS/VI/NSS/Laplace/Pathfinder closures
get recompiled on the second call with the same fitter?

If 2nd call ≈ 1st call → closure identity busts the cache → explicit caching needed.
If 2nd call << 1st call → JAX trace cache handles it → no work needed.
"""

import time
from pathlib import Path

import jax
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Gaussian,
    SEDModel,
    Uniform,
    load_filter_set,
    load_ssp_data,
)
from tengri import Parameters

data_dir = Path(__file__).resolve().parents[1] / "data"
ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

if not ssp_file.is_file():
    raise FileNotFoundError(f"SSP data not found: {ssp_file}")


def build_fitter():
    ssp = load_ssp_data(str(ssp_file))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

    spec = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_skew=Uniform(-0.5, 0.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 10.0),
        sfh_tsnorm_width_gyr=Uniform(0.1, 3.0),
        sfh_tsnorm_trunc=Uniform(0.01, 1.0),
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Gaussian(-0.3, 0.3, -1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=1.0,
    )
    model = SEDModel(spec, ssp, filters=filters)

    true_params = {
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

    mock = model.mock(true_params, snr=20.0, key=jr.PRNGKey(42))
    return Fitter(model, mock.flux_obs, mock.noise)


def time_method(fitter, method, kwargs, key):
    t0 = time.perf_counter()
    try:
        fitter.run(method=method, key=key, verbose=False, **kwargs)
        return time.perf_counter() - t0, True
    except Exception as e:
        return time.perf_counter() - t0, False


def main():
    print("=" * 80)
    print("  JIT Cache Hit Diagnostic")
    print("  Does the 2nd call recompile or cache-hit?")
    print("=" * 80)
    print()

    fitter = build_fitter()
    key = jr.PRNGKey(0)

    engines = [
        ("map/adam", "map", {"optimizer": "adam", "n_steps": 50}),
        ("map/lbfgs", "map", {"optimizer": "lbfgs", "n_steps": 50}),
        ("laplace", "laplace", {"n_map_steps": 50, "n_samples": 50}),
        ("mcmc_nuts", "mcmc_nuts", {"n_warmup": 50, "n_samples": 50}),
        ("pathfinder", "pathfinder", {"n_samples": 50}),
        ("vi", "vi", {"n_iterations": 2, "n_samples": 4}),
    ]

    print(f"{'Engine':<15} {'1st call':>10} {'2nd call':>10} {'3rd call':>10} {'Speedup':>10}")
    print("-" * 60)

    for label, method, kwargs in engines:
        key, k1, k2, k3 = jr.split(key, 4)

        t1, ok1 = time_method(fitter, method, kwargs, k1)
        t2, ok2 = time_method(fitter, method, kwargs, k2)
        t3, ok3 = time_method(fitter, method, kwargs, k3)

        if ok1 and ok2:
            speedup = t1 / t2
            print(f"{label:<15} {t1:>9.2f}s {t2:>9.2f}s {t3:>9.2f}s {speedup:>9.1f}×")
        elif ok1:
            print(f"{label:<15} {t1:>9.2f}s  FAIL  FAIL")
        else:
            print(f"{label:<15}  FAIL")

    print()
    print("Speedup = 1st / 2nd. If >5× → JAX caches properly. If ~1× → needs explicit caching.")


if __name__ == "__main__":
    main()
