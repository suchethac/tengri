"""Real-path compile-time benchmark for PopulationFitter / CatalogFitter.

Runs the actual production code with ``n_iterations=2`` to isolate compile
cost from execution cost.  Persistent JAX cache is enabled (default), so the
second run at the same N should be much faster — diff = compile time.

Reports per-N:
    setup_s        : PopulationFitter() construction + Python init loops
    cold_run_s     : first .run() call (compile + 2 KL steps)
    warm_run_s     : second .run() call (cache hit, ~just execution)
    compile_proxy  : cold - warm  (≈ compile cost)

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_jit_real_path.py \\
        --Ns 64 256 1024 4096 --method native_vi_linear --K 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import tengri  # auto-enables persistent compilation cache  # noqa: F401
from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.inference.hierarchical import PopulationFitter
from tengri.sps.dsps_wrapper import load_ssp_data

jax.config.update("jax_enable_x64", True)

SSP_FILE = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

FILTERS = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
]


def make_spec() -> Parameters:
    return Parameters(
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        mean_sfh_type=["tsnorm", "field"],
        n_grid=64,  # smaller for faster bench
    )


def make_galaxies(n_gal: int, ssp_data, obs: Observation, key) -> list[dict]:
    spec = make_spec()
    template = SEDModel(spec, ssp_data, observation=obs)
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        true_params = template.spec.sample(k)
        true_params["sfh_field_psd_sigma"] = jnp.array(2.0)
        true_params["sfh_field_psd_tau_myr"] = jnp.array(20.0)
        flux = template.predict_photometry(true_params)
        noise = jnp.abs(flux) * 0.05 + 1e-3
        flux_obs = flux + noise * jax.random.normal(k, shape=flux.shape)
        galaxies.append({"flux_obs": flux_obs, "noise": noise})
    return galaxies


def model_factory(psd_sigma=1.0, psd_tau_myr=50.0, *, ssp_data=None, obs=None):
    spec = make_spec()
    return SEDModel(spec, ssp_data, observation=obs)


def time_run(
    pop: PopulationFitter, method: str, K: int, n_iter: int = 2, n_samp: int = 3
) -> float:
    t0 = time.perf_counter()
    pop.run(
        method=method,
        n_iterations=n_iter,
        n_samples=n_samp,
        n_posterior_samples=2,
        forward_chunk_size=K,
        verbose=False,
    )
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--method", default="native_vi_linear")
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--out", default="bench/results/jit_real_path_benchmark.json")
    args = p.parse_args()

    print("Loading SSP data...")
    ssp_data = load_ssp_data(SSP_FILE)
    obs = Observation(photometry=Photometry.from_names(FILTERS), spectroscopy=None)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def factory(psd_sigma=1.0, psd_tau_myr=50.0):
        return model_factory(psd_sigma, psd_tau_myr, ssp_data=ssp_data, obs=obs)

    for N in args.Ns:
        print(f"\n=== N={N} ===")
        key = jax.random.PRNGKey(42)
        t0 = time.perf_counter()
        galaxies = make_galaxies(N, ssp_data, obs, key)
        pop = PopulationFitter(factory, galaxies, data_type="photometry")
        setup_s = time.perf_counter() - t0
        print(f"  setup: {setup_s:.2f}s")

        try:
            cold = time_run(pop, args.method, args.K)
            print(f"  cold run: {cold:.2f}s")
            warm = time_run(pop, args.method, args.K)
            print(f"  warm run: {warm:.2f}s")
            compile_proxy = cold - warm
            print(f"  compile_proxy: {compile_proxy:.2f}s")
            err = None
        except Exception as exc:
            cold = -1.0
            warm = -1.0
            compile_proxy = -1.0
            err = repr(exc)
            print(f"  ERROR: {err}")

        rows.append(
            {
                "N": N,
                "K": args.K,
                "method": args.method,
                "setup_s": setup_s,
                "cold_run_s": cold,
                "warm_run_s": warm,
                "compile_proxy_s": compile_proxy,
                "error": err,
            }
        )
        out_path.write_text(json.dumps(rows, indent=2))

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
