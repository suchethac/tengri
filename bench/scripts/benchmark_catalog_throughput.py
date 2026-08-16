#!/usr/bin/env python
"""Catalog throughput: vectorized per-galaxy NUTS sampling, galaxies/s vs K and devices.

Measures the Track-A catalog sampling path — ``CatalogFitter.run("mcmc_nuts",
forward_chunk_size=K, devices=...)`` — which fits many independent galaxies in
parallel and returns a posterior per galaxy.

Reports, for a mock photometric catalog:

* the **forward_chunk_size (K) sweep**: warm galaxies/second as K grows, so you
  can find the K that saturates the accelerator (cold compile is reported
  separately — it is paid once and cached).
* **device scaling** (only when more than one device is visible): single-device
  vs ``devices="all"`` throughput, i.e. how well the galaxy axis shards.

Device-agnostic: it does NOT force a platform, so on a CUDA box / Sherlock GPU
node it runs on the GPU. Emulate multiple devices on CPU with
``XLA_FLAGS=--xla_force_host_platform_device_count=N``.

Usage::

    # CPU (or GPU if JAX picks one):
    python bench/scripts/benchmark_catalog_throughput.py

    # Sherlock GPU node, bigger catalog:
    python bench/scripts/benchmark_catalog_throughput.py --n-gal 512 2048 --chunk 32 128 512

    # Emulate 4 devices on CPU to smoke the shard path:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python bench/scripts/benchmark_catalog_throughput.py --shard
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.inference.catalog_fitter import CatalogFitter
from tengri.observation.photometry import FilterCurve


def _load_or_synth_ssp():
    """Real SSP if on disk (realistic numbers), else a portable synthetic grid."""
    for name in (
        "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    ):
        path = os.path.join("data", name)
        if os.path.exists(path):
            from tengri.sps.dsps_wrapper import load_ssp_data

            return load_ssp_data(path), f"real:{name}"
    from tengri.sps.dsps_wrapper import SSPData

    wave = jnp.linspace(3000.0, 10000.0, 100)
    ages = jnp.linspace(-1.0, 1.14, 20)
    flux = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (3, 20, 100))) * 1e-3 + 1e-5
    ssp = SSPData(
        ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages, ssp_lgmet=jnp.array([-1.5, -0.5, 0.0])
    )
    return ssp, "synthetic"


def build_model(ssp, ssp_tag):
    """Small photometric dpl model: mass + alpha free (D=2), the rest pinned."""
    if ssp_tag.startswith("real"):
        phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    else:
        phot = Photometry(
            filters=tuple(
                FilterCurve(
                    wave=jnp.linspace(c * 0.9, c * 1.1, 40), trans=jnp.ones(40) * 0.5, name=f"b{i}"
                )
                for i, c in enumerate((3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
            )
        )
    obs = Observation(photometry=phot)
    met = Fixed(1.0) if ssp_tag == "synthetic" else Uniform(-1.5, 0.2)
    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=met,
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, ssp, observation=obs)


def make_catalog(model, n_gal, key):
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        tp = dict(model.spec.sample(k))
        flux = model.predict_photometry(tp)
        noise = jnp.abs(flux) * 0.05
        galaxies.append(
            {
                "flux_obs": flux + noise * jax.random.normal(jax.random.fold_in(k, 1), flux.shape),
                "noise": noise,
            }
        )
    return galaxies


def time_run(cat, K, devices, key, run_kw):
    t0 = time.perf_counter()
    cp = cat.run(
        "mcmc_nuts", key=key, forward_chunk_size=K, devices=devices, verbose=False, **run_kw
    )
    # block on the first galaxy's samples to ensure the async dispatch finished
    jax.block_until_ready(cp[0].samples)
    return time.perf_counter() - t0, cp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-gal", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--chunk", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--shard", action="store_true", help="also time devices='all' scaling")
    args = ap.parse_args(argv)

    dev = jax.devices()
    print("tengri catalog-throughput benchmark")
    print(
        f"  backend: {dev[0].platform}  |  devices: {len(dev)}  |  x64: {jax.config.x64_enabled}"
    )

    ssp, ssp_tag = _load_or_synth_ssp()
    model = build_model(ssp, ssp_tag)
    print(f"  ssp: {ssp_tag}  |  free params: {list(model.spec.free_params)}")
    run_kw = dict(n_warmup=args.warmup, n_burnin=args.burnin, n_samples=args.samples)
    key = jax.random.PRNGKey(0)

    print(f"\n  forward_chunk_size (K) sweep  [warmup={args.warmup}, samples={args.samples}]")
    print(f"  {'N':>6} {'K':>5} {'cold_s':>9} {'warm_s':>9} {'gal/s(warm)':>12}")
    biggest = max(args.n_gal)
    full_catalog = make_catalog(model, biggest, key)
    for n_gal in args.n_gal:
        cat = CatalogFitter(model, full_catalog[:n_gal], data_type="photometry")
        for K in args.chunk:
            if n_gal < K:
                continue
            cold, _ = time_run(cat, K, None, key, run_kw)  # includes compile
            warm, _ = time_run(cat, K, None, key, run_kw)  # cached
            print(
                f"  {n_gal:>6} {K:>5} {cold:>9.2f} {warm:>9.2f} {n_gal / warm:>12.1f}",
                flush=True,
            )

    if args.shard and len(dev) > 1:
        n_gal = biggest
        K = args.chunk[len(args.chunk) // 2]
        # pad-match n_gal to lcm(K, n_dev) already handled internally; just compare.
        cat = CatalogFitter(model, full_catalog[:n_gal], data_type="photometry")
        print(f"\n  device scaling  [N={n_gal}, K={K}, {len(dev)} devices]")
        _ = time_run(cat, K, None, key, run_kw)  # warm the single-device compile
        s1, _ = time_run(cat, K, None, key, run_kw)
        _ = time_run(cat, K, "all", key, run_kw)  # warm the sharded compile
        sN, _ = time_run(cat, K, "all", key, run_kw)
        print(f"  {'single':>8}: {s1:>8.2f} s  ({n_gal / s1:>8.1f} gal/s)")
        print(
            f"  {'sharded':>8}: {sN:>8.2f} s  ({n_gal / sN:>8.1f} gal/s)  speedup {s1 / sN:.2f}x"
        )
    elif args.shard:
        print(
            "\n  --shard requested but only 1 device visible; "
            "set XLA_FLAGS=--xla_force_host_platform_device_count=N to emulate."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
