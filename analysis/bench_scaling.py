#!/usr/bin/env python3
"""Benchmark: performance scaling across key dimensions.

Measures how forward-model and gradient wall time scale with:
  1. Parameter dimension D (GP grid size)
  2. Number of photometric bands
  3. Number of spectral pixels
  4. Catalog size (fit_batch)

Usage:
    python analysis/bench_scaling.py          # full run
    python analysis/bench_scaling.py --quick  # reduced iterations for CI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FIG_DIR, SSP_FILE, setup_matplotlib

from tengri import (
    Fitter,
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

# ── Paths ─────────────────────────────────────────────────────────
RESULTS_FILE = FIG_DIR / "scaling_results.json"

# ── Default VI iterations (minimal for benchmarking overhead) ─────
VI_ITER_DEFAULT = 2


# ── Bench utility ─────────────────────────────────────────────────
def bench(fn, n=200, warmup=3):
    """Time *fn*, returning (mean_us, last_result).

    Calls ``block_until_ready`` on JAX arrays to ensure accurate timing.
    """
    for _ in range(warmup):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    elapsed_us = (time.perf_counter() - t0) / n * 1e6
    return elapsed_us, r


def _bench_photometry(model, params, n_iter):
    """Bench forward + gradient for photometry, returning (fwd_us, grad_us)."""
    fwd_fn = jax.jit(model.predict_photometry)
    _ = fwd_fn(params)
    t_fwd, _ = bench(lambda: fwd_fn(params), n=n_iter)

    grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
    _ = grad_fn(params)
    t_grad, _ = bench(lambda: grad_fn(params), n=n_iter)
    return t_fwd, t_grad


def _bench_spectrum(model, params, n_iter):
    """Bench forward + gradient for spectroscopy, returning (fwd_us, grad_us)."""
    fwd_fn = jax.jit(model.predict_spectrum)
    _ = fwd_fn(params)
    t_fwd, _ = bench(lambda: fwd_fn(params), n=n_iter)

    grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_spectrum(p))))
    _ = grad_fn(params)
    t_grad, _ = bench(lambda: grad_fn(params), n=n_iter)
    return t_fwd, t_grad


# ── SSP loading (graceful skip) ───────────────────────────────────
def try_load_ssp():
    """Load SSP data, returning None if file is missing."""
    try:
        return load_ssp_data(str(SSP_FILE))
    except (FileNotFoundError, OSError) as exc:
        print(f"[WARN] SSP data unavailable ({exc}). Skipping tests that need it.")
        return None


# ── Shared ParamSpec factory ──────────────────────────────────────
def _make_spec(n_grid=64):
    """Create a stochastic ParamSpec with the given GP grid size."""
    return ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_field_psd_sigma=1.0,
        sfh_field_psd_tau_myr=50.0,
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        n_grid=n_grid,
    )


# ── 1. Parameter dimension D scaling ─────────────────────────────
def bench_dimension_scaling(ssp, n_iter, quick):
    """Measure forward + gradient time vs parameter dimension D."""
    print("\n" + "=" * 70)
    print("1. PARAMETER DIMENSION D SCALING")
    print("=" * 70)

    d_values = [7, 17, 37, 67, 137, 267]
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    key = jax.random.PRNGKey(0)

    results = {"D": [], "fwd_us": [], "grad_us": []}

    print(f"\n{'D':>6s}  {'Forward (us)':>14s}  {'Gradient (us)':>14s}")
    print("-" * 42)

    for n_grid in d_values:
        spec = _make_spec(n_grid=n_grid)
        model = Model(spec, ssp, observation=obs, precompute=True)
        params = spec.sample(key)

        t_fwd, t_grad = _bench_photometry(model, params, n_iter)

        d_total = len(spec.free_params)
        results["D"].append(d_total)
        results["fwd_us"].append(round(t_fwd, 1))
        results["grad_us"].append(round(t_grad, 1))

        print(f"{d_total:>6d}  {t_fwd:>14.1f}  {t_grad:>14.1f}")

    return results


# ── 2. Photometric bands scaling ─────────────────────────────────
def bench_band_scaling(ssp, n_iter, quick):
    """Measure forward + gradient time vs number of photometric bands."""
    print("\n" + "=" * 70)
    print("2. PHOTOMETRIC BANDS SCALING")
    print("=" * 70)

    # Pool of 20 distinct filters spanning UV through MIR
    all_filter_names = [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "2mass_j",
        "2mass_h",
        "2mass_ks",
        "wise_w1",
        "wise_w2",
        "wise_w3",
        "wise_w4",
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "jwst_f090w",
        "jwst_f115w",
    ]

    n_bands_values = [3, 5, 8, 10, 15, 20]
    key = jax.random.PRNGKey(1)

    results = {"n_bands": [], "fwd_us": [], "grad_us": []}

    print(f"\n{'N_bands':>8s}  {'Forward (us)':>14s}  {'Gradient (us)':>14s}")
    print("-" * 44)

    for n_bands in n_bands_values:
        filter_names = all_filter_names[:n_bands]
        obs_n = Observation(photometry=Photometry.from_names(filter_names))

        spec = _make_spec(n_grid=64)
        model = Model(spec, ssp, observation=obs_n, precompute=True)
        params = spec.sample(key)

        t_fwd, t_grad = _bench_photometry(model, params, n_iter)

        results["n_bands"].append(n_bands)
        results["fwd_us"].append(round(t_fwd, 1))
        results["grad_us"].append(round(t_grad, 1))

        print(f"{n_bands:>8d}  {t_fwd:>14.1f}  {t_grad:>14.1f}")

    return results


# ── 3. Spectral pixels scaling ────────────────────────────────────
def bench_spectral_scaling(ssp, n_iter, quick):
    """Measure forward + gradient time vs number of spectral pixels."""
    print("\n" + "=" * 70)
    print("3. SPECTRAL PIXELS SCALING")
    print("=" * 70)

    n_pix_values = [50, 100, 200, 500, 1000, 2000]
    key = jax.random.PRNGKey(2)

    results = {"n_pix": [], "fwd_us": [], "grad_us": []}

    print(f"\n{'N_pix':>8s}  {'Forward (us)':>14s}  {'Gradient (us)':>14s}")
    print("-" * 44)

    for n_pix in n_pix_values:
        spec = _make_spec(n_grid=64)
        # No filters needed for spectroscopy-only benchmark
        model = Model(spec, ssp, precompute=False)

        # Observed wavelength grid: 3800-7000 A rest-frame at z=0.1
        z = 0.1
        wave_obs = jnp.linspace(3800.0 * (1.0 + z), 7000.0 * (1.0 + z), n_pix)
        model.precompute_spectroscopy(wave_obs)

        params = spec.sample(key)

        t_fwd, t_grad = _bench_spectrum(model, params, n_iter)

        results["n_pix"].append(n_pix)
        results["fwd_us"].append(round(t_fwd, 1))
        results["grad_us"].append(round(t_grad, 1))

        print(f"{n_pix:>8d}  {t_fwd:>14.1f}  {t_grad:>14.1f}")

    return results


# ── 4. Catalog size scaling ──────────────────────────────────────
def bench_catalog_scaling(ssp, n_vi_iter, quick):
    """Measure fit_batch wall time vs number of galaxies."""
    print("\n" + "=" * 70)
    print("4. CATALOG SIZE SCALING (fit_batch, native_geovi)")
    print("=" * 70)

    n_gal_values = [1, 5, 10] if quick else [1, 5, 10, 50, 100]
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    key = jax.random.PRNGKey(3)

    results = {
        "n_gal": [],
        "total_s": [],
        "per_gal_s": [],
        "compile_frac": [],
    }

    print(f"\n{'N_gal':>6s}  {'Total (s)':>10s}  {'Per-gal (s)':>12s}  {'Compile frac':>13s}")
    print("-" * 50)

    spec = _make_spec(n_grid=64)
    model = Model(spec, ssp, observation=obs, precompute=True)

    for n_gal in n_gal_values:
        # Generate mock photometry batch
        keys = jax.random.split(key, n_gal + 1)
        key = keys[0]
        batch = []
        for i in range(n_gal):
            p_true = spec.sample(keys[i + 1])
            mock = model.mock(p_true, snr=20.0, key=jax.random.fold_in(keys[i + 1], 1))
            batch.append({"flux_obs": mock.flux_obs, "noise": mock.noise})

        # Time first galaxy (includes compile)
        fitter_first = Fitter(model, batch[0]["flux_obs"], batch[0]["noise"])
        t0_compile = time.perf_counter()
        _ = fitter_first.run(
            "vi_native",
            n_iterations=n_vi_iter,
            verbose=False,
            key=jax.random.PRNGKey(99),
        )
        t_compile = time.perf_counter() - t0_compile

        # Time full batch
        t0_batch = time.perf_counter()
        _ = fitter_first.fit_batch(
            batch,
            method="vi_native",
            n_iterations=n_vi_iter,
            verbose=False,
            key=jax.random.PRNGKey(100),
        )
        t_total = time.perf_counter() - t0_batch

        per_gal = t_total / n_gal
        compile_frac = t_compile / t_total if t_total > 0 else 0.0

        results["n_gal"].append(n_gal)
        results["total_s"].append(round(t_total, 2))
        results["per_gal_s"].append(round(per_gal, 3))
        results["compile_frac"].append(round(compile_frac, 3))

        print(f"{n_gal:>6d}  {t_total:>10.2f}  {per_gal:>12.3f}  {compile_frac:>13.1%}")

    return results


# ── Plotting ──────────────────────────────────────────────────────
def plot_results(all_results):
    """Generate 4 log-log scaling figures."""
    plt = setup_matplotlib()

    panels = [
        ("dimension", "D", "Parameter dimension D"),
        ("bands", "n_bands", "Number of photometric bands"),
        ("spectral", "n_pix", "Number of spectral pixels"),
    ]

    for tag, x_key, xlabel in panels:
        data = all_results.get(tag)
        if data is None:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.array(data[x_key])
        ax.loglog(x, data["fwd_us"], "o-", label="Forward", color="#2176AE")
        ax.loglog(x, data["grad_us"], "s--", label="Gradient", color="#D64045")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Wall time (us)")
        ax.set_title(f"Scaling: {xlabel}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"scaling_{tag}.pdf")
        fig.savefig(FIG_DIR / f"scaling_{tag}.png")
        plt.close(fig)
        print(f"  Saved scaling_{tag}.pdf/png")

    # Catalog scaling (linear-log)
    data = all_results.get("catalog")
    if data is not None:
        fig, ax1 = plt.subplots(figsize=(5, 4))
        x = np.array(data["n_gal"])
        ax1.plot(x, data["total_s"], "o-", label="Total time", color="#2176AE")
        ax1.set_xlabel("Number of galaxies")
        ax1.set_ylabel("Total time (s)", color="#2176AE")
        ax1.tick_params(axis="y", labelcolor="#2176AE")

        ax2 = ax1.twinx()
        ax2.plot(x, data["per_gal_s"], "s--", label="Per-galaxy", color="#D64045")
        ax2.set_ylabel("Per-galaxy time (s)", color="#D64045")
        ax2.tick_params(axis="y", labelcolor="#D64045")

        ax1.set_title("Catalog size scaling (native_geovi)")
        ax1.set_xscale("log")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "scaling_catalog.pdf")
        fig.savefig(FIG_DIR / "scaling_catalog.png")
        plt.close(fig)
        print("  Saved scaling_catalog.pdf/png")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark scaling across dimensions")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce iterations for CI testing",
    )
    args = parser.parse_args()

    quick = args.quick
    n_iter = 50 if quick else 200
    n_vi_iter = VI_ITER_DEFAULT

    print("tengri scaling benchmarks")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print(f"  Timing iterations: {n_iter}")
    print(f"  VI iterations: {n_vi_iter}")

    ssp = try_load_ssp()
    if ssp is None:
        print("\n[SKIP] All benchmarks require SSP data. Exiting.")
        return

    all_results = {}

    # 1. Dimension scaling
    all_results["dimension"] = bench_dimension_scaling(ssp, n_iter, quick)

    # 2. Band scaling
    all_results["bands"] = bench_band_scaling(ssp, n_iter, quick)

    # 3. Spectral pixel scaling
    all_results["spectral"] = bench_spectral_scaling(ssp, n_iter, quick)

    # 4. Catalog scaling
    try:
        all_results["catalog"] = bench_catalog_scaling(ssp, n_vi_iter, quick)
    except Exception as exc:
        print(f"\n[WARN] Catalog scaling failed: {exc}")
        print("  (fit_batch requires full inference stack; skipping)")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Plot
    plot_results(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
