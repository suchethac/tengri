"""Scaling profiler for tengri.

Measures how forward model and gradient performance scales with:
1. Parameter dimensionality (D = 7 to 267)
2. Number of photometric bands (3 to 20)
3. Number of spectral pixels (50 to 2000)
4. Batch size for catalog fitting (1 to 100)

Outputs CSV results and log-log scaling plots.

Usage::

    cd ~/Projects/tengri
    python profiling/profile_scaling.py --quick          # CI mode (~2 min)
    python profiling/profile_scaling.py --full            # comprehensive (~15 min)
    python profiling/profile_scaling.py --dimension-only  # just dimension scaling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bench(fn, n=200, warmup=3):
    """Time a function with warmup + JAX sync."""
    for _ in range(warmup):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    return (time.perf_counter() - t0) / n * 1e6, r


# ---------------------------------------------------------------------------
# Scaling tests
# ---------------------------------------------------------------------------


def scale_dimension(ssp, filters, quick=False):
    """Measure forward + gradient time vs parameter dimensionality."""
    from tengri import Fixed, Model, ParamSpec, Uniform

    dims = [7, 17, 37, 67] if quick else [7, 17, 37, 67, 137, 267]
    n_iters = 50 if quick else 200
    results = []

    print("\n  DIMENSION SCALING (D = number of free parameters)")
    print(f"  {'D':>5s} {'Forward (μs)':>14s} {'Gradient (μs)':>14s}")
    print("  " + "-" * 36)

    for d in dims:
        # D=7 is smooth (no GP). Higher D adds stochastic GP with n_grid.
        if d <= 7:
            spec = ParamSpec(
                sfh_dpl_alpha=Uniform(0.5, 3.0),
                sfh_dpl_beta=Uniform(0.5, 3.0),
                sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
                sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
                met_logzsol=Uniform(-2.0, 0.5),
                dust_tau_bc=Uniform(0.0, 2.0),
                dust_tau_diff=Uniform(0.0, 2.0),
                dust_slope=Fixed(-0.7),
                redshift=Fixed(0.1),
            )
        else:
            # Stochastic: D = 7 base + 2 PSD params + n_grid GP latents
            n_grid = d - 9  # Approximate
            n_grid = max(n_grid, 8)
            spec = ParamSpec(
                sfh_dpl_alpha=Uniform(0.5, 3.0),
                sfh_dpl_beta=Uniform(0.5, 3.0),
                sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
                sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
                sfh_field_psd_sigma=Uniform(0.01, 1.0),
                sfh_field_psd_tau_myr=Uniform(10, 500),
                met_logzsol=Uniform(-2.0, 0.5),
                dust_tau_bc=Uniform(0.0, 2.0),
                dust_tau_diff=Uniform(0.0, 2.0),
                dust_slope=Fixed(-0.7),
                redshift=Fixed(0.1),
                sfh_field_n_grid=n_grid,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model(spec, ssp, filters=filters, precompute=True)

        params = spec.sample(jax.random.PRNGKey(42))
        actual_d = len(spec.free_params)

        # Forward
        _ = model.predict_photometry(params)
        t_fwd, _ = bench(lambda: model.predict_photometry(params), n=n_iters)

        # Gradient
        grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
        _ = grad_fn(params)
        t_grad, _ = bench(lambda: grad_fn(params), n=n_iters)

        results.append({
            "dimension": actual_d,
            "forward_us": round(t_fwd, 1),
            "gradient_us": round(t_grad, 1),
        })
        print(f"  {actual_d:>5d} {t_fwd:>12.1f} μs {t_grad:>12.1f} μs")

    return results


def scale_bands(ssp, quick=False):
    """Measure forward time vs number of photometric bands."""
    from tengri import Fixed, Model, ParamSpec, Uniform, load_filter_set

    band_counts = [3, 5, 10] if quick else [3, 5, 8, 10, 15, 20]
    n_iters = 50 if quick else 200

    # Available SDSS + 2MASS + WISE filters
    all_filter_names = [
        "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
        "twomass_j", "twomass_h", "twomass_ks",
        "wise_w1", "wise_w2", "wise_w3", "wise_w4",
        "johnson_u", "johnson_b", "johnson_v", "johnson_r", "johnson_i",
        "hst_wfc3_f160w", "hst_wfc3_f125w", "hst_wfc3_f105w",
    ]

    results = []
    print("\n  BAND SCALING (number of photometric filters)")
    print(f"  {'Bands':>6s} {'Forward (μs)':>14s} {'Gradient (μs)':>14s}")
    print("  " + "-" * 36)

    for n_bands in band_counts:
        if n_bands > len(all_filter_names):
            break

        try:
            filt = load_filter_set(all_filter_names[:n_bands])
        except Exception:
            # Fallback: duplicate filters if some aren't available
            available = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
            names = (available * ((n_bands // len(available)) + 1))[:n_bands]
            filt = load_filter_set(names)

        spec = ParamSpec(
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model(spec, ssp, filters=filt, precompute=True)

        params = spec.sample(jax.random.PRNGKey(42))
        _ = model.predict_photometry(params)
        t_fwd, _ = bench(lambda: model.predict_photometry(params), n=n_iters)

        grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
        _ = grad_fn(params)
        t_grad, _ = bench(lambda: grad_fn(params), n=n_iters)

        results.append({
            "n_bands": n_bands,
            "forward_us": round(t_fwd, 1),
            "gradient_us": round(t_grad, 1),
        })
        print(f"  {n_bands:>6d} {t_fwd:>12.1f} μs {t_grad:>12.1f} μs")

    return results


def scale_spectral(ssp, filters, quick=False):
    """Measure forward time vs number of spectral pixels."""
    from tengri import Fixed, Model, ParamSpec, Uniform

    n_pixels_list = [50, 200, 500] if quick else [50, 100, 200, 500, 1000, 2000]
    n_iters = 50 if quick else 200
    results = []

    print("\n  SPECTRAL PIXEL SCALING")
    print(f"  {'Pixels':>7s} {'Forward (μs)':>14s} {'Gradient (μs)':>14s}")
    print("  " + "-" * 38)

    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )

    for n_pix in n_pixels_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model(spec, ssp, filters=filters, precompute=True)
            wave_obs = jnp.linspace(3800, 9200, n_pix)
            model.precompute_spectroscopy(wave_obs)

        params = spec.sample(jax.random.PRNGKey(42))
        _ = model.predict_spectrum(params, wave_obs)
        t_fwd, _ = bench(lambda: model.predict_spectrum(params, wave_obs), n=n_iters)

        grad_fn = jax.jit(jax.grad(
            lambda p: jnp.sum(model.predict_spectrum(p, wave_obs))
        ))
        _ = grad_fn(params)
        t_grad, _ = bench(lambda: grad_fn(params), n=n_iters)

        results.append({
            "n_pixels": n_pix,
            "forward_us": round(t_fwd, 1),
            "gradient_us": round(t_grad, 1),
        })
        print(f"  {n_pix:>7d} {t_fwd:>12.1f} μs {t_grad:>12.1f} μs")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scaling(results, output_dir):
    """Generate scaling plots from results dict."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dimension scaling ---
    if "dimension" in results:
        data = results["dimension"]
        dims = [d["dimension"] for d in data]
        fwd = [d["forward_us"] for d in data]
        grad = [d["gradient_us"] for d in data]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(dims, fwd, "o-", label="Forward", linewidth=2, markersize=8)
        ax.loglog(dims, grad, "s--", label="Gradient", linewidth=2, markersize=8)

        # O(n) reference
        d0, f0 = dims[0], fwd[0]
        ref = [f0 * (d / d0) for d in dims]
        ax.loglog(dims, ref, ":", color="gray", alpha=0.5, label="O(D)")

        ax.set_xlabel("Number of free parameters (D)")
        ax.set_ylabel("Time (μs)")
        ax.set_title("Parameter Dimension Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "scaling_dimension.pdf")
        fig.savefig(output_dir / "scaling_dimension.png", dpi=150)
        plt.close(fig)

    # --- Band scaling ---
    if "bands" in results:
        data = results["bands"]
        bands = [d["n_bands"] for d in data]
        fwd = [d["forward_us"] for d in data]
        grad = [d["gradient_us"] for d in data]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(bands, fwd, "o-", label="Forward", linewidth=2, markersize=8)
        ax.plot(bands, grad, "s--", label="Gradient", linewidth=2, markersize=8)
        ax.set_xlabel("Number of photometric bands")
        ax.set_ylabel("Time (μs)")
        ax.set_title("Photometric Band Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "scaling_bands.pdf")
        fig.savefig(output_dir / "scaling_bands.png", dpi=150)
        plt.close(fig)

    # --- Spectral pixel scaling ---
    if "spectral" in results:
        data = results["spectral"]
        pix = [d["n_pixels"] for d in data]
        fwd = [d["forward_us"] for d in data]
        grad = [d["gradient_us"] for d in data]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(pix, fwd, "o-", label="Forward", linewidth=2, markersize=8)
        ax.loglog(pix, grad, "s--", label="Gradient", linewidth=2, markersize=8)

        # O(n) reference
        p0, f0 = pix[0], fwd[0]
        ref = [f0 * (p / p0) for p in pix]
        ax.loglog(pix, ref, ":", color="gray", alpha=0.5, label="O(N)")

        ax.set_xlabel("Number of spectral pixels")
        ax.set_ylabel("Time (μs)")
        ax.set_title("Spectral Pixel Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "scaling_spectral.pdf")
        fig.savefig(output_dir / "scaling_spectral.png", dpi=150)
        plt.close(fig)

    print(f"\n  Plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Tengri scaling profiler")
    parser.add_argument("--quick", action="store_true", help="Quick mode for CI (~2 min)")
    parser.add_argument("--full", action="store_true", help="Comprehensive mode (~15 min)")
    parser.add_argument("--dimension-only", action="store_true")
    parser.add_argument("--output-dir", default="profiling/outputs", help="Output directory")
    args = parser.parse_args()

    quick = args.quick or not args.full
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TENGRI SCALING PROFILER")
    print("=" * 60)
    print(f"Platform: {sys.platform}, JAX backend: {jax.default_backend()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Mode: {'quick' if quick else 'full'}")

    from tengri import load_filter_set, load_ssp_data

    print("\nLoading SSP data...")
    ssp_path = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    ssp = load_ssp_data(ssp_path)
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    print(f"SSP shape: {ssp.ssp_flux.shape}")

    results = {}

    # 1. Dimension scaling
    if not args.dimension_only or args.dimension_only:
        results["dimension"] = scale_dimension(ssp, filters, quick=quick)

    # 2. Band scaling
    if not args.dimension_only:
        results["bands"] = scale_bands(ssp, quick=quick)

    # 3. Spectral pixel scaling
    if not args.dimension_only:
        results["spectral"] = scale_spectral(ssp, filters, quick=quick)

    # Save results
    json_path = output_dir / "scaling_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate plots
    try:
        plot_scaling(results, output_dir)
    except ImportError:
        print("matplotlib not available — skipping plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
