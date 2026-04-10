#!/usr/bin/env python3
"""Benchmark GL quadrature accuracy and timing for photometric precomputation.

Quantifies the dust-integration error of the precomputed-photometry approximation
(Zacharegkas+2025 §3) and its Gauss-Legendre extension, across three filter sets
chosen to match the testing in Zacharegkas+2025:

  (1) Gaussian synthetic filters (R = 3.4–25.5, matching their Fig B1)
  (2) SDSS ugriz
  (3) LSST grizy

All modes use the exact dense-grid trapezoidal SSP integral (Φ_{ijb}).
The ONLY source of error is the dust attenuation approximation:
  n_quad=1: dust at single T(λ)λ-weighted effective wavelength (= Zacharegkas)
  n_quad>1: dust GL-averaged with filter-weighted scale factors (our extension)

Reference: 10000-point trapezoidal rule (exact SSP × exact dust).
Test scenario: λ^{-2} power-law SSP + Charlot-Fall dust (τ_BC=1, n=-0.7).
This is a worst case: steep spectrum + broad fractional bandwidth.

Usage
-----
    # accuracy only (no SSP file needed):
    python scripts/benchmark_precompute_quad.py

    # accuracy + timing (needs data/ssp*.h5):
    python scripts/benchmark_precompute_quad.py --timing

    # specific n_quad values:
    python scripts/benchmark_precompute_quad.py --n-quad 3 5 7

    # N timing repeats:
    python scripts/benchmark_precompute_quad.py --timing --n-repeats 500
"""
from __future__ import annotations

import argparse
import glob
import sys
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── filter definitions ───────────────────────────────────────────────────────


def _gaussian_filters() -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Gaussian synthetic filters matching Zacharegkas+2025 Fig B1.

    9 bands spanning UV–NIR with resolution R = λ_mean/FWHM ranging from
    ~3.4 (broadest) to ~25.5 (narrowest).
    """
    # Centers and FWHM chosen to span 2000–20000 Å with varying resolution
    configs = [
        ("G2000_R3.4", 2000.0, 590.0),   # R ≈ 3.4
        ("G3000_R4.3", 3000.0, 700.0),   # R ≈ 4.3
        ("G4000_R5.7", 4000.0, 700.0),   # R ≈ 5.7
        ("G5500_R7.9", 5500.0, 700.0),   # R ≈ 7.9
        ("G7000_R10", 7000.0, 700.0),    # R ≈ 10
        ("G9000_R13", 9000.0, 700.0),    # R ≈ 12.9
        ("G12000_R17", 12000.0, 700.0),  # R ≈ 17.1
        ("G15000_R21", 15000.0, 700.0),  # R ≈ 21.4
        ("G18000_R26", 18000.0, 700.0),  # R ≈ 25.7
    ]
    names, fw_list, ft_list = [], [], []
    for name, center, fwhm in configs:
        sigma = fwhm / 2.3548
        w = np.linspace(center - 3 * fwhm, center + 3 * fwhm, 200)
        t = np.exp(-0.5 * ((w - center) / sigma) ** 2)
        t[t < 1e-4] = 0.0  # clip negligible tails
        names.append(name)
        fw_list.append(w)
        ft_list.append(t)
    return names, fw_list, ft_list


def _load_real_filters(filter_names: list[str]):
    """Load real filter curves via tengri."""
    try:
        from tengri.models.observation.filters import load_filter_set

        fw_list, ft_list, _ = load_filter_set(filter_names)
        return filter_names, fw_list, ft_list
    except Exception as e:
        print(f"  [warn] Could not load filters {filter_names}: {e}")
        return None


# ── photometry approximation helpers ─────────────────────────────────────────


def _exact_phot(
    ssp_lam: np.ndarray,
    ssp_sed: np.ndarray,
    dust: np.ndarray,
    wav: np.ndarray,
    thr: np.ndarray,
) -> float:
    """Reference photometry via dense trapezoidal rule (exact SSP × exact dust)."""
    t_i = np.interp(ssp_lam, wav, thr, left=0.0, right=0.0)
    num = np.trapezoid(ssp_sed * dust * t_i * ssp_lam, ssp_lam)
    den = np.trapezoid(t_i * ssp_lam, ssp_lam)
    return float(num / den) if abs(den) > 1e-30 else 0.0


def _approx_phot_nquad(
    ssp_lam: np.ndarray,
    ssp_sed: np.ndarray,
    dust_fn,
    wav: np.ndarray,
    thr: np.ndarray,
    n_quad: int,
) -> float:
    """Precomputed photometry: exact SSP trapz + n-point dust approximation.

    n_quad=1: dust at T(λ)λ-weighted effective wavelength (Zacharegkas method).
    n_quad>1: dust GL-averaged with filter-weighted scale factors (our extension).
    """
    wav_np, thr_np = np.asarray(wav), np.asarray(thr)

    # SSP: always exact dense-grid trapz
    t_i = np.interp(ssp_lam, wav_np, thr_np, left=0.0, right=0.0)
    num_ssp = np.trapezoid(ssp_sed * t_i * ssp_lam, ssp_lam)
    den = np.trapezoid(t_i * ssp_lam, ssp_lam)
    if abs(den) < 1e-30:
        return 0.0
    csp = float(num_ssp / den)

    if n_quad == 1:
        # Single effective wavelength (Zacharegkas+2025 §3)
        den_filt = np.trapezoid(thr_np * wav_np, wav_np)
        lam_eff = float(np.trapezoid(thr_np * wav_np**2, wav_np) / den_filt)
        dust_avg = float(dust_fn(np.array([lam_eff]))[0])
    else:
        # GL quadrature over the filter bandpass
        from tengri.models.sps.precompute import _gauss_legendre_nodes_for_filter

        nodes, weights, h = _gauss_legendre_nodes_for_filter(wav_np, n_quad, thr_np)
        t_at_nodes = np.interp(nodes, wav_np, thr_np)
        denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
        scale = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
        dust_at_nodes = dust_fn(nodes)
        dust_avg = float(np.sum(weights * dust_at_nodes * scale))

    return csp * dust_avg


# ── accuracy benchmark ────────────────────────────────────────────────────────


def run_accuracy(
    filter_names: list[str],
    fw_list: list,
    ft_list: list,
    n_quad_vals: list[int],
) -> dict:
    """Compute per-filter relative error for each n_quad."""
    filter_data = list(zip(filter_names, fw_list, ft_list))

    all_waves = np.concatenate([np.asarray(f[1]) for f in filter_data])
    lam_min, lam_max = all_waves.min() * 0.9, all_waves.max() * 1.1
    ssp_lam = np.linspace(lam_min, lam_max, 10_000)

    # Worst-case steep power-law SSP
    ssp_sed = (ssp_lam / ssp_lam.mean()) ** -2

    # Charlot-Fall dust (τ_BC=1, n=-0.7)
    tau_bc = 1.0
    dust_arr = np.exp(-tau_bc * (ssp_lam / 5500.0) ** -0.7)
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)  # noqa: E731

    results = {}
    for fname, wav, thr in filter_data:
        exact = _exact_phot(ssp_lam, ssp_sed, dust_arr, wav, thr)
        if abs(exact) < 1e-30:
            continue
        row: dict = {"exact": exact}

        for nq in n_quad_vals:
            approx = _approx_phot_nquad(ssp_lam, ssp_sed, dust_fn, wav, thr, nq)
            row[f"n{nq}_err_pct"] = 100.0 * abs(approx - exact) / abs(exact)

        results[fname] = row
    return results


def print_accuracy_table(
    label: str,
    results: dict,
    n_quad_vals: list[int],
    timings: dict | None = None,
) -> None:
    """Print combined accuracy (+timing) table for one filter set."""
    col_w = 10
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")

    # Header
    header = f"{'Filter':<16}"
    for nq in n_quad_vals:
        label_nq = f"n={nq}" if nq > 1 else "n=1 (Z+25)"
        header += f"  {label_nq:>{col_w}}"
    print(header)
    print("─" * len(header))

    all_errs: dict[int, list[float]] = {nq: [] for nq in n_quad_vals}
    for fname, row in results.items():
        line = f"{fname:<16}"
        for nq in n_quad_vals:
            e = row[f"n{nq}_err_pct"]
            line += f"  {e:>{col_w}.4f}%"
            all_errs[nq].append(e)
        print(line)

    # Summary row
    print("─" * len(header))
    line_max = f"{'Max':<16}"
    line_mean = f"{'Mean':<16}"
    for nq in n_quad_vals:
        line_max += f"  {max(all_errs[nq]):>{col_w}.4f}%"
        line_mean += f"  {np.mean(all_errs[nq]):>{col_w}.4f}%"
    print(line_max)
    print(line_mean)

    # Timing row (if available)
    if timings:
        line_time = f"{'µs/call':<16}"
        for nq in n_quad_vals:
            if nq in timings:
                line_time += f"  {timings[nq]:>{col_w}.1f}"
            else:
                line_time += f"  {'—':>{col_w}}"
        print(line_time)


# ── timing benchmark ──────────────────────────────────────────────────────────


def run_timing(
    ssp_file: str,
    filter_names: list[str],
    n_quad_vals: list[int],
    n_repeats: int,
) -> dict:
    """Benchmark fast_photometry JIT call for each n_quad."""
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    from tengri.models.observation.filters import load_filter_set
    from tengri.models.sps.dsps_wrapper import load_ssp_data
    from tengri.models.sps.precompute import fast_photometry, precompute_photometry

    ssp = load_ssp_data(ssp_file)
    fw_list, ft_list, _ = load_filter_set(filter_names)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = 0.1
    dl = cosmo.luminosity_distance(z).to(u.cm).value

    n_met, n_age = ssp.ssp_flux.shape[:2]
    csp_w = jnp.ones((n_met, n_age)) / (n_met * n_age)

    timings = {}
    for nq in n_quad_vals:
        pc = precompute_photometry(ssp, fw_list, ft_list, z, dl, n_quad=nq)

        n_dust_pts = pc.n_filters * pc.n_quad
        dust_bc = jnp.ones(n_dust_pts)
        dust_diff = jnp.ones(n_dust_pts)

        # warm-up (JIT compile)
        for _ in range(5):
            fast_photometry(csp_w, pc, dust_bc, dust_diff).block_until_ready()

        t0 = timeit.default_timer()
        for _ in range(n_repeats):
            fast_photometry(csp_w, pc, dust_bc, dust_diff).block_until_ready()
        dt_us = (timeit.default_timer() - t0) / n_repeats * 1e6

        timings[nq] = dt_us

    return timings


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-quad",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7],
        dest="n_quad_vals",
        help="n_quad values to test (default: 1 3 5 7)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Run fast_photometry JIT timing (requires data/ssp*.h5)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=300,
        dest="n_repeats",
        help="JIT timing repetitions",
    )
    args = parser.parse_args()

    n_quad_vals = sorted(args.n_quad_vals)

    print("=" * 72)
    print("Photometric precomputation: dust quadrature accuracy benchmark")
    print("SSP: λ^{-2} power law | Dust: Charlot-Fall (τ_BC=1, n=-0.7)")
    print("Reference: 10000-point dense trapezoidal rule")
    print(f"n_quad tested: {n_quad_vals}")
    print("=" * 72)

    # Timing (SDSS only, needs SSP data)
    timings = None
    if args.timing:
        ssp_files = glob.glob(str(ROOT / "data" / "ssp*.h5"))
        if ssp_files:
            sdss = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
            print(f"\nTiming: fast_photometry JIT (SSP: {Path(ssp_files[0]).name})")
            timings = run_timing(ssp_files[0], sdss, n_quad_vals, args.n_repeats)
        else:
            print("\n[timing] No data/ssp*.h5 found — skipping timing benchmark.")

    # --- Filter set 1: Gaussian synthetic ---
    g_names, g_fw, g_ft = _gaussian_filters()
    results_gauss = run_accuracy(g_names, g_fw, g_ft, n_quad_vals)
    print_accuracy_table(
        "Gaussian synthetic filters (R = 3.4–25.7, matching Zacharegkas+2025 Fig B1)",
        results_gauss,
        n_quad_vals,
    )

    # --- Filter set 2: SDSS ugriz ---
    sdss = _load_real_filters(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    if sdss is not None:
        results_sdss = run_accuracy(*sdss, n_quad_vals)
        print_accuracy_table(
            "SDSS ugriz", results_sdss, n_quad_vals, timings=timings
        )

    # --- Filter set 3: LSST grizy ---
    lsst = _load_real_filters(["lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"])
    if lsst is not None:
        results_lsst = run_accuracy(*lsst, n_quad_vals)
        print_accuracy_table(
            "LSST grizy (matching Zacharegkas+2025 Fig B2)",
            results_lsst,
            n_quad_vals,
        )

    print(f"\n{'─' * 72}")
    print("Notes:")
    print("  n=1 is the Zacharegkas+2025 §3 method (dust at single λ_eff per filter)")
    print("  n>1 is our GL extension (dust averaged over n GL nodes per filter)")
    print("  All modes use exact dense-grid SSP integral Φ_{ijb}")
    print("  Errors are DUST-ONLY: SSP contribution is always exact")
    if timings:
        print(f"  Timing: {args.n_repeats} reps after JIT warmup, CPU, float64")
    print(f"{'─' * 72}\n")


if __name__ == "__main__":
    main()
