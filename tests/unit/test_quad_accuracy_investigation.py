"""Full investigation of GL quadrature non-monotonic accuracy issue.

This test reproduces the exact benchmark scenario from
scripts/benchmark_precompute_quad.py to understand WHY accuracy is non-monotonic.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tengri.models.sps.precompute import _gauss_legendre_nodes_for_filter
from tengri.models.observation.filters import load_filter_set


def _exact_phot(ssp_lam, ssp_sed, dust, wav, thr):
    """Reference photometry via 10000-pt dense trapezoidal rule."""
    t_i = np.interp(ssp_lam, wav, thr, left=0.0, right=0.0)
    num = np.trapezoid(ssp_sed * dust * t_i * ssp_lam, ssp_lam)
    den = np.trapezoid(t_i * ssp_lam, ssp_lam)
    return float(num / den) if abs(den) > 1e-30 else 0.0


def _approx_phot_nquad(ssp_lam, ssp_sed, dust_fn, wav, thr, n_quad):
    """Approximate photometry using GL quadrature."""
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
        lam_eff = float(np.trapezoid(thr_np * wav_np**2, wav_np) / den)
        dust_avg = float(dust_fn(np.array([lam_eff]))[0])
    else:
        # GL quadrature over the filter bandpass
        nodes, weights, h = _gauss_legendre_nodes_for_filter(wav_np, n_quad)
        t_at_nodes = np.interp(nodes, wav_np, thr_np)
        denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
        scale = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
        dust_at_nodes = dust_fn(nodes)
        dust_avg = float(np.sum(weights * dust_at_nodes * scale))

    return csp * dust_avg


def test_sdss_z_nonmonotonic():
    """Investigate SDSS z-band non-monotonic accuracy."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z"])
        fw_z, ft_z = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS z: {e}")

    print("\n" + "="*80)
    print("SDSS Z-BAND: ACCURACY INVESTIGATION")
    print("="*80)

    # Worst-case steep power-law SSP (same as benchmark)
    all_waves = fw_z
    lam_min, lam_max = all_waves.min() * 0.9, all_waves.max() * 1.1
    ssp_lam = np.linspace(lam_min, lam_max, 10_000)
    ssp_sed = (ssp_lam / ssp_lam.mean()) ** -2

    # Charlot-Fall dust (τ_BC=1, n=-0.7)
    tau_bc = 1.0
    dust_arr = np.exp(-tau_bc * (ssp_lam / 5500.0) ** -0.7)
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)

    # Exact reference (10000-pt trapz)
    exact = _exact_phot(ssp_lam, ssp_sed, dust_arr, fw_z, ft_z)
    print(f"\nExact photometry (10000-pt trapz): {exact:.8f}")

    n_quad_vals = [1, 3, 5, 7]
    errors = {}

    print("\n" + "-"*80)
    print("Per-n_quad analysis:")
    print("-"*80)

    for n_quad in n_quad_vals:
        approx = _approx_phot_nquad(ssp_lam, ssp_sed, dust_fn, fw_z, ft_z, n_quad)
        err_pct = 100.0 * abs(approx - exact) / abs(exact)
        errors[n_quad] = err_pct

        print(f"\nn_quad = {n_quad}:")
        print(f"  Approximate: {approx:.8f}")
        print(f"  Error: {err_pct:.4f}%")

        if n_quad == 1:
            lam_eff = np.trapezoid(ft_z * fw_z**2, fw_z) / np.trapezoid(ft_z * fw_z, fw_z)
            dust_eff = dust_fn(np.array([lam_eff]))[0]
            print(f"  λ_eff = {lam_eff:.2f} Å")
            print(f"  A(λ_eff) = {dust_eff:.6f}")
        else:
            nodes, weights, h = _gauss_legendre_nodes_for_filter(fw_z, n_quad)
            t_at_nodes = np.interp(nodes, fw_z, ft_z)
            denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
            denom_exact = np.trapezoid(ft_z * fw_z, fw_z)

            print(f"  denom_quad = {denom_quad:.6e}")
            print(f"  denom_exact = {denom_exact:.6e}")
            print(f"  denom ratio (GL/exact) = {denom_quad / denom_exact:.6f}")

            scale = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
            dust_at_nodes = dust_fn(nodes)
            dust_avg_gl = float(np.sum(weights * dust_at_nodes * scale))

            print(f"  GL nodes: {nodes}")
            print(f"  T(λ_k): {t_at_nodes}")
            print(f"  A(λ_k): {dust_at_nodes}")
            print(f"  scales: {scale}")
            print(f"  ⟨A⟩_GL = {dust_avg_gl:.6f}")

    print("\n" + "-"*80)
    print("SUMMARY: Error vs n_quad")
    print("-"*80)
    for n_quad in n_quad_vals:
        marker = " ← NON-MONOTONIC!" if (
            n_quad > 1 and errors[n_quad] > errors[n_quad - 1]
        ) else ""
        print(f"  n={n_quad}: {errors[n_quad]:7.4f}%{marker}")

    print("\n" + "="*80)


def test_sdss_g_nonmonotonic():
    """Investigate SDSS g-band non-monotonic accuracy."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_g"])
        fw_g, ft_g = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS g: {e}")

    print("\n" + "="*80)
    print("SDSS G-BAND: ACCURACY INVESTIGATION")
    print("="*80)

    all_waves = fw_g
    lam_min, lam_max = all_waves.min() * 0.9, all_waves.max() * 1.1
    ssp_lam = np.linspace(lam_min, lam_max, 10_000)
    ssp_sed = (ssp_lam / ssp_lam.mean()) ** -2

    tau_bc = 1.0
    dust_arr = np.exp(-tau_bc * (ssp_lam / 5500.0) ** -0.7)
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)

    exact = _exact_phot(ssp_lam, ssp_sed, dust_arr, fw_g, ft_g)
    print(f"\nExact photometry: {exact:.8f}")

    n_quad_vals = [1, 3, 5, 7]
    errors = {}

    print("\n" + "-"*80)
    print("Per-n_quad analysis:")
    print("-"*80)

    for n_quad in n_quad_vals:
        approx = _approx_phot_nquad(ssp_lam, ssp_sed, dust_fn, fw_g, ft_g, n_quad)
        err_pct = 100.0 * abs(approx - exact) / abs(exact)
        errors[n_quad] = err_pct

        print(f"\nn_quad = {n_quad}: error = {err_pct:.4f}%")

        if n_quad > 1:
            nodes, weights, h = _gauss_legendre_nodes_for_filter(fw_g, n_quad)
            t_at_nodes = np.interp(nodes, fw_g, ft_g)
            denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
            denom_exact = np.trapezoid(ft_g * fw_g, fw_g)
            print(f"  denom ratio (GL/exact) = {denom_quad / denom_exact:.6f}")

    print("\n" + "-"*80)
    print("SUMMARY: Error vs n_quad")
    print("-"*80)
    for n_quad in n_quad_vals:
        marker = " ← NON-MONOTONIC!" if (
            n_quad > 1 and errors[n_quad] > errors[n_quad - 1]
        ) else ""
        print(f"  n={n_quad}: {errors[n_quad]:7.4f}%{marker}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_sdss_z_nonmonotonic()
    test_sdss_g_nonmonotonic()
