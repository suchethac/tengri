"""Critical analysis: denom_quad INCONSISTENCY in scale factor computation.

BUG HYPOTHESIS:

In precompute.py lines 376 and 399, the scale factors are normalized by:

    denom_quad = h * Σ w_k · T(λ_k) · λ_k

But this is computing the GL-approximated integral of T(λ)·λ, not the actual
photometric denominator ∫T(λ)·λ dλ.

The EXACT denominator is computed once for ssp_phot (line 338):
    denom = trapezoid(T·fw, fw)

But for GL dust averaging (quad_dust_scale), we use:
    denom_quad = h * sum(w * T(λ_k) * λ_k)

These are TWO DIFFERENT NORMALIZATIONS of the same integral!
- denom (trapz) ≈ ∫ T(λ) · λ dλ (EXACT, exact nodes)
- denom_quad (GL) ≈ ∫ T(λ) · λ dλ (APPROXIMATE, GL nodes)

For broad filters with irregular shapes (like SDSS z), GL nodes may not sample
the filter transmission well, leading to POOR estimates of denom_quad.

When we normalize the dust scale factors by denom_quad instead of denom, we're
introducing a systematic error that depends on how well GL nodes fit the filter.

PROPOSED FIX:
Use the EXACT denom (already computed for ssp_phot) instead of denom_quad:

    scale = t_at_nodes * nodes * h / denom  # NOT denom_quad!

This ensures the dust scale factors are properly normalized against the exact
photometric denominator, not a GL approximation of it.
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


def _exact_phot_current(ssp_lam, ssp_sed, dust, wav, thr):
    """Current implementation: exact SSP × exact dust."""
    t_i = np.interp(ssp_lam, wav, thr, left=0.0, right=0.0)
    num = np.trapezoid(ssp_sed * dust * t_i * ssp_lam, ssp_lam)
    den = np.trapezoid(t_i * ssp_lam, ssp_lam)
    return float(num / den) if abs(den) > 1e-30 else 0.0


def _approx_phot_current(ssp_lam, ssp_sed, dust_fn, wav, thr, n_quad):
    """Current: exact SSP trapz + GL dust with denom_quad normalization."""
    wav_np, thr_np = np.asarray(wav), np.asarray(thr)

    t_i = np.interp(ssp_lam, wav_np, thr_np, left=0.0, right=0.0)
    num_ssp = np.trapezoid(ssp_sed * t_i * ssp_lam, ssp_lam)
    denom_exact = np.trapezoid(t_i * ssp_lam, ssp_lam)  # EXACT photometric denom
    if abs(denom_exact) < 1e-30:
        return 0.0
    csp = float(num_ssp / denom_exact)

    if n_quad == 1:
        lam_eff = float(np.trapezoid(thr_np * wav_np**2, wav_np) / denom_exact)
        dust_avg = float(dust_fn(np.array([lam_eff]))[0])
    else:
        nodes, weights, h = _gauss_legendre_nodes_for_filter(wav_np, n_quad)
        t_at_nodes = np.interp(nodes, wav_np, thr_np)
        # CURRENT: Use denom_quad (GL approximation)
        denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
        scale_current = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
        dust_at_nodes = dust_fn(nodes)
        dust_avg = float(np.sum(weights * dust_at_nodes * scale_current))

    return csp * dust_avg


def _approx_phot_fixed(ssp_lam, ssp_sed, dust_fn, wav, thr, n_quad):
    """PROPOSED FIX: exact SSP trapz + GL dust with EXACT denom normalization."""
    wav_np, thr_np = np.asarray(wav), np.asarray(thr)

    t_i = np.interp(ssp_lam, wav_np, thr_np, left=0.0, right=0.0)
    num_ssp = np.trapezoid(ssp_sed * t_i * ssp_lam, ssp_lam)
    denom_exact = np.trapezoid(t_i * ssp_lam, ssp_lam)  # EXACT photometric denom
    if abs(denom_exact) < 1e-30:
        return 0.0
    csp = float(num_ssp / denom_exact)

    if n_quad == 1:
        lam_eff = float(np.trapezoid(thr_np * wav_np**2, wav_np) / denom_exact)
        dust_avg = float(dust_fn(np.array([lam_eff]))[0])
    else:
        nodes, weights, h = _gauss_legendre_nodes_for_filter(wav_np, n_quad)
        t_at_nodes = np.interp(nodes, wav_np, thr_np)
        # FIX: Use denom_exact (EXACT photometric normalization)
        scale_fixed = t_at_nodes * nodes * h / max(denom_exact, 1e-30)
        dust_at_nodes = dust_fn(nodes)
        dust_avg = float(np.sum(weights * dust_at_nodes * scale_fixed))

    return csp * dust_avg


def test_denom_quad_vs_exact():
    """Demonstrate the denom_quad vs denom_exact discrepancy."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z", "sdss_g"])
        filters = dict(z=(np.asarray(fw_list[0]), np.asarray(ft_list[0])),
                       g=(np.asarray(fw_list[1]), np.asarray(ft_list[1])))
    except Exception as e:
        pytest.skip(f"Could not load filters: {e}")

    print("\n" + "="*80)
    print("denom_quad vs denom_exact: THE CORE ISSUE")
    print("="*80)

    for band, (fw, ft) in filters.items():
        print(f"\n{band.upper()}-BAND:")
        print("-" * 80)

        denom_exact = np.trapezoid(ft * fw, fw)
        print(f"denom_exact = ∫T(λ)λ dλ = {denom_exact:.6e}")

        for n_quad in [3, 5, 7]:
            nodes, weights, h = _gauss_legendre_nodes_for_filter(fw, n_quad)
            t_at_nodes = np.interp(nodes, fw, ft)
            denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))

            ratio = denom_quad / denom_exact
            err_pct = 100 * abs(ratio - 1.0)

            print(f"n={n_quad}:")
            print(f"  denom_quad = {denom_quad:.6e}")
            print(f"  ratio (quad/exact) = {ratio:.6f}")
            print(f"  error = {err_pct:.2f}%")

            # Show how this affects scale normalization
            scale_exact = t_at_nodes * nodes * h / denom_exact
            scale_quad = t_at_nodes * nodes * h / denom_quad
            scale_ratio = scale_quad / scale_exact
            print(f"  scale factors ratio (quad/exact) = {scale_ratio[0]:.4f} ... {scale_ratio[-1]:.4f}")

    print("\n" + "="*80)


def test_accuracy_comparison():
    """Compare current vs fixed implementation."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z"])
        fw_z, ft_z = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS z: {e}")

    print("\n" + "="*80)
    print("ACCURACY: CURRENT vs FIXED")
    print("="*80)

    # Setup
    all_waves = fw_z
    lam_min, lam_max = all_waves.min() * 0.9, all_waves.max() * 1.1
    ssp_lam = np.linspace(lam_min, lam_max, 10_000)
    ssp_sed = (ssp_lam / ssp_lam.mean()) ** -2

    tau_bc = 1.0
    dust_arr = np.exp(-tau_bc * (ssp_lam / 5500.0) ** -0.7)
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)

    exact = _exact_phot_current(ssp_lam, ssp_sed, dust_arr, fw_z, ft_z)
    print(f"\nExact photometry: {exact:.8f}")

    print("\n" + "-" * 80)
    print("n_quad | Current error | Fixed error | Improvement")
    print("-" * 80)

    for n_quad in [1, 3, 5, 7]:
        approx_curr = _approx_phot_current(ssp_lam, ssp_sed, dust_fn, fw_z, ft_z, n_quad)
        approx_fix = _approx_phot_fixed(ssp_lam, ssp_sed, dust_fn, fw_z, ft_z, n_quad)

        err_curr = 100.0 * abs(approx_curr - exact) / abs(exact)
        err_fix = 100.0 * abs(approx_fix - exact) / abs(exact)
        improve = err_curr - err_fix

        print(f"{n_quad:6d} | {err_curr:13.4f}% | {err_fix:11.4f}% | {improve:+10.4f}%")

    print("\n" + "="*80)
    print("If 'Fixed error' is MONOTONICALLY DECREASING and always better,")
    print("then the bug hypothesis is CONFIRMED.")
    print("="*80)


if __name__ == "__main__":
    test_denom_quad_vs_exact()
    test_accuracy_comparison()
