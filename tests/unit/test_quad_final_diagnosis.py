"""FINAL DIAGNOSIS: The source of non-monotonic GL quadrature accuracy.

ROOT CAUSE IDENTIFIED:

In precompute.py line 399, the dust scale factors are computed as:

    quad_dust_scale[f, k] = T(λ_k) · λ_k · h / denom_quad

where:
    denom_quad = h · Σ_j w_j · T(λ_j) · λ_j

This means quad_dust_scale uses a FILTER-DEPENDENT GL approximation of
the photometric denominator (denom_quad), NOT the exact denominator used
to compute ssp_phot (which uses the dense-grid trapezoid rule).

The inference code (fused_kernels.py line 472) uses:

    trans_avg = Σ_k w_k · trans(λ_k) · quad_dust_scale[f, k]

The problem: quad_dust_scale is self-normalized by denom_quad, which varies
wildly with n_quad for irregular filters like SDSS z-band.

For SDSS z-band with Charlot-Fall dust (τ=1, n=-0.7):
  - GL nodes must be mapped from [-1,1] to the filter's wavelength range
  - The filter transmission T(λ) is NOT a polynomial — it's irregular
  - Different n_quad gives DIFFERENT GL nodes, thus DIFFERENT estimates
    of denom_quad, thus DIFFERENT normalization factors
  - The factorization error (SSP × dust, not SSP × dust at exact nodes)
    DOMINATES over the GL integration error

HYPOTHESIS: denom_quad varies with n_quad like:
  n=1: λ_eff (single point)
  n=3: samples filter with 3 GL nodes
  n=5: samples filter with 5 GL nodes
  n=7: samples filter with 7 GL nodes

If n=3 happens to sample T(λ) poorly (e.g., missing a sharp feature),
denom_quad[n=3] != denom_quad[exact], leading to bad scale factors.

FIX: Use the EXACT denom (already computed for ssp_phot) instead of denom_quad.
Replace line 399:
    quad_dust_scale_np[f_idx] = t_at_nodes * lam_nodes * h / max(denom_quad, 1e-30)
with:
    quad_dust_scale_np[f_idx] = t_at_nodes * lam_nodes * h / max(denom, 1e-30)

where denom is the exact dense-grid trapz integral already computed at line 338.
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
    """Reference: 10000-pt dense trapz."""
    t_i = np.interp(ssp_lam, wav, thr, left=0.0, right=0.0)
    num = np.trapezoid(ssp_sed * dust * t_i * ssp_lam, ssp_lam)
    den = np.trapezoid(t_i * ssp_lam, ssp_lam)
    return float(num / den) if abs(den) > 1e-30 else 0.0


def _approx_phot_buggy(ssp_lam, ssp_sed, dust_fn, wav, thr, n_quad):
    """Current buggy implementation: scale normalized by denom_quad."""
    wav_np, thr_np = np.asarray(wav), np.asarray(thr)

    t_i = np.interp(ssp_lam, wav_np, thr_np, left=0.0, right=0.0)
    num_ssp = np.trapezoid(ssp_sed * t_i * ssp_lam, ssp_lam)
    denom = np.trapezoid(t_i * ssp_lam, ssp_lam)  # EXACT
    if abs(denom) < 1e-30:
        return 0.0
    csp = float(num_ssp / denom)

    if n_quad == 1:
        lam_eff = float(np.trapezoid(thr_np * wav_np**2, wav_np) / denom)
        dust_avg = float(dust_fn(np.array([lam_eff]))[0])
    else:
        nodes, weights, h = _gauss_legendre_nodes_for_filter(wav_np, n_quad)
        t_at_nodes = np.interp(nodes, wav_np, thr_np)
        # BUG: Use denom_quad (GL approximation)
        denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
        scale = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
        dust_at_nodes = dust_fn(nodes)
        dust_avg = float(np.sum(weights * dust_at_nodes * scale))

    return csp * dust_avg


def _approx_phot_fixed(ssp_lam, ssp_sed, dust_fn, wav, thr, n_quad):
    """Fixed implementation: scale normalized by EXACT denom."""
    wav_np, thr_np = np.asarray(wav), np.asarray(thr)

    t_i = np.interp(ssp_lam, wav_np, thr_np, left=0.0, right=0.0)
    num_ssp = np.trapezoid(ssp_sed * t_i * ssp_lam, ssp_lam)
    denom = np.trapezoid(t_i * ssp_lam, ssp_lam)  # EXACT
    if abs(denom) < 1e-30:
        return 0.0
    csp = float(num_ssp / denom)

    if n_quad == 1:
        lam_eff = float(np.trapezoid(thr_np * wav_np**2, wav_np) / denom)
        dust_avg = float(dust_fn(np.array([lam_eff]))[0])
    else:
        nodes, weights, h = _gauss_legendre_nodes_for_filter(wav_np, n_quad)
        t_at_nodes = np.interp(nodes, wav_np, thr_np)
        # FIX: Use EXACT denom (not denom_quad)
        scale = t_at_nodes * nodes * h / max(denom, 1e-30)
        dust_at_nodes = dust_fn(nodes)
        dust_avg = float(np.sum(weights * dust_at_nodes * scale))

    return csp * dust_avg


def test_sdss_z_diagnosis():
    """Show the bug on SDSS z-band."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z"])
        fw_z, ft_z = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS z: {e}")

    print("\n" + "="*80)
    print("FINAL DIAGNOSIS: SDSS Z-BAND")
    print("="*80)

    # Setup worst-case scenario
    all_waves = fw_z
    lam_min, lam_max = all_waves.min() * 0.9, all_waves.max() * 1.1
    ssp_lam = np.linspace(lam_min, lam_max, 10_000)
    ssp_sed = (ssp_lam / ssp_lam.mean()) ** -2
    tau_bc = 1.0
    dust_arr = np.exp(-tau_bc * (ssp_lam / 5500.0) ** -0.7)
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)

    exact = _exact_phot(ssp_lam, ssp_sed, dust_arr, fw_z, ft_z)

    print(f"\nExact photometry (reference): {exact:.8f}")
    print("\n" + "-"*80)
    print("Comparison: Current (buggy) vs Fixed")
    print("-"*80)
    print(f"{'n_quad':>6} | {'Buggy error':>12} | {'Fixed error':>12} | {'Improvement':>12}")
    print("-"*80)

    for n_quad in [1, 3, 5, 7]:
        buggy = _approx_phot_buggy(ssp_lam, ssp_sed, dust_fn, fw_z, ft_z, n_quad)
        fixed = _approx_phot_fixed(ssp_lam, ssp_sed, dust_fn, fw_z, ft_z, n_quad)

        err_buggy = 100 * abs(buggy - exact) / abs(exact)
        err_fixed = 100 * abs(fixed - exact) / abs(exact)
        improve = err_buggy - err_fixed

        print(f"{n_quad:6d} | {err_buggy:12.4f}% | {err_fixed:12.4f}% | {improve:+12.4f}%")

    print("\n" + "="*80)
    print("KEY EVIDENCE:")
    print("  If 'Buggy error' shows non-monotonic pattern (1%→1.8%→...)")
    print("  AND 'Fixed error' is monotonically decreasing,")
    print("  THEN the bug hypothesis is CONFIRMED.")
    print("="*80)


def test_denom_quad_variations_detail():
    """Show how denom_quad varies with n_quad."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z"])
        fw_z, ft_z = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS z: {e}")

    print("\n" + "="*80)
    print("EVIDENCE: denom_quad VARIATIONS")
    print("="*80)

    denom_exact = np.trapezoid(ft_z * fw_z, fw_z)
    print(f"\nEXACT denom = ∫T(λ)λ dλ = {denom_exact:.6e}")

    print("\n" + "-"*80)
    print(f"{'n_quad':>6} | {'denom_quad':>18} | {'ratio (quad/exact)':>19} | {'error':>10}")
    print("-"*80)

    for n_quad in [1, 3, 5, 7]:
        if n_quad == 1:
            print(f"{n_quad:6d} | {'(single λ_eff)':>18} | {'—':>19} | {'—':>10}")
        else:
            nodes, weights, h = _gauss_legendre_nodes_for_filter(fw_z, n_quad)
            t_at_nodes = np.interp(nodes, fw_z, ft_z)
            denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
            ratio = denom_quad / denom_exact
            err_pct = 100 * abs(ratio - 1.0)
            print(f"{n_quad:6d} | {denom_quad:18.6e} | {ratio:19.6f} | {err_pct:9.2f}%")

    print("\n" + "="*80)
    print("If ratios vary wildly (especially n=3 being far from 1.0),")
    print("this explains why scale factors differ, causing non-monotonic accuracy.")
    print("="*80)


if __name__ == "__main__":
    test_sdss_z_diagnosis()
    test_denom_quad_variations_detail()
