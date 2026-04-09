"""Root cause analysis of non-monotonic GL quadrature accuracy.

KEY INSIGHT: The benchmark computes:
  f_b ≈ Φ_b · ⟨A⟩_GL

where:
  Φ_b = exact SSP filter integral (dense trapz) — ALWAYS EXACT
  ⟨A⟩_GL = Σ_k w_k · A(λ_k) · s_{bk}  (GL-averaged dust)

The error comes ENTIRELY from the dust averaging factorization: f_b ≈ Φ_b · ⟨A⟩_GL

The true integral is:
  f_b_true = ∫ SSP(λ) · A(λ) · T(λ) · λ dλ / ∫ T(λ) · λ dλ

The approximation assumes:
  f_b_approx = ∫ SSP(λ) · T(λ) · λ dλ / ∫ T(λ) · λ dλ  ×  ⟨A⟩_GL
             = Φ_b · ⟨A⟩_GL

This factorization is ONLY valid if A(λ) is slowly varying ("smooth") within the
filter bandpass. When n_quad changes, we get DIFFERENT estimates of ⟨A⟩_GL, but
the factorization error DOMINATES.

The key problem: different n_quad gives different denom_quad estimates, so
⟨A⟩_GL values are NOT directly comparable across n_quad.

For Charlot-Fall dust with τ_BC=1, n=-0.7:
  A(λ) = exp(-1.0 * (λ / 5500)^-0.7)

This is NOT a smooth polynomial! It's a steep power law. GL quadrature
is designed for polynomials, not power laws. GL nodes that optimize polynomial
integrals don't optimize power-law integrals.
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


def test_factorization_error_dominates():
    """Demonstrate that dust factorization error, not GL error, dominates."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z"])
        fw_z, ft_z = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS z: {e}")

    print("\n" + "="*80)
    print("ROOT CAUSE: DUST FACTORIZATION ERROR DOMINATES OVER GL ERROR")
    print("="*80)

    # Setup: worst-case steep power-law SSP
    all_waves = fw_z
    lam_min, lam_max = all_waves.min() * 0.9, all_waves.max() * 1.1
    ssp_lam_fine = np.linspace(lam_min, lam_max, 10_000)
    ssp_lam_coarse = np.linspace(lam_min, lam_max, 100)
    ssp_sed = (ssp_lam_fine / ssp_lam_fine.mean()) ** -2

    # Charlot-Fall dust
    tau_bc = 1.0
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)

    # === PART 1: Exact reference (dense 10000-pt grid) ===
    print("\nPart 1: EXACT REFERENCE (10000-pt dense grid)")
    print("-" * 80)

    t_i = np.interp(ssp_lam_fine, fw_z, ft_z, left=0.0, right=0.0)
    dust_fine = dust_fn(ssp_lam_fine)

    # Exact photometry
    num_exact = np.trapezoid(ssp_sed * dust_fine * t_i * ssp_lam_fine, ssp_lam_fine)
    den_exact = np.trapezoid(t_i * ssp_lam_fine, ssp_lam_fine)
    f_exact = num_exact / den_exact
    print(f"f_exact = {f_exact:.8f}")

    # Exact average dust on the fine grid
    a_exact = np.trapezoid(dust_fine * t_i * ssp_lam_fine, ssp_lam_fine) / den_exact
    print(f"⟨A⟩_exact = {a_exact:.6f}")

    # Exact CSP (no dust)
    t_i_fine = np.interp(ssp_lam_fine, fw_z, ft_z, left=0.0, right=0.0)
    csp_exact = np.trapezoid(ssp_sed * t_i_fine * ssp_lam_fine, ssp_lam_fine) / den_exact
    print(f"CSP_exact = {csp_exact:.8f}")
    print(f"CSP_exact × ⟨A⟩_exact = {csp_exact * a_exact:.8f}")
    print(f"Error in factorization: {100 * abs(csp_exact * a_exact - f_exact) / f_exact:.4f}%")

    # === PART 2: GL quadrature with different n_quad ===
    print("\n" + "-" * 80)
    print("Part 2: GL QUADRATURE WITH DIFFERENT n_quad")
    print("-" * 80)

    for n_quad in [1, 3, 5, 7]:
        print(f"\nn_quad = {n_quad}:")

        if n_quad == 1:
            # Single effective wavelength
            lam_eff = np.trapezoid(ft_z * fw_z**2, fw_z) / den_exact
            dust_eff = dust_fn(np.array([lam_eff]))[0]
            a_gl = dust_eff
            print(f"  λ_eff = {lam_eff:.2f} Å")
            print(f"  A(λ_eff) = {a_gl:.6f}")
        else:
            # GL nodes
            nodes, weights, h = _gauss_legendre_nodes_for_filter(fw_z, n_quad)
            t_at_nodes = np.interp(nodes, fw_z, ft_z)

            # Key computation: denom_quad
            denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))

            print(f"  GL nodes: {nodes}")
            print(f"  T(λ_k): {t_at_nodes}")
            print(f"  denom_quad = {denom_quad:.6e}")
            print(f"  denom_exact = {den_exact:.6e}")
            print(f"  ratio (GL/exact) = {denom_quad / den_exact:.6f}")

            # Dust at GL nodes
            dust_at_nodes = dust_fn(nodes)
            print(f"  A(λ_k): {dust_at_nodes}")

            # Scale factors
            scale = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
            print(f"  scales: {scale}")

            # GL-averaged dust
            a_gl = float(np.sum(weights * dust_at_nodes * scale))
            print(f"  ⟨A⟩_GL = Σ w_k A(λ_k) s_k = {a_gl:.6f}")

        # Factorization: Φ_b × ⟨A⟩_GL
        f_factorized = csp_exact * a_gl
        err_pct = 100 * abs(f_factorized - f_exact) / abs(f_exact)
        print(f"\n  CSP_exact × ⟨A⟩_GL = {f_factorized:.8f}")
        print(f"  Error: {err_pct:.4f}%")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("The non-monotonic accuracy is due to the factorization error")
    print("f ≈ CSP × ⟨A⟩_GL being LARGER than the GL quadrature error itself.")
    print("Different n_quad estimates of ⟨A⟩_GL are NOT directly comparable")
    print("because the scale factors depend on denom_quad, which varies with n_quad.")
    print("="*80)


def test_denom_quad_variations():
    """Show how denom_quad varies dramatically with n_quad."""
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z", "sdss_g"])
        filters = dict(z=(np.asarray(fw_list[0]), np.asarray(ft_list[0])),
                       g=(np.asarray(fw_list[1]), np.asarray(ft_list[1])))
    except Exception as e:
        pytest.skip(f"Could not load filters: {e}")

    print("\n" + "="*80)
    print("denom_quad VARIATIONS WITH n_quad")
    print("="*80)

    for band, (fw, ft) in filters.items():
        print(f"\n{band.upper()}-BAND:")
        print("-" * 40)

        # Exact denominator
        denom_exact = np.trapezoid(ft * fw, fw)
        print(f"Exact ∫T(λ)λ dλ = {denom_exact:.6e}")

        for n_quad in [1, 3, 5, 7]:
            if n_quad == 1:
                # Effective wavelength uses different normalization
                denom_eff = np.trapezoid(ft * fw, fw)
                print(f"n=1 (effective λ): {denom_eff:.6e} (same as exact)")
            else:
                nodes, weights, h = _gauss_legendre_nodes_for_filter(fw, n_quad)
                t_at_nodes = np.interp(nodes, fw, ft)
                denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
                ratio = denom_quad / denom_exact
                err = 100 * abs(ratio - 1.0)
                print(f"n={n_quad}: {denom_quad:.6e}  (ratio={ratio:.4f}, error={err:.2f}%)")


if __name__ == "__main__":
    test_factorization_error_dominates()
    test_denom_quad_variations()
