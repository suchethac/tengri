"""Diagnostic test for GL quadrature non-monotonic accuracy issue."""
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


def test_quad_diagnostic():
    """Analyze GL node sampling for SDSS z-band."""
    # Load SDSS z-band
    try:
        fw_list, ft_list, _ = load_filter_set(["sdss_z"])
        fw_z, ft_z = np.asarray(fw_list[0]), np.asarray(ft_list[0])
    except Exception as e:
        pytest.skip(f"Could not load SDSS z: {e}")

    # Test dust function (Charlot-Fall: τ_BC=1, n=-0.7)
    tau_bc = 1.0
    dust_fn = lambda lam: np.exp(-tau_bc * (np.asarray(lam) / 5500.0) ** -0.7)

    print("\n" + "="*80)
    print("SDSS Z-BAND: GL NODE ANALYSIS")
    print("="*80)

    print(f"\nFilter wavelength range: {fw_z.min():.1f}–{fw_z.max():.1f} Å")
    print(f"Filter grid: {len(fw_z)} points")

    # Compute reference normalization (exact dense grid)
    t_i = ft_z
    lam_exact_denom = np.trapezoid(t_i * fw_z, fw_z)
    print(f"\nExact (dense trapz) normalization ∫T(λ)λ dλ = {lam_exact_denom:.6e}")

    for n_quad in [1, 3, 5, 7]:
        print(f"\n{'-'*80}")
        print(f"n_quad = {n_quad}")
        print(f"{'-'*80}")

        if n_quad == 1:
            # Single effective wavelength method
            lam_eff = np.trapezoid(ft_z * fw_z**2, fw_z) / lam_exact_denom
            print(f"Single effective wavelength λ_eff = {lam_eff:.2f} Å")
            dust_eff = dust_fn(np.array([lam_eff]))[0]
            print(f"Dust at λ_eff: A(λ_eff) = {dust_eff:.6f}")
        else:
            # GL nodes
            nodes, weights, h = _gauss_legendre_nodes_for_filter(fw_z, n_quad)

            print(f"GL half-width h = (λ_max - λ_min)/2 = {h:.2f} Å")
            print(f"GL weights (sum={np.sum(weights):.4f}): {weights}")

            # Interpolate filter at nodes
            t_at_nodes = np.interp(nodes, fw_z, ft_z)
            print(f"\nGL nodes (Å) and filter transmission:")
            for i, (node, t) in enumerate(zip(nodes, t_at_nodes)):
                print(f"  Node {i}: λ = {node:8.2f} Å,  T(λ) = {t:.6f}")

            # Dust at nodes
            dust_at_nodes = dust_fn(nodes)
            print(f"\nDust attenuation A(λ) = exp(-1.0 * (λ/5500)^-0.7):")
            for i, (node, dust) in enumerate(zip(nodes, dust_at_nodes)):
                print(f"  Node {i}: λ = {node:8.2f} Å,  A(λ) = {dust:.6f}")

            # GL quadrature normalization (used in precompute)
            denom_quad = h * float(np.sum(weights * t_at_nodes * nodes))
            print(f"\nGL quadrature normalization ∫_GL T(λ)λ dλ = {denom_quad:.6e}")
            print(f"  = h * Σ w_k * T(λ_k) * λ_k")
            print(f"  = {h:.2f} * {np.sum(weights * t_at_nodes * nodes):.6e}")
            print(f"\nRatio (GL / exact) = {denom_quad / lam_exact_denom:.6f}")

            # Scale factors used for dust GL averaging
            scale = t_at_nodes * nodes * h / max(denom_quad, 1e-30)
            print(f"\nScale factors s_k = T(λ_k) * λ_k * h / denom_quad:")
            for i, (node, s) in enumerate(zip(nodes, scale)):
                print(f"  Scale[{i}]: λ = {node:8.2f} Å,  s_k = {s:.6e}")

            # GL-averaged dust
            dust_avg_gl = float(np.sum(weights * dust_at_nodes * scale))
            print(f"\nGL-averaged dust ⟨A⟩_GL = Σ w_k * A(λ_k) * s_k = {dust_avg_gl:.6f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_quad_diagnostic()
