# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for nthcomp gradient stability.

Tests the JAX autodiff behavior when division-by-near-zero occurs in where
clauses, which can produce NaN gradients even when the unselected branch is
masked. This tests the fix for the double-where safety pattern.
"""

import pytest

pytestmark = pytest.mark.gradient
import jax
import jax.numpy as jnp

from tengri.components.agn._nthcomp import (
    _clamp_interp_index,
    _is_table_available,
    nthcomp_lnu_interp,
)
from tests._grad_parity import assert_grad_matches_fd


@pytest.mark.skipif(not _is_table_available(), reason="nthcomp templates not loaded")
class TestNthcompGradientStability:
    """Test gradient stability of nthcomp functions at parameter grid edges."""

    def test_clamp_interp_index_zero_span_gradient(self):
        """Verify finite gradients when span approaches zero."""
        # This is the critical test: when two adjacent grid points coincide
        # (or very nearly so), the unselected branch still flows through
        # autodiff and can produce NaN if not handled carefully.
        grid = jnp.array([1.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        val = jnp.array(1.0, dtype=jnp.float32)

        def loss(v):
            _, frac = _clamp_interp_index(v, grid)
            return jnp.sum(frac)

        # This should NOT produce NaN or Inf
        grad = assert_grad_matches_fd(loss, val)
        assert jnp.isfinite(grad), f"Expected finite gradient, got {grad}"

    def test_clamp_interp_index_gradient_at_boundary(self):
        """Verify finite gradients at grid boundaries."""
        grid = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)

        def loss(v):
            _, frac = _clamp_interp_index(v, grid)
            return jnp.sum(frac)

        # Test at the boundary where clamping takes effect
        grad_at_min = jax.grad(loss)(grid[0])
        grad_at_max = jax.grad(loss)(grid[-1])
        grad_in_middle = jax.grad(loss)(grid[2])
        assert jnp.isfinite(grad_at_min), f"Gradient at min not finite: {grad_at_min}"
        assert jnp.isfinite(grad_at_max), f"Gradient at max not finite: {grad_at_max}"
        assert jnp.isfinite(grad_in_middle), f"Gradient in middle not finite: {grad_in_middle}"

    def test_nthcomp_lnu_interp_gradient_at_grid_edges(self):
        """Verify finite gradients when interpolating near grid boundaries."""
        if not _is_table_available():
            pytest.skip("nthcomp templates not loaded")
        # Use reasonable parameter values close to the grid boundaries
        nu = jnp.linspace(1e12, 1e19, 50, dtype=jnp.float32)

        def loss(gamma):
            lnu = nthcomp_lnu_interp(
                nu, gamma, jnp.array(5.0, dtype=jnp.float32), jnp.array(0.1, dtype=jnp.float32)
            )
            return jnp.sum(lnu)

        # Test at various gamma values
        for gamma_val in [1.3, 1.5, 2.0, 2.5]:
            gamma = jnp.array(gamma_val, dtype=jnp.float32)
            grad = jax.grad(loss)(gamma)
            assert jnp.isfinite(grad), f"Gradient not finite at gamma={gamma_val}: {grad}"

    def test_nthcomp_lnu_interp_gradient_near_zero_kTbb(self):
        """Verify stable gradients with very small seed temperature."""
        if not _is_table_available():
            pytest.skip("nthcomp templates not loaded")
        nu = jnp.linspace(1e12, 1e19, 50, dtype=jnp.float32)

        def loss(kte):
            lnu = nthcomp_lnu_interp(
                nu,
                jnp.array(1.8, dtype=jnp.float32),
                kte,
                jnp.array(0.01, dtype=jnp.float32),  # Very small kTbb
            )
            return jnp.sum(lnu)

        # This should remain finite even at small electron temperatures
        for kte_val in [1.0, 2.0, 5.0]:
            kte = jnp.array(kte_val, dtype=jnp.float32)
            grad = jax.grad(loss)(kte)
            assert jnp.isfinite(grad), f"Gradient not finite at kTe={kte_val}: {grad}"

    def test_nthcomp_lnu_interp_loss_gradient_finite(self):
        """Integration test: gradient of a mock likelihood is finite."""
        if not _is_table_available():
            pytest.skip("nthcomp templates not loaded")
        nu = jnp.linspace(1e12, 1e19, 100, dtype=jnp.float32)
        observed_lnu = jnp.ones_like(nu)
        uncertainty = jnp.ones_like(nu) * 0.1

        def mock_likelihood(gamma):
            model_lnu = nthcomp_lnu_interp(
                nu, gamma, jnp.array(5.0, dtype=jnp.float32), jnp.array(0.1, dtype=jnp.float32)
            )
            chi2 = jnp.sum(((model_lnu - observed_lnu) / uncertainty) ** 2)
            return chi2

        # Test gradient at various points in parameter space
        for gamma_val in [1.2, 1.5, 1.8, 2.2]:
            gamma = jnp.array(gamma_val, dtype=jnp.float32)
            grad = jax.grad(mock_likelihood)(gamma)
            assert jnp.isfinite(grad), (
                f"Mock likelihood gradient not finite at gamma={gamma_val}: {grad}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
