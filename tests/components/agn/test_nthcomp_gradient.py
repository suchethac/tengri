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


@pytest.mark.skipif(not _is_table_available(), reason="nthcomp templates not loaded")
class TestNthcompJVPRule:
    """The gamma rule is a ``custom_jvp``, so both autodiff modes work (#1206)."""

    NU = jnp.geomspace(1e16, 1e19, 7)
    ARGS = (jnp.array(2.0), jnp.array(100.0), jnp.array(0.01))

    def _grad_gamma(self, cotangent):
        gamma, kte, ktbb = self.ARGS

        def loss(g):
            return jnp.sum(nthcomp_lnu_interp(self.NU, g, kte, ktbb)) * cotangent

        return float(jax.grad(loss)(gamma))

    def test_forward_mode_autodiff_is_supported(self):
        """``jax.jvp`` must not raise — geoVI builds its metric with forward mode.

        A ``custom_vjp`` is opaque to forward mode and raised ``TypeError: can't
        apply forward-mode autodiff (jvp) to a custom_vjp function``, taking out
        every geoVI fit reaching an AGN model with this kernel.
        """
        gamma, kte, ktbb = self.ARGS
        primals = (self.NU, gamma, kte, ktbb)
        tangents = (jnp.zeros_like(self.NU), jnp.array(1.0), jnp.array(0.0), jnp.array(0.0))

        _, tangent_out = jax.jvp(nthcomp_lnu_interp, primals, tangents)

        assert tangent_out.shape == self.NU.shape
        assert bool(jnp.all(jnp.isfinite(tangent_out)))
        assert bool(jnp.any(tangent_out != 0.0)), "forward mode returned an all-zero tangent"

    def test_large_cotangent_does_not_collapse_to_zero(self):
        """A big incoming cotangent must scale the gradient, not zero it.

        The retired reverse rule divided the cotangent by ``max|fd_grad|`` ~1e-17
        "to avoid overflow", which sent a 1e30 cotangent to ~1e47 — past float32's
        3.4e38 — and a trailing ``where(isfinite, ..., 0.0)`` turned the ``inf``
        into a silent zero. Measured before the fix: exactly ``0.0``.

        Gradients are linear in the cotangent, so the ratio is the invariant;
        pinning it catches both the collapse and any rescaling that does not
        cancel.
        """
        unit = self._grad_gamma(1.0)
        assert unit != 0.0, "baseline gradient is zero — test cannot detect the collapse"

        scaled = self._grad_gamma(1e30)

        assert scaled != 0.0, "large cotangent collapsed to a silent zero gradient"
        assert jnp.isfinite(scaled)
        assert scaled / unit == pytest.approx(1e30, rel=1e-5)

    def test_float32_grid_with_float64_gamma_is_accepted(self):
        """A float32 SED grid with a float64 ``gamma`` must still differentiate.

        ``custom_jvp`` requires the tangent's dtype to MATCH the primal's — a
        contract ``custom_vjp`` never imposed, so it is the one way that
        conversion can regress. ``nu`` fixes the primal dtype while ``gamma``
        fixes the tangent's, so this mixed pairing promotes the tangent to
        float64 and JAX rejects the rule at trace time::

            TypeError: Custom JVP rule must produce primal and tangent outputs
            with corresponding shapes and dtypes.

        It is a hard error, not a wrong number, and it reached CI as the
        B1_agn_disc_torus scenario failing in the slow integration tier — no
        unit test paired the dtypes this way.
        """
        _, kte, ktbb = self.ARGS
        nu32 = self.NU.astype(jnp.float32)
        args = (nu32, jnp.float64(2.0), jnp.float64(float(kte)), jnp.float64(float(ktbb)))
        tangents = (
            jnp.zeros_like(nu32),
            jnp.float64(1.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
        )

        primal_out, tangent_out = jax.jvp(nthcomp_lnu_interp, args, tangents)

        assert tangent_out.dtype == primal_out.dtype
        assert bool(jnp.all(jnp.isfinite(tangent_out)))

        # Reverse mode over the same mixed pairing must also survive.
        def loss(g):
            return jnp.sum(nthcomp_lnu_interp(nu32, g, args[2], args[3])).astype(jnp.float64)

        assert jnp.isfinite(jax.grad(loss)(args[1]))
