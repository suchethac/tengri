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
        grad = jax.grad(loss)(val)
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
        """Finite *and* live inside the table; identically zero outside it (#1822).

        The gamma axis spans [1.5, 3.5] and kTe [0.05, 0.5]. This test used to
        evaluate at gamma=1.3 with kTe=5.0 — both outside — and assert only
        ``isfinite``. Every probe therefore sat in the clamped region where the
        gradient is exactly 0.0, which ``isfinite`` accepts, so a test named for
        the grid edges never reached a live one.
        """
        if not _is_table_available():
            pytest.skip("nthcomp templates not loaded")
        nu = jnp.linspace(1e12, 1e19, 50, dtype=jnp.float32)

        def loss(gamma):
            lnu = nthcomp_lnu_interp(
                nu, gamma, jnp.array(0.25, dtype=jnp.float32), jnp.array(0.1, dtype=jnp.float32)
            )
            return jnp.sum(lnu)

        # Inside the table: finite AND non-zero, including hard against both ends.
        for gamma_val in [1.51, 2.0, 2.5, 3.49]:
            gamma = jnp.array(gamma_val, dtype=jnp.float32)
            grad = jax.grad(loss)(gamma)
            assert jnp.isfinite(grad), f"Gradient not finite at gamma={gamma_val}: {grad}"
            assert grad != 0.0, (
                f"Gradient is exactly zero at gamma={gamma_val}, which is inside the "
                "table — the rule has stopped differentiating gamma"
            )

        # Outside: clamped, so exactly zero. Asserted rather than tolerated, so
        # that a fit initialized out here is a known-dead configuration and not a
        # surprise (#1521, #1684 are the same silent-clamp shape).
        for gamma_val in [1.0, 1.45, 3.6, 4.0]:
            gamma = jnp.array(gamma_val, dtype=jnp.float32)
            grad = jax.grad(loss)(gamma)
            assert grad == 0.0, (
                f"gamma={gamma_val} is outside the table's [1.5, 3.5] and should clamp "
                f"to a zero gradient, got {grad}"
            )

    def test_nthcomp_lnu_interp_gradient_near_zero_kTbb(self):
        """The kTe gradient is live at small seed temperature (#1822).

        This swept kTe = 1, 2, 5 keV — all above the table's 0.5 keV ceiling, so
        all clamped — and asserted only ``isfinite``. Zero is finite, so it was
        green on a derivative that was identically zero, which is precisely the
        failure this file's docstring says it exists to catch. Probes are now
        inside the grid, and agreement with a central difference is asserted:
        "non-zero" alone would pass on any wrong constant.
        """
        if not _is_table_available():
            pytest.skip("nthcomp templates not loaded")
        nu = jnp.linspace(1e12, 1e19, 50, dtype=jnp.float32)

        def loss(kte):
            lnu = nthcomp_lnu_interp(
                nu,
                jnp.array(2.5, dtype=jnp.float32),
                kte,
                jnp.array(0.01, dtype=jnp.float32),  # Very small kTbb
            )
            return jnp.sum(lnu)

        for kte_val in [0.12, 0.25, 0.45]:
            kte = jnp.array(kte_val, dtype=jnp.float32)
            grad = jax.grad(loss)(kte)
            assert jnp.isfinite(grad), f"Gradient not finite at kTe={kte_val}: {grad}"
            assert grad != 0.0, (
                f"d/d(kTe) is exactly zero at kTe={kte_val}, inside the table — the "
                "custom_jvp has stopped supplying the kTe tangent (#1822)"
            )
            h = 1e-4
            central = float((loss(kte + h) - loss(kte - h)) / (2 * h))
            assert central != 0.0, "setup: reference is zero and cannot judge the tangent"
            rel = abs(float(grad) - central) / abs(central)
            assert rel < 0.10, (
                f"d/d(kTe) = {float(grad):.5e} disagrees with the central difference "
                f"{central:.5e} by {rel:.1%} at kTe={kte_val}"
            )

    def test_nthcomp_lnu_interp_loss_gradient_finite(self):
        """A mock likelihood that is actually a function of gamma (#1822).

        The previous version compared a model of ~1e-18 against ``observed=1.0``
        with ``sigma=0.1``, so ``chi2 = sum((0 - 1)/0.1)**2 = 1.0000e+04``
        *exactly*, at every gamma — verified identical at 1.2 and 2.2. It
        differentiated a constant, and its analytic gradient (~1e-16) was
        numerical dust that ``isfinite`` happily accepted.

        The data are now generated from the model itself at a reference gamma,
        so chi2 has a genuine minimum there and a real slope away from it.
        """
        if not _is_table_available():
            pytest.skip("nthcomp templates not loaded")
        nu = jnp.linspace(1e12, 1e19, 100, dtype=jnp.float32)
        kte = jnp.array(0.25, dtype=jnp.float32)
        ktbb = jnp.array(0.1, dtype=jnp.float32)

        observed_lnu = nthcomp_lnu_interp(nu, jnp.array(2.0, dtype=jnp.float32), kte, ktbb)
        uncertainty = jnp.maximum(0.05 * observed_lnu, 1e-30)

        def mock_likelihood(gamma):
            model_lnu = nthcomp_lnu_interp(nu, gamma, kte, ktbb)
            return jnp.sum(((model_lnu - observed_lnu) / uncertainty) ** 2)

        # The objective must actually vary, or the gradient assertions below are
        # vacuous — this is the check whose absence let a constant through.
        chi2_at = [float(mock_likelihood(jnp.array(g, jnp.float32))) for g in (1.7, 2.0, 2.4)]
        assert chi2_at[1] < chi2_at[0] and chi2_at[1] < chi2_at[2], (
            f"chi2 has no minimum at the generating gamma: {chi2_at} — the mock "
            "likelihood is not a function of gamma"
        )

        for gamma_val in [1.7, 1.9, 2.2, 2.4]:
            gamma = jnp.array(gamma_val, dtype=jnp.float32)
            grad = jax.grad(mock_likelihood)(gamma)
            assert jnp.isfinite(grad), (
                f"Mock likelihood gradient not finite at gamma={gamma_val}: {grad}"
            )
            assert grad != 0.0, (
                f"Mock likelihood gradient is exactly zero at gamma={gamma_val} — "
                "the objective is not seeing gamma"
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

    def test_cotangent_at_the_scale_disc_py_actually_produces(self):
        """1e30 was not large enough to catch the real failure (#1822).

        ``disc.py`` multiplies this kernel's ~1e-19 shape by a ring luminosity,
        so the cotangent reverse mode hands back is ~1e66 — six orders past where
        the test above stops. While the kernel returned float32 regardless of the
        caller's dtype, that cotangent overflowed float32's 3.4e38 ceiling and
        ``jax.grad`` through ``kubota_done_disc`` returned **NaN** for
        ``agn_gamma_warm`` while ``jax.jvp`` returned 5.2e30.

        The sibling test passing at 1e30 is exactly why this went unnoticed: a
        magnitude sweep that stops short of the caller's real scale reports a
        healthy kernel.
        """
        unit = self._grad_gamma(1.0)
        assert unit != 0.0, "baseline gradient is zero — test cannot detect the overflow"

        for exponent in (40, 50, 66):
            scaled = self._grad_gamma(10.0**exponent)
            assert jnp.isfinite(scaled), (
                f"cotangent 1e{exponent} produced {scaled} — the kernel is forcing a "
                "float32 output again, so a realistic ring luminosity overflows"
            )
            assert scaled / unit == pytest.approx(10.0**exponent, rel=1e-5)

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
