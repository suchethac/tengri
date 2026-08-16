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
        """The clamp's derivative by value, because ``isfinite`` cannot see a dead axis.

        On a uniform grid of span 1 the fractional index has three distinct
        derivatives, and all three are worth pinning:

        * ``1/span`` = 1.0 strictly inside a cell
        * ``0.5`` exactly on a node -- JAX averages the two one-sided
          derivatives through the ``where``, so a fit that lands on a node takes
          half the step it would take anywhere else
        * exactly ``0.0`` outside the grid, where the clamp holds ``frac`` fixed

        The version this replaces evaluated at ``grid[0]``, ``grid[2]`` and
        ``grid[-1]`` -- all three ON nodes, all three 0.5 -- and asserted only
        that they were finite. It never reached the clamped region it is named
        for, and zero is finite, so it would have passed just as well if every
        one of them had been dead.
        """
        grid = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)

        def loss(v):
            _, frac = _clamp_interp_index(v, grid)
            return jnp.sum(frac)

        interior = float(jax.grad(loss)(jnp.array(2.5, dtype=jnp.float32)))
        assert interior == pytest.approx(1.0), (
            f"interior d(frac)/dv should be 1/span = 1.0, got {interior}"
        )

        for node in (grid[0], grid[2], grid[-1]):
            on_node = float(jax.grad(loss)(node))
            assert on_node == pytest.approx(0.5), (
                f"on node {float(node)} the two-sided average should be 0.5, got {on_node}"
            )

        for outside in (0.5, 6.0):
            clamped = float(jax.grad(loss)(jnp.array(outside, dtype=jnp.float32)))
            assert jnp.isfinite(clamped)
            assert clamped == 0.0, (
                f"outside the grid the clamp must hold frac fixed, so the "
                f"gradient is exactly zero; at v={outside} it is {clamped}. If "
                f"this became nonzero the clamp changed and the dead-zone "
                f"behaviour pinned in this file is stale."
            )

    #: Measured on the shipped table (#1822): the gamma axis is live on
    #: [1.5, 3.5]. Outside it the summed L_nu is bit-identical and the gradient
    #: is exactly 0.0. The endpoints themselves are excluded from the
    #: finite-difference checks -- a one-sided derivative at a clamp is not
    #: something a central difference can reference.
    GAMMA_LIVE = (1.7, 2.5, 3.0)
    GAMMA_CLAMPED = (1.0, 1.3, 4.0)

    @staticmethod
    def _sum_lnu(gamma, kte=5.0, ktbb=0.1, n_nu=50):
        nu = jnp.linspace(1e12, 1e19, n_nu, dtype=jnp.float32)
        return jnp.sum(
            nthcomp_lnu_interp(
                nu,
                gamma,
                jnp.array(kte, dtype=jnp.float32),
                jnp.array(ktbb, dtype=jnp.float32),
            )
        )

    @pytest.mark.parametrize("gamma_val", GAMMA_LIVE)
    def test_gamma_gradient_is_live_inside_the_table(self, gamma_val):
        """Inside [1.5, 3.5] the gamma gradient must be nonzero AND correct.

        The version this replaces asserted only ``isfinite``, and swept
        ``gamma = 1.3`` -- which is *below* the table, where the gradient is
        exactly zero. It was named for the grid edges and never reached a live
        one, so it could not have failed for a frozen gamma axis.

        The step and tolerance are explicit because nthcomp reads a **float32**
        lookup table -- and does so whatever dtype it is handed, so promoting
        the inputs to float64 does not help. Summing 50 float32 samples of a
        ~1e-18 spectrum leaves the central difference with about three good
        digits, and no step recovers more: measured relative agreement is
        9e-4 at h/x = 1e-3, degrading to 5e-2 by h/x = 1e-5 as cancellation
        takes over. 1e-3 is the optimum, so it is named rather than left to the
        helper's default of 1e-5.

        Before ``atol`` defaulted to 0.0 this comparison was inert: the
        gradients here are ~1e-18 against the old fixed floor of 1e-8, so
        substituting zero for the analytic gradient still passed. The
        ``!= 0.0`` assertion below was carrying the whole test.
        """
        gamma = jnp.array(gamma_val, dtype=jnp.float32)
        grad = assert_grad_matches_fd(self._sum_lnu, gamma, rtol=3e-2, eps=1e-3 * gamma_val)
        assert float(grad) != 0.0, (
            f"gamma={gamma_val} is inside the table but its gradient is exactly "
            f"zero, so a fit cannot move here"
        )

    @pytest.mark.parametrize("gamma_val", GAMMA_CLAMPED)
    def test_gamma_is_frozen_outside_the_table(self, gamma_val):
        """Outside [1.5, 3.5] the spectrum is frozen and the gradient is zero.

        Pinned as measured behaviour, not endorsed: a fit initialised out here
        cannot move, because the gradient is exactly 0.0 rather than merely
        small. Whether the clamp should instead raise is #1822. Asserting it
        makes the dead zone visible, and makes a future bounds change loud.
        """
        clamped_to = 1.5 if gamma_val < 1.5 else 3.5
        frozen = float(self._sum_lnu(jnp.array(clamped_to, dtype=jnp.float32)))
        here = float(self._sum_lnu(jnp.array(gamma_val, dtype=jnp.float32)))

        assert here == frozen, (
            f"gamma={gamma_val} should return the value at the clamped edge "
            f"{clamped_to}; got {here} vs {frozen}"
        )
        grad = float(jax.grad(self._sum_lnu)(jnp.array(gamma_val, dtype=jnp.float32)))
        assert grad == 0.0, f"expected an exactly zero gradient in the clamped region, got {grad}"

    @pytest.mark.xfail(
        reason=(
            "#1822: nthcomp_lnu_interp ignores kTe_keV. sum(L_nu) is bit-identical "
            "for kTe in (1, 20, 400) keV at every (gamma, kTbb) corner measured, and "
            "d/d(kTe) is exactly 0.0 at kTe = 0.5 ... 500. kTe sets the Comptonization "
            "rollover and the probe band straddles it, so this cannot be physical."
        ),
        strict=True,
    )
    def test_kte_changes_the_spectrum(self):
        """kTe must do something. It is an axis of the trilinear table.

        The test this replaces swept kTe = 1, 2, 5 and asserted only that the
        gradient was finite. Zero is finite, so it was green on an inert
        parameter -- the exact failure mode this file's docstring says it exists
        to catch.
        """
        cold = float(self._sum_lnu(jnp.array(1.8, dtype=jnp.float32), kte=1.0, ktbb=0.01))
        hot = float(self._sum_lnu(jnp.array(1.8, dtype=jnp.float32), kte=400.0, ktbb=0.01))
        assert cold != hot, (
            f"a 400x change in electron temperature left sum(L_nu) bit-identical at {cold!r}"
        )

    def test_kte_is_inert_and_the_neighbouring_axis_is_not(self):
        """Pin #1822 precisely, so the xfail above cannot pass for the wrong reason.

        If ``nthcomp_lnu_interp`` started returning a constant for *everything*,
        the strict xfail would flip to green and read as a fix. kTbb is the
        control: it shares the same trilinear interpolation and does respond.
        """
        gamma = jnp.array(1.8, dtype=jnp.float32)
        kte_vals = {float(self._sum_lnu(gamma, kte=k, ktbb=0.01)) for k in (1.0, 20.0, 400.0)}
        assert len(kte_vals) == 1, "kTe now varies the spectrum -- update #1822 and the xfail"

        ktbb_vals = {float(self._sum_lnu(gamma, kte=5.0, ktbb=b)) for b in (0.001, 0.05, 0.5)}
        assert len(ktbb_vals) == 3, (
            "kTbb is the live control for the kTe finding; if it too is inert, "
            "the whole interpolation is frozen and #1822 understates the problem"
        )

    def test_mock_likelihood_has_a_minimum_a_fit_could_find(self):
        """An integration check on a likelihood that is actually a function of gamma.

        The version this replaces compared a model of order 1e-18 against
        ``observed_lnu = ones``, so chi2 was ``sum((0 - 1) / 0.1)**2`` = 1.0000e4
        at *every* gamma -- verified identical at 1.2 and 2.2. It differentiated
        a constant, and its analytic gradient was numerical dust (~1e-16 against
        a finite difference of exactly 0).

        Generating the data from the model at a known gamma gives a chi2 with a
        real minimum, which is what an inference run actually descends.
        """
        truth = jnp.array(2.0, dtype=jnp.float32)
        # 50 points, matching _sum_lnu: a second grid size is a second XLA
        # compile of the same kernel, and this file is in the fast tier.
        nu = jnp.linspace(1e12, 1e19, 50, dtype=jnp.float32)

        def model(gamma):
            return nthcomp_lnu_interp(
                nu, gamma, jnp.array(5.0, dtype=jnp.float32), jnp.array(0.1, dtype=jnp.float32)
            )

        observed = model(truth)
        uncertainty = 0.1 * jnp.maximum(observed, jnp.max(observed) * 1e-3)

        def chi2(gamma):
            return jnp.sum(((model(gamma) - observed) / uncertainty) ** 2)

        assert float(chi2(truth)) == pytest.approx(0.0, abs=1e-6), (
            "chi2 at the generating gamma is not zero; the mock data and the "
            "model disagree, so nothing below means what it says"
        )

        for gamma_val, expected_sign in ((1.8, -1.0), (2.2, +1.0), (2.6, +1.0)):
            gamma = jnp.array(gamma_val, dtype=jnp.float32)
            grad = float(jax.grad(chi2)(gamma))
            assert jnp.isfinite(grad)
            assert grad != 0.0, f"chi2 gradient is exactly zero at gamma={gamma_val}"
            assert grad * expected_sign > 0, (
                f"at gamma={gamma_val} the chi2 gradient should point away from "
                f"the minimum at {float(truth)}, got {grad}"
            )
            assert float(chi2(gamma)) > 0.0


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
