# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for nthcomp gradient stability.

Covers the JAX autodiff behavior of the trilinear nthcomp lookup: that each of
its three axes is live where the table supports it, exactly frozen where the
clamp takes over, and that both autodiff modes survive the cotangent
magnitudes ``disc.py`` actually produces (#1206, #1822).
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

        # This should NOT produce NaN or Inf
        def loss(v):
            _, frac = _clamp_interp_index(v, grid)
            return jnp.sum(frac)

        grad = assert_grad_matches_fd(loss, val)
        assert jnp.isfinite(grad), f"Expected finite gradient, got {grad}"
        assert jnp.any(grad != 0.0), (
            "`grad` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

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
                f"behavior pinned in this file is stale."
            )

    #: Measured on the shipped table (#1822): the gamma axis is live on
    #: [1.5, 3.5]. Outside it the summed L_nu is bit-identical and the gradient
    #: is exactly 0.0. The endpoints themselves are excluded from the
    #: finite-difference checks -- a one-sided derivative at a clamp is not
    #: something a central difference can reference.
    GAMMA_LIVE = (1.7, 2.5, 3.0)
    GAMMA_CLAMPED = (1.0, 1.3, 4.0)

    #: The kTe axis, measured the same way: 15 nodes spanning [0.05, 0.5] keV at
    #: a spacing of 0.0321. This is RELAGN's *warm* Comptonization region, not a
    #: hot corona, and ``agn_kt_warm`` is declared ``Uniform(0.1, 0.5)`` --
    #: entirely inside it. Probes sit off-node: on a node the two one-sided
    #: derivatives average, so a central difference is referencing something
    #: else (see the 0.5 pinned above for the clamp's own version of this).
    KTE_LIVE = (0.12, 0.25, 0.32)
    KTE_CLAMPED = (0.01, 1.0, 400.0)

    @staticmethod
    def _sum_lnu(gamma, kte=0.25, ktbb=0.1, n_nu=50):
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
        assert jnp.all(jnp.isfinite(float(grad))), (
            "`float(grad)` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    @pytest.mark.parametrize("gamma_val", GAMMA_CLAMPED)
    def test_gamma_is_frozen_outside_the_table(self, gamma_val):
        """Outside [1.5, 3.5] the spectrum is frozen and the gradient is zero.

        Pinned as measured behavior, not endorsed: a fit initialized out here
        cannot move, because the gradient is exactly 0.0 rather than merely
        small. Asserting it makes the dead zone visible, and makes a future
        bounds change loud.
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

    def test_kte_changes_the_spectrum_inside_the_table(self):
        """kTe is an axis of the trilinear table, and it does vary the spectrum.

        #1822 reported the opposite -- "sum(L_nu) is bit-identical for kTe in
        (1, 20, 400) keV at every (gamma, kTbb) corner measured" -- and that
        measurement reproduces exactly. But all three probes are above the
        table's 0.5 keV ceiling, so all three clamp to the same edge and return
        ``f(0.50)``. That is correct edge-clamping, not an ignored parameter,
        and this test exists so nobody re-fixes a working interpolation.

        This file previously carried that reading as a ``strict=True`` xfail
        asserting kTe was inert. It passed -- the probes really are identical --
        which is the trap: a strict xfail is a claim that something is broken,
        and one written from a wrong diagnosis stays green while pinning the
        error in place.

        kTbb is swept alongside as a control. If the response were a global
        scale rather than a per-axis one, both would move together.
        """
        gamma = jnp.array(2.0, dtype=jnp.float32)

        cold = float(self._sum_lnu(gamma, kte=0.05))
        hot = float(self._sum_lnu(gamma, kte=0.5))
        assert hot / cold == pytest.approx(4.5, rel=0.1), (
            f"measured 4.5x across the kTe axis [0.05, 0.5]; got {hot / cold:.3f} "
            f"({cold!r} -> {hot!r})"
        )

        ktbb_vals = {float(self._sum_lnu(gamma, ktbb=b)) for b in (0.001, 0.05, 0.3)}
        assert len(ktbb_vals) == 3, (
            "kTbb is the control for the kTe measurement above; if it is inert "
            "too then the interpolation is frozen and neither number means what "
            "it says"
        )

    @pytest.mark.parametrize("kte_val", KTE_LIVE)
    def test_kte_gradient_is_live_inside_the_table(self, kte_val):
        """Inside [0.05, 0.5] keV the kTe gradient must be nonzero AND correct (#1822).

        The ``custom_jvp`` rule unpacked ``_, _, d_gamma, _, _`` and returned
        ``fd_grad * d_gamma``, discarding the kTe tangent entirely, so
        ``d/d(agn_kt_warm)`` was exactly 0.0 everywhere -- including across the
        whole of its declared ``Uniform(0.1, 0.5)`` prior, where the forward
        response is 4.5x. A gradient fit left it at its initial value and
        returned the prior as though it had been fitted.

        The test this replaces swept kTe = 1, 2, 5 keV -- all above the ceiling,
        so all clamped -- and asserted only ``isfinite``. Zero is finite, so it
        was green on a derivative that was identically zero, which is precisely
        the failure this file's docstring says it exists to catch.

        Step and tolerance are measured, as for gamma: at h/x = 1e-3 the
        analytic and central-difference values agree to between 0.1% and 0.7%
        across these probes, degrading to ~3% by h/x = 1e-4 as float32
        cancellation takes over.
        """
        kte = jnp.array(kte_val, dtype=jnp.float32)
        grad = assert_grad_matches_fd(
            lambda k: self._sum_lnu(jnp.array(2.0, dtype=jnp.float32), kte=k),
            kte,
            rtol=3e-2,
            eps=1e-3 * kte_val,
        )
        assert float(grad) != 0.0, (
            f"kTe={kte_val} is inside the table but d/d(kTe) is exactly zero -- "
            f"the custom_jvp has stopped supplying the kTe tangent (#1822)"
        )
        assert jnp.all(jnp.isfinite(float(grad))), (
            "`float(grad)` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    @pytest.mark.parametrize("kte_val", KTE_CLAMPED)
    def test_kte_is_frozen_outside_the_table(self, kte_val):
        """Outside [0.05, 0.5] keV the spectrum is frozen and the gradient is zero.

        The same dead zone pinned for gamma, and the one the issue's probes
        landed in. ``agn_kt_warm``'s declared prior does not reach here, so this
        is documentation of the clamp rather than a fit-reachable regime.
        """
        clamped_to = 0.05 if kte_val < 0.05 else 0.5
        gamma = jnp.array(2.0, dtype=jnp.float32)

        frozen = float(self._sum_lnu(gamma, kte=clamped_to))
        here = float(self._sum_lnu(gamma, kte=kte_val))
        assert here == frozen, (
            f"kTe={kte_val} should return the value at the clamped edge "
            f"{clamped_to}; got {here} vs {frozen}"
        )

        grad = float(
            jax.grad(lambda k: self._sum_lnu(gamma, kte=k))(jnp.array(kte_val, dtype=jnp.float32))
        )
        assert grad == 0.0, f"expected an exactly zero gradient in the clamped region, got {grad}"

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
                nu, gamma, jnp.array(0.25, dtype=jnp.float32), jnp.array(0.1, dtype=jnp.float32)
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
        assert jnp.any(tangent_out != 0.0), (
            "`tangent_out` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

        # Reverse mode over the same mixed pairing must also survive.
        def loss(g):
            return jnp.sum(nthcomp_lnu_interp(nu32, g, args[2], args[3])).astype(jnp.float64)

        assert jnp.isfinite(jax.grad(loss)(args[1]))
