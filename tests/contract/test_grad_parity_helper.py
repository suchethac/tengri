# SPDX-License-Identifier: BSD-3-Clause
"""The finite-difference gradient check must fail on a wrong gradient.

The failure mode is not hypothetical here: this tree carries hand-written
derivative rules (``tests/components/agn/test_nthcomp_custom_vjp.py``), and a
custom JVP/VJP with a wrong factor is precisely the bug that produces a
perfectly finite, perfectly plausible, wrong gradient. So the check is
exercised against exactly that — a ``jax.custom_jvp`` whose declared rule is
off by a constant — rather than against a synthetic mismatch.

The last class documents the check's own limit: on a non-smooth kernel it
reports a disagreement that is real but uninteresting, which is why converting
a finiteness assertion to this check is not mechanical everywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import _fd_noise_floor, assert_grad_matches_fd

pytestmark = pytest.mark.contract


class TestItPassesOnCorrectGradients:
    def test_scalar_input(self):
        g = assert_grad_matches_fd(lambda x: x**3, 2.0)
        assert float(g) == pytest.approx(12.0, rel=1e-9)

    def test_array_input_via_directional_derivative(self):
        g = assert_grad_matches_fd(lambda v: jnp.sum(jnp.sin(v)), jnp.linspace(0.1, 1.0, 8))
        np.testing.assert_allclose(np.asarray(g), np.cos(np.linspace(0.1, 1.0, 8)), rtol=1e-9)

    def test_the_analytic_gradient_is_returned(self):
        """Callers keep their existing assertions on the return value."""
        g = assert_grad_matches_fd(lambda x: 5.0 * x, 1.0)
        assert float(g) == pytest.approx(5.0)

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    def test_step_scales_with_the_input(self, scale):
        """A fixed absolute step would be either pure round-off at 1e6 or a
        huge excursion at 1e-6; the step tracks |x| instead."""
        assert_grad_matches_fd(lambda x: x**2, scale)


class TestItFailsOnWrongGradients:
    def test_catches_a_custom_jvp_with_the_wrong_factor(self):
        """The real bug: a hand-written rule that is finite and wrong."""

        @jax.custom_jvp
        def square(x):
            return x**2

        @square.defjvp
        def _square_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            return square(x), 3.0 * x * t  # correct rule is 2 * x * t

        with pytest.raises(AssertionError, match="finite difference"):
            assert_grad_matches_fd(square, 2.0)

    def test_the_wrong_gradient_would_have_passed_a_finiteness_check(self):
        """Proves the upgrade is not cosmetic — the assertion these 202 tests
        make today is green on the gradient the new check rejects."""

        @jax.custom_jvp
        def square(x):
            return x**2

        @square.defjvp
        def _square_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            return square(x), 3.0 * x * t

        assert jnp.isfinite(jax.grad(square)(2.0))  # the old check: passes

    def test_catches_a_silently_detached_parameter(self):
        """``stop_gradient`` left in by accident zeroes the gradient. Zero is
        finite, so only a numerical reference notices."""

        def detached(x):
            return jax.lax.stop_gradient(x) ** 2

        with pytest.raises(AssertionError, match="finite difference"):
            assert_grad_matches_fd(detached, 2.0)

    def test_catches_a_sign_error(self):
        @jax.custom_jvp
        def f(x):
            return jnp.exp(x)

        @f.defjvp
        def _f_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            return f(x), -jnp.exp(x) * t

        with pytest.raises(AssertionError, match="finite difference"):
            assert_grad_matches_fd(f, 0.5)


class TestTheFiniteDifferenceNoiseFloor:
    """Differencing amplifies f's own rounding error by 1/(2h). Without a floor
    scaled to that, every legitimately-zero gradient in float32 reads as wrong.
    """

    def test_a_true_zero_gradient_in_float32_is_not_reported_as_wrong(self):
        """The case that caught this: a float32 sum of order 4 whose derivative
        is exactly zero. One rounding quantum over 2h is ~2.4e-2, so a naive
        atol of 1e-8 calls 0.0 a disagreement."""
        const = jnp.float32(4.0)

        def f(x):
            # genuinely independent of x, evaluated in float32
            return jnp.sum(jnp.full((4,), const, dtype=jnp.float32)) + 0.0 * jnp.sum(
                jnp.asarray(x, dtype=jnp.float32)
            )

        g = assert_grad_matches_fd(f, jnp.arange(4.0))
        np.testing.assert_allclose(np.asarray(g), np.zeros(4), atol=0)

    def test_the_floor_scales_with_magnitude_and_shrinks_with_step(self):
        """It must track |f| and 1/h, or it is just another magic constant."""
        assert _fd_noise_floor(4.0, 4.0, 1e-5) > _fd_noise_floor(4.0, 4.0, 1e-3)
        assert _fd_noise_floor(400.0, 400.0, 1e-5) > _fd_noise_floor(4.0, 4.0, 1e-5)

    def test_float64_gets_a_far_tighter_floor_than_float32(self):
        """Otherwise the float32 allowance would mask real f64 errors — the
        draine_li2014 discrepancy is 7-17%, and must stay visible."""
        f32 = _fd_noise_floor(jnp.float32(4.0), jnp.float32(4.0), 1e-5)
        f64 = _fd_noise_floor(jnp.float64(4.0), jnp.float64(4.0), 1e-5)
        assert f64 < f32 / 1e7

    def test_the_floor_does_not_hide_a_wrong_gradient(self):
        """A sign error must still fail even with the floor active."""

        @jax.custom_jvp
        def f(x):
            return jnp.exp(x)

        @f.defjvp
        def _f_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            return f(x), -jnp.exp(x) * t

        with pytest.raises(AssertionError, match="finite difference"):
            assert_grad_matches_fd(f, 0.5)


class TestTheCheckKnowsItsOwnLimits:
    """Why the 202 remaining sites cannot all be converted mechanically."""

    def test_a_step_function_disagrees_and_that_is_expected(self):
        """The non-parametric SFHs are piecewise-constant by design (there is
        a regression test pinning that). Autodiff reports 0 inside a bin; a
        finite difference that straddles an edge reports a large slope. The
        disagreement is real and uninteresting — which is the documented
        reason to keep a finiteness assertion at such sites."""

        def step(x):
            return jnp.where(x < 1.0, 0.0, 1.0)

        # differentiate exactly at the discontinuity
        with pytest.raises(AssertionError):
            assert_grad_matches_fd(step, 1.0)

    def test_it_refuses_to_judge_where_f_is_not_evaluable(self):
        """A probe that lands outside the domain must report *that*, not
        silently pass or blame the gradient."""

        def only_positive(x):
            return jnp.where(x > 0, jnp.log(jnp.abs(x)), jnp.nan)

        with pytest.raises(AssertionError, match="probe itself was not finite"):
            assert_grad_matches_fd(only_positive, 0.0, eps=1.0)
