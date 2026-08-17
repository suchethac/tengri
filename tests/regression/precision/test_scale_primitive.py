# SPDX-License-Identifier: BSD-3-Clause
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.regression_bug

from tengri.utils.scale import apply_log10_scale


def test_f64_matches_naive_product():
    arr = jnp.asarray([1.3e23, 2.4e34, 5.0e28])  # spans 11 dex, like a real SED
    s = -57.6  # log10 of a z~0.01 flux_scale
    got = apply_log10_scale(arr, s)
    want = arr * (10.0**s)
    assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)


def test_pure_f32_stays_finite_and_accurate():
    # In pure f32, arr*10**s would go inf*0 -> nan; the helper must not.
    with jax.enable_x64(False):  # JAX ≥0.9 context manager (was jax.experimental.enable_x64)
        arr = jnp.asarray([1.3e23, 2.4e34, 5.0e28], dtype=jnp.float32)
        s = jnp.asarray(-57.6, dtype=jnp.float32)
        got = np.asarray(apply_log10_scale(arr, s))
    assert np.all(np.isfinite(got))
    # net magnitude ~ 2.4e34 * 10**-57.6 ~ 6e-24, representable in f32
    want = np.asarray([1.3e23, 2.4e34, 5.0e28]) * (10.0**-57.6)
    assert_allclose(got, want, rtol=1e-3)


# --- log10_add: signed base-10 logaddexp for additive log-domain seams -------


def test_log10_add_matches_naive_sum_in_f64():
    """Equal to log10 of the naive sum, across sign and magnitude combinations."""
    from tengri.utils.scale import log10_add

    for a, b in ((43.2, 42.1), (43.2, 43.2), (30.0, 55.0), (-12.0, 3.5)):
        for sa, sb in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
            naive = sa * 10.0**a + sb * 10.0**b
            if naive == 0.0:
                continue  # exact cancellation is covered separately
            got = float(log10_add(a, b, sign_a=sa, sign_b=sb))
            assert_allclose(got, np.log10(abs(naive)), rtol=1e-12)


def test_log10_add_handles_zero_terms():
    """``-inf`` denotes an exactly zero term and must not poison the sum."""
    from tengri.utils.scale import log10_add

    assert float(log10_add(-np.inf, 43.0)) == 43.0
    assert float(log10_add(43.0, -np.inf)) == 43.0
    assert float(log10_add(-np.inf, -np.inf)) == -np.inf
    # exact cancellation -> zero magnitude -> -inf
    assert float(log10_add(43.0, 43.0, sign_a=1.0, sign_b=-1.0)) == -np.inf


def test_log10_add_does_not_report_an_overflowed_term_as_zero():
    """``+inf`` is an overflow upstream, NOT an empty term.

    Both infinities fail ``isfinite``, but they mean opposite things: ``-inf``
    is "no term here", ``+inf`` is "something upstream left the range". Folding
    the latter into the ``-inf`` sentinel would report an overflowed term as
    exactly **zero** — a fail-open on precisely the axis this module exists to
    close, and indistinguishable from a legitimately absent term.
    """
    from tengri.utils.scale import log10_add

    assert float(log10_add(np.inf, 43.0)) == np.inf
    assert float(log10_add(43.0, np.inf)) == np.inf
    assert float(log10_add(np.inf, np.inf)) == np.inf
    # A sign flip does not make an overflow disappear either.
    assert float(log10_add(np.inf, 43.0, sign_a=-1.0)) == np.inf
    # +inf must still beat -inf: one real (overflowed) term plus one empty one.
    assert float(log10_add(np.inf, -np.inf)) == np.inf


def test_log10_add_stays_finite_in_pure_float32():
    """Summing two ~1e43 terms must never materialize them (float32 max 3.4e38)."""
    from tengri.utils.scale import log10_add, pow10

    ref = float(log10_add(43.2, 42.1))
    with jax.enable_x64(False):
        a = jnp.asarray(43.2, dtype=jnp.float32)
        b = jnp.asarray(42.1, dtype=jnp.float32)
        assert a.dtype == jnp.float32  # precondition: genuinely pure float32
        got = float(log10_add(a, b))
        naive = float(pow10(a) + pow10(b))
    assert np.isfinite(got), f"log10_add non-finite in float32: {got}"
    assert_allclose(got, ref, atol=1e-5)
    assert not np.isfinite(naive), "expected the naive linear sum to overflow float32"


def test_log10_add_gradient_is_finite():
    """Gradients stay finite, including where the sum vanishes."""
    from tengri.utils.scale import log10_add

    grad_a = jax.grad(lambda x: log10_add(x, 42.1))
    for x in (43.2, 20.0, 60.0):
        assert np.isfinite(float(grad_a(x))), f"grad non-finite at {x}"
    # exact cancellation takes the where-dummy branch
    assert np.isfinite(float(jax.grad(lambda x: log10_add(x, x, sign_b=-1.0))(43.0)))


def test_apply_log10_scale_of_zeros_is_zero_not_nan():
    r"""Scaling an all-zero array must give zeros, at any scale (#1206).

    ``apply_log10_scale`` normalizes ``arr`` by its peak and folds the peak's
    decades into the exponent. When ``arr`` is entirely zero the peak is zero,
    the where-dummy replaces it with 1.0, and the exponent collapses to the raw
    ``log10_scale`` — which for a dust-luminosity scale (~43 dex) overflows
    float32, so the result is ``0 * inf = nan`` instead of ``0``.

    Perfectly valid input (an SED with no emission) becoming NaN is the same
    fail-open shape this tier exists to remove. In float64 the bug is invisible
    because ``10**43`` is representable there.
    """
    scale = 43.17  # log10 of a dust IR luminosity, the scale that triggers it

    f64 = np.asarray(apply_log10_scale(jnp.zeros(5), scale))
    assert np.all(f64 == 0.0), f"float64 baseline already wrong: {f64}"

    with jax.enable_x64(False):
        zeros32 = jnp.zeros(5, dtype=jnp.float32)
        assert zeros32.dtype == jnp.float32  # precondition: genuinely pure float32
        got = np.asarray(apply_log10_scale(zeros32, jnp.float32(scale)))

    assert not np.any(np.isnan(got)), f"zeros scaled to NaN in float32: {got}"
    assert np.all(got == 0.0), f"zeros must scale to zeros, got {got}"


def test_apply_log10_scale_partially_zero_array_in_float32():
    """A mostly-underflowed array must keep its finite entries finite.

    The all-zero case above is the extreme; the realistic one is a template
    whose far-wing values underflow while the peak survives.
    """
    with jax.enable_x64(False):
        arr = jnp.asarray([0.0, 0.0, 1.0e-20, 3.0e-20], dtype=jnp.float32)
        got = np.asarray(apply_log10_scale(arr, jnp.float32(43.17)))

    assert np.all(np.isfinite(got)), f"non-finite under float32: {got}"
    # 3e-20 * 10**43.17 ~ 4.4e23 — comfortably representable.
    assert got[3] > got[2] > 0.0, f"ordering lost: {got}"
    assert got[0] == 0.0 and got[1] == 0.0
