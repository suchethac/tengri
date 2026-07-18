# SPDX-License-Identifier: BSD-3-Clause
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.regression_bug
jax.config.update("jax_enable_x64", True)

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
