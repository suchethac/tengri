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
