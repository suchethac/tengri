# SPDX-License-Identifier: BSD-3-Clause
r"""The float32 VJP of ``loglog_integral`` is finite and accurate at SED cotangents (#2767).

On a segment where the log-log slope is near :math:`-1` the argument of :math:`{\rm expm1}(u)/u`
is small (:math:`u = a + \ln(y_1/y_0) \sim 10^{-3}`). Differentiating the quotient forms
:math:`{\rm cot}\; x\, y\, a / u^2` before the cancellation: with the :math:`\sim 10^{14}`
cotangent of the summed kubota_done SED and :math:`x \sim 5\times10^{18}` Hz, that product passes
the float32 maximum and the gradient is NaN. Below :math:`|u| = 0.05` the integral uses the Taylor
series of the quotient, which has no :math:`1/u` in its derivative.

The gradient must be finite at that cotangent (for un-normalized ``y``, here scaled to 1e10) and
float32 must agree with float64.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import loglog_integral

pytestmark = pytest.mark.regression_bug

_COTANGENT = 2.07e14  # the cotangent that reached the corona normalization in #2767
_Y_SCALE = 1.0e10  # un-normalized y: the VJP overflows the float32 maximum without the series


def _grid(dtype, y_scale=1.0):
    x = jnp.asarray(np.geomspace(2.418e13, 2.418e21, 8000), dtype=dtype)
    y = jnp.asarray(y_scale * (np.asarray(x, np.float64) / 2.418e13) ** -0.9995, dtype=dtype)
    return x, y


def test_float32_vjp_is_finite_at_a_sed_gradient_cotangent():
    x, y = _grid(jnp.float32, _Y_SCALE)
    _, vjp = jax.vjp(loglog_integral, x, y)
    grad_y = np.asarray(vjp(jnp.float32(_COTANGENT))[1])
    assert np.all(np.isfinite(grad_y)), f"{np.sum(~np.isfinite(grad_y))} non-finite entries"
    # d(integral)/dy is positive for positive x and y, so a positive cotangent gives a positive,
    # non-collapsed gradient at every node.
    assert np.all(grad_y > 0.0), f"{np.sum(grad_y <= 0.0)} nodes with a zero or negative gradient"


def test_float32_vjp_matches_float64():
    x32, y32 = _grid(jnp.float32)
    g32 = np.asarray(jax.vjp(loglog_integral, x32, y32)[1](jnp.float32(1.0))[1], np.float64)
    with jax.enable_x64(True):
        x64, y64 = _grid(jnp.float64)
        g64 = np.asarray(jax.vjp(loglog_integral, x64, y64)[1](jnp.float64(1.0))[1])
    np.testing.assert_allclose(g32, g64, rtol=2e-3, atol=1e-6 * np.abs(g64).max())
