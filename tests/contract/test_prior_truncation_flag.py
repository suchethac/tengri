# SPDX-License-Identifier: BSD-3-Clause
"""Test truncation flag logic for bounded distributions.

Addresses issue #2233: truncation flag must be derived from the distribution's
natural support, not from CDF values that underflow beyond ~8 sigma.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.parameters.priors import Gaussian, Laplace, LogNormal, StudentT

pytestmark = pytest.mark.contract


def test_default_lognormal_is_not_truncated_and_is_exactly_affine():
    """Default LogNormal(mu, sigma, lo=0.0) is not truncated.

    The default lo=0.0 equals the natural support lower bound, so it should
    not trigger truncated-mode in unstandardize. Confirm that unstandardize
    is exactly the affine map exp(mu + sigma * xi) with no inverse-CDF clipping.
    """
    dist = LogNormal(1.2, 0.4)
    assert dist._truncated is False

    xi_vals = jnp.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    expected = jnp.exp(1.2 + 0.4 * xi_vals)
    actual = dist.unstandardize(xi_vals)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dist",
    [
        pytest.param(
            Gaussian(mu=3.6, sigma=0.3, lo=0.0),
            id="gaussian-deep-tail-bound",
        ),
        pytest.param(
            LogNormal(mu=5.0, sigma=0.3, lo=1e-3),
            id="lognormal-deep-tail-bound",
        ),
        pytest.param(
            Laplace(mu=0.0, b=1.0, lo=-800.0),
            id="laplace-deep-tail-bound",
        ),
        pytest.param(
            StudentT(mu=0.0, sigma=1.0, df=3.0, lo=-1e6),
            id="studentt-deep-tail-bound",
        ),
    ],
)
def test_deep_tail_bound_is_truncated(dist):
    """Deep tail bounds trigger truncated mode via structural rule.

    Bounds far in the tail (>8.3 sigma) would cause CDF underflow to 0.0/1.0,
    but the structural rule (bound inside natural support) correctly identifies
    them as truncations, so unstandardize stays inside [lo, hi] (inverse-CDF
    pushforward, not clipping).
    """
    assert dist._truncated is True

    # Sample latent values spanning the range
    for xi in [-60.0, -15.0, 15.0, 60.0]:
        theta = float(dist.unstandardize(jnp.asarray(xi)))
        lo, hi = dist.bounds
        assert lo <= theta <= hi, (
            f"{type(dist).__name__}.unstandardize({xi}) = {theta} is outside [{lo}, {hi}]"
        )
        assert math.isfinite(theta), (
            f"{type(dist).__name__}.unstandardize({xi}) = {theta} is not finite"
        )


def test_natural_support_bound_is_not_truncated():
    """Bounds equal to the natural support are not truncations.

    LogNormal has natural support (0, inf), so lo=0 is not a truncation.
    All others have natural support (-inf, inf), so no default bounds
    trigger truncation.
    """
    assert LogNormal(0.0, 0.5, lo=0.0)._truncated is False
    assert Gaussian(0.0, 1.0)._truncated is False
    assert Laplace(0.0, 1.0)._truncated is False
    assert StudentT(0.0, 1.0, 3.0)._truncated is False


def test_finite_bound_inside_support_is_truncated():
    """Finite bounds strictly inside natural support trigger truncation."""
    assert LogNormal(0.0, 0.5, lo=0.5)._truncated is True
    assert Gaussian(0.0, 1.0, hi=2.0)._truncated is True


def test_natural_support_declared():
    """Each distribution class declares its natural support correctly."""
    assert (0.0, math.inf) == LogNormal._NATURAL_SUPPORT
    assert (float("-inf"), float("inf")) == Gaussian._NATURAL_SUPPORT
    assert (float("-inf"), float("inf")) == Laplace._NATURAL_SUPPORT
    assert (float("-inf"), float("inf")) == StudentT._NATURAL_SUPPORT
