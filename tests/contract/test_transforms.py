# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parameter transform utilities."""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.utils.transforms import (
    sigmoid,
    to_bounded,
    to_unbounded,
)

pytestmark = pytest.mark.contract


class TestSigmoid:
    """Tests for sigmoid bounded transform."""

    def test_midpoint(self):
        """sigmoid(x0) = (ymin + ymax) / 2."""
        val = sigmoid(0.0, x0=0.0, k=0.1, ymin=-1.0, ymax=1.0)
        assert_allclose(float(val), 0.0, atol=1e-10)

    def test_bounds(self):
        """Output stays within (ymin, ymax)."""
        x = jnp.linspace(-100, 100, 1000)
        y = sigmoid(x, x0=0.0, k=0.1, ymin=-2.0, ymax=3.0)
        assert jnp.all(y > -2.0)
        assert jnp.all(y < 3.0)

    def test_monotone(self):
        """sigmoid is monotonically increasing."""
        x = jnp.linspace(-50, 50, 1000)
        y = sigmoid(x, x0=0.0, k=0.1, ymin=0.0, ymax=1.0)
        assert jnp.all(jnp.diff(y) > 0)


class TestRoundTrip:
    """Tests for bounded <-> unbounded roundtrip."""

    @pytest.mark.parametrize("lo,hi", [(-4.0, 1.5), (0.0, 10.0), (-1.0, 1.0)])
    def test_roundtrip(self, lo, hi):
        """to_unbounded(to_bounded(u)) ≈ u."""
        u = jnp.linspace(-3.0, 3.0, 20)
        bounded = to_bounded(u, lo, hi)
        recovered = to_unbounded(bounded, lo, hi)
        assert_allclose(recovered, u, rtol=1e-5)

    @pytest.mark.parametrize("lo,hi", [(-4.0, 1.5), (0.0, 10.0)])
    def test_inverse_roundtrip(self, lo, hi):
        """to_bounded(to_unbounded(x)) ≈ x."""
        mid = 0.5 * (lo + hi)
        x = jnp.linspace(lo + 0.1, hi - 0.1, 20)
        unbounded = to_unbounded(x, lo, hi)
        recovered = to_bounded(unbounded, lo, hi)
        assert_allclose(recovered, x, rtol=1e-5)
