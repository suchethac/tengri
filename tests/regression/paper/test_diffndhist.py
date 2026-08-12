# SPDX-License-Identifier: BSD-3-Clause
"""Tests for differentiable N-dimensional triweight histograms.

Vendored from diffsky (Hearin et al.) — tests verify:
1. Basic 1-D and 2-D histogram correctness
2. Weighted histogram correctness
3. Differentiability (jax.grad passes through)
4. JIT compatibility
5. Conservation (total weight ≈ npts for well-covered bins)
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.utils.diffndhist import tw_ndhist, tw_ndhist_weighted
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.regression_paper

jax.config.update("jax_enable_x64", True)


class TestTwNdhist1D:
    """1-D histogram smoke tests."""

    def test_single_point_in_single_bin(self):
        nddata = jnp.array([[5.0]])
        ndsig = jnp.array([[0.5]])
        ndbins_lo = jnp.array([[0.0]])
        ndbins_hi = jnp.array([[10.0]])
        result = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        chex.assert_shape(result, (1,))
        assert jnp.isclose(result[0], 1.0, atol=1e-6)

    def test_points_distribute_across_bins(self):
        nddata = jnp.array([[2.0], [5.0], [8.0]])
        ndsig = jnp.full((3, 1), 0.3)
        ndbins_lo = jnp.array([[0.0], [4.0], [6.0]])
        ndbins_hi = jnp.array([[4.0], [6.0], [10.0]])
        result = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        chex.assert_shape(result, (3,))
        assert jnp.isclose(result[0], 1.0, atol=0.05)
        assert jnp.isclose(result[1], 1.0, atol=0.05)
        assert jnp.isclose(result[2], 1.0, atol=0.05)

    def test_total_weight_conserved(self):
        key = jax.random.PRNGKey(42)
        npts = 100
        nddata = jax.random.uniform(key, shape=(npts, 1), minval=0.0, maxval=10.0)
        ndsig = jnp.full((npts, 1), 0.2)
        nbins = 20
        edges = jnp.linspace(0.0, 10.0, nbins + 1)
        ndbins_lo = edges[:-1].reshape(-1, 1)
        ndbins_hi = edges[1:].reshape(-1, 1)
        result = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        total = jnp.sum(result)
        assert jnp.isclose(total, float(npts), atol=5.0)


class TestTwNdhist2D:
    """2-D histogram tests."""

    def test_2d_single_point(self):
        nddata = jnp.array([[5.0, 5.0]])
        ndsig = jnp.array([[0.5, 0.5]])
        ndbins_lo = jnp.array([[0.0, 0.0]])
        ndbins_hi = jnp.array([[10.0, 10.0]])
        result = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        chex.assert_shape(result, (1,))
        assert jnp.isclose(result[0], 1.0, atol=1e-6)

    def test_2d_separation(self):
        nddata = jnp.array([[1.0, 1.0], [9.0, 9.0]])
        ndsig = jnp.full((2, 2), 0.3)
        ndbins_lo = jnp.array([[0.0, 0.0], [5.0, 5.0]])
        ndbins_hi = jnp.array([[5.0, 5.0], [10.0, 10.0]])
        result = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        assert jnp.isclose(result[0], 1.0, atol=0.05)
        assert jnp.isclose(result[1], 1.0, atol=0.05)


class TestTwNdhistWeighted:
    """Weighted histogram tests."""

    def test_uniform_weights(self):
        nddata = jnp.array([[2.0], [8.0]])
        ndsig = jnp.full((2, 1), 0.3)
        ydata = jnp.array([1.0, 1.0])
        ndbins_lo = jnp.array([[0.0], [5.0]])
        ndbins_hi = jnp.array([[5.0], [10.0]])
        result = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
        counts = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        assert jnp.allclose(result, counts, atol=1e-10)

    def test_weighted_mean(self):
        nddata = jnp.array([[2.0], [2.5], [8.0]])
        ndsig = jnp.full((3, 1), 0.3)
        ydata = jnp.array([10.0, 20.0, 100.0])
        ndbins_lo = jnp.array([[0.0], [5.0]])
        ndbins_hi = jnp.array([[5.0], [10.0]])

        weighted = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
        counts = tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi)
        mean_bin0 = weighted[0] / counts[0]
        mean_bin1 = weighted[1] / counts[1]
        assert 10.0 < float(mean_bin0) < 20.0
        assert jnp.isclose(mean_bin1, 100.0, atol=1.0)


class TestDifferentiability:
    """Verify jax.grad passes through the histogram."""

    def test_grad_through_ndhist(self):
        def loss(data):
            ndsig = jnp.full_like(data, 0.5)
            ndbins_lo = jnp.array([[0.0], [5.0]])
            ndbins_hi = jnp.array([[5.0], [10.0]])
            hist = tw_ndhist(data, ndsig, ndbins_lo, ndbins_hi)
            return jnp.sum(hist**2)

        data = jnp.array([[3.0], [7.0]])
        grads = assert_grad_matches_fd(loss, data)
        chex.assert_equal_shape([grads, data])
        chex.assert_tree_all_finite(grads)

    def test_grad_through_weighted_ndhist(self):
        def loss(data, y):
            ndsig = jnp.full_like(data, 0.5)
            ndbins_lo = jnp.array([[0.0], [5.0]])
            ndbins_hi = jnp.array([[5.0], [10.0]])
            wh = tw_ndhist_weighted(data, ndsig, y, ndbins_lo, ndbins_hi)
            return jnp.sum(wh)

        data = jnp.array([[3.0], [7.0]])
        y = jnp.array([1.0, 2.0])
        grads_data, grads_y = jax.grad(loss, argnums=(0, 1))(data, y)
        chex.assert_tree_all_finite(grads_data)
        chex.assert_tree_all_finite(grads_y)

    def test_grad_through_scatter(self):
        def loss(sig):
            data = jnp.array([[3.0], [7.0]])
            ndbins_lo = jnp.array([[0.0], [5.0]])
            ndbins_hi = jnp.array([[5.0], [10.0]])
            hist = tw_ndhist(data, sig, ndbins_lo, ndbins_hi)
            return jnp.sum(hist**2)

        sig = jnp.array([[0.5], [0.5]])
        grads = assert_grad_matches_fd(loss, sig)
        chex.assert_equal_shape([grads, sig])
        chex.assert_tree_all_finite(grads)


class TestJIT:
    """JIT compilation tests."""

    def test_jit_ndhist(self):
        @jax.jit
        def f(data):
            ndsig = jnp.full_like(data, 0.3)
            lo = jnp.array([[0.0]])
            hi = jnp.array([[10.0]])
            return tw_ndhist(data, ndsig, lo, hi)

        result = f(jnp.array([[5.0]]))
        assert jnp.isclose(result[0], 1.0, atol=1e-6)

    def test_jit_weighted(self):
        @jax.jit
        def f(data, y):
            ndsig = jnp.full_like(data, 0.3)
            lo = jnp.array([[0.0]])
            hi = jnp.array([[10.0]])
            return tw_ndhist_weighted(data, ndsig, y, lo, hi)

        result = f(jnp.array([[5.0]]), jnp.array([3.14]))
        assert jnp.isclose(result[0], 3.14, atol=0.01)
