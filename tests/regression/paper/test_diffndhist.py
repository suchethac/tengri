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


_BINS_LO = jnp.array([[0.0], [5.0]])
_BINS_HI = jnp.array([[5.0], [10.0]])
_SIGMA = 0.5

# The triweight kernel has COMPACT support of exactly +/-3 sigma, so where a
# point sits relative to a bin edge decides whether it has a derivative at
# all.  These two lie 0.6 and 1.8 sigma outside the edge at 5.0 — inside the
# support, and deliberately asymmetric about it: the symmetric pair
# [[4.5], [5.5]] balances the two bins and lands on a stationary point of
# sum(hist**2), where every gradient is zero for reasons that have nothing to
# do with differentiability.
_LIVE_DATA = jnp.array([[4.7], [5.9]])

# 4 sigma from the nearest edge: outside the kernel support entirely.
_DEAD_DATA = jnp.array([[3.0], [7.0]])


class TestDifferentiability:
    """jax.grad must carry real signal through the histogram, not finite zeros.

    Every test in this class used to evaluate at ``[[3.0], [7.0]]`` — both
    points 4 sigma from the nearest bin edge, so outside the triweight's
    compact support, where the true derivative is exactly 0.0.  The
    assertions were ``isfinite``, which zero satisfies.  Wrapping the
    histogram in ``jax.lax.stop_gradient`` would not have failed any of them.
    """

    def test_grad_through_ndhist(self):
        def loss(data):
            hist = tw_ndhist(data, jnp.full_like(data, _SIGMA), _BINS_LO, _BINS_HI)
            return jnp.sum(hist**2)

        grads = assert_grad_matches_fd(loss, _LIVE_DATA)
        chex.assert_equal_shape([grads, _LIVE_DATA])
        chex.assert_tree_all_finite(grads)
        for i, g in enumerate(grads.ravel()):
            assert float(g) != 0.0, f"data[{i}]: no gradient reaches the histogram"

    def test_grad_through_weighted_ndhist(self):
        """Uses sum-of-squares: a bare sum is position-invariant by construction.

        ``sum(tw_ndhist_weighted(...))`` equals ``sum(y)`` for any interior
        data — see TestWeightConservation — so the old ``jnp.sum(wh)`` loss
        had ``d/d(data) == 0`` identically, for every input, independent of
        where the points sat.  Squaring makes the loss depend on how the
        weight is *split* between bins, which is the thing being smoothed.
        """

        def loss(data, y):
            wh = tw_ndhist_weighted(data, jnp.full_like(data, _SIGMA), y, _BINS_LO, _BINS_HI)
            return jnp.sum(wh**2)

        y = jnp.array([1.0, 2.0])
        grads_data, grads_y = jax.grad(loss, argnums=(0, 1))(_LIVE_DATA, y)
        chex.assert_tree_all_finite(grads_data)
        chex.assert_tree_all_finite(grads_y)
        for i, g in enumerate(grads_data.ravel()):
            assert float(g) != 0.0, f"data[{i}]: weighted histogram is detached"
        for i, g in enumerate(grads_y.ravel()):
            assert float(g) != 0.0, f"y[{i}]: weights are detached"

    def test_grad_through_scatter(self):
        def loss(sig):
            hist = tw_ndhist(_LIVE_DATA, sig, _BINS_LO, _BINS_HI)
            return jnp.sum(hist**2)

        sig = jnp.full_like(_LIVE_DATA, _SIGMA)
        grads = assert_grad_matches_fd(loss, sig)
        chex.assert_equal_shape([grads, sig])
        chex.assert_tree_all_finite(grads)
        for i, g in enumerate(grads.ravel()):
            assert float(g) != 0.0, f"sigma[{i}]: scatter has no gradient"

    @pytest.mark.parametrize("n_sigma", [3.0, 4.0])
    def test_gradient_vanishes_outside_the_kernel_support(self, n_sigma):
        """Beyond 3 sigma the derivative is exactly zero — the trap, pinned.

        This is correct behaviour for a compactly supported kernel, not a
        bug, but it is why the three tests above were silently vacuous: the
        value stays finite the whole way out, so only a non-zero check can
        tell "differentiable" from "far enough away not to matter".
        """
        x = 5.0 - n_sigma * _SIGMA
        data = jnp.array([[float(x)], [7.0]])

        def loss(d):
            hist = tw_ndhist(d, jnp.full_like(d, _SIGMA), _BINS_LO, _BINS_HI)
            return jnp.sum(hist**2)

        g = float(jax.grad(loss)(data).ravel()[0])
        assert abs(g) <= 1e-15, f"{n_sigma} sigma out: expected no gradient, got {g:.3e}"

    def test_the_old_fixture_had_no_gradient_at_all(self):
        """The exact point the three tests above used to evaluate at.

        Both members of ``[[3.0], [7.0]]`` sit 4 sigma from the edge at 5.0.
        Every gradient there is identically zero — w.r.t. data and w.r.t.
        scatter — so ``isfinite`` was satisfied by a number carrying no
        information.  Kept as the standing proof that the fixture move was
        necessary rather than cosmetic.
        """

        def data_loss(d):
            hist = tw_ndhist(d, jnp.full_like(d, _SIGMA), _BINS_LO, _BINS_HI)
            return jnp.sum(hist**2)

        def sig_loss(s):
            return jnp.sum(tw_ndhist(_DEAD_DATA, s, _BINS_LO, _BINS_HI) ** 2)

        assert not jnp.any(jax.grad(data_loss)(_DEAD_DATA))
        assert not jnp.any(jax.grad(sig_loss)(jnp.full_like(_DEAD_DATA, _SIGMA)))


class TestWeightConservation:
    """The weighted histogram must preserve total weight where it can."""

    @pytest.mark.parametrize("x", [2.0, 3.0, 5.0, 7.0, 8.0])
    def test_total_weight_is_conserved_in_the_interior(self, x):
        """Smoothing redistributes weight between bins; it must not create it."""
        data = jnp.array([[float(x)], [7.0]])
        y = jnp.array([1.0, 2.0])
        wh = tw_ndhist_weighted(data, jnp.full_like(data, _SIGMA), y, _BINS_LO, _BINS_HI)
        assert float(jnp.sum(wh)) == pytest.approx(float(jnp.sum(y)), rel=1e-12)

    @pytest.mark.parametrize(
        ("x", "expected"), [(0.0, 2.5), (10.0, 2.5), (0.5, 2.82670325), (9.5, 2.82670325)]
    )
    def test_weight_leaks_off_the_ends_of_the_binned_domain(self, x, expected):
        """Sitting on the outer edge loses exactly half of that point's weight.

        The kernel spills past the first/last bin and that mass is dropped.
        Worth pinning: a catalog whose extreme objects sit at the domain edge
        is silently down-weighted, and the total still looks plausible.
        """
        data = jnp.array([[float(x)], [7.0]])
        y = jnp.array([1.0, 2.0])
        wh = tw_ndhist_weighted(data, jnp.full_like(data, _SIGMA), y, _BINS_LO, _BINS_HI)
        assert float(jnp.sum(wh)) == pytest.approx(expected, rel=1e-6)
        assert float(jnp.sum(wh)) < float(jnp.sum(y))


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
