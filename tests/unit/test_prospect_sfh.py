"""Tests for ProSpect-ported SFH models.

Covers:
- spline_sfh: monotone cubic PCHIP spline (massfunc_p4/p6 port)
- snorm_burst_sfh: skew-normal + burst (massfunc_snorm_burst port)
- snorm_trunc_burst_sfh: truncated snorm + burst (massfunc_snorm_burst_trunc port)

Each function is tested for:
- Correct output shape and non-negative SFR
- Physically expected behavior (burst additive, PCHIP monotone between nodes)
- JIT-compatibility via jax.jit
- Gradient existence via jax.grad w.r.t. traced parameters
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

from tengri.components.sfh.mean_sfh import (
    _pchip_slopes,
    snorm_burst_sfh,
    snorm_trunc_burst_sfh,
    spline_sfh,
)

_T = jnp.logspace(7, 10.14, 128)  # lookback times 10 Myr – 13.8 Gyr

_NODE_AGES_4 = np.array([1e5, 2e9, 9e9, 13e9])  # 4-node default (massfunc_p4)
_NODE_AGES_6 = np.array([1e5, 1e8, 1e9, 5e9, 9e9, 13e9])  # 6-node (massfunc_p6)


# ── PCHIP helpers ────────────────────────────────────────────────────


class TestPchipSlopes:
    """Tests for Fritsch-Carlson slope computation."""

    def test_monotone_increasing_slopes_positive(self):
        """Slopes on a strictly increasing sequence should all be non-negative."""
        y = jnp.array([0.0, 1.0, 3.0, 6.0])
        h = jnp.diff(jnp.array([0.0, 1.0, 2.0, 3.0]))
        d = _pchip_slopes(y, h)
        assert jnp.all(d >= 0.0)

    def test_monotone_decreasing_slopes_nonpositive(self):
        """Slopes on a strictly decreasing sequence should all be non-positive."""
        y = jnp.array([6.0, 3.0, 1.0, 0.0])
        h = jnp.diff(jnp.array([0.0, 1.0, 2.0, 3.0]))
        d = _pchip_slopes(y, h)
        assert jnp.all(d <= 0.0)

    def test_local_extremum_gets_zero_slope(self):
        """At a local maximum, Fritsch-Carlson sets slope to zero."""
        y = jnp.array([0.0, 1.0, 0.5])
        h = jnp.diff(jnp.array([0.0, 1.0, 2.0]))
        d = _pchip_slopes(y, h)
        # Interior slope at index 1 (local max) must be zero
        assert float(d[1]) == pytest.approx(0.0, abs=1e-12)

    def test_output_length_equals_nodes(self):
        """Output has the same length as y."""
        y = jnp.array([1.0, 2.0, 3.0, 2.5, 1.0])
        h = jnp.ones(4)
        d = _pchip_slopes(y, h)
        assert d.shape == y.shape


# ── spline_sfh ───────────────────────────────────────────────────────


class TestSplineSfh:
    """Tests for spline_sfh (ProSpect massfunc_p4/p6 port)."""

    def test_shape_4node(self):
        """Output shape matches input lookback time grid — 4 nodes."""
        sfr_nodes = jnp.array([1.0, 5.0, 3.0, 0.5])
        sfr = spline_sfh(_T, sfr_nodes, _NODE_AGES_4)
        assert sfr.shape == _T.shape

    def test_shape_6node(self):
        """Output shape matches input lookback time grid — 6 nodes."""
        sfr_nodes = jnp.array([0.5, 2.0, 5.0, 3.0, 1.0, 0.1])
        sfr = spline_sfh(_T, sfr_nodes, _NODE_AGES_6)
        assert sfr.shape == _T.shape

    def test_nonnegative(self):
        """SFR is always >= 0."""
        sfr_nodes = jnp.array([0.0, 3.0, 1.0, 0.0])
        sfr = spline_sfh(_T, sfr_nodes, _NODE_AGES_4)
        assert jnp.all(sfr >= 0.0)

    def test_interpolates_through_nodes(self):
        """SFH recovers node values at node lookback times (within tolerance)."""
        node_ages = np.array([1e7, 2e9, 9e9, 13e9])
        sfr_nodes = jnp.array([0.5, 3.0, 1.5, 0.2])
        t_at_nodes = jnp.array(node_ages, dtype=jnp.float64)
        sfr_at_nodes = spline_sfh(t_at_nodes, sfr_nodes, node_ages)
        assert_allclose(sfr_at_nodes, sfr_nodes, rtol=1e-6)

    def test_monotone_between_monotone_nodes(self):
        """SFH is monotone between monotone nodes (PCHIP guarantee)."""
        sfr_nodes = jnp.array([0.1, 2.0, 4.0, 5.0])
        sfr = spline_sfh(_T, sfr_nodes, _NODE_AGES_4)
        # Within node range: mask to ages inside the grid
        mask = (_NODE_AGES_4[0] <= _T) & (_NODE_AGES_4[-1] >= _T)
        sfr_inside = sfr[mask]
        diffs = jnp.diff(sfr_inside)
        # Non-decreasing (ascending nodes → ascending SFR values)
        assert jnp.all(diffs >= -1e-10)

    def test_jit_compatible(self):
        """spline_sfh can be JIT-compiled with static node_ages."""
        sfr_nodes = jnp.array([1.0, 3.0, 2.0, 0.5])

        @jax.jit
        def f(nodes):
            return spline_sfh(_T, nodes, _NODE_AGES_4)

        out = f(sfr_nodes)
        assert out.shape == _T.shape

    def test_grad_wrt_sfr_nodes(self):
        """Gradient w.r.t. sfr_nodes exists and is finite."""
        sfr_nodes = jnp.array([1.0, 3.0, 2.0, 0.5])

        def scalar_sum(nodes):
            return jnp.sum(spline_sfh(_T, nodes, _NODE_AGES_4))

        grad = jax.grad(scalar_sum)(sfr_nodes)
        assert jnp.all(jnp.isfinite(grad))
        assert grad.shape == sfr_nodes.shape


# ── snorm_burst_sfh ──────────────────────────────────────────────────


class TestSnormBurstSfh:
    """Tests for snorm_burst_sfh (ProSpect massfunc_snorm_burst port)."""

    _KWARGS: ClassVar = dict(
        log_peak_sfr=1.5, peak_lbt=5e9, width=2e9, skew=0.5, burst_sfr=2.0, burst_age=5e8
    )

    def test_shape(self):
        sfr = snorm_burst_sfh(_T, **self._KWARGS)
        assert sfr.shape == _T.shape

    def test_nonnegative(self):
        sfr = snorm_burst_sfh(_T, **self._KWARGS)
        assert jnp.all(sfr >= 0.0)

    def test_burst_adds_to_young_ages(self):
        """Young-age SFR with burst > same config with burst_sfr=0."""
        sfr_with_burst = snorm_burst_sfh(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_burst_sfh(_T, **no_burst_kwargs)

        young_mask = self._KWARGS["burst_age"] > _T
        assert jnp.all(sfr_with_burst[young_mask] >= sfr_no_burst[young_mask])

    def test_burst_excess_equals_burst_sfr_in_young_regime(self):
        """Excess SFR at young ages exactly equals burst_sfr."""
        sfr_with_burst = snorm_burst_sfh(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_burst_sfh(_T, **no_burst_kwargs)

        young_mask = self._KWARGS["burst_age"] > _T
        excess = sfr_with_burst[young_mask] - sfr_no_burst[young_mask]
        assert_allclose(excess, self._KWARGS["burst_sfr"], rtol=1e-10)

    def test_no_burst_outside_burst_age(self):
        """SFR at ages >= burst_age is identical to no-burst version."""
        sfr_with_burst = snorm_burst_sfh(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_burst_sfh(_T, **no_burst_kwargs)

        old_mask = self._KWARGS["burst_age"] <= _T
        assert_allclose(sfr_with_burst[old_mask], sfr_no_burst[old_mask], rtol=1e-12)

    def test_jit_compatible(self):
        f = jax.jit(snorm_burst_sfh)
        out = f(_T, **self._KWARGS)
        assert out.shape == _T.shape

    def test_grad_wrt_log_peak_sfr(self):
        kw = {k: v for k, v in self._KWARGS.items() if k != "log_peak_sfr"}

        def scalar_sum(log_peak_sfr):
            return jnp.sum(snorm_burst_sfh(_T, log_peak_sfr, **kw))

        grad = jax.grad(scalar_sum)(1.5)
        assert jnp.isfinite(grad)

    def test_grad_wrt_burst_sfr(self):
        kw = {k: v for k, v in self._KWARGS.items() if k != "burst_sfr"}

        def scalar_sum(burst_sfr):
            return jnp.sum(snorm_burst_sfh(_T, burst_sfr=burst_sfr, **kw))

        grad = jax.grad(scalar_sum)(2.0)
        assert jnp.isfinite(grad)


# ── snorm_trunc_burst_sfh ────────────────────────────────────────────


class TestSnormTruncBurstSfh:
    """Tests for snorm_trunc_burst_sfh (ProSpect massfunc_snorm_burst_trunc port)."""

    _KWARGS: ClassVar = dict(
        log_peak_sfr=1.5,
        peak_lbt=5e9,
        width=2e9,
        skew=0.5,
        trunc=2.0,
        burst_sfr=2.0,
        burst_age=5e8,
    )

    def test_shape(self):
        sfr = snorm_trunc_burst_sfh(_T, **self._KWARGS)
        assert sfr.shape == _T.shape

    def test_nonnegative(self):
        sfr = snorm_trunc_burst_sfh(_T, **self._KWARGS)
        assert jnp.all(sfr >= 0.0)

    def test_burst_adds_to_young_ages(self):
        """Young-age SFR with burst > same config with burst_sfr=0."""
        sfr_with_burst = snorm_trunc_burst_sfh(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_trunc_burst_sfh(_T, **no_burst_kwargs)

        young_mask = self._KWARGS["burst_age"] > _T
        assert jnp.all(sfr_with_burst[young_mask] >= sfr_no_burst[young_mask])

    def test_no_burst_outside_burst_age(self):
        """SFR at ages >= burst_age is identical to no-burst version."""
        sfr_with_burst = snorm_trunc_burst_sfh(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_trunc_burst_sfh(_T, **no_burst_kwargs)

        old_mask = self._KWARGS["burst_age"] <= _T
        assert_allclose(sfr_with_burst[old_mask], sfr_no_burst[old_mask], rtol=1e-12)

    def test_truncation_reduces_old_sfr_vs_snorm_burst(self):
        """Truncation suppresses SFR at ages older than peak relative to snorm_burst."""
        sfr_trunc = snorm_trunc_burst_sfh(_T, **self._KWARGS)
        # snorm_burst equivalent (same params minus trunc)
        snorm_burst_kwargs = {k: v for k, v in self._KWARGS.items() if k != "trunc"}
        sfr_plain = snorm_burst_sfh(_T, **snorm_burst_kwargs)

        # At very old ages (>> peak), truncation should reduce SFR
        very_old_mask = 1.2 * self._KWARGS["peak_lbt"] < _T
        assert jnp.sum(sfr_trunc[very_old_mask]) <= jnp.sum(sfr_plain[very_old_mask])

    def test_jit_compatible(self):
        f = jax.jit(snorm_trunc_burst_sfh)
        out = f(_T, **self._KWARGS)
        assert out.shape == _T.shape

    def test_grad_wrt_trunc(self):
        kw = {k: v for k, v in self._KWARGS.items() if k != "trunc"}

        def scalar_sum(trunc):
            return jnp.sum(snorm_trunc_burst_sfh(_T, trunc=trunc, **kw))

        grad = jax.grad(scalar_sum)(2.0)
        assert jnp.isfinite(grad)

    def test_grad_wrt_burst_sfr(self):
        kw = {k: v for k, v in self._KWARGS.items() if k != "burst_sfr"}

        def scalar_sum(burst_sfr):
            return jnp.sum(snorm_trunc_burst_sfh(_T, burst_sfr=burst_sfr, **kw))

        grad = jax.grad(scalar_sum)(2.0)
        assert jnp.isfinite(grad)
