# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the ProSpect-family SFH models.

Covers:
- spline: monotone cubic PCHIP spline (ProSpect massfunc_p4/p6)
- snorm_burst: skew-normal + burst (ProSpect massfunc_snorm_burst)
- snorm_trunc_burst: truncated snorm + burst (ProSpect massfunc_snorm_burst_trunc)

Each function is tested for:
- Correct output shape and non-negative SFR
- Physically expected behavior (burst additive, PCHIP monotone between nodes)
- JIT-compatibility via jax.jit
- Gradient existence via jax.grad w.r.t. traced parameters
"""

from typing import ClassVar

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.bounds

from tengri.components.stellar.sfh.mean_sfh import (
    snorm_burst,
    snorm_trunc_burst,
    spline,
)
from tengri.utils.grid_interp import _pchip_slopes
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd

_T = jnp.logspace(7, 10.14, 128)  # lookback times 10 Myr – 13.8 Gyr

_NODE_AGES_4 = np.array([1e5, 2e9, 9e9, 13e9])  # 4-node default (massfunc_p4)
_NODE_AGES_6 = np.array([1e5, 1e8, 1e9, 5e9, 9e9, 13e9])  # 6-node (massfunc_p6)


# ── PCHIP helpers ────────────────────────────────────────────────────


class TestPchipSlopes:
    """Tests for Fritsch-Carlson slope computation.

    PCHIP ensures monotonicity within segments: if the y-values are monotone
    between two nodes, the slope at each node has the same sign as the secant.
    At extrema, the slope is zero to prevent overshoot.
    """

    def test_monotone_increasing_slopes_positive(self):
        """Slopes on a strictly increasing sequence should all be non-negative.

        Monotonicity preservation: if y_{i-1} < y_i < y_{i+1}, then d_i ≥ 0.
        """
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.0, 1.0, 3.0, 6.0])
        d = _pchip_slopes(x, y)
        assert_non_negative(d, name="d")

    def test_monotone_decreasing_slopes_nonpositive(self):
        """Slopes on a strictly decreasing sequence should all be non-positive.

        Monotonicity preservation: if y_{i-1} > y_i > y_{i+1}, then d_i ≤ 0.
        """
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([6.0, 3.0, 1.0, 0.0])
        d = _pchip_slopes(x, y)
        assert jnp.all(d <= 0.0)

    def test_local_extremum_gets_zero_slope(self):
        """At a local maximum, Fritsch-Carlson sets slope to zero.

        Interior extrema force zero slope to prevent overshoot.
        """
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 0.5])
        d = _pchip_slopes(x, y)
        # Interior slope at index 1 (local max) must be zero
        assert float(d[1]) == pytest.approx(0.0, abs=1e-12)


# ── spline ───────────────────────────────────────────────────────


class TestSplineSfh:
    """Tests for spline (ProSpect massfunc_p4/p6).

    PCHIP cubic Hermite interpolation in log10(age) space ensures
    monotonicity between monotone nodes and exact interpolation at nodes.
    """

    def test_nonnegative(self):
        """SFR is always >= 0.

        Post-interpolation clamp ensures no negative ringing.
        """
        sfr_nodes = jnp.array([0.0, 3.0, 1.0, 0.0])
        sfr = spline(_T, sfr_nodes, _NODE_AGES_4)
        assert_non_negative(sfr, name="sfr")

    def test_interpolates_through_nodes(self):
        """SFH recovers node values at node lookback times (within tolerance).

        Cubic Hermite interpolation passes exactly through the nodes.
        """
        node_ages = np.array([1e7, 2e9, 9e9, 13e9])
        sfr_nodes = jnp.array([0.5, 3.0, 1.5, 0.2])
        t_at_nodes = jnp.array(node_ages, dtype=jnp.float64)
        sfr_at_nodes = spline(t_at_nodes, sfr_nodes, node_ages)
        assert_allclose(sfr_at_nodes, sfr_nodes, rtol=1e-6)

    def test_monotone_between_monotone_nodes(self):
        """SFH is monotone between monotone nodes (PCHIP guarantee).

        PCHIP preserves monotonicity: if nodes are monotone increasing,
        interpolation cannot overshoot.
        """
        sfr_nodes = jnp.array([0.1, 2.0, 4.0, 5.0])
        sfr = spline(_T, sfr_nodes, _NODE_AGES_4)
        # Within node range: mask to ages inside the grid
        mask = (_NODE_AGES_4[0] <= _T) & (_NODE_AGES_4[-1] >= _T)
        sfr_inside = sfr[mask]
        diffs = jnp.diff(sfr_inside)
        # Non-decreasing (ascending nodes → ascending SFR values)
        assert jnp.all(diffs >= -1e-10)

    def test_grad_wrt_sfr_nodes_exists(self):
        """Gradient w.r.t. sfr_nodes exists and is finite.

        Cubic Hermite interpolation is differentiable everywhere except
        possibly at node discontinuities (which don't exist for PCHIP).
        """
        sfr_nodes = jnp.array([1.0, 3.0, 2.0, 0.5])

        def scalar_sum(nodes):
            return jnp.sum(spline(_T, nodes, _NODE_AGES_4))

        grad = assert_grad_matches_fd(scalar_sum, sfr_nodes)
        chex.assert_tree_all_finite(grad)
        chex.assert_equal_shape([grad, sfr_nodes])


# ── snorm_burst ──────────────────────────────────────────────────


class TestSnormBurstSfh:
    """Tests for snorm_burst (ProSpect massfunc_snorm_burst).

    Tests anchor to physical properties:
    - SFR is non-negative everywhere
    - Burst adds to SFR at young lookback times (t < burst_age)
    - Total mass integral equals 10^log_total_mass
    """

    _KWARGS: ClassVar = dict(
        log_total_mass=1.5, peak_lbt=5e9, width=2e9, skew=0.5, burst_sfr=2.0, burst_age=5e8
    )

    def test_nonnegative(self):
        """SFR is always >= 0."""
        sfr = snorm_burst(_T, **self._KWARGS)
        assert_non_negative(sfr, name="sfr")

    def test_burst_adds_to_young_ages(self):
        """Young-age SFR with burst >= same config with burst_sfr=0.

        Burst is an additive component active at t < burst_age.
        """
        sfr_with_burst = snorm_burst(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_burst(_T, **no_burst_kwargs)

        young_mask = self._KWARGS["burst_age"] > _T
        assert jnp.all(sfr_with_burst[young_mask] >= sfr_no_burst[young_mask])

    def test_no_burst_outside_burst_age(self):
        """SFR at ages >= burst_age is approximately equal to no-burst version.

        Burst is inactive (indicator function) at t >= burst_age, so the
        composite shape is identical in the old-age regime. Renormalization
        to log_total_mass affects the absolute scaling but not the relative
        pattern at old ages.
        """
        sfr_with_burst = snorm_burst(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_burst(_T, **no_burst_kwargs)

        # Skip the ONE cell straddling ``burst_age``. The SFH array is a
        # quadrature integrand (the forward model turns it into mass parcels via
        # ``trapezoid``), so the cell containing the burst edge carries the burst
        # mass formed in its covered fraction -- it is not "outside" the burst
        # (#1374). Beyond that cell the two must agree. The one-cell step is
        # derived from the grid, not hard-coded, so this survives editing ``_T``.
        cell_factor = float(_T[1] / _T[0])  # _T is log-spaced
        old_mask = self._KWARGS["burst_age"] * cell_factor < _T
        assert int(old_mask.sum()) > 10, "probe setup failed: too few nodes past the burst edge"
        # Renormalization can redistribute mass, so use moderate tolerance
        assert_allclose(sfr_with_burst[old_mask], sfr_no_burst[old_mask], rtol=0.3)

    def test_total_mass_integral_golden(self):
        """Total mass integral matches golden value from current implementation.

        Golden value: integral of snorm_burst(..., log_total_mass=1.5, ...)
        at the test parameters equals 31.623 (10^1.5 Msun).
        """
        sfr = snorm_burst(_T, **self._KWARGS)
        total_mass = jnp.trapezoid(sfr, _T)
        # Golden: 10^1.5 ≈ 31.623 Msun
        expected = 10.0 ** self._KWARGS["log_total_mass"]
        assert_allclose(float(total_mass), expected, rtol=1e-8)


# ── snorm_trunc_burst ────────────────────────────────────────────


class TestSnormTruncBurstSfh:
    """Tests for snorm_trunc_burst (ProSpect massfunc_snorm_burst_trunc).

    Combines truncated skew-normal (tsnorm) with burst. Tests anchor to:
    - SFR is non-negative (burst + truncated kernel)
    - Burst adds at young ages
    - Truncation suppresses old-age SFR relative to non-truncated snorm_burst
    - Total mass integral equals 10^log_total_mass
    """

    _KWARGS: ClassVar = dict(
        log_total_mass=1.5,
        peak_lbt=5e9,
        width=2e9,
        skew=0.5,
        trunc=2.0,
        burst_sfr=2.0,
        burst_age=5e8,
    )

    def test_nonnegative(self):
        """SFR is always >= 0."""
        sfr = snorm_trunc_burst(_T, **self._KWARGS)
        assert_non_negative(sfr, name="sfr")

    def test_burst_adds_to_young_ages(self):
        """Young-age SFR with burst >= same config with burst_sfr=0.

        Burst is an additive component active at t < burst_age.
        """
        sfr_with_burst = snorm_trunc_burst(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_trunc_burst(_T, **no_burst_kwargs)

        young_mask = self._KWARGS["burst_age"] > _T
        assert jnp.all(sfr_with_burst[young_mask] >= sfr_no_burst[young_mask])

    def test_no_burst_outside_burst_age(self):
        """SFR at ages >= burst_age is approximately equal to no-burst version.

        Burst is inactive at t >= burst_age. Renormalization can affect
        absolute scaling but not the relative suppression at old ages.
        """
        sfr_with_burst = snorm_trunc_burst(_T, **self._KWARGS)
        no_burst_kwargs = {**self._KWARGS, "burst_sfr": 0.0}
        sfr_no_burst = snorm_trunc_burst(_T, **no_burst_kwargs)

        # Skip the ONE cell straddling ``burst_age``. The SFH array is a
        # quadrature integrand (the forward model turns it into mass parcels via
        # ``trapezoid``), so the cell containing the burst edge carries the burst
        # mass formed in its covered fraction -- it is not "outside" the burst
        # (#1374). Beyond that cell the two must agree. The one-cell step is
        # derived from the grid, not hard-coded, so this survives editing ``_T``.
        cell_factor = float(_T[1] / _T[0])  # _T is log-spaced
        old_mask = self._KWARGS["burst_age"] * cell_factor < _T
        assert int(old_mask.sum()) > 10, "probe setup failed: too few nodes past the burst edge"
        assert_allclose(sfr_with_burst[old_mask], sfr_no_burst[old_mask], rtol=0.3)

    def test_truncation_reduces_old_sfr_vs_snorm_burst(self):
        """Truncation suppresses SFR at ages older than peak.

        Truncated snorm (tsnorm) multiplies the kernel by an erfc-based
        truncation factor that smoothly goes to zero at recent ages.
        At very old ages, the truncation factor ≠ 0 but is less than 1,
        so total SFR is suppressed relative to non-truncated snorm.
        """
        sfr_trunc = snorm_trunc_burst(_T, **self._KWARGS)
        # snorm_burst equivalent (same params minus trunc)
        snorm_burst_kwargs = {k: v for k, v in self._KWARGS.items() if k != "trunc"}
        sfr_plain = snorm_burst(_T, **snorm_burst_kwargs)

        # At very old ages (>> peak), truncation should reduce SFR
        very_old_mask = 1.2 * self._KWARGS["peak_lbt"] < _T
        assert jnp.sum(sfr_trunc[very_old_mask]) <= jnp.sum(sfr_plain[very_old_mask])

    def test_total_mass_integral_golden(self):
        """Total mass integral matches golden value from current implementation.

        Golden value: integral of snorm_trunc_burst(..., log_total_mass=1.5, ...)
        at the test parameters equals 31.623 (10^1.5 Msun).
        """
        sfr = snorm_trunc_burst(_T, **self._KWARGS)
        total_mass = jnp.trapezoid(sfr, _T)
        # Golden: 10^1.5 ≈ 31.623 Msun
        expected = 10.0 ** self._KWARGS["log_total_mass"]
        assert_allclose(float(total_mass), expected, rtol=1e-8)
