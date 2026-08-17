# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the pure quantile-only dense_basis SFH.

``dense_basis_pure`` interpolates its cumulative-mass quantiles with the same
george-faithful GP as :func:`dense_basis`, matching ``interpolator='gp_george'``
— the default in Kartheik Iyer's ``dense_basis`` package. It previously used
PCHIP, which Iyer et al. (2019) explicitly set aside in favor of GP regression.
Contract tests for the shared PCHIP primitive now live in
``tests/contract/test_pchip_interp_1d.py``.

References
----------
- Iyer et al. (2019), ApJ 879, 116, Sec. 2.1. arXiv:1901.02877.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.bounds

from tengri.components.stellar.sfh.dense_basis import (
    _build_quantile_points_pure,
    dense_basis_pure,
)
from tests._bounds import assert_non_negative

# ── _build_quantile_points_pure ───────────────────────────────────


class TestBuildQuantilePointsPure:
    def test_first_point_is_zero(self):
        """First point is (0, 0)."""
        tx_fracs = jnp.array([0.4, 0.7])
        time_q, mass_q, _yerr = _build_quantile_points_pure(tx_fracs, 2)
        assert float(time_q[0]) == pytest.approx(0.0)
        assert float(mass_q[0]) == pytest.approx(0.0)

    def test_big_bang_constraint(self):
        """Second time point is 0.01 (Big Bang constraint)."""
        tx_fracs = jnp.array([0.4, 0.7])
        time_q, mass_q, _yerr = _build_quantile_points_pure(tx_fracs, 2)
        assert float(time_q[1]) == pytest.approx(0.01)
        assert float(mass_q[1]) == pytest.approx(0.0)

    def test_last_point_is_one(self):
        """Last point is (1, 1)."""
        tx_fracs = jnp.array([0.3, 0.5, 0.7])
        time_q, mass_q, _yerr = _build_quantile_points_pure(tx_fracs, 3)
        assert float(time_q[-1]) == pytest.approx(1.0)
        assert float(mass_q[-1]) == pytest.approx(1.0)

    def test_mass_quantiles_evenly_spaced(self):
        """mass_q interior values are linspace(0, 1, n_param+2)."""
        n_param = 4
        tx_fracs = jnp.linspace(0.2, 0.8, n_param)
        _time_q, mass_q, _yerr = _build_quantile_points_pure(tx_fracs, n_param)
        # Structure: [0, 0 (big bang), linspace(0,1,n+2)[1:]]
        # mass_q[2:] should equal linspace(0,1,n_param+2)[1:]
        expected_mass = jnp.linspace(0.0, 1.0, n_param + 2)
        assert_allclose(mass_q[2:], expected_mass[1:], atol=1e-6)


# ── dense_basis_pure ──────────────────────────────────────────


class TestDenseBasisPureSfh:
    @staticmethod
    def _default_age_grid():
        return jnp.linspace(1e6, 13e9, 200)

    def test_non_negative(self):
        """SFR is non-negative everywhere and finite."""
        age_yr = self._default_age_grid()
        sfr = dense_basis_pure(age_yr, log_total_mass=10.0, tx_frac_0=0.4, tx_frac_1=0.7)
        assert_non_negative(sfr, name="sfr")
        chex.assert_tree_all_finite(sfr)

    def test_total_mass_scaling(self):
        """Doubling log_total_mass by 1 dex increases integrated mass ~10x."""
        age_yr = self._default_age_grid()
        sfr_lo = dense_basis_pure(age_yr, log_total_mass=9.0, tx_frac_0=0.5)
        sfr_hi = dense_basis_pure(age_yr, log_total_mass=10.0, tx_frac_0=0.5)
        ratio = float(jnp.sum(sfr_hi)) / max(float(jnp.sum(sfr_lo)), 1e-30)
        assert 8.0 < ratio < 12.0, f"Expected ~10x mass scaling, got {ratio:.2f}"

    def test_no_tx_kwargs_raises(self):
        """Calling without any tx_frac_* raises ValueError."""
        age_yr = self._default_age_grid()
        with pytest.raises(ValueError, match="requires at least one"):
            dense_basis_pure(age_yr, log_total_mass=10.0)

    def test_missing_tx_frac_key_raises(self):
        """Passing tx_frac_1 without tx_frac_0 raises ValueError."""
        age_yr = self._default_age_grid()
        with pytest.raises(ValueError, match="Missing required parameter"):
            dense_basis_pure(age_yr, log_total_mass=10.0, tx_frac_1=0.5)

    def test_three_quantiles(self):
        """Works with three tx_frac parameters."""
        age_yr = self._default_age_grid()
        sfr = dense_basis_pure(
            age_yr, log_total_mass=10.0, tx_frac_0=0.2, tx_frac_1=0.5, tx_frac_2=0.8
        )
        chex.assert_equal_shape([sfr, age_yr])
        assert_non_negative(sfr, name="sfr")

    def test_age_universe_yr_custom(self):
        """Custom age_universe_yr changes the timescale."""
        age_yr = self._default_age_grid()
        sfr_default = dense_basis_pure(age_yr, log_total_mass=10.0, tx_frac_0=0.5)
        sfr_custom = dense_basis_pure(
            age_yr, log_total_mass=10.0, age_universe_yr=10e9, tx_frac_0=0.5
        )
        # Different universes → different SFH profiles
        assert not jnp.allclose(sfr_default, sfr_custom)

    def test_jit_parity_vs_eager(self):
        """JIT output matches eager evaluation (JAX correctness)."""
        age_yr = self._default_age_grid()
        sfr_eager = dense_basis_pure(age_yr, log_total_mass=10.0, tx_frac_0=0.5)
        sfr_jit = jax.jit(dense_basis_pure)(age_yr, log_total_mass=10.0, tx_frac_0=0.5)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)
