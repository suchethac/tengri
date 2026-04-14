"""Tests for PCHIP interpolation and pure quantile SFH in dense_basis.py.

Covers the uncovered branches: pchip_interpolate (lines 173-214),
_build_quantile_points_pure (481-506), and dense_basis_pure_sfh (547-582).
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

from tengri.models.sfh.dense_basis import (
    _build_quantile_points_pure,
    dense_basis_pure_sfh,
    pchip_interpolate,
)

# ---------------------------------------------------------------------------
# pchip_interpolate
# ---------------------------------------------------------------------------


class TestPchipInterpolate:
    def test_exact_at_knots(self):
        """PCHIP passes exactly through the training points."""
        x = jnp.linspace(0.0, 1.0, 5)
        y = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_pred = pchip_interpolate(x, y, x)
        assert_allclose(y_pred, y, atol=1e-6)

    def test_output_shape(self):
        """Output has the same length as x_eval."""
        x_train = jnp.linspace(0.0, 1.0, 6)
        y_train = x_train**2
        x_eval = jnp.linspace(0.1, 0.9, 20)
        out = pchip_interpolate(x_train, y_train, x_eval)
        assert out.shape == (20,)

    def test_monotone_on_monotone_input(self):
        """PCHIP preserves monotonicity on a strictly increasing dataset."""
        x = jnp.linspace(0.0, 1.0, 10)
        y = x**3  # monotone increasing
        x_eval = jnp.linspace(0.05, 0.95, 50)
        out = pchip_interpolate(x, y, x_eval)
        diffs = jnp.diff(out)
        assert jnp.all(diffs >= -1e-10), "PCHIP output is not monotone on monotone input"

    def test_linear_data_exact(self):
        """PCHIP recovers a linear function exactly."""
        x = jnp.linspace(0.0, 1.0, 5)
        y = 2.0 * x + 1.0
        x_eval = jnp.linspace(0.1, 0.9, 30)
        out = pchip_interpolate(x, y, x_eval)
        expected = 2.0 * x_eval + 1.0
        assert_allclose(out, expected, atol=1e-6)

    def test_clamped_at_boundaries(self):
        """x_eval outside training range is clamped (clips t to [0,1])."""
        x = jnp.linspace(0.0, 1.0, 5)
        y = x
        # Evaluate at points beyond both ends
        x_eval = jnp.array([-0.5, 1.5])
        out = pchip_interpolate(x, y, x_eval)
        # Should not raise and should return finite values
        assert jnp.all(jnp.isfinite(out))

    def test_finite_output(self):
        """All output values are finite for well-conditioned inputs."""
        x = jnp.linspace(0.0, 1.0, 8)
        y = jnp.sin(jnp.pi * x)
        x_eval = jnp.linspace(0.0, 1.0, 100)
        out = pchip_interpolate(x, y, x_eval)
        assert jnp.all(jnp.isfinite(out))

    def test_constant_data(self):
        """Constant training data → constant output."""
        x = jnp.linspace(0.0, 1.0, 6)
        y = jnp.ones(6) * 3.14
        x_eval = jnp.linspace(0.1, 0.9, 20)
        out = pchip_interpolate(x, y, x_eval)
        assert_allclose(out, 3.14, atol=1e-5)

    def test_gradients_finite(self):
        """Gradients through pchip_interpolate are finite."""
        x_train = jnp.linspace(0.0, 1.0, 5)
        x_eval = jnp.linspace(0.1, 0.9, 10)

        def loss(y_train):
            return jnp.sum(pchip_interpolate(x_train, y_train, x_eval) ** 2)

        g = jax.grad(loss)(jnp.linspace(0.0, 1.0, 5))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# _build_quantile_points_pure
# ---------------------------------------------------------------------------


class TestBuildQuantilePointsPure:
    def test_output_shapes(self):
        """Returns two arrays with n_param + 3 points each."""
        n_param = 3
        tx_fracs = jnp.array([0.3, 0.5, 0.7])
        time_q, mass_q = _build_quantile_points_pure(tx_fracs, n_param)
        # (0, big_bang, tx_0, tx_1, tx_2, 1) = n_param + 3
        assert time_q.shape == (n_param + 3,)
        assert mass_q.shape == (n_param + 3,)

    def test_first_point_is_zero(self):
        """First point is (0, 0)."""
        tx_fracs = jnp.array([0.4, 0.7])
        time_q, mass_q = _build_quantile_points_pure(tx_fracs, 2)
        assert float(time_q[0]) == pytest.approx(0.0)
        assert float(mass_q[0]) == pytest.approx(0.0)

    def test_big_bang_constraint(self):
        """Second time point is 0.01 (Big Bang constraint)."""
        tx_fracs = jnp.array([0.4, 0.7])
        time_q, mass_q = _build_quantile_points_pure(tx_fracs, 2)
        assert float(time_q[1]) == pytest.approx(0.01)
        assert float(mass_q[1]) == pytest.approx(0.0)

    def test_last_point_is_one(self):
        """Last point is (1, 1)."""
        tx_fracs = jnp.array([0.3, 0.5, 0.7])
        time_q, mass_q = _build_quantile_points_pure(tx_fracs, 3)
        assert float(time_q[-1]) == pytest.approx(1.0)
        assert float(mass_q[-1]) == pytest.approx(1.0)

    def test_mass_quantiles_evenly_spaced(self):
        """mass_q interior values are linspace(0, 1, n_param+2)."""
        n_param = 4
        tx_fracs = jnp.linspace(0.2, 0.8, n_param)
        _time_q, mass_q = _build_quantile_points_pure(tx_fracs, n_param)
        # Structure: [0, 0 (big bang), linspace(0,1,n+2)[1:]]
        # mass_q[2:] should equal linspace(0,1,n_param+2)[1:]
        expected_mass = jnp.linspace(0.0, 1.0, n_param + 2)
        assert_allclose(mass_q[2:], expected_mass[1:], atol=1e-6)

    def test_single_tx_frac(self):
        """Works with n_param=1."""
        tx_fracs = jnp.array([0.5])
        time_q, mass_q = _build_quantile_points_pure(tx_fracs, 1)
        assert time_q.shape == (4,)
        assert mass_q.shape == (4,)


# ---------------------------------------------------------------------------
# dense_basis_pure_sfh
# ---------------------------------------------------------------------------


class TestDenseBasisPureSfh:
    @staticmethod
    def _default_age_grid():
        return jnp.linspace(1e6, 13e9, 200)

    def test_output_shape(self):
        """Output SFR has same length as age grid."""
        age_yr = self._default_age_grid()
        sfr = dense_basis_pure_sfh(age_yr, log_total_mass=10.0, tx_frac_0=0.3, tx_frac_1=0.6)
        assert sfr.shape == age_yr.shape

    def test_non_negative(self):
        """SFR is non-negative everywhere."""
        age_yr = self._default_age_grid()
        sfr = dense_basis_pure_sfh(age_yr, log_total_mass=10.0, tx_frac_0=0.4, tx_frac_1=0.7)
        assert jnp.all(sfr >= 0.0)

    def test_finite_output(self):
        """SFR values are all finite."""
        age_yr = self._default_age_grid()
        sfr = dense_basis_pure_sfh(age_yr, log_total_mass=10.0, tx_frac_0=0.3, tx_frac_1=0.6)
        assert jnp.all(jnp.isfinite(sfr))

    def test_total_mass_scaling(self):
        """Doubling log_total_mass by 1 dex increases integrated mass ~10x."""
        age_yr = self._default_age_grid()
        sfr_lo = dense_basis_pure_sfh(age_yr, log_total_mass=9.0, tx_frac_0=0.5)
        sfr_hi = dense_basis_pure_sfh(age_yr, log_total_mass=10.0, tx_frac_0=0.5)
        ratio = float(jnp.sum(sfr_hi)) / max(float(jnp.sum(sfr_lo)), 1e-30)
        assert 8.0 < ratio < 12.0, f"Expected ~10x mass scaling, got {ratio:.2f}"

    def test_no_tx_kwargs_raises(self):
        """Calling without any tx_frac_* raises ValueError."""
        age_yr = self._default_age_grid()
        with pytest.raises(ValueError, match="requires at least one"):
            dense_basis_pure_sfh(age_yr, log_total_mass=10.0)

    def test_missing_tx_frac_key_raises(self):
        """Passing tx_frac_1 without tx_frac_0 raises ValueError."""
        age_yr = self._default_age_grid()
        with pytest.raises(ValueError, match="Missing required parameter"):
            dense_basis_pure_sfh(age_yr, log_total_mass=10.0, tx_frac_1=0.5)

    def test_three_quantiles(self):
        """Works with three tx_frac parameters."""
        age_yr = self._default_age_grid()
        sfr = dense_basis_pure_sfh(
            age_yr, log_total_mass=10.0, tx_frac_0=0.2, tx_frac_1=0.5, tx_frac_2=0.8
        )
        assert sfr.shape == age_yr.shape
        assert jnp.all(sfr >= 0.0)

    def test_age_universe_yr_custom(self):
        """Custom age_universe_yr changes the timescale."""
        age_yr = self._default_age_grid()
        sfr_default = dense_basis_pure_sfh(age_yr, log_total_mass=10.0, tx_frac_0=0.5)
        sfr_custom = dense_basis_pure_sfh(
            age_yr, log_total_mass=10.0, age_universe_yr=10e9, tx_frac_0=0.5
        )
        # Different universes → different SFH profiles
        assert not jnp.allclose(sfr_default, sfr_custom)

    def test_jittable(self):
        """dense_basis_pure_sfh can be JIT-compiled."""
        age_yr = self._default_age_grid()

        @jax.jit
        def run(log_m, t0):
            return dense_basis_pure_sfh(age_yr, log_total_mass=log_m, tx_frac_0=t0)

        sfr = run(10.0, 0.5)
        assert sfr.shape == age_yr.shape
        assert jnp.all(jnp.isfinite(sfr))
