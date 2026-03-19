"""Tests for [alpha/Fe] enhancement support.

Verifies:
1. effective_metallicity computes the correct offset
2. Alpha enhancement shifts the SED relative to solar ratios
3. Backward compatibility: alpha_fe=0 gives identical results
4. Gradients flow through the alpha enhancement path
5. ParamSpec correctly registers met_alpha_fe
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from diffsed.models.sps.dsps_wrapper import (
    _ALPHA_TO_Z_COEFF,
    effective_metallicity,
    interpolate_metallicity,
)
from diffsed.param_spec import ParamSpec

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Unit tests for effective_metallicity
# ---------------------------------------------------------------------------


class TestEffectiveMetallicity:
    """Test the effective_metallicity function."""

    def test_solar_ratios_no_shift(self):
        """[alpha/Fe] = 0 should return the input metallicity unchanged."""
        log_z = -1.5
        result = effective_metallicity(log_z, alpha_fe=0.0)
        assert_allclose(float(result), log_z, atol=1e-15)

    def test_positive_alpha_increases_z(self):
        """Positive [alpha/Fe] should increase total metallicity."""
        log_z = -1.5
        alpha_fe = 0.3
        result = effective_metallicity(log_z, alpha_fe)
        expected = log_z + _ALPHA_TO_Z_COEFF * alpha_fe
        assert_allclose(float(result), expected, atol=1e-15)
        assert float(result) > log_z

    def test_negative_alpha_decreases_z(self):
        """Negative [alpha/Fe] should decrease total metallicity."""
        log_z = -1.0
        alpha_fe = -0.1
        result = effective_metallicity(log_z, alpha_fe)
        assert float(result) < log_z

    def test_coefficient_value(self):
        """The coefficient should be 0.75 (Thomas+2003)."""
        assert_allclose(_ALPHA_TO_Z_COEFF, 0.75, atol=1e-10)

    def test_linearity(self):
        """effective_metallicity should be linear in both arguments."""
        log_z = -1.5
        alpha1 = 0.2
        alpha2 = 0.4
        r1 = effective_metallicity(log_z, alpha1)
        r2 = effective_metallicity(log_z, alpha2)
        # Doubling alpha should double the offset
        assert_allclose(
            float(r2) - log_z,
            2.0 * (float(r1) - log_z),
            atol=1e-14,
        )

    def test_jit_compatible(self):
        """Should compile under jax.jit without issues."""
        f = jax.jit(effective_metallicity)
        result = f(-1.5, 0.3)
        expected = -1.5 + 0.75 * 0.3
        assert_allclose(float(result), expected, atol=1e-14)

    def test_gradient_wrt_alpha(self):
        """Gradient of effective_metallicity w.r.t. alpha_fe should be 0.75."""
        g = jax.grad(effective_metallicity, argnums=1)(-1.5, 0.3)
        assert_allclose(float(g), _ALPHA_TO_Z_COEFF, atol=1e-14)

    def test_gradient_wrt_log_z(self):
        """Gradient of effective_metallicity w.r.t. log_z_fe should be 1.0."""
        g = jax.grad(effective_metallicity, argnums=0)(-1.5, 0.3)
        assert_allclose(float(g), 1.0, atol=1e-14)

    def test_vectorized(self):
        """Should work with array inputs via vmap."""
        log_z_arr = jnp.array([-2.0, -1.5, -1.0, -0.5])
        alpha_arr = jnp.array([0.0, 0.2, 0.4, 0.3])
        result = jax.vmap(effective_metallicity)(log_z_arr, alpha_arr)
        expected = log_z_arr + _ALPHA_TO_Z_COEFF * alpha_arr
        assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# ParamSpec integration
# ---------------------------------------------------------------------------


class TestAlphaFeParamSpec:
    """Test that ParamSpec correctly handles met_alpha_fe."""

    def test_default_is_fixed_zero(self):
        """met_alpha_fe should default to Fixed(0.0)."""
        spec = ParamSpec(
            sfh_dpl_alpha=1.0, sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=5.0, sfh_dpl_log_peak_sfr=1.0,
            mean_sfh_type="dpl",
        )
        dist = spec.get_distribution("met_alpha_fe")
        assert dist.is_fixed
        assert_allclose(dist.bounds[0], 0.0)

    def test_alpha_fe_in_all_params(self):
        """met_alpha_fe should appear in all_params."""
        spec = ParamSpec(mean_sfh_type="dpl")
        assert "met_alpha_fe" in spec.all_params

    def test_alpha_fe_free_when_set(self):
        """met_alpha_fe should be free when given a distribution."""
        from diffsed.distributions import Uniform
        spec = ParamSpec(
            mean_sfh_type="dpl",
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
        assert "met_alpha_fe" in spec.free_params
        dist = spec.get_distribution("met_alpha_fe")
        assert not dist.is_fixed
        lo, hi = dist.bounds
        assert_allclose(lo, -0.2)
        assert_allclose(hi, 0.6)

    def test_backward_compatible_no_alpha(self):
        """Models without explicit met_alpha_fe should work unchanged."""
        spec = ParamSpec(
            mean_sfh_type="dpl",
            met_logzsol=-0.3,
        )
        assert "met_alpha_fe" in spec.fixed_params

    def test_sample_includes_alpha_fe(self):
        """Sampling should include met_alpha_fe."""
        from diffsed.distributions import Uniform
        spec = ParamSpec(
            mean_sfh_type="dpl",
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
        params = spec.sample(jax.random.PRNGKey(42))
        assert "met_alpha_fe" in params
        val = float(params["met_alpha_fe"])
        assert -0.2 <= val <= 0.6


# ---------------------------------------------------------------------------
# Forward model integration (requires SSP data)
# ---------------------------------------------------------------------------


class TestAlphaFeForwardModel:
    """Test alpha enhancement in the full forward model."""

    @pytest.fixture
    def model_with_alpha(self):
        from diffsed import Model, ParamSpec, Fixed, Uniform, load_ssp_data
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        spec = ParamSpec(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_peak_sfr=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            met_alpha_fe=Uniform(-0.2, 0.6),
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
        )
        return Model(spec, ssp, precompute=False)

    def test_alpha_zero_matches_no_alpha(self, model_with_alpha):
        """alpha_fe=0 should give identical SED as no alpha enhancement."""
        sed_alpha0 = model_with_alpha.predict_sed({"met_alpha_fe": 0.0})
        # Build a model without alpha_fe (defaults to Fixed(0.0))
        from diffsed import Model, ParamSpec, Fixed, load_ssp_data
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        spec_no = ParamSpec(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_peak_sfr=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
        )
        model_no = Model(spec_no, ssp, precompute=False)
        sed_no = model_no.predict_sed({})
        assert_allclose(sed_alpha0, sed_no, rtol=1e-12)

    def test_positive_alpha_changes_sed(self, model_with_alpha):
        """Positive [alpha/Fe] should produce a different SED."""
        sed_0 = model_with_alpha.predict_sed({"met_alpha_fe": 0.0})
        sed_pos = model_with_alpha.predict_sed({"met_alpha_fe": 0.3})
        # SEDs should differ (alpha enhancement shifts metallicity)
        max_diff = float(jnp.max(jnp.abs(sed_pos - sed_0)))
        assert max_diff > 0, "Alpha enhancement should change the SED"

    def test_alpha_gradient_finite(self, model_with_alpha):
        """Gradient w.r.t. met_alpha_fe should be finite."""
        def loss(afe):
            return jnp.sum(model_with_alpha.predict_sed({"met_alpha_fe": afe}))
        g = jax.grad(loss)(0.3)
        assert jnp.isfinite(g), "Alpha-Fe gradient should be finite"
        assert abs(float(g)) > 0, "Alpha-Fe gradient should be non-zero"
