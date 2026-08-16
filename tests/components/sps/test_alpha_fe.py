# SPDX-License-Identifier: BSD-3-Clause
"""Tests for [alpha/Fe] enhancement support.

Verifies:
1. effective_metallicity computes the correct offset
2. Alpha enhancement shifts the SED relative to solar ratios
3. Backward compatibility: alpha_fe=0 gives identical results
4. Gradients flow through the alpha enhancement path
5. Parameters correctly registers met_alpha_fe
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.dsps_wrapper import (
    _ALPHA_TO_Z_COEFF,
    effective_metallicity,
)
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Unit tests for effective_metallicity ──────────────────────────


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


# ── Parameters integration ─────────────────────────────────────────


class TestAlphaFeParamSpec:
    """Test that Parameters correctly handles met_alpha_fe."""

    def test_default_is_fixed_zero(self):
        """met_alpha_fe should default to Fixed(0.0)."""
        spec = Parameters(
            sfh_dpl_alpha=1.0,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=5.0,
            sfh_dpl_log_total_mass=1.0,
            mean_sfh_type="dpl",
        )
        dist = spec.get_distribution("met_alpha_fe")
        assert dist.is_fixed
        assert_allclose(dist.bounds[0], 0.0)

    def test_alpha_fe_in_all_params(self):
        """met_alpha_fe should appear in all_params."""
        spec = Parameters(mean_sfh_type="dpl")
        assert "met_alpha_fe" in spec.all_params

    def test_alpha_fe_free_when_set(self):
        """met_alpha_fe should be free when given a distribution."""
        from tengri.parameters.priors import Uniform

        spec = Parameters(
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
        spec = Parameters(
            mean_sfh_type="dpl",
            met_logzsol=-0.3,
        )
        assert "met_alpha_fe" in spec.fixed_params

    def test_sample_includes_alpha_fe(self):
        """Sampling should include met_alpha_fe."""
        from tengri.parameters.priors import Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
        params = spec.sample(jax.random.PRNGKey(42))
        assert "met_alpha_fe" in params
        val = float(params["met_alpha_fe"])
        assert -0.2 <= val <= 0.6


# ── Forward model integration (requires SSP data) ─────────────────


class TestAlphaFeForwardModel:
    """Test alpha enhancement in the full forward model."""

    @pytest.fixture
    def model_with_alpha(self, ssp_data_fsps):
        from tengri import Fixed, Parameters, SEDModel, Uniform

        ssp = ssp_data_fsps
        spec = Parameters(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            met_alpha_fe=Uniform(-0.2, 0.6),
            sfh_dpl_age_gyr=Fixed(13.0),  # pin the otherwise-free formation time (#1021)
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
        )
        return SEDModel(spec, ssp, precompute=False)

    def test_alpha_zero_matches_no_alpha(self, model_with_alpha, ssp_data_fsps):
        """alpha_fe=0 gives bit-exact match the no-α path.

        Closed by extending closure-path-A to the α-aware branch in
        ``forward/pipeline.py``: when a 4D α-grid is loaded, the
        α-aware path does α-only bilinear interp on ``ssp_flux``,
        then feeds the resulting 3D (n_met, n_age, n_wave) cube to
        ``calc_rest_sed_sfh_table_lognormal_mdf`` — same kernel as
        the no-α delta-Z path. At ``α=0`` the α-only interp is
        bit-exact (α=0 is a grid point), so the two paths reduce
        to the same lognormal-MDF SED.
        """
        sed_alpha0 = model_with_alpha.predict_rest_sed({"met_alpha_fe": 0.0}).sed
        from tengri import Fixed, Parameters, SEDModel

        ssp = ssp_data_fsps
        spec_no = Parameters(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            sfh_dpl_age_gyr=Fixed(13.0),  # pin the otherwise-free formation time (#1021)
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
        )
        model_no = SEDModel(spec_no, ssp, precompute=False)
        sed_no = model_no.predict_rest_sed({}).sed
        assert_allclose(sed_alpha0, sed_no, rtol=1e-12)

    def test_positive_alpha_changes_sed(self, model_with_alpha):
        """Positive [alpha/Fe] should produce a different SED."""
        sed_0 = model_with_alpha.predict_rest_sed({"met_alpha_fe": 0.0}).sed
        sed_pos = model_with_alpha.predict_rest_sed({"met_alpha_fe": 0.3}).sed
        # SEDs should differ (alpha enhancement shifts metallicity)
        max_diff = float(jnp.max(jnp.abs(sed_pos - sed_0)))
        assert max_diff > 0, "Alpha enhancement should change the SED"

    def test_alpha_gradient_finite(self, model_with_alpha):
        """Gradient w.r.t. met_alpha_fe should be finite."""

        def loss(afe):
            return jnp.sum(model_with_alpha.predict_rest_sed({"met_alpha_fe": afe}).sed)

        grad_jax = float(jax.grad(loss)(0.3))
        grad_fd = fd_grad(loss, 0.3)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert abs(grad_jax) > 0, "Alpha-Fe gradient should be non-zero"
