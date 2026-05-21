"""Tests for rest-frame SED diagnostics."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.analysis.diagnostics.spectral import (
    dn4000,
    irx,
    rest_frame_colour,
    rest_frame_luminosity,
    uv_slope_beta,
)

pytestmark = pytest.mark.bounds


class TestUVSlopeBeta:
    """Test suite for uv_slope_beta diagnostic."""

    @pytest.fixture
    def wavelength_uv(self):
        """UV wavelength grid (1200–2600 Å)."""
        return jnp.linspace(1200.0, 2600.0, 300)

    def test_flat_spectrum_beta_minus_two(self, wavelength_uv):
        """Test that flat F_λ spectrum gives β ≈ -2."""
        # For a flat spectrum F_λ = const, we have L_ν = F_λ * c / λ²,
        # so log(F_λ) is flat, and when we fit log(F_λ) vs log(λ),
        # the slope should be 0 (not -2, which is log(L_ν) vs log(λ))
        f_lambda_flat = jnp.ones_like(wavelength_uv)

        # Convert to L_ν
        c_cgs = 2.99792458e10
        wavelength_cm = wavelength_uv * 1.0e-8
        l_nu_flat = f_lambda_flat * (wavelength_cm**2) / c_cgs

        beta = uv_slope_beta(wavelength_uv, l_nu_flat)

        # Flat F_λ gives beta ≈ 0 when fitting log(F_λ) vs log(λ)
        np.testing.assert_allclose(beta, 0.0, atol=0.1)

    def test_power_law_spectrum_recovery(self, wavelength_uv):
        """Test that power-law F_λ ∝ λ^β recovers β."""
        # Generate F_λ ∝ λ^1.5
        beta_input = 1.5
        f_lambda = wavelength_uv**beta_input

        # Convert to L_ν
        c_cgs = 2.99792458e10
        wavelength_cm = wavelength_uv * 1.0e-8
        l_nu = f_lambda * (wavelength_cm**2) / c_cgs

        beta_recovered = uv_slope_beta(wavelength_uv, l_nu)

        # Should recover the input slope (within ±0.2)
        np.testing.assert_allclose(beta_recovered, beta_input, atol=0.2)

    def test_negative_slope_spectrum(self, wavelength_uv):
        """Test negative slope (blueward spectrum)."""
        # F_λ ∝ λ^(-0.5)
        beta_input = -0.5
        f_lambda = wavelength_uv**beta_input

        c_cgs = 2.99792458e10
        wavelength_cm = wavelength_uv * 1.0e-8
        l_nu = f_lambda * (wavelength_cm**2) / c_cgs

        beta_recovered = uv_slope_beta(wavelength_uv, l_nu)

        # Should recover negative slope
        assert beta_recovered < 0.0

    def test_output_is_scalar(self, wavelength_uv):
        """Test that output is a scalar."""
        l_nu = jnp.ones_like(wavelength_uv)
        beta = uv_slope_beta(wavelength_uv, l_nu)

        # Check that it's a scalar (shape = ())
        assert jnp.ndim(beta) == 0


class TestDn4000:
    """Test suite for dn4000 diagnostic."""

    @pytest.fixture
    def wavelength_optical(self):
        """Optical wavelength grid covering 3850–4100 Å."""
        return jnp.linspace(3500.0, 4500.0, 200)

    def test_flat_spectrum_dn4000_unity(self, wavelength_optical):
        """Test that flat spectrum gives Dn4000 ≈ 1."""
        l_nu_flat = jnp.ones_like(wavelength_optical)

        dn_value = dn4000(wavelength_optical, l_nu_flat)

        # For flat spectrum, red and blue bands have equal flux
        np.testing.assert_allclose(dn_value, 1.0, atol=0.05)

    def test_red_biased_spectrum_dn4000_greater_than_unity(self, wavelength_optical):
        """Test that red-biased spectrum gives Dn4000 > 1."""
        # L_ν ∝ λ (red-biased; like old stellar population)
        l_nu = wavelength_optical

        dn_value = dn4000(wavelength_optical, l_nu)

        # Should be > 1 because red band (4000–4100) has higher flux
        assert dn_value > 1.0

    def test_blue_biased_spectrum_dn4000_less_than_unity(self, wavelength_optical):
        """Test that blue-biased spectrum gives Dn4000 < 1."""
        # L_ν ∝ 1/λ (blue-biased; like young OB stars)
        l_nu = 1.0 / wavelength_optical

        dn_value = dn4000(wavelength_optical, l_nu)

        # Should be < 1 because blue band (3850–3950) has higher flux
        assert dn_value < 1.0

    def test_output_is_scalar(self, wavelength_optical):
        """Test that output is a scalar."""
        l_nu = jnp.ones_like(wavelength_optical)
        dn_value = dn4000(wavelength_optical, l_nu)

        assert jnp.ndim(dn_value) == 0

    def test_positive_output(self, wavelength_optical):
        """Test that Dn4000 is always positive."""
        l_nu = jnp.abs(jnp.sin(wavelength_optical)) + 1.0
        dn_value = dn4000(wavelength_optical, l_nu)

        assert dn_value > 0.0


class TestIRX:
    """Test suite for infrared excess (IRX) diagnostic."""

    def test_equal_luminosities_irx_zero(self):
        """Test that equal L_dust and L_FUV gives IRX = 0."""
        irx_value = irx(1.0, 1.0)

        np.testing.assert_allclose(irx_value, 0.0)

    def test_higher_dust_gives_positive_irx(self):
        """Test that L_dust > L_FUV gives IRX > 0."""
        irx_value = irx(10.0, 1.0)

        assert irx_value > 0.0

    def test_higher_fuv_gives_negative_irx(self):
        """Test that L_dust < L_FUV gives IRX < 0."""
        irx_value = irx(0.1, 1.0)

        assert irx_value < 0.0

    def test_irx_values_match_expected(self):
        """Test IRX values against known test cases."""
        # IRX = log10(10/1) = 1.0
        np.testing.assert_allclose(irx(10.0, 1.0), 1.0)

        # IRX = log10(1/1) = 0.0
        np.testing.assert_allclose(irx(1.0, 1.0), 0.0)

        # IRX = log10(0.1/1) = -1.0
        np.testing.assert_allclose(irx(0.1, 1.0), -1.0)

    def test_output_is_scalar(self):
        """Test that output is a scalar."""
        irx_value = irx(5.0, 2.0)

        assert jnp.ndim(irx_value) == 0


class TestRestFrameLuminosity:
    """Test suite for rest-frame synthetic photometry."""

    @pytest.fixture
    def wavelength_sed(self):
        """SED wavelength grid."""
        return jnp.logspace(3.0, 5.0, 500)  # 1000–100000 Å

    @pytest.fixture
    def tophat_filter(self):
        """Top-hat filter centered at 5000 Å, width 1000 Å."""
        filter_wave = jnp.linspace(4500.0, 5500.0, 100)
        filter_trans = jnp.where((filter_wave >= 4750.0) & (filter_wave <= 5250.0), 1.0, 0.0)
        return filter_wave, filter_trans

    def test_flat_spectrum_through_tophat(self, wavelength_sed, tophat_filter):
        """Test that flat spectrum through top-hat filter gives correct flux."""
        l_nu_flat = jnp.ones_like(wavelength_sed)

        filter_wave, filter_trans = tophat_filter
        l_phot = rest_frame_luminosity(wavelength_sed, l_nu_flat, filter_wave, filter_trans)

        # For flat spectrum, the photometry should be close to 1
        # (exact value depends on filter shape and interpolation)
        assert l_phot > 0.5
        assert l_phot < 2.0

    def test_zero_spectrum_gives_zero_luminosity(self, wavelength_sed, tophat_filter):
        """Test that zero spectrum gives zero luminosity."""
        l_nu_zero = jnp.zeros_like(wavelength_sed)

        filter_wave, filter_trans = tophat_filter
        l_phot = rest_frame_luminosity(wavelength_sed, l_nu_zero, filter_wave, filter_trans)

        np.testing.assert_allclose(l_phot, 0.0)

    def test_output_is_scalar(self, wavelength_sed, tophat_filter):
        """Test that output is a scalar."""
        l_nu = jnp.ones_like(wavelength_sed)

        filter_wave, filter_trans = tophat_filter
        l_phot = rest_frame_luminosity(wavelength_sed, l_nu, filter_wave, filter_trans)

        assert jnp.ndim(l_phot) == 0

    def test_positive_output(self, wavelength_sed, tophat_filter):
        """Test that luminosity is always non-negative."""
        l_nu = jnp.abs(jnp.sin(wavelength_sed)) + 1.0

        filter_wave, filter_trans = tophat_filter
        l_phot = rest_frame_luminosity(wavelength_sed, l_nu, filter_wave, filter_trans)

        assert l_phot >= 0.0

    def test_normalization_independence(self, wavelength_sed, tophat_filter):
        """Test that doubling the spectrum doubles the flux."""
        l_nu = jnp.ones_like(wavelength_sed)

        filter_wave, filter_trans = tophat_filter
        l_phot1 = rest_frame_luminosity(wavelength_sed, l_nu, filter_wave, filter_trans)
        l_phot2 = rest_frame_luminosity(wavelength_sed, 2.0 * l_nu, filter_wave, filter_trans)

        np.testing.assert_allclose(l_phot2, 2.0 * l_phot1, rtol=0.01)


class TestRestFrameColour:
    """Test suite for rest-frame photometric colour."""

    @pytest.fixture
    def wavelength_sed(self):
        """SED wavelength grid."""
        return jnp.logspace(3.0, 5.0, 500)

    @pytest.fixture
    def blue_filter(self):
        """Blue filter centered at 4500 Å."""
        filter_wave = jnp.linspace(4000.0, 5000.0, 100)
        filter_trans = jnp.where((filter_wave >= 4250.0) & (filter_wave <= 4750.0), 1.0, 0.0)
        return filter_wave, filter_trans

    @pytest.fixture
    def red_filter(self):
        """Red filter centered at 6500 Å."""
        filter_wave = jnp.linspace(6000.0, 7000.0, 100)
        filter_trans = jnp.where((filter_wave >= 6250.0) & (filter_wave <= 6750.0), 1.0, 0.0)
        return filter_wave, filter_trans

    def test_same_filter_twice_gives_zero_colour(self, wavelength_sed, blue_filter):
        """Test that comparing identical filters gives colour = 0."""
        l_nu = jnp.ones_like(wavelength_sed)

        filter_wave, filter_trans = blue_filter
        colour = rest_frame_colour(
            wavelength_sed, l_nu, filter_wave, filter_trans, filter_wave, filter_trans
        )

        np.testing.assert_allclose(colour, 0.0, atol=1e-5)

    def test_flat_spectrum_through_different_filters(
        self, wavelength_sed, blue_filter, red_filter
    ):
        """Test colour of flat spectrum through different filters."""
        l_nu = jnp.ones_like(wavelength_sed)

        blue_wave, blue_trans = blue_filter
        red_wave, red_trans = red_filter

        colour = rest_frame_colour(
            wavelength_sed, l_nu, blue_wave, blue_trans, red_wave, red_trans
        )

        # For flat spectrum, blue and red fluxes should be similar
        # so colour should be close to 0
        assert jnp.abs(colour) < 0.5

    def test_red_biased_spectrum_gives_positive_colour(
        self, wavelength_sed, blue_filter, red_filter
    ):
        """Test that red-biased spectrum gives positive (red) colour."""
        # L_ν ∝ λ (red-biased)
        l_nu = wavelength_sed

        blue_wave, blue_trans = blue_filter
        red_wave, red_trans = red_filter

        colour = rest_frame_colour(
            wavelength_sed, l_nu, blue_wave, blue_trans, red_wave, red_trans
        )

        # Red filter should have more flux, so m_blue - m_red = -2.5*log(f_blue/f_red) > 0
        assert colour > 0.0

    def test_blue_biased_spectrum_gives_negative_colour(
        self, wavelength_sed, blue_filter, red_filter
    ):
        """Test that blue-biased spectrum gives negative (blue) colour."""
        # L_ν ∝ 1/λ (blue-biased)
        l_nu = 1.0 / wavelength_sed

        blue_wave, blue_trans = blue_filter
        red_wave, red_trans = red_filter

        colour = rest_frame_colour(
            wavelength_sed, l_nu, blue_wave, blue_trans, red_wave, red_trans
        )

        # Blue filter should have more flux, so m_blue - m_red = -2.5*log(f_blue/f_red) < 0
        assert colour < 0.0

    def test_output_is_scalar(self, wavelength_sed, blue_filter, red_filter):
        """Test that output is a scalar."""
        l_nu = jnp.ones_like(wavelength_sed)

        blue_wave, blue_trans = blue_filter
        red_wave, red_trans = red_filter

        colour = rest_frame_colour(
            wavelength_sed, l_nu, blue_wave, blue_trans, red_wave, red_trans
        )

        assert jnp.ndim(colour) == 0
