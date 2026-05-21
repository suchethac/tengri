"""Tests for Lopez+2024 IRX-based X-ray AGN model.

Validates:
- α_IRX → L_X derivation
- Power-law spectral shape and normalization
- Bolometric correction (Duras+2020)
- Energy range masking (λ < 124 Å)
- Anisotropy correction
- JIT-compatibility and gradient safety
- Combination with XRB component
"""

import chex
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.xray.xray import (
    xray_agn_corona_lopez24,
    xray_bolometric_correction_duras,
    xray_total_lopez24,
)


@pytest.fixture()
def xray_wavelength():
    """X-ray wavelength grid: 0.1 Å to 200 Å (covers 0.06-124 keV)."""
    return jnp.linspace(0.1, 200.0, 500)


class TestBolometricCorrectionDuras:
    def test_positive(self):
        k = xray_bolometric_correction_duras(1e43)
        assert k > 1.0

    def test_monotonically_increasing(self):
        k1 = xray_bolometric_correction_duras(1e40)
        k2 = xray_bolometric_correction_duras(1e42)
        k3 = xray_bolometric_correction_duras(1e44)
        assert k1 <= k2 <= k3

    def test_high_luminosity_larger(self):
        k_low = xray_bolometric_correction_duras(1e42)
        k_high = xray_bolometric_correction_duras(1e46)
        assert k_high > k_low

    def test_jit(self):
        fn = jax.jit(xray_bolometric_correction_duras)
        k = fn(1e44)
        assert jnp.isfinite(k)


class TestLopez24Corona:
    def test_output_shape(self, xray_wavelength):
        result = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=1e30)
        chex.assert_equal_shape([result, xray_wavelength])

    def test_nonnegative(self, xray_wavelength):
        result = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=1e30)
        assert jnp.all(result >= 0.0)

    def test_zero_outside_xray(self, xray_wavelength):
        result = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=1e30)
        optical_mask = xray_wavelength >= 124.0
        assert jnp.all(result[optical_mask] == 0.0)

    def test_nonzero_in_xray_band(self, xray_wavelength):
        result = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=1e30)
        xray_mask = xray_wavelength < 124.0
        assert jnp.any(result[xray_mask] > 0.0)

    def test_alpha_irx_scaling(self, xray_wavelength):
        low = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            alpha_irx=0.0,
        )
        high = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            alpha_irx=0.6,
        )
        assert jnp.sum(high) > jnp.sum(low)

    def test_l12um_scaling(self, xray_wavelength):
        low = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=1e28)
        high = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=1e30)
        assert jnp.sum(high) > jnp.sum(low)

    def test_zero_luminosity_gives_zero(self, xray_wavelength):
        result = xray_agn_corona_lopez24(xray_wavelength, l_12um_erg_hz=0.0)
        assert jnp.allclose(result, 0.0, atol=1e-50)

    def test_gamma_affects_slope(self, xray_wavelength):
        soft = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            gamma=1.4,
        )
        steep = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            gamma=3.0,
        )
        xray_mask = xray_wavelength < 50.0
        soft_xr = soft[xray_mask]
        steep_xr = steep[xray_mask]
        ratio_soft = jnp.max(soft_xr) / jnp.maximum(
            jnp.min(soft_xr[soft_xr > 0]),
            1e-100,
        )
        ratio_steep = jnp.max(steep_xr) / jnp.maximum(
            jnp.min(steep_xr[steep_xr > 0]),
            1e-100,
        )
        assert ratio_steep > ratio_soft

    def test_anisotropy_reduces_edge_on(self, xray_wavelength):
        face_on = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            cos_inc=1.0,
            apply_anisotropy=True,
        )
        edge_on = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            cos_inc=0.0,
            apply_anisotropy=True,
        )
        assert jnp.sum(face_on) > jnp.sum(edge_on)

    def test_no_anisotropy_flag(self, xray_wavelength):
        with_aniso = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            cos_inc=0.3,
            apply_anisotropy=True,
        )
        without_aniso = xray_agn_corona_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
            cos_inc=0.3,
            apply_anisotropy=False,
        )
        assert not jnp.allclose(with_aniso, without_aniso)

    def test_jit(self, xray_wavelength):
        fn = jax.jit(lambda w: xray_agn_corona_lopez24(w, l_12um_erg_hz=1e30))
        result = fn(xray_wavelength)
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_alpha_irx(self, xray_wavelength):
        def loss(a):
            return jnp.sum(
                xray_agn_corona_lopez24(
                    xray_wavelength,
                    l_12um_erg_hz=1e30,
                    alpha_irx=a,
                    apply_anisotropy=False,
                )
            )

        grad = jax.grad(loss)(0.3)
        assert jnp.isfinite(grad)
        assert grad > 0.0

    def test_gradient_wrt_l12um(self, xray_wavelength):
        def loss(l12):
            return jnp.sum(
                xray_agn_corona_lopez24(
                    xray_wavelength,
                    l_12um_erg_hz=l12,
                    apply_anisotropy=False,
                )
            )

        grad = jax.grad(loss)(1e30)
        assert jnp.isfinite(grad)
        assert grad > 0.0


class TestTotalLopez24:
    def test_output_shape(self, xray_wavelength):
        result = xray_total_lopez24(xray_wavelength)
        chex.assert_equal_shape([result, xray_wavelength])

    def test_xrb_only_when_no_agn(self, xray_wavelength):
        from tengri.components.xray.xray import xray_xrb

        total = xray_total_lopez24(
            xray_wavelength,
            sfr=1.0,
            stellar_mass=1e10,
            l_12um_erg_hz=0.0,
        )
        xrb = xray_xrb(xray_wavelength, sfr=1.0, stellar_mass=1e10, E_cut=300.0)
        assert jnp.allclose(total, xrb, atol=1e-50)

    def test_agn_adds_flux(self, xray_wavelength):
        no_agn = xray_total_lopez24(
            xray_wavelength,
            l_12um_erg_hz=0.0,
        )
        with_agn = xray_total_lopez24(
            xray_wavelength,
            l_12um_erg_hz=1e30,
        )
        assert jnp.sum(with_agn) > jnp.sum(no_agn)

    def test_jit(self, xray_wavelength):
        fn = jax.jit(lambda w: xray_total_lopez24(w, l_12um_erg_hz=1e30))
        result = fn(xray_wavelength)
        chex.assert_tree_all_finite(result)
