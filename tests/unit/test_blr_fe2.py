"""Tests for Fe II pseudo-continuum in BLR emission."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


from tengri.models.agn.blr import (
    _fe2_pseudo_continuum,
    blr_emission,
)


@pytest.fixture()
def wavelength():
    """Rest-frame wavelength grid covering UV through optical."""
    return jnp.linspace(1000.0, 8000.0, 5000)


class TestFe2PseudoContinuum:
    """Tests for the _fe2_pseudo_continuum helper."""

    def test_zero_strength_returns_zeros(self, wavelength):
        result = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=0.0)
        assert result.shape == wavelength.shape
        assert jnp.allclose(result, 0.0, atol=1e-30)

    def test_positive_strength_has_flux(self, wavelength):
        result = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=1.0)
        assert jnp.any(result > 0.0)

    def test_optical_bump_present(self, wavelength):
        """The 4570 A optical Fe II bump should be the dominant optical feature."""
        result = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=1.0)
        # Find flux near 4570 A
        mask_4570 = (wavelength > 4500.0) & (wavelength < 4650.0)
        # Find flux in a quiet region (e.g., 3500-3800 A between UV and optical)
        mask_quiet = (wavelength > 3500.0) & (wavelength < 3800.0)
        assert jnp.mean(result[mask_4570]) > 10.0 * jnp.mean(result[mask_quiet])

    def test_uv_bump_present(self, wavelength):
        """UV Fe II groups near 2400 and 2600 A should have significant flux."""
        result = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=1.0)
        mask_uv = (wavelength > 2200.0) & (wavelength < 2800.0)
        mask_quiet = (wavelength > 3500.0) & (wavelength < 3800.0)
        assert jnp.mean(result[mask_uv]) > 5.0 * jnp.mean(result[mask_quiet])

    def test_strength_scales_linearly(self, wavelength):
        """Doubling fe2_strength should double the output."""
        result_1 = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=1.0)
        result_2 = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=2.0)
        ratio = result_2 / jnp.maximum(result_1, 1e-30)
        # Only check where there is significant flux
        mask = result_1 > 1e-10 * jnp.max(result_1)
        assert jnp.allclose(ratio[mask], 2.0, rtol=1e-5)

    def test_broader_fwhm_spreads_flux(self, wavelength):
        """Larger FWHM should produce broader, lower-peak features."""
        narrow = _fe2_pseudo_continuum(wavelength, fwhm_kms=1000.0, fe2_strength=1.0)
        broad = _fe2_pseudo_continuum(wavelength, fwhm_kms=5000.0, fe2_strength=1.0)
        # Peak should be lower for broader FWHM
        assert jnp.max(broad) < jnp.max(narrow)

    def test_jit_compatible(self, wavelength):
        """Function should be JIT-compilable."""
        fn = jax.jit(lambda w: _fe2_pseudo_continuum(w, fwhm_kms=3000.0, fe2_strength=1.0))
        result = fn(wavelength)
        assert result.shape == wavelength.shape
        assert jnp.all(jnp.isfinite(result))

    def test_differentiable(self, wavelength):
        """Fe II output should be differentiable w.r.t. fe2_strength."""

        def loss(strength):
            fe2 = _fe2_pseudo_continuum(wavelength, fwhm_kms=3000.0, fe2_strength=strength)
            return jnp.sum(fe2)

        grad_fn = jax.grad(loss)
        g = grad_fn(1.0)
        assert jnp.isfinite(g)
        assert g > 0.0  # Positive gradient since more strength = more flux


class TestBlrEmissionWithFe2:
    """Tests for blr_emission with agn_fe2_strength parameter."""

    def test_backward_compatible_default(self, wavelength):
        """Default agn_fe2_strength=0.0 should match old behavior."""
        result_default = blr_emission(wavelength, l_disc_bol_erg=1e45, covering_fraction=0.1)
        result_explicit = blr_emission(
            wavelength,
            l_disc_bol_erg=1e45,
            covering_fraction=0.1,
            agn_fe2_strength=0.0,
        )
        assert jnp.allclose(result_default, result_explicit, rtol=1e-10)

    def test_fe2_adds_flux(self, wavelength):
        """Enabling Fe II should add flux, never subtract."""
        result_no_fe2 = blr_emission(wavelength, l_disc_bol_erg=1e45, agn_fe2_strength=0.0)
        result_with_fe2 = blr_emission(wavelength, l_disc_bol_erg=1e45, agn_fe2_strength=1.0)
        # Fe II should add flux in the optical bump region
        mask_opt = (wavelength > 4400.0) & (wavelength < 4700.0)
        assert jnp.all(result_with_fe2[mask_opt] >= result_no_fe2[mask_opt] - 1e-30)

    def test_fe2_increases_optical_bump(self, wavelength):
        """The 4400-5500 A region should show significant excess with Fe II."""
        no_fe2 = blr_emission(wavelength, l_disc_bol_erg=1e45, agn_fe2_strength=0.0)
        with_fe2 = blr_emission(wavelength, l_disc_bol_erg=1e45, agn_fe2_strength=1.5)
        mask = (wavelength > 4434.0) & (wavelength < 4684.0)
        excess = jnp.sum(with_fe2[mask]) - jnp.sum(no_fe2[mask])
        assert excess > 0.0

    def test_jit_with_fe2(self, wavelength):
        """blr_emission with Fe II should be JIT-compilable."""
        fn = jax.jit(lambda w: blr_emission(w, l_disc_bol_erg=1e45, agn_fe2_strength=1.0))
        result = fn(wavelength)
        assert jnp.all(jnp.isfinite(result))
        assert result.shape == wavelength.shape
