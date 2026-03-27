"""Unit tests for MAPPINGS V shock emission model."""

import jax
import jax.numpy as jnp
import pytest

from tengri.models.nebular.shock import (
    _N_LINES,
    shock_emission_sed,
    shock_line_ratios,
)

# ---------------------------------------------------------------------------
# shock_line_ratios
# ---------------------------------------------------------------------------


class TestShockLineRatios:
    """Tests for shock_line_ratios."""

    def test_all_ratios_positive(self):
        """All line ratios should be strictly positive."""
        for v in [100.0, 300.0, 500.0, 1000.0]:
            ratios = shock_line_ratios(v)
            for name, val in ratios.items():
                assert float(val) > 0.0, f"{name} at v={v} is not positive"

    def test_hbeta_is_unity(self):
        """Hbeta ratio should always be 1.0."""
        ratios = shock_line_ratios(300.0)
        assert float(ratios["Hbeta"]) == pytest.approx(1.0)

    def test_nii_enhanced_relative_to_case_b(self):
        """[NII]/Halpha should be elevated (shock diagnostic signature).

        In HII regions, [NII]/Halpha ~ 0.1-0.5 typically.
        In shocks, [NII]/Halpha > 0.5 at most velocities.
        """
        for v in [150.0, 200.0, 300.0]:
            ratios = shock_line_ratios(v)
            nii_ha = float(ratios["NII_6583"]) / float(ratios["Halpha"])
            assert nii_ha > 0.5, f"[NII]/Halpha={nii_ha:.2f} at v={v} not shock-like"

    def test_oiii_peaks_at_intermediate_velocity(self):
        """[OIII] should peak at intermediate velocities (~300-500 km/s)."""
        ratios_low = shock_line_ratios(100.0)
        ratios_mid = shock_line_ratios(400.0)
        ratios_high = shock_line_ratios(1000.0)

        oiii_low = float(ratios_low["OIII_5007"])
        oiii_mid = float(ratios_mid["OIII_5007"])
        oiii_high = float(ratios_high["OIII_5007"])

        assert oiii_mid > oiii_low, "[OIII] should increase from 100 to 400 km/s"
        assert oiii_mid > oiii_high, "[OIII] should decrease from 400 to 1000 km/s"

    def test_velocity_clipping(self):
        """Velocities outside [100, 1000] should be clipped, not error."""
        ratios_lo = shock_line_ratios(50.0)  # below grid
        ratios_hi = shock_line_ratios(2000.0)  # above grid
        ratios_lo_edge = shock_line_ratios(100.0)
        ratios_hi_edge = shock_line_ratios(1000.0)

        assert float(ratios_lo["Halpha"]) == pytest.approx(
            float(ratios_lo_edge["Halpha"]), rel=1e-5
        )
        assert float(ratios_hi["Halpha"]) == pytest.approx(
            float(ratios_hi_edge["Halpha"]), rel=1e-5
        )

    def test_doublet_ratios(self):
        """Doublet components should have correct atomic physics ratios."""
        ratios = shock_line_ratios(300.0)
        # [OIII] 5007/4959 = 2.98
        oiii_ratio = float(ratios["OIII_5007"]) / float(ratios["OIII_4959"])
        assert oiii_ratio == pytest.approx(2.98, rel=1e-3)

        # [NII] 6583/6548 = 2.94
        nii_ratio = float(ratios["NII_6583"]) / float(ratios["NII_6548"])
        assert nii_ratio == pytest.approx(2.94, rel=1e-3)

    def test_sii_total_conserved(self):
        """Sum of SII doublet should equal the total from the grid."""
        ratios = shock_line_ratios(300.0)
        sii_total = float(ratios["SII_6716"]) + float(ratios["SII_6731"])
        # At 300 km/s, _R_SII = 2.0 from the grid
        assert sii_total == pytest.approx(2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# shock_emission_sed
# ---------------------------------------------------------------------------


class TestShockEmissionSed:
    """Tests for shock_emission_sed."""

    @pytest.fixture()
    def wavelength(self):
        """Standard wavelength grid for testing."""
        return jnp.linspace(3000.0, 8000.0, 5000)

    def test_output_shape(self, wavelength):
        """Output shape should match input wavelength grid."""
        sed = shock_emission_sed(wavelength, 300.0, 1e6)
        assert sed.shape == wavelength.shape

    def test_zero_luminosity_gives_zero_sed(self, wavelength):
        """L_shock_halpha=0 should give zero SED."""
        sed = shock_emission_sed(wavelength, 300.0, 0.0)
        assert jnp.allclose(sed, 0.0)

    def test_sed_non_negative(self, wavelength):
        """SED should be non-negative everywhere."""
        sed = shock_emission_sed(wavelength, 300.0, 1e6)
        assert jnp.all(sed >= 0.0)

    def test_sed_has_peaks_at_line_wavelengths(self, wavelength):
        """SED should have peaks near the expected emission line positions."""
        sed = shock_emission_sed(wavelength, 300.0, 1e8, line_sigma_aa=2.0)
        # Check Halpha at 6563 A
        ha_region = jnp.abs(wavelength - 6563.0) < 10.0
        assert jnp.max(sed[ha_region]) > jnp.median(sed[sed > 0]) * 10

    def test_delta_function_mode(self, wavelength):
        """Delta function mode (sigma=0) should produce sparse SED."""
        sed = shock_emission_sed(wavelength, 300.0, 1e6, line_sigma_aa=0.0)
        n_nonzero = jnp.sum(sed > 0)
        # Should have at most N_LINES nonzero pixels (some lines may
        # land on the same pixel)
        assert int(n_nonzero) <= _N_LINES

    def test_gaussian_mode_broader(self, wavelength):
        """Gaussian mode should spread flux over more pixels."""
        sed_narrow = shock_emission_sed(wavelength, 300.0, 1e6, line_sigma_aa=1.0)
        sed_broad = shock_emission_sed(wavelength, 300.0, 1e6, line_sigma_aa=5.0)
        n_narrow = jnp.sum(sed_narrow > 1e-30)
        n_broad = jnp.sum(sed_broad > 1e-30)
        assert int(n_broad) > int(n_narrow)

    def test_luminosity_scales_linearly(self, wavelength):
        """Doubling L_shock_halpha should double the SED."""
        sed1 = shock_emission_sed(wavelength, 300.0, 1e6)
        sed2 = shock_emission_sed(wavelength, 300.0, 2e6)
        ratio = sed2 / jnp.maximum(sed1, 1e-50)
        nonzero = sed1 > 1e-30
        assert jnp.allclose(ratio[nonzero], 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestShockJIT:
    """Verify JIT compilation works."""

    def test_line_ratios_jittable(self):
        """shock_line_ratios should be JIT-compatible."""

        @jax.jit
        def _get_halpha(v):
            ratios = shock_line_ratios(v)
            return ratios["Halpha"]

        val = _get_halpha(300.0)
        assert float(val) > 0.0

    def test_sed_jittable(self):
        """shock_emission_sed should be JIT-compatible."""
        wave = jnp.linspace(3000.0, 8000.0, 1000)

        @jax.jit
        def _compute(v, lum):
            return shock_emission_sed(wave, v, lum)

        sed = _compute(300.0, 1e6)
        assert sed.shape == wave.shape


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestShockDifferentiable:
    """Verify differentiability w.r.t. shock parameters."""

    def test_grad_wrt_velocity(self):
        """Gradient w.r.t. shock_velocity should be nonzero."""
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def _total_flux(v):
            sed = shock_emission_sed(wave, v, 1e6, line_sigma_aa=2.0)
            return jnp.sum(sed)

        grad_fn = jax.grad(_total_flux)
        g = grad_fn(300.0)
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_wrt_luminosity(self):
        """Gradient w.r.t. L_shock_halpha should be nonzero."""
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def _total_flux(lum):
            sed = shock_emission_sed(wave, 300.0, lum, line_sigma_aa=2.0)
            return jnp.sum(sed)

        grad_fn = jax.grad(_total_flux)
        g = grad_fn(1e6)
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ---------------------------------------------------------------------------
# Integration with ParamSpec
# ---------------------------------------------------------------------------


class TestShockParamSpec:
    """Test shock parameter registration in ParamSpec."""

    def test_shock_params_registered(self):
        """When shock=True, shock params should be in the spec."""
        from tengri.core.param_spec import ParamSpec

        spec = ParamSpec(shock=True)
        params = spec.all_params
        assert "shock_frac" in params
        assert "shock_velocity" in params
        assert "shock_log_density" in params

    def test_shock_params_absent_by_default(self):
        """When shock=False (default), shock params should not be in the spec."""
        from tengri.core.param_spec import ParamSpec

        spec = ParamSpec()
        params = spec.all_params
        assert "shock_frac" not in params
        assert "shock_velocity" not in params

    def test_shock_frac_zero_default(self):
        """Default shock_frac should be Fixed(0.0)."""
        from tengri.core.param_spec import ParamSpec

        spec = ParamSpec(shock=True)
        dist = spec.get_distribution("shock_frac")
        assert dist.value == pytest.approx(0.0)

    def test_shock_velocity_bounds(self):
        """Shock velocity should have bounds [100, 1000]."""
        from tengri.core.param_spec import ParamSpec

        # Valid range should work
        spec = ParamSpec(shock=True, shock_velocity=(100.0, 1000.0))
        assert "shock_velocity" in spec.free_params

        # Invalid range should raise
        with pytest.raises(ValueError):
            ParamSpec(shock=True, shock_velocity=(50.0, 1000.0))
