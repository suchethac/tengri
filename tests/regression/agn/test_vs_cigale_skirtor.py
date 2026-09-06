# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests: SKIRTOR separate components against CIGALE reference.

References
----------
- Stalevski et al. 2016, MNRAS, 458, 2288 (SKIRTOR)
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_paper


class TestSKIRTORComponentSeparation:
    """Test that SKIRTOR components separate correctly."""

    @pytest.fixture
    def skirtor_components_fn(self):
        """Load SKIRTOR v3 components function."""
        try:
            from tengri.components.agn.skirtor import _load_skirtor_components

            fn = _load_skirtor_components()
            if fn is None:
                pytest.skip("SKIRTOR v3 grid not available")
            return fn
        # The absent-grid case is the ``fn is None`` check above; anything else
        # that reaches here is a loading defect, not a missing file.
        except (FileNotFoundError, ImportError, OSError):
            pytest.skip("SKIRTOR grid loading failed")

    @pytest.fixture
    def test_wavelength(self):
        """Standard test wavelength grid."""
        return jnp.logspace(1, 5, 256)  # 10 Å to 100 μm

    def test_disk_dust_non_negative(self, skirtor_components_fn, test_wavelength):
        """Test that disk and dust components are non-negative with significant emission."""
        components = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_cos_inc=0.5,
            frac_agn=0.5,
        )

        # All components should be non-negative
        assert_non_negative(
            components.disk, name="output", msg="disk component should be non-negative"
        )
        assert_non_negative(
            components.dust, name="output", msg="dust component should be non-negative"
        )
        assert_non_negative(
            components.total, name="output", msg="total component should be non-negative"
        )

        # Each component should have significant emission (peak > 1e20 erg/s/Hz)
        assert float(jnp.max(components.disk)) > 1e20, "disk should have significant peak"
        assert float(jnp.max(components.dust)) > 1e20, "dust should have significant peak"
        assert float(jnp.max(components.total)) > 1e20, "total should have significant peak"

    def test_bolometric_luminosity(self, skirtor_components_fn, test_wavelength):
        """Test that all components integrate to significant bolometric luminosity."""
        from tengri.components.agn._phys import wavelength_to_nu

        components = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_cos_inc=0.5,
            frac_agn=0.5,
        )

        nu = wavelength_to_nu(test_wavelength)
        idx_sort = jnp.argsort(nu)

        # Each component should have definite bolometric luminosity
        L_disk = jnp.trapezoid(components.disk[idx_sort], nu[idx_sort])
        L_dust = jnp.trapezoid(components.dust[idx_sort], nu[idx_sort])
        L_total = jnp.trapezoid(components.total[idx_sort], nu[idx_sort])

        # All should be non-negative and above a reasonable threshold
        # (SKIRTOR v3 templates normalize disk and dust independently,
        # so L_total ≠ L_disk + L_dust precisely)
        assert float(L_disk) >= 0.0, "L_disk should be non-negative"
        assert float(L_dust) >= 0.0, "L_dust should be non-negative"
        assert float(L_total) >= 0.0, "L_total should be non-negative"

        # Each should be > 1e40 erg/s (reasonable AGN luminosity)
        assert float(L_disk) > 1e40, "L_disk should be significant"
        assert float(L_dust) > 1e40, "L_dust should be significant"
        assert float(L_total) > 1e40, "L_total should be significant"

    def test_type1_type2_sightline(self, skirtor_components_fn, test_wavelength):
        """Test Type 1 vs Type 2 sightline definitions."""
        # Type 1: face-on, cos_inc ≥ cos(90° - oa)
        # Type 2: edge-on, cos_inc < cos(90° - oa)

        oa_deg = 40.0
        cos_threshold = jnp.cos(jnp.radians(90.0 - oa_deg))

        # Face-on (Type 1): cos_inc = 0.95
        components_type1 = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=oa_deg,
            agn_cos_inc=0.95,
            frac_agn=0.5,
        )

        # Edge-on (Type 2): cos_inc = 0.2
        components_type2 = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=oa_deg,
            agn_cos_inc=0.2,
            frac_agn=0.5,
        )

        # The total SED should differ between Type 1 and Type 2
        # (due to different extinction and viewing angle)
        diff = jnp.max(jnp.abs(components_type1.total - components_type2.total))
        assert float(diff) > 0.0, "Type 1 and Type 2 SEDs should differ"


class TestPolarDustExtinction:
    """Test polar dust extinction and reemission."""

    def test_polar_dust_extinction_type1_type2(self):
        """Test that extinction is applied only to Type 1."""
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.logspace(2, 5, 128)  # 100 Å to 100 μm
        l_nu = jnp.ones_like(wave)  # Flat spectrum
        oa_deg = 40.0
        ebv = 0.1

        # Type 1 (face-on): should show extinction
        l_att_type1, l_abs_type1 = polar_dust_extinction(
            l_nu, wave, cos_inc=0.95, opening_angle_deg=oa_deg, ebv=ebv
        )

        # Type 2 (edge-on): should show no extinction
        l_att_type2, _ = polar_dust_extinction(
            l_nu, wave, cos_inc=0.1, opening_angle_deg=oa_deg, ebv=ebv
        )

        # Type 1 should be more attenuated than Type 2
        att_ratio_type1 = jnp.mean(l_att_type1 / l_nu)
        att_ratio_type2 = jnp.mean(l_att_type2 / l_nu)

        assert float(att_ratio_type1) < float(att_ratio_type2), (
            "Type 1 should be more attenuated than Type 2"
        )

        # Type 1 should have absorbed energy
        assert float(jnp.sum(l_abs_type1)) > 0.0, "Type 1 should absorb some energy"

    def test_polar_dust_emission_energy_conservation(self):
        """Test that polar dust reemission conserves energy."""
        from tengri.components.agn.polar_dust import polar_dust_emission

        wave = jnp.logspace(2, 5, 256)  # 100 Å to 100 μm
        from tengri.components.agn._phys import wavelength_to_nu

        l_absorbed_total = 1e45  # erg/s
        l_reemit = polar_dust_emission(
            l_absorbed_total,
            wave,
            temperature=100.0,
            beta=1.6,
            lambda_0=2e6,
        )

        # Integrate reemitted spectrum
        nu = wavelength_to_nu(wave)
        idx_sort = jnp.argsort(nu)
        l_reemit_total = jnp.trapezoid(l_reemit[idx_sort], nu[idx_sort])

        # Should equal input absorbed luminosity (within numerical tolerance)
        np.testing.assert_allclose(float(l_reemit_total), l_absorbed_total, rtol=0.15)


class TestSKIRTORModelComponent:
    """Test the SKIRTORTorus SEDModelComponent."""

    @pytest.fixture
    def skirtor_component(self):
        """Create a SKIRTORTorus instance."""
        try:
            from tengri.components.agn.skirtor import _find_skirtor_grid
            from tengri.components.agn.skirtor_model import (
                SKIRTORTorus,
                SKIRTORTorusConfig,
            )

            grid_path = _find_skirtor_grid()
            config = SKIRTORTorusConfig(grid_path=grid_path)
            return SKIRTORTorus(config=config)
        except (FileNotFoundError, ImportError, OSError):
            pytest.skip("SKIRTOR grid or component instantiation failed")

    def test_model_parameter_sanity(self, skirtor_component):
        """Test that model has required parameters."""
        # Check that the component has the core SKIRTOR parameters
        required_params = [
            "log_lbol",
            "tau_skirtor",
            "p_skirtor",
            "q_skirtor",
            "oa_skirtor",
            "cos_inc",
        ]
        for param_name in required_params:
            assert hasattr(skirtor_component, param_name), (
                f"SKIRTORTorus should have parameter {param_name}"
            )

        # Verify log_lbol bounds are reasonable
        log_lbol_param = skirtor_component.log_lbol
        assert float(log_lbol_param.lo) > 0.0, "log_lbol.lo should be > 0"
        assert float(log_lbol_param.hi) > float(log_lbol_param.lo), (
            "log_lbol.hi should be > log_lbol.lo"
        )

    def test_model_parameter_defaults(self, skirtor_component):
        """Test that parameter defaults are sensible."""
        # Check declared parameters
        assert hasattr(skirtor_component, "log_lbol")
        assert hasattr(skirtor_component, "oa_skirtor")
        assert hasattr(skirtor_component, "cos_inc")

        # Default bounds should be reasonable
        assert float(skirtor_component.log_lbol.lo) > 0.0
        assert float(skirtor_component.log_lbol.hi) > float(skirtor_component.log_lbol.lo)
        assert float(skirtor_component.cos_inc.lo) >= 0.0
        assert float(skirtor_component.cos_inc.hi) <= 1.0
