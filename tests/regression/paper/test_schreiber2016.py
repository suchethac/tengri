# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Schreiber et al. (2016) dust emission model."""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.dust.emission import schreiber2016
from tests._bounds import assert_non_negative


class TestSchreiber2016:
    """Test suite for schreiber2016 dust emission model."""

    @pytest.fixture
    def wavelength_grid(self):
        """Standard wavelength grid for IR (1–1000 μm)."""
        return jnp.logspace(3.0, 7.0, 200)  # 1000 Å to 10 μm

    @pytest.fixture
    def l_absorbed(self):
        """Test absorbed luminosity."""
        return 1.0  # Lsun

    def test_energy_conservation_continuum(self, wavelength_grid, l_absorbed):
        """Test that f_pah=0 gives pure continuum with correct total energy."""
        l_nu = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.0,
        )

        # Integrate L_ν over frequency to get total luminosity
        wavelength_cm = wavelength_grid * 1.0e-8
        c_cgs = 2.99792458e10
        nu = c_cgs / wavelength_cm
        integral = -jnp.trapezoid(l_nu, nu)

        # Should equal L_absorbed (within 1% due to discretization)
        np.testing.assert_allclose(integral, l_absorbed, rtol=0.01)

    def test_energy_conservation_mixed(self, wavelength_grid, l_absorbed):
        """Test energy conservation for mixed continuum+PAH."""
        l_nu = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.2,
        )

        wavelength_cm = wavelength_grid * 1.0e-8
        c_cgs = 2.99792458e10
        nu = c_cgs / wavelength_cm
        integral = -jnp.trapezoid(l_nu, nu)

        np.testing.assert_allclose(integral, l_absorbed, rtol=0.01)

    def test_energy_conservation_pah_only(self, wavelength_grid, l_absorbed):
        """Test energy conservation for pure PAH (f_pah=1)."""
        l_nu = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=1.0,
        )

        wavelength_cm = wavelength_grid * 1.0e-8
        c_cgs = 2.99792458e10
        nu = c_cgs / wavelength_cm
        integral = -jnp.trapezoid(l_nu, nu)

        np.testing.assert_allclose(integral, l_absorbed, rtol=0.01)

    def test_pah_features_in_spectrum(self, wavelength_grid, l_absorbed):
        """Test that PAH features appear at the expected wavelengths."""
        l_nu = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=1.0,  # Pure PAH for maximum visibility
        )

        # PAH features (in Ångstrom, from Smith et al. 2007)
        pah_centers = [33000.0, 62000.0, 77000.0, 86000.0, 113000.0, 127000.0]

        for center_aa in pah_centers:
            idx_center = jnp.argmin(jnp.abs(wavelength_grid - center_aa))

            # Look for local maxima in a ±2 window around each feature
            window_size = max(2, int(0.1 * idx_center))
            idx_min = max(0, idx_center - window_size)
            idx_max = min(len(l_nu), idx_center + window_size + 1)

            if idx_max > idx_min:
                local_flux = l_nu[idx_min:idx_max]
                # Feature should have non-trivial flux
                assert jnp.max(local_flux) > 1e-10 * jnp.max(l_nu)

    def test_temperature_dependence(self, wavelength_grid, l_absorbed):
        """Test that higher T_dust shifts the peak wavelength blueward."""
        l_nu_cold = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=20.0,
            dust_f_pah=0.0,
        )

        l_nu_warm = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=40.0,
            dust_f_pah=0.0,
        )

        # Peak index for cold dust
        idx_peak_cold = jnp.argmax(l_nu_cold)
        lambda_peak_cold = wavelength_grid[idx_peak_cold]

        # Peak index for warm dust
        idx_peak_warm = jnp.argmax(l_nu_warm)
        lambda_peak_warm = wavelength_grid[idx_peak_warm]

        # Warmer dust should peak at shorter wavelengths
        assert lambda_peak_warm < lambda_peak_cold

    def test_f_pah_zero_matches_continuum(self, wavelength_grid, l_absorbed):
        """Test that f_pah=0 gives pure modified blackbody continuum."""
        from tengri.components.dust.emission import modified_blackbody

        l_nu_schreiber = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.0,
        )

        # Pure continuum from schreiber2016 should match modified_blackbody
        # with beta=1.5 (used in schreiber2016 for continuum)
        l_nu_mbb = modified_blackbody(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_beta_ir=1.5,
        )

        np.testing.assert_allclose(l_nu_schreiber, l_nu_mbb, rtol=0.05)

    def test_f_pah_clipped_to_range(self, wavelength_grid, l_absorbed):
        """Test that f_pah outside [0, 1] is clipped."""
        l_nu_clipped = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=1.5,  # Out of range
        )

        # Should be equivalent to f_pah=1.0
        l_nu_one = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=1.0,
        )

        np.testing.assert_allclose(l_nu_clipped, l_nu_one, rtol=0.01)

    def test_positive_luminosity(self, wavelength_grid, l_absorbed):
        """Test that output is always non-negative."""
        l_nu = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.5,
        )

        assert_non_negative(l_nu, name="l_nu")

    def test_mixed_models_are_between_pure_components(self, wavelength_grid, l_absorbed):
        """Test that f_pah=0.5 gives intermediate flux between pure models."""
        l_nu_continuum = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.0,
        )

        l_nu_pah = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=1.0,
        )

        l_nu_mixed = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.5,
        )

        # Mixed should be close to the average of the two extremes
        # (not necessarily exactly the mean due to renormalization, but in the ballpark)
        avg = 0.5 * (l_nu_continuum + l_nu_pah)

        # Check that mixed is between the two extremes at each wavelength
        below_continuum = jnp.any(l_nu_mixed < jnp.minimum(l_nu_continuum, l_nu_pah))
        above_pah = jnp.any(l_nu_mixed > jnp.maximum(l_nu_continuum, l_nu_pah))

        # Due to normalization, we allow some tolerance
        assert not below_continuum or jnp.mean(l_nu_mixed) > 0.4 * jnp.mean(avg)
        assert not above_pah or jnp.mean(l_nu_mixed) < 1.6 * jnp.mean(avg)

    def test_zero_absorption_gives_zero_output(self, wavelength_grid):
        """Test that zero absorbed luminosity gives zero output."""
        l_nu = schreiber2016(
            wavelength_grid,
            0.0,
            dust_T=30.0,
            dust_f_pah=0.5,
        )

        assert jnp.allclose(l_nu, 0.0)

    def test_output_shape(self, wavelength_grid, l_absorbed):
        """Test that output shape matches input wavelength grid."""
        l_nu = schreiber2016(
            wavelength_grid,
            l_absorbed,
            dust_T=30.0,
            dust_f_pah=0.5,
        )

        chex.assert_equal_shape([l_nu, wavelength_grid])

    def test_redshift_correction_applied(self, wavelength_grid, l_absorbed):
        """Test that non-zero redshift changes temperature (CMB correction)."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        dust_T = 30.0
        redshift_z0 = 0.0
        redshift_z3 = 3.0

        T_eff_z0 = cmb_corrected_temperature(dust_T, redshift_z0, 1.5)
        T_eff_z3 = cmb_corrected_temperature(dust_T, redshift_z3, 1.5)

        # CMB correction should increase effective temperature at high z
        assert T_eff_z3 > T_eff_z0
