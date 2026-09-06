# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for Lyman-continuum dust-absorption fraction (neb_fdust).

Tests the lyc_dust_escape_factor function and its integration with the
Cue nebular backend to verify the CIGALE-matched implementation.

References
----------
.. [1] Inoue, A. K., et al. 2011, "Dust Attenuation toward Star-forming
    Galaxies at z ~ 2", MNRAS, 411, 2336.
.. [2] CIGALE nebular module: pcigale/sed_modules/nebular.py, lines 94,
    156–162.

Markers
-------
- `@pytest.mark.contract` — Cross-component contract verification
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import grad

from tengri.components.nebular._recombination_coeffs import (
    ALPHA_1,
    ALPHA_B,
    lyc_dust_escape_factor,
)

pytestmark = pytest.mark.contract


class TestLycDustEscapeFactor:
    """Tests for the CIGALE k-factor (ionizing photon loss scaling)."""

    def test_zero_loss_returns_unity(self):
        """k-factor with f_esc=0, f_dust=0 should return 1.0."""
        k = lyc_dust_escape_factor(0.0, 0.0)
        assert jnp.allclose(k, 1.0, atol=1e-6)

    def test_escape_only_reduces_emission(self):
        """k-factor with non-zero f_esc (f_dust=0) should reduce emission."""
        k_esc30 = lyc_dust_escape_factor(0.3, 0.0)
        # CIGALE: k = (1 - f_esc) / (1 + (alpha_1/alpha_B) * f_esc)
        # With alpha_ratio ≈ 0.5969: k ≈ 0.7 / (1 + 0.5969*0.3) ≈ 0.5937
        alpha_ratio = ALPHA_1 / ALPHA_B
        expected = (1.0 - 0.3) / (1.0 + alpha_ratio * 0.3)
        assert jnp.allclose(k_esc30, expected, atol=1e-6)
        # Verify it's less than 1
        assert k_esc30 < 1.0

    def test_dust_only_reduces_emission(self):
        """k-factor with non-zero f_dust (f_esc=0) should reduce emission."""
        k_dust20 = lyc_dust_escape_factor(0.0, 0.2)
        # Same formula with f_esc=0, f_dust=0.2
        alpha_ratio = ALPHA_1 / ALPHA_B
        expected = (1.0 - 0.2) / (1.0 + alpha_ratio * 0.2)
        assert jnp.allclose(k_dust20, expected, atol=1e-6)
        assert k_dust20 < 1.0

    def test_escape_and_dust_combine_additively(self):
        """k-factor with both f_esc and f_dust uses combined total."""
        f_esc = 0.2
        f_dust = 0.1
        k = lyc_dust_escape_factor(f_esc, f_dust)
        # f_total = f_esc + f_dust = 0.3
        # k = (1 - 0.3) / (1 + alpha_ratio * 0.3)
        alpha_ratio = ALPHA_1 / ALPHA_B
        expected = (1.0 - (f_esc + f_dust)) / (1.0 + alpha_ratio * (f_esc + f_dust))
        assert jnp.allclose(k, expected, atol=1e-6)

    def test_high_combined_loss_approaches_zero(self):
        """k-factor with f_esc + f_dust → 1 should approach 0."""
        k_high = lyc_dust_escape_factor(0.5, 0.49)
        # f_total = 0.99 (very close to 1, but clamped to <1)
        # Should be very small
        assert 0.0 <= k_high < 0.1
        assert jnp.isfinite(k_high)

    def test_gradient_finite_at_zero(self):
        """Gradient of k-factor should be finite at f_esc=f_dust=0."""
        grad_fn = grad(lambda x: lyc_dust_escape_factor(x, 0.0))
        grad_at_zero = grad_fn(0.0)
        assert jnp.isfinite(grad_at_zero)
        assert jnp.any(grad_at_zero != 0.0), (
            "`grad_at_zero` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_gradient_finite_away_from_boundary(self):
        """Gradient should be finite away from the photon-loss boundary."""
        grad_fn = grad(lambda x: lyc_dust_escape_factor(x, 0.1))
        grad_at_point = grad_fn(0.2)
        assert jnp.isfinite(grad_at_point)
        assert jnp.any(grad_at_point != 0.0), (
            "`grad_at_point` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_vectorized_over_arrays(self):
        """k-factor should work with array inputs."""
        f_esc_arr = jnp.array([0.0, 0.1, 0.2, 0.3])
        f_dust_arr = jnp.array([0.0, 0.05, 0.1, 0.15])
        k_arr = lyc_dust_escape_factor(f_esc_arr, f_dust_arr)
        # Verify shape and all values are in (0, 1]
        assert k_arr.shape == f_esc_arr.shape
        assert jnp.all(k_arr > 0.0)
        assert jnp.all(k_arr <= 1.0)

    def test_consistency_with_cigale_formula(self):
        """Verify exact match with CIGALE nebular.py line 156-162."""
        # CIGALE formula (pcigale/sed_modules/nebular.py, lines 156-162):
        # k = (1.0 - f_esc - f_dust) / (1.0 + (alpha_1 / alpha_B) * (f_esc + f_dust))
        test_cases = [
            (0.0, 0.0),  # No loss
            (0.1, 0.0),  # Escape only
            (0.0, 0.1),  # Dust only
            (0.15, 0.05),  # Mixed
            (0.3, 0.2),  # Higher loss
        ]
        for f_esc, f_dust in test_cases:
            k = lyc_dust_escape_factor(f_esc, f_dust)
            alpha_ratio = ALPHA_1 / ALPHA_B
            f_total = f_esc + f_dust
            expected = (1.0 - f_total) / (1.0 + alpha_ratio * f_total)
            assert jnp.allclose(k, expected, atol=1e-6), (
                f"Mismatch at f_esc={f_esc}, f_dust={f_dust}: got {k}, expected {expected}"
            )


class TestNebularFdustIntegration:
    """Integration tests for neb_fdust parameter in nebular models."""

    def test_constants_match_cigale(self):
        """Verify recombination constants against CIGALE values."""
        # CIGALE nebular.py line 94: alpha_1 = 1.54e-19, alpha_B = 2.58e-19
        assert jnp.isclose(ALPHA_1, 1.54e-19)
        assert jnp.isclose(ALPHA_B, 2.58e-19)

    def test_alpha_ratio_near_cigale_convention(self):
        """Verify alpha_1 / alpha_B ≈ 0.597 as per CIGALE."""
        ratio = ALPHA_1 / ALPHA_B
        # Expected: 1.54 / 2.58 ≈ 0.5969
        assert jnp.isclose(ratio, 0.5969, atol=0.001)

    @pytest.mark.contract
    def test_cloudy_grid_fdust_reduces_lines(self):
        """Verify that neb_fdust > 0 reduces CloudyGrid line luminosity.

        The k-factor scales nebular lines. With fdust > 0, the k-factor is
        smaller than with fdust = 0, so luminosities should decrease.
        """
        pytest.importorskip("h5py")
        from pathlib import Path

        from tengri.components.nebular.cloudy_grid import CloudyGridBackend

        # Use synthetic test grid if available
        data_dir = Path(__file__).parents[2] / "data"
        grid_path = data_dir / "test_cloudy_grid.h5"

        if not grid_path.exists():
            pytest.skip("Test CLOUDY grid not available")

        try:
            from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

            ssp_data = load_ssp_data(data_dir / "test_ssp.h5")
        except Exception:
            pytest.skip("Test SSP data not available")

        backend = CloudyGridBackend(str(grid_path), ssp_data=ssp_data)

        # Mock SSP inputs
        ssp_weights = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])  # One young age bin
        ssp_log_ages_yr = jnp.array([6.0, 7.0, 7.5, 8.0, 9.0])  # log10(yr)
        log_z = -1.848  # Solar

        # Predict with fdust = 0
        waves_0, lums_0 = backend.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            neb_fdust=0.0,
        )

        # Predict with fdust = 0.1
        waves_dust, lums_dust = backend.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            neb_fdust=0.1,
        )

        # Check wavelengths are identical
        assert jnp.allclose(waves_0, waves_dust)

        # Check luminosities are reduced by fdust
        # k(0, 0.1) < k(0, 0) so lums_dust < lums_0
        assert jnp.all(lums_dust < lums_0), (
            f"Expected fdust=0.1 to reduce lines, but got "
            f"min(lums_0)={jnp.min(lums_0)}, min(lums_dust)={jnp.min(lums_dust)}"
        )

    @pytest.mark.contract
    def test_cb19_fdust_reduces_lines(self):
        """Verify that neb_fdust > 0 reduces CB19 line luminosity."""
        pytest.importorskip("h5py")
        from pathlib import Path

        from tengri.components.nebular.cloudy_cb19 import CB19Backend

        # Use test CB19 grid if available
        data_dir = Path(__file__).parents[2] / "data"
        grid_path = data_dir / "cb19_templates.h5"

        if not grid_path.exists():
            pytest.skip("CB19 grid not available")

        try:
            from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

            ssp_data = load_ssp_data(data_dir / "test_ssp.h5")
        except Exception:
            pytest.skip("Test SSP data not available")

        backend = CB19Backend(grid_path=str(grid_path), ssp_data=ssp_data)

        # Mock SSP inputs
        ssp_weights = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])  # One young age bin
        ssp_log_ages_yr = jnp.array([6.0, 7.0, 7.5, 8.0, 9.0])  # log10(yr)
        log_z = -1.848  # Solar

        # Predict with fdust = 0
        waves_0, lums_0 = backend.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            neb_fdust=0.0,
        )

        # Predict with fdust = 0.1
        waves_dust, lums_dust = backend.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            neb_fdust=0.1,
        )

        # Check wavelengths are identical
        assert jnp.allclose(waves_0, waves_dust)

        # Check luminosities are reduced by fdust
        assert jnp.all(lums_dust < lums_0), (
            f"Expected fdust=0.1 to reduce CB19 lines, but got "
            f"min(lums_0)={jnp.min(lums_0)}, min(lums_dust)={jnp.min(lums_dust)}"
        )
