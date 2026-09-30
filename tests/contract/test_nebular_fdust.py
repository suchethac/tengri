# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the Lyman-continuum budget split (neb_fdust_frac, #2436).

Tests the lyc_dust_escape_factor and lyc_shares functions and their
integration with the Cue/CloudyGrid/CB19 nebular backends to verify the
CIGALE-matched implementation and the #2436 additive-shares reparametrization
(neb_fdust retired in favor of neb_fdust_frac, the fraction of the
NON-escaping budget HII-region dust absorbs).

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

from tengri.components.lyc import lyc_shares
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
    """Integration tests for the neb_fdust_frac parameter in nebular models."""

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
        """Verify that neb_fdust_frac > 0 reduces CloudyGrid line luminosity.

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
        except (FileNotFoundError, ImportError, OSError):
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
            neb_fdust_frac=0.0,
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
            neb_fdust_frac=0.1,
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
        """Verify that neb_fdust_frac > 0 reduces CB19 line luminosity."""
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
        except (FileNotFoundError, ImportError, OSError):
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
            neb_fdust_frac=0.0,
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
            neb_fdust_frac=0.1,
        )

        # Check wavelengths are identical
        assert jnp.allclose(waves_0, waves_dust)

        # Check luminosities are reduced by fdust
        assert jnp.all(lums_dust < lums_0), (
            f"Expected fdust=0.1 to reduce CB19 lines, but got "
            f"min(lums_0)={jnp.min(lums_0)}, min(lums_dust)={jnp.min(lums_dust)}"
        )


class TestLycShares:
    """Tests for the #2436 additive-shares split (lyc_shares)."""

    def test_zero_zero_is_all_gas(self):
        """neb_fesc=0, neb_fdust_frac=0 -> all photons photoionize."""
        f_esc, f_dust, f_gas = lyc_shares(0.0, 0.0)
        assert jnp.allclose(f_esc, 0.0)
        assert jnp.allclose(f_dust, 0.0)
        assert jnp.allclose(f_gas, 1.0)

    def test_shares_nonnegative_and_sum_to_one_over_prior_box(self):
        """The three shares are nonnegative and sum to 1 over [0, 1]^2.

        Owner ruling (#2436): f_esc + f_dust <= 1 is a precondition of the
        additive per-photon budget, so the whole (neb_fesc, neb_fdust_frac)
        prior box must be physical -- unlike the retired absolute neb_fdust,
        which let neb_fesc + neb_fdust exceed 1.
        """
        fesc_grid = jnp.linspace(0.0, 1.0, 21)
        frac_grid = jnp.linspace(0.0, 1.0, 21)
        fesc_mesh, frac_mesh = jnp.meshgrid(fesc_grid, frac_grid)
        f_esc, f_dust, f_gas = lyc_shares(fesc_mesh, frac_mesh)

        assert jnp.all(f_esc >= -1e-12), f"f_esc has negative entries: min={f_esc.min()}"
        assert jnp.all(f_dust >= -1e-12), f"f_dust has negative entries: min={f_dust.min()}"
        assert jnp.all(f_gas >= -1e-12), f"f_gas has negative entries: min={f_gas.min()}"

        total = f_esc + f_dust + f_gas
        assert jnp.allclose(total, 1.0, atol=1e-10), (
            f"shares do not sum to 1 everywhere on the prior box: "
            f"min={total.min()}, max={total.max()}"
        )

    def test_dust_share_scales_nonescaping_budget(self):
        """f_dust = neb_fdust_frac * (1 - neb_fesc), not an independent absolute."""
        f_esc, f_dust, f_gas = lyc_shares(0.3, 0.5)
        assert jnp.allclose(f_esc, 0.3)
        assert jnp.allclose(f_dust, 0.5 * 0.7)
        assert jnp.allclose(f_gas, 0.5 * 0.7)

    def test_gradient_wrt_fdust_frac_finite_and_nonzero(self):
        """d(f_dust)/d(neb_fdust_frac) is finite and nonzero on the open box."""
        grad_fn = grad(lambda frac: lyc_shares(0.3, frac)[1])
        g = grad_fn(0.4)
        assert jnp.isfinite(g)
        assert g != 0.0, "gradient wrt neb_fdust_frac collapsed to zero"

    def test_gradient_wrt_fesc_finite_and_nonzero(self):
        """d(f_dust)/d(neb_fesc) is finite and nonzero on the open box."""
        grad_fn = grad(lambda fesc: lyc_shares(fesc, 0.4)[1])
        g = grad_fn(0.3)
        assert jnp.isfinite(g)
        assert g != 0.0, "gradient wrt neb_fesc collapsed to zero"


class TestNebFdustRetired:
    """The retired absolute neb_fdust must raise a rename-hint error (#2436)."""

    @staticmethod
    def _parse(**groups):
        from tengri.parameters import DEFAULT, Fixed, parse_groups

        return parse_groups(
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.5),
            **groups,
        )

    def test_neb_fdust_in_neb_dict_raises_with_new_name(self):
        """neb={'neb_fdust': ...} raises naming neb_fdust_frac."""
        from tengri.parameters import DEFAULT, Fixed

        with pytest.raises(ValueError, match=r"neb_fdust.*neb_fdust_frac.*2436"):
            self._parse(neb={"type": "cue", "neb_fdust": Fixed(0.2), "all_params": Fixed(DEFAULT)})

    def test_neb_fdust_frac_survives(self):
        """The surviving neb_fdust_frac name works for model construction."""
        from tengri.parameters import DEFAULT, Fixed

        groups = self._parse(
            neb={"type": "cue", "neb_fdust_frac": Fixed(0.2), "all_params": Fixed(DEFAULT)},
        )
        assert groups is not None
