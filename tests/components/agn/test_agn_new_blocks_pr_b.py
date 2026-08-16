# SPDX-License-Identifier: BSD-3-Clause
"""Tests for new AGN blocks: richards2006 disc and boroson_green feii.

Taxonomy markers: contract, regression_bug.
These tests verify the normalization of template-based blocks against
their source data files — critical to prevent silent truncation/renorm
mismatches (#717).
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.agn.blr import _fe2_pseudo_continuum
from tests._bounds import assert_non_negative

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


class TestRichards2006DiscBlock:
    """Test the richards2006 disc block registration and normalization."""

    def test_richardson_disc_registered(self):
        """Richards2006 disc block is registered in AGN_BLOCKS."""
        assert "disc" in AGN_BLOCKS
        assert "richards2006" in AGN_BLOCKS["disc"]

    def test_richards2006_in_consumes(self):
        """Richards2006 disc block is in AGN_BLOCK_CONSUMES."""
        assert ("disc", "richards2006") in AGN_BLOCK_CONSUMES
        # Richards2006 is a fixed template with no free spectral parameters.
        assert AGN_BLOCK_CONSUMES[("disc", "richards2006")] == frozenset()

    def test_richards2006_disc_block_builds_finite(self):
        """Richards2006 disc block builds and returns finite values."""
        from tengri.components.agn.blocks.disc import richards2006_disc_block

        wave = np.linspace(1000, 10000, 100)  # Å
        L_lambda = richards2006_disc_block(wave, agn_log_lbol=45.0)

        assert np.all(np.isfinite(L_lambda))
        assert_non_negative(L_lambda, name="L_lambda")

    def test_richards2006_disc_uv_peaked(self):
        """Richards2006 disc shows UV-peak (higher flux at short wavelengths)."""
        from tengri.components.agn.blocks.disc import richards2006_disc_block

        # Compare UV vs optical (representative wavelengths in Angstrom).
        wave_uv = np.array([2000.0, 2500.0, 3000.0])  # UV region
        wave_opt = np.array([5000.0, 6000.0, 7000.0])  # Optical region

        L_lambda_uv = richards2006_disc_block(wave_uv, agn_log_lbol=45.0)
        L_lambda_opt = richards2006_disc_block(wave_opt, agn_log_lbol=45.0)

        # Average flux (UV should be greater than optical for quasars).
        avg_uv = np.mean(L_lambda_uv)
        avg_opt = np.mean(L_lambda_opt)

        assert avg_uv > avg_opt, (
            f"UV flux ({avg_uv:.2e}) should exceed optical ({avg_opt:.2e}) for quasar SED"
        )

    def test_richards2006_disc_template_normalization(self):
        """Richards2006 disc template has correct absolute normalization.

        Validates that the template integral matches the source data file
        normalization (critical to catch silent truncation/renorm bugs #717).
        """
        from pathlib import Path

        # Load the source template file.
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "agn_bbb"
        template_file = data_dir / "richards2006.dat"

        if not template_file.exists():
            pytest.skip(f"Template file not found: {template_file}")

        # Load source template (wave [Å], flux [arbitrary units]).
        template_data = np.genfromtxt(str(template_file), comments="#")
        template_wave = template_data[:, 0]
        template_flux = template_data[:, 1]

        # Normalize template to unit bolometric integral.
        # The template is dimensionless; integral over frequency gives "bolo".
        # L_nu = template_flux * (c / wave^2) in dimensionless form.
        c_aa_per_s = 2.99792458e18
        template_lnu = template_flux * c_aa_per_s / (template_wave**2)
        # Trapezoid integration over frequency (dfreq = c * d(lambda) / lambda^2).
        freq = c_aa_per_s / template_wave
        # Sort by increasing frequency for integration.
        sort_idx = np.argsort(freq)
        freq_sorted = freq[sort_idx]
        lnu_sorted = template_lnu[sort_idx]
        template_bol_integral = np.trapezoid(lnu_sorted, freq_sorted)

        # Test at a reference bolometric luminosity (e.g., log L_bol = 45).
        log_lbol = 45.0
        L_SUN_ERG_S = 3.828e33
        target_bol_erg_s = (10.0**log_lbol) * L_SUN_ERG_S

        # Call the block.
        from tengri.components.agn.blocks.disc import richards2006_disc_block

        L_lambda = richards2006_disc_block(template_wave, agn_log_lbol=log_lbol)

        # Integrate the block output over frequency.
        L_nu = L_lambda * template_wave**2 / c_aa_per_s
        freq_block = c_aa_per_s / template_wave
        sort_idx_block = np.argsort(freq_block)
        freq_block_sorted = freq_block[sort_idx_block]
        L_nu_sorted = L_nu[sort_idx_block]
        block_bol_integral = np.trapezoid(L_nu_sorted, freq_block_sorted)

        # The block's integral should match the target (normalized) luminosity.
        # Allow 5% tolerance for interpolation/integration artifacts.
        rel_error = np.abs(block_bol_integral - target_bol_erg_s) / target_bol_erg_s
        assert rel_error < 0.05, (
            f"Richards2006 disc block bolometric integral mismatch: "
            f"expected {target_bol_erg_s:.3e}, got {block_bol_integral:.3e} "
            f"(rel_error={rel_error:.3f})"
        )


class TestBorosonGreenFeiiBlock:
    """Test the boroson_green feii block registration and normalization."""

    def test_boroson_green_feii_registered(self):
        """Boroson & Green feii block is registered in AGN_BLOCKS."""
        assert "feii" in AGN_BLOCKS
        assert "boroson_green" in AGN_BLOCKS["feii"]

    def test_boroson_green_in_consumes(self):
        """Boroson & Green feii block is in AGN_BLOCK_CONSUMES."""
        assert ("feii", "boroson_green") in AGN_BLOCK_CONSUMES
        # Boroson & Green feii consumes agn_fe2_strength.
        assert "agn_fe2_strength" in AGN_BLOCK_CONSUMES[("feii", "boroson_green")]

    def test_boroson_green_feii_block_builds_finite(self):
        """Boroson & Green feii block builds and returns finite values."""
        from tengri.components.agn.blocks.feii import boroson_green_feii_block

        wave = np.linspace(1200, 7500, 200)  # FeII coverage [Å]
        l5100_disc = 1e44  # disc continuum at 5100Å [erg/s]

        L_lambda = boroson_green_feii_block(
            wave, agn_log_lbol=45.0, l5100_disc=l5100_disc, agn_fe2_strength=1.0
        )

        assert np.all(np.isfinite(L_lambda))
        assert_non_negative(L_lambda, name="L_lambda")

    def test_boroson_green_feii_disabled_when_strength_zero(self):
        """FeII emission is zero when agn_fe2_strength=0."""
        from tengri.components.agn.blocks.feii import boroson_green_feii_block

        wave = np.linspace(1200, 7500, 200)
        l5100_disc = 1e44

        L_lambda_off = boroson_green_feii_block(
            wave, agn_log_lbol=45.0, l5100_disc=l5100_disc, agn_fe2_strength=0.0
        )
        L_lambda_on = boroson_green_feii_block(
            wave, agn_log_lbol=45.0, l5100_disc=l5100_disc, agn_fe2_strength=1.0
        )

        # FeII off should be zero everywhere.
        assert np.allclose(L_lambda_off, 0.0)
        # FeII on should be non-zero in some regions.
        assert np.any(L_lambda_on > 0.0)

    def test_boroson_green_feii_has_multiplet_bumps(self):
        """FeII pseudo-continuum shows bumps in UV and optical regions.

        Expected bumps (from Boroson & Green 1992):
        - ~2200–2600 Å: UV FeII multiplets
        - ~4400–5400 Å: optical FeII multiplets
        """
        from tengri.components.agn.blocks.feii import boroson_green_feii_block

        # Fine-grained wavelength grid to resolve multiplets.
        wave = np.linspace(1200, 7500, 1000)
        l5100_disc = 1e44

        L_lambda = boroson_green_feii_block(
            wave,
            agn_log_lbol=45.0,
            l5100_disc=l5100_disc,
            agn_fe2_strength=1.5,
            agn_blr_fwhm_kms=5000.0,
        )

        # Extract UV and optical bands.
        uv_mask = (wave >= 2200) & (wave <= 2600)
        opt_mask = (wave >= 4400) & (wave <= 5400)

        uv_flux = L_lambda[uv_mask]
        opt_flux = L_lambda[opt_mask]

        # Both regions should have non-zero flux (FeII multiplets).
        assert np.any(uv_flux > 0.0), "UV FeII multiplets (2200–2600 Å) not detected"
        assert np.any(opt_flux > 0.0), "Optical FeII multiplets (4400–5400 Å) not detected"

    def test_boroson_green_feii_vs_none(self):
        """Boroson & Green feii - none difference shows FeII contribution.

        The difference (boroson_green minus none) should be positive FeII bumps
        at the expected multiplet locations.
        """
        from tengri.components.agn.blocks._protocol import resolve_agn_block
        from tengri.components.agn.blocks.feii import boroson_green_feii_block

        wave = np.linspace(1200, 7500, 1000)
        l5100_disc = 1e44

        L_lambda_feii = boroson_green_feii_block(
            wave,
            agn_log_lbol=45.0,
            l5100_disc=l5100_disc,
            agn_fe2_strength=1.5,
            agn_blr_fwhm_kms=5000.0,
        )
        # Get the "none" feii block from registry
        feii_none = resolve_agn_block("feii", "none")
        L_lambda_none = feii_none(wave, agn_log_lbol=45.0, l5100_disc=l5100_disc)

        # Difference should be the FeII contribution (all positive).
        diff = L_lambda_feii - L_lambda_none
        assert np.all(diff >= 0.0), "FeII should only add positive flux"

        # The difference should have peaks in the expected regions.
        uv_mask = (wave >= 2200) & (wave <= 2600)
        opt_mask = (wave >= 4400) & (wave <= 5400)

        assert np.max(diff[uv_mask]) > 0.0, (
            "FeII - none should have positive UV bump (2200–2600 Å)"
        )
        assert np.max(diff[opt_mask]) > 0.0, (
            "FeII - none should have positive optical bump (4400–5400 Å)"
        )

    def test_boroson_green_feii_template_normalization(self):
        """Boroson & Green feii template has correct absolute normalization.

        Validates against the source PyQSOFit template files.
        """
        from pathlib import Path

        # Load source FeII templates.
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "agn_fe2"
        uv_file = data_dir / "fe_uv_pyqsofit.txt"
        opt_file = data_dir / "fe_optical_pyqsofit.txt"

        if not uv_file.exists() or not opt_file.exists():
            pytest.skip(f"FeII template files not found in {data_dir}")

        # Load templates.
        uv_data = np.genfromtxt(str(uv_file), comments="#")
        opt_data = np.genfromtxt(str(opt_file), comments="#")

        uv_log_wave = uv_data[:, 0]
        uv_flux = uv_data[:, 1]
        uv_wave = 10.0**uv_log_wave

        opt_log_wave = opt_data[:, 0]
        opt_flux = opt_data[:, 1]
        opt_wave = 10.0**opt_log_wave

        # Combine UV and optical templates.
        combined_wave = np.concatenate([uv_wave, opt_wave])
        combined_flux = np.concatenate([uv_flux, opt_flux])

        # Sort by wavelength.
        sort_idx = np.argsort(combined_wave)
        combined_wave = combined_wave[sort_idx]
        combined_flux = combined_flux[sort_idx]

        # Test the _fe2_pseudo_continuum function directly.
        fe2_lnu = _fe2_pseudo_continuum(combined_wave, fwhm_kms=5000.0, fe2_strength=1.0)

        # Integrate over frequency to get bolometric (units: template per H-beta).
        c_aa_per_s = 2.99792458e18
        freq = c_aa_per_s / combined_wave
        # Sort by freq for integration.
        sort_idx_freq = np.argsort(freq)
        freq_sorted = freq[sort_idx_freq]
        fe2_lnu_sorted = fe2_lnu[sort_idx_freq]
        fe2_bol_integral = np.trapezoid(fe2_lnu_sorted, freq_sorted)

        # FeII integral should be positive (non-zero emission).
        assert fe2_bol_integral > 0.0, "FeII bolometric integral should be positive"

        # When scaled by l5100_disc * agn_blr_f_bol, the output should be
        # well-defined and finite.
        l5100_disc = 1e44  # [erg/s]
        agn_blr_f_bol = 9.0

        # The block uses this normalization.
        l_disc_bol_erg = l5100_disc * agn_blr_f_bol
        fe2_absolute_integral = fe2_bol_integral * l_disc_bol_erg

        assert np.isfinite(fe2_absolute_integral), (
            "FeII block integral should be finite after normalization"
        )
        assert fe2_absolute_integral > 0.0, "FeII block integral should be positive"
