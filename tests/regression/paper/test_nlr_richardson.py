# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for Richardson+2014 NLR AGN emission-line template."""

import chex
import pytest

pytestmark = pytest.mark.regression_paper
import jax
import jax.numpy as jnp

from tengri.components.agn.nlr import compute_nlr_sed_richardson2014
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


class TestRichardsonNLR:
    """Test suite for compute_nlr_sed_richardson2014 function."""

    def test_output_shape(self):
        """Output shape matches input wavelength array."""
        wave = jnp.linspace(3000, 10000, 200)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        chex.assert_shape(sed, (200,))

    def test_non_negative(self):
        """All output values are non-negative."""
        wave = jnp.linspace(3000, 10000, 200)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        assert_non_negative(sed, name="sed")

    def test_finite(self):
        """All output values are finite (no NaN or Inf)."""
        wave = jnp.linspace(3000, 10000, 200)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        chex.assert_tree_all_finite(sed)

    def test_oiii_dominates(self):
        """[O III] 5007 is the strongest line (8.53x Hbeta)."""
        wave = jnp.linspace(4800, 5100, 500)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        peak_wave = wave[jnp.argmax(sed)]
        # Peak should be near 5007 Angstrom (within 50 Angstrom)
        assert abs(float(peak_wave) - 5007.0) < 50.0

    def test_zero_covering_fraction(self):
        """With zero covering fraction, all output is zero."""
        wave = jnp.linspace(3000, 10000, 200)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, covering_fraction=0.0)
        assert jnp.all(sed == 0.0)

    def test_jit_compatible(self):
        """Function is JIT-compilable."""
        wave = jnp.linspace(3000, 10000, 200)
        sed = assert_jit_matches_eager(compute_nlr_sed_richardson2014, wave, l_disc_bol_erg=1e44)
        chex.assert_tree_all_finite(sed)

    def test_scales_with_luminosity(self):
        """Output scales linearly with disc bolometric luminosity."""
        wave = jnp.linspace(3000, 10000, 200)
        sed1 = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        sed2 = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=2e44)
        assert jnp.allclose(sed2, 2.0 * sed1, rtol=1e-5)

    def test_scales_with_covering_fraction(self):
        """Output scales linearly with covering fraction."""
        wave = jnp.linspace(3000, 10000, 200)
        sed1 = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, covering_fraction=0.1)
        sed2 = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, covering_fraction=0.2)
        assert jnp.allclose(sed2, 2.0 * sed1, rtol=1e-5)

    def test_multiple_peaks(self):
        """Spectrum has multiple peaks (one per emission line)."""
        wave = jnp.linspace(3600, 7200, 2000)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        # Compute finite differences to find peaks
        diffs = jnp.diff(sed)
        sign_changes = jnp.diff(jnp.sign(diffs))
        n_peaks = jnp.sum(sign_changes < 0)  # Local maxima
        # Should have roughly 20-23 peaks (one per line)
        # Allow for some overlap or numerical issues
        assert n_peaks >= 15

    def test_hbeta_normalizes_correctly(self):
        """H-beta line is present in template (index 59, 4862.76 Angstrom)."""
        # The Richardson template includes H-beta (Balmer-beta at 4862.76 A)
        # with flux = 1.0 (normalized)
        wave = jnp.linspace(4800, 4900, 300)
        sed = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44)
        peak_wave = wave[jnp.argmax(sed)]
        # Peak should be near H-beta at 4862.76 Angstrom
        assert abs(float(peak_wave) - 4862.76) < 30.0

    def test_different_fwhm(self):
        """Different FWHM produces different line profiles."""
        wave = jnp.linspace(3000, 10000, 500)
        sed_narrow = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, fwhm_kms=300)
        sed_broad = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, fwhm_kms=1000)
        # Broad lines should have lower peak heights (flux spread over wider range)
        # but same integrated flux
        peak_narrow = jnp.max(sed_narrow)
        peak_broad = jnp.max(sed_broad)
        assert peak_narrow > peak_broad

    def test_line_efficiency(self):
        """Output scales with line efficiency parameter."""
        wave = jnp.linspace(3000, 10000, 200)
        sed1 = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, line_efficiency=0.05)
        sed2 = compute_nlr_sed_richardson2014(wave, l_disc_bol_erg=1e44, line_efficiency=0.10)
        assert jnp.allclose(sed2, 2.0 * sed1, rtol=1e-5)

    def test_vmap_compatibility(self):
        """Function can be vmapped over wavelength arrays."""
        waves = jnp.linspace(3000, 10000, 200)
        l_bolometric = jnp.array([1e44, 2e44, 5e44])

        def compute_one_sed(l_bol):
            return compute_nlr_sed_richardson2014(waves, l_disc_bol_erg=l_bol)

        seds = jax.vmap(compute_one_sed)(l_bolometric)
        chex.assert_shape(seds, (3, 200))
        chex.assert_tree_all_finite(seds)
        # Check that flux scales with luminosity
        assert jnp.allclose(seds[1], 2.0 * seds[0], rtol=1e-5)
        assert jnp.allclose(seds[2], 5.0 * seds[0], rtol=1e-5)
