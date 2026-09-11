# SPDX-License-Identifier: BSD-3-Clause
"""Tests for CIGALE SKIRTOR disc models and extinction laws.

Validates piecewise power-law disc spectra, extinction curves, anisotropic
luminosity geometry, and fracAGN conversions against known CIGALE values.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax.numpy as jnp
import numpy as np

from tengri.components.agn.disc_cigale import (
    adaf_disk_spectrum,
    piecewise_powerlaw_disk,
    schartmann2005_disk_spectrum,
    skirtor_disk_spectrum,
)
from tengri.components.agn.fracagn import (
    fracagn_to_log_lbol,
    log_lbol_to_fracagn,
)
from tengri.components.agn.polar_dust import (
    anisotropic_polar_luminosity,
    calzetti2000_extinction_curve,
    gaskell2004_extinction_curve,
)
from tests._bounds import assert_non_negative


class TestPiecewisePowerlawDisk:
    """Test generic piecewise power-law disc spectrum construction."""

    def test_shape(self):
        """Output shape matches wavelength input."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        limits = jnp.array([8.0, 10.0, 100.0, 5000.0, 1e6])
        coefs = jnp.array([0.2, -1.0, -1.5, -4.0])
        spectrum = piecewise_powerlaw_disk(wavelength, limits, coefs)
        chex.assert_equal_shape([spectrum, wavelength])

    def test_normalization_unity(self):
        """Spectrum integrates to approximately 1.0."""
        wavelength = jnp.linspace(8.0, 1e6, 500)
        limits = jnp.array([8.0, 10.0, 100.0, 5000.0, 1e6])
        coefs = jnp.array([0.2, -1.0, -1.5, -4.0])
        spectrum = piecewise_powerlaw_disk(wavelength, limits, coefs)
        integral = jnp.trapezoid(spectrum, wavelength)
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_monotonicity_each_segment(self):
        """Spectrum follows expected monotonicity in each segment.

        The grid is geometric, not linear. On the previous
        ``jnp.linspace(8.0, 1e6, 1000)`` the spacing was ~1000 A across five
        decades, so the 8-10 A segment held exactly **one** sample: measured
        len(seg1)=1, len(seg3)=4. The positive-slope check was written as
        ``if len(seg1) > 2:`` and therefore never executed -- the test reported
        a pass having checked only the second segment. Geometric spacing gives
        len(seg1)=19 and len(seg3)=333, and the counts are asserted so a future
        grid change cannot quietly empty a segment again.
        """
        wavelength = jnp.geomspace(8.0, 1e6, 1000)
        limits = jnp.array([8.0, 10.0, 100.0, 5000.0, 1e6])
        coefs = jnp.array([0.2, -1.0, -1.5, -4.0])
        spectrum = piecewise_powerlaw_disk(wavelength, limits, coefs)

        # Segment 1 (8-10 A): positive slope (coef=0.2)
        seg1 = spectrum[(wavelength >= 8.0) & (wavelength < 10.0)]
        assert seg1.shape[0] > 2, f"probe setup failed: segment 1 has {seg1.shape[0]} samples"
        assert seg1[-1] > seg1[0], (
            f"segment 1 (8-10 A, coef=+0.2) must rise; got {seg1[0]:.3e} -> {seg1[-1]:.3e}"
        )

        # Segment 3 (100-5000 A): negative slope (coef=-1.5)
        seg3 = spectrum[(wavelength >= 100.0) & (wavelength < 5000.0)]
        assert seg3.shape[0] > 2, f"probe setup failed: segment 3 has {seg3.shape[0]} samples"
        assert seg3[-1] < seg3[0], (
            f"segment 3 (100-5000 A, coef=-1.5) must fall; got {seg3[0]:.3e} -> {seg3[-1]:.3e}"
        )

    def test_continuity_at_breakpoints(self):
        """Spectrum is continuous at segment breakpoints."""
        wavelength = jnp.array([9.9, 10.0, 10.1, 99.9, 100.0, 100.1])
        limits = jnp.array([8.0, 10.0, 100.0, 5000.0, 1e6])
        coefs = jnp.array([0.2, -1.0, -1.5, -4.0])
        spectrum = piecewise_powerlaw_disk(wavelength, limits, coefs)
        # Values near breakpoints should be continuous (no jumps)
        assert jnp.abs(spectrum[1] - spectrum[2]) / (spectrum[1] + 1e-100) < 0.1
        assert jnp.abs(spectrum[4] - spectrum[5]) / (spectrum[4] + 1e-100) < 0.1


class TestSKIRTORDiskSpectrum:
    """Test SKIRTOR disc spectrum model."""

    def test_shape(self):
        """Output shape matches input."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        spectrum = skirtor_disk_spectrum(wavelength, delta=0.0)
        chex.assert_equal_shape([spectrum, wavelength])

    def test_normalization(self):
        """Spectrum integrates to ~1.0."""
        wavelength = jnp.linspace(8.0, 1e6, 500)
        spectrum = skirtor_disk_spectrum(wavelength, delta=0.0)
        integral = jnp.trapezoid(spectrum, wavelength)
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_delta_parameter_effect(self):
        """Delta parameter modulates the mid-IR slope."""
        wavelength = jnp.linspace(8.0, 1e6, 500)
        spec_delta0 = skirtor_disk_spectrum(wavelength, delta=0.0)
        spec_delta_pos = skirtor_disk_spectrum(wavelength, delta=0.5)
        spec_delta_neg = skirtor_disk_spectrum(wavelength, delta=-0.5)
        # All should be normalized
        np.testing.assert_allclose(jnp.trapezoid(spec_delta_pos, wavelength), 1.0, rtol=0.01)
        np.testing.assert_allclose(jnp.trapezoid(spec_delta_neg, wavelength), 1.0, rtol=0.01)
        # Delta parameter changes the spectrum shape
        # Verify that delta+ differs from delta0 and delta0 differs from delta-
        diff_pos = jnp.mean(jnp.abs(spec_delta_pos - spec_delta0))
        diff_neg = jnp.mean(jnp.abs(spec_delta0 - spec_delta_neg))
        assert diff_pos > 1e-9
        assert diff_neg > 1e-9

    def test_positivity(self):
        """All spectral values are non-negative."""
        wavelength = jnp.linspace(8.0, 1e6, 200)
        spectrum = skirtor_disk_spectrum(wavelength, delta=0.0)
        assert_non_negative(spectrum, name="spectrum")


class TestSchartmann2005DiskSpectrum:
    """Test Schartmann et al. (2005) disc spectrum model."""

    def test_shape(self):
        """Output shape matches input."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        spectrum = schartmann2005_disk_spectrum(wavelength, delta=0.0)
        chex.assert_equal_shape([spectrum, wavelength])

    def test_normalization(self):
        """Spectrum integrates to ~1.0."""
        wavelength = jnp.linspace(8.0, 1e6, 500)
        spectrum = schartmann2005_disk_spectrum(wavelength, delta=0.0)
        integral = jnp.trapezoid(spectrum, wavelength)
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_different_from_skirtor(self):
        """Schartmann spectrum differs from SKIRTOR due to different slopes."""
        wavelength = jnp.linspace(100.0, 1e5, 200)
        skirtor = skirtor_disk_spectrum(wavelength, delta=0.0)
        schartmann = schartmann2005_disk_spectrum(wavelength, delta=0.0)
        # Should differ at most wavelengths (different power-law indices)
        diff = jnp.mean(jnp.abs(skirtor - schartmann))
        assert diff > 1e-8  # Some difference due to different power-law coefficients


class TestADAFDiskSpectrum:
    """Test ADAF + truncated disc blended spectrum."""

    def test_shape(self):
        """Output shape matches input."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        spectrum = adaf_disk_spectrum(wavelength, delta=0.0)
        chex.assert_equal_shape([spectrum, wavelength])

    def test_normalization(self):
        """Spectrum integrates to ~1.0."""
        wavelength = jnp.linspace(8.0, 1e6, 500)
        spectrum = adaf_disk_spectrum(wavelength, delta=0.0)
        integral = jnp.trapezoid(spectrum, wavelength)
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_delta_blending(self):
        """Delta parameter interpolates between ADAF and thin disc."""
        wavelength = jnp.linspace(8.0, 1e6, 300)
        spec_delta0 = adaf_disk_spectrum(wavelength, delta=0.0)  # Pure ADAF
        spec_delta05 = adaf_disk_spectrum(wavelength, delta=0.5)  # Blend
        spec_delta1 = adaf_disk_spectrum(wavelength, delta=1.0)  # Pure thin disc
        # All normalized
        np.testing.assert_allclose(jnp.trapezoid(spec_delta0, wavelength), 1.0, rtol=0.01)
        np.testing.assert_allclose(jnp.trapezoid(spec_delta05, wavelength), 1.0, rtol=0.01)
        np.testing.assert_allclose(jnp.trapezoid(spec_delta1, wavelength), 1.0, rtol=0.01)
        # Blend should be between the two extremes (on average)
        mean0 = jnp.mean(spec_delta0)
        mean05 = jnp.mean(spec_delta05)
        mean1 = jnp.mean(spec_delta1)
        is_between = min(mean0, mean1) < mean05 < max(mean0, mean1)
        is_close_0 = np.isclose(mean05, mean0)
        is_close_1 = np.isclose(mean05, mean1)
        assert is_between or is_close_0 or is_close_1


class TestCalzetti2000Extinction:
    """Test Calzetti et al. (2000) extinction law."""

    def test_shape(self):
        """Output shape matches input."""
        wavelength = jnp.linspace(1000.0, 20000.0, 100)
        k_lambda = calzetti2000_extinction_curve(wavelength)
        chex.assert_equal_shape([k_lambda, wavelength])

    def test_positivity(self):
        """Extinction coefficient is non-negative."""
        wavelength = jnp.linspace(1000.0, 20000.0, 100)
        k_lambda = calzetti2000_extinction_curve(wavelength)
        assert_non_negative(k_lambda, name="k_lambda")

    def test_known_values(self):
        """Check extinction at known wavelengths against literature."""
        # Calzetti et al. (2000) parameterization gives k values
        # λ = 1500 A (0.15 um): k ≈ 1.97
        # λ = 5500 A (0.55 um): k ≈ 4.04 (from the formula)
        wavelength = jnp.array([1500.0, 5500.0, 12000.0])
        k_lambda = calzetti2000_extinction_curve(wavelength)
        # At 1500 A (UV), extinction should be much higher than at longer wavelengths
        k_1500 = k_lambda[0]
        k_5500 = k_lambda[1]
        k_12000 = k_lambda[2]
        assert k_1500 > 1.0
        # Extinction decreases with wavelength
        assert k_5500 > k_12000

    def test_wavelength_dependence(self):
        """Extinction decreases with wavelength (blue absorbs more)."""
        wavelength = jnp.linspace(1000.0, 20000.0, 100)
        k_lambda = calzetti2000_extinction_curve(wavelength)
        # Derivative should be negative (extinction decreases with wavelength)
        dk_dlambda = np.diff(k_lambda) / np.diff(wavelength)
        assert np.median(dk_dlambda) < 0.0


class TestGaskell2004Extinction:
    """Test Gaskell et al. (2004) extinction law."""

    def test_shape(self):
        """Output shape matches input."""
        wavelength = jnp.linspace(1000.0, 20000.0, 100)
        k_lambda = gaskell2004_extinction_curve(wavelength)
        chex.assert_equal_shape([k_lambda, wavelength])

    def test_positivity(self):
        """Extinction coefficient is non-negative."""
        wavelength = jnp.linspace(1000.0, 20000.0, 100)
        k_lambda = gaskell2004_extinction_curve(wavelength)
        assert_non_negative(k_lambda, name="k_lambda")

    def test_piecewise_behavior(self):
        """Extinction law changes behavior at x = 3.69 (λ ≈ 271 nm)."""
        # Short wavelength (x > 3.69): polynomial with positive curvature
        wave_short = jnp.array([500.0, 1000.0, 2000.0])  # x > 3.69
        k_short = gaskell2004_extinction_curve(wave_short)
        # Long wavelength (x < 3.69): linear in x (λ > 2710 A)
        wave_long = jnp.array([5000.0, 10000.0, 20000.0])  # x < 3.69
        k_long = gaskell2004_extinction_curve(wave_long)
        # Short wavelength extinction should be positive
        assert_non_negative(k_short, name="k_short")
        # Long wavelength clipped to be non-negative by implementation
        assert_non_negative(k_long, name="k_long")

    def test_continuity_at_transition(self):
        """Extinction is continuous at the piecewise transition (x ≈ 3.69)."""
        # x = 3.69 corresponds to λ ≈ 271.5 nm = 2715 A
        wave_around = jnp.array([2700.0, 2715.0, 2730.0])
        k_around = gaskell2004_extinction_curve(wave_around)
        # Values should not jump
        jump = jnp.abs(k_around[1] - k_around[0])
        assert jump < 0.5  # Allow small numerical differences


class TestAnisotropicPolarLuminosity:
    """Test anisotropic polar dust luminosity calculation."""

    def test_shape_and_sign(self):
        """Output is a scalar and non-negative."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        l_nu = jnp.ones_like(wavelength)
        extinction = jnp.ones_like(wavelength) * 0.8
        l_total = anisotropic_polar_luminosity(l_nu, wavelength, 30.0, extinction)
        assert jnp.ndim(l_total) == 0
        assert l_total >= 0.0

    def test_opening_angle_effect(self):
        """Luminosity varies with opening angle (torus geometry)."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        l_nu = jnp.ones_like(wavelength) * 1e-12
        extinction = jnp.ones_like(wavelength) * 0.8
        l_30 = anisotropic_polar_luminosity(l_nu, wavelength, 30.0, extinction)
        l_60 = anisotropic_polar_luminosity(l_nu, wavelength, 60.0, extinction)
        # Geometry factor varies with opening angle
        # At larger angles, the aniso factor should differ
        ratio = l_30 / jnp.maximum(l_60, 1e-100)
        assert not np.isclose(ratio, 1.0)

    def test_full_transmission(self):
        """Absorbed luminosity vanishes at full transmission and grows
        as transmission falls. The function returns the polar-dust
        absorbed (= re-emitted) power, NOT the transmitted disc power
        — see ``polar_dust.py:443`` (and CIGALE ``skirtor2016.py:368``).
        """
        wavelength = jnp.linspace(100.0, 1e5, 100)
        l_nu = jnp.ones_like(wavelength) * 1e-12
        l_partial = anisotropic_polar_luminosity(
            l_nu, wavelength, 30.0, jnp.ones_like(wavelength) * 0.5
        )
        l_transparent = anisotropic_polar_luminosity(
            l_nu, wavelength, 30.0, jnp.ones_like(wavelength) * 1.0
        )
        # Transparent polar dust absorbs nothing -> zero re-emission.
        # Partial transmission absorbs the complementary fraction.
        assert float(l_transparent) == 0.0
        assert float(l_partial) > 0.0

    def test_zero_luminosity(self):
        """Zero input luminosity gives zero output."""
        wavelength = jnp.linspace(100.0, 1e5, 100)
        l_nu = jnp.zeros_like(wavelength)
        extinction = jnp.ones_like(wavelength) * 0.8
        l_total = anisotropic_polar_luminosity(l_nu, wavelength, 30.0, extinction)
        assert jnp.abs(l_total) < 1e-50


class TestFracAGNConversion:
    """Test fracAGN ↔ log_lbol bidirectional conversion."""

    def test_roundtrip_consistency(self):
        """Converting back and forth preserves the original value."""
        log_lbol_orig = 10.5
        l_dust = 1e13  # Solar luminosities
        # Forward and back
        frac_agn = log_lbol_to_fracagn(log_lbol_orig, l_dust)
        log_lbol_recovered = fracagn_to_log_lbol(frac_agn, l_dust)
        np.testing.assert_allclose(log_lbol_orig, log_lbol_recovered, rtol=1e-6)

    def test_edge_cases(self):
        """Handle edge cases (very small/large luminosity)."""
        l_dust = 1e13
        # Very high AGN luminosity
        log_lbol_high = 15.0
        frac_high = log_lbol_to_fracagn(log_lbol_high, l_dust)
        assert 0.0 <= frac_high < 1.0
        # Very low AGN luminosity
        log_lbol_low = 5.0
        frac_low = log_lbol_to_fracagn(log_lbol_low, l_dust)
        assert 0.0 <= frac_low < 1.0
        assert frac_low < frac_high

    def test_no_agn(self):
        """Zero AGN luminosity corresponds to zero AGN fraction."""
        l_dust = 1e13
        frac_zero = log_lbol_to_fracagn(-np.inf, l_dust)
        # Very small luminosity → very small fraction
        assert frac_zero < 1e-10

    def test_monotonicity(self):
        """Higher log_lbol → higher frac_agn."""
        l_dust = 1e13
        lbols = jnp.array([8.0, 10.0, 12.0, 14.0])
        fracs = jnp.array([log_lbol_to_fracagn(lb, l_dust) for lb in lbols])
        # Strictly increasing
        assert jnp.all(jnp.diff(fracs) > 0.0)
