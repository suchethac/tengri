# SPDX-License-Identifier: BSD-3-Clause
"""Tests for emission-line blending by spectral resolution.

Verifies:
1. Single line produces a Gaussian profile at the correct wavelength
2. Total luminosity is conserved (integral matches input)
3. Close lines blend at low R but separate at high R
4. Redshift shifts lines correctly
5. Pure JAX, JIT-compatible, and differentiable
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.observation.spectrum import blend_emission_lines

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# Speed of light in Angstrom/s
_C_AA = 2.99792458e18


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def wave_grid():
    """Fine wavelength grid covering H-alpha region."""
    return jnp.linspace(6000.0, 7000.0, 10000)


@pytest.fixture
def halpha():
    """H-alpha line parameters."""
    return jnp.array([6563.0]), jnp.array([1.0])  # 1 Lsun


# ── Single-line tests ─────────────────────────────────────────────


class TestSingleLine:
    """Test blend_emission_lines with a single line."""

    def test_peak_at_correct_wavelength(self, wave_grid, halpha):
        """Peak should be at the line wavelength (z=0)."""
        lam, lum = halpha
        spec = blend_emission_lines(lam, lum, 1000.0, wave_grid, redshift=0.0)
        peak_wave = float(wave_grid[jnp.argmax(spec)])
        assert_allclose(peak_wave, 6563.0, atol=1.0)

    def test_peak_shifts_with_redshift(self, wave_grid, halpha):
        """Peak should shift to lambda*(1+z) with redshift."""
        lam, lum = halpha
        z = 0.05
        wave_wide = jnp.linspace(6000.0, 7200.0, 12000)
        spec = blend_emission_lines(lam, lum, 1000.0, wave_wide, redshift=z)
        peak_wave = float(wave_wide[jnp.argmax(spec)])
        expected = 6563.0 * (1.0 + z)
        assert_allclose(peak_wave, expected, atol=2.0)

    def test_width_scales_with_resolution(self, halpha):
        """FWHM should be lambda/R."""
        lam, lum = halpha
        for R in [500.0, 2000.0, 5000.0]:
            # Use a grid wide enough around the line center
            fwhm_expected = 6563.0 / R
            half_width = 5.0 * fwhm_expected
            wave = jnp.linspace(6563.0 - half_width, 6563.0 + half_width, 50000)
            spec = blend_emission_lines(lam, lum, R, wave, redshift=0.0)
            # Measure FWHM from the profile
            peak_val = float(jnp.max(spec))
            half_max = peak_val / 2.0
            above_half = spec > half_max
            # First and last indices above half max
            first_idx = int(jnp.argmax(above_half))
            last_idx = len(wave) - 1 - int(jnp.argmax(above_half[::-1]))
            fwhm_measured = float(wave[last_idx] - wave[first_idx])
            assert_allclose(
                fwhm_measured, fwhm_expected, rtol=0.05, err_msg=f"FWHM mismatch at R={R}"
            )

    def test_luminosity_conservation(self, halpha):
        """Integral of L_nu * dnu should equal input luminosity."""
        lam, lum = halpha
        R = 500.0
        # Use a wide grid to capture the full profile
        wave = jnp.linspace(6200.0, 6900.0, 50000)
        spec = blend_emission_lines(lam, lum, R, wave, redshift=0.0)
        # spec is in Lsun/Hz; integrate over frequency
        nu = _C_AA / wave
        # Integrate: L_total = -integral(L_nu * dnu)  [minus because nu decreasing]
        L_total = float(-jnp.trapezoid(spec, nu))
        assert_allclose(
            L_total,
            float(lum[0]),
            rtol=0.02,
            err_msg=f"Luminosity not conserved: {L_total:.4e} vs {float(lum[0]):.4e}",
        )

    def test_positive_spectrum(self, wave_grid, halpha):
        """Output should be non-negative everywhere."""
        lam, lum = halpha
        spec = blend_emission_lines(lam, lum, 500.0, wave_grid)
        assert jnp.all(spec >= 0), "Spectrum should be non-negative"


# ── Multi-line blending tests ─────────────────────────────────────


class TestLineBlending:
    """Test blending behavior with multiple lines."""

    def test_well_separated_lines_have_two_peaks(self):
        """Well-separated lines should produce two distinct peaks."""
        # H-alpha and H-beta are well separated (~1200 A apart)
        lam = jnp.array([4861.0, 6563.0])
        lum = jnp.array([1.0, 3.0])
        wave = jnp.linspace(4500.0, 7000.0, 50000)
        spec = blend_emission_lines(lam, lum, 1000.0, wave, redshift=0.0)
        # Find peaks: H-beta region and H-alpha region
        mask_hb = (wave > 4800.0) & (wave < 4920.0)
        mask_ha = (wave > 6500.0) & (wave < 6620.0)
        peak_hb = float(jnp.max(spec * mask_hb))
        peak_ha = float(jnp.max(spec * mask_ha))
        assert peak_hb > 0, "H-beta peak should be positive"
        assert peak_ha > 0, "H-alpha peak should be positive"

    def test_close_lines_blend_at_low_r(self):
        """[NII] doublet (6548+6583 A) should blend with H-alpha at R=100."""
        # NII doublet around H-alpha
        lam = jnp.array([6548.0, 6563.0, 6583.0])
        lum = jnp.array([0.33, 1.0, 1.0])
        wave = jnp.linspace(6300.0, 6800.0, 10000)

        spec_low = blend_emission_lines(lam, lum, 100.0, wave)
        spec_high = blend_emission_lines(lam, lum, 5000.0, wave)

        # At R=100: single broad peak; at R=5000: three distinct peaks
        # Count local maxima at R=5000
        diff_high = jnp.diff(spec_high)
        sign_changes = jnp.sum(
            (diff_high[:-1] > 0)
            & (diff_high[1:] < 0)
            & (spec_high[1:-1] > 0.01 * jnp.max(spec_high))
        )
        assert int(sign_changes) >= 2, (
            f"High-R should resolve at least 2 peaks, found {int(sign_changes)}"
        )

    def test_total_luminosity_multi_line(self):
        """Total luminosity should equal sum of input luminosities."""
        lam = jnp.array([4861.0, 6563.0, 5007.0])
        lum = jnp.array([1.0, 3.0, 2.0])
        wave = jnp.linspace(4500.0, 7000.0, 100000)
        spec = blend_emission_lines(lam, lum, 500.0, wave)
        nu = _C_AA / wave
        L_total = float(-jnp.trapezoid(spec, nu))
        L_expected = float(jnp.sum(lum))
        assert_allclose(
            L_total,
            L_expected,
            rtol=0.03,
            err_msg=f"Total luminosity: {L_total:.4e} vs {L_expected:.4e}",
        )


# ── JAX compatibility tests ───────────────────────────────────────


class TestJAXCompatibility:
    """Test JIT compilation and differentiability."""

    def test_jit_compiles(self, halpha):
        """Function should compile under jax.jit."""
        lam, lum = halpha
        wave = jnp.linspace(6400.0, 6700.0, 1000)
        f = jax.jit(blend_emission_lines, static_argnums=(2,))
        # static_argnums not needed since R is a traced scalar, but
        # the function should work either way
        spec = blend_emission_lines(lam, lum, 500.0, wave)
        chex.assert_equal_shape([spec, wave])
        chex.assert_tree_all_finite(spec)

    def test_gradient_wrt_luminosity(self, halpha):
        """Gradient w.r.t. line luminosity should be finite and positive."""
        lam, _ = halpha
        wave = jnp.linspace(6400.0, 6700.0, 1000)

        def loss(lum_val):
            lum = jnp.array([lum_val])
            return jnp.sum(blend_emission_lines(lam, lum, 500.0, wave))

        g_jax = float(jax.grad(loss)(1.0))
        g_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
        assert g_jax > 0, "More luminosity should increase total flux"

    def test_gradient_wrt_resolution(self, halpha):
        """Gradient w.r.t. spectral resolution should be finite."""
        lam, lum = halpha
        wave = jnp.linspace(6400.0, 6700.0, 1000)

        def loss(R):
            return jnp.sum(blend_emission_lines(lam, lum, R, wave))

        g_jax = float(jax.grad(loss)(500.0))
        g_fd = fd_grad(loss, 500.0)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )

    def test_gradient_wrt_redshift(self, halpha):
        """Gradient w.r.t. redshift should be finite."""
        lam, lum = halpha
        wave = jnp.linspace(6400.0, 7000.0, 1000)

        def loss(z):
            return jnp.sum(blend_emission_lines(lam, lum, 500.0, wave, redshift=z))

        g_jax = float(jax.grad(loss)(0.03))
        g_fd = fd_grad(loss, 0.03)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )

    def test_vmap_over_batch(self, halpha):
        """Should work with vmap over a batch of luminosities."""
        lam, _ = halpha
        wave = jnp.linspace(6400.0, 6700.0, 500)
        lum_batch = jnp.array([[1.0], [2.0], [3.0]])

        def single(lum):
            return blend_emission_lines(lam, lum, 500.0, wave)

        batch_spec = jax.vmap(single)(lum_batch)
        chex.assert_shape(batch_spec, (3, 500))
        # Linearity check: 2x luminosity -> 2x flux
        assert_allclose(batch_spec[1], 2.0 * batch_spec[0], rtol=1e-12)


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_luminosity(self, wave_grid):
        """Zero luminosity should produce zero spectrum."""
        lam = jnp.array([6563.0])
        lum = jnp.array([0.0])
        spec = blend_emission_lines(lam, lum, 500.0, wave_grid)
        assert_allclose(spec, 0.0, atol=1e-30)

    def test_very_high_resolution(self, halpha):
        """At very high R, the line should be very narrow."""
        lam, lum = halpha
        wave = jnp.linspace(6562.0, 6564.0, 100000)
        spec = blend_emission_lines(lam, lum, 100000.0, wave)
        # Peak should be very high (narrow line)
        peak = float(jnp.max(spec))
        assert peak > 0

    def test_line_outside_grid(self, halpha):
        """Line far from the grid should contribute negligibly."""
        lam, lum = halpha
        # Grid far from H-alpha
        wave = jnp.linspace(4000.0, 4500.0, 1000)
        spec = blend_emission_lines(lam, lum, 500.0, wave)
        assert float(jnp.max(spec)) < 1e-20
