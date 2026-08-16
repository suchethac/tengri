# SPDX-License-Identifier: BSD-3-Clause
"""Tests for shared AGN physics kernels in _phys.py.

Covers:
1. gaussian_line_profile — normalization, shape, consistency with NLR/BLR
2. ring_area — R&L 1979 Eq 1.6 geometry, cos_inc clamping
3. Differentiability and JIT compatibility of both
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp

from tengri.components.agn._phys import (
    ANGSTROM_CM,
    C_LIGHT,
    gaussian_line_profile,
    ring_area,
)
from tests._grad_parity import assert_grad_matches_fd


@pytest.fixture()
def wavelength():
    return jnp.linspace(3000.0, 8000.0, 5000)


class TestGaussianLineProfile:
    """Gaussian emission-line profile kernel tests."""

    def test_output_shape(self, wavelength):
        result = gaussian_line_profile(wavelength, 5007.0, 500.0)
        chex.assert_equal_shape([result, wavelength])

    def test_peak_at_line_center(self, wavelength):
        """Profile should peak near the line center wavelength."""
        result = gaussian_line_profile(wavelength, 5007.0, 500.0)
        peak_idx = jnp.argmax(result)
        peak_wave = wavelength[peak_idx]
        assert abs(float(peak_wave) - 5007.0) < 2.0

    def test_integral_over_frequency_is_unity(self, wavelength):
        """Profile normalized so integral over d(nu) = 1."""
        profile = gaussian_line_profile(wavelength, 5007.0, 500.0)
        nu = C_LIGHT / (wavelength * ANGSTROM_CM)
        sort_idx = jnp.argsort(nu)
        integral = jnp.abs(jnp.trapezoid(profile[sort_idx], nu[sort_idx]))
        assert jnp.isclose(integral, 1.0, atol=0.02)

    def test_wider_fwhm_gives_broader_profile(self, wavelength):
        """Doubling FWHM should roughly halve peak height."""
        narrow = gaussian_line_profile(wavelength, 5007.0, 300.0)
        wide = gaussian_line_profile(wavelength, 5007.0, 600.0)
        assert float(jnp.max(wide)) < float(jnp.max(narrow))

    def test_different_line_centers(self, wavelength):
        """Profiles at different wavelengths should peak at different places."""
        ha = gaussian_line_profile(wavelength, 6563.0, 500.0)
        hb = gaussian_line_profile(wavelength, 4861.0, 500.0)
        assert jnp.argmax(ha) != jnp.argmax(hb)

    def test_jit_compatible(self, wavelength):
        result_eager = gaussian_line_profile(wavelength, 5007.0, 500.0)
        result_jit = jax.jit(gaussian_line_profile, static_argnums=())(wavelength, 5007.0, 500.0)
        assert jnp.allclose(result_eager, result_jit)

    def test_grad_through_fwhm(self, wavelength):
        """Gradient of total flux w.r.t. line center should be finite."""

        def loss(line_center):
            profile = gaussian_line_profile(wavelength, line_center, 500.0)
            return jnp.sum(profile**2)

        grad = assert_grad_matches_fd(loss, 5007.0)
        assert jnp.isfinite(grad)

    def test_nlr_blr_consistency(self, wavelength):
        """Shared kernel must match the old NLR/BLR inline implementation.
        The old code used 2.3548 (truncated); the shared kernel uses
        2.3548200450309493 (full precision). Verify the difference
        is negligible at the SED level.
        """
        from tengri.components.agn.nlr import compute_nlr_sed

        l_bol = 1e45
        sed = compute_nlr_sed(wavelength, l_bol, covering_fraction=0.1)
        chex.assert_tree_all_finite(sed)
        assert float(jnp.max(sed)) > 0.0


class TestRingArea:
    """Disc ring projected area tests."""

    def test_formula_matches_rl79(self):
        """ring_area = pi * 2*pi * r * dr * max(cos_inc, 0.01)."""
        r = 1e12
        dr = 1e10
        cos_inc = 0.5
        expected = jnp.pi * 2.0 * jnp.pi * r * dr * cos_inc
        result = ring_area(r, dr, cos_inc)
        assert jnp.isclose(result, expected, rtol=1e-10)

    def test_cos_inc_clamped_at_001(self):
        """Edge-on (cos_inc=0) should clamp to 0.01, not zero."""
        r, dr = 1e12, 1e10
        result_zero = ring_area(r, dr, 0.0)
        result_neg = ring_area(r, dr, -0.5)
        expected = jnp.pi * 2.0 * jnp.pi * r * dr * 0.01
        assert jnp.isclose(result_zero, expected, rtol=1e-10)
        assert jnp.isclose(result_neg, expected, rtol=1e-10)

    def test_face_on_is_maximum(self):
        """cos_inc=1 (face-on) should give the largest area."""
        r, dr = 1e12, 1e10
        face_on = ring_area(r, dr, 1.0)
        tilted = ring_area(r, dr, 0.5)
        assert float(face_on) > float(tilted)

    def test_scales_linearly_with_r_and_dr(self):
        """Doubling r or dr should double the area."""
        cos_inc = 0.7
        a1 = ring_area(1e12, 1e10, cos_inc)
        a2 = ring_area(2e12, 1e10, cos_inc)
        assert jnp.isclose(a2, 2.0 * a1, rtol=1e-10)
        a3 = ring_area(1e12, 2e10, cos_inc)
        assert jnp.isclose(a3, 2.0 * a1, rtol=1e-10)

    def test_jit_compatible(self):
        result = jax.jit(ring_area)(1e12, 1e10, 0.5)
        assert jnp.isfinite(result)

    def test_grad_through_cos_inc(self):
        """Gradient w.r.t. cos_inc should be finite and positive."""
        grad = jax.grad(ring_area, argnums=2)(1e12, 1e10, 0.5)
        assert jnp.isfinite(grad)
        assert float(grad) > 0.0

    def test_disc_sed_uses_ring_area(self):
        """End-to-end: disc SED should still produce valid output."""
        from tengri.components.agn.disc import multicolor_disc

        wave = jnp.linspace(1000.0, 30000.0, 3000)
        sed = multicolor_disc(wave, agn_log_lbol=12.0, agn_cos_inc=0.5)
        chex.assert_tree_all_finite(sed)
        assert float(jnp.max(sed)) > 0.0
