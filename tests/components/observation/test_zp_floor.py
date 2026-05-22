# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the per-band photometric zero-point systematic-floor utility.

MISSING_FEATURES.md #6. Inflates the per-band noise to incorporate
ZP-level calibration uncertainty:

    sigma_eff^2 = sigma_data^2 + (f_floor * flux)^2

This adds a fractional floor in quadrature, so a 2% ZP uncertainty
caps the achievable per-band SNR at 50.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.noise import apply_zp_floor

pytestmark = pytest.mark.bounds


class TestApplyZpFloor:
    def test_zero_floor_is_noop(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        out = apply_zp_floor(flux, noise, jnp.zeros(3))
        np.testing.assert_allclose(np.asarray(out), np.asarray(noise))

    def test_uniform_scalar_floor(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        floor = 0.05  # 5%
        out = apply_zp_floor(flux, noise, floor)
        expected = np.sqrt(np.asarray(noise) ** 2 + (floor * np.asarray(flux)) ** 2)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)

    def test_per_band_floor(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        floor = jnp.array([0.01, 0.05, 0.10])
        out = apply_zp_floor(flux, noise, floor)
        expected = np.sqrt(np.asarray(noise) ** 2 + (np.asarray(floor) * np.asarray(flux)) ** 2)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)

    def test_caps_snr_at_inverse_floor(self):
        """A 2% floor caps SNR at 50 even for very-low statistical noise."""
        flux = jnp.array([1.0, 1.0, 1.0])
        noise = jnp.array([1e-6, 1e-8, 1e-10])
        out = apply_zp_floor(flux, noise, 0.02)
        snr = np.asarray(flux / out)
        assert np.all(snr <= 50.0 + 1e-6)

    def test_negative_floor_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            apply_zp_floor(jnp.array([1.0]), jnp.array([0.1]), -0.01)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            apply_zp_floor(jnp.array([1.0, 2.0]), jnp.array([0.1, 0.2]), jnp.array([0.01]))

    def test_handles_negative_flux(self):
        """Negative-flux non-detections should still produce finite, positive noise."""
        flux = jnp.array([-1.0, 2.0])
        noise = jnp.array([0.5, 0.2])
        out = apply_zp_floor(flux, noise, 0.05)
        out_np = np.asarray(out)
        chex.assert_tree_all_finite(out_np)
        assert np.all(out_np > 0)
