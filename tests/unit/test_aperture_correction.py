"""Tests for the aperture-correction preprocessing utility.

MISSING_FEATURES.md #10. tengri's pipeline assumes aperture-corrected
photometry; this utility lets a user apply per-band corrections to
``flux_obs`` and ``noise`` before constructing the Fitter.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.aperture import apply_aperture_correction

pytestmark = pytest.mark.unit


class TestApplyApertureCorrection:
    def test_unity_correction_is_noop(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        f2, n2 = apply_aperture_correction(flux, noise, jnp.ones(3))
        np.testing.assert_allclose(np.asarray(f2), np.asarray(flux))
        np.testing.assert_allclose(np.asarray(n2), np.asarray(noise))

    def test_uniform_scalar_correction(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        f2, n2 = apply_aperture_correction(flux, noise, 1.5)
        np.testing.assert_allclose(np.asarray(f2), 1.5 * np.asarray(flux))
        np.testing.assert_allclose(np.asarray(n2), 1.5 * np.asarray(noise))

    def test_per_band_correction(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        corr = jnp.array([1.1, 1.2, 1.3])
        f2, n2 = apply_aperture_correction(flux, noise, corr)
        np.testing.assert_allclose(np.asarray(f2), np.asarray(flux) * np.asarray(corr))
        np.testing.assert_allclose(np.asarray(n2), np.asarray(noise) * np.asarray(corr))

    def test_snr_preserved(self):
        """Correction scales flux and noise equally → SNR unchanged."""
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        corr = jnp.array([0.5, 1.5, 2.0])
        f2, n2 = apply_aperture_correction(flux, noise, corr)
        np.testing.assert_allclose(np.asarray(f2 / n2), np.asarray(flux / noise), rtol=1e-12)

    def test_shape_mismatch_raises(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="shape"):
            apply_aperture_correction(flux, noise, jnp.array([1.0, 1.0]))

    def test_negative_correction_raises(self):
        flux = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="positive"):
            apply_aperture_correction(flux, noise, jnp.array([1.0, -0.5, 1.0]))

    def test_returns_jax_arrays(self):
        flux = np.array([1.0, 2.0, 3.0])
        noise = np.array([0.1, 0.2, 0.3])
        f2, n2 = apply_aperture_correction(flux, noise, np.ones(3))
        assert hasattr(f2, "shape")
        assert hasattr(n2, "shape")
