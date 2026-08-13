# SPDX-License-Identifier: BSD-3-Clause
"""Tests for velocity dispersion broadening in spectroscopy.

Validates that:
1. Broadening reduces spectral resolution (smooths features)
2. Broadening conserves total flux (integral preserved)
3. sigma_v=0 gives identity (no change)
4. Larger sigma_v gives more smoothing
5. Gradients through the FFT convolution are finite
6. Results are physically reasonable for typical galaxy σ_v
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.observation.spectrum import velocity_broaden

pytestmark = pytest.mark.bounds
jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def wave():
    """Log-uniform wavelength grid (200 pixels, 3800-9200 Å).

    Log-uniform, not linear (#1742). ``velocity_broaden`` convolves with a
    constant Gaussian in ``ln(lambda)``, which is exact only on a grid uniform
    in ``ln(lambda)``; on the ``linspace`` this fixture used to return, the
    broadening came out low by ``wave[0]/lambda`` — reaching 3800/9200 = 0.41 at
    the red end of this very range. The function now refuses such a grid rather
    than returning a plausible wrong answer, so every test here was exercising a
    precondition violation.
    """
    return jnp.logspace(jnp.log10(3800.0), jnp.log10(9200.0), 200)


@pytest.fixture
def sharp_spectrum(wave):
    """Spectrum with a narrow emission line for testing broadening."""
    flux = jnp.ones_like(wave) * 1e-17
    # Add a narrow Gaussian line at 5500 Å (10 Å FWHM)
    line_center = 5500.0
    line_sigma = 4.25  # Å (FWHM ~ 10 Å)
    flux = flux + 5e-17 * jnp.exp(-0.5 * ((wave - line_center) / line_sigma) ** 2)
    return flux


@pytest.fixture
def flat_spectrum(wave):
    """Flat spectrum (should be unchanged by broadening)."""
    return jnp.ones_like(wave) * 1e-17


class TestVelocityBroadenBasics:
    """Basic properties of velocity broadening."""

    def test_output_shape(self, wave, sharp_spectrum):
        """Output has same shape as input."""
        result = velocity_broaden(sharp_spectrum, wave, 150.0)
        chex.assert_equal_shape([result, sharp_spectrum])

    def test_zero_sigma_identity(self, wave, sharp_spectrum):
        """sigma_v=0 gives back the original spectrum."""
        result = velocity_broaden(sharp_spectrum, wave, 0.0)
        assert_allclose(result, sharp_spectrum, rtol=1e-10)

    def test_flat_spectrum_unchanged(self, wave, flat_spectrum):
        """Broadening a flat spectrum doesn't change it."""
        result = velocity_broaden(flat_spectrum, wave, 200.0)
        # Interior pixels (avoid edge effects from FFT wrapping)
        interior = slice(10, -10)
        assert_allclose(
            result[interior],
            flat_spectrum[interior],
            rtol=1e-6,
        )

    def test_positive_output(self, wave, sharp_spectrum):
        """Broadened spectrum is non-negative (physical)."""
        result = velocity_broaden(sharp_spectrum, wave, 150.0)
        assert jnp.all(result >= -1e-25)  # allow tiny numerical noise


class TestVelocityBroadenPhysics:
    """Physical correctness of broadening."""

    def test_reduces_peak(self, wave, sharp_spectrum):
        """Broadening reduces the peak of a narrow line."""
        result = velocity_broaden(sharp_spectrum, wave, 150.0)
        assert float(jnp.max(result)) < float(jnp.max(sharp_spectrum))

    def test_larger_sigma_more_smoothing(self, wave, sharp_spectrum):
        """Larger sigma_v gives smaller peak (more smoothed)."""
        peak_100 = float(jnp.max(velocity_broaden(sharp_spectrum, wave, 100.0)))
        peak_200 = float(jnp.max(velocity_broaden(sharp_spectrum, wave, 200.0)))
        peak_300 = float(jnp.max(velocity_broaden(sharp_spectrum, wave, 300.0)))
        assert peak_100 > peak_200 > peak_300

    def test_conserves_flux(self, wave, sharp_spectrum):
        """Total flux is conserved (integral preserved)."""
        result = velocity_broaden(sharp_spectrum, wave, 150.0)
        total_in = float(jnp.sum(sharp_spectrum))
        total_out = float(jnp.sum(result))
        assert_allclose(total_out, total_in, rtol=1e-6)

    def test_typical_galaxy_sigma(self, wave, sharp_spectrum):
        """Works for typical galaxy velocity dispersions (50-300 km/s)."""
        for sigma in [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]:
            result = velocity_broaden(sharp_spectrum, wave, sigma)
            assert jnp.all(jnp.isfinite(result)), f"NaN at sigma_v={sigma}"


class TestVelocityBroadenGradients:
    """Gradients through the FFT convolution."""

    def test_gradient_wrt_sigma_finite(self, wave, sharp_spectrum):
        """Gradient w.r.t. sigma_v is finite."""

        def loss(sigma):
            return jnp.sum(velocity_broaden(sharp_spectrum, wave, sigma) ** 2)

        grad_jax = float(jax.grad(loss)(150.0))
        grad_fd = fd_grad(loss, 150.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="velocity_broaden: FD check ∂/∂sigma_v",
        )

    def test_gradient_wrt_flux_finite(self, wave):
        """Gradient w.r.t. input flux is finite."""
        flux = jnp.ones(len(wave)) * 1e-17

        def loss(f):
            return jnp.sum(velocity_broaden(f, wave, 150.0) ** 2)

        g = jax.grad(loss)(flux)
        chex.assert_tree_all_finite(g)

    def test_jit_compatible(self, wave, sharp_spectrum):
        """Works inside jax.jit."""
        fn = jax.jit(lambda f, s: velocity_broaden(f, wave, s))
        result = fn(sharp_spectrum, 150.0)
        chex.assert_tree_all_finite(result)
