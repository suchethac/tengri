# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the equivalent_width spectral diagnostic."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds
jax.config.update("jax_enable_x64", True)

from tengri.analysis.diagnostics.spectral import equivalent_width
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

_C_AA_S = 2.99792458e18


@pytest.fixture
def wave():
    """Fine wavelength grid around H-alpha (6564.61 Å)."""
    return jnp.linspace(6300.0, 6800.0, 5000)


def _flat_continuum_with_emission(wave, line_center, line_flux, sigma_aa=5.0):
    """Flat F_lambda=1 continuum + Gaussian emission line, returned as L_nu."""
    f_lambda = jnp.ones_like(wave) + line_flux * jnp.exp(
        -0.5 * ((wave - line_center) / sigma_aa) ** 2
    ) / (sigma_aa * jnp.sqrt(2.0 * jnp.pi))
    return f_lambda * (wave**2 / _C_AA_S)


class TestEquivalentWidthEmission:
    """Positive EW for emission lines."""

    def test_known_gaussian_ew(self, wave):
        """A Gaussian line with total flux A on unit continuum has EW = A."""
        line_flux = 50.0
        l_nu = _flat_continuum_with_emission(wave, 6564.61, line_flux)
        ew = equivalent_width(wave, l_nu, 6564.61, window_aa=30.0, continuum_width_aa=80.0)
        np.testing.assert_allclose(float(ew), line_flux, rtol=0.05)

    def test_emission_positive(self, wave):
        """Emission lines yield positive EW."""
        l_nu = _flat_continuum_with_emission(wave, 6564.61, 100.0)
        ew = equivalent_width(wave, l_nu, 6564.61)
        assert float(ew) > 0.0

    def test_flat_continuum_zero_ew(self, wave):
        """Pure continuum with no line has EW ≈ 0."""
        f_lambda = jnp.ones_like(wave)
        l_nu = f_lambda * (wave**2 / _C_AA_S)
        ew = equivalent_width(wave, l_nu, 6564.61)
        assert abs(float(ew)) < 0.5


class TestEquivalentWidthAbsorption:
    """Negative EW for absorption features."""

    def test_absorption_negative(self, wave):
        """Absorption dip yields negative EW."""
        l_nu = _flat_continuum_with_emission(wave, 6564.61, -30.0)
        ew = equivalent_width(wave, l_nu, 6564.61, window_aa=30.0, continuum_width_aa=80.0)
        assert float(ew) < 0.0


class TestEquivalentWidthJIT:
    """JIT compatibility and differentiability."""

    def test_jit(self, wave):
        l_nu = _flat_continuum_with_emission(wave, 6564.61, 50.0)
        ew = assert_jit_matches_eager(lambda w, s: equivalent_width(w, s, 6564.61), wave, l_nu)
        assert jnp.isfinite(ew)

    def test_gradient(self, wave):
        """Gradient w.r.t. l_nu is finite."""
        l_nu = _flat_continuum_with_emission(wave, 6564.61, 50.0)

        def loss(s):
            return equivalent_width(wave, s, 6564.61)

        grad = assert_grad_matches_fd(loss, l_nu)
        chex.assert_tree_all_finite(grad)
