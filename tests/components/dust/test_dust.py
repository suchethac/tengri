# SPDX-License-Identifier: BSD-3-Clause
"""Tests for two-component dust attenuation model."""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import two_component_dust
from tests._jit_parity import assert_jit_matches_eager

_CF_KWARGS = {"law_bc": "power_law", "law_diff": "power_law"}


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient of scalar f at x."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def _charlot_fall_hard(wavelength, age_grid, tau_v1, tau_v2, n_slope=-0.7, t_birth=1e7):
    """Step-function variant (for comparison/testing)."""
    wave_ratio = (wavelength / 5500.0) ** n_slope
    tau_young = (tau_v1 + tau_v2) * wave_ratio
    tau_old = tau_v2 * wave_ratio
    is_young = age_grid[:, None] < t_birth
    tau_lambda = jnp.where(is_young, tau_young[None, :], tau_old[None, :])
    return jnp.exp(-tau_lambda)


class TestCharlotFall:
    """Tests for the smooth two-component dust model (power-law curves)."""

    @pytest.fixture
    def wavelength(self):
        return jnp.linspace(1000, 10000, 100)

    @pytest.fixture
    def age_grid(self):
        return jnp.logspace(6, 10, 50)

    def test_output_shape(self, wavelength, age_grid):
        """Output has shape (n_ages, n_wave)."""
        atten = two_component_dust(wavelength, age_grid, 0.5, 0.3, **_CF_KWARGS)
        chex.assert_shape(atten, (50, 100))

    def test_values_between_0_and_1(self, wavelength, age_grid):
        """Attenuation factor is in [0, 1]."""
        atten = two_component_dust(wavelength, age_grid, 1.0, 0.5, **_CF_KWARGS)
        assert jnp.all(atten >= 0)
        assert jnp.all(atten <= 1)

    def test_no_dust_gives_unity(self, wavelength, age_grid):
        """Zero optical depth gives transmission = 1."""
        atten = two_component_dust(wavelength, age_grid, 0.0, 0.0, **_CF_KWARGS)
        assert_allclose(atten, 1.0, rtol=1e-10)

    def test_more_dust_less_transmission(self, wavelength, age_grid):
        """Higher tau_v gives lower transmission."""
        atten_low = two_component_dust(wavelength, age_grid, 0.3, 0.1, **_CF_KWARGS)
        atten_high = two_component_dust(wavelength, age_grid, 1.5, 0.5, **_CF_KWARGS)
        # Average transmission should be lower with more dust
        assert float(jnp.mean(atten_high)) < float(jnp.mean(atten_low))

    def test_young_stars_more_attenuated(self, wavelength, age_grid):
        """Young stars (< 10 Myr) are more attenuated than old stars."""
        atten = two_component_dust(wavelength, age_grid, 1.0, 0.3, **_CF_KWARGS)
        young_mask = age_grid < 1e7
        old_mask = age_grid > 1e8
        mean_young = float(jnp.mean(atten[young_mask]))
        mean_old = float(jnp.mean(atten[old_mask]))
        assert mean_young < mean_old

    def test_bluer_wavelengths_more_attenuated(self, age_grid):
        """Blue light is more attenuated than red (for n < 0)."""
        wave = jnp.array([3000.0, 5500.0, 10000.0])
        atten = two_component_dust(wave, age_grid, 0.5, 0.3, n_slope=-0.7, **_CF_KWARGS)
        # Average over ages: blue should be more attenuated
        mean_per_wave = jnp.mean(atten, axis=0)
        assert float(mean_per_wave[0]) < float(mean_per_wave[2])

    def test_v_band_optical_depth(self, age_grid):
        """At 5500 A, tau_lambda = tau_v for old stars (diffuse only)."""
        wave = jnp.array([5500.0])
        tau_v2 = 0.5
        atten = two_component_dust(wave, age_grid, 0.0, tau_v2, n_slope=-0.7, **_CF_KWARGS)
        # For old stars, tau_lambda = tau_v2 * (5500/5500)^n = tau_v2
        old_atten = float(atten[-1, 0])
        expected = float(jnp.exp(-tau_v2))
        assert_allclose(old_atten, expected, rtol=0.01)

    def test_is_jittable(self, wavelength, age_grid):
        """two_component_dust is JIT-compatible."""
        atten = assert_jit_matches_eager(
            lambda w, a, tv1, tv2: two_component_dust(w, a, tv1, tv2, **_CF_KWARGS),
            wavelength,
            age_grid,
            0.5,
            0.3,
        )
        chex.assert_shape(atten, (50, 100))

    def test_has_gradients(self, wavelength, age_grid):
        """Gradients of two_component_dust match central FD w.r.t. tau_v1 and tau_v2."""

        def loss(tau_v1, tau_v2):
            return jnp.sum(two_component_dust(wavelength, age_grid, tau_v1, tau_v2, **_CF_KWARGS))

        g1, g2 = jax.grad(loss, argnums=(0, 1))(0.5, 0.3)

        def f1(tau_v1: float) -> float:
            return float(loss(tau_v1, 0.3))

        def f2(tau_v2: float) -> float:
            return float(loss(0.5, tau_v2))

        np.testing.assert_allclose(
            float(g1),
            fd_grad(f1, 0.5),
            rtol=1e-3,
            err_msg="two_component_dust: FD check ∂(∑atten)/∂tau_bc",
        )
        np.testing.assert_allclose(
            float(g2),
            fd_grad(f2, 0.3),
            rtol=1e-3,
            err_msg="two_component_dust: FD check ∂(∑atten)/∂tau_diff",
        )


class TestCharlotFallHard:
    """Tests for the step-function variant (inlined reference implementation)."""

    def test_agrees_with_smooth_away_from_transition(self):
        """Hard and smooth agree far from t_birth."""
        wave = jnp.linspace(3000, 8000, 50)
        # Ages well below and well above 10 Myr
        ages = jnp.array([1e5, 1e6, 1e9, 1e10])
        atten_smooth = two_component_dust(wave, ages, 0.5, 0.3, **_CF_KWARGS)
        atten_hard = _charlot_fall_hard(wave, ages, 0.5, 0.3)
        # Should agree at very young and very old ages
        assert_allclose(atten_smooth[0], atten_hard[0], rtol=0.05)
        assert_allclose(atten_smooth[-1], atten_hard[-1], rtol=0.01)
