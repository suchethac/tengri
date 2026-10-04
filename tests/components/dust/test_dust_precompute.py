# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the fast dust attenuation entry point.

``two_component_dust_fast`` takes the per-node young mass fraction (the
stellar component's ``age_boundary_younger_fraction``) directly and mixes the
young and old populations' transmissions; it must agree with
``two_component_dust`` everywhere, which is the one definition.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import (
    two_component_dust,
    two_component_dust_fast,
)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient of scalar f at x."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def age_grid():
    """Typical SSP age grid (107 points, log-spaced)."""
    return 10.0 ** jnp.linspace(5.5, 10.14, 107)


@pytest.fixture
def filter_wavelengths():
    """Effective wavelengths for 5 SDSS bands (rest-frame Angstrom)."""
    return jnp.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])


@pytest.fixture
def spectral_wavelengths():
    """200 spectral pixel wavelengths for spectroscopy tests."""
    return jnp.linspace(3500.0, 9500.0, 200)


@pytest.fixture
def dust_age_weights(age_grid):
    """Per-node young mass fraction (hard step at 10 Myr; nodes wholly young or old)."""
    return (age_grid < 1e7).astype(float)


# ── charlot_fall_at_wavelengths_fast vs charlot_fall_at_wavelengths
_CF_KWARGS = {"law_bc": "power_law", "law_diff": "power_law"}


class TestFastDustAgreement:
    """Fast dust must exactly match the original per-call version."""

    def test_exact_agreement_photometry(self, age_grid, filter_wavelengths, dust_age_weights):
        """Fast and original agree exactly for photometric wavelengths."""
        tau_v1, tau_v2, dust_slope = 0.5, 0.3, -0.7
        result_original = two_component_dust(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            dust_slope=dust_slope,
            **_CF_KWARGS,
        )
        result_fast = two_component_dust_fast(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            dust_slope=dust_slope,
            **_CF_KWARGS,
        )
        assert_allclose(result_fast, result_original, rtol=1e-12)

    def test_exact_agreement_spectroscopy(self, age_grid, spectral_wavelengths, dust_age_weights):
        """Fast and original agree exactly for spectroscopic wavelengths."""
        tau_v1, tau_v2, dust_slope = 1.0, 0.5, -0.7
        result_original = two_component_dust(
            spectral_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            dust_slope=dust_slope,
            **_CF_KWARGS,
        )
        result_fast = two_component_dust_fast(
            spectral_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            dust_slope=dust_slope,
            **_CF_KWARGS,
        )
        assert_allclose(result_fast, result_original, rtol=1e-12)

    @pytest.mark.parametrize(
        "tau_v1,tau_v2,dust_slope",
        [
            (0.0, 0.0, -0.7),  # no dust
            (3.0, 1.5, -0.7),  # heavy dust
            (0.5, 0.3, -1.3),  # steep curve (Calzetti-like)
            (0.5, 0.3, -0.3),  # shallow curve
            (0.01, 0.01, -0.7),  # nearly no dust
        ],
    )
    def test_agreement_various_params(
        self,
        age_grid,
        filter_wavelengths,
        dust_age_weights,
        tau_v1,
        tau_v2,
        dust_slope,
    ):
        """Agreement holds across diverse dust parameter combinations."""
        result_original = two_component_dust(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            dust_slope=dust_slope,
            **_CF_KWARGS,
        )
        result_fast = two_component_dust_fast(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            dust_slope=dust_slope,
            **_CF_KWARGS,
        )
        assert_allclose(result_fast, result_original, rtol=1e-12)

    def test_output_shape(self, filter_wavelengths, dust_age_weights):
        """Output shape is (n_ages, n_filters)."""
        result = two_component_dust_fast(
            filter_wavelengths,
            dust_age_weights,
            tau_v1=0.5,
            tau_v2=0.3,
            **_CF_KWARGS,
        )
        assert result.shape == (len(dust_age_weights), len(filter_wavelengths))


# ── Gradient tests ────────────────────────────────────────────────
class TestFastDustGradients:
    """Gradients through the fast dust function are correct."""

    def test_gradients_finite(self, filter_wavelengths, dust_age_weights):
        """Gradients of two_component_dust_fast match central FD."""

        def loss(tau_v1, tau_v2):
            atten = two_component_dust_fast(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )
            return jnp.sum(atten)

        g1, g2 = jax.grad(loss, argnums=(0, 1))(0.5, 0.3)

        def f1(tau_v1: float) -> float:
            return float(loss(tau_v1, 0.3))

        def f2(tau_v2: float) -> float:
            return float(loss(0.5, tau_v2))

        np.testing.assert_allclose(
            float(g1),
            fd_grad(f1, 0.5),
            rtol=1e-3,
            err_msg="two_component_dust_fast: FD check ∂(∑atten)/∂tau_bc",
        )
        np.testing.assert_allclose(
            float(g2),
            fd_grad(f2, 0.3),
            rtol=1e-3,
            err_msg="two_component_dust_fast: FD check ∂(∑atten)/∂tau_diff",
        )

    def test_gradients_match_original(self, age_grid, filter_wavelengths, dust_age_weights):
        """Autodiff gradients match between fast and original."""

        def loss_original(tau_v1, tau_v2):
            atten = two_component_dust(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )
            return jnp.sum(atten)

        def loss_fast(tau_v1, tau_v2):
            atten = two_component_dust_fast(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )
            return jnp.sum(atten)

        g_orig = jax.grad(loss_original, argnums=(0, 1))(0.5, 0.3)
        g_fast = jax.grad(loss_fast, argnums=(0, 1))(0.5, 0.3)
        assert_allclose(g_fast[0], g_orig[0], rtol=1e-10)
        assert_allclose(g_fast[1], g_orig[1], rtol=1e-10)

    def test_jit_compatible(self, filter_wavelengths, dust_age_weights):
        """Fast dust function works inside jax.jit."""

        @jax.jit
        def fn(tau_v1, tau_v2):
            return two_component_dust_fast(
                filter_wavelengths,
                dust_age_weights,
                tau_v1=tau_v1,
                tau_v2=tau_v2,
                **_CF_KWARGS,
            )

        result = fn(0.5, 0.3)
        chex.assert_tree_all_finite(result)
