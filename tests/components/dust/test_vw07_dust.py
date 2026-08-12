# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Wild+2007 two-component dust attenuation curves.

Verifies:
1. vw07_bc and vw07_diff are registered in DUST_LAWS
2. Correct power-law slopes (n=-1.3 and n=-0.7)
3. Normalization at V-band (5500 A)
4. vw07_bc is steeper than vw07_diff in the UV
5. Compatible with two_component_dust framework
6. JIT compatibility and gradient flow
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax.numpy as jnp
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import (
    DUST_LAWS,
    power_law,
    two_component_dust,
    vw07_bc,
    vw07_diff,
)
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


@pytest.fixture
def wavelengths():
    return jnp.linspace(1000.0, 30000.0, 500)


class TestRegistry:
    def test_vw07_bc_registered(self):
        assert "vw07_bc" in DUST_LAWS

    def test_vw07_diff_registered(self):
        assert "vw07_diff" in DUST_LAWS


class TestSlopes:
    def test_vw07_bc_matches_powerlaw_n13(self, wavelengths):
        """vw07_bc should be power_law with n=-1.3."""
        k_vw = vw07_bc(wavelengths)
        k_ref = power_law(wavelengths, n_slope=-1.3)
        assert_allclose(k_vw, k_ref, rtol=1e-12)

    def test_vw07_diff_matches_powerlaw_n07(self, wavelengths):
        """vw07_diff should be power_law with n=-0.7."""
        k_vw = vw07_diff(wavelengths)
        k_ref = power_law(wavelengths, n_slope=-0.7)
        assert_allclose(k_vw, k_ref, rtol=1e-12)

    def test_normalized_at_vband(self):
        """Both curves should be 1.0 at 5500 A."""
        v = jnp.array([5500.0])
        assert_allclose(vw07_bc(v), 1.0, atol=1e-14)
        assert_allclose(vw07_diff(v), 1.0, atol=1e-14)


class TestPhysics:
    def test_bc_steeper_in_uv(self, wavelengths):
        """Birth cloud curve should be steeper (higher attenuation) in UV."""
        uv_mask = wavelengths < 3000.0
        k_bc = vw07_bc(wavelengths)
        k_diff = vw07_diff(wavelengths)
        assert jnp.all(k_bc[uv_mask] > k_diff[uv_mask])

    def test_bc_shallower_in_ir(self, wavelengths):
        """Birth cloud is shallower in the IR (k < 1 with steeper slope)."""
        ir_mask = wavelengths > 8000.0
        k_bc = vw07_bc(wavelengths)
        k_diff = vw07_diff(wavelengths)
        assert jnp.all(k_bc[ir_mask] < k_diff[ir_mask])

    def test_monotonic_decreasing(self, wavelengths):
        """Both curves should decrease monotonically with wavelength."""
        k_bc = vw07_bc(wavelengths)
        k_diff = vw07_diff(wavelengths)
        assert jnp.all(jnp.diff(k_bc) < 0)
        assert jnp.all(jnp.diff(k_diff) < 0)


class TestTwoComponentIntegration:
    def test_two_component_dust_with_vw07(self):
        """VW07 curves should work in the two_component_dust framework."""
        wave = jnp.linspace(1000.0, 25000.0, 200)
        ages = jnp.array([1e6, 5e6, 1e7, 5e7, 1e8, 1e9])
        trans = two_component_dust(
            wave,
            ages,
            tau_v1=1.0,
            tau_v2=0.5,
            law_bc="vw07_bc",
            law_diff="vw07_diff",
        )
        chex.assert_shape(trans, (6, 200))
        chex.assert_tree_all_finite(trans)
        assert_non_negative(trans, name="trans")
        assert jnp.all(trans <= 1.0)

    def test_young_stars_more_attenuated(self):
        """Young stars should be more attenuated (birth cloud + diffuse)."""
        wave = jnp.linspace(1000.0, 25000.0, 200)
        ages = jnp.array([1e6, 1e9])
        trans = two_component_dust(
            wave,
            ages,
            tau_v1=1.0,
            tau_v2=0.5,
            law_bc="vw07_bc",
            law_diff="vw07_diff",
        )
        assert jnp.all(trans[0, :] <= trans[1, :] + 1e-10)


class TestJITAndGradients:
    def test_jit_vw07_bc(self, wavelengths):
        k = assert_jit_matches_eager(vw07_bc, wavelengths)
        chex.assert_tree_all_finite(k)

    def test_jit_vw07_diff(self, wavelengths):
        k = assert_jit_matches_eager(vw07_diff, wavelengths)
        chex.assert_tree_all_finite(k)

    def test_extra_kwargs_ignored(self, wavelengths):
        """Should accept and ignore extra kwargs (like n_slope)."""
        k = vw07_bc(wavelengths, n_slope=999.0, dust_delta=0.5)
        k_ref = vw07_bc(wavelengths)
        assert_allclose(k, k_ref, rtol=1e-14)
