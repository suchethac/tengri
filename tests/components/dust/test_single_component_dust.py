# SPDX-License-Identifier: BSD-3-Clause
"""Tests for single-component (uniform screen) dust attenuation."""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def wavelengths():
    """Rest-frame wavelength grid (Angstrom)."""
    return jnp.linspace(1000.0, 20000.0, 200)


@pytest.fixture
def n_ages():
    return 107


# ── single_component_dust (base, returns 1-D) ─────────────────────
class TestSingleComponentDust:
    """Tests for ``single_component_dust`` (1-D output)."""

    def test_shape(self, wavelengths):
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=1.0)
        chex.assert_equal_shape([trans, wavelengths])

    def test_bounds(self, wavelengths):
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=2.0)
        assert_non_negative(trans, name="trans")
        assert jnp.all(trans <= 1.0)

    def test_zero_tau_is_unity(self, wavelengths):
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=0.0)
        assert jnp.allclose(trans, 1.0)

    def test_more_dust_less_transmission(self, wavelengths):
        from tengri.components.dust.attenuation import single_component_dust

        t1 = single_component_dust(wavelengths, tau_v=0.5)
        t2 = single_component_dust(wavelengths, tau_v=2.0)
        assert jnp.all(t2 <= t1)

    def test_bluer_more_attenuated_power_law(self, wavelengths):
        """For power_law with negative slope, blue < red transmission."""
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=1.0, law="power_law", n_slope=-0.7)
        # Blue end should have lower transmission than red end
        assert trans[0] < trans[-1]

    def test_vband_optical_depth_power_law(self):
        """At 5500A with power_law, tau_v should be recovered exactly."""
        from tengri.components.dust.attenuation import single_component_dust

        tau_v = 1.5
        wave = jnp.array([5500.0])
        trans = single_component_dust(wave, tau_v=tau_v, law="power_law", n_slope=-0.7)
        # power_law at 5500A: k = (5500/5500)^n = 1.0
        expected = jnp.exp(-tau_v)
        assert jnp.allclose(trans, expected, rtol=1e-10)

    def test_f_obscuration_floor(self, wavelengths):
        """With f_obscuration=0.3, transmission floor is 0.3."""
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=100.0, f_obscuration=0.3, law="power_law")
        assert jnp.allclose(trans, 0.3, atol=1e-6)

    def test_f_obscuration_zero_tau(self, wavelengths):
        """With f_obscuration and zero tau, transmission is 1.0."""
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=0.0, f_obscuration=0.5, law="power_law")
        assert jnp.allclose(trans, 1.0)

    @pytest.mark.parametrize(
        "law_name",
        [
            "power_law",
            "calzetti",
            "kriek_conroy",
            "smc",
            "cardelli",
            "li08",
            "salim",
        ],
    )
    def test_all_laws_work(self, wavelengths, law_name):
        from tengri.components.dust.attenuation import single_component_dust

        trans = single_component_dust(wavelengths, tau_v=1.0, law=law_name)
        chex.assert_equal_shape([trans, wavelengths])
        chex.assert_tree_all_finite(trans)
        assert_non_negative(trans, name="trans")
        assert jnp.all(trans <= 1.0)

    def test_jit_compilable(self, wavelengths):
        from tengri.components.dust.attenuation import single_component_dust

        trans = assert_jit_matches_eager(
            lambda w, t: single_component_dust(w, tau_v=t), wavelengths, 1.0
        )
        chex.assert_tree_all_finite(trans)

    def test_differentiable(self, wavelengths):
        from tengri.components.dust.attenuation import single_component_dust

        def loss(tau_v):
            return jnp.sum(single_component_dust(wavelengths, tau_v=tau_v))

        grad_jax = float(jax.grad(loss)(1.0))
        grad_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        # More dust -> less total transmission -> negative gradient
        assert grad_jax < 0.0


# ── single_component_dust_fast (broadcast to n_ages x n_wave) ─────
class TestSingleComponentDustFast:
    """Tests for ``single_component_dust_fast`` (2-D output)."""

    def test_shape(self, wavelengths, n_ages):
        from tengri.components.dust.attenuation import single_component_dust_fast

        trans = single_component_dust_fast(wavelengths, n_ages=n_ages, tau_v=1.0)
        assert trans.shape == (n_ages, len(wavelengths))

    def test_all_ages_identical(self, wavelengths, n_ages):
        """Single-component: every age row must be the same."""
        from tengri.components.dust.attenuation import single_component_dust_fast

        trans = single_component_dust_fast(wavelengths, n_ages=n_ages, tau_v=1.5)
        # All rows identical to first row
        for i in range(1, n_ages):
            assert jnp.allclose(trans[i], trans[0], rtol=1e-12)

    def test_matches_base(self, wavelengths, n_ages):
        """Fast version matches base single_component_dust."""
        from tengri.components.dust.attenuation import (
            single_component_dust,
            single_component_dust_fast,
        )

        base = single_component_dust(wavelengths, tau_v=1.0, law="calzetti")
        fast = single_component_dust_fast(wavelengths, n_ages=n_ages, tau_v=1.0, law="calzetti")
        assert jnp.allclose(fast[0], base, rtol=1e-12)

    def test_bounds(self, wavelengths, n_ages):
        from tengri.components.dust.attenuation import single_component_dust_fast

        trans = single_component_dust_fast(wavelengths, n_ages=n_ages, tau_v=2.0)
        assert_non_negative(trans, name="trans")
        assert jnp.all(trans <= 1.0)

    def test_zero_tau_is_unity(self, wavelengths, n_ages):
        from tengri.components.dust.attenuation import single_component_dust_fast

        trans = single_component_dust_fast(wavelengths, n_ages=n_ages, tau_v=0.0)
        assert jnp.allclose(trans, 1.0)

    def test_jit_compilable(self, wavelengths, n_ages):
        from tengri.components.dust.attenuation import single_component_dust_fast

        trans = assert_jit_matches_eager(
            lambda w, t: single_component_dust_fast(w, n_ages=n_ages, tau_v=t), wavelengths, 1.0
        )
        chex.assert_tree_all_finite(trans)

    def test_differentiable(self, wavelengths, n_ages):
        from tengri.components.dust.attenuation import single_component_dust_fast

        def loss(tau_v):
            return jnp.sum(single_component_dust_fast(wavelengths, n_ages=n_ages, tau_v=tau_v))

        grad_jax = float(jax.grad(loss)(1.0))
        grad_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert grad_jax < 0.0

    @pytest.mark.parametrize(
        "law_name",
        [
            "power_law",
            "calzetti",
            "kriek_conroy",
            "smc",
            "cardelli",
            "li08",
            "salim",
        ],
    )
    def test_all_laws_work(self, wavelengths, n_ages, law_name):
        from tengri.components.dust.attenuation import single_component_dust_fast

        trans = single_component_dust_fast(
            wavelengths,
            n_ages=n_ages,
            tau_v=1.0,
            law=law_name,
        )
        assert trans.shape == (n_ages, len(wavelengths))
        chex.assert_tree_all_finite(trans)
