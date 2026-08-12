# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust emission gradient compatibility.

jax.grad differentiability and finite-difference validation.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.emission import (
    casey2012,
    cmb_corrected_temperature,
    energy_balance_split,
    modified_blackbody,
    planck_bnu,
)
from tests._jit_parity import assert_jit_matches_eager

jax.config.update("jax_enable_x64", True)


pytestmark = pytest.mark.gradient


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def wavelengths():
    """IR wavelength grid (Angstrom), 1 -- 1000 um."""
    return jnp.linspace(1e4, 1e7, 500)


@pytest.fixture
def L_absorbed():
    """Typical absorbed luminosity in Lsun."""
    return 1e10


# ── JIT compatibility ─────────────────────────────────────────────


class TestJITCompatibility:
    """SEDModel is JIT-compilable."""

    def test_jit(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        sed = assert_jit_matches_eager(energy_balance_split, wavelengths, L_absorbed)
        chex.assert_equal_shape([sed, wavelengths])
        chex.assert_tree_all_finite(sed)

    def test_vmap(self, wavelengths):

        L_values = jnp.array([1e9, 5e9, 1e10])
        vmapped = jax.vmap(energy_balance_split, in_axes=(None, 0))
        seds = vmapped(wavelengths, L_values)
        assert seds.shape == (3, len(wavelengths))


# ── Gradient compatibility ────────────────────────────────────────


class TestGradientCompatibility:
    """SEDModel is differentiable w.r.t. all continuous parameters."""

    def test_grad_L_absorbed(self, wavelengths):

        def loss(L_abs):
            sed = energy_balance_split(wavelengths, L_abs)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g_jax = float(grad_fn(1e10))
        g_fd = fd_grad(loss, 1e10)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=2e-2,  # MBB integral over tabulated dust templates; 2% FD agreement is typical
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
        assert g_jax > 0.0

    def test_grad_f_cold(self, wavelengths, L_absorbed):

        def loss(f_cold):
            sed = energy_balance_split(wavelengths, L_absorbed, f_cold=f_cold)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g_jax = float(grad_fn(0.5))
        g_fd = fd_grad(loss, 0.5)
        np.testing.assert_allclose(
            g_jax, g_fd, rtol=1e-3, err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}"
        )

    def test_grad_temperatures(self, wavelengths, L_absorbed):

        def loss_warm(T_warm):
            sed = energy_balance_split(
                wavelengths,
                L_absorbed,
                dust_T_warm=T_warm,
                dust_T_cold=20.0,
            )
            return jnp.sum(sed)

        def loss_cold(T_cold):
            sed = energy_balance_split(
                wavelengths,
                L_absorbed,
                dust_T_warm=45.0,
                dust_T_cold=T_cold,
            )
            return jnp.sum(sed)

        g_warm_jax = float(jax.grad(loss_warm)(45.0))
        g_warm_fd = fd_grad(loss_warm, 45.0)
        np.testing.assert_allclose(
            g_warm_jax,
            g_warm_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_warm_jax:.4e}, FD={g_warm_fd:.4e}",
        )

        g_cold_jax = float(jax.grad(loss_cold)(20.0))
        g_cold_fd = fd_grad(loss_cold, 20.0)
        np.testing.assert_allclose(
            g_cold_jax,
            g_cold_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_cold_jax:.4e}, FD={g_cold_fd:.4e}",
        )

    def test_grad_eta(self, wavelengths, L_absorbed):

        def loss(eta):
            sed = energy_balance_split(wavelengths, L_absorbed, eta_balance=eta)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g_jax = float(grad_fn(1.0))
        g_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            g_jax, g_fd, rtol=1e-3, err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}"
        )
        assert g_jax > 0.0


class TestPlanckBnuGradient:
    """Planck function gradient compatibility."""

    @pytest.fixture
    def wave_ir(self):
        """IR wavelength grid (Angstrom), ~1 – 1000 μm."""
        return jnp.logspace(4, 8, 300)

    def test_jit_compatible(self, wave_ir):
        """planck_bnu is JIT-compilable."""

        from tengri.components.dust.emission import planck_bnu

        bnu = assert_jit_matches_eager(planck_bnu, wave_ir, 30.0)
        chex.assert_tree_all_finite(bnu)

    def test_gradient_wrt_temperature(self, wave_ir):
        """FD check: ∂(∑B_nu)/∂T > 0 (hotter → more luminous)."""

        def loss(T):
            return jnp.sum(planck_bnu(wave_ir, T))

        g_jax = float(jax.grad(loss)(30.0))
        g_fd = fd_grad(loss, 30.0, eps=0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0


class TestModifiedBlackbodyGradient:
    """Modified blackbody gradient compatibility."""

    @pytest.fixture
    def wave_fir(self):
        """Far-IR wavelength grid (Angstrom), 10 μm – 10 mm."""
        return jnp.logspace(5, 9, 400)

    def test_jit_compatible(self, wave_fir):

        from tengri.components.dust.emission import modified_blackbody

        sed = assert_jit_matches_eager(modified_blackbody, wave_fir, 1e10)
        chex.assert_tree_all_finite(sed)

    def test_gradient_wrt_L_absorbed(self, wave_fir):

        def loss(L_abs):
            return jnp.sum(modified_blackbody(wave_fir, L_abs))

        g_jax = float(jax.grad(loss)(1e10))
        g_fd = fd_grad(loss, 1e10, eps=1e7)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_gradient_wrt_temperature(self, wave_fir):

        def loss(T):
            return jnp.sum(modified_blackbody(wave_fir, L_absorbed=1e10, dust_T=T))

        g_jax = float(jax.grad(loss)(30.0))
        g_fd = fd_grad(loss, 30.0, eps=0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-2)


class TestCasey2012Gradient:
    """Casey2012 model gradient compatibility."""

    @pytest.fixture
    def wave_ir(self):
        """Broad IR wavelength grid (Angstrom), 1 μm – 10 mm."""
        return jnp.logspace(4, 9, 500)

    def test_jit_compatible(self, wave_ir):

        from tengri.components.dust.emission import casey2012

        sed = assert_jit_matches_eager(casey2012, wave_ir, 1e10)
        chex.assert_tree_all_finite(sed)

    def test_gradient_wrt_L_absorbed(self, wave_ir):

        def loss(L_abs):
            return jnp.sum(casey2012(wave_ir, L_abs))

        g_jax = float(jax.grad(loss)(1e10))
        g_fd = fd_grad(loss, 1e10, eps=1e7)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_gradient_wrt_temperature(self, wave_ir):

        def loss(T):
            return jnp.sum(casey2012(wave_ir, L_absorbed=1e10, dust_T=T))

        g_jax = float(jax.grad(loss)(35.0))
        g_fd = fd_grad(loss, 35.0, eps=0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-2)


class TestCmbCorrectedTemperatureGradient:
    """CMB temperature correction gradient compatibility."""

    def test_gradient_wrt_T_dust(self):
        """Gradient of T_eff w.r.t. T_dust is well-defined and positive.

        ∂T_eff / ∂T_dust > 0: hotter dust → hotter effective temperature.
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        grad_fn = jax.grad(lambda T: cmb_corrected_temperature(T, redshift=2.0))
        g = float(grad_fn(30.0))
        assert np.isfinite(g)
        assert g > 0.0

    def test_gradient_wrt_redshift(self):
        """Gradient of T_eff w.r.t. redshift is well-defined and positive.

        ∂T_eff / ∂z > 0: higher redshift → higher CMB temperature → higher T_eff.
        """

        grad_fn = jax.grad(lambda z: cmb_corrected_temperature(25.0, redshift=z))
        g = float(grad_fn(2.0))
        assert np.isfinite(g)
        assert g > 0.0


class TestCmbContrastFactorGradient:
    """CMB contrast factor gradient compatibility."""

    def test_gradient_compatible(self):
        """Gradient w.r.t. T_eff is finite and positive."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.logspace(5, 8, 50)

        def loss(T_eff):
            return jnp.sum(cmb_contrast_factor(wave, T_eff=T_eff, redshift=2.0))

        g = float(jax.grad(loss)(30.0))
        assert np.isfinite(g)
        assert g > 0.0  # hotter dust → contrast closer to 1 → sum increases


class TestAbsorbedLuminosityGradient:
    """Absorbed luminosity gradient compatibility."""

    def test_from_tau_gradient(self):
        """Gradient of absorbed luminosity w.r.t. tau is well-defined and positive.

        ∂L_abs / ∂τ > 0: more extinction → more absorption → higher absorbed luminosity.
        """
        from tengri.components.dust.emission import compute_absorbed_luminosity_from_tau

        wave = jnp.linspace(1e3, 1e7, 100)
        L_nu = jnp.ones_like(wave) * 1e9

        def loss(tau_scalar):
            return compute_absorbed_luminosity_from_tau(
                wave, L_nu, jnp.full_like(wave, tau_scalar)
            )

        g = float(jax.grad(loss)(1.0))
        assert np.isfinite(g)
        assert g > 0.0
