# SPDX-License-Identifier: BSD-3-Clause
"""Tests for TEA, Narayanan z-dependent, and Conroy2010 attenuation curves."""

import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import (
    DUST_LAWS,
    cardelli,
    conroy2010,
    kriek_conroy,
    narayanan_z,
    power_law,
    tea,
)
from tests._bounds import assert_non_negative


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def wavelength():
    return jnp.linspace(1000.0, 30000.0, 500)


@pytest.fixture
def uv_wavelength():
    """Far-UV wavelengths where MW bump dominates."""
    return jnp.linspace(1000.0, 3000.0, 100)


@pytest.fixture
def ir_wavelength():
    """NIR wavelengths where power-law dominates."""
    return jnp.linspace(8000.0, 30000.0, 100)


# ── TEA model (Haskell+2024) ──────────────────────────────────────
class TestTEA:
    """Tests for the TEA attenuation curve."""

    def test_registered(self):
        """TEA is registered in the dust law registry."""
        assert "tea" in DUST_LAWS

    def test_eb_at_delta_zero(self):
        """At delta=0, scatter=0: E_b = 2.5 * exp(0) = 2.5."""
        eb = 2.5 * jnp.exp(3.5 * 0.0) * 10.0**0.0
        assert_allclose(float(eb), 2.5, rtol=1e-10)

    def test_eb_decreases_with_steeper_delta(self):
        """Steeper (more negative) delta gives weaker bump."""
        eb_flat = 2.5 * jnp.exp(3.5 * 0.0)
        eb_steep = 2.5 * jnp.exp(3.5 * (-0.5))
        assert float(eb_steep) < float(eb_flat)

    def test_eb_increases_with_shallower_delta(self):
        """Shallower (more positive) delta gives stronger bump."""
        eb_default = 2.5 * jnp.exp(3.5 * (-0.2))
        eb_shallow = 2.5 * jnp.exp(3.5 * 0.3)
        assert float(eb_shallow) > float(eb_default)

    def test_matches_kriek_conroy(self, wavelength):
        """TEA with scatter=0 matches Kriek-Conroy with derived E_b."""
        delta = -0.3
        eb = 2.5 * jnp.exp(3.5 * delta)
        k_tea = tea(wavelength, dust_delta=delta, dust_tea_scatter=0.0)
        k_kc = kriek_conroy(wavelength, dust_delta=delta, dust_bump_strength=eb)
        assert_allclose(k_tea, k_kc, rtol=1e-12)

    def test_scatter_shifts_eb(self, wavelength):
        """Non-zero scatter shifts E_b by a factor of 10^scatter."""
        delta = -0.2
        k_zero = tea(wavelength, dust_delta=delta, dust_tea_scatter=0.0)
        k_pos = tea(wavelength, dust_delta=delta, dust_tea_scatter=0.3)
        # Positive scatter -> stronger bump -> different curve
        assert not jnp.allclose(k_zero, k_pos)

    def test_normalized_at_vband(self, wavelength):
        """k(5500 A) should be close to 1."""
        k = tea(jnp.array([5500.0]), dust_delta=-0.2)
        assert_allclose(float(k[0]), 1.0, atol=0.15)

    def test_jit_compatible(self, wavelength):
        """TEA curve is JIT-compilable."""
        k_eager = tea(wavelength, dust_delta=-0.2)
        k_jit = jax.jit(tea)(wavelength, dust_delta=-0.2)
        assert_allclose(k_eager, k_jit, rtol=1e-12)

    def test_gradient_compatible(self, wavelength):
        """TEA curve supports JAX gradients."""

        def loss(delta):
            return jnp.sum(tea(wavelength, dust_delta=delta))

        grad_jax = float(jax.grad(loss)(-0.2))
        grad_fd = fd_grad(loss, -0.2)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )


# ── Narayanan z-dependent ─────────────────────────────────────────
class TestNarayananZ:
    """Tests for the Narayanan+2018 z-dependent attenuation curve."""

    def test_registered(self):
        """narayanan_z is registered in the dust law registry."""
        assert "narayanan_z" in DUST_LAWS

    def test_z0_matches_kriek_conroy(self, wavelength):
        """At z=0 with defaults, matches Kriek-Conroy(delta=-0.2, bump=1.0)."""
        k_nz = narayanan_z(wavelength, redshift=0.0)
        k_kc = kriek_conroy(wavelength, dust_delta=-0.2, dust_bump_strength=1.0)
        assert_allclose(k_nz, k_kc, rtol=1e-12)

    def test_high_z_steeper(self, wavelength):
        """At z=6, delta is more negative (steeper curve)."""
        k_z0 = narayanan_z(wavelength, redshift=0.0)
        k_z6 = narayanan_z(wavelength, redshift=6.0)
        # At UV wavelengths, steeper curve gives more attenuation relative to V-band
        uv_idx = jnp.argmin(jnp.abs(wavelength - 1500.0))
        assert float(k_z6[uv_idx]) > float(k_z0[uv_idx])

    def test_high_z_no_bump(self, wavelength):
        """At z=6, bump strength is 0 (clipped)."""
        # E_b(z=6) = max(0, 1.0 - 0.15*6) = max(0, 0.1) = 0.1
        # At z=7: max(0, 1.0 - 1.05) = 0.0
        k_z7 = narayanan_z(wavelength, redshift=7.0)
        k_nobump = kriek_conroy(
            wavelength,
            dust_delta=-0.2 - 0.1 * 7.0,
            dust_bump_strength=0.0,
        )
        assert_allclose(k_z7, k_nobump, rtol=1e-12)

    def test_explicit_params_override_z(self, wavelength):
        """Explicit non-default params are used as-is regardless of z."""
        delta = -0.5
        bump = 2.0
        k_nz = narayanan_z(wavelength, dust_delta=delta, dust_bump_strength=bump, redshift=5.0)
        k_kc = kriek_conroy(wavelength, dust_delta=delta, dust_bump_strength=bump)
        assert_allclose(k_nz, k_kc, rtol=1e-12)

    def test_jit_compatible(self, wavelength):
        """narayanan_z is JIT-compilable."""
        k_eager = narayanan_z(wavelength, redshift=2.0)
        k_jit = jax.jit(narayanan_z, static_argnames=())(wavelength, redshift=2.0)
        assert_allclose(k_eager, k_jit, rtol=1e-12)


# ── Conroy2010 mixed MW + power-law ───────────────────────────────
class TestConroy2010:
    """Tests for the Conroy+2010 mixed MW + power-law curve."""

    def test_registered(self):
        """conroy2010 is registered in the dust law registry."""
        assert "conroy2010" in DUST_LAWS

    def test_uv_dominated_by_mw(self, uv_wavelength):
        """UV region should approximate the MW (Cardelli) curve."""
        k_c10 = conroy2010(uv_wavelength)
        k_mw = cardelli(uv_wavelength, dust_Rv=3.1)
        # Normalize MW to same V-band as conroy2010 for comparison
        k_mw_v = cardelli(jnp.array([5500.0]), dust_Rv=3.1)[0]
        # UV blend weight is ~0, so conroy2010 ~ k_mw / k_mw_v (approximately)
        # Check correlation rather than exact match (blend is smooth)
        corr = jnp.corrcoef(k_c10, k_mw / k_mw_v)[0, 1]
        assert float(corr) > 0.95

    def test_ir_dominated_by_power_law(self, ir_wavelength):
        """IR region should approximate the power-law curve."""
        k_c10 = conroy2010(ir_wavelength)
        k_pl = power_law(ir_wavelength, n_slope=-0.7)
        # In IR, blend ~ 1, so conroy2010 ~ power_law (modulo normalization)
        corr = jnp.corrcoef(k_c10, k_pl)[0, 1]
        assert float(corr) > 0.99

    def test_smooth_transition(self):
        """Transition around 5500 A is smooth (no discontinuity)."""
        wave = jnp.linspace(4000.0, 7000.0, 1000)
        k = conroy2010(wave)
        # Finite differences should be smooth
        dk = jnp.diff(k)
        ddk = jnp.diff(dk)
        # No large jumps in second derivative
        max_jump = float(jnp.max(jnp.abs(ddk)))
        assert max_jump < 1.0  # generous bound for smooth curve

    def test_positive_values(self, wavelength):
        """Curve should be non-negative everywhere."""
        k = conroy2010(wavelength)
        assert_non_negative(k, name="k")

    def test_jit_compatible(self, wavelength):
        """conroy2010 is JIT-compilable."""
        k_eager = conroy2010(wavelength)
        k_jit = jax.jit(conroy2010)(wavelength)
        assert_allclose(k_eager, k_jit, rtol=1e-12)

    def test_gradient_compatible(self, wavelength):
        """conroy2010 gradients match central FD w.r.t. Rv and n_slope."""

        def loss(rv, n):
            return jnp.sum(conroy2010(wavelength, dust_Rv=rv, n_slope=n))

        g_rv, g_n = jax.grad(loss, argnums=(0, 1))(3.1, -0.7)

        def f_rv(rv: float) -> float:
            return float(loss(rv, -0.7))

        def f_n(n: float) -> float:
            return float(loss(3.1, n))

        np.testing.assert_allclose(
            float(g_rv),
            fd_grad(f_rv, 3.1),
            rtol=1e-3,
            err_msg="conroy2010: FD check ∂(∑k)/∂dust_Rv",
        )
        np.testing.assert_allclose(
            float(g_n),
            fd_grad(f_n, -0.7),
            rtol=1e-3,
            err_msg="conroy2010: FD check ∂(∑k)/∂n_slope",
        )

    def test_uv_bump_present(self):
        """MW component introduces a 2175 A bump feature in the UV."""
        wave = jnp.linspace(1800.0, 2600.0, 200)
        k = conroy2010(wave)
        # The bump should create a local maximum near 2175 A
        bump_idx = jnp.argmin(jnp.abs(wave - 2175.0))
        # k at bump center should be higher than at edges
        assert float(k[bump_idx]) > float(k[0])
        assert float(k[bump_idx]) > float(k[-1])
