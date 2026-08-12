# SPDX-License-Identifier: BSD-3-Clause
"""Tests for self-consistent disc-corona X-ray model.

Tests alpha_ox_from_l2500, xray_anisotropy, and xray_agn_corona_from_disc.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.xray import (
    alpha_ox_from_l2500,
    xray_agn_corona_from_disc,
    xray_anisotropy,
    xray_xrb,
)
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# X-ray wavelength grid: 0.1-124 A (0.1-100 keV)
WAVE_XRAY = jnp.linspace(0.5, 120.0, 200)


# ---- alpha_ox_from_l2500 tests ----
class TestAlphaOxFromL2500:
    """Tests for the Just+2007 alpha_ox relation."""

    def test_known_value_l2500_1e30(self):
        """L_2500 = 1e30 erg/s/Hz => alpha_ox ~ -1.47."""
        alpha_ox = alpha_ox_from_l2500(1e30)
        assert jnp.isclose(alpha_ox, -1.472, atol=0.01)

    def test_known_value_l2500_1e31(self):
        """L_2500 = 1e31 erg/s/Hz => alpha_ox ~ -1.609."""
        # -0.137 * 31 + 2.638 = -4.247 + 2.638 = -1.609
        alpha_ox = alpha_ox_from_l2500(1e31)
        assert jnp.isclose(alpha_ox, -1.609, atol=0.001)

    def test_monotonic_decrease(self):
        """Brighter disc => more negative alpha_ox (X-ray quieter)."""
        l_values = jnp.array([1e28, 1e29, 1e30, 1e31, 1e32])
        alpha_values = jnp.array([alpha_ox_from_l2500(lv) for lv in l_values])
        # Each should be more negative than the previous
        diffs = jnp.diff(alpha_values)
        assert jnp.all(diffs < 0)

    def test_returns_scalar(self):
        """Output should be a scalar, not an array."""
        result = alpha_ox_from_l2500(1e30)
        assert result.ndim == 0


# ---- xray_anisotropy tests ----
class TestXrayAnisotropy:
    """Tests for viewing-angle X-ray anisotropy (Yang+2022)."""

    def test_face_on_double_edge_on(self):
        """Default a1=0.5: face-on should be 2x edge-on (denominator cancels).

        With denominator normalization (yang20.py:231-235):
            f(θ=0°) = [0.5 + 0 + 0.5] / [1 - 0.13397*0.5 - 0] = 1.0 / 0.933
            f(θ=90°) = [0 + 0 + 0.5] / [1 - 0.13397*0.5 - 0] = 0.5 / 0.933
            ratio = 1.0 / 0.5 = 2.0 (denominator cancels)
        """
        l_x = jnp.ones(10)
        face_on = xray_anisotropy(l_x, cos_inc=1.0)
        edge_on = xray_anisotropy(l_x, cos_inc=0.0)
        ratio = face_on / edge_on
        assert jnp.allclose(ratio, 2.0)

    def test_edge_on_half(self):
        """Default: edge-on factor = (1 - a1) / denom = 0.5 / 0.933 ≈ 0.536."""
        l_x = jnp.ones(5)
        result = xray_anisotropy(l_x, cos_inc=0.0)
        # (1 - 0.5) / (1 - 0.13397*0.5) = 0.5 / 0.933015 ≈ 0.536
        expected = 0.5 / (1.0 - 0.13397 * 0.5)
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_face_on_unity(self):
        """Face-on factor = 1.0 / denom ≈ 1.072 (with denominator normalization).

        The denominator (yang20.py:233) normalizes so bolometric corona
        luminosity at θ=0° is recovered:
            f(θ=0°) = [0.5 + 0 + 0.5] / [1 - 0.13397*0.5 - 0]
                    = 1.0 / 0.933015 ≈ 1.072
        """
        l_x = jnp.array([1.0, 2.0, 3.0])
        result = xray_anisotropy(l_x, cos_inc=1.0)
        # 1.0 / (1 - 0.13397*0.5) ≈ 1.072
        expected_factor = 1.0 / (1.0 - 0.13397 * 0.5)
        expected = l_x * expected_factor
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_intermediate_angle(self):
        """cos_inc=0.5 (60 deg) should give intermediate luminosity.

        With denominator normalization:
            f(θ, cos(θ)=0.5) = [0.5*0.5 + 0 + 0.5] / [1 - 0.13397*0.5]
                            = 0.75 / 0.933015 ≈ 0.804
        """
        l_x = jnp.ones(10)
        face_on = xray_anisotropy(l_x, cos_inc=1.0)
        edge_on = xray_anisotropy(l_x, cos_inc=0.0)
        mid = xray_anisotropy(l_x, cos_inc=0.5)
        # Should be between face-on and edge-on
        assert jnp.all(mid > edge_on)
        assert jnp.all(mid < face_on)
        # Exact with denominator: 0.75 / 0.933
        denom = 1.0 - 0.13397 * 0.5
        expected = (0.5 * 0.5 + 0.5) / denom
        assert jnp.allclose(mid, expected, rtol=1e-6)

    def test_isotropic_when_a1_zero(self):
        """a1=0, a2=0 => no anisotropy (factor=1 at all angles)."""
        l_x = jnp.ones(5)
        face_on = xray_anisotropy(l_x, cos_inc=1.0, a1=0.0, a2=0.0)
        edge_on = xray_anisotropy(l_x, cos_inc=0.0, a1=0.0, a2=0.0)
        assert jnp.allclose(face_on, edge_on)
        assert jnp.allclose(face_on, l_x)

    def test_quadratic_term(self):
        """Non-zero a2 adds quadratic cos^2 dependence (with denominator).

        With a1=0.3, a2=0.2:
            denom = 1 - 0.13397*0.3 - 0.25*0.2 = 1 - 0.04019 - 0.05 = 0.90981
            f(θ=0°) = (0.3 + 0.2 + 0.5) / 0.90981 = 1.0 / 0.90981 ≈ 1.099
            f(θ=90°) = (0 + 0 + 0.5) / 0.90981 = 0.5 / 0.90981 ≈ 0.550
        """
        l_x = jnp.ones(3)
        a1, a2 = 0.3, 0.2
        denom = 1.0 - 0.13397 * a1 - 0.25 * a2
        result_face = xray_anisotropy(l_x, cos_inc=1.0, a1=a1, a2=a2)
        result_edge = xray_anisotropy(l_x, cos_inc=0.0, a1=a1, a2=a2)
        expected_face = 1.0 / denom
        expected_edge = 0.5 / denom
        assert jnp.allclose(result_face, expected_face * l_x, rtol=1e-6)
        assert jnp.allclose(result_edge, expected_edge * l_x, rtol=1e-6)


# ---- xray_agn_corona_from_disc tests ----
class TestXrayAgnCoronaFromDisc:
    """Tests for the self-consistent disc-corona model."""

    def test_positive_luminosity(self):
        """Output should be non-negative everywhere."""
        result = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30)
        assert_non_negative(result, name="result")

    def test_zero_outside_xray(self):
        """L_nu should be zero at wavelengths > 124 A (outside X-ray)."""
        wave_uv = jnp.linspace(200.0, 5000.0, 50)
        result = xray_agn_corona_from_disc(wave_uv, l_2500_erg_hz=1e30)
        assert jnp.allclose(result, 0.0)

    def test_brighter_disc_more_xray(self):
        """Higher L_2500 should produce more X-ray luminosity."""
        l_low = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e29)
        l_high = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e31)
        # Integrate over the X-ray band
        total_low = jnp.trapezoid(l_low, WAVE_XRAY)
        total_high = jnp.trapezoid(l_high, WAVE_XRAY)
        assert total_high > total_low

    def test_positive_delta_alpha_ox_increases_xray(self):
        """Positive delta_alpha_ox => X-ray louder => more L_X."""
        l_base = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30, delta_alpha_ox=0.0)
        l_loud = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30, delta_alpha_ox=0.3)
        total_base = jnp.trapezoid(l_base, WAVE_XRAY)
        total_loud = jnp.trapezoid(l_loud, WAVE_XRAY)
        assert total_loud > total_base

    def test_anisotropy_reduces_edge_on(self):
        """Edge-on viewing should reduce X-ray flux vs face-on."""
        l_face = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30, cos_inc=1.0)
        l_edge = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30, cos_inc=0.0)
        total_face = jnp.trapezoid(l_face, WAVE_XRAY)
        total_edge = jnp.trapezoid(l_edge, WAVE_XRAY)
        assert total_face > total_edge
        # Default a1=0.5: face-on should be 2x edge-on
        ratio = total_face / total_edge
        assert jnp.isclose(ratio, 2.0, atol=0.01)

    def test_no_anisotropy_flag(self):
        """apply_anisotropy=False should give same result for all angles."""
        l_face = xray_agn_corona_from_disc(
            WAVE_XRAY,
            l_2500_erg_hz=1e30,
            cos_inc=1.0,
            apply_anisotropy=False,
        )
        l_edge = xray_agn_corona_from_disc(
            WAVE_XRAY,
            l_2500_erg_hz=1e30,
            cos_inc=0.0,
            apply_anisotropy=False,
        )
        assert jnp.allclose(l_face, l_edge)

    def test_jit_compilation(self):
        """Function should be JIT-compilable."""
        result = assert_jit_matches_eager(
            lambda w, l: xray_agn_corona_from_disc(w, l, apply_anisotropy=True), WAVE_XRAY, 1e30
        )
        chex.assert_equal_shape([result, WAVE_XRAY])
        chex.assert_tree_all_finite(result)

    def test_gradient(self):
        """Gradient through the function agrees with FD (more UV → more X-ray)."""

        def total_flux(l_2500):
            l_nu = xray_agn_corona_from_disc(WAVE_XRAY, l_2500, apply_anisotropy=True)
            return jnp.sum(l_nu)

        x0 = 1e30
        grad_jax = float(jax.grad(total_flux)(x0))

        def f_scalar(l_2500: float) -> float:
            return float(total_flux(l_2500))

        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f_scalar, x0, eps=1e24),
            rtol=1e-3,
            err_msg="xray_agn_corona_from_disc: FD check ∂(∑L_ν)/∂L_2500",
        )
        assert grad_jax > 0  # more UV => more X-ray

    def test_physically_sensible_lx(self):
        """L_nu at 2 keV should be consistent with alpha_ox prediction.
        For L_2500 = 1e30 erg/s/Hz, alpha_ox ~ -1.47, so
        L_2keV / L_2500 = 10^(alpha_ox/0.384) ~ 1.5e-4.
        L_nu(2keV) should be order ~1.5e26 erg/s/Hz.
        """
        l_nu = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30)
        wave_2kev = 6.2  # Angstrom
        idx = jnp.argmin(jnp.abs(WAVE_XRAY - wave_2kev))
        l_nu_2kev = float(l_nu[idx])
        assert 1e25 < l_nu_2kev < 1e27, (
            f"L_nu(2 keV) = {l_nu_2kev:.2e} erg/s/Hz, "
            "expected ~1.5e26 from alpha_ox = 0.384 log10(L_2keV/L_2500)"
        )
        # Peak of X-ray band must remain in a physical erg/s/Hz range.
        lmax = float(jnp.max(l_nu))
        assert 1e25 < lmax < 1e30, f"max L_nu = {lmax:.2e} erg/s/Hz"


# ---- xray_xrb offset tests ----
class TestXrbOffsets:
    """Verify that log_L_hmxb_offset and log_L_lmxb_offset work."""

    def test_hmxb_offset_increases_luminosity(self):
        """Positive HMXB offset should increase luminosity."""
        l_base = xray_xrb(WAVE_XRAY, sfr=10.0, stellar_mass=1e10)
        l_high = xray_xrb(WAVE_XRAY, sfr=10.0, stellar_mass=1e10, log_L_hmxb_offset=0.5)
        total_base = jnp.sum(l_base)
        total_high = jnp.sum(l_high)
        assert total_high > total_base

    def test_lmxb_offset_increases_luminosity(self):
        """Positive LMXB offset should increase luminosity."""
        l_base = xray_xrb(WAVE_XRAY, sfr=1.0, stellar_mass=1e11)
        l_high = xray_xrb(WAVE_XRAY, sfr=1.0, stellar_mass=1e11, log_L_lmxb_offset=0.5)
        total_base = jnp.sum(l_base)
        total_high = jnp.sum(l_high)
        assert total_high > total_base

    def test_zero_offset_unchanged(self):
        """Zero offsets should not change result."""
        l_default = xray_xrb(WAVE_XRAY, sfr=5.0, stellar_mass=1e10)
        l_explicit = xray_xrb(
            WAVE_XRAY,
            sfr=5.0,
            stellar_mass=1e10,
            log_L_hmxb_offset=0.0,
            log_L_lmxb_offset=0.0,
        )
        assert jnp.allclose(l_default, l_explicit)
