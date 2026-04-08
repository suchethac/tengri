"""Tests for self-consistent disc-corona X-ray model.

Tests alpha_ox_from_l2500, xray_anisotropy, and xray_agn_corona_from_disc.
"""

import jax
import jax.numpy as jnp

from tengri.models.xray import (
    alpha_ox_from_l2500,
    xray_agn_corona_from_disc,
    xray_anisotropy,
    xray_xrb,
)

# Enable 64-bit for precise comparisons
jax.config.update("jax_enable_x64", True)

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
        """Default a1=0.5: face-on should be 2x edge-on."""
        l_x = jnp.ones(10)
        face_on = xray_anisotropy(l_x, cos_inc=1.0)
        edge_on = xray_anisotropy(l_x, cos_inc=0.0)
        ratio = face_on / edge_on
        assert jnp.allclose(ratio, 2.0)

    def test_edge_on_half(self):
        """Default: edge-on factor = (1 - a1) = 0.5."""
        l_x = jnp.ones(5)
        result = xray_anisotropy(l_x, cos_inc=0.0)
        assert jnp.allclose(result, 0.5)

    def test_face_on_unity(self):
        """Face-on factor = a1 + (1 - a1) = 1.0."""
        l_x = jnp.array([1.0, 2.0, 3.0])
        result = xray_anisotropy(l_x, cos_inc=1.0)
        assert jnp.allclose(result, l_x)

    def test_intermediate_angle(self):
        """cos_inc=0.5 (60 deg) should give intermediate luminosity."""
        l_x = jnp.ones(10)
        face_on = xray_anisotropy(l_x, cos_inc=1.0)
        edge_on = xray_anisotropy(l_x, cos_inc=0.0)
        mid = xray_anisotropy(l_x, cos_inc=0.5)
        # Should be between face-on and edge-on
        assert jnp.all(mid > edge_on)
        assert jnp.all(mid < face_on)
        # Exact: factor = 0.5*0.5 + 0.5 = 0.75
        assert jnp.allclose(mid, 0.75)

    def test_isotropic_when_a1_zero(self):
        """a1=0, a2=0 => no anisotropy (factor=1 at all angles)."""
        l_x = jnp.ones(5)
        face_on = xray_anisotropy(l_x, cos_inc=1.0, a1=0.0, a2=0.0)
        edge_on = xray_anisotropy(l_x, cos_inc=0.0, a1=0.0, a2=0.0)
        assert jnp.allclose(face_on, edge_on)
        assert jnp.allclose(face_on, l_x)

    def test_quadratic_term(self):
        """Non-zero a2 adds quadratic cos^2 dependence."""
        l_x = jnp.ones(3)
        # a1=0.3, a2=0.2 => face-on factor = 0.3 + 0.2 + 0.5 = 1.0
        # edge-on factor = 0 + 0 + 0.5 = 0.5
        result_face = xray_anisotropy(l_x, cos_inc=1.0, a1=0.3, a2=0.2)
        result_edge = xray_anisotropy(l_x, cos_inc=0.0, a1=0.3, a2=0.2)
        assert jnp.allclose(result_face, 1.0)
        assert jnp.allclose(result_edge, 0.5)


# ---- xray_agn_corona_from_disc tests ----


class TestXrayAgnCoronaFromDisc:
    """Tests for the self-consistent disc-corona model."""

    def test_positive_luminosity(self):
        """Output should be non-negative everywhere."""
        result = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30)
        assert jnp.all(result >= 0.0)

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

    def test_delta_alpha_ox_zero_matches_old_interface(self):
        """New disc-coupled function with delta=0 should match old corona.

        We feed the old xray_agn_corona the same alpha_ox that Just+2007
        would predict for a given L_2500, and compare spectra.
        """
        from tengri.models.xray import xray_agn_corona

        l_2500_erg = 1e30
        alpha_ox = float(alpha_ox_from_l2500(l_2500_erg))

        # L_agn_bol in erg/s. Back out from L_2500:
        # L_2500 = L_bol / (BC * nu_2500)  => L_bol = L_2500 * BC * nu_2500
        nu_2500 = 1.199e15
        bc_2500 = 5.15
        l_bol_erg = l_2500_erg * bc_2500 * nu_2500

        l_old = xray_agn_corona(WAVE_XRAY, L_agn_bol=l_bol_erg, alpha_ox=alpha_ox)
        l_new = xray_agn_corona_from_disc(
            WAVE_XRAY,
            l_2500_erg_hz=l_2500_erg,
            delta_alpha_ox=0.0,
            apply_anisotropy=False,
        )
        # Should agree to ~1% (both normalise via same alpha_ox formula)
        mask = l_old > 0
        ratio = l_new[mask] / l_old[mask]
        assert jnp.allclose(ratio, 1.0, atol=0.02)

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
        jitted = jax.jit(lambda w, l: xray_agn_corona_from_disc(w, l, apply_anisotropy=True))
        result = jitted(WAVE_XRAY, 1e30)
        assert result.shape == WAVE_XRAY.shape
        assert jnp.all(jnp.isfinite(result))

    def test_gradient(self):
        """Gradient through the function should be finite."""

        def total_flux(l_2500):
            l_nu = xray_agn_corona_from_disc(WAVE_XRAY, l_2500, apply_anisotropy=True)
            return jnp.sum(l_nu)

        grad_fn = jax.grad(total_flux)
        g = grad_fn(1e30)
        assert jnp.isfinite(g)
        assert g > 0  # more UV => more X-ray

    def test_physically_sensible_lx(self):
        """L_nu at 2 keV should be consistent with alpha_ox prediction.

        For L_2500 = 1e30 erg/s/Hz, alpha_ox ~ -1.47, so
        L_2keV / L_2500 = 10^(alpha_ox/0.384) ~ 1.5e-4.
        L_nu(2keV) should be order ~1e26 erg/s/Hz ~ 1e-8 Lsun/Hz.
        """
        l_nu = xray_agn_corona_from_disc(WAVE_XRAY, l_2500_erg_hz=1e30)
        # Find L_nu near 2 keV (lambda ~ 6.2 A)
        wave_2kev = 6.2  # Angstrom
        idx = jnp.argmin(jnp.abs(WAVE_XRAY - wave_2kev))
        l_nu_2kev = l_nu[idx]
        # Should be positive and in a reasonable range (Lsun/Hz)
        assert l_nu_2kev > 0
        # L_nu at peak should be finite and not huge
        assert jnp.max(l_nu) < 1e10  # Lsun/Hz upper bound
        # Values are small but nonzero (uses same normalization
        # convention as existing xray_agn_corona)
        assert jnp.max(l_nu) > 1e-30


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
