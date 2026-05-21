"""Tests for X-ray emission module (models/xray.py).

Physics references:
- Grimm+2003, MNRAS 339, 793 (HMXB-SFR relation)
- Gilfanov 2004, MNRAS 349, 146 (LMXB-mass relation)
- Ranalli+2003, A&A 399, 39 (combined XRB calibration)
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.xray import xray_agn_corona, xray_total, xray_xrb
from tengri.utils.physics_constants import C_AA as _C_AA, KEV_TO_HZ as _KEV_TO_HZ


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


jax.config.update("jax_enable_x64", True)

# 2-10 keV wavelength/frequency grid for band integration
_E_GRID = jnp.linspace(2.0, 10.0, 500)  # keV
_NU_GRID = _E_GRID * _KEV_TO_HZ  # Hz
_WAVE_GRID = _C_AA / _NU_GRID  # Angstrom (x-ray)


class TestXRBNormalization:
    def test_hmxb_sfr_scaling(self):
        """HMXB luminosity scales linearly with SFR. Grimm+2003 MNRAS 339 Eq. 3."""
        L1 = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=0.0), _NU_GRID))
        L2 = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=2.0, stellar_mass=0.0), _NU_GRID))
        np.testing.assert_allclose(
            L2 / L1,
            2.0,
            rtol=0.01,
            err_msg="Grimm+2003: HMXB L_X ∝ SFR^1.0",
        )

    def test_lmxb_mass_scaling(self):
        """LMXB luminosity scales linearly with stellar mass. Gilfanov 2004 MNRAS 349 Eq. 1."""
        L1 = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=0.0, stellar_mass=1e10), _NU_GRID))
        L2 = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=0.0, stellar_mass=2e10), _NU_GRID))
        np.testing.assert_allclose(
            L2 / L1,
            2.0,
            rtol=0.01,
            err_msg="Gilfanov 2004: LMXB L_X ∝ M_star^1.0",
        )

    def test_hmxb_band_luminosity(self):
        """HMXB 2-10 keV luminosity at SFR=1 Msun/yr. Grimm+2003 Eq. 3: L=2.6e39 erg/s."""
        L_band = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=0.0), _NU_GRID))
        np.testing.assert_allclose(
            L_band,
            2.6e39,
            rtol=0.05,
            err_msg="Grimm+2003: HMXB L_2-10keV = 2.6e39 erg/s at SFR=1 Msun/yr",
        )

    def test_lmxb_band_luminosity(self):
        """LMXB 2-10 keV luminosity at M*=1e10 Msun. Gilfanov 2004: L=8.3e38 erg/s."""
        L_band = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=0.0, stellar_mass=1e10), _NU_GRID))
        np.testing.assert_allclose(
            L_band,
            8.3e38,
            rtol=0.05,
            err_msg="Gilfanov 2004: LMXB L_2-10keV = 8.3e28 erg/s/Msun × 1e10 Msun = 8.3e38 erg/s",
        )

    def test_combined_ranalli2003(self):
        """Combined XRB at SFR=1, M*=1e10 within 30% of Ranalli+2003 A&A 399 Eq. 3."""
        L_band = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=1e10), _NU_GRID))
        # Ranalli+2003 combined calibration: L_2-10keV ≈ 3.7e39 erg/s at SFR=1, M*=1e10
        # tengri uses Grimm+2003 (HMXB: 2.6e39) + Gilfanov 2004 (LMXB: 8.3e38 at 1e10)
        # combined ≈ 3.43e39, within 30% of Ranalli
        np.testing.assert_allclose(
            L_band,
            3.7e39,
            rtol=0.30,
            err_msg="Ranalli+2003 A&A 399 Eq. 3: L_2-10keV ≈ 3.7e39 erg/s at SFR=1, M*=1e10",
        )

    def test_xray_only_mask(self):
        """XRB emission is zero at optical wavelengths (> 124 Å = 0.1 keV)."""
        wave_opt = jnp.array([5500.0, 10000.0])  # optical Angstrom
        L_opt = xray_xrb(wave_opt, sfr=1.0, stellar_mass=1e10)
        assert float(jnp.sum(jnp.abs(L_opt))) == 0.0, (
            "XRB: L_nu must be zero at optical wavelengths (λ > 124 Å)"
        )

    def test_gradient_wrt_sfr(self):
        """FD check: ∂(∑ L_xrb)/∂sfr at sfr=1.0. Grimm+2003 linear calibration."""

        def f(sfr):
            return float(xray_xrb(_WAVE_GRID, sfr=sfr, stellar_mass=1e10).sum())

        grad_jax = float(
            jax.grad(lambda sfr: xray_xrb(_WAVE_GRID, sfr=sfr, stellar_mass=1e10).sum())(1.0)
        )
        grad_fd = fd_grad(f, 1.0, eps=0.01)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="xray_xrb: autodiff vs FD gradient w.r.t. sfr",
        )

    def test_gradient_wrt_stellar_mass(self):
        """FD check: ∂(∑ L_xrb)/∂stellar_mass. Gilfanov 2004 linear calibration."""

        def f(m):
            return float(xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=m).sum())

        grad_jax = float(
            jax.grad(lambda m: xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=m).sum())(1e10)
        )
        grad_fd = fd_grad(f, 1e10, eps=1e7)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="xray_xrb: autodiff vs FD gradient w.r.t. stellar_mass",
        )

    def test_jit_compatible(self):
        """xray_xrb is JIT-compilable."""
        f = jax.jit(lambda sfr: xray_xrb(_WAVE_GRID, sfr=sfr, stellar_mass=1e10).sum())
        result = float(f(1.0))
        assert jnp.isfinite(result), "xray_xrb is not JIT-compatible"


class TestXRayAGN:
    def test_agn_corona_finite(self):
        """xray_agn_corona returns finite values for typical AGN luminosity."""
        L_nu = xray_agn_corona(_WAVE_GRID, L_agn_bol=1e44)
        chex.assert_tree_all_finite(L_nu), "xray_agn_corona: non-finite output"

    def test_total_additive(self):
        """xray_total = xray_xrb + xray_agn_corona (additive)."""
        sfr = 1.0
        m = 1e10
        L_agn = 1e44
        L_xrb = xray_xrb(_WAVE_GRID, sfr=sfr, stellar_mass=m)
        L_agn_x = xray_agn_corona(_WAVE_GRID, L_agn_bol=L_agn)
        L_tot = xray_total(_WAVE_GRID, sfr=sfr, stellar_mass=m, L_agn_bol=L_agn)
        np.testing.assert_allclose(
            float(jnp.sum(L_tot)),
            float(jnp.sum(L_xrb + L_agn_x)),
            rtol=1e-4,
            err_msg="xray_total should equal xray_xrb + xray_agn_corona",
        )
