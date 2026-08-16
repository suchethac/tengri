# SPDX-License-Identifier: BSD-3-Clause
"""Tests for X-ray emission module (models/xray.py).

Physics references:
- Lehmer+2016, ApJ 825, 7 (HMXB quartic in Z, LMXB quartic in log τ).
- Yang+2020, MNRAS 491, 740 / X-CIGALE (canonical alpha_ox(L_2500) path).
- Just+2007, ApJ 665, 1004 (alpha_ox-L_2500 calibration).

Notes
-----
Post-#329 ``xray_agn_corona`` and ``xray_total`` take ``l_2500_30deg_erg_hz``
(not ``L_agn_bol``); we convert via Hopkins+2007 BC_2500 ≈ 5.15.
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.xray import xray_agn_corona, xray_total, xray_xrb
from tengri.utils.physics_constants import C_AA as _C_AA, KEV_TO_HZ as _KEV_TO_HZ


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# 2-10 keV wavelength/frequency grid for band integration
_E_GRID = jnp.linspace(2.0, 10.0, 500)  # keV
_NU_GRID = _E_GRID * _KEV_TO_HZ  # Hz
_WAVE_GRID = _C_AA / _NU_GRID  # Angstrom (x-ray)

# Hopkins+2007 BC_2500 ≈ 5.15 lets us back out L_2500 from L_agn_bol for the
# new xray_agn_corona signature.
_NU_2500 = 1.199e15  # Hz
_BC_2500 = 5.15

# Gilfanov 2004 / Yang+22-Lehmer+14 assume an old (10 Gyr) stellar population
# for LMXB normalization. Set explicitly so tests don't drift if the default
# ever changes.
_LMXB_AGE_GYR = 10.0


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
        """HMXB 2-10 keV luminosity at SFR=1 Msun/yr and the default Z.

        Yang+22 / Lehmer+19 quartic in Z:
            log L_HMXB(2-10 keV) [W] = 33.28 - 62.12 Z + 569.44 Z² - 1833.8 Z³
                                      + 1968.33 Z⁴

        The default is ``Z_SUN`` = 0.0142 (Asplund 2009), giving 3.22e39 erg/s.

        This asserted 1.78e39 while its own message called that value "Z_sun",
        which is the confusion #1755 was about: 1.78e39 is the quartic at
        **Z = 0.02**, a solar convention the rest of the codebase does not use.
        Evaluate the polynomial at the constant rather than restating a
        precomputed magnitude, so the two cannot drift apart again.
        """
        from tengri.utils.physics_constants import Z_SUN

        expected = 10.0 ** (
            40.28 - 62.12 * Z_SUN + 569.44 * Z_SUN**2 - 1833.80 * Z_SUN**3 + 1968.33 * Z_SUN**4
        )
        L_band = float(jnp.trapezoid(xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=0.0), _NU_GRID))
        np.testing.assert_allclose(
            L_band,
            expected,  # ~3.22e39 at Z_SUN = 0.0142
            rtol=0.05,
            err_msg=(
                f"Lehmer+19 (yang20.py:207-214): HMXB L_2-10keV should be "
                f"{expected:.3e} erg/s at the default Z={Z_SUN}"
            ),
        )

    def test_lmxb_band_luminosity(self):
        """LMXB 2-10 keV luminosity at M*=1e10 Msun, age=10 Gyr matches Lehmer+14.

        Yang+22 / Lehmer+14 quartic in log τ:
            log L_LMXB(2-10) [W / (M*/1e10 Msun)] = 33.276 - 1.503 logT - 0.423 logT²
                                                   + 0.425 logT³ + 0.136 logT⁴
        At logT = 1 (10 Gyr): log L = 31.911 W / 1e10 Msun → ~ 8.15e38 erg/s.
        Compatible with the Gilfanov+04 calibration (8.3e28 erg/s/Msun for old
        stellar populations).
        """
        L_band = float(
            jnp.trapezoid(
                xray_xrb(_WAVE_GRID, sfr=0.0, stellar_mass=1e10, stellar_age_gyr=_LMXB_AGE_GYR),
                _NU_GRID,
            )
        )
        np.testing.assert_allclose(
            L_band,
            10**31.911 * 1e7,  # W -> erg/s, ~ 8.15e38
            rtol=0.05,
            err_msg="Lehmer+14: LMXB L_2-10keV ≈ 8.15e38 erg/s at M*=1e10, age=10 Gyr",
        )

    def test_combined_ranalli2003(self):
        """Combined XRB at SFR=1, M*=1e10, age=10 Gyr sits inside Ranalli+2003.

        HMXB (Lehmer+19) + LMXB (Lehmer+14, 10 Gyr) ~ 8.15e38, against
        Ranalli+2003's 3.7e39.

        One assertion was doing two incompatible jobs — a literature
        consistency bound and a drift detector — so they are separate. That
        split earned itself immediately: the pin moved once and the bound did
        not, which is only legible because the two are apart.

        **The pin moved 2.5972e39 -> 4.032409e39 (x1.5526) in #1845**, which
        made the galaxy's metallicity reach the HMXB term (#1755). Before that
        the term ran at a hardcoded Z=0.02; it now uses the caller's value,
        defaulting to ``metallicity_z=Z_SUN=0.0142`` (Asplund 2009). Lehmer+2019
        anticorrelates HMXB emissivity with metallicity, so the lower Z raises
        L_HMXB — the direction the physics requires.

        Agreement with the observed relation improved threefold rather than
        degrading:

            old  2.5972e39   29.81% from 3.7e39   99.4% of the 30% band
            new  4.0324e39    8.98% from 3.7e39   29.9% of the band

        So the Ranalli bound not only still holds, it holds with far more room:
        the fixture is no longer sitting on its limit, where any 0.7% move the
        wrong way flipped it red and read as a physics regression.
        """
        L_band = float(
            jnp.trapezoid(
                xray_xrb(_WAVE_GRID, sfr=1.0, stellar_mass=1e10, stellar_age_gyr=_LMXB_AGE_GYR),
                _NU_GRID,
            )
        )

        # The science claim: consistent with the observed relation, whose own
        # scatter is ~30%. Kept as a bound, not as the drift signal.
        np.testing.assert_allclose(
            L_band,
            3.7e39,
            rtol=0.30,
            err_msg="Ranalli+2003 A&A 399 Eq. 3: L_2-10keV ~ 3.7e39 erg/s at SFR=1, M*=1e10",
        )

        # The drift detector: what this implementation actually produces today.
        # Tight, so a real change is caught precisely and reported as a change
        # in our number rather than as a brush with the literature band.
        np.testing.assert_allclose(
            L_band,
            4.032409e39,
            rtol=1e-3,
            err_msg=(
                "combined XRB luminosity moved. This is the implementation's own "
                "value, not a literature number — if the change is intended, "
                "update it here and check the Ranalli bound above still holds "
                "(it now has 20 percentage points of margin, up from 0.19)."
            ),
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
        """xray_agn_corona returns finite values for typical AGN luminosity.

        Post-#329 the canonical CIGALE-faithful path takes L_2500 (intrinsic,
        at 30 deg inclination), not L_bol. Convert via Hopkins+2007 BC_2500.
        """
        L_agn_bol = 1e44
        l_2500 = L_agn_bol / (_BC_2500 * _NU_2500)  # erg/s/Hz
        L_nu = xray_agn_corona(_WAVE_GRID, l_2500_30deg_erg_hz=l_2500)
        chex.assert_tree_all_finite(L_nu)

    def test_total_additive(self):
        """xray_total = xray_xrb + xray_agn_corona (additive) under the new L_2500 API."""
        sfr = 1.0
        m = 1e10
        L_agn_bol = 1e44
        l_2500 = L_agn_bol / (_BC_2500 * _NU_2500)
        L_xrb = xray_xrb(_WAVE_GRID, sfr=sfr, stellar_mass=m, stellar_age_gyr=_LMXB_AGE_GYR)
        L_agn_x = xray_agn_corona(_WAVE_GRID, l_2500_30deg_erg_hz=l_2500)
        L_tot = xray_total(
            _WAVE_GRID,
            sfr=sfr,
            stellar_mass=m,
            stellar_age_gyr=_LMXB_AGE_GYR,
            l_2500_30deg=l_2500,
        )
        np.testing.assert_allclose(
            float(jnp.sum(L_tot)),
            float(jnp.sum(L_xrb + L_agn_x)),
            rtol=1e-4,
            err_msg="xray_total should equal xray_xrb + xray_agn_corona",
        )
