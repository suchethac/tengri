# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests: Duras+2020 k_X below its calibrated range, pexrav gradient in float32.

Bug 1 (Duras): ``xray_bolometric_correction_duras`` evaluated ``(log L / b) ** c`` with
``log L`` taken in erg/s. Below 1 erg/s the base is negative and a non-integer power
(c = 17.79) returns NaN, in float64 as well. The paper (Duras et al. 2020, A&A 636, A73,
Eq. 2 and Table 1, "General" row, arXiv:2001.09984) writes the fit in
``log10(L_bol / L_sun)``, and the fit is flat for 7.5 < log10(L_bol/L_sun) < 11, so the
argument is clamped at the lower edge 7.5.

Bug 2 (pexrav): the branching ratio ``sigma_T / (sigma_T + sigma_phabs)`` was formed in
cm^2, where ``sigma_T + sigma_phabs`` is about 1e-22 cm^2. The reverse-mode backward pass
squares that sum, ~1e-44 is below the float32 range, and the wavelength gradient is NaN.
The fix works in units of 1e-22 cm^2, so the squared quantity is O(1e-5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray import xray as X

pytestmark = pytest.mark.regression_bug

#: Duras et al. (2020) Table 1, General row, K_X(L_bol): a, b, c.
DURAS_A, DURAS_B, DURAS_C = 10.96, 11.93, 17.79
#: Lower edge of the calibrated range, log10(L_bol / L_sun) [dex].
DURAS_LOG_LSUN_MIN = 7.5
#: Solar luminosity used by tengri (IAU 2015 nominal) [erg/s].
L_SUN = 3.828e33


def _duras_reference_float64(l_bol_erg: float) -> float:
    """Eq. 2 of Duras+2020 with the calibrated-range clamp, evaluated in float64 NumPy."""
    log_lsun = max(np.log10(l_bol_erg / L_SUN), DURAS_LOG_LSUN_MIN)
    return DURAS_A * (1.0 + (log_lsun / DURAS_B) ** DURAS_C)


def _old_pexrav_reflection(
    wave_angstrom: np.ndarray, l_primary: np.ndarray, R: float, cos_inc: float
) -> np.ndarray:
    """Pre-fix pexrav output, float64: branching in cm^2 from -ln T(N_H = 1e22) / 1e22.

    Reproduces the expression as it stood before the units change, term for term:
    ``sigma_phabs = -ln T / 1e22 [cm^2]``, ``g = sigma_T / (sigma_T + sigma_phabs)``,
    Klein-Nishina rolloff, angle factor, and the 124 Angstrom X-ray band mask.
    """
    nu = X._C_AA / jnp.asarray(wave_angstrom)
    e_kev = X._H_PLANCK * nu / X._KEV_TO_ERG
    sigma_phabs = -jnp.log(jnp.maximum(X.wabs_transmission(e_kev, 22.0), 1e-300)) / 10.0**22.0
    g_branch = X._SIGMA_THOMSON_CM2 / (X._SIGMA_THOMSON_CM2 + sigma_phabs)
    x = e_kev / 511.0
    g_kn = jnp.where(x > 0.5, 0.0, jnp.maximum(1.0 - 2.0 * x + (5.0 / 3.0) * x**2, 0.0))
    mu_factor = (2.0 * cos_inc + 1.0) / 3.0
    in_band = jnp.asarray(wave_angstrom) < 124.0
    return np.asarray(
        jnp.where(in_band, R * mu_factor * g_branch * g_kn * jnp.asarray(l_primary), 0.0)
    )


class TestDuras:
    """Bug 1: K_X is finite below the calibrated range and equals the clamp value there."""

    @pytest.mark.parametrize("l_bol", [1e-3, 1e-1, 1.0])
    def test_finite_below_one_erg_per_second(self, l_bol):
        """Below 1 erg/s the old power of a negative base returned NaN.

        Physical limit: the fit is flat below its range edge, so K_X is finite and
        equals its clamp value. Bounds: K_X >= a for any L, since the bracket is >= 1.
        """
        k = float(X.xray_bolometric_correction_duras(l_bol))
        assert np.isfinite(k)
        assert k >= DURAS_A

    @pytest.mark.parametrize("l_bol", [1e-3, 1e-1, 1.0])
    def test_equals_value_at_clamp(self, l_bol):
        """K_X below the range equals K_X at the lower edge, 10**7.5 L_sun."""
        l_clamp = 10.0**DURAS_LOG_LSUN_MIN * L_SUN
        k = float(X.xray_bolometric_correction_duras(l_bol))
        k_clamp = float(X.xray_bolometric_correction_duras(l_clamp))
        np.testing.assert_allclose(k, k_clamp, rtol=1e-12)
        np.testing.assert_allclose(k, _duras_reference_float64(l_bol), rtol=1e-12)

    @pytest.mark.parametrize("l_bol", [1e41, 1e43, 1e45])
    def test_inside_calibrated_range_matches_eq2(self, l_bol):
        """Inside the calibrated range the float64 value is Eq. 2 with Table 1 values."""
        k = float(X.xray_bolometric_correction_duras(l_bol))
        np.testing.assert_allclose(k, _duras_reference_float64(l_bol), rtol=1e-12)

    def test_float32_finite_below_range(self):
        """The clamp keeps the float32 path finite too."""
        with jax.enable_x64(False):
            k = X.xray_bolometric_correction_duras(jnp.float32(1e-3))
            assert np.isfinite(float(k))
            np.testing.assert_allclose(float(k), _duras_reference_float64(1e-3), rtol=1e-5)


class TestPexravFloat32:
    """Bug 2: the pexrav wavelength gradient is finite and non-zero in float32."""

    WAVE = np.geomspace(0.1, 120.0, 200)

    def test_float64_forward_unchanged_by_units(self):
        """The rescaled pexrav equals the pre-fix cm^2 expression to round-off (rtol 1e-12).

        The units change must be a pure rescaling, so the float64 forward value does not
        move; the closed form is the one documented in ``pexrav_reflection``.
        """
        wave = self.WAVE
        l_prim = np.linspace(1.0, 3.0, wave.size) * 1e30
        new = np.asarray(
            X.pexrav_reflection(jnp.asarray(wave), jnp.asarray(l_prim), R=0.5, cos_inc=0.5)
        )
        old = _old_pexrav_reflection(wave, l_prim, R=0.5, cos_inc=0.5)
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=0.0)

    def test_forward_float32_matches_float64(self):
        """The float32 forward value agrees with float64 to float32 precision."""
        wave = self.WAVE
        l_prim = np.linspace(1.0, 3.0, wave.size) * 1e30
        out64 = np.asarray(
            X.pexrav_reflection(jnp.asarray(wave), jnp.asarray(l_prim), R=0.5, cos_inc=0.5)
        )
        with jax.enable_x64(False):
            out32 = np.asarray(
                X.pexrav_reflection(
                    jnp.asarray(wave, jnp.float32),
                    jnp.asarray(l_prim, jnp.float32),
                    R=jnp.float32(0.5),
                    cos_inc=jnp.float32(0.5),
                )
            )
        scale = np.max(np.abs(out64))
        np.testing.assert_allclose(out32, out64, rtol=1e-5, atol=1e-6 * scale)

    def test_wavelength_gradient_finite_and_nonzero_float32(self):
        """Reverse-mode d(sum L_refl)/d(wave) in float32 is finite and not identically zero."""
        with jax.enable_x64(False):
            wave = jnp.asarray(self.WAVE, jnp.float32)
            l_prim = jnp.ones_like(wave)

            def total(w):
                return jnp.sum(X.pexrav_reflection(w, l_prim, R=0.5, cos_inc=0.5))

            grad = np.asarray(jax.grad(total)(wave))
        assert np.all(np.isfinite(grad))
        assert float(np.max(np.abs(grad))) > 0.0

    def test_reflection_gradient_float32_finite_and_nonzero(self):
        """d(sum L_refl)/dR and d/dcos_inc in float32 are finite and non-zero."""
        with jax.enable_x64(False):
            wave = jnp.asarray(self.WAVE, jnp.float32)
            l_prim = jnp.ones_like(wave)

            def total(r, c):
                return jnp.sum(X.pexrav_reflection(wave, l_prim, R=r, cos_inc=c))

            g_r, g_c = jax.grad(total, argnums=(0, 1))(jnp.float32(0.5), jnp.float32(0.5))
        for g in (g_r, g_c):
            assert np.isfinite(float(g))
            assert abs(float(g)) > 0.0
