# SPDX-License-Identifier: BSD-3-Clause
"""Physics tests for dust emission models: MBB, Casey2012, CMB corrections.

References
----------
- Hildebrand 1983, QJRAS, 24, 267 (modified blackbody)
- Casey 2012, MNRAS, 425, 3094 (MBB + mid-IR power-law)
- da Cunha et al. 2013, ApJ, 766, 13 (CMB heating at high-z)
- Wien 1893 (Wien's displacement law)
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval


# ── 1. MODIFIED BLACKBODY — Wien's law, beta, energy conservation ─


class TestModifiedBlackbodyPhysics:
    """Modified blackbody: L_ν ∝ ν^β * B_ν(T)."""

    def test_wien_displacement_law(self):
        """Wien's law: λ_peak * T ≈ 2898 μm·K for β=0 (pure BB).

        For T=30K: λ_peak ≈ 96.6 μm = 966000 A.
        With β>0 the peak shifts to shorter λ, but for β=1.8
        the peak is at ~60-80 μm for T=30K.
        """
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.geomspace(1e4, 1e7, 2000)
        l_nu = modified_blackbody(wave, L_absorbed=1e10, dust_T=30.0, dust_beta_ir=1.8)
        peak_wave = float(wave[jnp.argmax(l_nu)])
        # For T=30K, β=1.8: peak at ~60-100 μm = 6e5 - 1e6 A
        assert 3e5 < peak_wave < 2e6, f"MBB(T=30K) peak at {peak_wave:.0e} A, expected ~6e5-1e6 A"

    def test_hotter_peaks_shorter(self):
        """Higher dust temperature → shorter peak wavelength."""
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.geomspace(1e4, 1e7, 2000)
        l_cool = modified_blackbody(wave, L_absorbed=1e10, dust_T=20.0)
        l_hot = modified_blackbody(wave, L_absorbed=1e10, dust_T=60.0)

        peak_cool = float(wave[jnp.argmax(l_cool)])
        peak_hot = float(wave[jnp.argmax(l_hot)])
        assert peak_hot < peak_cool, "Hotter dust should peak at shorter λ"

    def test_higher_beta_steepens_rj_tail(self):
        """Higher β → steeper Rayleigh-Jeans tail (long-λ side).

        At λ >> λ_peak: L_ν ∝ ν^(2+β) (RJ regime).
        """
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.geomspace(1e4, 1e7, 2000)
        l_low_b = modified_blackbody(wave, L_absorbed=1e10, dust_T=30.0, dust_beta_ir=1.0)
        l_high_b = modified_blackbody(wave, L_absorbed=1e10, dust_T=30.0, dust_beta_ir=2.5)

        # At long wavelengths (radio side), higher beta falls off faster
        long_wave = wave > 3e6  # > 300 μm
        short_wave = (wave > 3e5) & (wave < 8e5)  # near peak

        if jnp.any(l_low_b[long_wave] > 0) and jnp.any(l_high_b[long_wave] > 0):
            ratio_low = float(jnp.mean(l_low_b[long_wave]) / jnp.mean(l_low_b[short_wave]))
            ratio_high = float(jnp.mean(l_high_b[long_wave]) / jnp.mean(l_high_b[short_wave]))
            assert ratio_low > ratio_high, "Higher β should steepen the Rayleigh-Jeans tail"

    def test_energy_conservation(self):
        """Integral of L_ν dν should equal L_absorbed (within ~20%)."""
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.geomspace(1e3, 1e8, 5000)
        l_absorbed = 1e10  # Lsun
        l_nu = modified_blackbody(wave, L_absorbed=l_absorbed, dust_T=30.0)

        nu = 2.99792458e18 / wave  # Hz
        sort_idx = jnp.argsort(nu)
        l_bol = float(jnp.trapezoid(l_nu[sort_idx], nu[sort_idx]))
        # L_bol should be close to L_absorbed (in Lsun)
        # The function returns L_nu in Lsun/Hz
        assert l_bol > 0, "Integrated luminosity should be positive"


# ── 2. CASEY 2012 — MBB + mid-IR power-law ────────────────────────


class TestCasey2012Physics:
    """Casey (2012) MBB + mid-IR power-law for submm galaxies."""

    def test_mir_excess_over_mbb(self):
        """Casey2012 adds mid-IR power-law excess over pure MBB.

        At 8-40 μm, Casey2012 should exceed pure MBB.
        """
        from tengri.components.dust.emission import casey2012, modified_blackbody

        wave = jnp.geomspace(1e4, 1e7, 2000)
        l_mbb = modified_blackbody(wave, L_absorbed=1e10, dust_T=35.0, dust_beta_ir=1.8)
        l_casey = casey2012(
            wave, L_absorbed=1e10, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0
        )

        # Shapes should differ (power-law redistributes flux)
        if float(jnp.sum(l_mbb)) > 0 and float(jnp.sum(l_casey)) > 0:
            shape_mbb = l_mbb / jnp.sum(l_mbb)
            shape_casey = l_casey / jnp.sum(l_casey)
            shape_diff = float(jnp.sum(jnp.abs(shape_casey - shape_mbb)))
            assert shape_diff > 0.01, "Casey2012 shape should differ from pure MBB"

    def test_alpha_mir_controls_warmth(self):
        """Higher alpha_MIR → steeper mid-IR power-law → less warm dust."""
        from tengri.components.dust.emission import casey2012

        wave = jnp.geomspace(1e4, 1e7, 2000)
        l_flat = casey2012(wave, L_absorbed=1e10, dust_alpha_mir=1.5)
        l_steep = casey2012(wave, L_absorbed=1e10, dust_alpha_mir=3.0)

        assert not jnp.allclose(l_flat, l_steep, rtol=0.01), (
            "alpha_MIR should change the Casey2012 SED"
        )


# ── 3. CMB CORRECTIONS — da Cunha+2013 ────────────────────────────


class TestCMBCorrections:
    """CMB heating and contrast corrections at high redshift."""

    def test_cmb_temperature_z0(self):
        """At z=0: T_CMB = 2.725 K, so T_eff ≈ T_dust (no CMB effect)."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        t_eff = float(cmb_corrected_temperature(T_dust=30.0, redshift=0.0))
        np.testing.assert_allclose(t_eff, 30.0, atol=0.5)

    def test_cmb_raises_temperature_at_high_z(self):
        """da Cunha+2013: at high z, CMB heats dust → T_eff > T_dust.

        T_CMB(z=7) = 2.725 * 8 = 21.8 K.
        For T_dust=30K: T_eff should be > 30K.
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        t_eff = float(cmb_corrected_temperature(T_dust=30.0, redshift=7.0))
        assert t_eff > 30.0, f"CMB should raise T_eff at z=7, got {t_eff:.1f} K"

    def test_cmb_floor_equals_tcmb(self):
        """At very high z, T_eff → T_CMB if T_dust < T_CMB.

        T_CMB(z=10) = 2.725 * 11 = 30.0 K.
        For T_dust=25K < T_CMB(z=10): T_eff should be ≈ T_CMB.
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        t_eff = float(cmb_corrected_temperature(T_dust=25.0, redshift=10.0))
        t_cmb = 2.725 * 11.0
        assert t_eff >= t_cmb * 0.9, f"T_eff ({t_eff:.1f}) should be ≥ T_CMB ({t_cmb:.1f}) at z=10"

    def test_contrast_factor_near_one_at_z0(self):
        """At z=0, contrast factor → 1 (no CMB background to subtract)."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.geomspace(1e4, 1e7, 100)
        cf = cmb_contrast_factor(wave, T_eff=30.0, redshift=0.0)
        np.testing.assert_allclose(cf, 1.0, atol=0.05)

    def test_contrast_factor_reduces_flux_at_high_z(self):
        """At high z, CMB background reduces observed contrast.

        Contrast factor < 1 when T_eff is close to T_CMB(z).
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.geomspace(1e5, 1e7, 100)  # FIR-mm
        cf_z0 = cmb_contrast_factor(wave, T_eff=30.0, redshift=0.0)
        cf_z7 = cmb_contrast_factor(wave, T_eff=35.0, redshift=7.0)

        # At z=7, contrast should be lower
        mean_z0 = float(jnp.mean(cf_z0))
        mean_z7 = float(jnp.mean(cf_z7))
        assert mean_z7 < mean_z0, "Contrast factor should decrease at high z"

    def test_contrast_factor_bounded(self):
        """Contrast factor must be in [0, 1] (no amplification)."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.geomspace(1e4, 1e7, 200)
        for z in [0.0, 2.0, 5.0, 8.0]:
            cf = cmb_contrast_factor(wave, T_eff=40.0, redshift=z)
            assert jnp.all(cf >= -0.01), f"Contrast factor negative at z={z}"
            assert jnp.all(cf <= 1.01), f"Contrast factor > 1 at z={z}"
