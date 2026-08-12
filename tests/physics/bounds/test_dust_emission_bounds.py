# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust emission physical bounds and limits.

Non-negativity, temperature monotonicity, and limiting cases.
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def wavelengths():
    """IR wavelength grid (Angstrom), 1 -- 1000 um."""
    return jnp.linspace(1e4, 1e7, 500)


@pytest.fixture
def L_absorbed():
    """Typical absorbed luminosity in Lsun."""
    return 1e10


# ── f_cold extremes ───────────────────────────────────────────────


class TestFColdExtremes:
    """f_cold=0 gives all warm, f_cold=1 gives all cold.

    Limit cases for the dust mixture parameter.
    """

    def test_f_cold_zero_all_warm(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split, modified_blackbody

        sed = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=0.0,
            dust_T_warm=45.0,
            dust_T_cold=20.0,
        )
        expected = modified_blackbody(
            wavelengths,
            L_absorbed=L_absorbed,
            dust_T=45.0,
            dust_beta_ir=1.5,
        )
        assert jnp.allclose(sed, expected, rtol=1e-10)

    def test_f_cold_one_all_cold(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split, modified_blackbody

        sed = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=1.0,
            dust_T_warm=45.0,
            dust_T_cold=20.0,
        )
        expected = modified_blackbody(
            wavelengths,
            L_absorbed=L_absorbed,
            dust_T=20.0,
            dust_beta_ir=2.0,
        )
        assert jnp.allclose(sed, expected, rtol=1e-10)


# ── Non-negativity and physical sanity ────────────────────────────


class TestEdgeCases:
    """Edge cases and physical sanity checks."""

    def test_zero_luminosity(self, wavelengths):
        from tengri.components.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, 0.0)
        assert jnp.allclose(sed, 0.0)

    def test_output_non_negative(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, L_absorbed)
        assert_non_negative(sed, name="sed")

    def test_warm_peaks_at_shorter_wavelength(self, wavelengths, L_absorbed):
        """Warm component peaks at shorter wavelength than cold (Wien's law).

        hotter dust → peaks at shorter wavelength
        """
        from tengri.components.dust.emission import energy_balance_split

        sed_warm_only = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=0.0,
            dust_T_warm=50.0,
            dust_T_cold=15.0,
        )
        sed_cold_only = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=1.0,
            dust_T_warm=50.0,
            dust_T_cold=15.0,
        )
        peak_warm = wavelengths[jnp.argmax(sed_warm_only)]
        peak_cold = wavelengths[jnp.argmax(sed_cold_only)]
        # Wien's law: hotter dust peaks at shorter wavelength
        assert peak_warm < peak_cold


class TestPlanckBnuBounds:
    """Direct tests for the planck_bnu Planck function bounds."""

    @pytest.fixture
    def wave_ir(self):
        """IR wavelength grid (Angstrom), ~1 – 1000 μm."""
        return jnp.logspace(4, 8, 300)

    def test_finite_positive(self, wave_ir):
        from tengri.components.dust.emission import planck_bnu

        bnu = planck_bnu(wave_ir, temperature=30.0)
        chex.assert_tree_all_finite(bnu)
        assert jnp.all(bnu > 0.0)

    def test_output_shape(self, wave_ir):
        from tengri.components.dust.emission import planck_bnu

        bnu = planck_bnu(wave_ir, temperature=30.0)
        chex.assert_equal_shape([bnu, wave_ir])

    def test_hotter_peaks_at_shorter_wavelength(self, wave_ir):
        """Wien's displacement law: hotter BB peaks at shorter λ."""
        from tengri.components.dust.emission import planck_bnu

        bnu_cold = planck_bnu(wave_ir, temperature=20.0)
        bnu_warm = planck_bnu(wave_ir, temperature=60.0)
        peak_cold = float(wave_ir[jnp.argmax(bnu_cold)])
        peak_warm = float(wave_ir[jnp.argmax(bnu_warm)])
        assert peak_warm < peak_cold

    def test_hotter_brighter(self, wave_ir):
        """At fixed wavelength, higher T → higher B_nu (Stefan-Boltzmann).

        More hot photons → higher total emission
        """
        from tengri.components.dust.emission import planck_bnu

        bnu_low = planck_bnu(wave_ir, temperature=20.0)
        bnu_high = planck_bnu(wave_ir, temperature=50.0)
        assert jnp.sum(bnu_high) > jnp.sum(bnu_low)

    def test_short_wavelengths_finite(self):
        """UV/EUV wavelengths (clipped x) should not overflow."""
        from tengri.components.dust.emission import planck_bnu

        wave_uv = jnp.array([10.0, 100.0, 1000.0])  # Angstrom
        bnu = planck_bnu(wave_uv, temperature=1e4)
        chex.assert_tree_all_finite(bnu)
        assert_non_negative(bnu, name="bnu")


class TestModifiedBlackbodyBounds:
    """Standalone tests for modified_blackbody bounds."""

    @pytest.fixture
    def wave_fir(self):
        """Far-IR wavelength grid (Angstrom), 10 μm – 10 mm."""
        return jnp.logspace(5, 9, 400)

    def test_finite_non_negative(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed = modified_blackbody(wave_fir, L_absorbed=1e10)
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

    def test_output_shape(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed = modified_blackbody(wave_fir, L_absorbed=1e10)
        chex.assert_equal_shape([sed, wave_fir])

    def test_hotter_peaks_shorter_wavelength(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed_cold = modified_blackbody(wave_fir, L_absorbed=1e10, dust_T=20.0)
        sed_warm = modified_blackbody(wave_fir, L_absorbed=1e10, dust_T=50.0)
        peak_cold = float(wave_fir[jnp.argmax(sed_cold)])
        peak_warm = float(wave_fir[jnp.argmax(sed_warm)])
        assert peak_warm < peak_cold

    def test_zero_luminosity(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed = modified_blackbody(wave_fir, L_absorbed=0.0)
        assert jnp.allclose(sed, 0.0)

    def test_higher_beta_steeper_rayleigh_jeans(self, wave_fir):
        """Higher dust_beta_ir → steeper slope on Rayleigh-Jeans side.

        τ_λ ∝ λ^(-β): higher β suppresses long-wavelength emission more.
        """
        from tengri.components.dust.emission import modified_blackbody

        # Use radio-wavelength end where RJ slope is dominant
        wave_radio = jnp.logspace(8, 10, 100)  # 1 cm – 1 m
        sed_low_beta = modified_blackbody(wave_radio, L_absorbed=1e10, dust_beta_ir=1.0)
        sed_high_beta = modified_blackbody(wave_radio, L_absorbed=1e10, dust_beta_ir=2.5)

        # High beta suppresses long-wavelength (Rayleigh-Jeans) emission more
        ratio_low = float(sed_low_beta[-1]) / float(sed_low_beta[0])
        ratio_high = float(sed_high_beta[-1]) / float(sed_high_beta[0])
        # High-beta SED has steeper drop → smaller ratio at longest wavelength
        assert ratio_high < ratio_low


class TestCasey2012Bounds:
    """Standalone tests for casey2012 (MBB + mid-IR power law) bounds."""

    @pytest.fixture
    def wave_ir(self):
        """Broad IR wavelength grid (Angstrom), 1 μm – 10 mm."""
        return jnp.logspace(4, 9, 500)

    def test_finite_non_negative(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        sed = casey2012(wave_ir, L_absorbed=1e10)
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

    def test_output_shape(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        sed = casey2012(wave_ir, L_absorbed=1e10)
        chex.assert_equal_shape([sed, wave_ir])

    def test_zero_luminosity(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        sed = casey2012(wave_ir, L_absorbed=0.0)
        assert jnp.allclose(sed, 0.0)

    def test_hotter_peaks_shorter_wavelength(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        # Use only FIR range where MBB dominates
        wave_fir = jnp.logspace(5.5, 8, 300)
        sed_cold = casey2012(wave_fir, L_absorbed=1e10, dust_T=20.0)
        sed_warm = casey2012(wave_fir, L_absorbed=1e10, dust_T=50.0)
        peak_cold = float(wave_fir[jnp.argmax(sed_cold)])
        peak_warm = float(wave_fir[jnp.argmax(sed_warm)])
        assert peak_warm < peak_cold


class TestCmbCorrectedTemperatureBounds:
    """Tests for cmb_corrected_temperature bounds and limits."""

    def test_z0_no_change(self):
        """At z=0 with z=0 CMB, T_eff should equal T_dust.

        T_eff(z=0) ≈ T_dust: CMB correction negligible at z=0.
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 35.0
        # At z=0 the CMB terms cancel: T_cmb_z == T_CMB_0, so inner = T_dust^exponent
        T_eff = float(cmb_corrected_temperature(T_dust, redshift=0.0, beta_ir=1.6))
        assert abs(T_eff - T_dust) < 0.1

    def test_high_z_raises_temperature(self):
        """At high redshift the CMB floor raises the effective temperature.

        T_eff increases with z: CMB heating dominates at high z
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 20.0
        T_eff_z0 = float(cmb_corrected_temperature(T_dust, redshift=0.0))
        T_eff_z5 = float(cmb_corrected_temperature(T_dust, redshift=5.0))
        assert T_eff_z5 > T_eff_z0

    def test_always_finite(self):
        """Finite output even for very cold or hot dust.

        T_eff must be bounded and finite for all physical dust temperatures.
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        for T_dust in (0.01, 1.0, 50.0, 200.0):
            T_eff = float(cmb_corrected_temperature(T_dust, redshift=2.0))
            assert np.isfinite(T_eff), f"T_eff not finite for T_dust={T_dust}"
            assert T_eff > 0.0

    def test_negative_T_dust_clamped(self):
        """Negative T_dust values are clamped to 1 K — no NaN.

        Robustness: handle unphysical inputs gracefully.
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_eff = float(cmb_corrected_temperature(-10.0, redshift=0.5))
        assert np.isfinite(T_eff)
        assert T_eff > 0.0


class TestCmbContrastFactorBounds:
    """Tests for cmb_contrast_factor bounds.

    Contrast must be in [0, 1] by construction: 1 - B_cmb/B_eff.
    """

    def test_z0_contrast_near_one_at_fir_peak(self):
        """At z=0 near the FIR peak (~50-500 μm), CMB is negligible vs T=40 K dust.

        0 ≤ contrast ≤ 1: CMB < dust radiation at typical dust frequencies.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        # 50–500 μm = 5e5–5e6 Å: FIR peak of 40 K dust, Wien side of 2.7 K CMB
        wave = jnp.logspace(5.7, 6.7, 100)
        factor = cmb_contrast_factor(wave, T_eff=40.0, redshift=0.0)
        assert jnp.all(factor > 0.99)

    def test_high_z_reduces_contrast(self):
        """At high z, contrast factor is appreciably below 1 for cold dust.

        At z=5: T_CMB ≈ 16 K, approaching T_dust=20 K, so CMB contribution > 1%.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.logspace(5, 8, 100)
        factor_z5 = cmb_contrast_factor(wave, T_eff=20.0, redshift=5.0)
        # At z=5, T_CMB = 2.725*6 ≈ 16 K; contrast against T_eff=20 K is suppressed
        assert jnp.any(factor_z5 < 0.9)

    def test_output_in_unit_interval(self):
        """Contrast factor must be in [0, 1] by construction.

        0 ≤ 1 - B_cmb/B_eff ≤ 1 for all redshifts.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        for z in (0.0, 2.0, 5.0, 10.0):
            wave = jnp.logspace(4, 8, 200)
            factor = cmb_contrast_factor(wave, T_eff=30.0, redshift=z)
            assert_non_negative(factor, name="factor", msg=f"Negative contrast at z={z}")
            assert jnp.all(factor <= 1.0), f"Contrast > 1 at z={z}"

    def test_all_finite(self):
        """No NaN/Inf values on the output grid."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.logspace(3, 9, 300)
        factor = cmb_contrast_factor(wave, T_eff=35.0, redshift=3.0)
        chex.assert_tree_all_finite(factor)


# ── Additional TestCasey2012 bounds tests ─────────────────────────


class TestCasey2012DetailedBounds:
    """Additional detailed bounds tests for casey2012."""

    @pytest.fixture
    def wave_ir(self):
        """Broad IR wavelength grid (Angstrom), 1 μm – 10 mm."""
        return jnp.logspace(4, 9, 500)

    def test_alpha_affects_mid_ir(self, wave_ir):
        """Larger dust_alpha_mir increases mid-IR power-law contribution."""
        from tengri.components.dust.emission import casey2012

        mir_mask = (wave_ir > 8e4) & (wave_ir < 4e5)
        sed_low = casey2012(wave_ir, L_absorbed=1e10, dust_alpha_mir=1.5)
        sed_high = casey2012(wave_ir, L_absorbed=1e10, dust_alpha_mir=3.0)
        # Different alpha → different MIR shapes
        assert not jnp.allclose(sed_low[mir_mask], sed_high[mir_mask], rtol=0.01)

    def test_mid_ir_excess_vs_pure_mbb(self, wave_ir):
        """casey2012 has MORE 8–40 μm flux than a pure MBB — the mid-IR excess.

        The mid-IR power law is the point of Casey (2012) Eq. 1: it fills the
        8–40 μm side that a single-temperature MBB underpredicts, with its
        amplitude tied to the graybody at the turnover λ_c (Eq. 2). Before
        #1004 this test asserted the opposite ordering, because the closure
        carried a spurious Wien exp(-hν/kT) factor that annihilated the power
        law (e⁻⁴¹ at 10 μm for T = 35 K) — the docstring rationalized the bug.
        """
        from tengri.components.dust.emission import casey2012, modified_blackbody

        L_abs = 1e10
        sed_casey = casey2012(wave_ir, L_absorbed=L_abs, dust_T=35.0)
        sed_mbb = modified_blackbody(wave_ir, L_absorbed=L_abs, dust_T=35.0)

        # 8–40 μm in Angstrom — the Casey power law adds mid-IR flux
        mir_mask = (wave_ir > 8e4) & (wave_ir < 4e5)
        assert jnp.sum(sed_casey[mir_mask]) > jnp.sum(sed_mbb[mir_mask])
