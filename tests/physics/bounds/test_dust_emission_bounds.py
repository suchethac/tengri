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

    def test_contrast_lower_bound_short_wavelengths(self):
        """Contrast >= 1 - 1e-12 for all wavelengths.

        At short wavelengths (UV), the contrast should approach 1 from below
        as the CMB Planck function vanishes much faster than the dust one.
        Note: T_eff must be > T_cmb(z) for physical validity.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        # Test at multiple temperature and redshift points
        # Note: T_cmb(z) = 2.725 K * (1 + z), so T_eff must exceed this
        test_cases = [
            (25.0, 0.0),  # T_eff=25 K >> T_cmb(0)=2.7 K
            (50.0, 10.0),  # T_eff=50 K >> T_cmb(10)=30 K
            (100.0, 0.0),  # T_eff=100 K >> T_cmb(0)=2.7 K
            (100.0, 10.0),  # T_eff=100 K >> T_cmb(10)=30 K
        ]

        for T_eff, redshift in test_cases:
            wave = jnp.logspace(2, 4, 200)  # 100 Å to 10,000 Å
            factor = cmb_contrast_factor(wave, T_eff=T_eff, redshift=redshift)
            assert jnp.all(factor >= 1.0 - 1e-12), (
                f"Contrast < 1 - 1e-12 at T_eff={T_eff}, z={redshift}: min={jnp.min(factor)}"
            )

    def test_contrast_monotonic_decreasing_with_wavelength(self):
        """Contrast is monotonically non-increasing with wavelength.

        At short wavelengths (UV), contrast ~ 1. As wavelength increases
        (moving to longer wavelengths where CMB is more significant),
        the contrast can only decrease or stay the same.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        T_eff = 25.0
        redshift = 0.0
        wave = jnp.logspace(2, 8, 500)  # 100 Å to 100 mm
        factor = cmb_contrast_factor(wave, T_eff=T_eff, redshift=redshift)

        # Check monotonicity: each value <= previous value
        diffs = jnp.diff(factor)
        # Allow small numerical noise (1e-14)
        assert jnp.all(diffs <= 1e-14), f"Contrast not monotonic; max increase: {jnp.max(diffs)}"

    def test_energy_balance_casey2012_with_contrast(self):
        """Energy balance: casey2012 integrates to absorbed luminosity * contrast.

        On a UV-to-radio grid, the integral of casey2012 SED over frequency
        should equal L_absorbed (or very close with small allowed loss).
        Uses the convention: integral over frequency = integral of nu*L_nu d(ln(wave)).
        """
        from tengri.components.dust.emission import casey2012
        from tengri.components.dust.emission._physics import integrate_lnu_over_nu

        # Test parameters from specification
        test_cases = [
            {"T": 25.0, "alpha": 2.0},
            {"T": 25.0, "alpha": 1.4},
            {"T": 100.0, "alpha": 2.0},
        ]

        for case in test_cases:
            # UV-to-radio grid: 0.01 dex spacing from 1e2 to 1e8 Angstrom
            wave = jnp.array(10.0 ** np.arange(2.0, 8.0, 0.01))
            L_abs = 1e10
            sed = casey2012(
                wave,
                L_absorbed=L_abs,
                dust_T=case["T"],
                dust_beta_ir=2.0,
                dust_alpha_mir=case["alpha"],
                redshift=0.0,
            )

            # Use the canonical integration function from _physics
            total_absorbed = integrate_lnu_over_nu(sed, wave)

            # Use pytest.approx with abs=0 and relative tolerance
            np.testing.assert_allclose(
                float(total_absorbed),
                L_abs,
                rtol=1e-4,
                atol=0,
                err_msg=(
                    f"Energy balance failed for T={case['T']}, alpha={case['alpha']}: "
                    f"integrated={float(total_absorbed)}, expected={L_abs}"
                ),
            )

    def test_contrast_long_wavelengths_numerical_values(self):
        """Contrast at long wavelengths matches hand-computed values from expm1.

        For T_eff=50 K, z=10, wavelength in {100 um, 1 mm, 1 cm},
        verify the new function matches 1 - expm1(x_eff)/expm1(x_cmb)
        computed with numpy float64 (no clipping).
        Note: T_eff=50 K >> T_cmb(10)=30 K ensures physical validity.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        T_eff = 50.0
        redshift = 10.0
        T_cmb_z = 2.725 * (1.0 + redshift)

        # Hand-compute expected values using numpy float64
        h_planck = 6.62607015e-27  # erg*s
        c_cgs = 2.99792458e10  # cm/s
        k_boltz = 1.380649e-16  # erg/K

        wavelengths_um = np.array([100.0, 1e3, 1e4])  # 100 um, 1 mm, 1 cm
        wavelengths_aa = wavelengths_um * 1e4
        wavelengths_cm = wavelengths_aa * 1e-8

        expected_contrasts = []
        for wl_cm in wavelengths_cm:
            nu = c_cgs / wl_cm
            x_eff = h_planck * nu / (k_boltz * T_eff)
            x_cmb = h_planck * nu / (k_boltz * T_cmb_z)
            ratio = np.expm1(x_eff) / np.expm1(x_cmb)
            contrast = 1.0 - ratio
            expected_contrasts.append(contrast)

        # Call the jax function
        wave = jnp.array(wavelengths_aa)
        factor = cmb_contrast_factor(wave, T_eff=T_eff, redshift=redshift)

        # Compare with tolerance
        for i, expected in enumerate(expected_contrasts):
            np.testing.assert_allclose(
                float(factor[i]),
                expected,
                rtol=1e-10,
                atol=0,
                err_msg=(
                    f"Contrast mismatch at wavelength {wavelengths_um[i]} um: "
                    f"computed={float(factor[i])}, expected={expected}"
                ),
            )

    def test_gradient_safety_float64(self):
        """Gradient is finite on UV-to-radio grid in float64.

        jax.grad of sum of cmb_contrast_factor over wavelengths must be finite.
        """
        import jax

        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.array(10.0 ** np.arange(2.0, 8.0, 0.01))

        def loss_fn(T):
            return jnp.sum(cmb_contrast_factor(wave, T_eff=T, redshift=10.0))

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(25.0)
        assert jnp.isfinite(grad_val), f"Gradient not finite in float64: {grad_val}"

    def test_gradient_safety_float32(self):
        """Gradient is finite on UV-to-radio grid in float32 (with x64 disabled).

        jax.grad of sum of cmb_contrast_factor over wavelengths must be finite
        even when JAX_ENABLE_X64 is False.
        """
        import jax
        from jax import config

        from tengri.components.dust.emission import cmb_contrast_factor

        # Save current state and temporarily disable x64
        old_state = config.jax_enable_x64
        try:
            config.update("jax_enable_x64", False)
            wave = jnp.array(10.0 ** np.arange(2.0, 8.0, 0.01), dtype=jnp.float32)

            def loss_fn(T):
                return jnp.sum(cmb_contrast_factor(wave, T_eff=jnp.float32(T), redshift=10.0))

            grad_fn = jax.grad(loss_fn)
            grad_val = grad_fn(jnp.float32(50.0))
            assert jnp.isfinite(grad_val), f"Gradient not finite in float32: {grad_val}"
        finally:
            config.update("jax_enable_x64", old_state)


# ── Graybody (general-opacity graybody) tests ─────────────────────


class TestGraybodyBounds:
    """Tests for graybody (general-opacity graybody) physical bounds."""

    @pytest.fixture
    def wave_ir(self):
        """Broad IR wavelength grid (Angstrom), 1 μm – 10 mm."""
        return jnp.logspace(4, 9, 500)

    def test_graybody_finite_non_negative(self, wave_ir):
        from tengri.components.dust.emission import graybody

        sed = graybody(wave_ir, L_absorbed=1e10, dust_lambda_0_um=200.0)
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

    def test_graybody_output_shape(self, wave_ir):
        from tengri.components.dust.emission import graybody

        sed = graybody(wave_ir, L_absorbed=1e10, dust_lambda_0_um=200.0)
        chex.assert_equal_shape([sed, wave_ir])

    def test_graybody_zero_luminosity(self, wave_ir):
        from tengri.components.dust.emission import graybody

        sed = graybody(wave_ir, L_absorbed=0.0, dust_lambda_0_um=200.0)
        assert jnp.allclose(sed, 0.0)

    def test_graybody_hotter_peaks_shorter_wavelength(self, wave_ir):
        """Higher temperature → peak at shorter wavelength (Wien)."""
        from tengri.components.dust.emission import graybody

        sed_cold = graybody(wave_ir, L_absorbed=1e10, dust_T=20.0, dust_lambda_0_um=200.0)
        sed_warm = graybody(wave_ir, L_absorbed=1e10, dust_T=50.0, dust_lambda_0_um=200.0)
        peak_cold = float(wave_ir[jnp.argmax(sed_cold)])
        peak_warm = float(wave_ir[jnp.argmax(sed_warm)])
        assert peak_warm < peak_cold

    def test_graybody_beta_ir_zero_is_finite(self, wave_ir):
        """graybody with beta_ir=0 should be finite and reasonable (pure blackbody).

        At beta_ir=0, the emissivity is 1 everywhere. The spectrum is well-defined
        and should be smooth like a pure blackbody. The peak of L_nu (per Hz) is at
        a longer wavelength than Wien's displacement law peak (which applies to
        L_lambda).
        """
        from tengri.components.dust.emission import graybody

        sed = graybody(
            wave_ir,
            L_absorbed=1e10,
            dust_T=30.0,
            dust_beta_ir=0.0,
            dust_lambda_0_um=200.0,
        )

        # Must be finite, non-negative, and have reasonable shape
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

        # Peak of L_nu for 30 K blackbody is at ~ h*nu/(k*T) ≈ 2.82
        # which corresponds to lambda ~ c / (2.82 * k*T / h) ~ 160-180 um
        peak_idx = jnp.argmax(sed)
        peak_wavelength_um = float(wave_ir[peak_idx]) / 1e4
        # For 30 K blackbody, L_nu peak is ~ 160-180 um
        assert 140 < peak_wavelength_um < 200, (
            f"Peak at {peak_wavelength_um} um is outside expected range [140-200]"
        )

    def test_graybody_large_lambda_0_optically_thin_limit(self, wave_ir):
        """When lambda_0_um >> wavelength, graybody → optically-thin limit."""
        from tengri.components.dust.emission import graybody, modified_blackbody

        # lambda_0 = 1e6 um means tau << 1 everywhere on the IR grid
        sed_graybody = graybody(
            wave_ir,
            L_absorbed=1e10,
            dust_T=30.0,
            dust_beta_ir=1.8,
            dust_lambda_0_um=1e6,
        )
        sed_optically_thin = modified_blackbody(
            wave_ir, L_absorbed=1e10, dust_T=30.0, dust_beta_ir=1.8
        )

        # Shapes should match in the thin limit (contrast factor nearly cancels)
        # Compare on wavelengths where contrast ~ 1 (FIR)
        wave_fir_idx = wave_ir >= 1e5  # 10 um and longer
        if jnp.any(wave_fir_idx):
            max_sb = jnp.max(sed_graybody[wave_fir_idx]) + 1e-30
            max_ot = jnp.max(sed_optically_thin[wave_fir_idx]) + 1e-30
            shape_graybody = sed_graybody[wave_fir_idx] / max_sb
            shape_optically_thin = sed_optically_thin[wave_fir_idx] / max_ot
            np.testing.assert_allclose(shape_graybody, shape_optically_thin, rtol=1e-6)

    def test_graybody_energy_balance(self):
        """Graybody integral over frequency equals L_absorbed."""
        from tengri.components.dust.emission import graybody
        from tengri.components.dust.emission._physics import integrate_lnu_over_nu

        # UV-to-radio grid: 0.01 dex spacing from 1e2 to 1e8 Angstrom
        wave = jnp.array(10.0 ** np.arange(2.0, 8.0, 0.01))
        L_abs = 1e10

        sed = graybody(
            wave,
            L_absorbed=L_abs,
            dust_T=30.0,
            dust_beta_ir=2.0,
            dust_lambda_0_um=200.0,
            redshift=0.0,
        )

        # Use the canonical integration function from _physics
        total_absorbed = integrate_lnu_over_nu(sed, wave)

        np.testing.assert_allclose(
            float(total_absorbed),
            L_abs,
            rtol=1e-4,
            atol=0,
            err_msg="Graybody energy balance failed",
        )

    def test_graybody_different_pivots_different_shapes(self):
        """Different lambda_0_um values produce different spectral shapes."""
        from tengri.components.dust.emission import graybody

        wave_ir = jnp.logspace(5, 8, 300)

        sed_100 = graybody(
            wave_ir,
            L_absorbed=1e10,
            dust_T=25.0,
            dust_beta_ir=2.0,
            dust_lambda_0_um=100.0,
        )
        sed_200 = graybody(
            wave_ir,
            L_absorbed=1e10,
            dust_T=25.0,
            dust_beta_ir=2.0,
            dust_lambda_0_um=200.0,
        )

        # The spectra should differ (different optical depth profiles)
        # At short wavelengths (where tau is large), the two should differ more
        assert not jnp.allclose(sed_100, sed_200, rtol=1e-3)

        # Find peaks
        peak_100 = float(wave_ir[jnp.argmax(sed_100)])
        peak_200 = float(wave_ir[jnp.argmax(sed_200)])

        # The 100 um pivot should peak blueward of 200 um pivot
        assert peak_100 < peak_200


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
