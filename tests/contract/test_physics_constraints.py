# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for physical consistency of tengri models.

These tests validate that the code obeys known astrophysical relationships,
conservation laws, and analytic limits. No SSP data files are needed — all
tests use synthetic data or standalone functions.

References
----------
- Calzetti et al. 2000, ApJ, 533, 682 — starburst attenuation law
- Cardelli, Clayton & Mathis 1989, ApJ, 345, 245 — MW extinction
- Osterbrock & Ferland 2006, AGN3 — Case B recombination
- Murphy et al. 2011, ApJ, 737, 67 — radio-SFR calibration
- Lehmer et al. 2010, ApJ, 724, 559 — XRB scaling
- Bell 2003, ApJ, 586, 794 — FIR-radio correlation
- Planck Collaboration 2018 — cosmological parameters
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds


# ── 1. Balmer decrement physics (Case B recombination + dust) ─────


class TestBalmerDecrementPhysics:
    """Dust reddening of Balmer lines obeys radiative transfer predictions.

    Intrinsic Ha/Hb = 2.86 (Case B, T=10^4 K, Osterbrock & Ferland 2006).
    Dust increases the ratio: Ha/Hb_obs = 2.86 * 10^(0.4 * A_V * (k_Hb - k_Ha)).
    """

    def test_calzetti_differential_reddening(self):
        """Calzetti law predicts Ha/Hb ~ 3.9 at A_V = 1."""
        from tengri.components.dust.attenuation import calzetti

        wave = jnp.array([4862.76, 6564.72])  # Hb, Ha
        k = calzetti(wave)
        k_hb, k_ha = float(k[0]), float(k[1])

        # Ha/Hb must increase with dust (k_Hb > k_Ha for normal curves)
        assert k_hb > k_ha, "Hbeta must be more attenuated than Halpha"

        # Predicted Balmer decrement at A_V = 1 (tau_V = A_V / 1.086)
        a_v = 1.0
        delta_k = k_hb - k_ha
        predicted_ratio = 2.86 * 10 ** (0.4 * a_v * delta_k)

        # Should be ~3.5-4.2 depending on exact curve normalization
        assert 3.0 < predicted_ratio < 5.0, (
            f"Balmer decrement at A_V=1 = {predicted_ratio:.2f}, expected 3.0-5.0"
        )

    def test_zero_dust_intrinsic_ratio(self):
        """At zero dust, differential reddening is zero → ratio stays 2.86."""
        from tengri.components.dust.attenuation import calzetti

        wave = jnp.array([4862.76, 6564.72])
        k = calzetti(wave)

        # Zero optical depth: transmission = exp(0) = 1 at both wavelengths
        a_v = 0.0
        predicted_ratio = 2.86 * 10 ** (0.4 * a_v * (float(k[0]) - float(k[1])))
        np.testing.assert_allclose(predicted_ratio, 2.86, rtol=1e-10)

    def test_decrement_increases_with_dust(self):
        """Balmer decrement must increase monotonically with A_V."""
        from tengri.components.dust.attenuation import calzetti

        wave = jnp.array([4862.76, 6564.72])
        k = calzetti(wave)
        delta_k = float(k[0]) - float(k[1])

        ratios = []
        for a_v in [0.0, 0.5, 1.0, 2.0, 4.0]:
            ratios.append(2.86 * 10 ** (0.4 * a_v * delta_k))

        # Monotonically increasing
        assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))


# ── 2. IRX-beta direction (Meurer+1999) ───────────────────────────


class TestIRXBetaDirection:
    """Dust moves galaxies upper-right on the IRX-beta diagram.

    More dust → higher IRX (more IR relative to UV) AND redder beta.
    """

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 5000)

    def test_dust_increases_irx_and_reddens_beta(self, wave):
        """Adding dust to a UV-bright SED increases IRX and reddens beta."""
        from tengri.components.dust.attenuation import calzetti
        from tengri.utils.sed_quantities import (
            compute_irx,
            compute_l_tir,
            compute_uv_luminosity_1600,
            compute_uv_slope_beta,
        )

        # Intrinsic power-law SED: f_nu ~ nu^0.5 (blue, star-forming)
        c_aa = 2.99792458e18
        nu = c_aa / wave
        sed_intrinsic = 1e30 * (nu / 1e15) ** 0.5

        k_curve = calzetti(wave)
        irx_values = []
        beta_values = []

        for tau_v in [0.0, 0.5, 1.0, 2.0]:
            transmission = jnp.exp(-tau_v * k_curve)
            sed_atten = sed_intrinsic * transmission

            # Add IR emission scaled to absorbed energy
            absorbed = sed_intrinsic - sed_atten
            nu_arr = c_aa / wave
            l_abs = -jnp.trapezoid(absorbed, nu_arr)
            # Simple MBB dust emission at 30K in FIR
            t_dust = 30.0
            h_cgs, k_b = 6.626e-27, 1.381e-16
            bb = 2 * h_cgs * nu**3 / c_aa**2 / (jnp.exp(h_cgs * nu / (k_b * t_dust)) - 1.0 + 1e-30)
            bb_norm = -jnp.trapezoid(bb, nu)
            sed_dust = jnp.where(bb_norm > 0, bb * l_abs / bb_norm, 0.0)

            sed_total = sed_atten + sed_dust
            l_tir = compute_l_tir(sed_total, wave)
            l_uv = compute_uv_luminosity_1600(sed_total, wave)
            irx = compute_irx(l_tir, l_uv)
            beta = compute_uv_slope_beta(sed_atten, wave)  # beta from stellar UV

            irx_values.append(float(irx))
            beta_values.append(float(beta))

        # IRX must increase with dust (skip tau=0 where L_TIR ~ 0)
        for i in range(1, len(irx_values) - 1):
            assert irx_values[i] < irx_values[i + 1], (
                f"IRX not increasing: tau_v step {i} → {i + 1}"
            )

        # Beta must redden (increase) with dust
        for i in range(len(beta_values) - 1):
            assert beta_values[i] < beta_values[i + 1], f"Beta not reddening: step {i} → {i + 1}"


# ── 3. SFR indicator consistency ──────────────────────────────────


class TestSFRIndicatorConsistency:
    """SFR tracers should be self-consistent (Murphy+2011, Lehmer+2010)."""

    def test_radio_sfr_roundtrip(self):
        """SFR → L_radio → SFR round-trip within 1%."""
        from tengri.utils.sed_quantities import compute_l_radio_1p4ghz_from_sfr

        for sfr_input in [0.1, 1.0, 10.0, 100.0]:
            l_radio = float(compute_l_radio_1p4ghz_from_sfr(jnp.array(sfr_input)))
            # Murphy+2011: SFR = 5.52e-22 * L_1.4GHz (erg/s/Hz → Msun/yr)
            sfr_recovered = 5.52e-22 * l_radio
            np.testing.assert_allclose(
                sfr_recovered,
                sfr_input,
                rtol=0.01,
                err_msg=f"Radio SFR round-trip failed for SFR={sfr_input}",
            )

    def test_hmxb_scales_linearly_with_sfr(self):
        """HMXB X-ray luminosity scales linearly with SFR (Lehmer+2010)."""
        from tengri.utils.sed_quantities import compute_l_x_xrb

        # Fix stellar mass to isolate HMXB (SFR-dependent) component
        m_star = 0.0  # no LMXB contribution
        sfrs = [1.0, 10.0, 100.0]
        lx = [float(compute_l_x_xrb(jnp.array(s), jnp.array(m_star))) for s in sfrs]

        # L_X should scale linearly: L(10 SFR) / L(SFR) = 10
        np.testing.assert_allclose(lx[1] / lx[0], 10.0, rtol=1e-10)
        np.testing.assert_allclose(lx[2] / lx[0], 100.0, rtol=1e-10)

    def test_lmxb_scales_linearly_with_mass(self):
        """LMXB X-ray luminosity scales linearly with M* (Lehmer+2010)."""
        from tengri.utils.sed_quantities import compute_l_x_xrb

        sfr = 0.0  # no HMXB contribution
        masses = [1e9, 1e10, 1e11]
        lx = [float(compute_l_x_xrb(jnp.array(sfr), jnp.array(m))) for m in masses]

        np.testing.assert_allclose(lx[1] / lx[0], 10.0, rtol=1e-10)
        np.testing.assert_allclose(lx[2] / lx[0], 100.0, rtol=1e-10)

    def test_hmxb_calibration_value(self):
        """HMXB coefficient matches Lehmer+2010: 2.6e39 erg/s per Msun/yr."""
        from tengri.utils.sed_quantities import compute_l_x_xrb

        lx = float(compute_l_x_xrb(jnp.array(1.0), jnp.array(0.0)))
        np.testing.assert_allclose(lx, 2.6e39, rtol=0.01)


# ── 4. Age of universe constraints ────────────────────────────────


class TestAgeOfUniverseConstraint:
    """Cosmological age must obey Planck 2018 predictions."""

    def test_age_at_z0(self):
        """Age of universe at z=0 ~ 13.8 Gyr (Planck 2018)."""
        from tengri.utils.cosmology import age_at_z

        age_gyr = float(age_at_z(0.0))
        np.testing.assert_allclose(
            age_gyr,
            13.8,
            atol=0.3,
            err_msg=f"Age at z=0 = {age_gyr:.2f} Gyr, expected ~13.8",
        )

    def test_age_decreases_with_redshift(self):
        """Age must decrease monotonically with redshift."""
        from tengri.utils.cosmology import age_at_z

        ages = [float(age_at_z(z)) for z in [0.0, 0.5, 1.0, 2.0, 5.0]]
        assert all(ages[i] > ages[i + 1] for i in range(len(ages) - 1))

    def test_age_at_z1_about_6gyr(self):
        """Age at z=1 ~ 5.9 Gyr (Planck 2018)."""
        from tengri.utils.cosmology import age_at_z

        age_gyr = float(age_at_z(1.0))
        np.testing.assert_allclose(
            age_gyr,
            5.9,
            atol=0.5,
            err_msg=f"Age at z=1 = {age_gyr:.2f} Gyr, expected ~5.9",
        )

    def test_age_positive_at_high_z(self):
        """Age must be positive even at very high redshift."""
        from tengri.utils.cosmology import age_at_z

        for z in [5.0, 10.0, 20.0]:
            age = float(age_at_z(z))
            assert age > 0, f"Negative age at z={z}: {age}"


# ── 5. Radio-FIR correlation ──────────────────────────────────────


class TestRadioFIRCorrelation:
    """FIR-radio correlation parameter q_TIR (Bell 2003)."""

    def test_q_ir_roundtrip(self):
        """q_TIR definition round-trips correctly (Bell 2003).

        Construct L_TIR and L_radio that should give q_TIR = 2.64 by
        definition, then verify compute_q_ir reproduces it.
        """
        from tengri.utils.physics_constants import L_SUN
        from tengri.utils.sed_quantities import compute_q_ir

        # Bell 2003: q_TIR = log10(L_TIR / 3.75e12 W) - log10(L_1.4GHz / W/Hz)
        # Set L_1.4GHz = 1e22 W/Hz (typical LIRG) and solve for L_TIR at q=2.64
        q_target = 2.64
        l_radio_w_hz = 1e22  # W/Hz
        l_radio_erg = l_radio_w_hz * 1e7  # erg/s/Hz
        l_tir_w = 3.75e12 * 10**q_target * l_radio_w_hz
        l_tir_lsun = l_tir_w / (L_SUN * 1e-7)

        q = float(compute_q_ir(jnp.array(l_tir_lsun), jnp.array(l_radio_erg)))
        np.testing.assert_allclose(
            q, q_target, atol=0.01, err_msg=f"q_TIR round-trip: got {q:.3f}"
        )

    def test_q_ir_finite_for_physical_inputs(self):
        """q_TIR must be finite for any positive L_TIR and L_radio."""
        from tengri.utils.sed_quantities import compute_q_ir

        for l_tir in [1e8, 1e10, 1e12, 1e14]:
            for l_radio in [1e20, 1e22, 1e24]:
                q = float(compute_q_ir(jnp.array(l_tir), jnp.array(l_radio)))
                assert np.isfinite(q), f"q_TIR not finite for L_TIR={l_tir}, L_radio={l_radio}"


# ── 6. Dust attenuation curve analytic values ─────────────────────


class TestDustAttenuationAnalyticValues:
    """Dust curves must match published analytic formulae."""

    def test_calzetti_vband_normalization(self):
        """Calzetti k(5500A) ~ 1.0 (normalized at V-band)."""
        from tengri.components.dust.attenuation import calzetti

        k_v = float(calzetti(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.05)

    def test_calzetti_uv_value(self):
        """Calzetti k(1500A) ~ 2.55 (normalized A(lambda)/A(V)).

        The Calzetti function returns k(lambda) = (k'(lambda)+R_V)/R_V,
        normalized so k(V) ~ 1. At 1500A this gives ~2.5-2.6.
        """
        from tengri.components.dust.attenuation import calzetti

        k_uv = float(calzetti(jnp.array([1500.0]))[0])
        # Much higher than k(V)=1 (strong UV attenuation)
        assert k_uv > 2.0, f"k(1500A) = {k_uv:.2f}, expected > 2"
        np.testing.assert_allclose(k_uv, 2.55, rtol=0.10, err_msg=f"Calzetti k(1500A)={k_uv:.2f}")

    def test_calzetti_nir_low(self):
        """Calzetti k(2.2um) should be small (<0.15)."""
        from tengri.components.dust.attenuation import calzetti

        k_nir = float(calzetti(jnp.array([22000.0]))[0])
        assert 0.0 <= k_nir < 0.2, f"k(2.2um) = {k_nir:.3f}, expected < 0.2"

    def test_power_law_exact(self):
        """Power-law k(lambda) = (lambda/5500)^n must be exact."""
        from tengri.components.dust.attenuation import power_law

        n = -0.7
        for lam in [1500.0, 2800.0, 5500.0, 10000.0]:
            k = float(power_law(jnp.array([lam]), n_slope=n)[0])
            expected = (lam / 5500.0) ** n
            np.testing.assert_allclose(
                k, expected, rtol=1e-10, err_msg=f"Power-law k({lam}) mismatch"
            )

    def test_cardelli_vband_unity(self):
        """Cardelli A(V)/A(V) = 1.0 by definition."""
        from tengri.components.dust.attenuation import cardelli

        k_v = float(cardelli(jnp.array([5500.0]), dust_Rv=3.1)[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.05, err_msg=f"Cardelli k(V)={k_v:.3f}")

    def test_cardelli_bband(self):
        """Cardelli at B-band: A(B)/A(V) = 1 + 1/R_V = 1.323 for R_V=3.1."""
        from tengri.components.dust.attenuation import cardelli

        k_b = float(cardelli(jnp.array([4400.0]), dust_Rv=3.1)[0])
        expected = 1.0 + 1.0 / 3.1  # = 1.3226
        np.testing.assert_allclose(
            k_b, expected, atol=0.1, err_msg=f"Cardelli k(B)={k_b:.3f}, expected {expected:.3f}"
        )


# ── 7. Dust physical monotonicity ─────────────────────────────────


class TestDustPhysicalMonotonicity:
    """Dust attenuation curves must obey basic physical constraints."""

    @pytest.fixture
    def optical_uv_wave(self):
        """Wavelength grid from 1200A to 10000A (UV to optical/NIR)."""
        return jnp.linspace(1200.0, 10000.0, 500)

    def test_calzetti_positive_everywhere(self, optical_uv_wave):
        """k(lambda) >= 0 at all wavelengths."""
        from tengri.components.dust.attenuation import calzetti

        k = np.asarray(calzetti(optical_uv_wave))
        assert_non_negative(k, name="k", msg="Calzetti k(lambda) has negative values")

    def test_smc_positive_everywhere(self, optical_uv_wave):
        """SMC k(lambda) >= 0 at all wavelengths."""
        from tengri.components.dust.attenuation import smc

        k = np.asarray(smc(optical_uv_wave))
        assert_non_negative(k, name="k", msg="SMC k(lambda) has negative values")

    def test_calzetti_uv_steeper_than_nir(self):
        """k(UV) must be much larger than k(NIR) for Calzetti."""
        from tengri.components.dust.attenuation import calzetti

        k = calzetti(jnp.array([1500.0, 5500.0, 10000.0]))
        k_uv, k_v, k_nir = float(k[0]), float(k[1]), float(k[2])

        assert k_uv > k_v > k_nir, (
            f"Calzetti not monotonically decreasing: "
            f"k(1500)={k_uv:.2f}, k(5500)={k_v:.2f}, k(10000)={k_nir:.2f}"
        )
        # UV should be at least 5x steeper than NIR
        assert k_uv / k_nir > 5, f"UV/NIR ratio = {k_uv / k_nir:.1f}, expected > 5"

    def test_transmission_bounded(self):
        """Transmission exp(-tau * k) must be in (0, 1] for tau > 0."""
        from tengri.components.dust.attenuation import calzetti

        wave = jnp.linspace(1500.0, 10000.0, 100)
        k = calzetti(wave)

        for tau_v in [0.1, 0.5, 1.0, 3.0, 5.0]:
            trans = np.asarray(jnp.exp(-tau_v * k))
            assert np.all(trans > 0), f"Transmission <= 0 at tau_v={tau_v}"
            assert np.all(trans <= 1.0), f"Transmission > 1 at tau_v={tau_v}"

    def test_smc_steeper_than_calzetti_in_uv(self):
        """SMC curve rises more steeply into the UV than Calzetti."""
        from tengri.components.dust.attenuation import calzetti, smc

        wave = jnp.array([1500.0, 5500.0])
        calz_ratio = float(calzetti(wave)[0] / calzetti(wave)[1])
        smc_ratio = float(smc(wave)[0] / smc(wave)[1])
        assert smc_ratio > calz_ratio, (
            f"SMC UV/V ratio ({smc_ratio:.1f}) not steeper than Calzetti ({calz_ratio:.1f})"
        )

    def test_kriek_conroy_bump_increases_2175(self):
        """Kriek-Conroy with E_b > 0 must show enhanced 2175A absorption."""
        from tengri.components.dust.attenuation import kriek_conroy

        wave = jnp.array([2175.0, 3000.0])

        k_no_bump = kriek_conroy(wave, dust_bump_strength=0.0)
        k_with_bump = kriek_conroy(wave, dust_bump_strength=1.0)

        # At 2175A, the bump version should have higher k
        assert float(k_with_bump[0]) > float(k_no_bump[0]), (
            "2175A bump not visible in Kriek-Conroy"
        )
        # At 3000A (away from bump), difference should be smaller
        bump_2175 = float(k_with_bump[0]) - float(k_no_bump[0])
        bump_3000 = float(k_with_bump[1]) - float(k_no_bump[1])
        assert bump_2175 > bump_3000, "Bump not peaked at 2175A"


# ── 8. UV slope beta analytic values ──────────────────────────────


class TestUVSlopeBetaAnalytic:
    """UV slope beta must match analytic predictions for power-law spectra."""

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2.8, 4.0, 2000)  # 630A to 10000A

    def test_flat_fnu_gives_beta_minus2(self, wave):
        """Flat f_nu → beta = -2.0 (definition of f_lambda ~ lambda^beta)."""
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        sed = jnp.ones_like(wave) * 1e30
        beta = float(compute_uv_slope_beta(sed, wave))
        np.testing.assert_allclose(beta, -2.0, atol=0.02)

    def test_power_law_fnu_alpha1(self, wave):
        """f_nu ~ nu^1 (blue spectrum) → beta = -3.0.

        Since f_nu ~ nu^alpha and beta = d(ln f_nu)/d(ln lambda) - 2,
        and nu = c/lambda, so f_nu ~ lambda^(-alpha), hence
        d(ln f_nu)/d(ln lambda) = -alpha, and beta = -alpha - 2.
        """
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        c_aa = 2.99792458e18
        nu = c_aa / wave
        sed = 1e20 * (nu / 1e15) ** 1.0  # f_nu ~ nu^1
        beta = float(compute_uv_slope_beta(sed, wave))
        np.testing.assert_allclose(beta, -3.0, atol=0.05)

    def test_power_law_fnu_alpha_minus1(self, wave):
        """f_nu ~ nu^(-1) (red spectrum) → beta = -1.0."""
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        c_aa = 2.99792458e18
        nu = c_aa / wave
        sed = 1e40 * (nu / 1e15) ** (-1.0)
        beta = float(compute_uv_slope_beta(sed, wave))
        np.testing.assert_allclose(beta, -1.0, atol=0.05)

    def test_redder_spectrum_gives_higher_beta(self, wave):
        """Redder f_nu (more negative alpha) gives higher (redder) beta.

        Since beta = -alpha - 2, decreasing alpha gives increasing beta.
        Test with ascending alpha order so betas are descending, then check.
        """
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        c_aa = 2.99792458e18
        nu = c_aa / wave
        alphas = [-2.0, -1.0, 0.0, 1.0]
        betas = []
        for alpha in alphas:
            sed = 1e30 * (nu / 1e15) ** alpha
            betas.append(float(compute_uv_slope_beta(sed, wave)))

        # beta = -alpha - 2, so more negative alpha → higher beta
        # betas should decrease as alpha increases
        assert all(betas[i] > betas[i + 1] for i in range(len(betas) - 1)), (
            f"Beta not monotonically reddening with decreasing alpha: "
            f"alphas={alphas}, betas={betas}"
        )


# ── 9. Dn4000 physical limits ─────────────────────────────────────


class TestDn4000PhysicalLimits:
    """Dn4000 must obey known physical bounds."""

    @pytest.fixture
    def wave(self):
        return jnp.linspace(3500.0, 4500.0, 1000)

    def test_flat_spectrum_gives_dn4000_unity(self, wave):
        """Flat SED should give Dn4000 = 1.0."""
        from tengri.utils.sed_quantities import compute_dn4000

        sed = jnp.ones_like(wave) * 1e30
        dn = float(compute_dn4000(sed, wave))
        np.testing.assert_allclose(dn, 1.0, atol=0.01)

    def test_red_spectrum_gives_dn4000_above_1(self, wave):
        """Red (decreasing f_nu with lambda) spectrum → Dn4000 > 1.0.

        If red side (4000-4100) has more flux than blue side (3850-3950)
        in f_nu units, Dn4000 > 1.
        """
        from tengri.utils.sed_quantities import compute_dn4000

        # f_nu increasing with lambda (redder = more flux in f_nu)
        sed = 1e30 * (wave / 4000.0)
        dn = float(compute_dn4000(sed, wave))
        assert dn > 1.0, f"Red spectrum Dn4000 = {dn:.3f}, expected > 1"

    def test_step_function_at_4000(self, wave):
        """Step function with 2x flux above 4000A → Dn4000 ~ 2.0."""
        from tengri.utils.sed_quantities import compute_dn4000

        sed = jnp.where(wave >= 4000.0, 2e30, 1e30)
        dn = float(compute_dn4000(sed, wave))
        np.testing.assert_allclose(dn, 2.0, atol=0.05)


# ── 10. Cosmological luminosity distance ──────────────────────────


class TestLuminosityDistancePhysics:
    """Luminosity distance must obey basic relativistic scaling."""

    def test_dl_positive(self):
        """dL must be positive for z > 0."""
        from tengri.utils.cosmology import luminosity_distance

        for z in [0.01, 0.1, 1.0, 5.0]:
            dl = float(luminosity_distance(z))
            assert dl > 0, f"dL({z}) = {dl}"

    def test_dl_monotonic(self):
        """dL must increase monotonically with z."""
        from tengri.utils.cosmology import luminosity_distance

        dls = [float(luminosity_distance(z)) for z in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]]
        assert all(dls[i] < dls[i + 1] for i in range(len(dls) - 1))

    def test_dl_low_z_hubble_law(self):
        """At low z, dL ~ c*z/H0 (Hubble law)."""
        from tengri.utils.cosmology import DEFAULT_H0, luminosity_distance
        from tengri.utils.physics_constants import C_KM_S, MPC_CM

        z = 0.01
        dl = float(luminosity_distance(z))
        # Hubble law: dL = c*z*(1+z)/H0  (first-order, flat)
        dl_hubble = C_KM_S * z * (1 + z) / DEFAULT_H0 * MPC_CM
        np.testing.assert_allclose(
            dl, dl_hubble, rtol=0.02, err_msg="Low-z dL should match Hubble law"
        )
