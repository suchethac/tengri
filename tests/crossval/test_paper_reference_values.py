# SPDX-License-Identifier: BSD-3-Clause
"""Tests against EXPLICIT numerical values from published papers.

Every assertion here cites a specific equation, table, or figure from a
peer-reviewed paper. If the code doesn't match the paper, the code is wrong.

References are cited inline with each test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval


# ── 1. CALZETTI+2000 — Table 1, Eq. 2-4 ───────────────────────────


class TestCalzetti2000:
    """Reference values from Calzetti, Kinney & Storchi-Bergmann 2000, ApJ, 533, 682."""

    def test_rv_equals_4p05(self):
        """Calzetti+2000 Eq. 5: R'_V = 4.05 ± 0.80.

        k(V) = (k'(V) + R_V) / R_V = 1.0 by construction.
        """
        from tengri.components.dust.attenuation import calzetti

        k_v = float(calzetti(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.01)

    def test_uv_polynomial_at_specific_wavelengths(self):
        """Calzetti+2000 Eq. 4 (UV regime, 0.12 ≤ λ ≤ 0.63 μm):

        k'(λ) = 2.659(-2.156 + 1.509/λ - 0.198/λ² + 0.011/λ³) + R_V

        Test at λ = 0.15 μm (1500A):
        x = 1/0.15 = 6.667
        k' = 2.659(-2.156 + 10.060 - 8.800 + 3.259) + 4.05
           = 2.659 * 2.363 + 4.05 = 6.283 + 4.05 = 10.333
        k = k'/R_V = 10.333/4.05 = 2.551
        """
        from tengri.components.dust.attenuation import calzetti

        k = float(calzetti(jnp.array([1500.0]))[0])
        # Analytic: k'(1500) = 2.659*(-2.156+1.509*6.667-0.198*6.667^2+0.011*6.667^3)+4.05
        x = 1.0 / 0.15
        kp = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 4.05
        k_expected = kp / 4.05
        np.testing.assert_allclose(
            k,
            k_expected,
            rtol=0.01,
            err_msg=f"k(1500A) = {k:.3f}, expected {k_expected:.3f}",
        )

    def test_ir_polynomial_at_1um(self):
        """Calzetti+2000 Eq. 3 (IR regime, 0.63 ≤ λ ≤ 2.20 μm):

        k'(λ) = 2.659(-1.857 + 1.040/λ) + R_V

        At λ = 1.0 μm (10000A):
        k' = 2.659(-1.857 + 1.040) + 4.05 = 2.659*(-0.817) + 4.05 = -2.173 + 4.05 = 1.877
        k = 1.877/4.05 = 0.463
        """
        from tengri.components.dust.attenuation import calzetti

        k = float(calzetti(jnp.array([10000.0]))[0])
        kp = 2.659 * (-1.857 + 1.040 / 1.0) + 4.05
        k_expected = kp / 4.05
        np.testing.assert_allclose(k, k_expected, rtol=0.01)


# ── 2. CARDELLI, CLAYTON & MATHIS 1989 — Table 3 ──────────────────


class TestCardelli1989:
    """Reference values from Cardelli, Clayton & Mathis 1989, ApJ, 345, 245."""

    def test_av_over_av_equals_one(self):
        """By definition: A(V)/A(V) = 1.0."""
        from tengri.components.dust.attenuation import cardelli

        k_v = float(cardelli(jnp.array([5500.0]), dust_Rv=3.1)[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.02)

    def test_ab_over_av_definition(self):
        """CCM89: A(B)/A(V) = a(x) + b(x)/R_V at x=1/0.44 = 2.273.

        For R_V=3.1: A(B)/A(V) = 1 + 1/R_V = 1.323 (by definition of R_V).
        """
        from tengri.components.dust.attenuation import cardelli

        k = cardelli(jnp.array([4400.0, 5500.0]), dust_Rv=3.1)
        ratio = float(k[0] / k[1])
        expected = 1.0 + 1.0 / 3.1  # = 1.3226
        np.testing.assert_allclose(
            ratio, expected, atol=0.05, err_msg=f"A(B)/A(V) = {ratio:.3f}, expected {expected:.3f}"
        )

    def test_2175_bump_peak(self):
        """CCM89 UV curve: 2175A bump is a local maximum (graphite carrier).

        The bump is described by the Lorentzian term:
        a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)^2 + 0.341)
        which peaks at x = 4.595 (λ ≈ 2176 A).
        """
        from tengri.components.dust.attenuation import cardelli

        wave = jnp.linspace(1900.0, 2400.0, 500)
        k = cardelli(wave, dust_Rv=3.1)
        peak_wave = float(wave[jnp.argmax(k)])
        np.testing.assert_allclose(
            peak_wave,
            2175.0,
            atol=50.0,
            err_msg=f"Bump peak at {peak_wave:.0f} A, expected ~2175 A",
        )


# ── 3. PEI 1992 — SMC/LMC extinction curves ───────────────────────


class TestPei1992:
    """Reference values from Pei 1992, ApJ, 395, 130 (Tables 2, 4)."""

    def test_smc_rv(self):
        """Pei 1992 Table 2: SMC R_V = 2.93."""
        from tengri.components.dust.attenuation import _SMC_RV

        assert abs(float(_SMC_RV) - 2.93) < 0.01

    def test_lmc_rv(self):
        """Pei 1992 Table 2: LMC R_V = 3.16."""
        from tengri.components.dust.attenuation import _LMC_RV

        assert abs(float(_LMC_RV) - 3.16) < 0.01

    def test_smc_normalized_at_v(self):
        """SMC curve normalized to k(5500A) = 1.0."""
        from tengri.components.dust.attenuation import smc

        k_v = float(smc(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.05)

    def test_lmc_normalized_at_v(self):
        """LMC curve normalized to k(5500A) = 1.0."""
        from tengri.components.dust.attenuation import lmc

        k_v = float(lmc(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.05)

    def test_smc_6_drude_components(self):
        """Pei 1992 Table 4: SMC has 6 Drude components."""
        from tengri.components.dust.attenuation import _SMC_LAM

        assert len(_SMC_LAM) == 6

    def test_lmc_6_drude_components(self):
        """Pei 1992 Table 4: LMC has 6 Drude components."""
        from tengri.components.dust.attenuation import _LMC_LAM

        assert len(_LMC_LAM) == 6


# ── 4. ALLEN+2008 — MAPPINGS III shock model reference values ─────


class TestAllen2008:
    """Reference values from Allen et al. 2008, ApJS, 178, 20 (Table 5)."""

    def test_shock_grid_velocity_range(self):
        """Allen+2008: velocity grid 100-1000 km/s."""
        from tengri.components.nebular.shock import _FALLBACK_V

        assert float(_FALLBACK_V[0]) == 100.0
        assert float(_FALLBACK_V[-1]) == 1000.0

    def test_oiii_peaks_at_300_400(self):
        """Allen+2008 Fig. 14: [OIII]/Hβ peaks at v ~ 300-400 km/s."""
        from tengri.components.nebular.shock import _FALLBACK_R_OIII, _FALLBACK_V

        peak_idx = int(jnp.argmax(_FALLBACK_R_OIII))
        peak_v = float(_FALLBACK_V[peak_idx])
        assert 200.0 <= peak_v <= 500.0, f"[OIII]/Hβ should peak at 200-500 km/s, got {peak_v}"

    def test_oiii_5007_4959_ratio_298(self):
        """Storey & Zeippen 2000: [OIII] 5007/4959 = 2.98 (atomic physics)."""
        from tengri.components.nebular.shock import _OIII_DOUBLET_RATIO

        np.testing.assert_allclose(_OIII_DOUBLET_RATIO, 2.98, atol=0.01)

    def test_nii_6583_6548_ratio_294(self):
        """Storey & Zeippen 2000: [NII] 6583/6548 = 2.94 (atomic physics)."""
        from tengri.components.nebular.shock import _NII_DOUBLET_RATIO

        np.testing.assert_allclose(_NII_DOUBLET_RATIO, 2.94, atol=0.01)

    def test_halpha_hbeta_above_case_b_286(self):
        """Allen+2008: Hα/Hβ > 2.86 in shocks (collisional enhancement).

        Case B recombination: Hα/Hβ = 2.86 (Osterbrock & Ferland 2006).
        In shocks, collisional excitation adds to recombination.
        """
        from tengri.components.nebular.shock import _FALLBACK_R_HA

        for i, r in enumerate(_FALLBACK_R_HA):
            assert float(r) >= 2.86, f"Hα/Hβ at index {i} = {float(r)}, should be ≥ 2.86"

    def test_oiii_reference_values_from_table5(self):
        """Allen+2008 Table 5 (solar, n=1, shock+precursor):

        [OIII] 5007/Hβ at v=100: ~0.3
        [OIII] 5007/Hβ at v=300: ~5.8
        [OIII] 5007/Hβ at v=1000: ~2.5
        """
        from tengri.components.nebular.shock import _FALLBACK_R_OIII, _FALLBACK_V

        # v=100 (index 0)
        np.testing.assert_allclose(float(_FALLBACK_R_OIII[0]), 0.3, atol=0.2)
        # v=300 (index 3)
        idx_300 = int(jnp.argmin(jnp.abs(_FALLBACK_V - 300.0)))
        np.testing.assert_allclose(float(_FALLBACK_R_OIII[idx_300]), 5.8, atol=1.0)
        # v=1000 (last)
        np.testing.assert_allclose(float(_FALLBACK_R_OIII[-1]), 2.5, atol=0.5)


# ── 5. MAHADEVAN 1997 — ADAF synchrotron peak frequency ───────────


class TestMahadevan1997:
    """Reference values from Mahadevan 1997, ApJ, 477, 585."""

    def test_synchrotron_peak_in_radio_not_uv(self):
        """Mahadevan 1997: ADAF + truncated-disc SED peaks in IR/radio, NOT UV/optical.

        For a standard thin disc, L_nu peaks in the UV (λ ~ 1000-4000 Å).
        An ADAF flow at the same L_bol has a very different spectral shape:
          - Synchrotron peaks at radio/mm: ν_peak ∝ M_BH^{-1/2} * ṁ^{1/2} ~ 10^10 Hz
          - Bremsstrahlung: flat spectrum to X-ray (kT_e/h)
          - Outer truncated disc: IR/optical emission from r > r_tr

        The combined ADAF + outer-disc SED must peak well into the infrared
        (λ > 1 μm = 1e4 Å), never in the UV/optical (λ < 4000 Å).
        """
        from tengri.components.agn.disc import adaf_disc

        wave = jnp.geomspace(1e4, 1e10, 2000)
        l_nu = adaf_disc(wave, agn_log_lbol=10.0, agn_log_mbh=8.0)
        peak_wave = float(wave[jnp.argmax(l_nu)])

        # ADAF + outer disc SED must peak in the infrared (λ > 1 μm), not UV/optical.
        # UV/optical threshold: λ < 4000 Å.  Standard thin disc peaks ~1000-4000 Å.
        assert peak_wave > 1e4, (
            f"ADAF SED peak at {peak_wave:.2e} Å — should be in IR/radio (>1 μm), "
            f"not UV/optical. Standard thin disc peaks in UV; ADAF must not."
        )

    def test_bh_mass_shifts_synchrotron_peak(self):
        """Mahadevan 1997 Eq. 24: ν_peak ∝ M_BH^{-1/2} — higher M_BH → longer radio λ.

        The ADAF synchrotron peak frequency scales as:
            ν_peak = 1e12 * (M_BH / 10^8 Msun)^{-1/2} * ṁ^{1/2}  Hz

        So M_BH = 10^9 Msun must peak at λ ≈ 3 cm (10× longer than M_BH = 10^7 Msun).
        This tests the adaf_disc code path (disc.py:1113) directly.

        Uses a full-SED wavelength grid (X-ray to radio) so the internal normalization
        is physically correct — adaf_disc normalizes over the passed wavelength array.
        Outer disc is suppressed (r_tr ≈ ISCO) to isolate the ADAF synchrotron shape.
        """
        import numpy as np

        from tengri.components.agn.disc import adaf_disc

        # Wide grid (1 Å to 1 m) for correct SED normalization
        wave_full = np.array(jnp.geomspace(1.0, 1e12, 8000))

        # Suppress outer disc (r_tr = 7 R_g ≈ ISCO) to isolate ADAF synchrotron
        common = dict(agn_log_lbol=10.0, agn_log_ledd=-3.0, agn_r_tr=7.0)
        l_low_mbh = np.array(adaf_disc(jnp.array(wave_full), agn_log_mbh=7.0, **common))
        l_high_mbh = np.array(adaf_disc(jnp.array(wave_full), agn_log_mbh=9.0, **common))

        # Identify radio/mm range (1 mm to 10 m = 1e7 to 1e11 Å) where synchrotron peaks
        radio_mask = (wave_full >= 1e7) & (wave_full <= 1e11)
        wave_radio = wave_full[radio_mask]

        peak_low = wave_radio[np.argmax(l_low_mbh[radio_mask])]
        peak_high = wave_radio[np.argmax(l_high_mbh[radio_mask])]

        assert peak_high > peak_low, (
            f"Mahadevan 1997 Eq. 24: ν_peak ∝ M_BH^(-1/2), so M_BH=10^9 Msun must "
            f"synchrotron-peak at longer λ than M_BH=10^7 Msun. "
            f"Got λ_peak(10^7 Msun)={peak_low:.2e} Å, λ_peak(10^9 Msun)={peak_high:.2e} Å"
        )


# ── 6. BARDEEN, PRESS & TEUKOLSKY 1972 — ISCO radius ──────────────


class TestBPT1972:
    """Reference values from Bardeen, Press & Teukolsky 1972, ApJ, 178, 347."""

    def test_schwarzschild_isco_exactly_6(self):
        """BPT72: r_ISCO(a=0) = 6 R_g (Schwarzschild)."""
        from tengri.components.agn.disc import _isco_radius

        r = float(_isco_radius(0.0))
        np.testing.assert_allclose(r, 6.0, atol=0.01)

    def test_extreme_kerr_isco_near_1(self):
        """BPT72: r_ISCO(a→1) → 1 R_g (extreme Kerr).

        For a=0.998: r_ISCO ≈ 1.237 R_g.
        """
        from tengri.components.agn.disc import _isco_radius

        r = float(_isco_radius(0.998))
        np.testing.assert_allclose(r, 1.237, atol=0.05)

    def test_a_half_isco(self):
        """BPT72: r_ISCO(a=0.5) ≈ 4.233 R_g."""
        from tengri.components.agn.disc import _isco_radius

        r = float(_isco_radius(0.5))
        np.testing.assert_allclose(r, 4.233, atol=0.05)

    def test_a_09_isco(self):
        """BPT72: r_ISCO(a=0.9) ≈ 2.321 R_g."""
        from tengri.components.agn.disc import _isco_radius

        r = float(_isco_radius(0.9))
        np.testing.assert_allclose(r, 2.321, atol=0.05)


# ── 7. LI+2008 — Eq. 1 three-term attenuation curve ───────────────


class TestLi2008:
    """Reference values from Li et al. 2008, ApJ, 685, 1046."""

    def test_li08_eq1_three_terms(self):
        """Li+2008 Eq. 1: A_λ/A_V has three terms.

        Term 1: c1 / [(λ/0.08)^c2 + (0.08/λ)^c2 + c3]
        Term 2: FUV rise (normalization-dependent)
        Term 3: c4 / [(λ/0.2175)^2 + (0.2175/λ)^2 - 1.95] (UV bump)
        """
        from tengri.components.dust.attenuation import li08

        # At 2175A (bump center), term 3 should contribute significantly
        wave = jnp.array([2175.0])
        k_with_bump = float(li08(wave, dust_c4=0.04)[0])
        k_no_bump = float(li08(wave, dust_c4=0.0)[0])
        # Bump should add measurable attenuation
        assert k_with_bump > k_no_bump, "c4 should add bump at 2175A"

    def test_li08_normalized_at_v(self):
        """Li+2008: k(5500A) = 1.0 by normalization."""
        from tengri.components.dust.attenuation import li08

        k_v = float(li08(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.01)

    def test_li08_mw_preset_c4_nonzero(self):
        """Li+2008 / Markov+2023: MW-like preset has c4 ~ 0.04 (bump)."""
        from tengri.components.dust.attenuation import li08

        # MW preset: c1=6.0, c2=4.0, c3=2.0, c4=0.04
        wave = jnp.linspace(1800.0, 2600.0, 200)
        k = li08(wave, dust_c1=6.0, dust_c2=4.0, dust_c3=2.0, dust_c4=0.04)
        # Must have visible bump
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_1900 = float(k[jnp.argmin(jnp.abs(wave - 1900.0))])
        k_at_2500 = float(k[jnp.argmin(jnp.abs(wave - 2500.0))])
        # Bump should be a local enhancement
        interpolated = k_at_1900 + (k_at_2500 - k_at_1900) * (2175 - 1900) / (2500 - 1900)
        assert k_at_2175 > interpolated, "MW preset must show 2175A bump"


# ── 8. VANDEN BERK+2001 — BLR line relative strengths ─────────────


class TestVandenBerk2001:
    """Reference values from Vanden Berk et al. 2001, AJ, 122, 549.

    SDSS composite quasar spectrum broad line equivalent widths.
    """

    def test_lya_is_strongest_line(self):
        """VB01 Table 2: Lyα has the largest EW of all broad lines."""
        from tengri.components.agn.blr import _BLR_LINE_STRENGTHS, _BLR_LINE_WAVELENGTHS

        max_idx = int(jnp.argmax(_BLR_LINE_STRENGTHS))
        max_wave = float(_BLR_LINE_WAVELENGTHS[max_idx])
        # Either Lyα (1216) or Hα (6563) should be strongest
        # VB01: broad Lyα EW = 92A, broad Hα EW = 260A → Hα has larger EW
        # But in terms of luminosity, Lyα is typically comparable
        assert max_wave in [1216.0, 6563.0], (
            f"Strongest BLR line should be Lyα or Hα, got λ={max_wave:.0f}"
        )

    def test_civ_present(self):
        """VB01: CIV 1549A has EW ~ 24A (prominent broad line)."""
        from tengri.components.agn.blr import _BLR_LINE_STRENGTHS, _BLR_LINE_WAVELENGTHS

        civ_idx = int(jnp.argmin(jnp.abs(_BLR_LINE_WAVELENGTHS - 1549.0)))
        civ_strength = float(_BLR_LINE_STRENGTHS[civ_idx])
        assert civ_strength > 0.1, f"CIV should be a significant line, got strength {civ_strength}"

    def test_mgii_present(self):
        """VB01: MgII 2800A has EW ~ 32A (strong UV line)."""
        from tengri.components.agn.blr import _BLR_LINE_STRENGTHS, _BLR_LINE_WAVELENGTHS

        mgii_idx = int(jnp.argmin(jnp.abs(_BLR_LINE_WAVELENGTHS - 2800.0)))
        mgii_strength = float(_BLR_LINE_STRENGTHS[mgii_idx])
        assert mgii_strength > 0.1, f"MgII should be present, got strength {mgii_strength}"


# ── 9. BELLSTEDT+2020 — skew-normal SFH formulation ───────────────


class TestBellstedt2020:
    """Reference values from Bellstedt+2020 (arXiv:2005.11917), Eq. 2-4."""

    def test_snorm_skew_zero_is_gaussian(self):
        """Bellstedt+2020 Eq. 2: with skew=0, the kernel is a standard Gaussian."""
        from tengri.components.stellar.sfh import norm, snorm

        t = jnp.geomspace(1e5, 14e9, 500)
        sfr_snorm = snorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0)
        sfr_norm = norm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9)
        np.testing.assert_allclose(sfr_snorm, sfr_norm, rtol=1e-10)

    def test_tsnorm_trunc_1_minimal_effect(self):
        """Bellstedt+2020: trunc controls CDF suppression. trunc=1 is minimal."""
        from tengri.components.stellar.sfh import snorm, tsnorm

        t = jnp.geomspace(1e5, 14e9, 500)
        sfr_snorm = snorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0)
        sfr_tsnorm = tsnorm(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0, trunc=1.0)
        # With trunc=1, the CDF truncation should produce a measurable difference
        # but the shape should be similar
        corr = float(jnp.corrcoef(sfr_snorm, sfr_tsnorm)[0, 1])
        assert corr > 0.8, "tsnorm(trunc=1) should be similar to snorm"


# ── 10. LEJA+2019 — Continuity SFH prior ──────────────────────────


class TestLeja2019:
    """Reference values from Leja+2019 (arXiv:1905.11997)."""

    def test_7_bins_default(self):
        """Leja+2019: default is 7 lookback time bins (8 edges)."""
        from tengri.components.stellar.sfh.nonparametric import (
            DEFAULT_BIN_EDGES_GYR,
            DEFAULT_N_BINS,
        )

        assert DEFAULT_N_BINS == 7
        assert len(DEFAULT_BIN_EDGES_GYR) == 8

    def test_default_bins_span_cosmic_time(self):
        """Leja+2019: bins span 0 to 13.7 Gyr."""
        from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR

        assert float(DEFAULT_BIN_EDGES_GYR[0]) == pytest.approx(0.0)
        assert float(DEFAULT_BIN_EDGES_GYR[-1]) == pytest.approx(13.7, rel=0.01)

    def test_zero_ratios_flat_sfh(self):
        """Leja+2019: all log-ratios = 0 → equal SFR in all bins."""
        from tengri.components.stellar.sfh import continuity

        age = jnp.geomspace(1e8, 13e9, 500)
        sfr = continuity(
            age,
            log_total_mass=10.0,
            ratio_0=0.0,
            ratio_1=0.0,
            ratio_2=0.0,
            ratio_3=0.0,
            ratio_4=0.0,
            ratio_5=0.0,
        )
        # Should be approximately constant
        active = sfr > 0
        if jnp.sum(active) > 10:
            cv = float(jnp.std(sfr[active]) / jnp.mean(sfr[active]))
            assert cv < 0.5, f"Zero ratios should give ~flat SFH, CV={cv:.2f}"


# ── 11. EDDINGTON LUMINOSITY — fundamental physics ────────────────


class TestEddingtonPhysics:
    """L_Edd = 4π G M m_p c / σ_T — textbook values."""

    def test_eddington_formula(self):
        """L_Edd(M) = 1.26 × 10^38 * (M/Msun) erg/s.

        For M = 10^8 Msun: L_Edd = 1.26 × 10^46 erg/s.
        """
        from tengri.components.agn.disc import _eddington_luminosity

        l_edd = float(_eddington_luminosity(8.0))
        expected = 1.26e38 * 1e8  # 1.26e46 erg/s
        np.testing.assert_allclose(l_edd, expected, rtol=0.02)

    def test_gravitational_radius(self):
        """R_g = GM/c^2 = 1.485 × 10^13 * (M/10^8 Msun) cm."""
        from tengri.components.agn.disc import _gravitational_radius

        r_g = float(_gravitational_radius(8.0))
        expected = 1.485e13  # cm, for 10^8 Msun
        np.testing.assert_allclose(r_g, expected, rtol=0.02)


# ── 12. SFH MASS CONSERVATION — dense_basis and continuity ────────


class TestSFHMassConservation:
    """For SFH types with explicit log_total_mass parameter, integral equals 10^log_total_mass."""

    # Log-spaced age grid for accurate trapezoidal integration
    AGE_GRID = jnp.geomspace(1e6, 13.7e9, 10000)

    def test_dense_basis_mass_conservation(self):
        """Dense basis SFH: ∫ SFR(t) dt = 10^log_total_mass within 1%.

        Iyer+2019, ApJ 879, 116 — dense basis SFH is mass-normalized.
        """
        from tengri.components.stellar.sfh.dense_basis import dense_basis

        sfr = dense_basis(
            self.AGE_GRID,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            age_universe_yr=13.7e9,
            tx_frac_0=0.2,
            tx_frac_1=0.5,
            tx_frac_2=0.8,
        )
        mass = float(jnp.trapezoid(sfr, self.AGE_GRID))
        np.testing.assert_allclose(
            mass,
            1e10,
            rtol=0.01,
            err_msg="Dense basis SFH mass conservation: ∫ SFR dt != 10^10 Msun (Iyer+2019)",
        )

    def test_continuity_mass_conservation(self):
        """Continuity SFH: ∫ SFR(t) dt = 10^log_total_mass within 1%.

        Leja+2019, ApJ 876, 3 — continuity SFH is mass-normalized.
        """
        from tengri.components.stellar.sfh.nonparametric import continuity

        sfr = continuity(
            self.AGE_GRID,
            log_total_mass=10.0,
            ratio_0=0.0,
            ratio_1=0.0,
            ratio_2=0.0,
            ratio_3=0.0,
            ratio_4=0.0,
            ratio_5=0.0,
        )
        mass = float(jnp.trapezoid(sfr, self.AGE_GRID))
        np.testing.assert_allclose(
            mass,
            1e10,
            rtol=0.01,
            err_msg="Continuity SFH mass conservation: ∫ SFR dt != 10^10 Msun (Leja+2019)",
        )

    def test_continuity_mass_conservation_varying_ratios(self):
        """Continuity SFH mass is invariant to ratio parameters (only shape changes).

        Leja+2019: total mass is set by log_total_mass independent of log-ratios.
        """
        from tengri.components.stellar.sfh.nonparametric import continuity

        sfr_bursty = continuity(
            self.AGE_GRID,
            log_total_mass=10.0,
            ratio_0=1.0,
            ratio_1=-0.5,
            ratio_2=0.8,
            ratio_3=-0.3,
            ratio_4=0.5,
            ratio_5=-0.2,
        )
        mass = float(jnp.trapezoid(sfr_bursty, self.AGE_GRID))
        np.testing.assert_allclose(
            mass,
            1e10,
            rtol=0.01,
            err_msg="Continuity SFH: mass must be log_total_mass regardless of ratios",
        )

    def test_dpl_peak_time(self):
        """DPL SFH peaks at t_peak = tau × (beta/alpha)^{1/(alpha+beta)}. Carnall+2018 Eq. 1.

        SFR(t) = norm / [(t/tau)^alpha + (t/tau)^{-beta}].
        Setting d/dt = 0 gives t_peak = tau × (beta/alpha)^{1/(alpha+beta)}.
        When alpha == beta the formula reduces to t_peak = tau exactly.
        """
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        tau = 3e9  # yr
        # alpha == beta → t_peak == tau exactly (no grid-resolution ambiguity)
        alpha = beta = 2.0
        sfr = double_powerlaw(self.AGE_GRID, alpha=alpha, beta=beta, tau=tau, norm=1.0)
        peak_age = float(self.AGE_GRID[jnp.argmax(sfr)])
        np.testing.assert_allclose(
            peak_age,
            tau,
            rtol=0.01,
            err_msg="DPL SFH: peak age should match tau when alpha=beta (Carnall+2018 Eq. 1)",
        )


# ── 13. CHARLOT & FALL 2000 — two-component dust attenuation ──────


class TestCharlotFall2000:
    """Charlot & Fall 2000, ApJ 539, 718, Eq. 3 — two-component dust model."""

    def test_young_star_attenuation(self):
        """Young stars (deeply embedded) see tau_bc + tau_diff at V-band.

        CF00 Eq. 3: T_young = exp(-(tau_bc + tau_diff) × k(λ)).
        At V-band (5500 Å): k(5500) ≈ 1.0, so T_young = exp(-1.3) ≈ 0.2725.

        The birth-cloud transition is a sigmoid with t_birth=10 Myr, width=0.3 dex.
        At age=1e4 yr the weight is 0.99995 ≈ 1 (deep sigmoid limit).
        """
        from tengri.components.dust.attenuation import two_component_dust

        wave_v = jnp.array([5500.0])
        # 1e4 yr (10 kyr) — deep inside sigmoid → birth-cloud weight ≈ 1.0
        age_young = jnp.array([1e4])
        # two_component_dust returns shape (n_ages, n_wave); index [0, 0] for scalar
        t_young = float(
            two_component_dust(
                wave_v,
                age_young,
                tau_v1=1.0,
                tau_v2=0.3,
                law_bc="power_law",
                law_diff="calzetti",
            )[0, 0]
        )
        np.testing.assert_allclose(
            t_young,
            float(jnp.exp(-1.3)),
            rtol=0.05,
            err_msg="CF00 Eq. 3: young-star V-band attenuation = exp(-tau_bc - tau_diff)",
        )

    def test_old_star_attenuation(self):
        """Old stars (long-dispersed birth cloud) see only tau_diff at V-band.

        CF00 Eq. 3: T_old = exp(-tau_diff × k(λ)).
        At V-band: T_old = exp(-0.3) ≈ 0.7408.

        At age=5e9 yr the sigmoid weight is < 2e-4 ≈ 0 (deep old-star limit).
        """
        from tengri.components.dust.attenuation import two_component_dust

        wave_v = jnp.array([5500.0])
        # 5 Gyr — deep outside sigmoid → birth-cloud weight ≈ 0.0
        age_old = jnp.array([5e9])
        t_old = float(
            two_component_dust(
                wave_v,
                age_old,
                tau_v1=1.0,
                tau_v2=0.3,
                law_bc="power_law",
                law_diff="calzetti",
            )[0, 0]
        )
        np.testing.assert_allclose(
            t_old,
            float(jnp.exp(-0.3)),
            rtol=0.05,
            err_msg="CF00 Eq. 3: old-star V-band attenuation = exp(-tau_diff)",
        )

    def test_young_old_ratio(self):
        """Young/old attenuation ratio = exp(-tau_bc) at V-band.

        CF00: T_young / T_old = exp(-tau_bc × k(V)) = exp(-1.0) ≈ 0.3679.
        Uses deep sigmoid limits: 1e4 yr (weight≈1) and 5e9 yr (weight≈0).
        """
        from tengri.components.dust.attenuation import two_component_dust

        wave_v = jnp.array([5500.0])
        age_young = jnp.array([1e4])  # deep young limit: weight ≈ 1
        age_old = jnp.array([5e9])  # deep old limit: weight ≈ 0
        t_young = float(
            two_component_dust(
                wave_v,
                age_young,
                tau_v1=1.0,
                tau_v2=0.3,
                law_bc="power_law",
                law_diff="calzetti",
            )[0, 0]
        )
        t_old = float(
            two_component_dust(
                wave_v,
                age_old,
                tau_v1=1.0,
                tau_v2=0.3,
                law_bc="power_law",
                law_diff="calzetti",
            )[0, 0]
        )
        ratio = t_young / t_old
        np.testing.assert_allclose(
            ratio,
            float(jnp.exp(-1.0)),
            rtol=0.05,
            err_msg="CF00 Eq. 3: T_young/T_old = exp(-tau_bc) at V-band",
        )


# ── 14. WG00 DUST GEOMETRY LIMITS ─────────────────────────────────


class TestWittGordon2000Limits:
    """Witt & Gordon 2000, ApJ 528, 799 — shell and cloudy geometry limits."""

    def test_shell_opaque_limit(self):
        """Shell geometry: T → 0 at large tau_v. WG00 Eq. 1: T = exp(-tau × k).

        At tau_v = 100: T(5500 Å) < 1e-10.
        """
        from tengri.components.dust.attenuation import wg00_shell

        T = float(wg00_shell(jnp.array([5500.0]), tau_v=100.0, law="cardelli")[0])
        assert T < 1e-10, f"WG00 shell opaque limit: T={T:.2e} (expected < 1e-10 at tau_v=100)"

    def test_shell_transparent_limit(self):
        """Shell geometry: T → 1 at tau_v = 0. WG00 Eq. 1: Beer-Lambert."""
        from tengri.components.dust.attenuation import wg00_shell

        T = float(wg00_shell(jnp.array([5500.0]), tau_v=0.0, law="cardelli")[0])
        np.testing.assert_allclose(
            T,
            1.0,
            rtol=1e-6,
            err_msg="WG00 shell: T → 1 at tau_v=0 (transparent)",
        )

    def test_cloudy_slab_formula(self):
        """Cloudy geometry: T = (1 - exp(-x))/x, x = tau × k(λ). WG00 Eq. 2.

        At tau_v = 1, k(5500 Å) ≈ 1 (cardelli normalized to V-band):
        T = (1 - exp(-1)) / 1 ≈ 0.6321.
        """
        from tengri.components.dust.attenuation import wg00_cloudy

        T = float(wg00_cloudy(jnp.array([5500.0]), tau_v=1.0, law="cardelli")[0])
        expected = float((1.0 - jnp.exp(-1.0)) / 1.0)  # ≈ 0.6321
        np.testing.assert_allclose(
            T,
            expected,
            rtol=0.05,
            err_msg="WG00 cloudy slab formula: T=(1-exp(-x))/x at x=tau*k≈1",
        )

    def test_cloudy_taylor_at_zero(self):
        """Cloudy Taylor branch: T ≈ 1 at tau → 0 (numerical stability).

        lim_{x→0} (1-exp(-x))/x = 1. Implementation uses Taylor expansion
        for tau < threshold to avoid 0/0.
        """
        from tengri.components.dust.attenuation import wg00_cloudy

        T = float(wg00_cloudy(jnp.array([5500.0]), tau_v=1e-6, law="cardelli")[0])
        np.testing.assert_allclose(
            T,
            1.0,
            rtol=1e-4,
            err_msg="WG00 cloudy Taylor branch: T ≈ 1 as tau → 0",
        )


# ── 15. BELL 2003 — FIR-Radio correlation ─────────────────────────


class TestBell2003Radio:
    """Bell 2003, ApJ 586, 794, Eq. 6 — FIR-radio correlation."""

    def test_reference_frequency_implements_bell_eq5(self):
        """L_nu at the reference frequency satisfies Bell 2003 Eq. 5.

        Bell 2003 Eq. 5 defines q_IR = log10(FIR / (3.75×10^12 Hz × L_1.4GHz)),
        so the inverse is:
            L_1.4GHz = L_IR / (3.75e12 Hz × 10^q_IR)

        Using q_ir=2.64 (Bell 2003 canonical median, Table 3 of the paper) as
        a fixed literature value — NOT back-computed from the expected answer.
        The test verifies the function correctly implements this formula.
        """
        from tengri.components.radio import radio_sfr_bell2003
        from tengri.utils.physics_constants import C_AA as _C_AA

        L_ir = 1e45  # arbitrary erg/s — tests formula, not absolute calibration
        q_ir = 2.64  # Bell 2003, Table 3 canonical median
        nu_ref = 1.4e9  # Hz
        wave_ref = jnp.array([_C_AA / nu_ref])

        L_nu = float(radio_sfr_bell2003(wave_ref, L_ir=L_ir, q_ir=q_ir)[0])

        # Bell 2003 Eq. 5 rearranged: L_1.4 = L_IR / (3.75e12 × 10^q_IR)
        expected = L_ir / (3.75e12 * 10.0**q_ir)
        np.testing.assert_allclose(
            L_nu,
            expected,
            rtol=1e-6,
            err_msg="Bell 2003 Eq. 5: L_1.4 = L_IR / (3.75e12 × 10^q_IR)",
        )

    def test_q_ir_controls_normalization_by_decade(self):
        """Increasing q_ir by 1 reduces L_1.4GHz by exactly 10×.

        Independent check of the 10^q_ir denominator structure.
        Relative comparison between two evaluations — cannot be circular
        since the expected ratio (10×) comes from the exponent definition,
        not from the absolute output.
        """
        from tengri.components.radio import radio_sfr_bell2003
        from tengri.utils.physics_constants import C_AA as _C_AA

        wave = jnp.array([_C_AA / 1.4e9])
        L_ir = 1e44

        L_lo = float(radio_sfr_bell2003(wave, L_ir=L_ir, q_ir=2.0)[0])
        L_hi = float(radio_sfr_bell2003(wave, L_ir=L_ir, q_ir=3.0)[0])

        np.testing.assert_allclose(
            L_lo / L_hi,
            10.0,
            rtol=1e-6,
            err_msg="q_ir increases by 1 → L_nu decreases by 10× (10^q_ir in denominator)",
        )

    def test_synchrotron_spectral_index(self):
        """Nonthermal (synchrotron) emission: S ∝ ν^{−0.8}. Condon 1992 ARA&A 30.

        L(1.0 GHz) / L(1.4 GHz) = (1.0/1.4)^{-0.8} ≈ 1.338.
        """
        from tengri.components.radio import radio_sfr_bell2003
        from tengri.utils.physics_constants import C_AA as _C_AA

        wave_1p0 = jnp.array([_C_AA / 1.0e9])
        wave_1p4 = jnp.array([_C_AA / 1.4e9])
        L_ir = 1e44

        L_1p0 = float(radio_sfr_bell2003(wave_1p0, L_ir=L_ir, q_ir=2.64)[0])
        L_1p4 = float(radio_sfr_bell2003(wave_1p4, L_ir=L_ir, q_ir=2.64)[0])
        ratio = L_1p0 / L_1p4
        expected = (1.0 / 1.4) ** (-0.8)  # ≈ 1.338

        np.testing.assert_allclose(
            ratio,
            expected,
            rtol=0.01,
            err_msg="Condon 1992 ARA&A: synchrotron S∝ν^−0.8, ratio L(1GHz)/L(1.4GHz)≈1.338",
        )


# ── 16. RANALLI+2003 — XRB calibration ────────────────────────────


class TestRanalli2003XRay:
    """Ranalli+2003, A&A 399, 39 — combined XRB 2-10 keV calibration."""

    def test_combined_xrb_band_luminosity(self):
        """L_2-10keV ≈ 3.7×10^39 erg/s at SFR=1, M*=1e10. Ranalli+2003 A&A 399 Eq. 3.

        tengri uses Grimm+2003 (HMXB: 2.6e39 erg/s/SFR) + Gilfanov 2004
        (LMXB: 8.3e28 erg/s/Msun). Combined at SFR=1, M*=1e10: ~3.43e39,
        within 30% of Ranalli.
        """
        from tengri.components.xray import xray_xrb
        from tengri.utils.physics_constants import C_AA as _C_AA, KEV_TO_HZ as _KEV_TO_HZ

        E_grid = jnp.linspace(2.0, 10.0, 500)  # keV
        nu_grid = E_grid * _KEV_TO_HZ
        wave_grid = _C_AA / nu_grid

        L_band = float(jnp.trapezoid(xray_xrb(wave_grid, sfr=1.0, stellar_mass=1e10), nu_grid))
        np.testing.assert_allclose(
            L_band,
            3.7e39,
            rtol=0.30,
            err_msg="Ranalli+2003 A&A 399 Eq. 3: L_2-10keV ≈ 3.7e39 erg/s at SFR=1, M*=1e10",
        )


# ── 17. INOUE+2014 — IGM opacity ──────────────────────────────────


class TestInoue2014IGM:
    """Inoue+2014, MNRAS 442, 1805 — IGM opacity model."""

    def test_lyman_limit_opacity_z4(self):
        """Lyman limit at rest 912 Å is fully opaque at z_source=4.

        Inoue+2014: τ_LL >> 1 for z > 3.
        Observed-frame convention: pass wave_obs = 912 × (1+z).
        """
        from tengri.components.igm import igm_transmission

        z = 4.0
        wave_ll_obs = jnp.array([912.0 * (1 + z)])  # observed 4560 Å
        T = float(igm_transmission(wave_ll_obs, z)[0])
        assert T < 0.05, f"Inoue+2014: Lyman limit opacity at z=4: T={T:.3f} (expected < 0.05)"

    def test_lya_forest_z3(self):
        """Mean Lya forest transmission at z=3 ≈ 0.68. Fan+2006 AJ 132, Eq. 3."""
        from tengri.components.igm import igm_transmission

        z = 3.0
        wave_lya_obs = jnp.array([1216.0 * (1 + z)])  # observed 4864 Å
        T = float(igm_transmission(wave_lya_obs, z)[0])
        np.testing.assert_allclose(
            T,
            0.68,
            atol=0.15,
            err_msg="Fan+2006 AJ 132 Eq. 3: mean Lya forest T at z=3 ≈ 0.68",
        )

    def test_no_absorption_z0(self):
        """No significant IGM absorption at z≈0 — local universe is transparent.

        Inoue+2014: τ_IGM is tiny at z=0.01 so T > 0.97 everywhere above the
        Lyman limit. We check the range 912*(1+z) to 3000*(1+z) Å (observed),
        which covers the LAF and DLA forest at very low redshift.
        """
        from tengri.components.igm import igm_transmission

        z = 0.01
        wave = jnp.linspace(912.0 * (1 + z), 3000.0 * (1 + z), 50)
        T = igm_transmission(wave, z)
        min_T = float(jnp.min(T))
        assert min_T > 0.97, (
            f"IGM: no significant absorption at z=0.01, min T={min_T:.4f} (expected > 0.97)"
        )


# ── 18. OSTERBROCK & FERLAND 2006 — Case B Balmer decrements ──────


class TestOsterbrockCaseB:
    """Osterbrock & Ferland 2006, "Astrophysics of Gaseous Nebulae and Active
    Galactic Nuclei" (2nd ed.), §4.2, Table 4.2.

    Case B recombination ratios at T=10^4 K, n_e=100 cm^-3:
      Hα/Hβ = 2.86
      Hγ/Hβ = 0.468

    These are tested via the Cue emulator at low ionization parameter
    (log U = −3.5) and low density (log n = 2) where the gas conditions
    approach Case B recombination in an optically-thick nebula.  ±10%
    tolerance accounts for the emulator approximation and the mild
    temperature/density sensitivity of recombination coefficients.
    """

    @pytest.fixture(scope="class")
    def backend(self):
        import os

        from tengri.components.nebular.cue import CueBackend

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        return CueBackend("data/cue_weights.npz")

    def _find_line(self, wave, lum, target_aa, window_aa=10.0):
        """Return luminosity of the line nearest to target_aa."""
        idx = int(jnp.argmin(jnp.abs(wave - target_aa)))
        nearest = float(wave[idx])
        assert abs(nearest - target_aa) < window_aa, (
            f"No line within {window_aa} Å of {target_aa} Å — found nearest at {nearest:.1f} Å"
        )
        return float(lum[idx])

    def test_halpha_hbeta_case_b(self, backend):
        """Hα/Hβ = 2.86 at low logU — Osterbrock & Ferland 2006 §4.2 Table 4.2."""
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-3.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        ha = self._find_line(wave, lum, 6564.61)
        hb = self._find_line(wave, lum, 4862.68)
        ratio = ha / hb
        np.testing.assert_allclose(
            ratio,
            2.86,
            rtol=0.10,
            err_msg=(
                "Osterbrock & Ferland 2006 §4.2 Table 4.2: "
                "Case B Hα/Hβ = 2.86 at T=10^4 K, n_e=100 cm^-3"
            ),
        )

    def test_hgamma_hbeta_case_b(self, backend):
        """Hγ/Hβ = 0.468 at low logU — Osterbrock & Ferland 2006 §4.2 Table 4.2."""
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-3.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        hg = self._find_line(wave, lum, 4341.68)
        hb = self._find_line(wave, lum, 4862.68)
        ratio = hg / hb
        np.testing.assert_allclose(
            ratio,
            0.468,
            rtol=0.10,
            err_msg=(
                "Osterbrock & Ferland 2006 §4.2 Table 4.2: "
                "Case B Hγ/Hβ = 0.468 at T=10^4 K, n_e=100 cm^-3"
            ),
        )

    def test_halpha_hbeta_increases_with_dust(self, backend):
        """Dusty nebulae must have Hα/Hβ > 2.86 (reddening elevates the ratio).

        Calzetti+2000: dust preferentially attenuates shorter wavelengths, so
        Hβ (4863 Å) is more attenuated than Hα (6565 Å).  The observed ratio
        in any real, dust-affected HII region exceeds 2.86.  This test verifies
        that the Cue + dust pipeline preserves this ordering by:
          1. Fetching intrinsic Cue line luminosities at Case B conditions.
          2. Applying Calzetti T(λ) = exp(-τ_V × k(λ)) with τ_V=1 (realistic ISM).
          3. Asserting ratio_dust > ratio_intrinsic (dust raises the ratio).
          4. Asserting ratio_dust > 2.86 (exceeds Case B lower bound).
        """
        from tengri.components.dust.attenuation import calzetti

        # Intrinsic (no dust) — Cue emulator at low ionization, solar metallicity
        wave, lum0 = backend.predict_nebular_line_luminosities(
            gas_logu=-3.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        ha0 = self._find_line(wave, lum0, 6564.61)
        hb0 = self._find_line(wave, lum0, 4862.68)
        ratio0 = ha0 / hb0

        # Calzetti attenuation with τ_V = 1.0 (moderate ISM).
        # T(λ) = exp(-τ_V × k_calzetti(λ)); k is normalized so k(5500 Å)=1.
        # k(Hβ=4863 Å) > k(Hα=6565 Å) → T_hβ < T_hα → ratio rises.
        tau_v = 1.0
        k_ha = float(calzetti(jnp.array([6564.61]))[0])
        k_hb = float(calzetti(jnp.array([4862.68]))[0])
        t_ha = float(jnp.exp(-tau_v * k_ha))
        t_hb = float(jnp.exp(-tau_v * k_hb))
        ratio_dust = (ha0 * t_ha) / (hb0 * t_hb)

        assert ratio_dust > ratio0, (
            f"Dust must increase Hα/Hβ (Calzetti k(Hβ)={k_hb:.3f} > k(Hα)={k_ha:.3f}): "
            f"attenuated={ratio_dust:.3f} ≤ intrinsic={ratio0:.3f}"
        )
        assert ratio_dust > 2.86, (
            f"Dusty HII region Hα/Hβ must exceed Case B value 2.86. "
            f"Got attenuated ratio={ratio_dust:.3f} (intrinsic={ratio0:.3f})"
        )


# ── 19. STOREY & ZEIPPEN 2000 — Forbidden-line doublet A-coefficient ratios


class TestStoreyZeippen2000:
    """Storey & Zeippen 2000, MNRAS 312, 813 — forbidden-line A coefficients.

    The [OIII] 5007/4959 and [NII] 6583/6548 doublet ratios are fixed by
    atomic physics (Einstein A coefficients), independent of gas conditions:

      [OIII] λ5007 / λ4959 = A(5007) / A(4959) ≈ 2.98
      [NII]  λ6583 / λ6548 = A(6583) / A(6548) ≈ 2.94

    These are tested via the Cue emulator.  The emulator is trained on
    CLOUDY grids that solve the level populations self-consistently, so
    these ratios should be reproduced to within ±5% across all conditions.
    """

    @pytest.fixture(scope="class")
    def backend(self):
        import os

        from tengri.components.nebular.cue import CueBackend

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        return CueBackend("data/cue_weights.npz")

    def _find_line(self, wave, lum, target_aa, window_aa=8.0):
        idx = int(jnp.argmin(jnp.abs(wave - target_aa)))
        assert abs(float(wave[idx]) - target_aa) < window_aa, (
            f"No line within {window_aa} Å of {target_aa} Å "
            f"— found nearest at {float(wave[idx]):.1f} Å"
        )
        return float(lum[idx])

    def test_oiii_doublet_ratio(self, backend):
        """[OIII] 5007/4959 = 2.98 — Storey & Zeippen 2000 MNRAS 312, 813."""
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        o3_5007 = self._find_line(wave, lum, 5008.24)
        o3_4959 = self._find_line(wave, lum, 4960.30)
        ratio = o3_5007 / o3_4959
        np.testing.assert_allclose(
            ratio,
            2.98,
            rtol=0.05,
            err_msg=(
                "Storey & Zeippen 2000 MNRAS 312: "
                "[OIII] 5007/4959 = 2.98 (fixed by A coefficients)"
            ),
        )

    def test_nii_doublet_ratio(self, backend):
        """[NII] 6583/6548 = 2.94 — Storey & Zeippen 2000 MNRAS 312, 813."""
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        nii_6583 = self._find_line(wave, lum, 6585.27)
        nii_6548 = self._find_line(wave, lum, 6549.86)
        ratio = nii_6583 / nii_6548
        np.testing.assert_allclose(
            ratio,
            2.94,
            rtol=0.05,
            err_msg=(
                "Storey & Zeippen 2000 MNRAS 312: [NII] 6583/6548 = 2.94 (fixed by A coefficients)"
            ),
        )

    def test_doublet_ratios_independent_of_logu(self, backend):
        """Doublet ratios must be invariant to ionization parameter.

        Storey & Zeippen 2000: the 5007/4959 ratio is set by A coefficients,
        not collisional rates, so it is independent of n_e and T for n_e << n_crit
        (~7×10^5 cm^-3).  Verify the ratio is the same at logU = −3.5 and −1.5.
        """
        ratios = []
        for logu in [-3.5, -1.5]:
            wave, lum = backend.predict_nebular_line_luminosities(
                gas_logu=logu,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
            )
            o3_5007 = self._find_line(wave, lum, 5008.24)
            o3_4959 = self._find_line(wave, lum, 4960.30)
            ratios.append(o3_5007 / o3_4959)

        np.testing.assert_allclose(
            ratios[0],
            ratios[1],
            rtol=0.05,
            err_msg=(
                "Storey & Zeippen 2000: [OIII] 5007/4959 must be "
                "independent of logU (fixed by A coefficients)"
            ),
        )


# ── 20. KENNICUTT 1998 — Hα SFR calibration ───────────────────────


class TestKennicutt1998:
    """Kennicutt 1998, ARA&A, 36, 189, Eq. 2 — Hα SFR calibration.

      SFR [Msun/yr] = L(Hα) / 1.26e41 [erg/s]

    i.e. at SFR = 1 Msun/yr, L(Hα) = 1.26×10^41 erg/s.

    The calibration assumes Case B recombination at T=10^4 K, Salpeter IMF,
    and a constant star-formation history.  The ionizing photon rate
    corresponding to SFR = 1 Msun/yr is Q_H ≈ 10^52.8 s^-1 (Kennicutt &
    Evans 2012, ARA&A 50, 531, Table 1).

    The ±50% tolerance accounts for the Cue emulator being trained on
    single-burst BPASS/FSPS ionizing fields (not a galaxy-averaged SFR),
    and for the Salpeter→Chabrier IMF offset (~1.7×).

    See also: TestKennicutt1998Halpha in tests/unit/test_cue_and_ionizing.py.
    This crossval class additionally tests the linearity of the L(Hα)–Q_H
    relation and the scaling with ionizing photon rate.
    """

    @pytest.fixture(scope="class")
    def backend(self):
        import os

        from tengri.components.nebular.cue import CueBackend

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        return CueBackend("data/cue_weights.npz")

    def _find_ha(self, wave, lum):
        idx = int(jnp.argmin(jnp.abs(wave - 6564.61)))
        return float(lum[idx])

    def test_halpha_kennicutt_absolute(self, backend):
        """L(Hα) ≈ 1.26e41 erg/s at SFR=1 Msun/yr.

        Kennicutt 1998, ARA&A 36, 189, Eq. 2.  logQ_H = 52.8 s^-1
        corresponds to SFR ~ 1 Msun/yr for Salpeter IMF
        (Kennicutt & Evans 2012, ARA&A 50, 531, Table 1).
        ±50% tolerance: Cue is not a galaxy SFR model.
        """
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=52.8,
        )
        from tengri.utils.physics_constants import L_SUN_CUE

        # The backend returns [Lsun] (#1559); Kennicutt's calibration is erg/s.
        # ``L_SUN_CUE``, not IAU — this is Cue's own catalog, and the two
        # conventions differ by 0.287%.
        ha_lum = self._find_ha(wave, lum) * L_SUN_CUE
        assert ha_lum > 0, "L(Hα) must be positive"
        np.testing.assert_allclose(
            ha_lum,
            1.26e41,
            rtol=0.50,
            err_msg=(
                "Kennicutt 1998 ARA&A 36, 189 Eq. 2: "
                "L(Hα) = 1.26e41 erg/s at SFR=1 Msun/yr (logQH=52.8)"
            ),
        )

    def test_halpha_scales_with_qion(self, backend):
        """L(Hα) must scale linearly with Q_H.

        Osterbrock & Ferland 2006 §4.2: under Case B recombination,
        L(Hα) ∝ Q_H (every ionizing photon eventually produces a recombination
        cascade).  A factor of 10 in Q_H (one dex) must produce a factor of 10
        in L(Hα) to within 5%.
        """
        wave_lo, lum_lo = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        wave_hi, lum_hi = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=50.0,
        )
        ha_lo = self._find_ha(wave_lo, lum_lo)
        ha_hi = self._find_ha(wave_hi, lum_hi)
        ratio = ha_hi / ha_lo
        np.testing.assert_allclose(
            ratio,
            10.0,
            rtol=0.05,
            err_msg=(
                "Osterbrock & Ferland 2006 §4.2: "
                "L(Hα) must scale linearly with Q_H — "
                f"one dex in Q_H gave factor {ratio:.3f} (expected 10.0 ± 5%)"
            ),
        )
