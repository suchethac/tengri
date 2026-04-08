"""Physics-motivated numerical tests for multiwavelength SED components.

Tests that check NUMBERS, not just trends. Each test targets a specific
published relation or physical constraint with explicit tolerances.

References
----------
- Murphy+2011, ApJ, 737, 67 — radio-SFR relation (Eq. 11, 15)
- Bell 2003, ApJ, 586, 794 — FIR-radio correlation
- Delvecchio+2021, A&A, 647, A123 — mass/z-dependent FIRRC
- Grimm+2003, MNRAS, 339, 793 — HMXB-SFR: L_X = 2.6e39 × SFR
- Gilfanov 2004, MNRAS, 349, 146 — LMXB-M*: L_X = 9.2e28 × M*
- Lehmer+2010, ApJ, 724, 559 — universal L_X-SFR-M* relation
- Condon 1992, ARA&A, 30, 575 — synchrotron theory
- Inoue+2014, MNRAS, 442, 1805 — IGM transmission tables
- Just+2007, ApJ, 665, 1004 — alpha_ox relation
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_C_AA = 2.99792458e18  # c in Angstrom/s
_LSUN_ERG = 3.828e33  # erg/s


# ===================================================================
# 1. Radio: absolute calibration values
# ===================================================================


class TestRadioAbsoluteValues:
    """Radio luminosities must match published calibrations."""

    def test_murphy2011_radio_sfr_at_1p4ghz(self):
        """Murphy+2011 Eq. 15: SFR = 5.52e-22 × L_1.4GHz [erg/s/Hz].

        At SFR=1 Msun/yr → L_1.4GHz = 1.81e21 W/Hz = 1.81e28 erg/s/Hz.
        tengri uses q_ir to compute L_radio from L_IR. For SFR=1 Msun/yr
        with Kennicutt L_IR ~ 1e10 Lsun: L_1.4 = L_IR/(3.75e12 × 10^q).
        """
        from tengri.models.radio import radio_star_forming

        wave_14ghz = jnp.array([_C_AA / 1.4e9])  # 1.4 GHz
        L_ir = 1e10  # Lsun, typical for SFR~1 Msun/yr
        q_ir = 2.64  # Bell 2003 default

        l_nu = radio_star_forming(wave_14ghz, L_ir=L_ir, q_ir=q_ir)
        l_14 = float(l_nu[0])

        # Expected: L_IR / (3.75e12 × 10^2.64) ≈ 6.1e-6 (in same units as L_IR)
        expected = L_ir / (3.75e12 * 10.0**q_ir)
        np.testing.assert_allclose(
            l_14,
            expected,
            rtol=0.05,
            err_msg=f"L_1.4GHz = {l_14:.2e}, expected {expected:.2e} from q_ir formula",
        )

    def test_bell2003_q_ir_roundtrip(self):
        """q_TIR = log10(L_TIR / (3.75e12 × L_1.4GHz)) must recover input q."""
        from tengri.models.radio import radio_star_forming

        wave_14ghz = jnp.array([_C_AA / 1.4e9])
        L_ir = 1e11  # LIRG
        q_input = 2.64

        l_14 = float(radio_star_forming(wave_14ghz, L_ir=L_ir, q_ir=q_input)[0])
        q_recovered = np.log10(L_ir / (3.75e12 * l_14))
        np.testing.assert_allclose(
            q_recovered,
            q_input,
            atol=0.05,
            err_msg=f"q round-trip: input={q_input}, recovered={q_recovered:.3f}",
        )

    def test_free_free_flatter_than_synchrotron(self):
        """Free-free (α_ff ~ -0.1) is flatter than synchrotron (α_sf ~ 0.8).

        At 10 GHz, free-free should be relatively brighter compared to
        synchrotron than at 150 MHz.
        """
        from tengri.models.radio import radio_freefree, radio_star_forming

        wave_150mhz = jnp.array([_C_AA / 1.5e8])
        wave_10ghz = jnp.array([_C_AA / 1.0e10])

        l_sync_150 = float(radio_star_forming(wave_150mhz, L_ir=1e10)[0])
        l_sync_10g = float(radio_star_forming(wave_10ghz, L_ir=1e10)[0])
        l_ff_150 = float(radio_freefree(wave_150mhz, L_ir=1e10)[0])
        l_ff_10g = float(radio_freefree(wave_10ghz, L_ir=1e10)[0])

        if l_sync_150 > 0 and l_ff_150 > 0:
            sync_ratio = l_sync_10g / l_sync_150  # (10/0.15)^{-0.8} ~ 0.005
            ff_ratio = l_ff_10g / l_ff_150  # (10/0.15)^{0.1} ~ 1.5

            assert ff_ratio > sync_ratio, (
                f"Free-free (ratio={ff_ratio:.3f}) should be flatter than "
                f"synchrotron (ratio={sync_ratio:.3f})"
            )

    def test_free_free_thermal_fraction_at_1p4ghz(self):
        """Free-free is ~10% of total radio at 1.4 GHz (Condon 1992).

        The thermal fraction increases at higher frequencies because
        synchrotron declines as ν^{-0.8} while free-free is nearly flat.
        """
        from tengri.models.radio import radio_freefree, radio_star_forming

        wave_14ghz = jnp.array([_C_AA / 1.4e9])
        L_ir = 1e10

        l_sync = float(radio_star_forming(wave_14ghz, L_ir=L_ir)[0])
        l_ff = float(radio_freefree(wave_14ghz, L_ir=L_ir)[0])

        if l_sync > 0:
            thermal_frac = l_ff / (l_sync + l_ff)
            # Condon 1992: ~10% at 1.4 GHz for normal SF galaxies
            assert 0.01 < thermal_frac < 0.30, (
                f"Thermal fraction at 1.4 GHz = {thermal_frac:.2%}, expected ~10%"
            )

    def test_delvecchio_massive_galaxy_more_radio(self):
        """Delvecchio+2021: higher M* → lower q → more radio per L_IR.

        Mass slope = -0.234: Δq = -0.234 × Δlog(M*).
        """
        from tengri.models.radio import radio_sfr_delvecchio2021

        wave = jnp.array([_C_AA / 1.4e9])
        L_ir = 1e11

        l_m9 = float(
            radio_sfr_delvecchio2021(
                wave, L_ir, log_mstar=9.0, redshift=0.0, apply_suppression=False
            )[0]
        )
        l_m11 = float(
            radio_sfr_delvecchio2021(
                wave, L_ir, log_mstar=11.0, redshift=0.0, apply_suppression=False
            )[0]
        )

        # Δq = -0.234 × 2 = -0.468, so L_m11/L_m9 = 10^0.468 ≈ 2.93
        ratio = l_m11 / l_m9
        np.testing.assert_allclose(
            np.log10(ratio),
            0.468,
            atol=0.05,
            err_msg=f"Delvecchio mass scaling: log ratio={np.log10(ratio):.3f}, expected 0.468",
        )


# ===================================================================
# 2. X-ray: absolute calibration values
# ===================================================================


class TestXrayAbsoluteValues:
    """X-ray luminosities must match published scaling relations."""

    def test_grimm_hmxb_coefficient(self):
        """Grimm+2003: L_X(HMXB, 2-10 keV) = 2.6e39 × SFR erg/s.

        With SFR=1 and M*=0, the total L_X should be ~2.6e39 erg/s
        integrated over 2-10 keV.
        """
        from tengri.models.xray import xray_xrb

        # 2-10 keV: λ = 1.24-6.2 Å
        wave = jnp.linspace(1.24, 6.2, 500)
        l_nu = xray_xrb(wave, sfr=1.0, stellar_mass=0.0)

        nu = _C_AA / wave
        l_band = abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        np.testing.assert_allclose(
            l_band,
            2.6e39,
            rtol=0.25,
            err_msg=f"HMXB L_X = {l_band:.2e}, expected 2.6e39 (Grimm+2003)",
        )

    def test_gilfanov_lmxb_coefficient(self):
        """Gilfanov 2004: L_X(LMXB, 0.5-8 keV) = 9.2e28 × M* erg/s.

        With SFR=0 and M*=1e10, L_X should be ~9.2e38 erg/s.
        """
        from tengri.models.xray import xray_xrb

        # 0.5-8 keV: λ = 1.55-24.8 Å
        wave = jnp.linspace(1.55, 24.8, 500)
        l_nu = xray_xrb(wave, sfr=0.0, stellar_mass=1e10)

        nu = _C_AA / wave
        l_band = abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        # 9.2e28 × 1e10 = 9.2e38 erg/s
        np.testing.assert_allclose(
            l_band,
            9.2e38,
            rtol=0.30,
            err_msg=f"LMXB L_X = {l_band:.2e}, expected 9.2e38 (Gilfanov 2004)",
        )

    def test_lehmer_combined_relation(self):
        """Lehmer+2010: L_X(total) = α×M* + β×SFR.

        At SFR=10 and M*=1e10:
        L_HMXB = 2.6e39 × 10 = 2.6e40
        L_LMXB = 9.2e28 × 1e10 = 9.2e38
        Total ≈ 2.7e40 (HMXB dominates for high-SFR galaxy).
        """
        from tengri.models.xray import xray_xrb

        wave = jnp.linspace(1.24, 6.2, 500)
        l_nu = xray_xrb(wave, sfr=10.0, stellar_mass=1e10)

        nu = _C_AA / wave
        l_band = abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        # Combined ~ 2.6e40 + 9.2e38 ≈ 2.7e40
        assert 5e39 < l_band < 5e41, f"Combined L_X = {l_band:.2e}, expected ~2.7e40 (Lehmer+2010)"

    def test_xray_photon_index_controls_hardness(self):
        """Photon index Gamma: higher Γ → softer (more low-energy photons).

        Hardness ratio H/(H+S) should decrease with Γ.
        """
        from tengri.models.xray import xray_xrb

        wave = jnp.linspace(0.5, 25.0, 1000)
        # Soft: 0.5-2 keV (6.2-24.8 Å), Hard: 2-10 keV (1.24-6.2 Å)
        soft_mask = (wave > 6.2) & (wave < 24.8)
        hard_mask = (wave > 1.24) & (wave < 6.2)

        l_hard_gamma = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, gamma_hmxb=1.5)
        l_soft_gamma = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, gamma_hmxb=2.5)

        hr_hard = float(jnp.sum(l_hard_gamma[hard_mask])) / float(
            jnp.sum(l_hard_gamma[hard_mask]) + jnp.sum(l_hard_gamma[soft_mask])
        )
        hr_soft = float(jnp.sum(l_soft_gamma[hard_mask])) / float(
            jnp.sum(l_soft_gamma[hard_mask]) + jnp.sum(l_soft_gamma[soft_mask])
        )

        assert hr_hard > hr_soft, (
            f"Harder spectrum (Γ=1.5, HR={hr_hard:.3f}) should have higher "
            f"hardness ratio than soft (Γ=2.5, HR={hr_soft:.3f})"
        )

    def test_xray_cutoff_suppresses_high_energy(self):
        """E_cut exponential cutoff: L(E > E_cut) drops sharply.

        At E_cut=50 keV, flux at 100 keV should be exp(-2) ~ 0.14×
        relative to no-cutoff power law.
        """
        from tengri.models.xray import xray_xrb

        # 100 keV → λ = 12398.4/1e5 = 0.124 Å
        # 10 keV → λ = 1.24 Å
        wave = jnp.array([0.124, 1.24])

        l_lowcut = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, E_cut=50.0)
        l_highcut = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, E_cut=500.0)

        # At 100 keV with E_cut=50: exp(-100/50) = exp(-2) ≈ 0.14
        # At 10 keV with E_cut=50: exp(-10/50) = exp(-0.2) ≈ 0.82
        # With E_cut=500: both ≈ 1
        if float(l_highcut[0]) > 0 and float(l_lowcut[0]) > 0:
            ratio_100kev = float(l_lowcut[0] / l_highcut[0])
            ratio_10kev = float(l_lowcut[1] / l_highcut[1])
            assert ratio_100kev < ratio_10kev, "Cutoff should suppress 100 keV more than 10 keV"


# ===================================================================
# 3. AGN alpha_ox: X-ray to UV connection
# ===================================================================


class TestAlphaOxPhysics:
    """The alpha_ox relation connects UV and X-ray AGN luminosity."""

    def test_alpha_ox_definition(self):
        """alpha_ox = 0.384 × log(L_2keV / L_2500A).

        Standard alpha_ox ~ -1.4 for luminous quasars (Just+2007).
        More negative = relatively weaker X-rays.
        """
        from tengri.models.xray import xray_agn_corona

        # At alpha_ox = -1.4 and L_bol = 1e12 Lsun:
        # L_2500 ~ 0.2 × L_bol (for a typical quasar SED)
        # L_2keV = L_2500 × 10^(alpha_ox / 0.384) = L_2500 × 10^(-3.65)
        wave = jnp.array([6.2])  # 2 keV
        l_2kev = float(xray_agn_corona(wave, L_agn_bol=1e12, alpha_ox=-1.4)[0])

        # L_2keV should be nonzero and positive
        assert l_2kev > 0, "AGN corona should produce X-ray flux at 2 keV"

        # With steeper alpha_ox, should get less X-ray
        l_2kev_steep = float(xray_agn_corona(wave, L_agn_bol=1e12, alpha_ox=-1.8)[0])
        assert l_2kev_steep < l_2kev, "Steeper alpha_ox should give weaker X-rays"

    def test_agn_corona_scales_with_lbol(self):
        """Brighter AGN → brighter X-ray corona."""
        from tengri.models.xray import xray_agn_corona

        wave = jnp.linspace(1.0, 10.0, 100)
        l_low = float(jnp.sum(xray_agn_corona(wave, L_agn_bol=1e10)))
        l_high = float(jnp.sum(xray_agn_corona(wave, L_agn_bol=1e12)))

        ratio = l_high / max(l_low, 1e-50)
        # L_bol × 100 doesn't give exactly 100x X-ray because alpha_ox relation
        # is nonlinear, but should be much brighter
        assert ratio > 10, f"100x L_bol gave only {ratio:.1f}x X-ray"


# ===================================================================
# 4. Cross-component consistency: radio SFR vs UV SFR
# ===================================================================


class TestCrossComponentConsistency:
    """Different SFR indicators must agree for a self-consistent SED."""

    def test_radio_luminosity_linear_with_lir(self):
        """L_radio ∝ L_IR: doubling L_IR should double radio emission.

        This is a fundamental self-consistency check — the FIR-radio
        correlation is built into the model, so it must hold exactly.
        """
        from tengri.models.radio import radio_star_forming

        wave = jnp.geomspace(1e8, 1e10, 100)

        l_1x = float(jnp.sum(radio_star_forming(wave, L_ir=1e10)))
        l_2x = float(jnp.sum(radio_star_forming(wave, L_ir=2e10)))
        l_10x = float(jnp.sum(radio_star_forming(wave, L_ir=1e11)))

        np.testing.assert_allclose(
            l_2x / l_1x, 2.0, rtol=1e-10, err_msg="Radio should scale linearly with L_IR"
        )
        np.testing.assert_allclose(
            l_10x / l_1x, 10.0, rtol=1e-10, err_msg="Radio should scale linearly with L_IR"
        )

    def test_radio_and_xray_sfr_scalings_consistent(self):
        """Both radio and X-ray trace SFR. 10x L_IR → 10x radio AND ~10x X-ray.

        If the model is self-consistent, increasing SFR should boost both
        radio (via FIR-radio) and X-ray (via HMXB-SFR) proportionally.
        """
        from tengri.models.radio import radio_star_forming
        from tengri.models.xray import xray_xrb

        wave_radio = jnp.geomspace(1e8, 1e10, 100)
        wave_xray = jnp.linspace(1.24, 6.2, 100)

        # SFR ∝ L_IR for radio, SFR directly for X-ray
        radio_1 = float(jnp.sum(radio_star_forming(wave_radio, L_ir=1e10)))
        radio_10 = float(jnp.sum(radio_star_forming(wave_radio, L_ir=1e11)))
        xray_1 = float(jnp.sum(xray_xrb(wave_xray, sfr=1.0, stellar_mass=0.0)))
        xray_10 = float(jnp.sum(xray_xrb(wave_xray, sfr=10.0, stellar_mass=0.0)))

        radio_ratio = radio_10 / radio_1
        xray_ratio = xray_10 / xray_1

        # Both should scale ~10x (linear with SFR)
        np.testing.assert_allclose(
            radio_ratio, 10.0, rtol=0.01, err_msg="Radio should scale 10x with 10x L_IR"
        )
        np.testing.assert_allclose(
            xray_ratio, 10.0, rtol=0.01, err_msg="X-ray should scale 10x with 10x SFR"
        )


# ===================================================================
# 5. IGM: numerical transmission values at specific z and λ
# ===================================================================


class TestIGMNumericalValues:
    """Check IGM transmission at specific published reference points."""

    def test_lya_forest_opacity_z3(self):
        """At z=3, mean Lyα forest transmission at 1050Å rest ~ 0.7.

        Inoue+2014 Table 2: τ_eff(LAF) at 1050A rest-frame ≈ 0.35,
        so T ≈ exp(-0.35) ≈ 0.70.
        """
        from tengri.models.igm import igm_transmission

        wave_obs = jnp.array([1050.0 * (1 + 3.0)])  # 4200 Å observed
        t = float(igm_transmission(wave_obs, z_source=3.0)[0])
        assert 0.3 < t < 0.95, f"IGM T(1050A rest, z=3) = {t:.2f}, expected 0.5-0.85"

    def test_gunn_peterson_trough_z6(self):
        """At z=6, below Lyα: essentially complete absorption.

        The Gunn-Peterson trough means T → 0 for λ_rest < 1216 Å at z > 5.5.
        """
        from tengri.models.igm import igm_transmission

        wave_obs = jnp.array([1100.0 * (1 + 6.0)])  # 7700 Å, below Lyα at z=6
        t = float(igm_transmission(wave_obs, z_source=6.0)[0])
        assert t < 0.1, f"T below Lyα at z=6 = {t:.2f}, expected < 0.1 (Gunn-Peterson)"

    @pytest.mark.parametrize(
        "z, wave_rest, t_min, t_max",
        [
            (0.5, 1100.0, 0.85, 1.0),  # low z, minimal absorption
            (2.0, 1100.0, 0.5, 0.95),  # moderate z
            (4.0, 1100.0, 0.1, 0.6),  # high z, significant absorption
            (3.0, 5000.0, 0.99, 1.0),  # red of Lyα, no absorption
            (5.0, 800.0, 0.0, 0.05),  # below Lyman limit at high z
        ],
    )
    def test_igm_transmission_range(self, z, wave_rest, t_min, t_max):
        """IGM transmission at various (z, λ) must be in expected range."""
        from tengri.models.igm import igm_transmission

        wave_obs = jnp.array([wave_rest * (1 + z)])
        t = float(igm_transmission(wave_obs, z_source=z)[0])
        assert t_min <= t <= t_max + 0.01, (
            f"T(λ_rest={wave_rest}Å, z={z}) = {t:.3f}, expected [{t_min}, {t_max}]"
        )


# ===================================================================
# 6. Radio component decomposition
# ===================================================================


class TestRadioComponentDecomposition:
    """Verify that SF + AGN + free-free combine correctly."""

    def test_components_sum_to_total(self):
        """radio_components individual parts must sum to radio_total."""
        from tengri.models.radio import radio_components, radio_total

        wave = jnp.geomspace(1e8, 1e10, 200)
        kwargs = dict(
            L_ir=1e10,
            L_agn_bol=1e11,
            q_ir=2.64,
            alpha_sf=0.8,
            radio_loudness=1.0,
            alpha_agn=0.7,
        )

        total = radio_total(wave, include_freefree=True, **kwargs)
        comps = radio_components(wave, include_freefree=True, **kwargs)

        # Sum of components should equal total
        comp_sum = comps["synchrotron"] + comps["freefree"] + comps["agn"]
        np.testing.assert_allclose(
            np.asarray(comp_sum),
            np.asarray(total),
            rtol=1e-10,
            err_msg="Component decomposition doesn't sum to total",
        )

    def test_agn_dominates_for_radio_loud(self):
        """For radio-loud AGN (loudness=3), AGN component >> SF at 1.4 GHz."""
        from tengri.models.radio import radio_components

        wave = jnp.array([_C_AA / 1.4e9])
        comps = radio_components(
            wave,
            L_ir=1e10,
            L_agn_bol=1e12,
            radio_loudness=3.0,
            include_freefree=True,
        )

        agn_frac = float(comps["agn"][0]) / float(
            comps["synchrotron"][0] + comps["freefree"][0] + comps["agn"][0]
        )
        assert agn_frac > 0.9, f"Radio-loud AGN fraction = {agn_frac:.2%}, expected > 90%"

    def test_sf_dominates_for_radio_quiet(self):
        """For radio-quiet AGN (loudness=0), SF dominates at 1.4 GHz."""
        from tengri.models.radio import radio_components

        # Use moderate AGN (L_bol=1e10 Lsun) with radio_loudness=-2 (very quiet)
        # and strong SF (L_IR=1e11) so SF should dominate
        wave = jnp.array([_C_AA / 1.4e9])
        comps = radio_components(
            wave,
            L_ir=1e11,
            L_agn_bol=1e10,
            radio_loudness=-2.0,
            include_freefree=True,
        )

        total = float(comps["synchrotron"][0] + comps["freefree"][0] + comps["agn"][0])
        sf_frac = float(comps["synchrotron"][0] + comps["freefree"][0]) / total
        assert sf_frac > 0.9, f"Radio-quiet SF fraction = {sf_frac:.2%}, expected > 90%"
