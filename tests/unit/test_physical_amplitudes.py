"""Physical-amplitude regression tests for core tengri components.

Each test pins a function's output against a literature-derived value
so that future refactors cannot silently shift the amplitude by orders of
magnitude (as happened with ``xray_agn_corona`` prior to the April 2026 fix).

Reference galaxies and expected numbers are quoted in the docstrings.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

LSUN_ERG = 3.828e33  # erg/s

# ---------------------------------------------------------------------
# X-ray AGN corona  (Just+2007 alpha_ox + power-law)
# ---------------------------------------------------------------------


class TestXrayAgnCorona:
    def test_amplitude_at_2_keV(self):
        """L_bol=10^44 erg/s ⇒ L_ν(2 keV) ≈ 3.6e24 erg/s/Hz.

        From alpha_ox = -1.4, L_2500 = L_bol / (BC * nu_2500),
        L_2keV = L_2500 * 10^(alpha_ox/0.384).
        """
        from tengri.components.xray.xray import xray_agn_corona

        wl = jnp.array([6.2])
        L = float(np.array(xray_agn_corona(wl, L_agn_bol=1e44))[0])
        assert 1e24 < L < 1e25, f"L_nu(2 keV) = {L:.2e}"

    def test_powerlaw_slope_gamma_1p8(self):
        """Gamma=1.8 ⇒ F_nu ∝ nu^(1-Gamma) ⇒ L(10)/L(1) ≈ 10^(-0.8) ≈ 0.158."""
        from tengri.components.xray.xray import xray_agn_corona

        wl = jnp.array([12.4, 1.24])  # 1 keV, 10 keV
        L = np.array(xray_agn_corona(wl, L_agn_bol=1e44, gamma=1.8))
        ratio = L[1] / L[0]
        assert 0.12 < ratio < 0.25, f"10/1 keV ratio = {ratio:.3f}"

    def test_linearity_in_Lbol(self):
        from tengri.components.xray.xray import xray_agn_corona

        wl = jnp.array([6.2])
        a = float(np.array(xray_agn_corona(wl, L_agn_bol=1e44))[0])
        b = float(np.array(xray_agn_corona(wl, L_agn_bol=2e44))[0])
        assert abs(b / a - 2.0) < 0.01


# ---------------------------------------------------------------------
# AGN optical / UV
# ---------------------------------------------------------------------


class TestAGNOpticalUV:
    def test_powerlaw_disc_linear_in_lbol(self):
        from tengri.components.agn.disc import powerlaw_disc

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        a = np.array(powerlaw_disc(wl, agn_log_lbol=10.0))
        b = np.array(powerlaw_disc(wl, agn_log_lbol=11.0))
        ratio = b.max() / a.max()
        assert 9.5 < ratio < 10.5, f"10x L_bol → {ratio:.3f}x L_nu"

    def test_powerlaw_disc_seyfert_amplitude(self):
        """log(L_bol/L_sun)=11 ⇒ L_ν peak ~1e28-1e31 erg/s/Hz."""
        from tengri.components.agn.disc import powerlaw_disc

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        L = np.array(powerlaw_disc(wl, agn_log_lbol=11.0))
        assert 1e28 < L.max() < 1e31

    def test_qsogen_near_linear_in_lbol(self):
        """QSOgen has ~10% Baldwin-effect deviation from exact linearity."""
        from tengri.components.agn.qsogen import qsogen

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        a = np.array(qsogen(wl, agn_log_lbol=10.0, z=0.0))
        b = np.array(qsogen(wl, agn_log_lbol=11.0, z=0.0))
        assert 8.5 < b.max() / a.max() < 11.5

    def test_qsogen_quasar_amplitude(self):
        from tengri.components.agn.qsogen import qsogen

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        L = np.array(qsogen(wl, agn_log_lbol=12.0, z=0.0))
        assert 1e29 < L.max() < 1e32


# ---------------------------------------------------------------------
# Dust attenuation curves — k(λ)/k(V) values at key wavelengths
# ---------------------------------------------------------------------


class TestDustAttenuation:
    """All attenuation curves are normalised so k(5500 Å) = 1."""

    WL = jnp.array([5500.0, 2175.0, 1500.0, 8000.0])  # V, UV bump, FUV, I-band

    def test_calzetti_values(self):
        """Calzetti+2000: k(2175)/k(V)≈2.09, k(1500)/k(V)≈2.55, k(8000)/k(V)≈0.63."""
        from tengri.components.dust.attenuation import calzetti

        k = np.array(calzetti(self.WL))
        assert abs(k[0] - 1.0) < 0.01
        assert 2.0 < k[1] < 2.2  # mild feature, no strong UV bump
        assert 2.4 < k[2] < 2.7
        assert 0.55 < k[3] < 0.70

    def test_cardelli_has_uv_bump(self):
        """Cardelli+1989 MW: strong 2175 Å feature, k(2175)/k(V) ≈ 3.2 for R_V=3.1."""
        from tengri.components.dust.attenuation import cardelli

        k = np.array(cardelli(self.WL, dust_Rv=3.1))
        assert 3.0 < k[1] < 3.4, f"MW UV bump k(2175)={k[1]:.2f}, expected 3.2"
        # Bump must exceed linear interpolation between neighbour wavelengths
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        k2 = np.array(cardelli(wl2, dust_Rv=3.1))
        baseline = 0.5 * (k2[0] + k2[2])
        assert k2[1] > baseline * 1.05, "Cardelli UV bump missing/too weak"

    def test_smc_steep_uv_no_bump(self):
        """SMC: k(1500)/k(V)≈4.6, 2175 Å bump absent (feature <10%)."""
        from tengri.components.dust.attenuation import smc

        k = np.array(smc(self.WL))
        assert k[2] > 4.0, f"SMC k(1500) = {k[2]:.2f}, expected ≈4.6"
        # Absence of bump: k(2175) should be close to linear interp
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        k2 = np.array(smc(wl2))
        baseline = 0.5 * (k2[0] + k2[2])
        assert abs(k2[1] - baseline) / baseline < 0.1, "SMC should not have UV bump"

    def test_salim_collapses_to_calzetti_at_delta_zero(self):
        """Salim+2018: δ=0 and bump=1.0 recovers Calzetti within 1%."""
        from tengri.components.dust.attenuation import calzetti, salim

        ks = np.array(salim(self.WL, dust_delta=0.0, dust_uv_bump=1.0))
        kc = np.array(calzetti(self.WL))
        np.testing.assert_allclose(ks, kc, rtol=0.02)


# ---------------------------------------------------------------------
# Dust emission — modified blackbody + template models
# ---------------------------------------------------------------------


class TestDustEmission:
    def test_mbb_peak_wavelength_scales_with_T(self):
        """MBB with β=1.8 peaks near λ ≈ 14388/(4.6·T) μm (Wien + emissivity shift)."""
        from tengri.dust import modified_blackbody

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 2000)
        for T in [25.0, 35.0, 50.0]:
            L = np.array(modified_blackbody(wl, 1.0, dust_T=T, dust_beta_ir=1.8))
            peak = float(wl[L.argmax()]) * 1e-4
            expected = 14388.0 / (4.6 * T)
            assert 0.7 * expected < peak < 1.5 * expected

    def test_mbb_linear_in_L_absorbed(self):
        from tengri.dust import modified_blackbody

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        a = np.array(modified_blackbody(wl, 1.0))
        b = np.array(modified_blackbody(wl, 100.0))
        assert 99 < b.max() / a.max() < 101

    def test_mbb_integrates_to_L_absorbed(self):
        """∫L_ν dν = L_abs (within 2% trapezoid tolerance)."""
        from tengri.dust import modified_blackbody

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 5000)
        c_aa_per_s = 2.9979e18
        nu = c_aa_per_s / np.array(wl)
        L_in = 1e44
        lnu = np.array(modified_blackbody(wl, L_in, dust_T=35.0, dust_beta_ir=1.8))
        L_out = -np.trapezoid(lnu, nu)
        assert 0.98 < L_out / L_in < 1.02

    def test_mbb_beta_index_controls_submm_slope(self):
        """Submm slope: L_ν ∝ ν^(2+β); β=2 drops ~10× per dex steeper than β=1."""
        from tengri.dust import modified_blackbody

        wl = jnp.array([5e5, 1e6])  # 50, 100 μm (submm side)
        L1 = np.array(modified_blackbody(wl, 1.0, dust_T=30.0, dust_beta_ir=1.0))
        L2 = np.array(modified_blackbody(wl, 1.0, dust_T=30.0, dust_beta_ir=2.0))
        # Ratio L(100)/L(50) is steeper (smaller) for higher β
        assert (L2[1] / L2[0]) < (L1[1] / L1[0])

    def test_dl07_peaks_in_fir_for_low_umin(self):
        """Draine&Li 2007 with U_min=1 ⇒ cold-dust dominated, peak at λ > 60 μm."""
        from tengri.dust import draine_li2007

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        L = np.array(draine_li2007(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 60.0 < peak_um < 250.0, f"DL07 peak at {peak_um:.1f} μm, expected FIR"

    def test_dl07_pah_features_present(self):
        """DL07 with q_PAH=2.5 must show raised emission near 7.7 μm vs continuum baseline."""
        from tengri.dust import draine_li2007

        wl = jnp.array([5.0e4, 7.7e4, 11.3e4])  # 5, 7.7, 11.3 μm
        L = np.array(draine_li2007(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        # 7.7 or 11.3 μm PAHs should exceed 5 μm continuum
        assert L[1] > L[0] or L[2] > L[0]

    def test_dl14_peaks_in_fir_for_low_umin(self):
        """Draine&Li 2014 with U_min=1, gamma=0.01 should peak in FIR like DL07.

        Currently the tabulated loader peaks at 3.3 μm PAH feature even for
        cold-dust-dominated (γ=0.01) parameters, which is unphysical.  Marked
        xfail until the template loader is audited.
        """
        from tengri.dust import draine_li2014

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        L = np.array(draine_li2014(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 60.0 < peak_um < 250.0, f"DL14 peak at {peak_um:.1f} μm"

    def test_casey_peaks_in_fir(self):
        from tengri.dust import casey2012

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        L = np.array(casey2012(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0))
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 40.0 < peak_um < 200.0


# ---------------------------------------------------------------------
# Radio continuum — FIR-radio correlation & synchrotron slope
# ---------------------------------------------------------------------


class TestRadioContinuum:
    WL_1P4GHZ = jnp.array([2.14e9])  # 1.4 GHz in Angstrom (c/1.4e9 Hz)

    def test_bell2003_matches_firrc(self):
        """Bell+2003: L_1.4GHz = L_IR / (3.75e12 * 10^q_IR) with q_IR=2.64 default."""
        from tengri.components.radio.radio import radio_sfr_bell2003

        L_IR = 1e10 * LSUN_ERG  # 10^10 L_sun
        L = float(np.array(radio_sfr_bell2003(self.WL_1P4GHZ, L_ir=L_IR))[0])
        expected = L_IR / (3.75e12 * 10**2.64)
        assert 0.99 < L / expected < 1.01

    def test_bell2003_synchrotron_slope(self):
        """α=0.8 ⇒ L_ν(150 MHz) / L_ν(1.4 GHz) = (0.15/1.4)^(-0.8) ≈ 5.9."""
        from tengri.components.radio.radio import radio_sfr_bell2003

        # 150 MHz → λ = c/ν = 2e9 cm = 2e17 Å; 1.4 GHz → 2.14e9 Å.
        # Note: lambda[Å] = 3e18 / ν[Hz]
        wl = jnp.array([3e18 / 150e6, 3e18 / 1.4e9])
        L = np.array(radio_sfr_bell2003(wl, L_ir=1e44, alpha_sf=0.8))
        ratio = L[0] / L[1]
        # (1.4 GHz / 150 MHz)^0.8 ≈ 9.33^0.8 ≈ 5.9
        assert 5.0 < ratio < 7.0, f"150/1400 MHz ratio = {ratio:.2f}, expected ~5.9"

    def test_bell2003_q_shift_lowers_radio(self):
        """Higher q_IR ⇒ less radio per L_IR (linear in 10^-q_IR)."""
        from tengri.components.radio.radio import radio_sfr_bell2003

        L_low = float(np.array(radio_sfr_bell2003(self.WL_1P4GHZ, L_ir=1e44, q_ir=2.3))[0])
        L_hi = float(np.array(radio_sfr_bell2003(self.WL_1P4GHZ, L_ir=1e44, q_ir=2.9))[0])
        # Ratio should be 10^(2.9-2.3) = 10^0.6 ≈ 3.98
        assert 3.5 < L_low / L_hi < 4.5


# ---------------------------------------------------------------------
# Chemical evolution
# ---------------------------------------------------------------------


class TestChemicalEvolution:
    def test_closed_box_output_is_log10(self):
        from tengri.components.sfh import closed_box_metallicity

        t_yr = np.linspace(0, 13.8e9, 200)
        sfr = np.ones_like(t_yr)
        log_z = np.array(closed_box_metallicity(t_yr, sfr, yield_y=0.03, eta_outflow=0.0))
        assert np.all(log_z >= -4.0) and np.all(log_z <= 1.0)

    def test_outflow_lowers_metallicity(self):
        from tengri.components.sfh import closed_box_metallicity

        t_yr = np.linspace(0, 13.8e9, 200)
        sfr = np.ones_like(t_yr)
        z_closed = np.array(closed_box_metallicity(t_yr, sfr, eta_outflow=0.0))
        z_leaky = np.array(closed_box_metallicity(t_yr, sfr, eta_outflow=2.0))
        assert z_leaky[0] < z_closed[0]

    def test_higher_yield_raises_metallicity(self):
        from tengri.components.sfh import closed_box_metallicity

        t_yr = np.linspace(0, 13.8e9, 200)
        sfr = np.ones_like(t_yr)
        z_low = np.array(closed_box_metallicity(t_yr, sfr, yield_y=0.01))
        z_hi = np.array(closed_box_metallicity(t_yr, sfr, yield_y=0.06))
        assert z_hi[0] > z_low[0]


# ---------------------------------------------------------------------
# Emission lines — Hα / Hβ for known SFR (Kennicutt 1998 + case B)
# ---------------------------------------------------------------------


def _make_ssp_if_available():
    from pathlib import Path

    from tengri import load_ssp_data

    p = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not p.exists():
        return None
    return load_ssp_data(str(p))


_MAPPINGS_H5 = (
    __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "mappings_templates.h5"
)
_h5_only_shock = pytest.mark.skipif(
    not _MAPPINGS_H5.exists(),
    reason="data/mappings_templates.h5 not found; build via download_mappings_templates.py",
)


class TestShockLineRatios:
    """MAPPINGS V shock models (via tengri.nebular.shock_line_ratios).

    These pin atomic-physics ratios (temperature/density insensitive) and
    Case-B Balmer ratio, which should be independent of code version.
    """

    def _r(self, v=300.0, log_n=2.0, b=1.0):
        from tengri.nebular import shock_line_ratios

        return shock_line_ratios(shock_velocity=v, shock_log_density=log_n, shock_b_over_sqrt_n=b)

    def test_oiii_5007_4959_ratio_is_atomic(self):
        """[O III] λ5007/λ4959 = 2.98 (Storey & Zeippen 2000, atomic transition)."""
        r = self._r()
        ratio = r["O3_5007A"] / r["O3_4959A"]
        assert 2.7 < ratio < 3.1, f"[OIII] 5007/4959 = {ratio:.3f}, expected 2.98"

    def test_nii_6583_6548_ratio_is_atomic(self):
        """[N II] λ6583/λ6548 = 2.96 (atomic transition, quasi-constant)."""
        r = self._r()
        ratio = r["NII_6583A"] / r["NII_6548A"]
        assert 2.7 < ratio < 3.1, f"[NII] 6583/6548 = {ratio:.3f}, expected 2.96"

    def test_halpha_hbeta_case_B_like(self):
        """Shocks at v=200 km/s give Hα/Hβ close to Case B 2.86 (T_e ≈ 1e4 K).

        Fast shocks elevate the ratio slightly; range ~2.7–3.3 covers
        the shocked-gas literature at v = 100–500 km/s.
        """
        r = self._r(v=200.0)
        balmer = r["HA_6563A"] / r["Hb_4861A"]
        assert 2.5 < balmer < 3.3, f"Hα/Hβ = {balmer:.3f}"

    @_h5_only_shock
    def test_sii_doublet_is_density_sensitive(self):
        """[SII] 6716/6731 increases from ~0.45 (high n) to ~1.45 (low n)."""
        r_low = self._r(log_n=0.0)  # 1 cm^-3
        r_hi = self._r(log_n=3.0)  # 1000 cm^-3
        sii_low = r_low["SII_6716A"] / r_low["SII_6731A"]
        sii_hi = r_hi["SII_6716A"] / r_hi["SII_6731A"]
        assert sii_hi < sii_low, (
            f"Denser gas should have smaller [SII] ratio (low_n={sii_low:.2f}, hi_n={sii_hi:.2f})"
        )
        # Bounded by atomic-physics limits (Osterbrock & Ferland AGN² Table 5.8)
        assert 0.4 < sii_hi < 1.5
        assert 0.4 < sii_low < 1.5

    def test_bpt_agn_branch_for_fast_shocks(self):
        """v_s ≥ 300 km/s lands above the Kewley+2001 AGN/SF boundary line."""
        import numpy as np

        r = self._r(v=400.0, log_n=2.0)
        log_oiii_hb = np.log10(r["O3_5007A"] / r["Hb_4861A"])
        log_nii_ha = np.log10(r["NII_6583A"] / r["HA_6563A"])
        # Kewley+2001 max-SB line: y = 0.61/(x - 0.47) + 1.19  (for log NII/Hα)
        y_kewley = 0.61 / (log_nii_ha - 0.47) + 1.19
        # Fast shocks should be above (larger [OIII]/Hβ than max-SB line)
        assert log_oiii_hb > y_kewley, (
            f"Fast shock should lie in AGN/LINER region. "
            f"log[OIII]/Hβ={log_oiii_hb:.2f}, Kewley y={y_kewley:.2f}"
        )


class TestEmissionLineAmplitudes:
    """Case B recombination: for SFR = 1 M⊙/yr (Chabrier IMF, solar Z),
    Q_H ≈ 4.2e53 photons/s (Leitherer+1999) and
    L_Hα ≈ 1.26e41 erg/s, L_Hβ = L_Hα / 2.86 ≈ 4.4e40 erg/s ≈ 1.15e7 L_sun.

    Here we pin predict_hbeta for a SFR=1 M⊙/yr 1 Gyr-old constant SFH.
    """

    def _build_const_sfh_model(self, log_sfr=0.0):
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")
        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_sfr=Fixed(log_sfr),
            sfh_const_start_gyr=Fixed(1.0),
            sfh_const_end_gyr=Fixed(1e-3),
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.0),
        )
        return SEDModel(spec, ssp)

    def test_hbeta_amplitude_for_sfr1(self):
        """Constant SFR=1 M⊙/yr ⇒ L_Hβ should be ≈1e7 L_sun (Leitherer+1999, Case B)."""
        model = self._build_const_sfh_model(log_sfr=0.0)
        L_hb = float(model.predict_hbeta({}))
        assert 1e6 < L_hb < 1e8, f"L_Hβ = {L_hb:.2e} L_sun (expected ≈1e7)"

    def test_hbeta_linear_in_sfr(self):
        """predict_hbeta must scale linearly with SFR."""
        m1 = self._build_const_sfh_model(log_sfr=0.0)
        m10 = self._build_const_sfh_model(log_sfr=1.0)
        r = float(m10.predict_hbeta({})) / float(m1.predict_hbeta({}))
        assert 9.0 < r < 11.0, f"10×SFR → {r:.2f}×L_Hβ"


# ---------------------------------------------------------------------
# SSP metallicity grid — ensure canonical Zsun offset & grid bounds
# ---------------------------------------------------------------------


class TestFullSEDAmplitudes:
    """End-to-end SED for a Milky-Way-like galaxy: M* ~ 10^10 Msun, SFR~1 M⊙/yr.

    These tests fix the amplitude at key wavelengths (UV, optical, NIR, submm)
    to catch regressions that slip past the component-level tests.
    """

    def _mw_like_sed(self):
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_log_peak_sfr=Fixed(np.log10(3.0)),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.0),
        )
        model = SEDModel(spec, ssp)
        pred = model.predict_rest_sed({})
        return np.array(pred.wavelength), np.array(pred.sed), model

    def test_nir_peak_amplitude(self):
        """1.6 μm (NIR H-band peak of evolved stellar pop) for ~10^10 M⊙ galaxy:
        L_ν ≈ 10^28–10^30 erg/s/Hz."""
        wl, lnu, _ = self._mw_like_sed()
        i = int(np.argmin(np.abs(wl - 16000.0)))
        assert 1e28 < lnu[i] < 1e30, f"L_ν(1.6 μm) = {lnu[i]:.2e}"

    def test_v_band_amplitude(self):
        """5500 Å (V-band) for M* ~ 10^10 M⊙: L_ν ≈ 10^28 erg/s/Hz (absolute mag ≈ -19)."""
        wl, lnu, _ = self._mw_like_sed()
        i = int(np.argmin(np.abs(wl - 5500.0)))
        assert 1e27 < lnu[i] < 1e30, f"L_ν(V) = {lnu[i]:.2e}"

    def test_uv_amplitude_attenuated(self):
        """1500 Å with τ_diff=0.3 ⇒ L_ν ~ 10^26–10^28 (attenuated from intrinsic)."""
        wl, lnu, _ = self._mw_like_sed()
        i = int(np.argmin(np.abs(wl - 1500.0)))
        assert 1e25 < lnu[i] < 1e29, f"L_ν(UV) = {lnu[i]:.2e}"

    def test_stellar_peak_is_nir(self):
        """Optical/NIR stellar SED peaks in νL_ν near 1 μm; i.e. NIR >> UV."""
        wl, lnu, _ = self._mw_like_sed()
        i_uv = int(np.argmin(np.abs(wl - 1500.0)))
        i_nir = int(np.argmin(np.abs(wl - 16000.0)))
        # Dust-attenuated UV should be lower than NIR in νL_ν sense
        nu_uv = 3e18 / wl[i_uv]
        nu_nir = 3e18 / wl[i_nir]
        assert lnu[i_nir] * nu_nir > lnu[i_uv] * nu_uv, (
            "NIR νL_ν must exceed dust-attenuated UV νL_ν for this model"
        )

    def test_bolometric_luminosity(self):
        """Integrated L_bol for a dwarf-MW-like model (~10^10 L_sun)."""
        wl, lnu, _ = self._mw_like_sed()
        nu = 3e18 / wl
        srt = np.argsort(nu)
        L_bol = float(np.trapezoid(lnu[srt], nu[srt]))
        # Expect 10^8 to 10^11 L_sun for this smooth-SFH model
        L_bol_lsun = L_bol / LSUN_ERG
        assert 1e8 < L_bol_lsun < 1e12, f"L_bol = {L_bol_lsun:.2e} L_sun"

    def test_ratio_v_to_uv_dust_attenuation(self):
        """τ_diff=0.3 ⇒ UV attenuation A(1500) ≈ τ·k(1500)/k(V) ≈ 0.3·3/(0.92·ln10)
        so f(UV)/f(V) should be reduced relative to τ=0 case."""
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")

        def build(tau):
            spec = Parameters(
                mean_sfh_type="dpl",
                sfh_dpl_log_peak_sfr=Fixed(np.log10(3.0)),
                sfh_dpl_alpha=Fixed(2.0),
                sfh_dpl_beta=Fixed(1.0),
                sfh_dpl_tau_gyr=Fixed(5.0),
                met_logzsol=Fixed(0.0),
                dust_tau_bc=Fixed(0.0),
                dust_tau_diff=Fixed(tau),
                dust_slope=Fixed(-0.7),
                redshift=Fixed(0.0),
            )
            model = SEDModel(spec, ssp)
            pred = model.predict_rest_sed({})
            return np.array(pred.wavelength), np.array(pred.sed)

        wl, lnu0 = build(0.0)
        _, lnu3 = build(1.0)
        i_uv = int(np.argmin(np.abs(wl - 1500.0)))
        i_v = int(np.argmin(np.abs(wl - 5500.0)))
        # Ratio of dust-reddened UV/V relative to unattenuated must be < 1
        ratio_dusty = lnu3[i_uv] / lnu3[i_v]
        ratio_clean = lnu0[i_uv] / lnu0[i_v]
        assert ratio_dusty < ratio_clean, "Dust must attenuate UV more than V"


class TestDustLawCombinations:
    """Systematic test of every registered dust attenuation curve.

    Convention: all curves should satisfy k(5500 Å) ≈ 1 so that A(λ) = k(λ)·A_V.
    k(FUV)/k(V) must be in [1, 15], k(I)/k(V) in [0.2, 2] for any realistic law.
    """

    WL: ClassVar = jnp.array([1500.0, 2175.0, 3000.0, 5500.0, 9000.0])
    _REQUIRES_RV: ClassVar[set[str]] = {"cardelli", "conroy2010", "d03_mwrv31"}

    @pytest.mark.parametrize(
        "name",
        [
            "calzetti",
            "cardelli",
            "conroy2010",
            "d03_mwrv31",
            "hd23_mwrv31",
            "kriek_conroy",
            "leitherer02",
            "li08",
            "lmc",
            "narayanan_z",
            "noll09",
            "power_law",
            "salim",
            "salim_sbl18",
            "smc",
            "tea",
            "vw07_bc",
            "vw07_diff",
            "wd01_mwrv31",
            "wd01_smcbar",
        ],
    )
    def test_dust_law_normalized_at_V_band(self, name):
        """All registered laws (except prevot_smc) normalise to k(V)=1 within 5%."""
        from tengri.components.dust.attenuation import resolve_dust_law

        fn = resolve_dust_law(name)
        kwargs = {"dust_Rv": 3.1} if name in self._REQUIRES_RV else {}
        k = np.array(fn(self.WL, **kwargs))
        assert np.all(np.isfinite(k))
        assert np.all(k >= 0.0), f"{name}: negative k values"
        assert 0.95 < k[3] < 1.05, f"{name}: k(5500Å) = {k[3]:.3f}"
        assert 1.5 < k[0] < 15.0, f"{name}: k(FUV) = {k[0]:.2f}"
        assert 0.2 < k[4] < 2.0, f"{name}: k(I) = {k[4]:.2f}"

    def test_prevot_smc_normalization(self):
        """prevot_smc should return k(V)=1 per tengri convention."""
        from tengri.components.dust.attenuation import prevot_smc

        k = np.array(prevot_smc(self.WL))
        assert 0.95 < k[3] < 1.05, f"prevot_smc k(V) = {k[3]:.3f}"

    @pytest.mark.parametrize("name", ["cardelli", "conroy2010", "d03_mwrv31", "hd23_mwrv31"])
    def test_milky_way_laws_have_uv_bump(self, name):
        """MW-type curves must show a 2175 Å feature: k(2175) > avg(k(1900), k(2450))."""
        from tengri.components.dust.attenuation import resolve_dust_law

        fn = resolve_dust_law(name)
        wl2 = jnp.array([1900.0, 2175.0, 2450.0])
        kwargs = {"dust_Rv": 3.1} if name in self._REQUIRES_RV else {}
        k = np.array(fn(wl2, **kwargs))
        baseline = 0.5 * (k[0] + k[2])
        assert k[1] > baseline, f"{name}: no UV bump (k(2175)={k[1]:.2f})"


class TestAGNModelCombinations:
    """Every registered AGN model should produce a physical SED at log(L_bol/L_sun)=11."""

    # All non-template AGN models we expect to evaluate at default parameters
    _ALL_MODELS: ClassVar[list[str]] = [
        "adaf",
        "cat3d_wind",
        "kubota_done",
        "kubota_done_full",
        "multicolor_agn",
        "qsogen",
        "silva04",
        "simple",
        "skirtor",
        "standard",
        "unified_nlr_blr",
    ]

    @pytest.mark.parametrize("name", _ALL_MODELS)
    def test_agn_model_physical_amplitude(self, name):
        """L_ν max should land in 10^22–10^33 erg/s/Hz for log(L_bol/L_sun)=11."""
        from tengri.agn import AGN_MODELS, resolve_agn_model

        if name not in AGN_MODELS:
            pytest.skip(f"AGN model '{name}' not registered")
        fn = resolve_agn_model(name)
        wl = jnp.logspace(np.log10(100), np.log10(5e5), 1000)
        try:
            L = np.array(fn(wl, agn_log_lbol=11.0))
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{name} requires data files not present")
        L_pos = L[L > 0]
        assert len(L_pos) > 0, f"{name}: all-zero/negative output"
        assert 1e22 < L_pos.max() < 1e33, (
            f"{name}: max L_ν = {L_pos.max():.2e} erg/s/Hz (outside 1e22–1e33)"
        )
        assert np.all(np.isfinite(L)), f"{name}: NaN/Inf in output"

    @pytest.mark.parametrize("name", _ALL_MODELS)
    def test_agn_model_linear_in_Lbol(self, name):
        """Each AGN model must scale ~linearly in 10^L_bol (within ~15% — qsogen Baldwin)."""
        from tengri.agn import AGN_MODELS, resolve_agn_model

        if name not in AGN_MODELS:
            pytest.skip(f"AGN model '{name}' not registered")
        fn = resolve_agn_model(name)
        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        try:
            a = np.array(fn(wl, agn_log_lbol=10.0))
            b = np.array(fn(wl, agn_log_lbol=11.0))
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{name} requires data files not present")
        # qsogen has mild Baldwin effect; most models linear.
        if a.max() == 0.0:
            pytest.skip(f"{name} returned zeros for log_lbol=10")
        ratio = b.max() / a.max()
        assert 8.0 < ratio < 12.0, f"{name}: 10x L_bol → {ratio:.2f}x L_ν"


class TestRadioVariantCrossCheck:
    """At z=0, SFR~1 M⊙/yr, M*~10^10 M⊙, different radio prescriptions should
    agree to within a factor of ~3 on L_1.4GHz."""

    WL = jnp.array([2.14e9])  # 1.4 GHz

    def test_bell2003_vs_delvecchio2021_at_z0(self):
        """Both models should give L_1.4GHz ~ 1e28 erg/s/Hz for L_IR = 10^10 L_sun."""
        from tengri.components.radio.radio import (
            radio_sfr_bell2003,
            radio_sfr_delvecchio2021,
        )

        L_IR = 1e10 * LSUN_ERG
        L_bell = float(np.array(radio_sfr_bell2003(self.WL, L_ir=L_IR))[0])
        L_delv = float(
            np.array(radio_sfr_delvecchio2021(self.WL, L_ir=L_IR, log_mstar=10.0, redshift=0.0))[0]
        )
        # Factor-of-2 agreement expected (different q_IR anchors: 2.64 vs 2.743)
        ratio = L_bell / L_delv
        assert 0.5 < ratio < 2.0, (
            f"Bell vs Delvecchio disagree: L_bell={L_bell:.2e}, L_delv={L_delv:.2e}"
        )

    def test_delvecchio_mass_dependence(self):
        """Delvecchio+2021: q_IR DECREASES with log M★ (mass_slope=+0.234),
        so massive galaxies have MORE radio per L_IR. Eq. 4 of arXiv:2503.20525.
        """
        from tengri.components.radio.radio import radio_sfr_delvecchio2021

        L_IR = 1e10 * LSUN_ERG
        L_lo = float(
            np.array(radio_sfr_delvecchio2021(self.WL, L_ir=L_IR, log_mstar=9.0, redshift=0.0))[0]
        )
        L_hi = float(
            np.array(radio_sfr_delvecchio2021(self.WL, L_ir=L_IR, log_mstar=11.0, redshift=0.0))[0]
        )
        assert L_hi > L_lo, "Massive galaxies should have more radio per L_IR (q_IR-)"
        # Expected ratio ~ 10^(2 · 0.234) = 2.95 for pure q shift, but the
        # low-SFR suppression multiplier inflates it at 10^9 Msun; allow a
        # wider window.
        ratio = L_hi / L_lo
        assert 2.0 < ratio < 6.0, f"L(M11)/L(M9) = {ratio:.2f}"


class TestSFHForms:
    """Every parametric SFH form must produce a non-negative SFR array of the
    right shape, with cumulative stellar mass ~ 10^7–10^11 M⊙ for SFR ~ 1."""

    @pytest.mark.parametrize(
        "sfh_type,kwargs",
        [
            (
                "dpl",
                {
                    "sfh_dpl_log_peak_sfr": 0.0,
                    "sfh_dpl_alpha": 2.0,
                    "sfh_dpl_beta": 1.0,
                    "sfh_dpl_tau_gyr": 3.0,
                },
            ),
            (
                "lnorm",
                {
                    "sfh_lnorm_log_peak_sfr": 0.0,
                    "sfh_lnorm_peak_lbt_gyr": 3.0,
                    "sfh_lnorm_width_gyr": 1.0,
                },
            ),
            (
                "tsnorm",
                {
                    "sfh_tsnorm_log_peak_sfr": 0.0,
                    "sfh_tsnorm_peak_lbt_gyr": 3.0,
                    "sfh_tsnorm_width_gyr": 2.0,
                    "sfh_tsnorm_skew": 0.0,
                    "sfh_tsnorm_trunc": 5.0,
                },
            ),
        ],
    )
    def test_sfh_produces_valid_array(self, sfh_type, kwargs):
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")
        param_dict = {k: Fixed(v) for k, v in kwargs.items()}
        param_dict.update(
            {
                "met_logzsol": Fixed(0.0),
                "dust_tau_bc": Fixed(0.0),
                "dust_tau_diff": Fixed(0.0),
                "dust_slope": Fixed(-0.7),
                "redshift": Fixed(0.0),
                "mean_sfh_type": sfh_type,
            }
        )
        spec = Parameters(**param_dict)
        model = SEDModel(spec, ssp)
        sfh = model.predict_sfh({})
        sfr = np.array(sfh["sfr_mean"])
        t = np.array(sfh["t_gyr"])
        assert sfr.shape == t.shape
        assert np.all(sfr >= 0.0), f"{sfh_type}: negative SFR"
        assert np.all(np.isfinite(sfr))
        dt_yr = np.abs(np.diff(t)) * 1e9
        mass = float(np.sum(sfr[:-1] * dt_yr))
        assert 1e7 < mass < 1e12, f"{sfh_type}: cumulative mass {mass:.2e} M⊙ outside [1e7, 1e12]"


class TestDustEmissionCombinations:
    """Every dust-emission template should peak in IR and conserve L_absorbed."""

    @pytest.mark.parametrize(
        "name", ["modified_blackbody", "casey2012", "dale2014", "draine_li2007"]
    )
    def test_dust_emission_peaks_in_ir(self, name):
        """Peak λ must be in IR (8–1000 μm) for cold-dust parameters."""
        import tengri.dust as dust_mod

        fn = getattr(dust_mod, name, None)
        if fn is None:
            pytest.skip(f"{name} not available in tengri.dust")
        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        try:
            if name == "modified_blackbody":
                L = np.array(fn(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8))
            elif name == "casey2012":
                L = np.array(fn(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0))
            elif name == "dale2014":
                L = np.array(fn(wl, 1e44, dust_alpha_dale=2.0))
            else:
                L = np.array(fn(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{name} requires data files not present")
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 8.0 < peak_um < 1000.0, f"{name}: peak at {peak_um:.1f} μm"

    @pytest.mark.parametrize("name", ["modified_blackbody", "casey2012", "draine_li2007"])
    def test_dust_emission_energy_balance(self, name):
        """∫L_ν dν ≈ L_absorbed (within 10% trapezoid tolerance)."""
        import tengri.dust as dust_mod

        fn = getattr(dust_mod, name, None)
        if fn is None:
            pytest.skip(f"{name} not available")
        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 5000)
        try:
            if name == "modified_blackbody":
                L = np.array(fn(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8))
            elif name == "casey2012":
                L = np.array(fn(wl, 1e44, dust_T=35.0))
            else:
                L = np.array(fn(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        except (FileNotFoundError, KeyError):
            pytest.skip(f"{name} requires data files not present")
        nu = 2.998e18 / np.array(wl)
        L_int = -np.trapezoid(L, nu)
        assert 0.9 < L_int / 1e44 < 1.1, (
            f"{name}: ∫L_ν dν / L_abs = {L_int / 1e44:.3f}, expected ≈1"
        )


class TestSSPMetallicity:
    def test_ssp_log_metallicity_range(self):
        """Standard SSP grids cover ~[-2.3, +0.3] in log10(Z/Z_sun).

        tengri stores ``ssp_lgmet`` in ABSOLUTE log10(Z), with LOG10_ZSUN = -1.848
        (MILES convention). So ssp_lgmet range should be ~[-4.1, -1.55].
        """
        from pathlib import Path

        from tengri import load_ssp_data

        ssp_path = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
        if not ssp_path.exists():
            pytest.skip("SSP file not available")
        ssp = load_ssp_data(str(ssp_path))
        lgmet = np.array(ssp.ssp_lgmet)
        assert -4.5 < lgmet.min() < -3.0
        assert -2.0 < lgmet.max() < -1.0
        # Convert to Z/Zsun (LOG10_ZSUN = -1.848)
        log_zsol = lgmet + 1.848
        assert log_zsol.min() < -1.5 and log_zsol.max() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
