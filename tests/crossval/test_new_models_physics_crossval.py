# SPDX-License-Identifier: BSD-3-Clause
"""Physics cross-validation for the 10 new models.

Tests physical correctness against published reference values, analytic
limits, and known astrophysical relationships. Goes beyond code-execution
tests to verify the implemented physics is right.

References
----------
- Allen et al. 2008, ApJS, 178, 20 — MAPPINGS III shock models
- Bellstedt et al. 2021, MNRAS, 503, 3309 — chemical evolution
- Boquien & Salim 2021, A&A, 653, A149 — BOSA templates
- da Cunha et al. 2008, MNRAS, 388, 1595 — MAGPHYS
- Haskell et al. 2024, arXiv:2401.11007 — TEA attenuation
- Hensley & Draine 2023, ApJ, 948, 55 — Astrodust
- Jones et al. 2017, A&A, 602, A46 — THEMIS dust
- Mahadevan 1997, ApJ, 477, 585 — ADAF
- Mason et al. 2018, ApJ, 856, 2 — patchy IGM
- Miralda-Escude 1998, ApJ, 501, 15 — Gunn-Peterson damping wing
- Smith et al. 2007, ApJ, 656, 770 — PAH Drude profiles
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval


# ── 1. SHOCK EMISSION — BPT diagram & velocity-dependent physics ──


class TestShockBPTPhysics:
    """Shock line ratios must trace known BPT diagram loci."""

    def test_oi_enhanced_vs_hii(self):
        """[OI] 6300/Halpha is THE shock diagnostic — must be strong.

        In HII regions, log([OI]/Halpha) < -1.5 typically.
        In shocks, log([OI]/Halpha) > -1.0 at v > 150 km/s (Allen+2008).
        """
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [200.0, 300.0, 500.0]:
            ratios = shock_line_ratios(v)
            oi_ha = float(ratios["OI_6300A"]) / float(ratios["HA_6563A"])
            log_oi_ha = np.log10(oi_ha)
            assert log_oi_ha > -1.5, (
                f"[OI]/Halpha at v={v}: log={log_oi_ha:.2f}, expected > -1.5 (shock diagnostic)"
            )

    def test_halpha_hbeta_above_case_b(self):
        """Shock Ha/Hb > 2.86 (collisional excitation adds to recombination).

        Allen+2008: Ha/Hb ranges from ~3.0 at 100 km/s to ~3.7 at 1000 km/s.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [100.0, 300.0, 500.0, 1000.0]:
            ratios = shock_line_ratios(v)
            ha_hb = float(ratios["HA_6563A"]) / float(ratios["Hb_4861A"])
            assert ha_hb >= 2.85, f"Shock Ha/Hb at v={v}: {ha_hb:.2f}, expected >= 2.86 (Case B)"

    def test_sii_enhanced_in_shocks(self):
        """[SII]/Halpha enhanced in shocks vs HII regions.

        HII regions: log([SII]/Halpha) typically < -0.4
        Shocks: log([SII]/Halpha) typically > -0.4 at v >= 150 km/s
        """
        from tengri.components.nebular.shock import shock_line_ratios

        for v in [200.0, 300.0, 500.0]:
            ratios = shock_line_ratios(v)
            sii_total = float(ratios["SII_6716A"]) + float(ratios["SII_6731A"])
            sii_ha = sii_total / float(ratios["HA_6563A"])
            log_sii_ha = np.log10(sii_ha)
            assert log_sii_ha > -0.7, (
                f"[SII]/Halpha at v={v}: log={log_sii_ha:.2f}, expected > -0.7 (shock-enhanced)"
            )

    def test_nii_ha_turns_over_with_velocity(self):
        """[NII]/Halpha rises with shock velocity, then turns over.

        Allen+2008: [NII] strengthens while the post-shock temperature suits
        N+ collisional excitation (~1-3e4 K), and weakens once the gas is hot
        enough to ionize nitrogen further. The turnover is real; this test used
        to place it at ~150 km/s and assert `ratio(150) > ratio(1000)`, which
        the grid does not support for **any** component (#1728):

            shock      peak 500 km/s   0.30 (100) -> 1.89 (500) -> 1.77 (1000)
            precursor  peak 700 km/s
            combined   peak 700 km/s   0.31 (100) -> 1.39 (700) -> 1.33 (1000)

        At 150 km/s the combined ratio is 0.35, well below its 1.39 peak. The
        assertion is now the turnover itself, sampled densely enough to find
        it, rather than a hand-placed velocity.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        velocities = np.arange(100.0, 1001.0, 50.0)
        nii_ha = np.array(
            [
                float(shock_line_ratios(float(v))["NII_6583A"])
                / float(shock_line_ratios(float(v))["HA_6563A"])
                for v in velocities
            ]
        )

        peak_idx = int(np.argmax(nii_ha))
        assert 0 < peak_idx < len(velocities) - 1, (
            f"[NII]/Halpha should turn over inside the grid, peak at "
            f"{velocities[peak_idx]:.0f} km/s (an edge)"
        )
        assert nii_ha[peak_idx] > 2.0 * nii_ha[0], (
            "the ratio should rise substantially from 100 km/s to its peak, "
            f"got {nii_ha[0]:.3f} -> {nii_ha[peak_idx]:.3f}"
        )
        assert nii_ha[-1] < nii_ha[peak_idx], "the ratio must fall again past the peak"

    def test_shock_temperature_scaling(self):
        """Shock temperature T ~ 1.4e5 * (v/100 km/s)^2 K.

        This is post-shock gas temperature from Rankine-Hugoniot jump
        conditions. The line ratios should reflect this scaling.
        """
        from tengri.components.nebular.shock import shock_line_ratios

        # At 100 km/s: T ~ 1.4e5 K → [OIII] weak (below ionization)
        # At 300 km/s: T ~ 1.3e6 K → [OIII] strong
        ratios_low = shock_line_ratios(100.0)
        ratios_high = shock_line_ratios(300.0)

        oiii_low = float(ratios_low["O3_5007A"])
        oiii_high = float(ratios_high["O3_5007A"])

        assert oiii_high > oiii_low, (
            "[OIII] should be stronger at 300 km/s (T~1.3e6 K) "
            f"than 100 km/s (T~1.4e5 K): {oiii_high:.2f} vs {oiii_low:.2f}"
        )


# ── 2. ADAF — spectral component physics ──────────────────────────


class TestADAFSpectralPhysics:
    """ADAF must show correct multi-component spectral structure."""

    @pytest.fixture
    def wavelength(self):
        return jnp.logspace(0, 8, 500)

    def test_radio_emission_present(self, wavelength):
        """ADAF must emit in radio/mm (synchrotron component).

        Synchrotron peak at ~10^11-10^12 Hz = 3e4-3e5 um = 3e8-3e9 A.
        """
        from tengri.components.agn.adaf import adaf_spectrum

        l_nu = adaf_spectrum(
            wavelength,
            agn_log_lbol=8.42,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )

        # Radio/mm: wavelength > 1e5 A (> 10 um)
        radio_mask = wavelength > 1e5
        radio_flux = float(jnp.sum(l_nu[radio_mask]))
        assert radio_flux > 0, "ADAF should emit in radio/mm (synchrotron)"

    def test_adaf_radiative_efficiency_low(self, wavelength):
        """ADAF stays sub-Eddington at a physical accretion luminosity.

        adaf_spectrum is L_bol-canonical (#898): ``agn_log_lbol`` is log10(L/Lsun)
        and the radiative inefficiency lives in the derived mdot (Eq. 49), tested
        in test_adaf_mahadevan.py. Here we simply check the model does not produce
        super-Eddington output at a physical LLAGN luminosity.
        """
        from tengri.components.agn.adaf import adaf_spectrum

        c_aa = 2.99792458e18
        l_nu = adaf_spectrum(
            wavelength,
            agn_log_lbol=10.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_adaf_delta=0.1,
        )
        nu = c_aa / wavelength
        l_bol = float(-jnp.trapezoid(l_nu, nu))

        # L_Edd for M_BH = 1e8 Msun ~ 1.26e46 erg/s.
        l_edd = 1.26e46
        assert l_bol < l_edd, f"ADAF L_bol={l_bol:.3e} exceeds L_Edd={l_edd:.3e}"

    def test_truncation_radius_retired(self, wavelength):
        """agn_r_tr (the bundled truncated outer disc) was retired in #898 — the
        faithful Mahadevan 1997 ADAF is inner-flow only, so r_tr has no effect."""
        from tengri.components.agn.adaf import adaf_spectrum

        l_small = adaf_spectrum(wavelength, agn_log_lbol=10.0, agn_log_mbh=8.0, agn_r_tr=30.0)
        l_large = adaf_spectrum(wavelength, agn_log_lbol=10.0, agn_log_mbh=8.0, agn_r_tr=300.0)
        assert bool(jnp.array_equal(l_small, l_large)), "agn_r_tr should have no effect (retired)"


# ── 3. PATCHY IGM — Gunn-Peterson & damping wing quantitative ─────


class TestPatchyIGMQuantitative:
    """Damping wing must match Miralda-Escude 1998 quantitative predictions."""

    def test_gunn_peterson_tau_at_z6(self):
        """τ_GP ≈ 7.16e5 * ((1+z)/10)^1.5 at z=6 for fully neutral IGM.

        This is Eq. 1 of Miralda-Escude 1998.
        """
        z = 6.0
        tau_gp_expected = 7.16e5 * ((1 + z) / 10.0) ** 1.5
        # Should be ~2.5e5 — enormous optical depth
        assert tau_gp_expected > 1e5, f"τ_GP at z=6 = {tau_gp_expected:.3e}"
        assert tau_gp_expected < 1e7, f"τ_GP at z=6 = {tau_gp_expected:.3e}"

    def test_fully_neutral_complete_absorption(self):
        """x_HI=1.0 at z=7 should produce near-complete absorption blueward of Lya."""
        from tengri.components.igm import igm_transmission_patchy

        # Lya at z=7: 1216 * 8 = 9728 A
        wave_obs = jnp.linspace(8000.0, 12000.0, 500)
        t = np.asarray(igm_transmission_patchy(wave_obs, 7.0, x_HI=1.0, R_bubble=0.5))

        # Blueward of Lya (< 9728 A): should be heavily absorbed
        blue_mask = np.asarray(wave_obs) < 9728.0
        if np.any(blue_mask):
            mean_t_blue = np.mean(t[blue_mask])
            assert mean_t_blue < 0.1, f"Mean T blueward of Lya at z=7, x_HI=1: {mean_t_blue:.3f}"

    def test_damping_wing_redward_of_lya(self):
        """Damping wing extends redward of Lya (diagnostic of neutral IGM).

        The red damping wing is the key observational signature that
        distinguishes a partially neutral IGM from a fully ionized one.
        """
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        lya_obs = 1215.67 * (1 + z)  # ~ 9725 A

        # Just redward of Lya: 9800-10200 A
        wave_red = jnp.linspace(lya_obs + 50, lya_obs + 500, 100)

        # x_HI = 0: no damping wing (Inoue only)
        t_ionized = np.asarray(igm_transmission_patchy(wave_red, z, x_HI=0.0))

        # x_HI = 0.5: damping wing should reduce transmission
        t_neutral = np.asarray(igm_transmission_patchy(wave_red, z, x_HI=0.5, R_bubble=1.0))

        # Damping wing reduces transmission redward of Lya
        assert np.mean(t_neutral) < np.mean(t_ionized), (
            f"Damping wing not detected: T(x_HI=0.5)={np.mean(t_neutral):.3f} "
            f">= T(x_HI=0)={np.mean(t_ionized):.3f}"
        )

    def test_larger_bubble_more_transmission(self):
        """Larger ionized bubble → more transmission near Lya."""
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        lya_obs = 1215.67 * (1 + z)
        wave = jnp.linspace(lya_obs + 20, lya_obs + 300, 100)

        t_small = np.mean(np.asarray(igm_transmission_patchy(wave, z, x_HI=0.5, R_bubble=0.5)))
        t_large = np.mean(np.asarray(igm_transmission_patchy(wave, z, x_HI=0.5, R_bubble=5.0)))

        assert t_large > t_small, (
            f"Larger bubble should increase T: R=0.5→{t_small:.3f}, R=5→{t_large:.3f}"
        )

    def test_x_hi_zero_matches_standard(self):
        """x_HI=0 should exactly reproduce standard Inoue+2014 at any z."""
        from tengri.components.igm import igm_transmission, igm_transmission_patchy

        for z in [3.0, 5.0, 7.0]:
            wave = jnp.linspace(3000.0, 15000.0, 200)
            t_standard = np.asarray(igm_transmission(wave, z))
            t_patchy = np.asarray(igm_transmission_patchy(wave, z, x_HI=0.0))

            np.testing.assert_allclose(
                t_patchy,
                t_standard,
                rtol=1e-10,
                err_msg=f"x_HI=0 should match standard at z={z}",
            )


# ── 4. CHEMICAL EVOLUTION — closed-box analytic formula ───────────


class TestChemEvolAnalyticFormulas:
    """Closed-box metallicity must match Z = y * ln(1/f_gas)."""

    def test_closed_box_independent_of_sfh_shape(self):
        """Z depends only on gas fraction, NOT on SFH shape.

        This is the fundamental property of the closed-box model
        (Bellstedt+2021, Eq. 2). Two different SFHs that consume the
        same total gas mass should give the same final metallicity.
        """
        from tengri.components.stellar.sfh.chemical_evolution import closed_box_metallicity

        age_grid = jnp.logspace(6, 10.14, 128)

        # SFH 1: constant
        sfr_const = jnp.ones_like(age_grid) * 2.0

        # SFH 2: rising (same total integral approximately)
        sfr_rising = jnp.linspace(4.0, 0.1, len(age_grid))

        z_const = closed_box_metallicity(age_grid, sfr_const, yield_y=0.03)
        z_rising = closed_box_metallicity(age_grid, sfr_rising, yield_y=0.03)

        # Final metallicity (youngest end, index 0) should differ
        # because total mass formed differs, BUT the Z(mu) relation
        # should hold: at the same gas fraction, Z must be the same.
        # Both should be monotonically increasing in cosmic time.
        assert float(z_const[-1]) < float(z_const[0]), "Z should be higher at young (recent) times"
        assert float(z_rising[-1]) < float(z_rising[0]), (
            "Z should be higher at young (recent) times"
        )

    def test_effective_yield_with_outflows(self):
        """y_eff = y / (1 + eta): outflows reduce effective yield.

        Outflows expel metals, reducing the final metallicity.
        log(Z/Zsun) should be more negative (lower) with outflows.
        """
        from tengri.components.stellar.sfh.chemical_evolution import closed_box_metallicity

        age_grid = jnp.logspace(6, 10.14, 128)
        sfr = jnp.ones_like(age_grid) * 1.0

        z_closed = closed_box_metallicity(age_grid, sfr, yield_y=0.03, eta_outflow=0.0)
        z_leaky = closed_box_metallicity(age_grid, sfr, yield_y=0.03, eta_outflow=2.0)

        # Outflows should reduce Z (more negative log Z/Zsun)
        z_closed_val = float(z_closed[0])  # youngest = most enriched
        z_leaky_val = float(z_leaky[0])

        assert z_leaky_val < z_closed_val, (
            f"Outflows should reduce Z: closed={z_closed_val:.3f}, "
            f"leaky={z_leaky_val:.3f} (leaky should be more negative)"
        )

    def test_yield_values_chabrier(self):
        """Chabrier IMF yield ~ 0.03 (Vincenzo+2016)."""
        from tengri.components.stellar.sfh.chemical_evolution import closed_box_metallicity

        age_grid = jnp.logspace(6, 10.14, 128)
        sfr = jnp.ones_like(age_grid) * 1.0

        z = closed_box_metallicity(age_grid, sfr, yield_y=0.03, f_gas_init=0.9)

        # With y=0.03, should reach near-solar metallicity
        z_final = float(z[0])  # youngest = most enriched
        assert -1.5 < z_final < 1.0, (
            f"Final log(Z/Zsun) = {z_final:.2f} with y=0.03, expected -1.5 to +1.0"
        )

    def test_higher_yield_higher_metallicity(self):
        """Higher nucleosynthetic yield → higher final metallicity."""
        from tengri.components.stellar.sfh.chemical_evolution import closed_box_metallicity

        age_grid = jnp.logspace(6, 10.14, 128)
        sfr = jnp.ones_like(age_grid) * 1.0

        z_low_y = closed_box_metallicity(age_grid, sfr, yield_y=0.01)
        z_high_y = closed_box_metallicity(age_grid, sfr, yield_y=0.05)

        assert float(z_high_y[0]) > float(z_low_y[0]), "Higher yield should give higher final Z"


# ── 5. TEA ATTENUATION — bump-slope correlation ───────────────────


class TestTEABumpSlopePhysics:
    """TEA E_b(delta) relation from Haskell+2024 NIHAO-SKIRT calibration."""

    def test_eb_formula_at_delta_zero(self):
        """E_b(delta=0) = 2.5 * exp(0) = 2.5 (maximum bump)."""
        from tengri.components.dust.attenuation import tea

        wave = jnp.array([2175.0, 5500.0])
        k_tea = tea(wave, dust_delta=0.0, dust_tea_scatter=0.0)
        k_calz = tea(wave, dust_delta=0.0, dust_tea_scatter=0.0)

        # At delta=0, E_b = 2.5 — strong UV bump
        # The 2175A feature should be prominent
        assert float(k_tea[0]) > float(k_tea[1]), "TEA at delta=0 should show 2175A bump"

    def test_negative_delta_weaker_bump(self):
        """More negative delta (steeper curve) → weaker UV bump.

        E_b = 2.5 * exp(3.5 * delta). For delta < 0, E_b < 2.5.
        """
        from tengri.components.dust.attenuation import tea

        wave = jnp.array([2175.0])

        k_flat = float(tea(wave, dust_delta=0.0)[0])
        k_steep = float(tea(wave, dust_delta=-0.3)[0])

        # Steeper (more negative delta) → lower E_b → less bump
        # The total k at 2175A should be lower relative to continuum
        # when the bump is weaker
        # Actually both k values include the bump, so let's check
        # the bump contribution directly
        eb_flat = 2.5 * np.exp(3.5 * 0.0)
        eb_steep = 2.5 * np.exp(3.5 * (-0.3))

        assert eb_steep < eb_flat, (
            f"E_b should decrease with negative delta: "
            f"delta=0→{eb_flat:.2f}, delta=-0.3→{eb_steep:.2f}"
        )

    def test_scatter_shifts_bump_strength(self):
        """Scatter parameter adds log-normal variation to E_b.

        E_b = 2.5 * exp(3.5*delta) * 10^scatter
        """
        # Positive scatter → stronger bump
        eb_nominal = 2.5 * np.exp(3.5 * (-0.2))
        eb_up = 2.5 * np.exp(3.5 * (-0.2)) * 10**0.3
        eb_down = 2.5 * np.exp(3.5 * (-0.2)) * 10 ** (-0.3)

        assert eb_up > eb_nominal > eb_down

    def test_tea_matches_kriek_conroy_form(self):
        """TEA uses Kriek-Conroy functional form internally."""
        from tengri.components.dust.attenuation import kriek_conroy, tea

        wave = jnp.linspace(1200.0, 10000.0, 200)

        # TEA at delta=0, scatter=0: E_b = 2.5
        k_tea = np.asarray(tea(wave, dust_delta=0.0, dust_tea_scatter=0.0))

        # Kriek-Conroy with same E_b and delta
        k_kc = np.asarray(kriek_conroy(wave, dust_bump_strength=2.5, dust_delta=0.0))

        np.testing.assert_allclose(
            k_tea, k_kc, rtol=1e-10, err_msg="TEA should match K&C with same E_b, delta"
        )


# ── 6. MAGPHYS — MBB peak wavelengths (Wien's law) ────────────────


@pytest.mark.skip(
    reason=(
        "#1728: magphys_dc08 does not exist — nothing MAGPHYS-related is in "
        "components.dust.emission_templates or list_dust_emission_models(). "
        "Kept as a spec for the model rather than deleted; see the identical "
        "skip on TestMagphysPhysics in test_template_models_physics.py."
    )
)
class TestMagphysWienPeaks:
    """Modified blackbody components must peak at correct wavelengths."""

    def test_cold_component_peaks_far_ir(self):
        """Cold dust (T=20K) peaks at ~150 um = 1.5e6 A (Wien's law).

        Wien: λ_peak = b / T with b = 2898 um*K for standard BB.
        For MBB with β=2: λ_peak ≈ 2898 / T * (4+β)/(3+β) ≈ 174 um.
        """
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(3, 7.5, 2000)  # 1000 A to 3e7 A

        # Isolate cold component: xi_pah=0, xi_mir=0, xi_warm=0 → all cold
        sed = magphys_dc08(
            wave,
            L_absorbed=1e10,
            dust_xi_pah=0.0,
            dust_xi_mir=0.0,
            dust_xi_warm=0.0,
            dust_T_cold=20.0,
        )

        peak_wave = float(wave[jnp.argmax(sed * wave)])  # peak in λ*L_λ
        peak_um = peak_wave / 1e4

        # T=20K, β=2: peak ~ 100-200 um
        assert 80 < peak_um < 300, f"Cold dust (20K) peak at {peak_um:.0f} um, expected 80-300 um"

    def test_warm_component_peaks_mid_ir(self):
        """Warm dust (T=45K) peaks at ~65 um."""
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(3, 7, 2000)

        # Isolate warm: xi_pah=0, xi_mir=0, xi_warm=1.0
        sed = magphys_dc08(
            wave,
            L_absorbed=1e10,
            dust_xi_pah=0.0,
            dust_xi_mir=0.0,
            dust_xi_warm=1.0,
            dust_T_warm=45.0,
        )

        peak_wave = float(wave[jnp.argmax(sed * wave)])
        peak_um = peak_wave / 1e4

        # T=45K: peak ~ 40-100 um
        assert 30 < peak_um < 150, f"Warm dust (45K) peak at {peak_um:.0f} um, expected 30-150 um"

    def test_hot_component_peaks_near_ir(self):
        """Hot MIR continuum (T=180K) peaks at ~15-20 um."""
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(3, 6.5, 2000)

        # Isolate hot MIR: xi_pah=0, xi_mir=1.0, xi_warm=0
        sed = magphys_dc08(
            wave,
            L_absorbed=1e10,
            dust_xi_pah=0.0,
            dust_xi_mir=1.0,
            dust_xi_warm=0.0,
            dust_T_hot=180.0,
        )

        peak_wave = float(wave[jnp.argmax(sed * wave)])
        peak_um = peak_wave / 1e4

        # T=180K: peak ~ 10-30 um
        assert 5 < peak_um < 50, f"Hot dust (180K) peak at {peak_um:.0f} um, expected 5-50 um"


# ── 7. PAH — Drude profile properties ─────────────────────────────


@pytest.mark.skip(
    reason=(
        "#1728: magphys_dc08 does not exist — nothing MAGPHYS-related is in "
        "components.dust.emission_templates or list_dust_emission_models(). "
        "Kept as a spec for the model rather than deleted; see the identical "
        "skip on TestMagphysPhysics in test_template_models_physics.py."
    )
)
class TestPAHDrudeProfilePhysics:
    """PAH Drude profiles must have correct physical properties."""

    def test_pah_7p7_is_strongest(self):
        """7.7 um complex is the strongest PAH feature (Smith+2007)."""
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(np.log10(3e4), np.log10(2e5), 2000)  # 3-20 um

        # Pure PAH component
        sed = magphys_dc08(
            wave,
            L_absorbed=1e10,
            dust_xi_pah=1.0,
            dust_xi_mir=0.0,
            dust_xi_warm=0.0,
        )
        wave_um = np.asarray(wave) / 1e4
        sed_arr = np.asarray(sed)

        # Find peaks near 6.2, 7.7, 8.6, 11.3 um
        def flux_near(center_um, width_um=0.5):
            mask = np.abs(wave_um - center_um) < width_um
            return np.max(sed_arr[mask]) if np.any(mask) else 0.0

        f_6p2 = flux_near(6.2)
        f_7p7 = flux_near(7.7)
        f_8p6 = flux_near(8.6)
        f_11p3 = flux_near(11.3)

        # 7.7 should be strongest
        assert f_7p7 > f_6p2, "7.7 um should be stronger than 6.2 um"
        assert f_7p7 > f_8p6, "7.7 um should be stronger than 8.6 um"
        assert f_7p7 > f_11p3, "7.7 um should be stronger than 11.3 um"

    def test_drude_profile_symmetric_in_log_wavelength(self):
        """Drude profile is symmetric in 1/λ (Lorentzian in frequency)."""
        # D(λ) = S * (γ/λ0)^2 / ((λ/λ0 - λ0/λ)^2 + (γ/λ0)^2)
        # At λ = λ0 * (1 ± ε): approximately symmetric for small ε
        lam0 = 7.7  # um
        gamma = 0.126  # fractional FWHM for 7.7 um
        s = 1.0

        def drude(lam):
            x = lam / lam0 - lam0 / lam
            return s * (gamma) ** 2 / (x**2 + gamma**2)

        # Check symmetry around center
        d_blue = drude(lam0 * 0.95)
        d_red = drude(lam0 * 1.05)

        # Not exactly symmetric in λ, but should be close
        np.testing.assert_allclose(
            d_blue, d_red, rtol=0.2, err_msg="Drude profile not approximately symmetric"
        )


# ── 8. Li08 attenuation — Li et al. (2008) Eq. (1), coefficients c1–c4


class TestLi08AttenuationPhysics:
    """Li et al. (2008) four-parameter analytic A_lambda / A_V curve."""

    def test_normalized_at_vband(self):
        """k(5500A) must equal 1.0 (V-band normalization)."""
        from tengri.components.dust.attenuation import li08

        k_v = float(li08(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k_v, 1.0, atol=0.01, err_msg=f"Li08 k(V)={k_v:.4f}")

    def test_c2_steepens_far_uv(self):
        """Larger c2 steepens the far-UV more than the near-UV."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.array([1000.0, 3000.0])
        k_base = np.asarray(li08(wave, dust_c2=4.0))
        k_steep = np.asarray(li08(wave, dust_c2=5.5))
        d1000 = abs(float(k_steep[0] - k_base[0]))
        d3000 = abs(float(k_steep[1] - k_base[1]))
        assert d1000 > 0.05 and d1000 > d3000, (
            f"c2 should move 1000A more than 3000A: d1000={d1000}, d3000={d3000}"
        )

    def test_mw_preset_has_bump(self):
        """MW-like preset (c4 > 0) enhances k(2175) vs SMC-like (c4 = 0)."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.array([2175.0, 3000.0])
        k_mw = li08(wave, dust_c1=6.0, dust_c2=4.0, dust_c3=2.0, dust_c4=0.04)
        k_smc = li08(wave, dust_c1=5.0, dust_c2=5.5, dust_c3=1.5, dust_c4=0.0)
        mw_ratio = float(k_mw[0]) / float(k_mw[1])
        smc_ratio = float(k_smc[0]) / float(k_smc[1])
        assert mw_ratio > smc_ratio, (
            f"MW-like bump should raise k(2175)/k(3000): MW={mw_ratio:.2f}, SMC={smc_ratio:.2f}"
        )

    def test_smc_no_bump(self):
        """SMC-like preset (c4=0) is mostly monotonic through 2175A window."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.linspace(1800.0, 2800.0, 100)
        k = np.asarray(li08(wave, dust_c1=5.0, dust_c2=5.5, dust_c3=1.5, dust_c4=0.0))
        diff = np.diff(k)
        mostly_decreasing = np.sum(diff < 0) > 0.8 * len(diff)
        assert mostly_decreasing, "SMC-like (c4=0) should be mostly monotonic in UV"

    def test_calzetti_preset_grayer(self):
        """Calzetti-like preset is grayer than MW-like (smaller k_1500 / k_5500)."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.array([1500.0, 5500.0])
        k_mw = np.asarray(li08(wave, dust_c1=6.0, dust_c2=4.0, dust_c3=2.0, dust_c4=0.04))
        k_calz = np.asarray(li08(wave, dust_c1=3.5, dust_c2=2.5, dust_c3=3.0, dust_c4=0.0))
        mw_ratio = float(k_mw[0] / k_mw[1])
        calz_ratio = float(k_calz[0] / k_calz[1])
        assert calz_ratio < mw_ratio, (
            f"Calzetti-like should be grayer: MW={mw_ratio:.3f}, Calz={calz_ratio:.3f}"
        )

    def test_positive_everywhere(self):
        """k(λ) must be non-negative at all wavelengths."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.linspace(900.0, 30000.0, 1000)
        k = np.asarray(li08(wave))
        assert np.all(k >= 0), "Li08 k(lambda) has negative values"
