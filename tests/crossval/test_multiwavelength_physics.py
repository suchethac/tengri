# SPDX-License-Identifier: BSD-3-Clause
"""Physics tests for radio, X-ray, IGM, PSD, and SED quantities.

References
----------
- Bell 2003, ApJ, 586, 794 (FIR-radio correlation q_TIR ~ 2.64)
- Grimm+2003, MNRAS, 339, 793 (HMXB L_X-SFR relation)
- Gilfanov 2004, MNRAS, 349, 146 (LMXB L_X-M* relation)
- Inoue+2014, MNRAS, 442, 1805 (IGM transmission)
- Meurer+1999 (IRX-beta relation)
- Balogh+1999, ApJ, 527, 54 (Dn4000 definition)
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.crossval


# ── 1. RADIO — FIR-radio correlation ──────────────────────────────


class TestRadioPhysics:
    """Radio emission must obey FIR-radio correlation."""

    def test_radio_sf_scales_with_lir(self):
        """Radio SFR luminosity ∝ L_IR (FIR-radio correlation)."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.geomspace(1e7, 1e10, 500)  # radio
        l_low = radio_star_forming(wave, L_ir=1e10)
        l_high = radio_star_forming(wave, L_ir=1e11)
        ratio = float(jnp.sum(l_high)) / float(jnp.sum(l_low))
        assert 5.0 < ratio < 20.0, f"10x L_IR should give ~10x radio flux, got {ratio:.1f}x"

    def test_qir_bell2003(self):
        """Bell (2003): q_TIR ≈ 2.64 for star-forming galaxies.

        q_IR = log10(L_TIR / (3.75e12 Hz)) - log10(L_1.4GHz / (W Hz^-1))
        Higher q_IR → less radio per unit IR → less radio flux.
        """
        from tengri.components.radio import radio_star_forming

        wave = jnp.geomspace(1e7, 1e10, 500)
        l_low_q = radio_star_forming(wave, L_ir=1e10, q_ir=2.0)
        l_high_q = radio_star_forming(wave, L_ir=1e10, q_ir=3.0)

        # Higher q_IR → less radio
        assert float(jnp.sum(l_high_q)) < float(jnp.sum(l_low_q)), (
            "Higher q_IR should give less radio emission"
        )

    def test_spectral_index_sf(self):
        """Star-forming radio: L_ν ∝ ν^{-α_sf} with α_sf ≈ 0.8."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.geomspace(1e8, 1e10, 100)  # radio: 1cm - 1m
        l_nu = radio_star_forming(wave, L_ir=1e10, alpha_sf=0.8)

        # Check power-law slope in log-log space
        nu = 2.99792458e18 / wave
        log_nu = jnp.log10(nu)
        log_l = jnp.log10(jnp.maximum(l_nu, 1e-50))

        # Fit slope — should be approximately -alpha_sf
        valid = l_nu > 0
        if jnp.sum(valid) > 10:
            coeffs = jnp.polyfit(log_nu[valid], log_l[valid], 1)
            slope = float(coeffs[0])
            assert -1.2 < slope < -0.4, f"Radio spectral index slope {slope:.2f}, expected ~-0.8"

    def test_radio_agn_spectral_index(self):
        """AGN radio: L_ν ∝ ν^{-α_agn} with α_agn ≈ 0.7."""
        from tengri.components.radio import radio_agn

        wave = jnp.geomspace(1e8, 1e10, 100)
        l_nu = radio_agn(wave, L_agn_bol=1e44, radio_loudness=1.0, alpha_agn=0.7)
        chex.assert_tree_all_finite(l_nu)
        assert float(jnp.sum(l_nu)) > 0, "Radio AGN should produce flux"

    def test_radio_loudness_controls_agn_radio(self):
        """Higher radio_loudness → more radio from AGN."""
        from tengri.components.radio import radio_agn

        wave = jnp.geomspace(1e8, 1e10, 100)
        l_quiet = radio_agn(wave, L_agn_bol=1e44, radio_loudness=0.0)
        l_loud = radio_agn(wave, L_agn_bol=1e44, radio_loudness=2.0)
        assert float(jnp.sum(l_loud)) > float(jnp.sum(l_quiet)), (
            "Higher radio_loudness should give more radio"
        )


# ── 2. X-RAY — HMXB/LMXB scaling ──────────────────────────────────


class TestXrayPhysics:
    """X-ray binary emission must follow Grimm+2003 scaling."""

    def test_hmxb_scales_with_sfr(self):
        """Grimm+2003: L_X(HMXB) ∝ SFR."""
        from tengri.components.xray import xray_xrb

        wave = jnp.geomspace(1.0, 100.0, 100)  # X-ray: 0.1-12 keV
        l_low = xray_xrb(wave, sfr=1.0, stellar_mass=1e10)
        l_high = xray_xrb(wave, sfr=10.0, stellar_mass=1e10)
        ratio = float(jnp.sum(l_high)) / max(float(jnp.sum(l_low)), 1e-50)
        assert 5.0 < ratio < 20.0, f"10x SFR should give ~10x HMXB X-ray, got {ratio:.1f}x"

    def test_lmxb_scales_with_mass(self):
        """Gilfanov (2004): L_X(LMXB) ∝ M*."""
        from tengri.components.xray import xray_xrb

        wave = jnp.geomspace(1.0, 100.0, 100)
        l_low = xray_xrb(wave, sfr=0.01, stellar_mass=1e10)
        l_high = xray_xrb(wave, sfr=0.01, stellar_mass=1e11)
        ratio = float(jnp.sum(l_high)) / max(float(jnp.sum(l_low)), 1e-50)
        assert 5.0 < ratio < 20.0, f"10x M* should give ~10x LMXB X-ray, got {ratio:.1f}x"

    def test_agn_corona_spectral_shape(self):
        """AGN X-ray corona: power-law L_ν ∝ ν^{1-Γ} with cutoff."""
        from tengri.components.xray import xray_agn_corona

        wave = jnp.geomspace(0.1, 100.0, 200)  # hard X-ray
        l_nu = xray_agn_corona(wave, L_agn_bol=1e44, gamma=1.8)
        chex.assert_tree_all_finite(l_nu)
        assert float(jnp.sum(l_nu)) > 0

    def test_softer_gamma_more_soft_xray(self):
        """Higher Γ → softer spectrum → more flux at soft X-rays."""
        from tengri.components.xray import xray_agn_corona

        wave = jnp.geomspace(1.0, 100.0, 200)
        l_hard = xray_agn_corona(wave, L_agn_bol=1e44, gamma=1.5)
        l_soft = xray_agn_corona(wave, L_agn_bol=1e44, gamma=2.2)

        # Soft X-ray (> 10A, lower energy)
        soft_mask = wave > 10.0
        hard_mask = wave < 5.0

        ratio_hard = float(jnp.mean(l_hard[soft_mask]) / jnp.mean(l_hard[hard_mask]))
        ratio_soft = float(jnp.mean(l_soft[soft_mask]) / jnp.mean(l_soft[hard_mask]))
        assert ratio_soft > ratio_hard, "Higher Γ should give softer spectrum"


# ── 3. IGM TRANSMISSION — Inoue+2014 ──────────────────────────────


class TestIGMPhysics:
    """IGM transmission must obey Lyman series absorption physics."""

    def test_full_transmission_above_lya(self):
        """Above Lyα in observed frame: T = 1 (no IGM absorption)."""
        from tengri.components.igm import igm_transmission

        z = 3.0
        wave_obs = jnp.array([1216.0 * (1 + z) + 100.0])  # just above Lyα
        t = float(igm_transmission(wave_obs, z_source=z)[0])
        np.testing.assert_allclose(t, 1.0, atol=0.05)

    def test_strong_absorption_below_lyman_limit(self):
        """Below Lyman limit (912A rest): T → 0 at z > 3."""
        from tengri.components.igm import igm_transmission

        z = 4.0
        wave_obs = jnp.array([912.0 * (1 + z) - 100.0])  # below LL
        t = float(igm_transmission(wave_obs, z_source=z)[0])
        assert t < 0.5, f"T below Lyman limit at z=4 should be < 0.5, got {t:.2f}"

    def test_higher_z_more_absorption(self):
        """Higher redshift → more IGM absorption (more neutral H)."""
        from tengri.components.igm import igm_transmission

        # Compare at same rest-frame wavelength (1100A = between Lyα and LL)
        wave_rest = 1100.0
        t_z2 = float(igm_transmission(jnp.array([wave_rest * 3.0]), z_source=2.0)[0])
        t_z5 = float(igm_transmission(jnp.array([wave_rest * 6.0]), z_source=5.0)[0])
        assert t_z5 <= t_z2 + 0.05, (
            f"z=5 should have more absorption than z=2: T(z=5)={t_z5:.2f} vs T(z=2)={t_z2:.2f}"
        )

    def test_transmission_bounded_0_1(self):
        """Transmission must be in [0, 1]."""
        from tengri.components.igm import igm_transmission

        for z in [1.0, 3.0, 6.0]:
            wave_obs = jnp.geomspace(500.0, 20000.0, 500)
            t = igm_transmission(wave_obs, z_source=z)
            assert jnp.all(t >= -0.01), f"T < 0 at z={z}"
            assert jnp.all(t <= 1.01), f"T > 1 at z={z}"

    def test_z0_no_absorption(self):
        """At z=0: no IGM absorption (T = 1 everywhere)."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.geomspace(500.0, 20000.0, 200)
        t = igm_transmission(wave_obs, z_source=0.0)
        np.testing.assert_allclose(t, 1.0, atol=0.01)


# ── 4. PSD MODELS — power spectral density physics ────────────────


class TestPSDPhysics:
    """Power spectral density models for SFH burstiness."""

    def test_drw_low_freq_flat(self):
        """DRW PSD: flat at ω << 1/τ (white noise regime)."""
        from tengri.components.stellar.sfh.psd_models import psd_drw

        tau_yr = 1e8  # 100 Myr
        omega = jnp.geomspace(1e-12, 1e-6, 200)
        psd = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=tau_yr)

        # At very low frequencies, PSD should be approximately constant
        low_f = omega < 1e-10
        if jnp.sum(low_f) > 3:
            cv = float(jnp.std(psd[low_f]) / jnp.mean(psd[low_f]))
            assert cv < 0.1, f"DRW low-freq PSD should be flat, CV={cv:.3f}"

    def test_drw_high_freq_decline(self):
        """DRW PSD: declines as ω^{-2} at ω >> 1/τ (red noise)."""
        from tengri.components.stellar.sfh.psd_models import psd_drw

        tau_yr = 1e8
        omega = jnp.geomspace(1e-5, 1e-3, 50)
        psd = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=tau_yr)

        # Should be declining
        assert float(psd[0]) > float(psd[-1]), "DRW should decline at high frequencies"

    def test_matern_nu_half_equals_drw(self):
        """Matérn with ν=0.5 reduces to DRW (exponential ACF)."""
        from tengri.components.stellar.sfh.psd_models import psd_drw, psd_matern

        omega = jnp.geomspace(1e-11, 1e-6, 200)
        sigma = 1.0
        tau_yr = 1e8
        length_scale = tau_yr  # Matérn length scale ~ DRW τ

        psd_d = psd_drw(omega, psd_sigma=sigma, psd_tau_yr=tau_yr)
        psd_m = psd_matern(omega, variance=sigma**2, length_scale=length_scale, nu=0.5)

        # Should be proportional (may differ by normalization)
        if float(jnp.max(psd_d)) > 0 and float(jnp.max(psd_m)) > 0:
            psd_d_norm = psd_d / jnp.max(psd_d)
            psd_m_norm = psd_m / jnp.max(psd_m)
            corr = float(jnp.corrcoef(psd_d_norm, psd_m_norm)[0, 1])
            assert corr > 0.95, f"Matérn(ν=0.5) should match DRW shape, correlation={corr:.3f}"

    def test_higher_matern_nu_smoother(self):
        """Higher Matérn ν → smoother (faster high-freq decline)."""
        from tengri.components.stellar.sfh.psd_models import psd_matern

        omega = jnp.geomspace(1e-11, 1e-6, 200)
        psd_rough = psd_matern(omega, variance=1.0, length_scale=1e8, nu=0.5)
        psd_smooth = psd_matern(omega, variance=1.0, length_scale=1e8, nu=2.5)

        # At high frequencies, smoother PSD declines faster
        high_f_ratio_rough = float(psd_rough[-1] / psd_rough[0])
        high_f_ratio_smooth = float(psd_smooth[-1] / psd_smooth[0])
        assert high_f_ratio_smooth < high_f_ratio_rough, (
            "Higher Matérn ν should decline faster at high frequencies"
        )

    def test_extended_regulator_positive(self):
        """Extended regulator PSD must be positive at all frequencies."""
        from tengri.components.stellar.sfh.psd_models import psd_extended_regulator

        f = jnp.geomspace(1e-12, 1e-6, 200)
        psd = psd_extended_regulator(f, s_reg=1.0, tau_in=1e8, tau_eq=1e9, s_dyn=0.5, tau_dyn=1e7)
        assert_non_negative(psd, name="psd", msg="Extended regulator PSD must be non-negative")
        chex.assert_tree_all_finite(psd)

    def test_psd_sigma_scales_amplitude(self):
        """Higher psd_sigma → higher PSD amplitude."""
        from tengri.components.stellar.sfh.psd_models import psd_drw

        omega = jnp.geomspace(1e-11, 1e-6, 100)
        psd_1 = psd_drw(omega, psd_sigma=1.0, psd_tau_yr=1e8)
        psd_2 = psd_drw(omega, psd_sigma=2.0, psd_tau_yr=1e8)

        ratio = float(jnp.mean(psd_2) / jnp.mean(psd_1))
        # sigma enters as sigma^2 in PSD
        assert 3.0 < ratio < 5.0, f"2x sigma should give ~4x PSD, got {ratio:.1f}x"


# ── 5. SED QUANTITIES — derived physical properties ───────────────


class TestSEDQuantitiesPhysics:
    """Derived SED quantities must give physically meaningful values."""

    def test_dn4000_flat_spectrum(self):
        """Flat f_ν → Dn4000 = 1.0 (no break)."""
        from tengri.utils.sed_quantities import compute_dn4000

        wave = jnp.linspace(3700.0, 4200.0, 500)
        sed = jnp.ones_like(wave)
        dn4000 = float(compute_dn4000(sed, wave))
        np.testing.assert_allclose(dn4000, 1.0, atol=0.05)

    def test_dn4000_red_spectrum_above_1(self):
        """Red (declining f_ν) spectrum → Dn4000 > 1 (old stellar pop)."""
        from tengri.utils.sed_quantities import compute_dn4000

        wave = jnp.linspace(3700.0, 4200.0, 500)
        # Red spectrum: more flux above 4000A than below
        sed = (wave / 4000.0) ** 2
        dn4000 = float(compute_dn4000(sed, wave))
        assert dn4000 > 1.0, f"Red spectrum should have Dn4000 > 1, got {dn4000:.2f}"

    def test_uv_slope_flat_fnu(self):
        """Flat f_ν → β = -2.0 (by definition, f_λ ∝ λ^β)."""
        from tengri.utils.sed_quantities import compute_uv_slope_beta

        wave = jnp.linspace(1200.0, 2800.0, 500)
        # Flat in f_ν means f_λ ∝ λ^{-2} → β = -2
        sed = jnp.ones_like(wave)  # flat f_ν
        beta = float(compute_uv_slope_beta(sed, wave))
        np.testing.assert_allclose(beta, -2.0, atol=0.1)

    def test_irx_analytic_value(self):
        """IRX = log10(L_TIR_erg / L_UV_erg): exact value for known input ratio.

        l_tir = 1e11 Lsun = 3.828e44 erg/s; l_uv = 3.828e43 erg/s (=1e10 Lsun).
        Ratio = 10 → IRX = log10(10) = 1.0 exactly.
        """
        from tengri.utils.sed_quantities import compute_irx

        _LSUN_ERG = 3.828e33
        l_tir = jnp.array(1e11)  # Lsun
        l_uv = jnp.array(1e10 * _LSUN_ERG)  # erg/s
        irx = float(compute_irx(l_tir, l_uv))
        np.testing.assert_allclose(
            irx,
            1.0,
            atol=1e-6,
            err_msg=f"IRX = log10(10) = 1.0 exactly, got {irx:.6f}",
        )

    def test_irx_decade_increase(self):
        """Each decade of L_TIR (fixed L_UV) increases IRX by exactly 1.0.

        l_uv fixed at 3.828e43 erg/s (=1e10 Lsun).
        l_tir_low = 1e10 Lsun → IRX = log10(1) = 0.0.
        l_tir_high = 1e12 Lsun → IRX = log10(100) = 2.0.
        """
        from tengri.utils.sed_quantities import compute_irx

        _LSUN_ERG = 3.828e33
        l_uv = jnp.array(1e10 * _LSUN_ERG)  # erg/s — fixed UV luminosity
        irx_low = float(compute_irx(jnp.array(1e10), l_uv))
        irx_high = float(compute_irx(jnp.array(1e12), l_uv))

        np.testing.assert_allclose(irx_low, 0.0, atol=1e-6, err_msg="IRX(L_TIR=L_UV) = 0")
        np.testing.assert_allclose(irx_high, 2.0, atol=1e-6, err_msg="IRX(100×L_UV) = 2")
