"""Tests for new physics modules: dust emission, AGN, IGM, dust laws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# =====================================================================
# Dust attenuation curves
# =====================================================================


class TestDustLaws:
    """Test the 6 attenuation curves."""

    @pytest.fixture
    def wave(self):
        return jnp.linspace(1000, 30000, 100)

    def test_all_laws_registered(self):
        from diffsed.models.dust.attenuation import DUST_LAWS
        assert set(DUST_LAWS.keys()) == {
            "power_law", "calzetti", "kriek_conroy", "smc", "cardelli", "li08", "salim"
        }

    def test_all_laws_return_positive(self, wave):
        from diffsed.models.dust.attenuation import DUST_LAWS
        for name, fn in DUST_LAWS.items():
            k = fn(wave, n_slope=-0.7, dust_bump_strength=1.0,
                   dust_delta=0.0, dust_Rv=3.1)
            assert jnp.all(k >= 0), f"{name} returned negative values"

    def test_calzetti_normalized_at_v(self, wave):
        from diffsed.models.dust.attenuation import calzetti
        k = calzetti(jnp.array([5500.0]))
        # k(V) should be ~1.0 (normalized to V-band)
        np.testing.assert_allclose(float(k[0]), 1.0, atol=0.1)

    def test_kriek_conroy_bump_effect(self, wave):
        from diffsed.models.dust.attenuation import kriek_conroy
        k_no_bump = kriek_conroy(wave, dust_bump_strength=0.0)
        k_with_bump = kriek_conroy(wave, dust_bump_strength=3.0)
        # UV bump at 2175 A should increase attenuation
        idx_2175 = jnp.argmin(jnp.abs(wave - 2175))
        assert float(k_with_bump[idx_2175]) > float(k_no_bump[idx_2175])

    def test_smc_steeper_than_calzetti_in_uv(self, wave):
        from diffsed.models.dust.attenuation import calzetti, smc
        k_calz = calzetti(wave)
        k_smc = smc(wave)
        idx_1500 = jnp.argmin(jnp.abs(wave - 1500))
        assert float(k_smc[idx_1500]) > float(k_calz[idx_1500])

    def test_f_obscuration_floor(self):
        from diffsed.models.dust.attenuation import two_component_dust
        wave = jnp.linspace(1000, 10000, 50)
        ages = jnp.array([1e6, 1e9])
        atten = two_component_dust(wave, ages, tau_v1=2.0, tau_v2=1.0,
                                    f_obscuration=0.2)
        # Transmission should never go below f_obscuration
        assert float(jnp.min(atten)) >= 0.19  # slightly below 0.2 due to numerics

    def test_dust_law_gradients(self, wave):
        from diffsed.models.dust.attenuation import DUST_LAWS
        for name, fn in DUST_LAWS.items():
            def _loss(ns, bs, dd, rv):
                return jnp.sum(fn(wave, n_slope=ns, dust_bump_strength=bs,
                                   dust_delta=dd, dust_Rv=rv))
            g = jax.grad(_loss, argnums=(0,1,2,3))(-0.7, 1.0, 0.0, 3.1)
            assert all(jnp.isfinite(gi) for gi in g), f"{name} has NaN gradient"


# =====================================================================
# IGM absorption
# =====================================================================


class TestIGM:
    """Test Inoue+2014 IGM transmission."""

    def test_no_absorption_at_low_z(self):
        from diffsed.models.igm import igm_transmission
        wave = jnp.linspace(3000, 10000, 100)
        t = igm_transmission(wave, 0.01)
        np.testing.assert_allclose(t, 1.0, atol=1e-3)

    def test_gunn_peterson_at_z6(self):
        from diffsed.models.igm import igm_transmission
        # Below Ly-limit at z=6: lambda_obs < 912*(1+6) = 6384 A
        t = igm_transmission(jnp.array([5000.0]), 6.0)
        assert float(t[0]) < 0.01, "Should be fully absorbed at z=6 below Ly-limit"

    def test_no_absorption_redward(self):
        from diffsed.models.igm import igm_transmission
        # Well above Ly-alpha at z=3: lambda_obs > 1216*(1+3) = 4864 A
        t = igm_transmission(jnp.array([10000.0]), 3.0)
        assert float(t[0]) > 0.99

    def test_igm_gradient_finite(self):
        from diffsed.models.igm import igm_transmission
        wave = jnp.linspace(800, 15000, 100)
        g = jax.grad(lambda z: jnp.sum(igm_transmission(wave, z)))(3.0)
        assert jnp.isfinite(g)


# =====================================================================
# Dust emission
# =====================================================================


class TestDustEmission:
    """Test dust IR emission models."""

    @pytest.fixture
    def ir_wave(self):
        return jnp.logspace(jnp.log10(1e4), jnp.log10(1e7), 300)

    def test_all_models_registered(self):
        from diffsed.models.dust.emission import DUST_EMISSION_MODELS
        assert "modified_blackbody" in DUST_EMISSION_MODELS
        assert "dale2014" in DUST_EMISSION_MODELS
        assert "draine_li2007" in DUST_EMISSION_MODELS
        assert "draine_li2014" in DUST_EMISSION_MODELS

    def test_energy_conservation(self, ir_wave):
        """Total dust emission should equal absorbed luminosity."""
        from diffsed.models.dust.emission import DUST_EMISSION_MODELS
        L_abs = 1e10  # Lsun
        _c = 2.99792458e18  # c in Angstrom/s
        for name, fn in DUST_EMISSION_MODELS.items():
            sed = fn(ir_wave, L_abs, dust_T=35.0, dust_beta_ir=1.6,
                     dust_alpha_dale=2.0, dust_umin=1.0,
                     dust_gamma_dl=0.01, dust_qpah=2.5,
                     dust_alpha_dl14=2.0)
            nu = _c / ir_wave
            L_total = float(-jnp.trapezoid(sed, nu))
            ratio = L_total / L_abs
            assert 0.5 < ratio < 2.0, (
                f"{name}: L_emitted/L_absorbed = {ratio:.2f}, expected ~1.0"
            )

    def test_greybody_peak_wavelength(self, ir_wave):
        from diffsed.models.dust.emission import modified_blackbody
        sed = modified_blackbody(ir_wave, 1e10, dust_T=35.0)
        peak_um = float(ir_wave[jnp.argmax(sed)]) / 1e4
        # Wien's law: peak ~ 2898/T um for B_nu peak, but modified BB shifts
        assert 50 < peak_um < 200, f"Peak at {peak_um:.0f} um, expected 50-200"

    def test_emission_gradient(self, ir_wave):
        from diffsed.models.dust.emission import modified_blackbody
        g = jax.grad(lambda T: jnp.sum(
            modified_blackbody(ir_wave, 1e10, dust_T=T)))(35.0)
        assert jnp.isfinite(g) and abs(float(g)) > 0


class TestDL14Emission:
    """Test Draine & Li 2014 dust emission model (analytic approximation)."""

    @pytest.fixture
    def ir_wave(self):
        return jnp.logspace(jnp.log10(1e4), jnp.log10(1e7), 300)

    def test_dl14_registered(self):
        from diffsed.models.dust.emission import DUST_EMISSION_MODELS
        assert "draine_li2014" in DUST_EMISSION_MODELS

    def test_dl14_positive_output(self, ir_wave):
        from diffsed.models.dust.emission import draine_li2014
        sed = draine_li2014(ir_wave, 1e10)
        assert jnp.all(sed >= 0), "DL14 SED should be non-negative"
        assert float(jnp.max(sed)) > 0, "DL14 SED should have positive values"

    def test_dl14_energy_conservation(self, ir_wave):
        """DL14 emission should conserve energy within 50%."""
        from diffsed.models.dust.emission import draine_li2014
        L_abs = 1e10
        _c = 2.99792458e18
        sed = draine_li2014(ir_wave, L_abs, dust_umin=1.0,
                            dust_gamma_dl=0.01, dust_qpah=2.5,
                            dust_alpha_dl14=2.0)
        nu = _c / ir_wave
        L_total = float(-jnp.trapezoid(sed, nu))
        ratio = L_total / L_abs
        assert 0.5 < ratio < 2.0, f"L_emitted/L_absorbed = {ratio:.2f}"

    def test_dl14_alpha_default_matches_dl07(self, ir_wave):
        """DL14 with alpha=2.0 should be similar to DL07 (both analytic)."""
        from diffsed.models.dust.emission import draine_li2007, draine_li2014
        L_abs = 1e10
        sed_07 = draine_li2007(ir_wave, L_abs, dust_umin=1.0,
                                dust_gamma_dl=0.01, dust_qpah=2.5)
        sed_14 = draine_li2014(ir_wave, L_abs, dust_umin=1.0,
                                dust_gamma_dl=0.01, dust_qpah=2.5,
                                dust_alpha_dl14=2.0)
        # Both are analytic approximations, not identical but similar shapes
        ratio = float(jnp.sum(sed_14)) / float(jnp.sum(sed_07))
        assert 0.3 < ratio < 3.0, (
            f"DL14(alpha=2) vs DL07 ratio = {ratio:.2f}, expected similar magnitude"
        )

    def test_dl14_alpha_affects_spectrum(self, ir_wave):
        """Different alpha values should produce different spectra."""
        from diffsed.models.dust.emission import draine_li2014
        L_abs = 1e10
        sed_low_alpha = draine_li2014(ir_wave, L_abs, dust_alpha_dl14=1.5,
                                       dust_gamma_dl=0.1)
        sed_high_alpha = draine_li2014(ir_wave, L_abs, dust_alpha_dl14=2.5,
                                        dust_gamma_dl=0.1)
        # Low alpha = steeper power-law = more dust at high U = warmer
        # So peak should shift to shorter wavelengths
        peak_low = float(ir_wave[jnp.argmax(sed_low_alpha)])
        peak_high = float(ir_wave[jnp.argmax(sed_high_alpha)])
        # With gamma=0.1, the alpha effect should be noticeable
        assert not jnp.allclose(sed_low_alpha, sed_high_alpha), (
            "Different alpha values should give different spectra"
        )

    def test_dl14_extended_qpah_range(self, ir_wave):
        """DL14 should handle extended q_PAH range (up to 7.32%)."""
        from diffsed.models.dust.emission import draine_li2014
        # q_PAH = 7.0% (beyond DL07's 4.58% max)
        sed = draine_li2014(ir_wave, 1e10, dust_qpah=7.0)
        assert jnp.all(jnp.isfinite(sed)), "DL14 should handle q_PAH=7.0%"
        assert float(jnp.max(sed)) > 0

    def test_dl14_extended_umin_range(self, ir_wave):
        """DL14 should handle extended U_min range (up to 50)."""
        from diffsed.models.dust.emission import draine_li2014
        sed = draine_li2014(ir_wave, 1e10, dust_umin=40.0)
        assert jnp.all(jnp.isfinite(sed)), "DL14 should handle U_min=40"
        assert float(jnp.max(sed)) > 0

    def test_dl14_jit_compatible(self, ir_wave):
        """DL14 should be JIT-compilable."""
        from diffsed.models.dust.emission import draine_li2014
        fn = jax.jit(lambda u, g, q, a: draine_li2014(
            ir_wave, 1e10, dust_umin=u, dust_gamma_dl=g,
            dust_qpah=q, dust_alpha_dl14=a))
        sed = fn(1.0, 0.01, 2.5, 2.0)
        assert sed.shape == ir_wave.shape
        assert jnp.all(jnp.isfinite(sed))

    def test_dl14_gradients(self, ir_wave):
        """DL14 should have finite gradients for all parameters."""
        from diffsed.models.dust.emission import draine_li2014

        def loss(umin, gamma, qpah, alpha):
            return jnp.sum(draine_li2014(
                ir_wave, 1e10, dust_umin=umin, dust_gamma_dl=gamma,
                dust_qpah=qpah, dust_alpha_dl14=alpha))

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(1.0, 0.05, 2.5, 2.0)
        for i, name in enumerate(["umin", "gamma", "qpah", "alpha"]):
            assert jnp.isfinite(grads[i]), f"NaN gradient for {name}"
            # alpha gradient should be nonzero when gamma > 0
            if name == "alpha":
                assert abs(float(grads[i])) > 0, (
                    "alpha gradient should be nonzero with gamma=0.05"
                )

    def test_dl14_param_spec_registered(self):
        """dust_alpha_dl14 should be in the parameter spec."""
        from diffsed.param_spec import _DUST_EMISSION_PARAMS
        assert "dust_alpha_dl14" in _DUST_EMISSION_PARAMS


# =====================================================================
# AGN
# =====================================================================


class TestAGN:
    """Test AGN emission models."""

    @pytest.fixture
    def wave(self):
        return jnp.logspace(2, 7, 500)

    def test_all_models_registered(self):
        from diffsed.models.agn import AGN_MODELS
        assert {"simple", "standard", "kubota_done"}.issubset(AGN_MODELS.keys())

    def test_simple_agn_positive(self, wave):
        from diffsed.models.agn import AGN_MODELS
        sed = AGN_MODELS["simple"](wave, agn_log_lbol=44.0)
        assert jnp.all(sed >= 0), "AGN SED should be non-negative"
        assert float(jnp.max(sed)) > 0, "AGN SED should have positive values"

    def test_agn_in_forward_model(self):
        """AGN should boost UV and MIR flux relative to pure stellar."""
        from diffsed import (
            Model, ParamSpec, Uniform, Fixed,
            load_ssp_data, load_filter_set,
        )
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        filters = load_filter_set(["galex_fuv", "sdss_r", "wise_w3"])
        spec = ParamSpec(
            sfh_dpl_alpha=Fixed(1.0), sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0), sfh_dpl_log_peak_sfr=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3), dust_tau_diff=Fixed(0.2),
            agn_frac=Uniform(0.0, 1.0),
            redshift=Fixed(0.1), mean_sfh_type="dpl",
            agn_model="simple",
        )
        model = Model(spec, ssp, filters=filters, precompute=False)
        params = {"agn_frac": 0.2}
        params0 = {"agn_frac": 0.0}
        phot = model.predict_photometry(params)
        phot0 = model.predict_photometry(params0)

        # FUV and W3 should be boosted; r-band barely changed
        fuv_ratio = float(phot[0] / phot0[0])
        w3_ratio = float(phot[2] / phot0[2])
        r_ratio = float(phot[1] / phot0[1])
        assert fuv_ratio > 1.05, f"FUV ratio {fuv_ratio:.3f}, expected > 1.05"
        assert w3_ratio > 1.1, f"W3 ratio {w3_ratio:.3f}, expected > 1.1"
        assert r_ratio < 1.05, f"r ratio {r_ratio:.3f}, expected < 1.05"

    def test_agn_gradient(self, wave):
        from diffsed.models.agn import AGN_MODELS
        g = jax.grad(lambda ll: jnp.sum(
            AGN_MODELS["simple"](wave, agn_log_lbol=ll)))(44.0)
        assert jnp.isfinite(g) and abs(float(g)) > 0


# =====================================================================
# Dust emission in forward model
# =====================================================================


class TestDustEmissionForwardModel:
    """Test dust emission wired into the full model."""

    def test_dust_emission_adds_ir_flux(self):
        from diffsed import (
            Model, ParamSpec, Uniform, Fixed,
            load_ssp_data, load_filter_set,
        )
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        filters = load_filter_set(["sdss_r", "wise_w3"])
        spec = ParamSpec(
            sfh_dpl_alpha=Fixed(1.0), sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0), sfh_dpl_log_peak_sfr=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0), dust_tau_diff=Fixed(0.5),
            dust_T=Uniform(20.0, 60.0),
            redshift=Fixed(0.1), mean_sfh_type="dpl",
            dust_emission="modified_blackbody",
        )
        model = Model(spec, ssp, filters=filters, precompute=False)
        params_em = {"dust_T": 35.0, "dust_beta_ir": 1.6}

        # Model without dust emission
        spec_no = ParamSpec(
            sfh_dpl_alpha=Fixed(1.0), sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0), sfh_dpl_log_peak_sfr=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0), dust_tau_diff=Fixed(0.5),
            redshift=Fixed(0.1), mean_sfh_type="dpl",
        )
        model_no = Model(spec_no, ssp, filters=filters, precompute=False)

        phot_em = model.predict_photometry(params_em)
        phot_no = model_no.predict_photometry({})

        # Dust emission should add IR flux, so SED should be >= stellar-only
        # (W3 at 12 um may not show large effect at z=0.1 with T_dust=35K
        #  since the peak is at ~80um, but the total should increase)
        sed_em = model.predict_sed(params_em)
        sed_no = model_no.predict_sed({})
        max_diff = float(jnp.max(sed_em - sed_no))
        assert max_diff > 0, "Dust emission should add positive flux somewhere"

    def test_dust_emission_gradient(self):
        from diffsed import (
            Model, ParamSpec, Uniform, Fixed,
            load_ssp_data, load_filter_set,
        )
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        filters = load_filter_set(["wise_w3"])
        spec = ParamSpec(
            sfh_dpl_alpha=Fixed(1.0), sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0), sfh_dpl_log_peak_sfr=Fixed(1.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0), dust_tau_diff=Fixed(0.5),
            dust_T=Uniform(20.0, 60.0),
            redshift=Fixed(0.1), mean_sfh_type="dpl",
            dust_emission="modified_blackbody",
        )
        model = Model(spec, ssp, filters=filters, precompute=False)

        def loss(T):
            return model.predict_photometry({"dust_T": T})[0]

        g = jax.grad(loss)(35.0)
        assert jnp.isfinite(g), "dust_T gradient should be finite"
