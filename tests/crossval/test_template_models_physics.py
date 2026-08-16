# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation for template-based dust emission models.

Tests that the REAL template-loaded models (not analytic fallbacks)
produce physically correct SEDs verified against the original papers.
These tests will SKIP if template data files are missing.

References
----------
- Boquien & Salim 2021, A&A, 653, A149 — BOSA templates
- da Cunha et al. 2008, MNRAS, 388, 1595 — MAGPHYS
- Dale et al. 2014, ApJ, 784, 83 — Dale IR templates
- Draine & Li 2007, ApJ, 657, 810 — DL07 dust models
- Hensley & Draine 2023, ApJ, 948, 55 — Astrodust+PAH
- Jones et al. 2017, A&A, 602, A46 — THEMIS/DustEM
- Smith et al. 2007, ApJ, 656, 770 — PAH Drude profiles
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _template_exists(filename: str) -> bool:
    return (_DATA_DIR / filename).is_file()


def _no_fallback_warning(recwarn) -> bool:
    """Check that no fallback warnings were issued."""
    return not any("fallback" in str(w.message).lower() for w in recwarn)


# ── DL07 — Draine & Li 2007 ───────────────────────────────────────

_DL07_EXISTS = _template_exists("dl07_templates_v2.h5") or _template_exists("dl07_templates.h5")


@pytest.mark.skipif(not _DL07_EXISTS, reason="DL07 templates not found")
class TestDL07TemplatePhysics:
    """Draine & Li 2007 template-based model physics validation."""

    def test_loads_real_templates(self):
        """SEDModel must load templates, not fall back to analytic."""
        from tengri.components.dust.emission import draine_li2007

        wave = jnp.logspace(4, 7, 500)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sed = draine_li2007(wave, 1e10)
            assert _no_fallback_warning(w), "DL07 fell back to analytic!"
        assert float(jnp.max(sed)) > 0

    def test_energy_conservation(self):
        """Integral of L_nu dnu must equal L_absorbed."""
        from tengri.components.dust.emission import draine_li2007

        wave = jnp.logspace(4, 8, 3000)  # wide range for full integral
        l_absorbed = 1e10
        sed = draine_li2007(wave, l_absorbed)

        c_aa = 2.99792458e18
        nu = c_aa / wave
        l_bol = float(-jnp.trapezoid(sed, nu))

        np.testing.assert_allclose(
            l_bol, l_absorbed, rtol=0.10, err_msg=f"DL07 energy: {l_bol:.3e} vs {l_absorbed:.3e}"
        )

    def test_higher_qpah_stronger_mir(self):
        """Higher q_PAH → stronger MIR emission (PAH features).

        DL07 q_PAH ranges from 0.47% to 4.58%. Higher values mean
        more PAH mass → stronger 7.7 um feature.
        """
        from tengri.components.dust.emission import draine_li2007

        wave = jnp.logspace(np.log10(5e4), np.log10(2e5), 500)  # 5-20 um
        sed_low = draine_li2007(wave, 1e10, dust_qpah=0.5)
        sed_high = draine_li2007(wave, 1e10, dust_qpah=4.0)

        # 7.7 um region (7-8.5 um = 7e4-8.5e4 A)
        pah_mask = (wave > 7e4) & (wave < 8.5e4)
        pah_low = float(jnp.mean(sed_low[pah_mask]))
        pah_high = float(jnp.mean(sed_high[pah_mask]))

        assert pah_high > pah_low, (
            f"Higher qPAH should strengthen 7.7um: qPAH=0.5→{pah_low:.3e}, qPAH=4→{pah_high:.3e}"
        )

    def test_higher_umin_warmer_sed(self):
        """Higher U_min → warmer dust → peak at shorter wavelength."""
        from tengri.components.dust.emission import draine_li2007

        wave = jnp.logspace(4, 7.5, 1000)

        sed_cold = draine_li2007(wave, 1e10, dust_umin=0.5)
        sed_warm = draine_li2007(wave, 1e10, dust_umin=10.0)

        peak_cold = float(wave[jnp.argmax(sed_cold * wave)])
        peak_warm = float(wave[jnp.argmax(sed_warm * wave)])

        assert peak_warm < peak_cold, (
            f"Higher Umin should peak shorter: "
            f"U=0.5→{peak_cold / 1e4:.0f}um, U=10→{peak_warm / 1e4:.0f}um"
        )

    def test_pah_features_present(self):
        """DL07 must show PAH features at 6.2, 7.7, 11.3 um."""
        from tengri.components.dust.emission import draine_li2007

        wave = jnp.logspace(np.log10(3e4), np.log10(2e5), 2000)  # 3-20 um
        sed = draine_li2007(wave, 1e10, dust_qpah=3.0)

        wave_um = np.asarray(wave) / 1e4
        sed_arr = np.asarray(sed)

        # Check 7.7 um feature exceeds continuum
        pah_region = np.abs(wave_um - 7.7) < 0.5
        continuum_region = (wave_um > 9.5) & (wave_um < 10.5)

        pah_flux = np.max(sed_arr[pah_region])
        cont_flux = np.median(sed_arr[continuum_region])

        assert pah_flux > 2 * cont_flux, (
            f"7.7um PAH should exceed continuum: PAH={pah_flux:.3e}, cont={cont_flux:.3e}"
        )

    def test_gamma_zero_no_pdr(self):
        """gamma=0 means all dust at single U_min, no high-U PDR component."""
        from tengri.components.dust.emission import draine_li2007

        wave = jnp.logspace(4, 7, 500)

        sed_g0 = draine_li2007(wave, 1e10, dust_gamma_dl=0.0, dust_umin=1.0)
        sed_g1 = draine_li2007(wave, 1e10, dust_gamma_dl=0.5, dust_umin=1.0)

        # Different gamma should give different SED shapes
        ratio = np.asarray(sed_g1) / np.maximum(np.asarray(sed_g0), 1e-50)
        nonzero = np.asarray(sed_g0) > 1e-30
        assert not np.allclose(ratio[nonzero], 1.0, rtol=0.01), (
            "gamma should change SED shape (PDR component adds warm emission)"
        )


# ── ASTRODUST — Hensley & Draine 2023 ─────────────────────────────

_AD_EXISTS = _template_exists("astrodust_templates.h5")


@pytest.mark.skipif(not _AD_EXISTS, reason="Astrodust templates not found")
class TestAstrodustTemplatePhysics:
    """Hensley & Draine 2023 Astrodust template-based model validation."""

    def test_loads_real_templates(self):
        """Must load templates, not analytic fallback."""
        from tengri.components.dust.emission import astrodust

        wave = jnp.logspace(4, 7, 500)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sed = astrodust(wave, 1e10)
            assert _no_fallback_warning(w), "Astrodust fell back to analytic!"

    def test_energy_conservation(self):
        """Energy must be conserved to within 10%."""
        from tengri.components.dust.emission import astrodust

        wave = jnp.logspace(3.5, 8, 3000)
        l_absorbed = 1e10
        sed = astrodust(wave, l_absorbed)
        c_aa = 2.99792458e18
        nu = c_aa / wave
        l_bol = float(-jnp.trapezoid(sed, nu))
        np.testing.assert_allclose(l_bol, l_absorbed, rtol=0.10)

    def test_qpah_does_not_apply_to_astrodust(self):
        """Astrodust has no qPAH dimension — the published grid has one PAH population.

        From the module docstring of ``astrodust_hd23``: *"Single fiducial PAH
        size distribution and ionization fraction (Hensley & Draine 2022) —
        there is no published (qpah, size) grid for this model."* So
        ``dust_qpah`` is inapplicable here, not accidentally dropped, and the
        output is identical to full precision across its whole range.

        This test previously asserted that higher qPAH strengthens the 7.7 um
        feature, i.e. asserted a dimension the model does not have (#1728). It
        is inverted rather than deleted because the invariance is worth
        pinning: ``astrodust`` takes ``**kwargs``, so a caller can pass
        ``dust_qpah`` and have it silently accepted. Anyone who later wires a
        qPAH grid in should see this test fail and update it deliberately.
        """
        from tengri.components.dust.emission import astrodust

        wave = jnp.logspace(np.log10(5e4), np.log10(1.5e5), 500)
        seds = [np.asarray(astrodust(wave, 1e10, dust_qpah=q)) for q in (0.5, 2.0, 4.6)]

        # Not bit-equality: a handful of points drift by ~7e-15 relative, which
        # is float noise, not a qPAH dependence. Any real grid interpolation
        # would move the 7.7 um feature by percent, not by machine epsilon.
        for other in seds[1:]:
            np.testing.assert_allclose(
                seds[0],
                other,
                rtol=1e-10,
                atol=0.0,
                err_msg="astrodust has no qPAH grid; the SED must not depend on dust_qpah",
            )

    def test_umin_is_the_knob_astrodust_does_respond_to(self):
        """The radiation-field intensity is astrodust's real free parameter.

        Guards the test above: an invariance test passes trivially if the model
        ignores *everything*. This pins that the component is live.
        """
        from tengri.components.dust.emission import astrodust

        wave = jnp.logspace(np.log10(3e4), np.log10(1e7), 800)
        cool = np.asarray(astrodust(wave, 1e10, dust_umin=0.5))
        hot = np.asarray(astrodust(wave, 1e10, dust_umin=5.0))

        assert not np.allclose(cool, hot, rtol=1e-6, atol=0.0), (
            "astrodust must respond to dust_umin; identical SEDs mean the "
            "template interpolation is not wired"
        )

    def test_pah_features_present(self):
        """Astrodust must show PAH emission features at 7.7 um."""
        from tengri.components.dust.emission import astrodust

        wave = jnp.logspace(np.log10(3e4), np.log10(2e5), 2000)
        sed = astrodust(wave, 1e10, dust_qpah=3.5)
        wave_um = np.asarray(wave) / 1e4
        sed_arr = np.asarray(sed)

        pah_region = np.abs(wave_um - 7.7) < 0.5
        cont_region = (wave_um > 9.5) & (wave_um < 10.5)
        pah_flux = np.max(sed_arr[pah_region])
        cont_flux = np.median(sed_arr[cont_region])

        assert pah_flux > 1.5 * cont_flux, (
            f"Astrodust 7.7um PAH should exceed continuum: "
            f"PAH={pah_flux:.3e}, cont={cont_flux:.3e}"
        )


# ── THEMIS — Jones et al. 2017 ────────────────────────────────────

_TH_EXISTS = _template_exists("themis_templates.h5")


@pytest.mark.skipif(not _TH_EXISTS, reason="THEMIS templates not found")
class TestTHEMISTemplatePhysics:
    """Jones+2017 THEMIS template-based model validation."""

    def test_loads_real_templates(self):
        """Must load templates, not analytic fallback."""
        from tengri.components.dust.emission import themis

        wave = jnp.logspace(4, 7, 500)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sed = themis(wave, 1e10)
            assert _no_fallback_warning(w), "THEMIS fell back to analytic!"

    def test_energy_conservation(self):
        """Energy must be conserved to within 10%."""
        from tengri.components.dust.emission import themis

        wave = jnp.logspace(3.5, 8, 3000)
        l_absorbed = 1e10
        sed = themis(wave, l_absorbed)
        c_aa = 2.99792458e18
        nu = c_aa / wave
        l_bol = float(-jnp.trapezoid(sed, nu))
        np.testing.assert_allclose(l_bol, l_absorbed, rtol=0.10)

    def test_qhac_affects_aromatic_features(self):
        """qhac controls a-C(:H) aromatic emission strength.

        Jones+2017: small a-C nanoparticles (<20nm) are aromatic
        and produce PAH-like MIR features. Higher qhac = more
        small aromatic grains.
        """
        from tengri.components.dust.emission import themis

        wave = jnp.logspace(np.log10(5e4), np.log10(1.5e5), 500)
        sed_low = themis(wave, 1e10, dust_qhac=0.05)
        sed_high = themis(wave, 1e10, dust_qhac=0.25)

        pah_mask = (wave > 7e4) & (wave < 8.5e4)
        assert float(jnp.mean(sed_high[pah_mask])) > float(jnp.mean(sed_low[pah_mask])), (
            "Higher qhac should strengthen aromatic MIR features"
        )


# ── BOSA — Boquien & Salim 2021 ───────────────────────────────────

_BO_EXISTS = _template_exists("bosa_templates.h5")


@pytest.mark.skipif(not _BO_EXISTS, reason="BOSA templates not found")
class TestBOSATemplatePhysics:
    """Boquien & Salim 2021 BOSA template-based model validation."""

    def test_loads_real_templates(self):
        """Must load templates, not analytic fallback."""
        from tengri.components.dust.emission import bosa

        wave = jnp.logspace(4, 7, 500)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sed = bosa(wave, 1e10)
            assert _no_fallback_warning(w), "BOSA fell back to analytic!"

    def test_energy_conservation(self):
        """Total IR emission must equal L_absorbed."""
        from tengri.components.dust.emission import bosa

        wave = jnp.logspace(3.5, 8, 3000)
        l_absorbed = 1e10
        sed = bosa(wave, l_absorbed, dust_log_ssfr=-10.0)
        c_aa = 2.99792458e18
        nu = c_aa / wave
        l_bol = float(-jnp.trapezoid(sed, nu))
        np.testing.assert_allclose(l_bol, l_absorbed, rtol=0.10)

    def test_higher_ssfr_warmer_peak(self):
        """Higher sSFR → warmer dust → peak at shorter wavelength.

        Boquien & Salim 2021: sSFR is an independent parameter
        beyond L_TIR that captures the temperature-SFR connection.
        """
        from tengri.components.dust.emission import bosa

        wave = jnp.logspace(4, 7.5, 1000)

        sed_quiescent = bosa(wave, 1e10, dust_log_ssfr=-11.5)
        sed_starburst = bosa(wave, 1e10, dust_log_ssfr=-9.0)

        peak_q = float(wave[jnp.argmax(sed_quiescent * wave)])
        peak_sb = float(wave[jnp.argmax(sed_starburst * wave)])

        assert peak_sb < peak_q, (
            f"Higher sSFR should peak shorter: "
            f"sSFR=-11.5→{peak_q / 1e4:.0f}um, sSFR=-9→{peak_sb / 1e4:.0f}um"
        )

    def test_linear_scaling_with_luminosity(self):
        """Doubling L_absorbed should approximately double emission."""
        from tengri.components.dust.emission import bosa

        wave = jnp.logspace(4, 7, 500)
        sed1 = bosa(wave, 1e10, dust_log_ssfr=-10.0)
        sed2 = bosa(wave, 2e10, dust_log_ssfr=-10.0)

        # BOSA templates are parameterized on log L_TIR, so scaling
        # is not exact (different template at different L_TIR), but
        # within the same L_TIR bin it should be close
        ratio = float(jnp.sum(sed2)) / float(jnp.sum(sed1))
        assert 1.5 < ratio < 2.5, f"BOSA luminosity scaling: ratio={ratio:.2f}, expected ~2"


# ── Dale 2014 templates ───────────────────────────────────────────

_DA_EXISTS = _template_exists("dale2014_templates.h5")


@pytest.mark.skipif(not _DA_EXISTS, reason="Dale2014 templates not found")
class TestDale2014TemplatePhysics:
    """Dale et al. 2014 template-based model validation."""

    def test_loads_real_templates(self):
        """Must load templates, not analytic fallback."""
        from tengri.components.dust.emission import dale2014

        wave = jnp.logspace(4, 7, 500)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sed = dale2014(wave, 1e10)
            assert _no_fallback_warning(w), "Dale2014 fell back to analytic!"

    def test_energy_conservation(self):
        """Energy must be conserved."""
        from tengri.components.dust.emission import dale2014

        wave = jnp.logspace(3.5, 8, 3000)
        l_absorbed = 1e10
        sed = dale2014(wave, l_absorbed)
        c_aa = 2.99792458e18
        nu = c_aa / wave
        l_bol = float(-jnp.trapezoid(sed, nu))
        np.testing.assert_allclose(l_bol, l_absorbed, rtol=0.10)

    def test_alpha_dale_affects_sed_shape(self):
        """dust_alpha_dale controls warm/cool dust balance.

        Higher alpha → more weight at low U → cooler SED.
        Parameter name is dust_alpha_dale (not dust_alpha).
        """
        from tengri.components.dust.emission import dale2014

        wave = jnp.logspace(4, 7.5, 1000)

        sed_warm = dale2014(wave, 1e10, dust_alpha_dale=0.5)
        sed_cool = dale2014(wave, 1e10, dust_alpha_dale=3.5)

        # SEDs should be different
        nonzero = np.asarray(sed_warm) > 1e-30
        ratio = np.asarray(sed_cool[nonzero]) / np.maximum(np.asarray(sed_warm[nonzero]), 1e-50)
        assert not np.allclose(ratio, 1.0, rtol=0.01), "dust_alpha_dale should change SED shape"

        # MIR (5-30 um) vs FIR (80-300 um) ratio
        mir_mask = (wave > 5e4) & (wave < 3e5)
        fir_mask = (wave > 8e5) & (wave < 3e6)

        mir_warm = float(jnp.sum(sed_warm[mir_mask]))
        fir_warm = float(jnp.sum(sed_warm[fir_mask]))
        mir_cool = float(jnp.sum(sed_cool[mir_mask]))
        fir_cool = float(jnp.sum(sed_cool[fir_mask]))

        ratio_warm = mir_warm / max(fir_warm, 1e-50)
        ratio_cool = mir_cool / max(fir_cool, 1e-50)

        assert ratio_warm > ratio_cool, (
            f"Higher alpha_dale should reduce MIR/FIR ratio: "
            f"α=0.5→{ratio_warm:.3f}, α=3.5→{ratio_cool:.3f}"
        )


# ── MAGPHYS — da Cunha+2008 (NOT IMPLEMENTED) ─────────────────────


@pytest.mark.skip(
    reason=(
        "#1728: magphys_dc08 does not exist. Nothing MAGPHYS-related is in "
        "components.dust.emission_templates or in list_dust_emission_models(); "
        "these nine tests and tests/crossval/test_magphys_crossval.py (already "
        "module-skipped for the same reason) describe a model that was never "
        "built. Kept rather than deleted because they are a usable spec for it: "
        "the xi sum constraint, the four-component decomposition and the "
        "temperature ordering are all da Cunha+2008 Eq. 6 and would be the right "
        "tests on the day it lands. Implement or delete — but do not leave a "
        "filename implying coverage that does not exist."
    )
)
class TestMagphysPhysics:
    """MAGPHYS 4-component dust emission physics from da Cunha+2008."""

    def test_xi_sum_constraint(self):
        """xi_PAH + xi_MIR + xi_W must equal 1 (residual goes to cold).

        da Cunha+2008 Eq. 6: the fractional luminosities sum to unity
        for each component (BC and ISM separately).
        """
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(4, 7, 500)

        # Set fractions that sum to 1: PAH=0.1, MIR=0.2, warm=0.7, cold=0
        sed = magphys_dc08(wave, 1e10, dust_xi_pah=0.1, dust_xi_mir=0.2, dust_xi_warm=0.7)
        assert float(jnp.max(sed)) > 0

    def test_energy_balance_strict(self):
        """Total L_IR must equal L_absorbed to within 5% (da Cunha+2008)."""
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(3.5, 8, 3000)
        l_absorbed = 1e10
        sed = magphys_dc08(wave, l_absorbed)

        c_aa = 2.99792458e18
        nu = c_aa / wave
        l_bol = float(-jnp.trapezoid(sed, nu))

        np.testing.assert_allclose(
            l_bol,
            l_absorbed,
            rtol=0.05,
            err_msg=f"MAGPHYS energy: {l_bol:.3e} vs {l_absorbed:.3e}",
        )

    def test_pah_features_at_correct_wavelengths(self):
        """PAH features must appear at 3.3, 6.2, 7.7, 8.6, 11.3, 12.7 um.

        Smith+2007: these 6 features are the dominant PAH emission bands.
        """
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(np.log10(2e4), np.log10(2e5), 3000)  # 2-20 um
        sed = magphys_dc08(
            wave,
            1e10,
            dust_xi_pah=1.0,
            dust_xi_mir=0.0,
            dust_xi_warm=0.0,
        )

        wave_um = np.asarray(wave) / 1e4
        sed_arr = np.asarray(sed)

        # Check the two strongest PAH features (7.7, 11.3 um)
        # 6.2 um is weaker and may not exceed nearby continuum
        for center_um in [7.7, 11.3]:
            pah_mask = np.abs(wave_um - center_um) < 0.5
            cont_mask = np.abs(wave_um - (center_um + 2.0)) < 0.5
            if np.any(pah_mask) and np.any(cont_mask):
                pah_peak = np.max(sed_arr[pah_mask])
                cont_level = np.median(sed_arr[cont_mask])
                assert pah_peak > cont_level, (
                    f"PAH feature at {center_um}um not detected: "
                    f"peak={pah_peak:.3e}, cont={cont_level:.3e}"
                )

    def test_temperature_ordering_of_components(self):
        """Wien peaks: T_hot > T_warm > T_cold → λ_hot < λ_warm < λ_cold."""
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(3.5, 7.5, 2000)

        # Isolate each component
        sed_hot = magphys_dc08(wave, 1e10, dust_xi_pah=0, dust_xi_mir=1.0, dust_xi_warm=0)
        sed_warm = magphys_dc08(wave, 1e10, dust_xi_pah=0, dust_xi_mir=0, dust_xi_warm=1.0)
        sed_cold = magphys_dc08(wave, 1e10, dust_xi_pah=0, dust_xi_mir=0, dust_xi_warm=0)

        peak_hot = float(wave[jnp.argmax(sed_hot * wave)])
        peak_warm = float(wave[jnp.argmax(sed_warm * wave)])
        peak_cold = float(wave[jnp.argmax(sed_cold * wave)])

        assert peak_hot < peak_warm < peak_cold, (
            f"Temperature ordering wrong: hot={peak_hot / 1e4:.0f}um, "
            f"warm={peak_warm / 1e4:.0f}um, cold={peak_cold / 1e4:.0f}um"
        )

    def test_cmb_correction_at_high_z(self):
        """CMB heating raises dust temperature at high z.

        da Cunha+2013: T_eff = (T^(4+β) + T_CMB(z)^(4+β) - T_CMB(0)^(4+β))^(1/(4+β))
        At z=5, T_CMB = 2.725*(1+5) = 16.35 K. Cold dust (20K) is significantly
        affected — the peak should shift to shorter wavelengths.
        """
        from tengri.components.dust.emission import magphys_dc08

        wave = jnp.logspace(4, 7.5, 1000)

        sed_z0 = magphys_dc08(wave, 1e10, dust_T_cold=20.0, redshift=0.0)
        sed_z5 = magphys_dc08(wave, 1e10, dust_T_cold=20.0, redshift=5.0)

        peak_z0 = float(wave[jnp.argmax(sed_z0 * wave)])
        peak_z5 = float(wave[jnp.argmax(sed_z5 * wave)])

        # At z=5, CMB heats cold dust → peak shifts blueward
        # (T_CMB = 16.35K is close to T_cold = 20K, so effect is large)
        assert peak_z5 <= peak_z0, (
            f"CMB should warm dust at z=5: "
            f"peak(z=0)={peak_z0 / 1e4:.0f}um, peak(z=5)={peak_z5 / 1e4:.0f}um"
        )
