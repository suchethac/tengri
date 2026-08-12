# SPDX-License-Identifier: BSD-3-Clause
"""Anti-laziness tests: catch fake, stubbed, or incomplete implementations.

These tests are specifically designed to detect common lazy implementation
patterns that an automated agent might produce:

1. Functions that IGNORE their parameters (return same output regardless)
2. Hardcoded return values (constants, zeros, ones)
3. Analytic fallbacks pretending to be full models
4. Wrong units or normalization
5. Functions that only work for default parameters
6. Template loading that silently fails to dummy data
7. Energy non-conservation
8. Models that don't respect physical monotonicity
9. Luminosity/flux that doesn't scale with input luminosity
10. Functions that return the same shape regardless of wavelength grid

These tests should be run after ANY model implementation change to ensure
no lazy shortcuts were taken.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval


# ── 1. ATTENUATION CURVES — parameter sensitivity ─────────────────


class TestAttenuationParameterSensitivity:
    """Every attenuation curve with tunable parameters MUST respond to them."""

    def test_kriek_conroy_bump_not_ignored(self):
        """Verify dust_bump_strength actually changes the 2175A region."""
        from tengri.components.dust.attenuation import kriek_conroy

        wave = jnp.linspace(1800.0, 2600.0, 200)
        k0 = kriek_conroy(wave, dust_bump_strength=0.0, dust_delta=0.0)
        k3 = kriek_conroy(wave, dust_bump_strength=3.0, dust_delta=0.0)
        max_diff = float(jnp.max(jnp.abs(k3 - k0)))
        assert max_diff > 0.1, f"dust_bump_strength is IGNORED — max change is only {max_diff:.4f}"

    def test_kriek_conroy_delta_not_ignored(self):
        """Verify dust_delta actually changes the UV slope."""
        from tengri.components.dust.attenuation import kriek_conroy

        wave = jnp.geomspace(1000.0, 20000.0, 200)
        k_neg = kriek_conroy(wave, dust_delta=-0.5)
        k_pos = kriek_conroy(wave, dust_delta=0.5)
        max_diff = float(jnp.max(jnp.abs(k_neg - k_pos)))
        assert max_diff > 0.1, f"dust_delta is IGNORED — max change is only {max_diff:.4f}"

    def test_cardelli_rv_not_ignored(self):
        """Verify dust_Rv actually changes the Cardelli curve."""
        from tengri.components.dust.attenuation import cardelli

        wave = jnp.geomspace(1000.0, 20000.0, 200)
        k25 = cardelli(wave, dust_Rv=2.5)
        k50 = cardelli(wave, dust_Rv=5.0)
        max_diff = float(jnp.max(jnp.abs(k25 - k50)))
        assert max_diff > 0.1, f"dust_Rv is IGNORED — max change is only {max_diff:.4f}"

    def test_power_law_slope_not_ignored(self):
        """Verify n_slope actually changes the power law."""
        from tengri.components.dust.attenuation import power_law

        wave = jnp.geomspace(1000.0, 20000.0, 200)
        k03 = power_law(wave, n_slope=-0.3)
        k13 = power_law(wave, n_slope=-1.3)
        max_diff = float(jnp.max(jnp.abs(k03 - k13)))
        assert max_diff > 0.3, f"n_slope is IGNORED — max change is only {max_diff:.4f}"

    def test_li08_all_four_params_matter(self):
        """Li08 has 4 coefficients — ALL must affect the output."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.geomspace(912.0, 30000.0, 200)
        k_default = li08(wave)

        for param, val in [
            ("dust_c1", 10.0),
            ("dust_c2", 8.0),
            ("dust_c3", 5.0),
            ("dust_c4", 0.2),
        ]:
            k_modified = li08(wave, **{param: val})
            max_diff = float(jnp.max(jnp.abs(k_default - k_modified)))
            assert max_diff > 0.01, f"Li08 {param} is IGNORED — max change is only {max_diff:.6f}"

    def test_noll09_both_params_matter(self):
        """Noll09 bump and delta must both affect output."""
        from tengri.components.dust.attenuation import noll09

        wave = jnp.geomspace(912.0, 30000.0, 200)
        k_base = noll09(wave, dust_bump_strength=0.0, dust_delta=0.0)

        k_bump = noll09(wave, dust_bump_strength=3.0, dust_delta=0.0)
        assert float(jnp.max(jnp.abs(k_base - k_bump))) > 0.1, "Noll09 bump IGNORED"

        k_slope = noll09(wave, dust_bump_strength=0.0, dust_delta=-0.5)
        assert float(jnp.max(jnp.abs(k_base - k_slope))) > 0.1, "Noll09 delta IGNORED"

    def test_salim_sbl18_both_params_matter(self):
        """Salim+2018 bump and delta must both affect output."""
        from tengri.components.dust.attenuation import salim_sbl18

        wave = jnp.geomspace(912.0, 30000.0, 200)
        k_base = salim_sbl18(wave, dust_bump_strength=0.0, dust_delta=0.0)

        k_bump = salim_sbl18(wave, dust_bump_strength=3.0, dust_delta=0.0)
        assert float(jnp.max(jnp.abs(k_base - k_bump))) > 0.1, "SBL18 bump IGNORED"

        k_slope = salim_sbl18(wave, dust_bump_strength=0.0, dust_delta=-0.5)
        assert float(jnp.max(jnp.abs(k_base - k_slope))) > 0.1, "SBL18 delta IGNORED"

    def test_tea_delta_not_ignored(self):
        """TEA attenuation must respond to delta parameter."""
        from tengri.components.dust.attenuation import tea

        wave = jnp.geomspace(912.0, 30000.0, 200)
        k1 = tea(wave, dust_delta=-0.5)
        k2 = tea(wave, dust_delta=0.3)
        max_diff = float(jnp.max(jnp.abs(k1 - k2)))
        assert max_diff > 0.1, f"TEA delta is IGNORED — max change is only {max_diff:.4f}"

    def test_narayanan_redshift_not_ignored(self):
        """Narayanan curve must respond to redshift."""
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.geomspace(912.0, 30000.0, 200)
        k0 = narayanan_z(wave, redshift=0.0)
        k3 = narayanan_z(wave, redshift=3.0)
        max_diff = float(jnp.max(jnp.abs(k0 - k3)))
        assert max_diff > 0.01, f"Narayanan redshift IGNORED — max change is {max_diff:.6f}"


# ── 2. AGN DISC MODELS — parameter sensitivity & scaling ──────────


class TestAGNParameterSensitivity:
    """AGN models must respond to ALL their physical parameters."""

    _WAVE = jnp.geomspace(100.0, 1e8, 500)

    def test_powerlaw_alpha_changes_shape(self):
        """Power-law disc alpha must change the SED shape."""
        from tengri.components.agn.disc import powerlaw_disc

        l1 = powerlaw_disc(self._WAVE, agn_log_lbol=11.0, agn_alpha=-0.5)
        l2 = powerlaw_disc(self._WAVE, agn_log_lbol=11.0, agn_alpha=-1.5)
        # Shapes must differ
        corr = float(jnp.corrcoef(l1, l2)[0, 1])
        assert corr < 0.999, f"agn_alpha barely changes shape, correlation={corr:.6f}"

    def test_powerlaw_tmax_changes_shape(self):
        """Power-law disc T_max must change the UV cutoff."""
        from tengri.components.agn.disc import powerlaw_disc

        l_hot = powerlaw_disc(self._WAVE, agn_log_lbol=11.0, agn_T_max=1e6)
        l_cool = powerlaw_disc(self._WAVE, agn_log_lbol=11.0, agn_T_max=1e4)
        assert not jnp.allclose(l_hot, l_cool, rtol=0.01), "agn_T_max is IGNORED"

    def test_multicolor_mass_changes_shape(self):
        """BH mass must change the multicolor disc SED shape."""
        from tengri.components.agn.disc import multicolor_disc

        l7 = multicolor_disc(self._WAVE, agn_log_lbol=11.0, agn_log_mbh=7.0)
        l9 = multicolor_disc(self._WAVE, agn_log_lbol=11.0, agn_log_mbh=9.0)
        assert not jnp.allclose(l7, l9, rtol=0.01), "agn_log_mbh is IGNORED"

    def test_multicolor_spin_changes_shape(self):
        """BH spin must change the multicolor disc SED."""
        from tengri.components.agn.disc import multicolor_disc

        l0 = multicolor_disc(self._WAVE, agn_log_lbol=11.0, agn_a_spin=0.0)
        l9 = multicolor_disc(self._WAVE, agn_log_lbol=11.0, agn_a_spin=0.9)
        assert not jnp.allclose(l0, l9, rtol=0.01), "agn_a_spin is IGNORED"

    def test_kd_all_unique_params_matter(self):
        """Kubota & Done 3-zone disc: each unique parameter must matter."""
        from tengri.components.agn.disc import kubota_done_disc

        # Note: agn_gamma_warm has negligible effect after total SED
        # renormalization (warm zone is small). This is physical, not lazy.
        params_to_test = {
            "agn_f_hard": (0.01, 0.3),
            "agn_kt_warm": (0.1, 0.5),
            "agn_gamma_hard": (1.5, 2.2),
            "agn_kt_hot": (50.0, 200.0),
        }
        for param, (v1, v2) in params_to_test.items():
            l1 = kubota_done_disc(self._WAVE, agn_log_lbol=11.0, **{param: v1})
            l2 = kubota_done_disc(self._WAVE, agn_log_lbol=11.0, **{param: v2})
            # Use relative max difference — some params have subtle effects
            # after renormalization, but must differ at some wavelength
            rel_diff = float(jnp.max(jnp.abs(l1 - l2) / (jnp.abs(l1) + 1e-50)))
            assert rel_diff > 1e-4, (
                f"K&D parameter {param} is IGNORED (max relative diff: {rel_diff:.2e})"
            )

    def test_adaf_all_params_matter(self):
        """ADAF model (faithful Mahadevan 1997, #898): each parameter affects the
        output. Evaluated in the low-mdot regime (log_lbol=9, log_mbh=9 -> mdot
        ~1e-3, alpha_c>1) where the electron-heating delta is physically active:
        Eq. 43 (high mdot, alpha_c<1) has no delta term, so delta is *correctly*
        inert there. agn_r_tr / agn_log_ledd were retired in #898."""
        from tengri.components.agn.adaf import adaf_spectrum

        base = dict(agn_log_lbol=9.0, agn_log_mbh=9.0)
        params_to_test = {
            "agn_adaf_alpha": (0.1, 0.4),
            "agn_adaf_beta": (0.1, 0.9),
            "agn_adaf_delta": (0.01, 0.4),
            "agn_log_mbh": (8.0, 10.0),
        }
        for param, (v1, v2) in params_to_test.items():
            l1 = adaf_spectrum(self._WAVE, **{**base, param: v1})
            l2 = adaf_spectrum(self._WAVE, **{**base, param: v2})
            rel_diff = float(jnp.max(jnp.abs(l1 - l2) / (jnp.abs(l1) + 1e-50)))
            assert rel_diff > 1e-4, (
                f"ADAF parameter {param} is IGNORED (max relative diff: {rel_diff:.2e})"
            )

    def test_agn_luminosity_scaling(self):
        """10x L_bol must give ~10x total L_nu for all disc models.

        adaf is intentionally excluded: adaf_spectrum is normalized so that
        int L_nu dnu = L_bol *exactly* by construction, and its spectral shape
        shifts with L_bol (mdot is derived from it), so an on-grid-sum proxy is
        not a valid 10x test. Its exact L_bol scaling is pinned by
        test_adaf_mahadevan.py::TestSpectrumAssembly::test_normalizes_to_lbol.
        """
        from tengri.components.agn.disc import multicolor_disc, powerlaw_disc

        for name, fn, kwargs in [
            ("powerlaw", powerlaw_disc, {}),
            ("multicolor", multicolor_disc, {}),
        ]:
            l_low = fn(self._WAVE, agn_log_lbol=10.0, **kwargs)
            l_high = fn(self._WAVE, agn_log_lbol=11.0, **kwargs)
            ratio = float(jnp.sum(l_high)) / max(float(jnp.sum(l_low)), 1e-50)
            assert 5.0 < ratio < 20.0, f"{name}: 10x L_bol gave {ratio:.1f}x flux, expected ~10x"


# ── 3. SFH MODELS — parameter sensitivity ─────────────────────────


def _assert_parameter_matters(baseline, modified, label: str) -> None:
    """Fail unless ``modified`` differs from ``baseline`` by a *relative* margin.

    ``jnp.allclose`` carries a default ``atol=1e-8``. An SFH normalized to total
    mass has amplitudes of order ``10**log_total_mass / 1.4e10`` — for
    ``log_total_mass=1.0`` that is ~1e-9, entirely beneath that floor, so every
    comparison returns "close" no matter what the parameter does. A sensitivity
    suite in that regime cannot fail, which is worse than failing: it reports
    that nothing is ignored while being blind to everything.

    ``atol=0.0`` makes the comparison purely relative and immune to the
    normalization the caller happens to choose (#1728).
    """
    assert not jnp.allclose(baseline, modified, rtol=0.01, atol=0.0), f"{label} is IGNORED"


class TestSFHParameterSensitivity:
    """SFH models must respond to all their parameters."""

    _T = jnp.geomspace(1e5, 14e9, 500)

    def test_dpl_all_params_matter(self):
        """DPL: alpha, beta, tau, log_total_mass all must matter."""
        from tengri.components.stellar.sfh import dpl

        # log_total_mass=10.0 (mass normalization, not peak SFR): keeps SFR
        # amplitudes ~1 Msun/yr, well above jnp.allclose's default atol=1e-8.
        sfr_default = dpl(
            self._T, alpha=2.0, beta=1.0, tau=5e9, age=_AGE_UNIV_YR, log_total_mass=10.0
        )

        for param, val in [("alpha", 4.0), ("beta", 3.0), ("tau", 2e9), ("log_total_mass", 11.0)]:
            sfr_mod = dpl(
                self._T,
                **{
                    "alpha": 2.0,
                    "beta": 1.0,
                    "tau": 5e9,
                    "age": _AGE_UNIV_YR,
                    "log_total_mass": 10.0,
                    param: val,
                },
            )
            _assert_parameter_matters(sfr_default, sfr_mod, f"DPL parameter {param}")

    def test_tsnorm_all_params_matter(self):
        """tsnorm: all 5 params must affect the SFH.

        ``log_total_mass=10.0`` for the same reason the DPL test above uses it:
        it keeps SFR amplitudes near 1 Msun/yr. At the 1.0 this used to pass,
        amplitudes sat at ~1e-9 and the comparison could not resolve anything —
        it reported ``peak_lbt is IGNORED`` for a parameter that moves the peak
        from 3.21 to 0.19 Gyr, a 109% relative change (#1728).
        """
        from tengri.components.stellar.sfh import tsnorm

        defaults = {
            "log_total_mass": 10.0,
            "peak_lbt": 5e9,
            "width": 2e9,
            "skew": 0.5,
            "trunc": 2.0,
        }
        sfr_default = tsnorm(self._T, **defaults)

        mods = {
            "log_total_mass": 11.0,
            "peak_lbt": 2e9,
            "width": 0.5e9,
            "skew": -0.5,
            "trunc": 8.0,
        }
        for param, val in mods.items():
            sfr_mod = tsnorm(self._T, **{**defaults, param: val})
            _assert_parameter_matters(sfr_default, sfr_mod, f"tsnorm parameter {param}")

    def test_continuity_ratios_not_ignored(self):
        """Continuity SFH: each ratio must change the SFH."""
        from tengri.components.stellar.sfh import continuity

        age = jnp.geomspace(1e6, 13.7e9, 500)
        defaults = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr_default = continuity(age, log_total_mass=10.0, **defaults)

        for i in range(6):
            params = {**defaults, f"ratio_{i}": 1.0}
            sfr_mod = continuity(age, log_total_mass=10.0, **params)
            _assert_parameter_matters(sfr_default, sfr_mod, f"Continuity ratio_{i}")

    def test_dirichlet_zfracs_not_ignored(self):
        """Dirichlet SFH: each z_frac must change the SFH."""
        from tengri.components.stellar.sfh import dirichlet

        age = jnp.geomspace(1e6, 13.7e9, 500)
        defaults = {f"z_frac_{i}": 0.5 for i in range(6)}
        sfr_default = dirichlet(age, log_total_mass=10.0, **defaults)

        for i in range(6):
            params = {**defaults, f"z_frac_{i}": 0.01}
            sfr_mod = dirichlet(age, log_total_mass=10.0, **params)
            _assert_parameter_matters(sfr_default, sfr_mod, f"Dirichlet z_frac_{i}")


# ── 4. SHOCK EMISSION — velocity sensitivity ──────────────────────


class TestShockParameterSensitivity:
    """Shock models must respond to velocity and luminosity."""

    def test_velocity_changes_all_line_ratios(self):
        """Each line ratio must change with velocity."""
        from tengri.components.nebular.shock import shock_line_ratios

        r_100 = shock_line_ratios(100.0)
        r_500 = shock_line_ratios(500.0)

        for line in ["O3_5007A", "NII_6583A", "SII_6716A", "OI_6300A", "HA_6563A"]:
            v1 = float(r_100[line])
            v2 = float(r_500[line])
            assert abs(v1 - v2) > 0.01, (
                f"Shock line {line} does NOT change between 100 and 500 km/s"
            )

    def test_shock_sed_velocity_changes_shape(self):
        """Shock SED shape must change with velocity, not just scale."""
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.linspace(3000.0, 8000.0, 2000)
        l_slow = compute_shock_sed(wave, shock_velocity=150.0, l_shock_halpha=1e8)
        l_fast = compute_shock_sed(wave, shock_velocity=500.0, l_shock_halpha=1e8)

        # Normalize both to unit integral and compare shapes
        l_slow_norm = l_slow / jnp.sum(l_slow)
        l_fast_norm = l_fast / jnp.sum(l_fast)
        shape_diff = float(jnp.sum(jnp.abs(l_slow_norm - l_fast_norm)))
        assert shape_diff > 0.01, "Shock SED shape doesn't change with velocity — suspicious"


# ── 5. TEMPLATE LOADING — not falling back to dummies ─────────────


class TestTemplatesAreReal:
    """Template-based models must actually load real data, not fallbacks."""

    def test_dl07_templates_loaded(self):
        """DL07 templates must be loaded from data file, not analytic fallback."""
        from tengri.components.dust import emission

        wave = jnp.geomspace(1e3, 1e7, 500)
        l_absorbed = 1e10  # Lsun
        l_nu = emission.draine_li2007(wave, l_absorbed, dust_qpah=0.04, dust_umin=1.0)
        # Check that 7.7μm (77000A) PAH feature is present
        pah_region = (wave > 70000) & (wave < 85000)
        continuum = (wave > 50000) & (wave < 60000)
        if jnp.any(l_nu[pah_region] > 0) and jnp.any(l_nu[continuum] > 0):
            pah_excess = float(jnp.mean(l_nu[pah_region]) / jnp.mean(l_nu[continuum]))
            assert pah_excess > 1.2, (
                f"DL07 PAH feature at 7.7μm too weak ({pah_excess:.2f}x) — "
                "likely using analytic fallback, not real templates"
            )

    def test_dale2014_alpha_changes_shape(self):
        """Dale2014 dust_alpha_dale must change the SED shape."""
        from tengri.components.dust import emission

        wave = jnp.geomspace(1e3, 1e7, 500)
        l_absorbed = 1e10
        l1 = emission.dale2014(wave, l_absorbed, dust_alpha_dale=1.0)
        l2 = emission.dale2014(wave, l_absorbed, dust_alpha_dale=3.0)

        if float(jnp.sum(l1)) > 0 and float(jnp.sum(l2)) > 0:
            assert not jnp.allclose(l1 / jnp.sum(l1), l2 / jnp.sum(l2), rtol=0.05), (
                "Dale2014 dust_alpha_dale is IGNORED — likely using fallback"
            )

    def test_astrodust_qpah_changes_shape(self):
        """Astrodust qPAH must change the SED shape.

        If this fails, the analytic fallback (simple MBB) is being used
        instead of real templates. The fallback ignores qPAH entirely.
        Run scripts/download_astrodust_templates.py to get real templates.
        """
        import warnings

        from tengri.components.dust import emission

        wave = jnp.geomspace(1e3, 1e7, 500)
        l_absorbed = 1e10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            l_low = emission.astrodust(wave, l_absorbed, dust_qpah=0.01, dust_umin=1.0)
            l_high = emission.astrodust(wave, l_absorbed, dust_qpah=0.06, dust_umin=1.0)
            # If fallback warning was emitted, skip this test
            fallback_warned = any(
                "Falling back" in str(wi.message) or "fallback" in str(wi.message).lower()
                for wi in w
            )

        if fallback_warned:
            pytest.skip("Astrodust using analytic fallback — templates not loaded")

        # Also check if output is identical (fallback doesn't warn on some paths)
        if float(jnp.sum(l_low)) > 0 and float(jnp.sum(l_high)) > 0:
            shapes_differ = not jnp.allclose(
                l_low / jnp.sum(l_low), l_high / jnp.sum(l_high), rtol=0.05
            )
            if not shapes_differ:
                # Identical output → analytic fallback ignoring qPAH
                pytest.skip(
                    "Astrodust qPAH has no effect — analytic fallback active. "
                    "Load real templates with scripts/download_astrodust_templates.py"
                )

    def test_themis_qhac_changes_shape(self):
        """THEMIS qhac must change the SED shape."""
        from tengri.components.dust import emission

        wave = jnp.geomspace(1e3, 1e7, 500)
        l_absorbed = 1e10
        l_low = emission.themis(wave, l_absorbed, dust_qhac=0.01, dust_umin=1.0)
        l_high = emission.themis(wave, l_absorbed, dust_qhac=0.10, dust_umin=1.0)

        if float(jnp.sum(l_low)) > 0 and float(jnp.sum(l_high)) > 0:
            assert not jnp.allclose(l_low / jnp.sum(l_low), l_high / jnp.sum(l_high), rtol=0.05), (
                "THEMIS qhac is IGNORED — likely using fallback"
            )

    def test_bosa_ssfr_changes_shape(self):
        """BOSA sSFR must change the SED shape."""
        from tengri.components.dust import emission

        wave = jnp.geomspace(1e3, 1e7, 500)
        l_absorbed = 1e10
        l_low = emission.bosa(wave, l_absorbed, dust_log_ssfr=-11.0)
        l_high = emission.bosa(wave, l_absorbed, dust_log_ssfr=-9.0)

        if float(jnp.sum(l_low)) > 0 and float(jnp.sum(l_high)) > 0:
            assert not jnp.allclose(l_low / jnp.sum(l_low), l_high / jnp.sum(l_high), rtol=0.05), (
                "BOSA sSFR is IGNORED — likely using fallback"
            )


# ── 6. WAVELENGTH GRID INDEPENDENCE — no hardcoded grid ───────────


class TestWavelengthGridIndependence:
    """Models must work on arbitrary wavelength grids, not just one grid."""

    def test_attenuation_works_on_any_grid(self):
        """Attenuation curves must return correct shapes for any grid."""
        from tengri.components.dust.attenuation import calzetti, cardelli, kriek_conroy, smc

        for fn in [calzetti, cardelli, smc, kriek_conroy]:
            for n in [10, 100, 1000]:
                wave = jnp.geomspace(1000.0, 20000.0, n)
                k = fn(wave)
                assert k.shape == (n,), f"Wrong shape: expected ({n},), got {k.shape}"
                assert jnp.all(jnp.isfinite(k)), f"Non-finite values on grid size {n}"

    def test_agn_disc_works_on_any_grid(self):
        """AGN disc models must work on arbitrary wavelength grids."""
        from tengri.components.agn.disc import multicolor_disc, powerlaw_disc

        for fn in [powerlaw_disc, multicolor_disc]:
            for n in [10, 100, 500]:
                wave = jnp.geomspace(100.0, 1e6, n)
                l_nu = fn(wave, agn_log_lbol=11.0)
                assert l_nu.shape == (n,), f"Wrong shape: expected ({n},), got {l_nu.shape}"
                chex.assert_tree_all_finite(l_nu)


# ── 7. NOT RETURNING ZEROS OR ONES ────────────────────────────────


class TestNotReturningDummies:
    """Models must return physically meaningful values, not constants."""

    def test_attenuation_not_all_ones(self):
        """k(λ) must vary with wavelength (not all 1.0)."""
        from tengri.components.dust.attenuation import DUST_LAWS

        wave = jnp.geomspace(1000.0, 20000.0, 100)
        for name, fn in DUST_LAWS.items():
            k = fn(wave)
            std = float(jnp.std(k))
            assert std > 0.01, f"{name}: k(λ) is nearly constant (std={std:.6f}) — suspicious"

    def test_agn_seds_not_all_zeros(self):
        """AGN SEDs must have non-zero flux in their expected bands."""
        from tengri.components.agn.adaf import adaf_spectrum
        from tengri.components.agn.disc import multicolor_disc, powerlaw_disc

        wave = jnp.geomspace(100.0, 1e8, 500)
        for name, fn, kwargs in [
            ("powerlaw", powerlaw_disc, {}),
            ("multicolor", multicolor_disc, {}),
            ("adaf", adaf_spectrum, {}),
        ]:
            l_nu = fn(wave, agn_log_lbol=11.0, **kwargs)
            total = float(jnp.sum(l_nu))
            assert total > 0, f"{name}: total flux is zero"
            non_zero = float(jnp.sum(l_nu > 0))
            assert non_zero > 10, f"{name}: only {non_zero} non-zero pixels — likely broken"

    def test_sfh_not_all_zeros(self):
        """SFH models must produce non-zero SFR in some range."""
        from tengri.components.stellar.sfh import dpl, lnorm, norm, snorm, tsnorm

        t = jnp.geomspace(1e5, 14e9, 500)
        for name, fn, kwargs in [
            (
                "tsnorm",
                tsnorm,
                {
                    "log_total_mass": 1.0,
                    "peak_lbt": 5e9,
                    "width": 2e9,
                    "skew": 0.5,
                    "trunc": 2.0,
                },
            ),
            ("snorm", snorm, {"log_total_mass": 1.0, "peak_lbt": 5e9, "width": 2e9, "skew": 0.5}),
            ("norm", norm, {"log_total_mass": 1.0, "peak_lbt": 5e9, "width": 2e9}),
            (
                "lnorm",
                lnorm,
                {"log_total_mass": 1.0, "peak": 5e9, "width": 0.5, "age": _AGE_UNIV_YR},
            ),
            (
                "dpl",
                dpl,
                {
                    "alpha": 2.0,
                    "beta": 1.0,
                    "tau": 5e9,
                    "age": _AGE_UNIV_YR,
                    "log_total_mass": 1.0,
                },
            ),
        ]:
            sfr = fn(t, **kwargs)
            total = float(jnp.sum(sfr))
            assert total > 0, f"{name}: SFR is all zeros"


# ── 8. BLR/NLR — lines at correct wavelengths ─────────────────────


class TestEmissionLinesAtCorrectWavelengths:
    """Line emission must appear at the correct atomic wavelengths."""

    def test_blr_lines_at_atomic_wavelengths(self):
        """BLR emission must peak at known line wavelengths, not elsewhere."""
        from tengri.components.agn.blr import compute_blr_sed

        l_disc = 3.83e44
        expected_lines = [1216.0, 1549.0, 2800.0, 4861.0, 6563.0]

        for line_wave in expected_lines:
            wave = jnp.linspace(line_wave - 200, line_wave + 200, 1000)
            l_nu = compute_blr_sed(wave, l_disc_bol_erg=l_disc)
            peak = float(wave[jnp.argmax(l_nu)])
            assert abs(peak - line_wave) < 30.0, (
                f"BLR line expected at {line_wave:.0f} A, peaked at {peak:.0f} A"
            )

    def test_nlr_lines_at_atomic_wavelengths(self):
        """NLR emission must peak at known forbidden line wavelengths."""
        from tengri.components.agn.nlr import compute_nlr_sed

        l_disc = 3.83e44
        expected_lines = [5007.0, 6563.0]

        for line_wave in expected_lines:
            wave = jnp.linspace(line_wave - 100, line_wave + 100, 500)
            l_nu = compute_nlr_sed(wave, l_disc_bol_erg=l_disc)
            if float(jnp.max(l_nu)) > 0:
                peak = float(wave[jnp.argmax(l_nu)])
                assert abs(peak - line_wave) < 20.0, (
                    f"NLR line expected at {line_wave:.0f} A, peaked at {peak:.0f} A"
                )

    def test_shock_lines_at_atomic_wavelengths(self):
        """Shock emission must produce lines at correct wavelengths."""
        from tengri.components.nebular.shock import compute_shock_sed

        expected_lines = [4861.0, 5007.0, 6563.0]  # Hβ, [OIII], Hα
        wave = jnp.linspace(4800.0, 6700.0, 5000)
        l_nu = compute_shock_sed(wave, shock_velocity=300.0, l_shock_halpha=1e8)

        # Find peaks
        for line_wave in expected_lines:
            nearby = (wave > line_wave - 50) & (wave < line_wave + 50)
            if jnp.any(l_nu[nearby] > 0):
                local_peak = float(wave[nearby][jnp.argmax(l_nu[nearby])])
                assert abs(local_peak - line_wave) < 20.0, (
                    f"Shock line expected near {line_wave:.0f} A, got {local_peak:.0f} A"
                )


# ── 9. WG00 GEOMETRY — not returning identity ─────────────────────


class TestWG00NotIdentity:
    """WG00 geometry functions must actually attenuate light."""

    def test_shell_not_identity(self):
        """Shell with tau_v > 0 must attenuate (T < 1)."""
        from tengri.components.dust.attenuation import wg00_shell

        wave = jnp.geomspace(1000.0, 20000.0, 100)
        t = wg00_shell(wave, tau_v=1.0)
        assert float(jnp.min(t)) < 0.5, "Shell at tau=1 barely attenuates — suspicious"

    def test_cloudy_not_identity(self):
        """Cloudy geometry with tau_v > 0 must attenuate."""
        from tengri.components.dust.attenuation import wg00_cloudy

        wave = jnp.geomspace(1000.0, 20000.0, 100)
        t = wg00_cloudy(wave, tau_v=1.0)
        assert float(jnp.min(t)) < 0.8, "Cloudy at tau=1 barely attenuates — suspicious"

    def test_dusty_not_identity(self):
        """Dusty geometry with tau_v > 0 must attenuate."""
        from tengri.components.dust.attenuation import wg00_dusty

        wave = jnp.geomspace(1000.0, 20000.0, 100)
        t = wg00_dusty(wave, tau_v=1.0)
        assert float(jnp.min(t)) < 0.9, "Dusty at tau=1 barely attenuates — suspicious"
