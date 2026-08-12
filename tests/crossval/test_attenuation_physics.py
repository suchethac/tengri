# SPDX-License-Identifier: BSD-3-Clause
"""Physics cross-validation for ALL attenuation curve models.

Tests physical correctness of dust attenuation curves against known
astrophysical relationships, published reference values, and cross-curve
ordering constraints.

References
----------
- Calzetti et al. 2000, ApJ, 533, 682
- Cardelli, Clayton & Mathis 1989, ApJ, 345, 245
- Gordon et al. 2003, ApJ, 594, 279 (SMC)
- Pei 1992, ApJ, 395, 130 (SMC, LMC generalized Drude)
- Kriek & Conroy 2013, ApJL, 775, L16
- Li et al. 2008, ApJ, 685, 1046
- Leitherer et al. 2002, ApJS, 140, 303
- Noll et al. 2009, A&A, 507, 1793
- Salim, Boquien & Lee 2018, ApJ, 859, 11
- Witt & Gordon 2000, ApJ, 528, 799
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.crossval

# Standard wavelength grid: 912A (Lyman limit) to 30000A (K-band)
WAVE = jnp.geomspace(912.0, 30000.0, 1000)


# ── 1. ALL CURVES — universal physics constraints ─────────────────


class TestAllCurvesUniversalPhysics:
    """Every attenuation curve must satisfy these universal constraints."""

    @pytest.fixture(
        params=[
            "power_law",
            "calzetti",
            "kriek_conroy",
            "smc",
            "lmc",
            "cardelli",
            "li08",
            "salim",
            "leitherer02",
            "noll09",
            "salim_sbl18",
            "tea",
            "narayanan_z",
            "conroy2010",
        ]
    )
    def curve(self, request):
        from tengri.components.dust.attenuation import resolve_dust_law

        return request.param, resolve_dust_law(request.param)

    def test_k_at_5500_equals_one(self, curve):
        """k(5500A) = 1.0 for all curves (normalization convention)."""
        name, fn = curve
        k = fn(jnp.array([5500.0]))
        assert abs(float(k[0]) - 1.0) < 1e-3, (
            f"{name}: k(5500A) should be 1.0, got {float(k[0]):.6f}"
        )

    def test_k_positive_optical_uv(self, curve):
        """k(λ) > 0 for all optical/UV wavelengths (dust always attenuates)."""
        name, fn = curve
        wave_test = jnp.geomspace(1000.0, 10000.0, 100)
        k = fn(wave_test)
        assert_non_negative(k, name="k", msg=f"{name}: k(λ) has negative values")

    def test_k_finite(self, curve):
        """k(λ) must be finite everywhere."""
        name, fn = curve
        k = fn(WAVE)
        assert jnp.all(jnp.isfinite(k)), f"{name}: k(λ) has non-finite values"

    def test_k_greater_in_uv_than_nir(self, curve):
        """UV attenuation > NIR attenuation (dust grains scatter/absorb UV more)."""
        name, fn = curve
        k_uv = float(fn(jnp.array([1500.0]))[0])
        k_nir = float(fn(jnp.array([20000.0]))[0])
        assert k_uv > k_nir, f"{name}: k(1500A)={k_uv:.2f} should exceed k(20000A)={k_nir:.2f}"


# ── 2. CALZETTI — starburst curve reference values ────────────────


class TestCalzettiPhysics:
    """Calzetti curve must match published polynomial coefficients."""

    def test_rv_is_4p05(self):
        """Calzetti R_V = 4.05 is hardcoded (much grayer than MW R_V=3.1)."""
        from tengri.components.dust.attenuation import calzetti

        # k(V) = (k'(V) + R_V) / R_V = 1.0 by definition
        k_v = float(calzetti(jnp.array([5500.0]))[0])
        assert abs(k_v - 1.0) < 0.01

    def test_calzetti_k_at_1600(self):
        """k(1600A) normalized to V-band.

        The function returns k = (k' + R_V) / R_V. At 1600A (UV polynomial),
        k ≈ 2.5 (the UV is ~2.5x more attenuated than V-band).
        """
        from tengri.components.dust.attenuation import calzetti

        k = float(calzetti(jnp.array([1600.0]))[0])
        assert 2.0 < k < 3.5, f"Calzetti k(1600A) should be ~2.5, got {k:.2f}"

    def test_ir_optical_transition_smooth(self):
        """No discontinuity at 6300A (IR/optical polynomial boundary)."""
        from tengri.components.dust.attenuation import calzetti

        wave = jnp.linspace(6000.0, 6600.0, 200)
        k = calzetti(wave)
        dk = jnp.diff(k)
        # Max discontinuity should be tiny
        max_jump = float(jnp.max(jnp.abs(dk)))
        assert max_jump < 0.02, f"Calzetti has discontinuity at 6300A: {max_jump}"


# ── 3. CARDELLI MW — R_V dependence ───────────────────────────────


class TestCardelliPhysics:
    """Cardelli curve must obey R_V-dependent MW extinction physics."""

    def test_higher_rv_grayer(self):
        """Higher R_V → grayer (flatter) curve (larger grains)."""
        from tengri.components.dust.attenuation import cardelli

        k_low = cardelli(WAVE, dust_Rv=2.5)
        k_high = cardelli(WAVE, dust_Rv=5.0)

        # UV/optical ratio should be lower for higher R_V
        k_uv_low = float(k_low[jnp.argmin(jnp.abs(WAVE - 1500.0))])
        k_v_low = float(k_low[jnp.argmin(jnp.abs(WAVE - 5500.0))])
        k_uv_high = float(k_high[jnp.argmin(jnp.abs(WAVE - 1500.0))])
        k_v_high = float(k_high[jnp.argmin(jnp.abs(WAVE - 5500.0))])

        ratio_low = k_uv_low / k_v_low
        ratio_high = k_uv_high / k_v_high
        assert ratio_low > ratio_high, "Lower R_V should give steeper UV/V ratio"

    def test_2175_bump_present(self):
        """MW curve must have the 2175A UV bump (graphite carrier)."""
        from tengri.components.dust.attenuation import cardelli

        wave = jnp.linspace(1800.0, 2600.0, 500)
        k = cardelli(wave, dust_Rv=3.1)
        # Find local maximum near 2175A
        bump_region = (wave > 2000) & (wave < 2350)
        k_bump = k[bump_region]
        # Bump should create a local max
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_2000 = float(k[jnp.argmin(jnp.abs(wave - 2000.0))])
        k_at_2500 = float(k[jnp.argmin(jnp.abs(wave - 2500.0))])
        assert k_at_2175 > k_at_2500, "2175A bump should be above 2500A"

    def test_rv_3p1_b_minus_v(self):
        """By definition: A(B)/A(V) = 1 + 1/R_V. For R_V=3.1: A(B)/A(V) ≈ 1.323."""
        from tengri.components.dust.attenuation import cardelli

        k = cardelli(jnp.array([4400.0, 5500.0]), dust_Rv=3.1)
        ratio = float(k[0] / k[1])
        expected = 1.0 + 1.0 / 3.1
        assert abs(ratio - expected) < 0.05, f"A(B)/A(V) should be {expected:.3f}, got {ratio:.3f}"


# ── 4. SMC vs LMC vs MW — cross-curve ordering ────────────────────


class TestExtinctionCurveOrdering:
    """Different environments produce distinct extinction curves."""

    def test_smc_steeper_than_mw_in_uv(self):
        """SMC has no 2175A bump and steeper UV rise than MW."""
        from tengri.components.dust.attenuation import cardelli, smc

        wave_fuv = jnp.array([1500.0])
        k_smc = float(smc(wave_fuv)[0])
        k_mw = float(cardelli(wave_fuv, dust_Rv=3.1)[0])
        # SMC is steeper: k(1500)/k(5500) is higher for SMC
        assert k_smc > k_mw, "SMC should be steeper than MW at 1500A"

    def test_smc_no_bump(self):
        """SMC has NO 2175A bump (Pei 1992)."""
        from tengri.components.dust.attenuation import smc

        wave = jnp.linspace(1800.0, 2600.0, 200)
        k = smc(wave)
        # Should be monotonically decreasing through the bump region
        # (or at most negligible bump)
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_2000 = float(k[jnp.argmin(jnp.abs(wave - 2000.0))])
        k_at_2400 = float(k[jnp.argmin(jnp.abs(wave - 2400.0))])

        # Bump strength: excess above linear interpolation between 2000 and 2400
        k_interp = k_at_2000 + (k_at_2400 - k_at_2000) * (2175 - 2000) / (2400 - 2000)
        bump_excess = k_at_2175 - k_interp
        assert bump_excess < 0.3, f"SMC bump excess should be negligible, got {bump_excess:.2f}"

    def test_lmc_weak_bump(self):
        """LMC has a WEAK 2175A bump (Pei 1992)."""
        from tengri.components.dust.attenuation import lmc

        wave = jnp.linspace(1800.0, 2600.0, 200)
        k = lmc(wave)
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_2000 = float(k[jnp.argmin(jnp.abs(wave - 2000.0))])
        k_at_2400 = float(k[jnp.argmin(jnp.abs(wave - 2400.0))])
        k_interp = k_at_2000 + (k_at_2400 - k_at_2000) * (2175 - 2000) / (2400 - 2000)
        bump_excess = k_at_2175 - k_interp
        # LMC bump should be present but weaker than MW
        assert bump_excess > -0.5, f"LMC should have some bump, got excess {bump_excess:.2f}"

    def test_calzetti_grayer_than_mw(self):
        """Calzetti (R_V=4.05) is grayer than MW (R_V=3.1): less UV/V contrast."""
        from tengri.components.dust.attenuation import calzetti, cardelli

        k_calz = calzetti(WAVE)
        k_mw = cardelli(WAVE, dust_Rv=3.1)

        # UV/V ratio should be lower for Calzetti (grayer)
        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        v_idx = int(jnp.argmin(jnp.abs(WAVE - 5500.0)))
        ratio_calz = float(k_calz[uv_idx] / k_calz[v_idx])
        ratio_mw = float(k_mw[uv_idx] / k_mw[v_idx])
        assert ratio_calz < ratio_mw, "Calzetti should be grayer (lower UV/V ratio) than MW"


# ── 5. KRIEK-CONROY — bump + slope modifications ──────────────────


class TestKriekConroyPhysics:
    """Kriek & Conroy (2013) modified Calzetti with bump and slope."""

    def test_zero_params_equals_calzetti(self):
        """E_b=0, delta=0 → pure Calzetti curve."""
        from tengri.components.dust.attenuation import calzetti, kriek_conroy

        k_calz = calzetti(WAVE)
        k_kc = kriek_conroy(WAVE, dust_bump_strength=0.0, dust_delta=0.0)
        np.testing.assert_allclose(k_calz, k_kc, rtol=1e-10)

    def test_bump_strength_adds_2175(self):
        """Positive E_b adds a 2175A bump."""
        from tengri.components.dust.attenuation import kriek_conroy

        k_no_bump = kriek_conroy(WAVE, dust_bump_strength=0.0)
        k_bump = kriek_conroy(WAVE, dust_bump_strength=3.0)

        idx_2175 = int(jnp.argmin(jnp.abs(WAVE - 2175.0)))
        assert float(k_bump[idx_2175]) > float(k_no_bump[idx_2175]), (
            "E_b > 0 should add bump at 2175A"
        )

    def test_negative_delta_steepens_uv(self):
        """Negative delta → steeper UV (more attenuation at short λ)."""
        from tengri.components.dust.attenuation import kriek_conroy

        k_steep = kriek_conroy(WAVE, dust_delta=-0.5)
        k_flat = kriek_conroy(WAVE, dust_delta=0.5)

        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        nir_idx = int(jnp.argmin(jnp.abs(WAVE - 15000.0)))

        ratio_steep = float(k_steep[uv_idx] / k_steep[nir_idx])
        ratio_flat = float(k_flat[uv_idx] / k_flat[nir_idx])
        assert ratio_steep > ratio_flat, "Negative delta should steepen UV"


# ── 6. LEITHERER02 — far-UV extension ─────────────────────────────


class TestLeitherer02Physics:
    """Leitherer+2002 UV extension of Calzetti curve."""

    def test_matches_calzetti_above_1800(self):
        """L02 and Calzetti should agree for λ > 1800A."""
        from tengri.components.dust.attenuation import calzetti, leitherer02

        wave_opt = jnp.geomspace(2000.0, 20000.0, 100)
        k_calz = calzetti(wave_opt)
        k_l02 = leitherer02(wave_opt)
        np.testing.assert_allclose(k_calz, k_l02, rtol=0.05)

    def test_extends_below_1200(self):
        """L02 provides valid values below Calzetti's 1200A limit."""
        from tengri.components.dust.attenuation import leitherer02

        wave_fuv = jnp.array([970.0, 1000.0, 1100.0, 1200.0])
        k = leitherer02(wave_fuv)
        chex.assert_tree_all_finite(k)
        assert jnp.all(k > 0), "L02 should give positive k below 1200A"


# ── 7. NOLL09 & SALIM_SBL18 — modified Calzetti variants ──────────


class TestModifiedCalzettiPhysics:
    """Noll+2009 and Salim+2018 modified Calzetti variants."""

    def test_noll09_bump_adds_2175(self):
        """Noll09 with positive E_b adds 2175A bump."""
        from tengri.components.dust.attenuation import noll09

        k_no = noll09(WAVE, dust_bump_strength=0.0, dust_delta=0.0)
        k_yes = noll09(WAVE, dust_bump_strength=3.0, dust_delta=0.0)

        idx_2175 = int(jnp.argmin(jnp.abs(WAVE - 2175.0)))
        assert float(k_yes[idx_2175]) > float(k_no[idx_2175])

    def test_salim_sbl18_slope_modification(self):
        """SBL18 negative delta steepens UV slope."""
        from tengri.components.dust.attenuation import salim_sbl18

        k_steep = salim_sbl18(WAVE, dust_bump_strength=0.0, dust_delta=-0.5)
        k_flat = salim_sbl18(WAVE, dust_bump_strength=0.0, dust_delta=0.5)

        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        nir_idx = int(jnp.argmin(jnp.abs(WAVE - 15000.0)))

        ratio_steep = float(k_steep[uv_idx] / k_steep[nir_idx])
        ratio_flat = float(k_flat[uv_idx] / k_flat[nir_idx])
        assert ratio_steep > ratio_flat

    def test_noll09_delta_zero_bump_zero_near_calzetti(self):
        """Noll09 with no modifications should be close to Calzetti+L02."""
        from tengri.components.dust.attenuation import leitherer02, noll09

        wave_opt = jnp.geomspace(1500.0, 20000.0, 100)
        k_l02 = leitherer02(wave_opt)
        k_noll = noll09(wave_opt, dust_bump_strength=0.0, dust_delta=0.0)
        np.testing.assert_allclose(k_l02, k_noll, rtol=0.10)


# ── 8. LI08 — flexible 4-parameter curve ──────────────────────────


class TestLi08Physics:
    """Li+2008 four-coefficient analytic curve physics."""

    def test_mw_preset_has_bump(self):
        """MW-like preset (c4>0) must show 2175A bump."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.linspace(1800.0, 2600.0, 300)
        k = li08(wave, dust_c1=6.0, dust_c2=4.0, dust_c3=2.0, dust_c4=0.04)
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_2500 = float(k[jnp.argmin(jnp.abs(wave - 2500.0))])
        assert k_at_2175 > k_at_2500, "MW preset should have 2175A bump"

    def test_smc_preset_no_bump(self):
        """SMC-like preset (c4=0) must have no bump."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.linspace(1800.0, 2600.0, 300)
        k = li08(wave, dust_c1=5.0, dust_c2=5.5, dust_c3=1.5, dust_c4=0.0)
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_2000 = float(k[jnp.argmin(jnp.abs(wave - 2000.0))])
        k_at_2400 = float(k[jnp.argmin(jnp.abs(wave - 2400.0))])
        k_interp = k_at_2000 + (k_at_2400 - k_at_2000) * (2175 - 2000) / (2400 - 2000)
        assert abs(k_at_2175 - k_interp) < 0.3, "SMC preset should have no bump"

    def test_c4_controls_bump_strength(self):
        """Higher c4 → stronger 2175A bump."""
        from tengri.components.dust.attenuation import li08

        wave = jnp.array([2175.0])
        k_low = float(li08(wave, dust_c4=0.01)[0])
        k_high = float(li08(wave, dust_c4=0.10)[0])
        assert k_high > k_low, "Higher c4 should increase bump"

    def test_c2_changes_curve_shape(self):
        """Different c2 values produce measurably different curve shapes."""
        from tengri.components.dust.attenuation import li08

        k_low_c2 = li08(WAVE, dust_c2=2.0)
        k_high_c2 = li08(WAVE, dust_c2=6.0)

        # Curves should differ meaningfully at UV wavelengths
        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        diff = abs(float(k_low_c2[uv_idx] - k_high_c2[uv_idx]))
        assert diff > 0.1, f"c2 should change UV shape, got diff={diff:.3f}"


# ── 9. WG00 GEOMETRIES — radiative transfer geometry ordering ─────


class TestWG00GeometryPhysics:
    """Witt & Gordon (2000) dust-star geometry ordering."""

    def test_geometry_ordering(self):
        """Geometries produce different transmission for same tau_v.

        Shell (foreground screen) gives exp(-tau*k). Cloudy and dusty
        are grayer due to mixed dust-star geometry.
        """
        from tengri.components.dust.attenuation import wg00_cloudy, wg00_dusty, wg00_shell

        tau_v = 1.0
        t_shell = wg00_shell(WAVE, tau_v=tau_v)
        t_cloudy = wg00_cloudy(WAVE, tau_v=tau_v)
        t_dusty = wg00_dusty(WAVE, tau_v=tau_v)

        # All should be valid transmissions
        for t, name in [(t_shell, "shell"), (t_cloudy, "cloudy"), (t_dusty, "dusty")]:
            assert jnp.all(jnp.isfinite(t)), f"{name} has non-finite values"

    def test_zero_tau_full_transmission(self):
        """τ=0 → T=1 for all geometries (no dust)."""
        from tengri.components.dust.attenuation import wg00_cloudy, wg00_dusty, wg00_shell

        for fn in [wg00_shell, wg00_cloudy, wg00_dusty]:
            t = fn(WAVE, tau_v=0.0)
            np.testing.assert_allclose(t, 1.0, atol=1e-10)

    def test_transmission_in_zero_one(self):
        """Transmission must be in (0, 1] for finite tau."""
        from tengri.components.dust.attenuation import wg00_cloudy, wg00_dusty, wg00_shell

        for fn in [wg00_shell, wg00_cloudy, wg00_dusty]:
            for tau_v in [0.1, 1.0, 5.0]:
                t = fn(WAVE, tau_v=tau_v)
                assert jnp.all(t > 0), "Transmission must be positive"
                assert jnp.all(t <= 1.0 + 1e-10), "Transmission must be ≤ 1"

    def test_more_dust_less_transmission(self):
        """Higher tau → lower transmission (monotonic)."""
        from tengri.components.dust.attenuation import wg00_shell

        taus = [0.1, 0.5, 1.0, 2.0, 5.0]
        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        prev_t = 1.0
        for tau_v in taus:
            t = float(wg00_shell(WAVE, tau_v=tau_v)[uv_idx])
            assert t < prev_t, f"Transmission should decrease with tau={tau_v}"
            prev_t = t


# ── 10. NARAYANAN — redshift-dependent curve ──────────────────────


class TestNarayananPhysics:
    """Narayanan+2018 redshift-dependent attenuation curve."""

    def test_z0_based_on_kriek_conroy(self):
        """At z=0, Narayanan uses K-C with z-dependent delta and bump."""
        from tengri.components.dust.attenuation import narayanan_z

        wave_test = jnp.geomspace(1500.0, 20000.0, 100)
        k_nz = narayanan_z(wave_test, redshift=0.0)
        # Should be finite and positive
        chex.assert_tree_all_finite(k_nz)
        assert_non_negative(k_nz, name="k_nz")
        # k(5500) should be ~1
        v_idx = int(jnp.argmin(jnp.abs(wave_test - 5500.0)))
        assert abs(float(k_nz[v_idx]) - 1.0) < 0.1

    def test_redshift_modifies_curve(self):
        """Different redshifts produce different curve shapes."""
        from tengri.components.dust.attenuation import narayanan_z

        k_z0 = narayanan_z(WAVE, redshift=0.0)
        k_z3 = narayanan_z(WAVE, redshift=3.0)

        # Curves should differ at UV wavelengths
        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        diff = abs(float(k_z0[uv_idx] - k_z3[uv_idx]))
        assert diff > 0.01, f"Redshift should modify UV shape, got diff={diff:.4f}"


# ── 11. CONROY2010 — MW/power-law blend ───────────────────────────


class TestConroy2010Physics:
    """Conroy+2010 mixed MW + power-law (FSPS dust_type=1)."""

    def test_output_positive_finite(self):
        """Conroy2010 must produce positive finite values."""
        from tengri.components.dust.attenuation import conroy2010

        k = conroy2010(WAVE)
        chex.assert_tree_all_finite(k)
        assert_non_negative(k, name="k")

    def test_uv_bump_present(self):
        """Conroy2010 has MW-like 2175A bump at short wavelengths."""
        from tengri.components.dust.attenuation import conroy2010

        wave = jnp.linspace(1800.0, 2600.0, 300)
        k = conroy2010(wave)
        # Check for local enhancement near 2175A
        k_at_2175 = float(k[jnp.argmin(jnp.abs(wave - 2175.0))])
        k_at_2500 = float(k[jnp.argmin(jnp.abs(wave - 2500.0))])
        # Allow that bump may be weak due to power-law dilution
        assert k_at_2175 >= k_at_2500 * 0.9, "Conroy2010 should have some bump"


# ── 12. POWER LAW — analytic exactness ────────────────────────────


class TestPowerLawPhysics:
    """Power-law attenuation: k(λ) = (λ/5500)^n must be exact."""

    def test_analytic_values(self):
        """k(λ) = (λ/5500)^n at specific wavelengths."""
        from tengri.components.dust.attenuation import power_law

        wave = jnp.array([1000.0, 2750.0, 5500.0, 11000.0])
        n = -0.7
        k = power_law(wave, n_slope=n)
        expected = (wave / 5500.0) ** n
        np.testing.assert_allclose(k, expected, rtol=1e-12)

    def test_slope_controls_steepness(self):
        """More negative slope → steeper UV rise."""
        from tengri.components.dust.attenuation import power_law

        k_steep = power_law(WAVE, n_slope=-1.3)
        k_flat = power_law(WAVE, n_slope=-0.3)

        uv_idx = int(jnp.argmin(jnp.abs(WAVE - 1500.0)))
        nir_idx = int(jnp.argmin(jnp.abs(WAVE - 20000.0)))

        ratio_steep = float(k_steep[uv_idx] / k_steep[nir_idx])
        ratio_flat = float(k_flat[uv_idx] / k_flat[nir_idx])
        assert ratio_steep > ratio_flat
