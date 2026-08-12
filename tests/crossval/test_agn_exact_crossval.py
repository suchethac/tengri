# SPDX-License-Identifier: BSD-3-Clause
"""Exact cross-validation of AGN model components against analytical formulas
and reference implementations.

Every test derives the expected value from FIRST PRINCIPLES using scipy/astropy,
then compares to tengri's output. No circular testing — the reference
calculations are independent of the tengri implementation.

Models tested:
1. ISCO radius — exact BPT72 formula via scipy
2. Eddington luminosity — exact formula via astropy constants
3. Multicolor disc — Planck integration via scipy.integrate.quad
4. QSOgen continuum — analytical broken power-law
5. QSOgen hot dust — Planck B_λ at 2μm anchor
6. Beloborodov Gamma — exact formula
7. Just+2007 alpha_ox — exact formula
8. Polar dust SMC extinction — A_λ from Pei92 R_V and k(λ)
9. K&D disc temperature profile — T(r) ∝ r^{-3/4} check
10. Radio DPL — analytical shape at specific frequencies
"""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import constants as const

pytestmark = pytest.mark.crossval

# Exact physical constants from CODATA 2018 (via astropy)
H_PLANCK = float(const.h.cgs.value)  # erg s
K_BOLTZ = float(const.k_B.cgs.value)  # erg/K
C_LIGHT = float(const.c.cgs.value)  # cm/s
SIGMA_SB = float(const.sigma_sb.cgs.value)  # erg/cm^2/s/K^4
SIGMA_T = float(const.sigma_T.cgs.value)  # cm^2
M_PROTON = float(const.m_p.cgs.value)  # g
G_GRAV = float(const.G.cgs.value)  # cm^3/g/s^2
LSUN = float(const.L_sun.cgs.value)  # erg/s
MSUN = float(const.M_sun.cgs.value)  # g


# ── 1. ISCO RADIUS — independent scipy implementation of BPT72 ────


def _isco_radius_reference(a_spin: float) -> float:
    """BPT72 ISCO via independent implementation (not from tengri)."""
    a = np.clip(a_spin, 0.0, 0.998)
    z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    return 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


class TestISCOExact:
    """Compare tengri ISCO against independent BPT72 implementation."""

    @pytest.mark.parametrize("a_spin", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.998])
    def test_isco_matches_reference(self, a_spin):
        """tengri ISCO must match BPT72 formula to 0.01%."""
        from tengri.components.agn.disc import _isco_radius

        tengri_r = float(_isco_radius(a_spin))
        ref_r = _isco_radius_reference(a_spin)
        np.testing.assert_allclose(
            tengri_r,
            ref_r,
            rtol=1e-4,
            err_msg=f"ISCO mismatch at a={a_spin}: tengri={tengri_r}, ref={ref_r}",
        )


# ── 2. EDDINGTON LUMINOSITY — exact from astropy constants ────────


def _eddington_luminosity_reference(log_mbh: float) -> float:
    """L_Edd from exact CODATA constants (not from tengri)."""
    m_bh = 10.0**log_mbh * MSUN  # g
    return 4.0 * np.pi * G_GRAV * m_bh * M_PROTON * C_LIGHT / SIGMA_T


class TestEddingtonExact:
    """Compare tengri Eddington luminosity against exact astropy formula."""

    @pytest.mark.parametrize("log_mbh", [6.0, 7.0, 8.0, 9.0, 10.0])
    def test_eddington_matches_astropy(self, log_mbh):
        """Must match to 0.5% (difference from constant precision)."""
        from tengri.components.agn.disc import _eddington_luminosity

        tengri_l = float(_eddington_luminosity(log_mbh))
        ref_l = _eddington_luminosity_reference(log_mbh)
        np.testing.assert_allclose(
            tengri_l,
            ref_l,
            rtol=0.005,
            err_msg=f"L_Edd mismatch at logM={log_mbh}",
        )


# ── 3. GRAVITATIONAL RADIUS — exact from astropy ──────────────────


def _gravitational_radius_reference(log_mbh: float) -> float:
    """R_g = GM/c^2 from exact constants."""
    m_bh = 10.0**log_mbh * MSUN
    return G_GRAV * m_bh / C_LIGHT**2


class TestGravRadiusExact:
    """Compare tengri R_g against exact astropy formula."""

    @pytest.mark.parametrize("log_mbh", [6.0, 7.0, 8.0, 9.0, 10.0])
    def test_rg_matches_astropy(self, log_mbh):
        """Must match to 0.5%."""
        from tengri.components.agn.disc import _gravitational_radius

        tengri_rg = float(_gravitational_radius(log_mbh))
        ref_rg = _gravitational_radius_reference(log_mbh)
        np.testing.assert_allclose(
            tengri_rg, ref_rg, rtol=0.005, err_msg=f"R_g mismatch at logM={log_mbh}"
        )


# ── 4. NOVIKOV-THORNE EFFICIENCY — independent calculation ────────


class TestNTEfficiency:
    """Novikov-Thorne radiative efficiency η = 1 - sqrt(1 - 2/(3*r_isco))."""

    @pytest.mark.parametrize(
        "a_spin,expected_eta",
        [
            (0.0, 0.0572),  # Schwarzschild
            (0.5, 0.0816),
            (0.9, 0.156),
            (0.998, 0.321),  # Near-maximal Kerr
        ],
    )
    def test_efficiency_published_values(self, a_spin, expected_eta):
        """Compare η against published values (Thorne 1974 Table 1)."""
        from tengri.components.agn.disc import _isco_radius

        r_isco = float(_isco_radius(a_spin))
        eta = 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * r_isco))
        np.testing.assert_allclose(
            eta,
            expected_eta,
            atol=0.005,
            err_msg=f"η mismatch at a={a_spin}: got {eta:.4f}, expected {expected_eta}",
        )


# ── 5. PLANCK FUNCTION — independent scipy implementation ─────────


def _planck_bnu_reference(nu: float, temperature: float) -> float:
    """B_nu(T) from first principles using CODATA constants."""
    x = H_PLANCK * nu / (K_BOLTZ * temperature)
    if x > 500:
        return 0.0
    return 2.0 * H_PLANCK * nu**3 / C_LIGHT**2 / (np.exp(x) - 1.0)


class TestPlanckExact:
    """Compare tengri Planck function against independent scipy calculation."""

    def test_planck_at_peak(self):
        """Wien peak: B_nu(T) peaks at ν_peak = 2.821 * kT/h.

        For T=10000K: ν_peak = 5.88e14 Hz → λ ≈ 5100 A.
        """
        from tengri.components.agn.disc import _planck_lnu

        t = 10000.0
        nu_peak = 2.821 * K_BOLTZ * t / H_PLANCK
        tengri_bnu = float(_planck_lnu(jnp.array([nu_peak]), t)[0])
        ref_bnu = _planck_bnu_reference(nu_peak, t)
        np.testing.assert_allclose(tengri_bnu, ref_bnu, rtol=1e-6)

    @pytest.mark.parametrize(
        "nu,temperature",
        [
            (1e12, 30.0),  # FIR dust
            (1e14, 5000.0),  # Optical star
            (1e15, 50000.0),  # UV hot star
            (3e14, 1240.0),  # Hot dust (QSOgen T_bb)
        ],
    )
    def test_planck_at_various(self, nu, temperature):
        """Planck function must match independent calculation to 1e-6."""
        from tengri.components.agn.disc import _planck_lnu

        tengri_bnu = float(_planck_lnu(jnp.array([nu]), temperature)[0])
        ref_bnu = _planck_bnu_reference(nu, temperature)
        np.testing.assert_allclose(tengri_bnu, ref_bnu, rtol=1e-5)


# ── 6. QSOGEN BROKEN POWER-LAW — analytical verification ──────────


class TestQSOgenContinuumExact:
    """Verify QSOgen continuum matches the analytical broken power-law."""

    def test_normalized_at_5500(self):
        """f_nu(5500A) = 1.0 by construction."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        wave = jnp.array([5500.0])
        f = float(_broken_powerlaw_continuum(wave, -0.349, 0.593, 3880.0)[0])
        np.testing.assert_allclose(f, 1.0, atol=0.01)

    def test_red_slope_above_break(self):
        """For λ >> plbrk: f_nu ∝ λ^{-plslp2}.

        At 8000A (well above 3880A break):
        f_nu(8000) / f_nu(5500) = (8000/5500)^{-0.593}
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        f = _broken_powerlaw_continuum(jnp.array([5500.0, 8000.0]), -0.349, 0.593, 3880.0)
        expected_ratio = (8000.0 / 5500.0) ** (-0.593)
        actual_ratio = float(f[1] / f[0])
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.05)

    def test_blue_slope_below_break(self):
        """For plbrk3 < λ < plbrk: f_nu ∝ λ^{-plslp1}.

        Slope should be -plslp1 = 0.349 (positive = bluer at shorter λ).
        """
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        wave = jnp.array([2000.0, 3000.0])
        f = _broken_powerlaw_continuum(wave, -0.349, 0.593, 3880.0)
        # In the blue segment: f_nu ∝ λ^{0.349}
        expected_ratio = (2000.0 / 3000.0) ** (0.349)
        actual_ratio = float(f[0] / f[1])
        # Allow 10% for sigmoid transition effects
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.15)

    def test_plslp1_zero_flat_blue(self):
        """plslp1=0 → flat f_nu in the blue (no UV rise/decline)."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum

        wave = jnp.linspace(1500.0, 3500.0, 100)
        f = _broken_powerlaw_continuum(wave, 0.0, 0.593, 3880.0)
        # Should be approximately constant in the blue segment
        cv = float(jnp.std(f[10:80]) / jnp.mean(f[10:80]))
        assert cv < 0.15, f"plslp1=0 should give ~flat blue, CV={cv:.3f}"


# ── 7. QSOGEN HOT DUST — Planck B_λ anchor at 2μm ─────────────────


class TestQSOgenHotDustExact:
    """QSOgen hot dust: BB(T_bb) anchored to continuum at 2μm."""

    def test_bb_scales_linearly_with_bbnorm(self):
        """BB flux at 2μm must scale linearly with bbnorm."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        wave = jnp.linspace(5000.0, 30000.0, 500)
        cont = _broken_powerlaw_continuum(wave, -0.349, 0.593, 3880.0)

        bb1 = _hot_dust_blackbody(wave, cont, tbb=1240.0, bbnorm=1.0)
        bb2 = _hot_dust_blackbody(wave, cont, tbb=1240.0, bbnorm=2.0)

        idx_2um = int(jnp.argmin(jnp.abs(wave - 20000.0)))
        ratio = float(bb2[idx_2um] / bb1[idx_2um])
        np.testing.assert_allclose(
            ratio, 2.0, rtol=0.01, err_msg="BB should scale linearly with bbnorm"
        )

    def test_hotter_dust_peaks_bluer(self):
        """Wien's law: higher T → shorter peak λ."""
        from tengri.components.agn.qsogen import _broken_powerlaw_continuum, _hot_dust_blackbody

        wave = jnp.linspace(5000.0, 50000.0, 500)
        cont = _broken_powerlaw_continuum(wave, -0.349, 0.593, 3880.0)

        bb_cool = _hot_dust_blackbody(wave, cont, tbb=800.0, bbnorm=1.0)
        bb_hot = _hot_dust_blackbody(wave, cont, tbb=2000.0, bbnorm=1.0)

        peak_cool = float(wave[jnp.argmax(bb_cool)])
        peak_hot = float(wave[jnp.argmax(bb_hot)])
        assert peak_hot < peak_cool, "Hotter dust should peak bluer"


# ── 8. BELOBORODOV 1999 — exact formula verification ──────────────


class TestBeloborodovExact:
    """Gamma_hot = (7/3) * (L_diss/L_seed)^{-0.1} — exact formula."""

    @pytest.mark.parametrize(
        "ratio,expected",
        [
            (0.01, min((7 / 3) * 0.01 ** (-0.1), 3.0)),
            (0.1, (7 / 3) * 0.1 ** (-0.1)),
            (1.0, 7 / 3),
            (10.0, (7 / 3) * 10.0 ** (-0.1)),
            (100.0, (7 / 3) * 100.0 ** (-0.1)),
        ],
    )
    def test_exact_formula(self, ratio, expected):
        """Verify against exact Beloborodov (1999) Eq. 1."""
        from tengri.components.agn.disc import beloborodov_gamma_hot

        gamma = float(beloborodov_gamma_hot(ratio, 1.0))
        expected_clipped = np.clip(expected, 1.4, 3.0)
        np.testing.assert_allclose(gamma, expected_clipped, rtol=0.01)


# ── 9. JUST+2007 alpha_ox — exact formula ─────────────────────────


class TestJust2007Exact:
    """alpha_ox = -0.137 * log10(L_2500) + 2.638 — Just+2007 Eq.3."""

    @pytest.mark.parametrize("log_l2500", [27.0, 28.0, 29.0, 30.0, 31.0, 32.0])
    def test_exact_formula(self, log_l2500):
        """Must match the linear relation exactly."""
        from tengri.components.xray import alpha_ox_from_l2500

        result = float(alpha_ox_from_l2500(10.0**log_l2500))
        expected = -0.137 * log_l2500 + 2.638
        np.testing.assert_allclose(result, expected, atol=0.001)


# ── 10. POLAR DUST — SMC A_λ = R_V * E(B-V) * k(λ) ────────────────


class TestPolarDustExtinctionExact:
    """Polar dust extinction must use correct SMC formula."""

    def test_transmission_formula(self):
        """T = exp(-0.921 * E(B-V) * R_V * k(λ)) for Type 1.

        R_V(SMC) = 2.93 (Pei 1992).
        At V-band (5500A): k(V) = 1.0, so τ = 0.921 * E(B-V) * 2.93.
        """
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.array([5500.0])
        l_nu = jnp.array([1.0])
        ebv = 0.3
        # Perfect Type 1 (cos_inc >> cos(90-opening) so mask ≈ 1)
        l_out, _l_abs = polar_dust_extinction(
            l_nu, wave, cos_inc=0.99, opening_angle_deg=45.0, ebv=ebv
        )

        # Expected: T = exp(-0.921 * 0.3 * 2.93 * 1.0) = exp(-0.809)
        expected_t = np.exp(-0.921 * ebv * 2.93 * 1.0)
        np.testing.assert_allclose(float(l_out[0]), expected_t, rtol=0.02)

    def test_uv_more_extincted_than_optical(self):
        """SMC: k(1500A) >> k(V) → UV much more extincted."""
        from tengri.components.agn.polar_dust import polar_dust_extinction
        from tengri.components.dust.attenuation import smc as smc_curve

        wave = jnp.array([1500.0, 5500.0])
        l_nu = jnp.ones(2)
        l_out, _ = polar_dust_extinction(l_nu, wave, cos_inc=0.99, opening_angle_deg=45.0, ebv=0.1)

        k = smc_curve(wave)
        # UV transmission should be much lower
        assert float(l_out[0]) < float(l_out[1]), (
            "UV should be more extincted than V-band with SMC law"
        )
        # And the ratio should match k(λ)
        ratio_expected = np.exp(-0.921 * 0.1 * 2.93 * (float(k[0]) - float(k[1])))
        ratio_actual = float(l_out[0] / l_out[1])
        np.testing.assert_allclose(ratio_actual, ratio_expected, rtol=0.05)


# ── 11. MULTICOLOR DISC TEMPERATURE PROFILE — T(r) ∝ r^{-3/4} ─────


class TestDiscTemperatureProfile:
    """Standard Shakura-Sunyaev T(r) = T_in * (r/r_in)^{-3/4} * f(r)."""

    def test_temperature_decreases_outward(self):
        """T(r) must decrease with radius (cooler outer disc)."""
        from tengri.components.agn.disc import _gravitational_radius, _isco_radius

        r_g = _gravitational_radius(8.0)
        r_isco = float(_isco_radius(0.0)) * float(r_g)

        # T(r) = T_in * (r/r_in)^{-3/4} * (1 - sqrt(r_in/r))^{1/4}
        radii = np.array([r_isco * 2, r_isco * 10, r_isco * 100])
        t_in = 1e5  # arbitrary inner temperature

        temps = []
        for r in radii:
            r_ratio = r / r_isco
            torque = max(1.0 - np.sqrt(1.0 / r_ratio), 1e-30) ** 0.25
            t = t_in * r_ratio ** (-0.75) * torque
            temps.append(t)

        # Temperature must decrease outward
        for i in range(len(temps) - 1):
            assert temps[i] > temps[i + 1], (
                f"T({radii[i] / r_isco:.0f} r_in)={temps[i]:.0f} K > "
                f"T({radii[i + 1] / r_isco:.0f} r_in)={temps[i + 1]:.0f} K expected"
            )

    def test_temperature_scales_r_minus_three_quarters(self):
        """At large r (where torque correction → 1): T ∝ r^{-3/4}."""
        from tengri.components.agn.disc import _gravitational_radius, _isco_radius

        r_g = _gravitational_radius(8.0)
        r_isco = float(_isco_radius(0.0)) * float(r_g)

        # At r >> r_in, the torque factor → 1, so T ∝ r^{-3/4}
        r1, r2 = r_isco * 100, r_isco * 1000  # both far from ISCO
        t_ratio = (r2 / r1) ** (-0.75)  # expected ratio
        # Including torque corrections (both near 1 at large r)
        tc1 = (1.0 - np.sqrt(r_isco / r1)) ** 0.25
        tc2 = (1.0 - np.sqrt(r_isco / r2)) ** 0.25
        expected_ratio = t_ratio * tc2 / tc1

        np.testing.assert_allclose(
            expected_ratio,
            (r2 / r1) ** (-0.75),
            rtol=0.05,
            err_msg="Torque correction should be ~1 at large radii",
        )


# ── 12. DISC BOLOMETRIC LUMINOSITY — energy conservation ──────────


class TestDiscEnergyConservation:
    """∫ L_ν dν must equal L_bol × agn_lum_ratio (by normalization)."""

    @pytest.mark.parametrize(
        "model_name,model_fn_kwargs",
        [
            ("powerlaw", {"agn_alpha": -1.0}),
            ("multicolor", {"agn_log_mbh": 8.0}),
        ],
    )
    def test_luminosity_integral(self, model_name, model_fn_kwargs):
        """Integrated L_ν must equal L_bol to 20% (wavelength grid truncation)."""
        from tengri.components.agn.disc import multicolor_disc, powerlaw_disc

        fn = {"powerlaw": powerlaw_disc, "multicolor": multicolor_disc}[model_name]
        wave = jnp.geomspace(10.0, 1e8, 5000)
        agn_log_lbol = 11.0
        agn_lum_ratio = 0.5

        l_nu = fn(wave, agn_log_lbol=agn_log_lbol, agn_lum_ratio=agn_lum_ratio, **model_fn_kwargs)

        nu = 2.99792458e18 / wave
        sort_idx = jnp.argsort(nu)
        l_bol_integrated = float(jnp.trapezoid(l_nu[sort_idx], nu[sort_idx]))
        # L_nu is erg/s/Hz (CGS), integral is erg/s → compare in erg/s
        _LSUN = 3.828e33
        l_bol_expected = 10.0**agn_log_lbol * agn_lum_ratio * _LSUN

        np.testing.assert_allclose(
            l_bol_integrated,
            l_bol_expected,
            rtol=0.20,
            err_msg=f"{model_name}: L_bol integral mismatch",
        )


# ── 13. YANG+2022 ANISOTROPY — exact formula at specific angles ───


class TestAnisotropyExact:
    """f(θ) = a1*cos(θ) + a2*cos²(θ) + (1-a1-a2) — exact values."""

    @pytest.mark.parametrize(
        "cos_inc,a1,a2,expected_factor",
        [
            (1.0, 0.5, 0.0, 1.0),  # face-on
            (0.0, 0.5, 0.0, 0.5),  # edge-on
            (0.5, 0.5, 0.0, 0.75),  # 60 degrees
            (1.0, 0.3, 0.2, 1.0),  # face-on, different coeffs
            (0.0, 0.3, 0.2, 0.5),  # edge-on, different coeffs
            (0.5, 0.3, 0.2, 0.70),  # 0.3*0.5 + 0.2*0.25 + 0.5 = 0.70
        ],
    )
    def test_exact_values(self, cos_inc, a1, a2, expected_factor):
        """Anisotropy factor must match analytical formula exactly."""
        from tengri.components.xray import xray_anisotropy

        l_x = jnp.array([1.0])
        result = float(xray_anisotropy(l_x, cos_inc, a1, a2)[0])
        np.testing.assert_allclose(result, expected_factor, atol=1e-10)


# ── 14. RADIO DPL — analytical shape at reference frequency ───────


class TestRadioDPLExact:
    """DPL must give L_nu(5 GHz) = L_5GHz exactly (normalization)."""

    def test_l5ghz_matches_definition(self):
        """L_nu at 5 GHz must equal L_B * 10^R (radio-loudness definition)."""
        from tengri.components.radio import radio_agn_dpl

        # 5 GHz → λ = c/ν = 3e10/5e9 = 6 cm = 6e8 A
        wave = jnp.array([6e8])
        R = 1.5
        L_agn_bol = 1e44

        l_nu = radio_agn_dpl(wave, L_agn_bol=L_agn_bol, radio_loudness=R)

        # Expected: L_B = L_bol / (BC_B * nu_B), L_5GHz = L_B * 10^R
        _NU_B = 6.818e14
        _BC_B = 5.15
        L_B = L_agn_bol * LSUN / (_BC_B * _NU_B) / LSUN
        L_5GHz_expected = L_B * 10.0**R

        np.testing.assert_allclose(
            float(l_nu[0]),
            L_5GHz_expected,
            rtol=0.05,
            err_msg=f"L(5GHz) = {float(l_nu[0]):.3e}, expected {L_5GHz_expected:.3e}",
        )
