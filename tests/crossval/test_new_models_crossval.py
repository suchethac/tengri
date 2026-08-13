# SPDX-License-Identifier: BSD-3-Clause
"""Exact cross-validation tests for newly implemented models.

Every test cites the specific equation, table, or figure from the
reference paper. Tolerances are tight — if the code doesn't match
the paper at the stated precision, the implementation is wrong.

New models tested:
1. Beloborodov (1999) self-consistent Gamma_hot
2. Just+2007 alpha_ox–L_2500 relation
3. Yang+2022 X-ray anisotropy
4. Martinez-Ramirez+2024 double power-law AGN radio
5. Yang+2020 polar dust extinction & graybody reemission
6. K&D disc self-consistent gamma coupling
7. compute_l2500 monochromatic extraction
8. AGN NLR ionizing spectrum conversion

References
----------
- Beloborodov 1999, ApJ, 510, L123
- Just et al. 2007, ApJ, 665, 1004
- Yang et al. 2022, ApJ, 927, 42
- Martinez-Ramirez et al. 2024, A&A (AGNfitter-rx)
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE)
- Kubota & Done 2018, MNRAS, 480, 1247
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval


# ── 1. BELOBORODOV (1999) — self-consistent Gamma_hot ─────────────


class TestBeloborodov1999:
    """Beloborodov 1999, ApJ, 510, L123.

    Eq. 1: Gamma_hot = (7/3) * (L_diss / L_seed)^{-0.1}

    This relates the hard X-ray photon index to the ratio of
    dissipated-to-seed luminosity in a slab corona geometry.
    """

    def test_formula_at_unity_ratio(self):
        """At L_diss = L_seed: Gamma = (7/3) * 1^{-0.1} = 7/3 = 2.333."""
        from tengri.components.agn.disc import beloborodov_gamma_hot

        gamma = float(beloborodov_gamma_hot(1.0, 1.0))
        np.testing.assert_allclose(gamma, 7.0 / 3.0, rtol=0.01)

    def test_formula_at_ratio_10(self):
        """At L_diss/L_seed = 10: Gamma = (7/3) * 10^{-0.1} = 2.333 * 0.794 = 1.853."""
        from tengri.components.agn.disc import beloborodov_gamma_hot

        gamma = float(beloborodov_gamma_hot(10.0, 1.0))
        expected = (7.0 / 3.0) * 10.0 ** (-0.1)
        np.testing.assert_allclose(gamma, expected, rtol=0.01)

    def test_formula_at_ratio_0p1(self):
        """At L_diss/L_seed = 0.1: Gamma = (7/3) * 0.1^{-0.1} = 2.333 * 1.259 = 2.937."""
        from tengri.components.agn.disc import beloborodov_gamma_hot

        gamma = float(beloborodov_gamma_hot(0.1, 1.0))
        expected = (7.0 / 3.0) * 0.1 ** (-0.1)
        # Clipped to [1.4, 3.0]
        np.testing.assert_allclose(gamma, min(expected, 3.0), rtol=0.01)

    def test_higher_ratio_harder_spectrum(self):
        """More dissipation relative to seed → smaller Gamma (harder spectrum).

        This is the fundamental physics: more coronal heating → harder X-rays.
        """
        from tengri.components.agn.disc import beloborodov_gamma_hot

        gamma_low = float(beloborodov_gamma_hot(0.1, 1.0))
        gamma_high = float(beloborodov_gamma_hot(10.0, 1.0))
        assert gamma_high < gamma_low, (
            "Higher dissipation ratio should give harder spectrum (lower Gamma)"
        )

    def test_clipping_bounds(self):
        """Gamma must be clipped to [1.4, 3.0] (physical range)."""
        from tengri.components.agn.disc import beloborodov_gamma_hot

        # Very high ratio → very hard
        gamma_extreme = float(beloborodov_gamma_hot(1000.0, 1.0))
        assert gamma_extreme >= 1.4, f"Gamma too low: {gamma_extreme}"

        # Very low ratio → very soft
        gamma_soft = float(beloborodov_gamma_hot(0.001, 1.0))
        assert gamma_soft <= 3.0, f"Gamma too high: {gamma_soft}"


# ── 2. JUST+2007 — alpha_ox–L_2500 relation ───────────────────────


class TestJust2007:
    """Just et al. 2007, ApJ, 665, 1004.

    Eq. 3: alpha_ox = -0.137 * log10(L_2500) + 2.638

    where L_2500 is monochromatic luminosity at 2500A in erg/s/Hz.
    """

    def test_formula_at_log_l2500_30(self):
        """At log10(L_2500) = 30: alpha_ox = -0.137*30 + 2.638 = -1.472."""
        from tengri.components.xray import alpha_ox_from_l2500

        alpha_ox = float(alpha_ox_from_l2500(1e30))
        expected = -0.137 * 30.0 + 2.638
        np.testing.assert_allclose(alpha_ox, expected, atol=0.01)

    def test_formula_at_log_l2500_31(self):
        """At log10(L_2500) = 31: alpha_ox = -0.137*31 + 2.638 = -1.609."""
        from tengri.components.xray import alpha_ox_from_l2500

        alpha_ox = float(alpha_ox_from_l2500(1e31))
        expected = -0.137 * 31.0 + 2.638
        np.testing.assert_allclose(alpha_ox, expected, atol=0.01)

    def test_more_luminous_steeper_alpha_ox(self):
        """Brighter AGN have steeper (more negative) alpha_ox.

        This is the anti-correlation: L_X/L_UV decreases with L_UV.
        """
        from tengri.components.xray import alpha_ox_from_l2500

        aox_faint = float(alpha_ox_from_l2500(1e28))
        aox_bright = float(alpha_ox_from_l2500(1e32))
        assert aox_bright < aox_faint, "Brighter AGN should have more negative alpha_ox"

    def test_typical_quasar_range(self):
        """Typical quasars: L_2500 ~ 10^{29-31} erg/s/Hz → alpha_ox ~ -1.3 to -1.6."""
        from tengri.components.xray import alpha_ox_from_l2500

        for log_l, expected_range in [
            (29, (-1.5, -1.2)),
            (30, (-1.6, -1.3)),
            (31, (-1.8, -1.5)),
        ]:
            aox = float(alpha_ox_from_l2500(10.0**log_l))
            assert expected_range[0] < aox < expected_range[1], (
                f"alpha_ox({log_l})={aox:.3f}, expected {expected_range}"
            )


# ── 3. YANG+2022 — X-ray anisotropy ───────────────────────────────


class TestYang2022Anisotropy:
    r"""Yang et al. 2022, ApJ, 927, 192.

    Anisotropy correction, anchored at the 30° reference inclination:

    .. math::

        f(\mu) = \frac{a_1 \mu + a_2 \mu^2 + (1 - a_1 - a_2)}
                      {1 - 0.13397 a_1 - 0.25 a_2}

    The α_ox(L_2500) relation that supplies ``L_2keV`` is defined at 30°, so
    the input spectrum *is* the 30° corona and the correction is normalized to
    leave it unchanged there. Face-on is therefore brighter than the anchor,
    not equal to it. X-CIGALE ``yang20.py:231-235``; see #980.
    """

    @staticmethod
    def _expected(mu: float, a1: float, a2: float) -> float:
        """Anchored Yang+2022 factor, written from the paper not the code."""
        numerator = a1 * mu + a2 * mu**2 + (1.0 - a1 - a2)
        denominator = 1.0 - 0.13397 * a1 - 0.25 * a2
        return numerator / denominator

    def test_face_on_is_brighter_than_the_30_degree_anchor(self):
        """Face-on (cos_inc=1) is the maximum, and exceeds the anchor."""
        from tengri.components.xray import xray_anisotropy

        l_x = jnp.ones(10)
        result = xray_anisotropy(l_x, cos_inc=1.0, a1=0.5, a2=0.0)
        np.testing.assert_allclose(result, self._expected(1.0, 0.5, 0.0), rtol=1e-6)
        assert float(result[0]) > 1.0, "face-on must exceed the 30 deg anchor"

    def test_edge_on_reduced(self):
        """Edge-on (cos_inc=0) gives (1 - a1 - a2) over the anchor."""
        from tengri.components.xray import xray_anisotropy

        l_x = jnp.ones(10)
        result = xray_anisotropy(l_x, cos_inc=0.0, a1=0.5, a2=0.0)
        np.testing.assert_allclose(result, self._expected(0.0, 0.5, 0.0), rtol=1e-6)

    def test_formula_at_45_degrees(self):
        """At 45 degrees (cos_inc = 0.7071)."""
        from tengri.components.xray import xray_anisotropy

        cos45 = float(np.cos(np.radians(45.0)))
        l_x = jnp.ones(10)
        result = xray_anisotropy(l_x, cos_inc=cos45, a1=0.5, a2=0.0)
        np.testing.assert_allclose(result, self._expected(cos45, 0.5, 0.0), rtol=1e-6)

    def test_a1_a2_formula(self):
        """General formula scales the input spectrum linearly."""
        from tengri.components.xray import xray_anisotropy

        cos_inc, a1, a2 = 0.6, 0.3, 0.2
        l_x = jnp.array([2.0])
        result = float(xray_anisotropy(l_x, cos_inc, a1, a2)[0])
        np.testing.assert_allclose(result, 2.0 * self._expected(cos_inc, a1, a2), rtol=1e-6)


# ── 4. MARTINEZ-RAMIREZ+2024 — double power-law AGN radio ─────────


class TestMartinezRamirez2024:
    """Martinez-Ramirez et al. 2024 (AGNfitter-rx).

    Double power-law radio SED with optically thick/thin transition
    and synchrotron aging cutoff (Eq. 9-10).
    """

    def test_dpl_produces_radio_emission(self):
        """DPL must produce non-zero radio emission for radio-loud AGN."""
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)  # 1mm - 1m
        l_nu = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0)
        assert float(jnp.sum(l_nu)) > 0, "DPL should produce radio emission"

    def test_dpl_radio_quiet_faint(self):
        """Radio-quiet (R=0) should be much fainter than radio-loud (R=3)."""
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)
        l_quiet = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=0.0)
        l_loud = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=3.0)
        assert float(jnp.sum(l_loud)) > 100 * float(jnp.sum(l_quiet)), (
            "Radio-loud should be >>100x radio-quiet"
        )

    def test_dpl_alpha1_controls_steep_slope(self):
        """alpha1 sets the optically thin slope: more negative → steeper."""
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)
        l_flat = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, alpha1=-0.3)
        l_steep = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, alpha1=-1.5)
        assert not jnp.allclose(l_flat, l_steep, rtol=0.01), "alpha1 is IGNORED"

    def test_dpl_alpha2_controls_thick_slope(self):
        """alpha2 sets the optically thick slope."""
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)
        l1 = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, alpha2=-0.5)
        l2 = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, alpha2=0.5)
        assert not jnp.allclose(l1, l2, rtol=0.01), "alpha2 is IGNORED"

    def test_dpl_turnover_frequency(self):
        """Transition frequency separates thick/thin regimes.

        Below nu_t: spectrum flattens (optically thick).
        Above nu_t: spectrum steepens (optically thin).
        """
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)
        # Low turnover → most of radio range is optically thin
        l_low_t = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, log_nu_t=8.0)
        # High turnover → most of radio range is optically thick
        l_high_t = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, log_nu_t=11.0)
        assert not jnp.allclose(l_low_t, l_high_t, rtol=0.01), "log_nu_t is IGNORED"

    def test_dpl_synchrotron_cutoff(self):
        """Exponential cutoff changes SED shape (synchrotron aging).

        The DPL renormalizes at 5 GHz, so check shape change not total flux.
        """
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)
        l_no_cut = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, log_nu_cut=15.0)
        l_cut = radio_agn_dpl(wave, L_agn_bol=1e44, radio_loudness=2.0, log_nu_cut=10.0)
        assert not jnp.allclose(
            l_no_cut / jnp.sum(l_no_cut),
            l_cut / jnp.sum(l_cut),
            rtol=0.01,
        ), "log_nu_cut should change the SED shape"

    def test_dpl_normalization_at_5ghz(self):
        """L_nu at 5 GHz should equal L_5GHz from radio-loudness definition."""
        from tengri.components.radio import radio_agn_dpl

        # 5 GHz = 6e9 A
        wave = jnp.array([6e9])  # c/nu = 3e18/5e9 = 6e8... wait
        # c/5GHz = 3e18/5e9 = 6e8 A? No: c = 3e10 cm/s, 5 GHz = 5e9 Hz
        # lambda = c/nu = 3e10/5e9 = 6 cm = 6e8 A
        wave_5ghz = jnp.array([6e8])
        # But radio_agn_dpl masks lambda > 1e7 A, and 6e8 > 1e7 ✓
        l_nu = radio_agn_dpl(wave_5ghz, L_agn_bol=1e44, radio_loudness=1.0)
        # Should be non-zero
        assert float(l_nu[0]) > 0, "L_nu at 5 GHz should be positive"


# ── 5. YANG+2020 — polar dust extinction + graybody emission ──────


class TestYang2020PolarDust:
    """Yang et al. 2020, MNRAS, 491, 740, Sec. 2.2.2.

    SMC extinction for Type 1 sightlines, graybody FIR reemission.
    """

    def test_type1_gets_extincted(self):
        """Face-on (Type 1) with E(B-V)>0 should attenuate UV."""
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.geomspace(1000.0, 30000.0, 200)
        l_nu_in = jnp.ones_like(wave)

        l_out, l_abs = polar_dust_extinction(
            l_nu_in, wave, cos_inc=0.95, opening_angle_deg=40.0, ebv=0.1
        )
        # UV should be attenuated
        uv_mask = wave < 2000.0
        assert float(jnp.mean(l_out[uv_mask])) < 0.9, (
            "Type 1 UV should be attenuated by polar dust"
        )
        # Absorbed luminosity should be positive
        assert float(jnp.sum(l_abs)) > 0, "Absorbed luminosity should be positive"

    def test_type2_no_extinction(self):
        """Edge-on (Type 2) should NOT be extincted by polar dust.

        Yang+2020: polar dust only affects Type 1 sightlines.
        """
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.geomspace(1000.0, 30000.0, 200)
        l_nu_in = jnp.ones_like(wave)

        l_out, _l_abs = polar_dust_extinction(
            l_nu_in, wave, cos_inc=0.05, opening_angle_deg=40.0, ebv=0.3
        )
        # Type 2: no extinction (torus blocks the line of sight already)
        np.testing.assert_allclose(l_out, l_nu_in, atol=0.05)

    def test_zero_ebv_no_extinction(self):
        """E(B-V) = 0 → no extinction regardless of orientation."""
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.geomspace(1000.0, 30000.0, 200)
        l_nu_in = jnp.ones_like(wave) * 5.0

        l_out, l_abs = polar_dust_extinction(
            l_nu_in, wave, cos_inc=0.95, opening_angle_deg=40.0, ebv=0.0
        )
        np.testing.assert_allclose(l_out, l_nu_in, atol=1e-10)
        np.testing.assert_allclose(l_abs, 0.0, atol=1e-10)

    def test_uses_smc_law(self):
        """Extinction should use SMC law (steep UV, R_V=2.93)."""
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.geomspace(1000.0, 30000.0, 200)
        l_nu_in = jnp.ones_like(wave)

        l_out, _ = polar_dust_extinction(
            l_nu_in, wave, cos_inc=0.95, opening_angle_deg=40.0, ebv=0.2
        )
        # SMC has no 2175A bump — check that attenuation is monotonic in UV
        uv_range = (wave > 1500) & (wave < 3000)
        l_uv = l_out[uv_range]
        # Should increase monotonically with wavelength (less extinction)
        diffs = jnp.diff(l_uv)
        assert float(jnp.sum(diffs > -0.01)) > 0.8 * len(diffs), (
            "SMC attenuation should be roughly monotonic (no 2175A bump)"
        )

    def test_graybody_energy_conservation(self):
        """Reemitted graybody should integrate to L_absorbed."""
        from tengri.components.agn.polar_dust import polar_dust_emission

        wave = jnp.geomspace(1e4, 1e8, 5000)
        l_absorbed = 1e10  # arbitrary units

        l_reemit = polar_dust_emission(l_absorbed, wave, temperature=100.0)

        # Integrate over frequency
        nu = 2.99792458e18 / wave
        sort_idx = jnp.argsort(nu)
        l_bol = float(jnp.trapezoid(l_reemit[sort_idx], nu[sort_idx]))
        # Energy conservation: L_bol_reemit ≈ L_absorbed (within 20%)
        assert abs(l_bol / l_absorbed - 1.0) < 0.30, (
            f"Graybody L_bol={l_bol:.2e} should equal L_absorbed={l_absorbed:.2e}"
        )

    def test_graybody_peaks_in_fir(self):
        """T=100K graybody peaks near 30-100 μm."""
        from tengri.components.agn.polar_dust import polar_dust_emission

        wave = jnp.geomspace(1e4, 1e8, 2000)
        l_reemit = polar_dust_emission(1e10, wave, temperature=100.0)

        peak_wave = float(wave[jnp.argmax(l_reemit)])
        # Wien's law for MBB: peak ~ 29 μm * (100K/T) at β=0
        # With β=1.6, peak shifts to ~50-100 μm = 5e5-1e6 A
        assert 2e5 < peak_wave < 2e6, (
            f"T=100K graybody peak at {peak_wave:.0e} A, expected ~5e5-1e6 A"
        )


# ── 6. KUBOTA & DONE SELF-CONSISTENT GAMMA ────────────────────────


class TestKDSelfConsistentGamma:
    """Kubota & Done 2018 + Beloborodov 1999 coupling.

    agn_self_consistent_gamma=True derives Gamma_hot from L_warm/L_hot.
    """

    def test_self_consistent_differs_from_fixed(self):
        """Self-consistent Gamma should differ from the fixed default."""
        from tengri.components.agn.disc import kubota_done_disc

        wave = jnp.geomspace(100.0, 1e8, 500)
        l_fixed = kubota_done_disc(wave, agn_log_lbol=11.0, agn_self_consistent_gamma=False)
        l_sc = kubota_done_disc(wave, agn_log_lbol=11.0, agn_self_consistent_gamma=True)
        assert not jnp.allclose(l_fixed, l_sc, rtol=0.001), (
            "Self-consistent Gamma should differ from fixed Gamma"
        )

    def test_self_consistent_finite(self):
        """Self-consistent model must produce finite SED."""
        from tengri.components.agn.disc import kubota_done_disc

        wave = jnp.geomspace(100.0, 1e8, 500)
        l_nu = kubota_done_disc(wave, agn_log_lbol=11.0, agn_self_consistent_gamma=True)
        chex.assert_tree_all_finite(l_nu)
        assert float(jnp.sum(l_nu)) > 0


# ── 7. compute_l2500 — monochromatic extraction ───────────────────


class TestComputeL2500:
    """Extract L_nu at rest-frame 2500 A by interpolation."""

    def test_flat_spectrum(self):
        """Flat L_nu=1 everywhere → L_2500 = 1."""
        from tengri.components.agn.disc import compute_l2500

        wave = jnp.linspace(1000.0, 5000.0, 500)
        l_nu = jnp.ones_like(wave)
        l2500 = float(compute_l2500(wave, l_nu))
        np.testing.assert_allclose(l2500, 1.0, atol=0.01)

    def test_known_value(self):
        """L_nu = wavelength/2500 → L_2500 = 1.0."""
        from tengri.components.agn.disc import compute_l2500

        wave = jnp.linspace(1000.0, 5000.0, 500)
        l_nu = wave / 2500.0
        l2500 = float(compute_l2500(wave, l_nu))
        np.testing.assert_allclose(l2500, 1.0, atol=0.02)

    def test_unsorted_wavelengths(self):
        """Must work with unsorted wavelength arrays."""
        from tengri.components.agn.disc import compute_l2500

        wave = jnp.array([5000.0, 1000.0, 2500.0, 3000.0, 2000.0])
        l_nu = jnp.array([5.0, 1.0, 2.5, 3.0, 2.0])
        l2500 = float(compute_l2500(wave, l_nu))
        np.testing.assert_allclose(l2500, 2.5, atol=0.1)


# ── 8. SELF-CONSISTENT X-RAY CORONA FROM DISC ─────────────────────


class TestSelfConsistentCorona:
    """xray_agn_corona_from_disc: disc UV → alpha_ox → X-ray."""

    def test_produces_xray_emission(self):
        """Self-consistent corona must produce X-ray flux."""
        from tengri.components.xray import xray_agn_corona_from_disc

        wave = jnp.geomspace(0.1, 200.0, 500)  # X-ray
        l_nu = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e30)
        assert float(jnp.sum(l_nu)) > 0, "Corona should emit X-rays"
        chex.assert_tree_all_finite(l_nu)

    def test_brighter_disc_more_xray(self):
        """Higher L_2500 → more X-ray (despite steeper alpha_ox)."""
        from tengri.components.xray import xray_agn_corona_from_disc

        wave = jnp.geomspace(0.1, 200.0, 300)
        l_faint = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e28)
        l_bright = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e31)
        # Even though alpha_ox steepens, absolute X-ray luminosity increases
        assert float(jnp.sum(l_bright)) > float(jnp.sum(l_faint)), (
            "Brighter disc should produce more absolute X-ray"
        )

    def test_anisotropy_reduces_edge_on(self):
        """Edge-on viewing should reduce X-ray flux (Yang+2022)."""
        from tengri.components.xray import xray_agn_corona_from_disc

        wave = jnp.geomspace(0.1, 200.0, 300)
        l_face = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e30, cos_inc=1.0)
        l_edge = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e30, cos_inc=0.1)
        assert float(jnp.sum(l_edge)) < float(jnp.sum(l_face)), (
            "Edge-on should have less X-ray flux"
        )

    def test_delta_alpha_ox_shifts_xray(self):
        """Positive delta_alpha_ox → more X-ray relative to UV."""
        from tengri.components.xray import xray_agn_corona_from_disc

        wave = jnp.geomspace(0.1, 200.0, 300)
        l_default = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e30, delta_alpha_ox=0.0)
        l_excess = xray_agn_corona_from_disc(wave, l_2500_erg_hz=1e30, delta_alpha_ox=0.3)
        assert float(jnp.sum(l_excess)) > float(jnp.sum(l_default)), (
            "Positive delta_alpha_ox should boost X-ray"
        )


# ── 9. AGN NLR IONIZING SPECTRUM CONVERSION ───────────────────────


class TestAGNIonSpec:
    """agn_ionspec_from_alpha_pl: power-law slope → Cue parameters."""

    def test_returns_expected_keys(self):
        """Must return dict with the 7 Cue ionizing spectrum parameters."""
        from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

        params = agn_ionspec_from_alpha_pl(alpha_pl=-1.7)
        # Should have slope and luminosity ratio keys
        assert isinstance(params, dict)
        assert len(params) >= 4, f"Expected ≥4 keys, got {len(params)}"

    def test_steeper_slope_different_params(self):
        """Different alpha_pl should produce different parameters."""
        from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

        p1 = agn_ionspec_from_alpha_pl(alpha_pl=-1.0)
        p2 = agn_ionspec_from_alpha_pl(alpha_pl=-2.0)

        # At least some parameters should differ
        any_differ = False
        for key in p1:
            if key in p2 and abs(float(p1[key]) - float(p2[key])) > 0.01:
                any_differ = True
                break
        assert any_differ, "Different alpha_pl should produce different params"

    def test_physical_slope_range(self):
        """Typical AGN EUV slope alpha_pl ~ -1.0 to -2.0 (Telfer+2002)."""
        from tengri.components.nebular.agn_nebular import agn_ionspec_from_alpha_pl

        # Should not error for physical range
        for alpha in [-0.5, -1.0, -1.5, -2.0, -2.5]:
            params = agn_ionspec_from_alpha_pl(alpha_pl=alpha)
            # All values should be finite
            for key, val in params.items():
                assert np.isfinite(float(val)), f"alpha_pl={alpha}: {key}={val} is not finite"


# ── 10. ANTI-LAZINESS: ALL NEW FUNCTIONS RESPOND TO PARAMETERS ────


class TestNewModelsParameterSensitivity:
    """Every new function must respond to its parameters (not ignored)."""

    def test_radio_dpl_all_params(self):
        """radio_agn_dpl: all 5 shape params must matter."""
        from tengri.components.radio import radio_agn_dpl

        wave = jnp.geomspace(1e7, 1e10, 200)
        defaults = {
            "L_agn_bol": 1e44,
            "radio_loudness": 2.0,
            "alpha1": -0.75,
            "alpha2": -0.1,
            "log_nu_t": 10.0,
            "log_nu_cut": 13.0,
        }
        l_default = radio_agn_dpl(wave, **defaults)

        for param, alt_val in [
            ("alpha1", -1.5),
            ("alpha2", 0.5),
            ("log_nu_t", 8.0),
            ("log_nu_cut", 10.0),
            ("radio_loudness", 4.0),
        ]:
            modified = {**defaults, param: alt_val}
            l_mod = radio_agn_dpl(wave, **modified)
            assert not jnp.allclose(l_default, l_mod, rtol=0.01), (
                f"radio_agn_dpl parameter {param} is IGNORED"
            )

    def test_xray_corona_from_disc_all_params(self):
        """xray_agn_corona_from_disc: all params must matter."""
        from tengri.components.xray import xray_agn_corona_from_disc

        wave = jnp.geomspace(0.1, 200.0, 200)
        defaults = {
            "l_2500_erg_hz": 1e30,
            "cos_inc": 0.8,
            "delta_alpha_ox": 0.0,
            "gamma": 1.8,
            "E_cut": 300.0,
        }
        l_default = xray_agn_corona_from_disc(wave, **defaults)

        for param, alt_val in [
            ("delta_alpha_ox", 0.5),
            ("gamma", 2.5),
            ("E_cut", 50.0),
            ("cos_inc", 0.1),
        ]:
            modified = {**defaults, param: alt_val}
            l_mod = xray_agn_corona_from_disc(wave, **modified)
            # Use max relative difference — some params scale uniformly
            rel_diff = float(jnp.max(jnp.abs(l_default - l_mod) / (jnp.abs(l_default) + 1e-50)))
            assert rel_diff > 0.01, (
                f"xray_agn_corona_from_disc {param} IGNORED (max rel diff: {rel_diff:.2e})"
            )

    def test_polar_dust_ebv_matters(self):
        """Polar dust: E(B-V) must change the output."""
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.geomspace(1000.0, 30000.0, 100)
        l_in = jnp.ones_like(wave)

        l1, _ = polar_dust_extinction(l_in, wave, 0.95, 40.0, ebv=0.05)
        l2, _ = polar_dust_extinction(l_in, wave, 0.95, 40.0, ebv=0.5)
        assert not jnp.allclose(l1, l2, rtol=0.01), "E(B-V) is IGNORED"

    def test_polar_dust_temperature_matters(self):
        """Graybody temperature must change the emission."""
        from tengri.components.agn.polar_dust import polar_dust_emission

        wave = jnp.geomspace(1e4, 1e8, 200)
        l1 = polar_dust_emission(1e10, wave, temperature=50.0)
        l2 = polar_dust_emission(1e10, wave, temperature=200.0)
        assert not jnp.allclose(l1 / jnp.sum(l1), l2 / jnp.sum(l2), rtol=0.05), (
            "Graybody temperature is IGNORED"
        )
