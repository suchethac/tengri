# SPDX-License-Identifier: BSD-3-Clause
"""Physics-motivated numerical tests for multiwavelength SED components.

ALL inputs use CGS units (erg/s for luminosities) to match the documented
function interfaces. ALL outputs are checked as absolute erg/s/Hz values,
not just ratios or trends.

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
_LSUN = 3.828e33  # erg/s

# Standard L_IR values in erg/s for SFR calibration
_L_IR_SFR1 = 1e10 * _LSUN  # L_IR for SFR ~ 1 Msun/yr (Kennicutt 1998)
_L_IR_SFR10 = 1e11 * _LSUN  # L_IR for SFR ~ 10 Msun/yr


# ── 1. Radio: absolute values in erg/s/Hz ─────────────────────────


class TestRadioAbsoluteValues:
    """Radio luminosities in erg/s/Hz against published calibrations."""

    def test_bell2003_l14ghz_absolute(self):
        """Bell 2003: L_1.4GHz = L_IR / (3.75e12 × 10^q) erg/s/Hz.

        At L_IR = 1e10 Lsun = 3.828e43 erg/s, q=2.64:
        L_1.4GHz = 3.828e43 / (3.75e12 × 10^2.64) = 2.34e28 erg/s/Hz.
        Murphy+2011 gives 1.81e28 erg/s/Hz for SFR=1 (Kroupa→Chabrier offset).
        """
        from tengri.components.radio import radio_star_forming

        wave = jnp.array([_C_AA / 1.4e9])
        l_14 = float(radio_star_forming(wave, L_ir=_L_IR_SFR1, q_ir=2.64)[0])

        expected = _L_IR_SFR1 / (3.75e12 * 10.0**2.64)
        np.testing.assert_allclose(
            l_14, expected, rtol=0.01, err_msg=f"L_1.4GHz = {l_14:.2e}, expected {expected:.2e}"
        )

        # Absolute range check: ~2e28 erg/s/Hz for SFR~1
        assert 1e27 < l_14 < 1e29, f"L_1.4GHz = {l_14:.2e}, expected ~2e28 erg/s/Hz"

    def test_bell2003_q_ir_slope(self):
        """q_IR controls normalization: each unit increase halves L_1.4GHz by 10x.

        Cross-checks the 10^q_IR denominator using two independent q values
        from the observed distribution (Bell 2003, Table 3: median 2.64, spread ~0.2).
        Ratio test is non-circular: expected ratio (10^1.0 = 10) comes from
        the exponent definition, not from the function output.
        """
        from tengri.components.radio import radio_star_forming

        wave = jnp.array([_C_AA / 1.4e9])
        l_lo = float(radio_star_forming(wave, L_ir=_L_IR_SFR10, q_ir=2.44)[0])
        l_hi = float(radio_star_forming(wave, L_ir=_L_IR_SFR10, q_ir=3.44)[0])

        # Δq = 1.0 → L_lo/L_hi = 10 exactly (Bell 2003 Eq. 5 denominator)
        np.testing.assert_allclose(
            l_lo / l_hi,
            10.0,
            rtol=1e-6,
            err_msg="Δq_IR = 1 should change L_1.4GHz by exactly 10× (Bell 2003 Eq. 5)",
        )

    def test_free_free_absolute_at_1p4ghz(self):
        """Murphy+2011 Eq. 11: free-free L_nu at 1.4 GHz for SFR=1.

        L_ff(1.4 GHz) ≈ 2.1e27 erg/s/Hz per Msun/yr (Murphy+2011 Table 1).
        With L_IR = 1e10 Lsun → SFR ≈ 0.58 Msun/yr (Kennicutt 1998).
        Expected: ~1.2e27 erg/s/Hz.
        """
        from tengri.components.radio import radio_freefree

        wave = jnp.array([_C_AA / 1.4e9])
        l_ff = float(radio_freefree(wave, L_ir=_L_IR_SFR1)[0])

        # Should be ~10% of synchrotron (~2e28), so ~1e27
        assert 1e26 < l_ff < 1e28, f"L_ff(1.4GHz) = {l_ff:.2e}, expected ~1e27 erg/s/Hz"

    def test_free_free_flatter_than_synchrotron(self):
        """Free-free (α_ff ~ -0.1) is flatter than synchrotron (α_sf ~ 0.8)."""
        from tengri.components.radio import radio_freefree, radio_star_forming

        wave_150mhz = jnp.array([_C_AA / 1.5e8])
        wave_10ghz = jnp.array([_C_AA / 1.0e10])

        sync_ratio = float(radio_star_forming(wave_10ghz, L_ir=_L_IR_SFR1)[0]) / float(
            radio_star_forming(wave_150mhz, L_ir=_L_IR_SFR1)[0]
        )
        ff_ratio = float(radio_freefree(wave_10ghz, L_ir=_L_IR_SFR1)[0]) / float(
            radio_freefree(wave_150mhz, L_ir=_L_IR_SFR1)[0]
        )

        assert ff_ratio > sync_ratio, (
            f"Free-free (10G/150M = {ff_ratio:.3f}) should be flatter than "
            f"synchrotron ({sync_ratio:.3f})"
        )

    def test_thermal_fraction_at_1p4ghz(self):
        """Free-free is ~5-15% of total radio at 1.4 GHz (Condon 1992)."""
        from tengri.components.radio import radio_freefree, radio_star_forming

        wave = jnp.array([_C_AA / 1.4e9])
        l_sync = float(radio_star_forming(wave, L_ir=_L_IR_SFR1)[0])
        l_ff = float(radio_freefree(wave, L_ir=_L_IR_SFR1)[0])

        thermal_frac = l_ff / (l_sync + l_ff)
        assert 0.01 < thermal_frac < 0.30, (
            f"Thermal fraction = {thermal_frac:.2%}, expected 5-15% (Condon 1992)"
        )

    def test_delvecchio_mass_scaling_absolute(self):
        """Delvecchio+2021: Δlog(L) = 0.468 per 2 dex in M* at fixed L_IR."""
        from tengri.components.radio import radio_sfr_delvecchio2021

        wave = jnp.array([_C_AA / 1.4e9])

        l_m9 = float(
            radio_sfr_delvecchio2021(
                wave, _L_IR_SFR10, log_mstar=9.0, redshift=0.0, apply_suppression=False
            )[0]
        )
        l_m11 = float(
            radio_sfr_delvecchio2021(
                wave, _L_IR_SFR10, log_mstar=11.0, redshift=0.0, apply_suppression=False
            )[0]
        )

        np.testing.assert_allclose(
            np.log10(l_m11 / l_m9),
            0.468,
            atol=0.05,
            err_msg="Delvecchio mass scaling mismatch",
        )

    def test_synchrotron_suppression_function(self):
        """Bell+2003 suppression: L_corr/L → 0 for L << L0 = 3e28 erg/s/Hz.

        Tests the _synchrotron_suppression helper directly with CGS inputs.
        """
        from tengri.components.radio.radio import _L0_SYNCH, _synchrotron_suppression

        # At L >= L*: n(L) = 0.9, NOT 1. The 0.9 is the non-thermal fraction —
        # the remaining ~10% of the radio continuum is thermal free-free — so
        # "unsuppressed" means 0.9, not unity. This test asserted 1.0 and so
        # asserted the absence of the very fraction the function returns
        # (#1728). Bell 2003 Eq. 3, as the implementation documents.
        L_bright = jnp.array(100 * _L0_SYNCH)
        ratio_bright = float(_synchrotron_suppression(L_bright) / L_bright)
        np.testing.assert_allclose(ratio_bright, 0.9, atol=1e-6)

        # The plateau is flat: no residual L-dependence above L*.
        for multiple in (1.0, 10.0, 1e4, 1e8):
            L = jnp.array(multiple * _L0_SYNCH)
            np.testing.assert_allclose(
                float(_synchrotron_suppression(L) / L),
                0.9,
                atol=1e-6,
                err_msg=f"n(L) must be flat at 0.9 above L*, differs at L = {multiple:g} L*",
            )

        # Below L*: n(L) = 0.9 (L/L*)**0.3, the gentle power law of Eq. 3.
        for multiple in (0.01, 0.1, 0.5):
            L = jnp.array(multiple * _L0_SYNCH)
            np.testing.assert_allclose(
                float(_synchrotron_suppression(L) / L),
                0.9 * multiple**0.3,
                rtol=1e-6,
                err_msg=f"n(L) must follow 0.9 (L/L*)^0.3 below L*, at L = {multiple:g} L*",
            )

        # Faint galaxies are suppressed relative to the 0.9 plateau, and
        # monotonically so. This block used to assert a *quadratic* form,
        # `1/(1 + (L0/L)^2) ~ 1e-6` at L = 1e-3 L*, which is a different model
        # from the one implemented and documented: Bell 2003 Eq. 3 is a gentle
        # power law, giving 0.9 * (1e-3)^0.3 = 0.113 there (#1728). The gentle
        # index is the point — cosmic-ray escape mildly reduces the radio-to-IR
        # ratio in low-mass systems, it does not switch synchrotron off.
        faint_ratios = [
            float(_synchrotron_suppression(jnp.array(m * _L0_SYNCH)) / jnp.array(m * _L0_SYNCH))
            for m in (1e-3, 1e-2, 1e-1, 1.0)
        ]
        assert faint_ratios == sorted(faint_ratios), (
            f"n(L) must increase monotonically with luminosity, got {faint_ratios}"
        )
        assert faint_ratios[0] < 0.9 / 5.0, (
            f"a galaxy at 1e-3 L* should be well below the 0.9 plateau, got {faint_ratios[0]:.3f}"
        )


# ── 2. X-ray: absolute values in erg/s ────────────────────────────


class TestXrayAbsoluteValues:
    """X-ray luminosities in erg/s against published scaling relations."""

    def test_lehmer2016_hmxb_2_10kev(self):
        r"""Lehmer+2016 Eq. 15: L_X(HMXB, 2-10 keV) per unit SFR at the default Z.

        .. math::

            \log(L_X/\mathrm{SFR}) = 40.28 - 62.12 Z + 569.44 Z^2
                                     - 1833.80 Z^3 + 1968.33 Z^4

        At the module's default — ``Z_SUN`` = 0.0142 — that is 10**39.508 =
        3.22e39, and the implementation reproduces it.

        This test used to assert **2.6e39**, taken from the docstring's
        parenthetical claim that the equation "yields ~2.6e39 at Z=0.02,
        consistent with Grimm et al. 2003". That claim is arithmetically false —
        the polynomial gives 1.78e39 there, a factor of 1.46 — and Lehmer+2016
        and Grimm+2003/Mineo+2012 genuinely differ by ~30-45% in this band
        rather than agreeing (#1755). Asserting the equation, not the prose.

        The default then moved from 0.02 to the project's canonical solar
        (Asplund 2009), which is the other half of #1755: one module carrying a
        second definition of "solar" put its HMXB baseline 1.8x off from what
        the rest of the code means by the same word. Taken from the constant,
        never restated as a literal.
        """
        from tengri.components.xray import xray_xrb
        from tengri.utils.physics_constants import Z_SUN

        wave = jnp.linspace(1.24, 6.2, 500)  # 2-10 keV
        l_nu = xray_xrb(wave, sfr=1.0, stellar_mass=0.0)

        nu = _C_AA / wave
        l_band = abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        z_default = Z_SUN
        log_l = (
            40.28
            - 62.12 * z_default
            + 569.44 * z_default**2
            - 1833.80 * z_default**3
            + 1968.33 * z_default**4
        )
        expected = 10.0**log_l  # 3.2177e39 at Z_SUN = 0.0142

        np.testing.assert_allclose(
            l_band,
            expected,
            rtol=0.10,
            err_msg=f"HMXB L_X = {l_band:.3e}, Lehmer+2016 at Z={z_default} gives {expected:.3e}",
        )

    def test_lmxb_matches_gilfanov_for_an_old_population(self):
        """LMXB traces stellar mass, and its normalization is age-dependent.

        Gilfanov 2004 quotes L_X(LMXB) = 9.2e28 x M*, i.e. 9.2e38 at 1e10 Msun,
        for an **old** population. Lehmer+2016 — what tengri implements — is a
        function of stellar age, and reproduces that at ~10 Gyr (8.15e38). At
        the module default of 1 Gyr it is 1.89e40, a factor of 23 higher.

        This test used to call ``xray_xrb`` at its default age and assert
        Gilfanov's old-population value, so it was comparing a 1 Gyr population
        against a ~10 Gyr normalization and failing by a factor of 27 (#1755).
        Pinning the age makes the comparison meaningful.
        """
        from tengri.components.xray import xray_xrb

        wave = jnp.linspace(1.55, 24.8, 500)  # 0.5-8 keV
        l_nu = xray_xrb(wave, sfr=0.0, stellar_mass=1e10, stellar_age_gyr=10.0)

        nu = _C_AA / wave
        l_band = abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        # Gilfanov's 9.2e38 is 2-10 keV; this band is 0.5-8 keV and so collects
        # more, hence the wide tolerance. The point is the order of magnitude
        # and that the age dependence is wired, not a precise band match.
        assert 5e38 < l_band < 5e39, (
            f"LMXB L_X(0.5-8 keV) at 10 Gyr = {l_band:.2e}, expected within a "
            "factor of a few of Gilfanov 2004's 9.2e38"
        )

    def test_lmxb_declines_with_stellar_age(self):
        """The age dependence itself — old populations host fainter LMXBs.

        Pins the mechanism the test above relies on, so a regression that
        flattened the age term would fail here with a message about age.
        """
        from tengri.components.xray import xray_xrb

        wave = jnp.linspace(1.55, 24.8, 500)
        nu = _C_AA / wave

        def band(age_gyr: float) -> float:
            l_nu = xray_xrb(wave, sfr=0.0, stellar_mass=1e10, stellar_age_gyr=age_gyr)
            return abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        young, middle, old = band(1.0), band(5.0), band(10.0)
        assert young > middle > old, (
            f"LMXB luminosity must fall with stellar age, got {young:.2e}, "
            f"{middle:.2e}, {old:.2e} at 1, 5, 10 Gyr"
        )

    def test_lehmer_combined(self):
        """Lehmer+2010 combined: L_X ≈ 2.7e40 for SFR=10, M*=1e10."""
        from tengri.components.xray import xray_xrb

        wave = jnp.linspace(1.24, 6.2, 500)
        l_nu = xray_xrb(wave, sfr=10.0, stellar_mass=1e10)
        nu = _C_AA / wave
        l_band = abs(float(jnp.trapezoid(l_nu[::-1], nu[::-1])))

        assert 5e39 < l_band < 5e41, f"Combined L_X = {l_band:.2e}, expected ~2.7e40"

    def test_xray_l_nu_at_2kev_absolute(self):
        """L_nu(2 keV) for SFR=1 should be ~1e21 erg/s/Hz.

        L_X = 2.6e39 spread over ~2e18 Hz (2-10 keV) → L_nu ~ 1e21 erg/s/Hz.
        """
        from tengri.components.xray import xray_xrb

        wave = jnp.array([6.2])  # 2 keV
        l_nu = float(xray_xrb(wave, sfr=1.0, stellar_mass=0.0)[0])

        assert 1e20 < l_nu < 1e23, f"L_nu(2keV) = {l_nu:.2e}, expected ~1e21 erg/s/Hz"

    def test_photon_index_hardness(self):
        """Higher Γ → softer spectrum → lower hardness ratio."""
        from tengri.components.xray import xray_xrb

        wave = jnp.linspace(0.5, 25.0, 1000)
        soft_mask = (wave > 6.2) & (wave < 24.8)
        hard_mask = (wave > 1.24) & (wave < 6.2)

        l_g15 = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, gamma_hmxb=1.5)
        l_g25 = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, gamma_hmxb=2.5)

        hr_hard = float(jnp.sum(l_g15[hard_mask])) / float(
            jnp.sum(l_g15[hard_mask]) + jnp.sum(l_g15[soft_mask])
        )
        hr_soft = float(jnp.sum(l_g25[hard_mask])) / float(
            jnp.sum(l_g25[hard_mask]) + jnp.sum(l_g25[soft_mask])
        )

        assert hr_hard > hr_soft, f"Γ=1.5 HR={hr_hard:.3f} should > Γ=2.5 HR={hr_soft:.3f}"

    def test_ecut_suppression(self):
        """E_cut cutoff: 100 keV more suppressed than 10 keV at E_cut=50."""
        from tengri.components.xray import xray_xrb

        wave = jnp.array([0.124, 1.24])  # 100 keV, 10 keV
        l_cut50 = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, E_cut=50.0)
        l_cut500 = xray_xrb(wave, sfr=1.0, stellar_mass=0.0, E_cut=500.0)

        if float(l_cut500[0]) > 0 and float(l_cut50[0]) > 0:
            r_100 = float(l_cut50[0] / l_cut500[0])
            r_10 = float(l_cut50[1] / l_cut500[1])
            assert r_100 < r_10, "Cutoff should suppress 100 keV more than 10 keV"

    def test_agn_corona_absolute_at_2kev(self):
        r"""AGN corona at L_bol=1e45 erg/s: check L_nu(2 keV) against the definition.

        :math:`\alpha_{ox} = 0.384 \log_{10}(L_\nu(2\,keV) / L_\nu(2500))`
        relates two **monochromatic** luminosities, so

        .. math::
            L_\nu(2\,keV) = L_\nu(2500) \times 10^{\alpha_{ox}/0.384}

        and no frequency conversion enters. This test used to divide that by
        ``KEV_TO_HZ * 2``, treating ``L_2keV`` as an integrated erg/s — a
        factor of 4.84e17 (#1728). It also passed ``alpha_ox=`` and
        ``L_agn_bol=``; the corona now takes ``l_2500_30deg_erg_hz`` and
        *derives* alpha_ox from it via Just+2007 (#980), with
        ``delta_alpha_ox`` as the offset knob.

        Tolerance covers the high-energy cutoff, ``exp(-2/300) = 0.9934`` at
        2 keV against a 300 keV E_cut.
        """
        from tengri.components.xray import alpha_ox_from_l2500, xray_agn_corona

        L_bol = 1e45  # erg/s (bright quasar)
        BC_2500 = 5.15
        nu_2500 = 1.199e15
        L_2500 = L_bol / (BC_2500 * nu_2500)

        wave = jnp.array([6.2])  # 2 keV
        l_2kev = float(xray_agn_corona(wave, L_2500)[0])

        alpha_ox = float(alpha_ox_from_l2500(L_2500))
        expected = L_2500 * 10.0 ** (alpha_ox / 0.384)

        np.testing.assert_allclose(
            l_2kev,
            expected,
            rtol=0.10,
            err_msg=f"L_nu(2keV) = {l_2kev:.2e}, expected {expected:.2e}",
        )

    def test_agn_corona_steeper_alpha_ox(self):
        """Steeper alpha_ox → weaker X-rays relative to UV.

        alpha_ox is derived from L_2500 rather than set, so the offset knob
        ``delta_alpha_ox`` is what steepens it. A more negative offset must
        reduce the 2 keV luminosity at fixed UV.
        """
        from tengri.components.xray import xray_agn_corona

        wave = jnp.array([6.2])
        l_2500 = 1e45 / (5.15 * 1.199e15)

        nominal = float(xray_agn_corona(wave, l_2500)[0])
        steeper = float(xray_agn_corona(wave, l_2500, delta_alpha_ox=-0.4)[0])

        assert steeper < nominal, "steeper alpha_ox must give less X-ray at fixed UV"
        # 0.4 dex of alpha_ox is 10**(-0.4/0.384) in the 2 keV luminosity.
        np.testing.assert_allclose(steeper / nominal, 10.0 ** (-0.4 / 0.384), rtol=0.02)


# ── 3. Cross-component self-consistency ───────────────────────────


class TestCrossComponentConsistency:
    """Self-consistency between radio, X-ray, and IR tracers."""

    def test_radio_linear_with_lir(self):
        """L_radio ∝ L_IR must hold exactly (built into the model)."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.geomspace(1e8, 1e10, 100)
        l_1x = float(jnp.sum(radio_star_forming(wave, L_ir=_L_IR_SFR1)))
        l_10x = float(jnp.sum(radio_star_forming(wave, L_ir=_L_IR_SFR10)))

        np.testing.assert_allclose(l_10x / l_1x, 10.0, rtol=1e-10)

    def test_radio_xray_sfr_scaling_parallel(self):
        """10× SFR → 10× radio AND 10× X-ray (both linear with SFR)."""
        from tengri.components.radio import radio_star_forming
        from tengri.components.xray import xray_xrb

        wave_r = jnp.geomspace(1e8, 1e10, 100)
        wave_x = jnp.linspace(1.24, 6.2, 100)

        radio_1 = float(jnp.sum(radio_star_forming(wave_r, L_ir=_L_IR_SFR1)))
        radio_10 = float(jnp.sum(radio_star_forming(wave_r, L_ir=_L_IR_SFR10)))
        xray_1 = float(jnp.sum(xray_xrb(wave_x, sfr=1.0, stellar_mass=0.0)))
        xray_10 = float(jnp.sum(xray_xrb(wave_x, sfr=10.0, stellar_mass=0.0)))

        np.testing.assert_allclose(radio_10 / radio_1, 10.0, rtol=0.01)
        np.testing.assert_allclose(xray_10 / xray_1, 10.0, rtol=0.01)

    def test_free_free_and_synchrotron_same_sfr(self):
        """Both free-free and synchrotron trace the same L_IR → same SFR.

        Doubling L_IR doubles BOTH components.
        """
        from tengri.components.radio import radio_freefree, radio_star_forming

        wave = jnp.array([_C_AA / 1.4e9])
        sync_1 = float(radio_star_forming(wave, L_ir=_L_IR_SFR1)[0])
        sync_2 = float(radio_star_forming(wave, L_ir=2 * _L_IR_SFR1)[0])
        ff_1 = float(radio_freefree(wave, L_ir=_L_IR_SFR1)[0])
        ff_2 = float(radio_freefree(wave, L_ir=2 * _L_IR_SFR1)[0])

        np.testing.assert_allclose(sync_2 / sync_1, 2.0, rtol=1e-10)
        np.testing.assert_allclose(ff_2 / ff_1, 2.0, rtol=1e-10)


# ── 4. IGM: numerical transmission values ─────────────────────────


class TestIGMNumericalValues:
    """IGM transmission at specific (z, λ) against Inoue+2014."""

    def test_lya_forest_opacity_z3(self):
        """At z=3, Lyα forest T at 1050Å rest ~ 0.5-0.85."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.array([1050.0 * (1 + 3.0)])
        t = float(igm_transmission(wave_obs, z_source=3.0)[0])
        assert 0.3 < t < 0.95, f"IGM T = {t:.2f}"

    def test_gunn_peterson_z6(self):
        """At z=6, below Lyα: T < 0.1 (Gunn-Peterson trough)."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.array([1100.0 * (1 + 6.0)])
        t = float(igm_transmission(wave_obs, z_source=6.0)[0])
        assert t < 0.1, f"T = {t:.2f}, expected < 0.1"

    @pytest.mark.parametrize(
        "z, wave_rest, t_min, t_max",
        [
            (0.5, 1100.0, 0.85, 1.0),
            (2.0, 1100.0, 0.5, 0.95),
            (4.0, 1100.0, 0.1, 0.6),
            (3.0, 5000.0, 0.99, 1.0),
            (5.0, 800.0, 0.0, 0.05),
        ],
    )
    def test_igm_range(self, z, wave_rest, t_min, t_max):
        """IGM transmission at (z, λ) in expected range."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.array([wave_rest * (1 + z)])
        t = float(igm_transmission(wave_obs, z_source=z)[0])
        assert t_min <= t <= t_max + 0.01, f"T = {t:.3f}, expected [{t_min}, {t_max}]"


# ── 5. Radio component decomposition (CGS throughout) ─────────────


class TestRadioComponentDecomposition:
    """SF + AGN + free-free decomposition with CGS inputs."""

    def test_components_sum_to_total(self):
        """Components must sum to total exactly."""
        from tengri.components.radio import compute_radio_components, radio_total

        wave = jnp.geomspace(1e8, 1e10, 200)
        kwargs = dict(
            L_ir=_L_IR_SFR1,
            L_agn_bol=1e44,
            q_ir=2.64,
            alpha_sf=0.8,
            radio_loudness=1.0,
            alpha_agn=0.7,
        )

        total = radio_total(wave, include_freefree=True, **kwargs)
        comps = compute_radio_components(wave, include_freefree=True, **kwargs)
        comp_sum = comps["synchrotron"] + comps["freefree"] + comps["agn"]

        np.testing.assert_allclose(np.asarray(comp_sum), np.asarray(total), rtol=1e-10)

    def test_agn_dominates_radio_loud(self):
        """Radio-loud AGN (loudness=3): AGN > 90% at 1.4 GHz."""
        from tengri.components.radio import compute_radio_components

        wave = jnp.array([_C_AA / 1.4e9])
        comps = compute_radio_components(
            wave,
            L_ir=_L_IR_SFR1,
            L_agn_bol=1e45,
            radio_loudness=3.0,
            include_freefree=True,
        )
        total = float(comps["synchrotron"][0] + comps["freefree"][0] + comps["agn"][0])
        agn_lum_ratio = float(comps["agn"][0]) / total
        assert agn_lum_ratio > 0.9, f"AGN fraction = {agn_lum_ratio:.2%}"

    def test_sf_dominates_radio_quiet(self):
        """Radio-quiet with strong SF: SF > 90% at 1.4 GHz."""
        from tengri.components.radio import compute_radio_components

        wave = jnp.array([_C_AA / 1.4e9])
        comps = compute_radio_components(
            wave,
            L_ir=_L_IR_SFR10,
            L_agn_bol=1e42,
            radio_loudness=-2.0,
            include_freefree=True,
        )
        total = float(comps["synchrotron"][0] + comps["freefree"][0] + comps["agn"][0])
        sf_frac = float(comps["synchrotron"][0] + comps["freefree"][0]) / total
        assert sf_frac > 0.9, f"SF fraction = {sf_frac:.2%}"

    def test_synchrotron_absolute_value(self):
        """Synchrotron at 1.4 GHz for SFR~1 should be ~2e28 erg/s/Hz."""
        from tengri.components.radio import compute_radio_components

        wave = jnp.array([_C_AA / 1.4e9])
        comps = compute_radio_components(
            wave,
            L_ir=_L_IR_SFR1,
            L_agn_bol=0.0,
            include_freefree=True,
        )
        l_sync = float(comps["synchrotron"][0])
        assert 1e27 < l_sync < 1e29, f"L_sync(1.4GHz) = {l_sync:.2e}, expected ~2e28"

    def test_freefree_absolute_value(self):
        """Free-free at 1.4 GHz for SFR~1 should be ~1e27 erg/s/Hz."""
        from tengri.components.radio import compute_radio_components

        wave = jnp.array([_C_AA / 1.4e9])
        comps = compute_radio_components(
            wave,
            L_ir=_L_IR_SFR1,
            L_agn_bol=0.0,
            include_freefree=True,
        )
        l_ff = float(comps["freefree"][0])
        assert 1e25 < l_ff < 1e28, f"L_ff(1.4GHz) = {l_ff:.2e}, expected ~1e27"
