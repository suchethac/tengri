# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate radio, X-ray, and AGN models against CIGALE formulas.

Radio: tengri and CIGALE both use the FIR-radio correlation
  q_IR = log10(L_IR / (3.75e12 * L_1.4GHz))
with a synchrotron power-law L_nu ~ nu^{-alpha}. The formula is
analytical, so agreement should be exact for matching parameters.

X-ray: tengri uses Grimm+2003 (HMXB-SFR) and Gilfanov+2004
(LMXB-M*) scaling relations, same as CIGALE's Yang+2020 module.
We verify the scaling and spectral shapes.

AGN: tengri implements power-law disc + torus (simple, standard,
kubota_done). CIGALE uses Fritz+2006 radiative transfer templates.
We compare torus peak wavelength and UV/IR balance qualitatively.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ── 1. Radio: FIR-radio correlation ───────────────────────────────


class TestRadioCrossval:
    """Compare tengri radio against CIGALE formula (analytical)."""

    def test_fir_radio_formula_matches_cigale(self):
        """L_1.4GHz from FIR-radio correlation should match CIGALE exactly.

        Both codes use: L_1.4GHz = L_IR / (3.75e12 * 10^q_IR)
        """
        from tengri.components.radio import radio_star_forming

        wave = jnp.logspace(7, 10, 500)  # 1mm to 100m in Angstrom
        L_ir = 1e10  # Lsun

        for q_ir, alpha in [(2.58, 0.8), (2.64, 0.8), (2.34, 0.7)]:
            l_nu = np.asarray(radio_star_forming(wave, L_ir, q_ir=q_ir, alpha_sf=alpha))

            # CIGALE formula: L_1.4GHz = L_IR / (3.75e12 * 10^q_IR)
            L_14_expected = L_ir / (3.75e12 * 10.0**q_ir)

            # Find L at 1.4 GHz (21 cm = 2.1e9 A)
            idx_14 = np.argmin(np.abs(np.asarray(wave) - 2.1e9))
            L_14_tengri = l_nu[idx_14]

            np.testing.assert_allclose(
                L_14_tengri,
                L_14_expected,
                rtol=0.03,  # 3% tolerance for wavelength grid discretization
                err_msg=f"L_1.4GHz mismatch for q={q_ir}, alpha={alpha}",
            )

    def test_spectral_index(self):
        """Power-law slope should match spectral index alpha."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.logspace(7, 10, 500)
        alpha = 0.8

        l_nu = np.asarray(radio_star_forming(wave, 1e10, q_ir=2.64, alpha_sf=alpha))
        wave_np = np.asarray(wave)

        # Measure slope between two radio wavelengths
        w1, w2 = 5e8, 5e9  # 5cm and 50cm
        i1 = np.argmin(np.abs(wave_np - w1))
        i2 = np.argmin(np.abs(wave_np - w2))

        nu1 = 2.998e18 / wave_np[i1]
        nu2 = 2.998e18 / wave_np[i2]

        measured_alpha = -np.log(l_nu[i1] / l_nu[i2]) / np.log(nu1 / nu2)
        np.testing.assert_allclose(
            measured_alpha,
            alpha,
            rtol=0.01,
            err_msg=f"Spectral index: measured={measured_alpha:.3f}, expected={alpha}",
        )

    def test_radio_zero_below_1mm(self):
        """Radio emission should be zero at wavelengths < 1mm (optical/IR)."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.array([5000.0, 10000.0, 1e6, 5e6])  # optical to mid-IR
        l_nu = np.asarray(radio_star_forming(wave, 1e10))
        assert np.all(l_nu == 0), "Radio should be zero at optical/IR wavelengths"

    def test_radio_scales_with_lir(self):
        """Doubling L_IR should double radio luminosity."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.logspace(8, 10, 100)
        l1 = np.asarray(radio_star_forming(wave, 1e10))
        l2 = np.asarray(radio_star_forming(wave, 2e10))

        mask = l1 > 0
        np.testing.assert_allclose(l2[mask] / l1[mask], 2.0, rtol=1e-10)

    @pytest.mark.parametrize("q_ir", [2.34, 2.58, 2.80])
    def test_higher_qir_means_less_radio(self, q_ir):
        """Higher q_IR means more IR per unit radio — less radio."""
        from tengri.components.radio import radio_star_forming

        wave = jnp.array([2.1e9])  # 21 cm
        l_low = float(radio_star_forming(wave, 1e10, q_ir=2.34)[0])
        l_high = float(radio_star_forming(wave, 1e10, q_ir=q_ir)[0])
        if q_ir > 2.34:
            assert l_high < l_low, f"q_IR={q_ir} should give less radio than q_IR=2.34"


# ── 2. X-ray: scaling relations ───────────────────────────────────


class TestXrayCrossval:
    """Validate X-ray binary scaling relations against literature."""

    def test_hmxb_scales_with_sfr(self):
        """HMXB luminosity should scale linearly with SFR."""
        from tengri.components.xray import xray_xrb

        wave = jnp.array([1.0])  # 1 Angstrom ~ 12.4 keV
        l1 = float(xray_xrb(wave, sfr=1.0, stellar_mass=1e10)[0])
        l2 = float(xray_xrb(wave, sfr=2.0, stellar_mass=1e10)[0])

        # HMXB component doubles, LMXB stays same
        # Total should increase but not exactly double (LMXB baseline)
        assert l2 > l1, "Higher SFR should give more X-ray"

    def test_lmxb_scales_with_mass(self):
        """LMXB luminosity should scale linearly with stellar mass."""
        from tengri.components.xray import xray_xrb

        wave = jnp.array([1.0])
        l1 = float(xray_xrb(wave, sfr=0.0, stellar_mass=1e10)[0])
        l2 = float(xray_xrb(wave, sfr=0.0, stellar_mass=2e10)[0])

        np.testing.assert_allclose(l2 / l1, 2.0, rtol=0.01)

    def test_grimm_relation_integrated(self):
        r"""HMXB integrated 2-10 keV L_X matches the implemented Lehmer+2016 relation.

        tengri implements Lehmer+2016 Eq. 15, which is metallicity-dependent:

        .. math::

            \log(L_X/\mathrm{SFR}) = 40.28 - 62.12 Z + 569.44 Z^2
                                     - 1833.80 Z^3 + 1968.33 Z^4

        At the module default ``metallicity_z=0.02`` that is 1.78e39, not the
        2.6e39 this test used to assert. The 2.6e39 comes from
        Grimm+2003/Mineo+2012, which genuinely differ from Lehmer+2016 by
        ~30-45% in this band — the `xray.py` docstring claims the two agree,
        which is arithmetically false (#1755).
        """
        from tengri.components.xray import xray_xrb

        # 2-10 keV in Angstrom: E=hc/λ → λ = 12398.4/E(eV)
        # 2 keV → 6.199 A, 10 keV → 1.240 A
        wave = jnp.linspace(1.24, 6.20, 500)  # 2-10 keV band
        l_nu = np.asarray(xray_xrb(wave, sfr=1.0, stellar_mass=0.0))

        # Integrate L_nu over frequency to get L_bol in 2-10 keV
        c_aa = 2.99792458e18
        nu = c_aa / np.asarray(wave)
        l_band = abs(np.trapezoid(l_nu[::-1], nu[::-1]))

        z_default = 0.02
        log_l = (
            40.28
            - 62.12 * z_default
            + 569.44 * z_default**2
            - 1833.80 * z_default**3
            + 1968.33 * z_default**4
        )
        expected = 10.0**log_l  # 1.7825e39

        np.testing.assert_allclose(
            l_band,
            expected,
            rtol=0.10,
            err_msg=(
                f"HMXB 2-10 keV L_X = {l_band:.3e}; Lehmer+2016 at Z=0.02 gives "
                f"{expected:.3e} erg/s per Msun/yr"
            ),
        )

    def test_xray_exponential_cutoff(self):
        """X-ray spectrum should have exponential cutoff at high energies."""
        from tengri.components.xray import xray_xrb

        wave_soft = jnp.array([2.0])  # ~6 keV
        wave_hard = jnp.array([0.1])  # ~124 keV

        l_soft = float(xray_xrb(wave_soft, sfr=1.0, stellar_mass=1e10)[0])
        l_hard = float(xray_xrb(wave_hard, sfr=1.0, stellar_mass=1e10, E_cut=100.0)[0])

        assert l_soft > l_hard, "Soft X-ray should be brighter than hard (cutoff)"


# ── 3. AGN models ─────────────────────────────────────────────────


class TestAGNCrossval:
    """Validate AGN disc + torus models."""

    def test_disc_conserves_luminosity(self):
        """Disc L_bol integral should match input L_bol."""
        from tengri.components.agn.disc import powerlaw_disc

        wave = jnp.linspace(100, 100000, 5000)
        l_nu = np.asarray(powerlaw_disc(wave, agn_log_lbol=11.0))

        c_cgs = 2.998e10
        nu = c_cgs / (np.asarray(wave) * 1e-8)
        l_bol = abs(np.trapezoid(l_nu[::-1], nu[::-1]))

        lsun = 3.828e33
        expected = 10**11.0 * lsun  # erg/s (L_nu is in erg/s/Hz after CGS standardization)
        # Should be within factor 2 (numerical integration over finite grid)
        ratio = l_bol / expected
        assert 0.5 < ratio < 2.0, f"Disc L_bol ratio = {ratio:.2f}"

    def test_torus_peaks_in_mir(self):
        """Silva+04 torus should peak in the mid-IR (1-100 um)."""
        from tengri.components.agn.silva04 import silva04_analytic

        wave = jnp.linspace(5000, 200000, 5000)
        l_nu = np.asarray(silva04_analytic(wave, agn_log_lbol=11.0))
        peak = float(wave[np.argmax(l_nu)]) / 1e4  # um

        # Silva+04 smooth torus peaks in the mid-IR for typical column densities
        assert 1.0 < peak < 100.0, f"Torus peak at {peak:.1f} um, expected 1-100 um"

    def test_unified_total_luminosity(self):
        """Unified AGN (disc+torus) should conserve total luminosity."""
        from tengri.components.agn.unified import multicolor_agn

        wave = jnp.linspace(100, 500000, 10000)
        l_nu = np.asarray(
            multicolor_agn(
                wave,
                agn_log_lbol=11.0,
                agn_lum_ratio=1.0,
                agn_torus_frac=0.5,
            )
        )

        c_cgs = 2.998e10
        nu = c_cgs / (np.asarray(wave) * 1e-8)
        l_bol = abs(np.trapezoid(l_nu[::-1], nu[::-1]))

        lsun = 3.828e33
        expected = 10**11.0 * lsun  # erg/s
        ratio = l_bol / expected
        assert 0.3 < ratio < 3.0, f"AGN L_bol ratio = {ratio:.2f}"

    def test_type1_vs_type2_covering(self):
        """Higher torus covering should shift UV→IR balance."""
        from tengri.components.agn.unified import multicolor_agn

        wave = jnp.linspace(500, 200000, 5000)

        l_type1 = np.asarray(
            multicolor_agn(
                wave,
                agn_log_lbol=11.0,
                agn_lum_ratio=1.0,
                agn_torus_frac=0.2,
            )
        )
        l_type2 = np.asarray(
            multicolor_agn(
                wave,
                agn_log_lbol=11.0,
                agn_lum_ratio=1.0,
                agn_torus_frac=0.8,
            )
        )

        uv = (np.asarray(wave) > 1000) & (np.asarray(wave) < 3000)
        ir = (np.asarray(wave) > 30000) & (np.asarray(wave) < 100000)

        uv_ir_type1 = np.mean(l_type1[uv]) / np.mean(l_type1[ir])
        uv_ir_type2 = np.mean(l_type2[uv]) / np.mean(l_type2[ir])

        assert uv_ir_type2 < uv_ir_type1, "Type 2 should have more IR, less UV"

    def test_agn_radio_component(self):
        """Radio-loud AGN should produce radio emission."""
        from tengri.components.radio import radio_agn

        wave = jnp.array([2.1e9])  # 21 cm
        l_quiet = float(radio_agn(wave, L_agn_bol=1e11, radio_loudness=0.0)[0])
        l_loud = float(radio_agn(wave, L_agn_bol=1e11, radio_loudness=3.0)[0])

        assert l_loud > l_quiet * 100, "Radio-loud should be >> radio-quiet"
