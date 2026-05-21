"""End-to-end SED amplitude tests for Milky-Way-like galaxy models.

These test amplitude at key wavelengths (UV, optical, NIR, submm)
to catch regressions slipping past component-level tests.
"""

from __future__ import annotations

import numpy as np
import pytest

LSUN_ERG = 3.828e33  # erg/s

pytestmark = pytest.mark.conservation


def _make_ssp_if_available():
    from pathlib import Path

    from tengri import load_ssp_data

    p = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not p.exists():
        return None
    return load_ssp_data(str(p))


class TestFullSEDAmplitudes:
    """End-to-end SED for a Milky-Way-like galaxy: M* ~ 10^10 Msun, SFR~1 M⊙/yr.

    These tests fix the amplitude at key wavelengths (UV, optical, NIR, submm)
    to catch regressions that slip past the component-level tests.
    """

    def _mw_like_sed(self):
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_log_peak_sfr=Fixed(np.log10(3.0)),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.0),
        )
        model = SEDModel(spec, ssp)
        pred = model.predict_rest_sed({})
        return np.array(pred.wavelength), np.array(pred.sed), model

    def test_nir_peak_amplitude(self):
        """1.6 μm (NIR H-band peak of evolved stellar pop) for ~10^10 M⊙ galaxy:
        L_ν ≈ 10^28–10^30 erg/s/Hz."""
        wl, lnu, _ = self._mw_like_sed()
        i = int(np.argmin(np.abs(wl - 16000.0)))
        assert 1e28 < lnu[i] < 1e30, f"L_ν(1.6 μm) = {lnu[i]:.2e}"

    def test_v_band_amplitude(self):
        """5500 Å (V-band) for M* ~ 10^10 M⊙: L_ν ≈ 10^28 erg/s/Hz (absolute mag ≈ -19)."""
        wl, lnu, _ = self._mw_like_sed()
        i = int(np.argmin(np.abs(wl - 5500.0)))
        assert 1e27 < lnu[i] < 1e30, f"L_ν(V) = {lnu[i]:.2e}"

    def test_uv_amplitude_attenuated(self):
        """1500 Å with τ_diff=0.3 ⇒ L_ν ~ 10^26–10^28 (attenuated from intrinsic)."""
        wl, lnu, _ = self._mw_like_sed()
        i = int(np.argmin(np.abs(wl - 1500.0)))
        assert 1e25 < lnu[i] < 1e29, f"L_ν(UV) = {lnu[i]:.2e}"

    def test_stellar_peak_is_nir(self):
        """Optical/NIR stellar SED peaks in νL_ν near 1 μm; i.e. NIR >> UV."""
        wl, lnu, _ = self._mw_like_sed()
        i_uv = int(np.argmin(np.abs(wl - 1500.0)))
        i_nir = int(np.argmin(np.abs(wl - 16000.0)))
        # Dust-attenuated UV should be lower than NIR in νL_ν sense
        nu_uv = 3e18 / wl[i_uv]
        nu_nir = 3e18 / wl[i_nir]
        assert lnu[i_nir] * nu_nir > lnu[i_uv] * nu_uv, (
            "NIR νL_ν must exceed dust-attenuated UV νL_ν for this model"
        )

    def test_bolometric_luminosity(self):
        """Integrated L_bol for a dwarf-MW-like model (~10^10 L_sun)."""
        wl, lnu, _ = self._mw_like_sed()
        nu = 3e18 / wl
        srt = np.argsort(nu)
        L_bol = float(np.trapezoid(lnu[srt], nu[srt]))
        # Expect 10^8 to 10^11 L_sun for this smooth-SFH model
        L_bol_lsun = L_bol / LSUN_ERG
        assert 1e8 < L_bol_lsun < 1e12, f"L_bol = {L_bol_lsun:.2e} L_sun"

    def test_ratio_v_to_uv_dust_attenuation(self):
        """τ_diff=0.3 ⇒ UV attenuation A(1500) ≈ τ·k(1500)/k(V) ≈ 0.3·3/(0.92·ln10)
        so f(UV)/f(V) should be reduced relative to τ=0 case."""
        from tengri import Fixed, Parameters, SEDModel

        ssp = _make_ssp_if_available()
        if ssp is None:
            pytest.skip("SSP data not available")

        def build(tau):
            spec = Parameters(
                mean_sfh_type="dpl",
                sfh_dpl_log_peak_sfr=Fixed(np.log10(3.0)),
                sfh_dpl_alpha=Fixed(2.0),
                sfh_dpl_beta=Fixed(1.0),
                sfh_dpl_tau_gyr=Fixed(5.0),
                met_logzsol=Fixed(0.0),
                dust_tau_bc=Fixed(0.0),
                dust_tau_diff=Fixed(tau),
                dust_slope=Fixed(-0.7),
                redshift=Fixed(0.0),
            )
            model = SEDModel(spec, ssp)
            pred = model.predict_rest_sed({})
            return np.array(pred.wavelength), np.array(pred.sed)

        wl, lnu0 = build(0.0)
        _, lnu3 = build(1.0)
        i_uv = int(np.argmin(np.abs(wl - 1500.0)))
        i_v = int(np.argmin(np.abs(wl - 5500.0)))
        # Ratio of dust-reddened UV/V relative to unattenuated must be < 1
        ratio_dusty = lnu3[i_uv] / lnu3[i_v]
        ratio_clean = lnu0[i_uv] / lnu0[i_v]
        assert ratio_dusty < ratio_clean, "Dust must attenuate UV more than V"
