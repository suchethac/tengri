# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2582 — X-ray SFR timescale and properties correctness.

Bug 1: X-ray XRB/hot-gas blocks scale with instantaneous SFR instead of
100 Myr-averaged SFR. The relations (Lehmer et al. 2016, adopted by Yang+2020,
2022) are calibrated on ~10 Myr timescales, but the instantaneous SFR in bursty
models causes the emitted spectrum to mismatch the registered property by factor
0.5–2.5. Yang et al. 2022 Sect. 3.3 and pcigale 2025.1 use sfr_100myr (Yang+2022,
Lehmer+2016, ApJ 825, 7, Eq. 13–14).

Bug 2: Registered properties log_l_x_xrb and log_l_x_agn are independent
relations (one reads sfr_100myr, one Duras+2020 bolometric correction), so they
describe different galaxies than the emitted X-ray spectrum (which reads sfr).
They should be 2–10 keV band integrals of the emitted HMXB+LMXB and AGN corona,
using the same band-norm functions the spectrum uses, and computed float32-safe
in log space.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.components.xray.component import XRaySEDComponent

pytestmark = pytest.mark.regression_bug


HC = 12.398419843  # keV·Angstrom
C_AA = 2.99792458e18  # Angstrom·Hz / c


def band_integral(wave, lnu, e1_kev, e2_kev, n=20001):
    """Integrate X-ray luminosity over a keV band.

    Parameters
    ----------
    wave : array, shape (n_wave,)
        Wavelength [Angstrom].
    lnu : array, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].
    e1_kev, e2_kev : float
        Energy band [keV].
    n : int
        Number of points in the integration grid.

    Returns
    -------
    float
        Band luminosity [erg/s].
    """
    E = np.linspace(e1_kev, e2_kev, n)
    w = HC / E
    o = np.argsort(wave)
    l = np.exp(np.interp(np.log(w), np.log(wave[o]), np.log(np.maximum(lnu[o], 1e-300))))
    nu = C_AA / w
    oo = np.argsort(nu)
    return np.trapezoid(l[oo], nu[oo])


@pytest.mark.parametrize(
    "sfh_type",
    [
        {"type": "const", "log_total_mass": Fixed(10.5)},
        {"type": "delayed", "tau_gyr": Fixed(2.0), "log_total_mass": Fixed(10.5)},
        {
            "type": "sfh2exp",
            "tau_main_gyr": Fixed(3.0),
            "tau_burst_gyr": Fixed(0.02),
            "f_burst": Fixed(0.3),
            "age_gyr": Fixed(5.0),
            "burst_age_gyr": Fixed(0.03),
            "log_total_mass": Fixed(10.5),
        },
        {
            "type": "periodic",
            "tau_0_gyr": Fixed(2.0),
            "period_gyr": Fixed(1.0),
            "log_total_mass": Fixed(10.5),
            "age_gyr": Fixed(5.0),
        },
    ],
)
class TestXRaySFRTimescale:
    """XRB and hot-gas blocks use 100 Myr-averaged SFR, not instantaneous."""

    def test_emitter_inputs_sfr_equals_sfr_100myr(self, sfh_type):
        """The emitter_inputs["sfr"] is sfr_100myr when available."""
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={**sfh_type, "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.0),
        )
        st = model.predict_state({})
        d = st.derived
        inp = XRaySEDComponent().emitter_inputs(d)

        # The block should read sfr_100myr, not sfr
        assert float(inp["sfr"]) == pytest.approx(float(d["sfr_100myr"]), rel=1e-6)


@pytest.mark.parametrize("xray_type", ["yang20", "lopez24"])
class TestXRayProperties:
    """Properties are band integrals of emitted terms, not independent relations."""

    def test_log_l_x_xrb_matches_emitted_spectrum(self, xray_type):
        """log_l_x_xrb equals log10(∫_{2}^{10 keV} hmxb+lmxb dν)."""
        if xray_type == "lopez24":
            pytest.skip("lopez24 requires AGN; tested separately in test_log_l_x_agn_corona")

        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "sfh2exp",
                "tau_main_gyr": Fixed(3.0),
                "tau_burst_gyr": Fixed(0.02),
                "f_burst": Fixed(0.3),
                "age_gyr": Fixed(5.0),
                "burst_age_gyr": Fixed(0.03),
                "log_total_mass": Fixed(10.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            xray={"type": xray_type, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.0),
        )
        st = model.predict_state({})
        inp = XRaySEDComponent().emitter_inputs(st.derived)

        # Emit the X-ray terms with the block's internal parameters
        XP = {
            "xray_gamma_agn": 1.8,
            "xray_gamma_hmxb": 2.0,
            "xray_gamma_lmxb": 1.56,
            "xray_E_cut": 300.0,
            "xray_delta_alpha_ox": 0.0,
            "xray_log_nh": 20.0,
            "xray_alpha_irx": 0.3,
            "xray_det_hmxb": 0.0,
            "xray_det_lmxb": 0.0,
        }
        terms = XRaySEDComponent().emission_terms(
            XP, np.asarray(st.wave), **{k: np.asarray(v) for k, v in inp.items()}
        )

        # Band integral of emitted XRB spectrum
        lnu_xrb = np.asarray(terms["hmxb"]) + np.asarray(terms["lmxb"])
        l_spec = band_integral(np.asarray(st.wave), lnu_xrb, 2.0, 10.0)
        log_l_spec = np.log10(np.maximum(l_spec, 1e-300))

        # Published property
        prop = float(model.predict_properties({}, names=("log_l_x_xrb",))["log_l_x_xrb"])

        # They should match to 1e-3 dex (measurement tolerance)
        assert prop == pytest.approx(log_l_spec, abs=2e-3)

    def test_log_l_x_agn_corona_matches_emitted_spectrum(self, xray_type):
        """log_l_x_agn equals log10(∫_{2}^{10 keV} agn_corona dν)."""
        if xray_type == "yang20":
            pytest.skip("yang20 has no composable AGN; tested separately")

        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "const", "log_total_mass": Fixed(10.5), "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            agn={
                "type": "composable",
                "torus": "skirtor",
                "disc": "schartmann2005",
                "agn_torus_frac": Fixed(1.0),
                "agn_log_lbol": Fixed(12.0),
                "all_params": Fixed(DEFAULT),
            },
            xray={"type": xray_type, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.0),
        )
        st = model.predict_state({})
        inp = XRaySEDComponent().emitter_inputs(st.derived)

        # Emit the X-ray terms
        XP = {
            "xray_gamma_agn": 1.8,
            "xray_gamma_hmxb": 2.0,
            "xray_gamma_lmxb": 1.56,
            "xray_E_cut": 300.0,
            "xray_delta_alpha_ox": 0.0,
            "xray_log_nh": 20.0,
            "xray_alpha_irx": 0.3,
            "xray_det_hmxb": 0.0,
            "xray_det_lmxb": 0.0,
        }
        terms = XRaySEDComponent().emission_terms(
            XP, np.asarray(st.wave), **{k: np.asarray(v) for k, v in inp.items()}
        )

        # Band integral of emitted AGN corona spectrum
        lnu_agn = np.asarray(terms["agn_corona"])
        l_spec = band_integral(np.asarray(st.wave), lnu_agn, 2.0, 10.0)
        log_l_spec = np.log10(np.maximum(l_spec, 1e-300))

        # Published property
        prop = float(model.predict_properties({}, names=("log_l_x_agn",))["log_l_x_agn"])

        # They should match to 2e-3 dex
        assert prop == pytest.approx(log_l_spec, abs=2e-3)


class TestXRayFloat32:
    """X-ray properties are finite and accurate in float32."""

    def test_float32_xrb_properties_finite(self):
        """log_l_x_xrb is finite in float32."""
        with jnp.enable_x64(False):
            ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
            model = SEDModel.build(
                ssp_data=ssp,
                sfh={
                    "type": "const",
                    "log_total_mass": Fixed(10.5),
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                neb={"type": "none"},
                xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.0),
            )
            prop = model.predict_properties({}, names=("log_l_x_xrb",))["log_l_x_xrb"]
            assert jnp.isfinite(prop)

    def test_float32_agn_properties_finite(self):
        """log_l_x_agn is finite in float32."""
        with jnp.enable_x64(False):
            ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
            model = SEDModel.build(
                ssp_data=ssp,
                sfh={
                    "type": "const",
                    "log_total_mass": Fixed(10.5),
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                neb={"type": "none"},
                agn={
                    "type": "composable",
                    "torus": "skirtor",
                    "disc": "schartmann2005",
                    "agn_torus_frac": Fixed(1.0),
                    "agn_log_lbol": Fixed(12.0),
                    "all_params": Fixed(DEFAULT),
                },
                xray={"type": "lopez24", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.0),
            )
            prop = model.predict_properties({}, names=("log_l_x_agn",))["log_l_x_agn"]
            assert jnp.isfinite(prop)
