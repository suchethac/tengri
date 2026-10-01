# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2582: X-ray SFR timescale and registered properties.

The HMXB and hot-gas relations (Lehmer et al. 2016, ApJ 825, 7, Eqs. 13-14, as
adopted by Yang et al. 2022, ApJ 927, 192, Sect. 3.3) are calibrated on the SFR
averaged over the last 100 Myr, not on the instantaneous SFR. The emitted
spectrum must therefore read ``sfr_100myr``.

The registered properties ``log_l_x_xrb`` and ``log_l_x_agn`` must be the
2-10 keV band luminosities of the terms ``emission_terms`` emits (one
definition), evaluated here by an independent log-log quadrature of the emitted
spectrum.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.components.xray.component import XRaySEDComponent, XRaySEDComponentConfig

pytestmark = pytest.mark.regression_bug

_SSP_PATH = Path(__file__).resolve().parents[3] / "data" / "fsps_prsc_miles_chabrier.h5"
HC = 12.398419843  # keV * Angstrom
C_AA = 2.99792458e18  # Angstrom * Hz

#: Declared defaults of the ``xray_*`` parameters the builds below leave Fixed.
_XP = {
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

_BURST = {
    "type": "sfh2exp",
    "tau_main_gyr": Fixed(3.0),
    "tau_burst_gyr": Fixed(0.02),
    "f_burst": Fixed(0.3),
    "age_gyr": Fixed(5.0),
    "burst_age_gyr": Fixed(0.03),
    "log_total_mass": Fixed(10.5),
}
_SFHS = {
    "const": {"type": "const", "log_total_mass": Fixed(10.5)},
    "delayed": {
        "type": "delayed",
        "tau_gyr": Fixed(2.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(10.5),
    },
    "sfh2exp_burst": _BURST,
    "periodic": {
        "type": "periodic",
        "log_total_mass": Fixed(10.5),
        "delta_bursts_gyr": Fixed(0.3),
        "tau_bursts_gyr": Fixed(0.05),
        "age_gyr": Fixed(5.0),
    },
}
_AGN = {
    "type": "composable",
    "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
    "torus": {"type": "skirtor", "agn_torus_frac": Fixed(1.0), "all_params": Fixed(DEFAULT)},
    "atten": {"type": "none"},
    "agn_log_lbol": Fixed(12.0),
    "agn_cos_inc": Fixed(float(np.cos(np.radians(30.0)))),
    "all_params": Fixed(DEFAULT),
}


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_PATH))


def _build(ssp, sfh, xray_type, agn=None):
    kwargs = {"agn": agn} if agn is not None else {}
    return SEDModel.build(
        ssp_data=ssp,
        sfh={**sfh, "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        xray={"type": xray_type, "log_nh": Fixed(20.0), "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
        **kwargs,
    )


def _band(wave, lnu, e1, e2, n=20001):
    """log-log quadrature of an emitted spectrum over [e1, e2] keV, in erg/s."""
    energy = np.linspace(e1, e2, n)
    lam = HC / energy
    o = np.argsort(wave)
    interp = np.exp(
        np.interp(np.log(lam), np.log(wave[o]), np.log(np.maximum(lnu[o], 1e-300)))
    )
    nu = C_AA / lam
    oo = np.argsort(nu)
    return np.trapezoid(interp[oo], nu[oo])


def _emitted(model, xray_type):
    state = model.predict_state({})
    comp = XRaySEDComponent(config=XRaySEDComponentConfig(model=xray_type))
    inputs = comp.emitter_inputs(state.derived)
    terms = comp.emission_terms(
        _XP, np.asarray(state.wave), **{k: np.asarray(v) for k, v in inputs.items()}
    )
    return comp, inputs, np.asarray(state.wave), {k: np.asarray(v) for k, v in terms.items()}


@pytest.mark.parametrize("xray_type", ["yang20", "lopez24"])
@pytest.mark.parametrize("sfh_name", list(_SFHS))
def test_block_reads_the_100myr_sfr(ssp, sfh_name, xray_type):
    """The HMXB / hot-gas amplitude SFR is the 100 Myr average, not the instantaneous one."""
    model = _build(ssp, _SFHS[sfh_name], xray_type)
    derived = model.predict_state({}).derived
    block = XRaySEDComponent().emitter_inputs(derived)
    np.testing.assert_allclose(float(block["sfr"]), float(derived["sfr_100myr"]), rtol=1e-6)


def test_burst_hmxb_luminosity_is_lehmer_per_sfr_times_the_100myr_sfr(ssp):
    """Emitted HMXB L(2-10 keV) = 10**(Lehmer+2016 Eq. 13 quartic in Z) * sfr_100myr.

    For the burst SFH of the issue (instantaneous 147.07, 100 Myr 96.45 Msun/yr).
    """
    model = _build(ssp, _BURST, "yang20")
    derived = model.predict_state({}).derived
    assert float(derived["sfr"]) != pytest.approx(float(derived["sfr_100myr"]), rel=1e-2)
    _, inputs, wave, terms = _emitted(model, "yang20")
    z = float(inputs["metallicity_z"])
    log_per_sfr = 40.28 - 62.12 * z + 569.44 * z**2 - 1833.80 * z**3 + 1968.33 * z**4
    expected = 10.0**log_per_sfr * float(derived["sfr_100myr"])
    assert _band(wave, terms["hmxb"], 2.0, 10.0) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("xray_type", ["yang20", "lopez24"])
@pytest.mark.parametrize("sfh_name", ["const", "sfh2exp_burst"])
def test_log_l_x_xrb_is_the_emitted_band_luminosity(ssp, sfh_name, xray_type):
    """log_l_x_xrb == log10 of the 2-10 keV integral of the emitted HMXB + LMXB."""
    model = _build(ssp, _SFHS[sfh_name], xray_type)
    _, _, wave, terms = _emitted(model, xray_type)
    expected = np.log10(_band(wave, terms["hmxb"] + terms["lmxb"], 2.0, 10.0))
    prop = float(model.predict_properties({}, names=("log_l_x_xrb",))["log_l_x_xrb"])
    assert prop == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("xray_type", ["yang20", "lopez24"])
def test_log_l_x_agn_is_the_emitted_corona_band_luminosity(ssp, xray_type):
    """log_l_x_agn == log10 of the 2-10 keV integral of the emitted corona."""
    model = _build(ssp, _SFHS["delayed"], xray_type, agn=_AGN)
    _, _, wave, terms = _emitted(model, xray_type)
    expected = np.log10(_band(wave, terms["agn"], 2.0, 10.0))
    prop = float(model.predict_properties({}, names=("log_l_x_agn",))["log_l_x_agn"])
    assert prop == pytest.approx(expected, abs=2e-3)


def test_hot_gas_band_luminosity_is_the_emitted_one(ssp):
    """The shared helper's hot-gas 0.5-2 keV luminosity equals the emitted term's."""
    model = _build(ssp, _BURST, "yang20")
    comp, inputs, wave, terms = _emitted(model, "yang20")
    logs = comp.log_band_luminosities(_XP, **inputs)
    expected = np.log10(_band(wave, terms["hotgas"], 0.5, 2.0))
    assert float(logs["hotgas"]) == pytest.approx(expected, abs=2e-3)


def test_no_agn_property_is_minus_infinity(ssp):
    """Without an AGN the corona luminosity is exactly zero, i.e. -inf in dex."""
    model = _build(ssp, _SFHS["const"], "yang20")
    prop = float(model.predict_properties({}, names=("log_l_x_agn",))["log_l_x_agn"])
    assert prop == -np.inf


def _properties(ssp, dtype64, xray_type, sfh, agn):
    with jax.enable_x64(dtype64):
        model = _build(ssp, sfh, xray_type, agn=agn)
        props = model.predict_properties({}, names=("log_l_x_xrb", "log_l_x_agn"))
        return {k: float(v) for k, v in props.items()}


@pytest.mark.parametrize("xray_type", ["yang20", "lopez24"])
def test_properties_are_finite_and_float64_accurate_in_float32(ssp, xray_type):
    """Both properties are finite in pure float32 and within 1e-3 dex of float64."""
    ref = _properties(ssp, True, xray_type, _BURST, _AGN)
    f32 = _properties(ssp, False, xray_type, _BURST, _AGN)
    for name in ("log_l_x_xrb", "log_l_x_agn"):
        assert np.isfinite(f32[name]), f"{name} is not finite in float32: {f32[name]}"
        assert f32[name] == pytest.approx(ref[name], abs=1e-3), name
