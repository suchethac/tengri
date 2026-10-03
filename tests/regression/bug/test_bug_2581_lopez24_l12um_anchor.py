# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2581: the lopez24 corona is anchored to the AGN's own 12 um.

Lopez et al. (2024, A&A 692, A209) tie the corona to the nuclear 12 um luminosity
of the AGN components, ``L(2-10 keV) = nu L_nu(12 um) / 10**alpha_IRX`` (Asmus et
al. 2015 convention), with the 12 um luminosity summed over accretion disc, torus
and polar dust. The composable AGN must therefore publish ``log_L_12um``, the
X-ray block must read it, and the amplitude must be formed in log space so that
pure float32 keeps the X-ray wing finite.
"""

from functools import cache
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
LAMBDA_12UM = 1.2e5  # Angstrom
ALPHA_IRX = 0.3  # declared default of xray_alpha_irx

#: Declared defaults of the ``xray_*`` parameters the builds below leave Fixed.
_XP = {
    "xray_gamma_agn": 1.8,
    "xray_gamma_hmxb": 2.0,
    "xray_gamma_lmxb": 1.56,
    "xray_E_cut": 300.0,
    "xray_delta_alpha_ox": 0.0,
    "xray_log_nh": 0.0,
    "xray_alpha_irx": ALPHA_IRX,
    "xray_det_hmxb": 0.0,
    "xray_det_lmxb": 0.0,
}


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_PATH))


def _agn(torus, disc, cos_inc_deg):
    return {
        "type": "composable",
        "disc": {"type": disc, "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus, "agn_torus_frac": Fixed(1.0), "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(12.0),
        "agn_cos_inc": Fixed(float(np.cos(np.radians(cos_inc_deg)))),
        "all_params": Fixed(DEFAULT),
    }


def _build(ssp, agn):
    kwargs = {"agn": agn} if agn is not None else {}
    return SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
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
        xray={"type": "lopez24", "log_nh": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
        **kwargs,
    )


def _band(wave, lnu, e1, e2, n=20001):
    """log-log quadrature of an emitted spectrum over [e1, e2] keV, in erg/s."""
    wave = np.asarray(wave, dtype=np.float64)
    lnu = np.asarray(lnu, dtype=np.float64)
    energy = np.linspace(e1, e2, n)
    lam = HC / energy
    o = np.argsort(wave)
    interp = np.exp(np.interp(np.log(lam), np.log(wave[o]), np.log(np.maximum(lnu[o], 1e-300))))
    nu = C_AA / lam
    oo = np.argsort(nu)
    return np.trapezoid(interp[oo], nu[oo])


def _log_nu_lnu_12um(wave, lnu):
    """log10 of nu L_nu at 12 um by log-log interpolation of an L_nu spectrum [dex re erg/s]."""
    wave = np.asarray(wave, dtype=np.float64)
    lnu = np.asarray(lnu, dtype=np.float64)
    log_lnu = np.interp(np.log10(LAMBDA_12UM), np.log10(wave), np.log10(np.maximum(lnu, 1e-300)))
    return log_lnu + np.log10(C_AA / LAMBDA_12UM)


@cache
def _measure(torus, disc, cos_inc_deg, x64):
    """Everything the cells below read, from one forward pass of one build."""
    ssp = load_ssp_data(str(_SSP_PATH))
    with jax.enable_x64(x64):
        model = _build(ssp, _agn(torus, disc, cos_inc_deg))
        state = model.predict_state({})
        derived = state.derived
        wave = np.asarray(state.wave)
        agn_sed = (
            np.asarray(derived["sed_agn_disc"], dtype=np.float64)
            + np.asarray(derived["sed_agn_torus"], dtype=np.float64)
            + np.asarray(derived["sed_agn_polar"], dtype=np.float64)
        )
        comp = XRaySEDComponent(config=XRaySEDComponentConfig(model="lopez24"))
        inputs = comp.emitter_inputs(derived)
        terms = comp.emission_terms(
            _XP,
            state.wave,
            **{k: np.asarray(v, dtype=state.wave.dtype) for k, v in inputs.items()},
        )
        published = derived.get("log_L_12um")
        return {
            "published": None if published is None else float(published),
            "expected": float(_log_nu_lnu_12um(wave, agn_sed)),
            "band_agn": _band(wave, np.asarray(terms["agn"]), 2.0, 10.0),
            "sed_xray_finite": bool(np.all(np.isfinite(np.asarray(derived["sed_xray"])))),
        }


_CASES = [
    pytest.param(torus, disc, cos_deg, id=f"{torus}-{disc}-i{cos_deg}")
    for torus in ("skirtor", "fritz")
    for disc in ("schartmann2005", "skirtor")
    for cos_deg in (0.0, 30.0, 70.0)
]
_DTYPES = [pytest.param(True, id="float64"), pytest.param(False, id="float32")]


@pytest.mark.parametrize("x64", _DTYPES)
@pytest.mark.parametrize(("torus", "disc", "cos_deg"), _CASES)
def test_agn_publishes_log_l_12um_of_disc_torus_and_polar_dust(torus, disc, cos_deg, x64):
    """log_L_12um == log10 nu L_nu(12 um) of sed_agn_disc + sed_agn_torus + sed_agn_polar."""
    m = _measure(torus, disc, cos_deg, x64)
    assert m["published"] is not None, "the composable AGN does not publish log_L_12um"
    assert m["published"] == pytest.approx(m["expected"], abs=2e-3)


@pytest.mark.parametrize("x64", _DTYPES)
@pytest.mark.parametrize(("torus", "disc", "cos_deg"), _CASES)
def test_corona_band_luminosity_is_the_12um_over_alpha_irx(torus, disc, cos_deg, x64):
    """int_{2-10 keV} agn dnu = 1.01 * 10**(log_L_12um - alpha_IRX) at log N_H = 0.

    The 1.01 is the 1 % scattered fraction on top of the unabsorbed primary.
    """
    m = _measure(torus, disc, cos_deg, x64)
    assert m["published"] is not None, "the composable AGN does not publish log_L_12um"
    expected = 1.01 * 10.0 ** (m["published"] - ALPHA_IRX)
    assert m["band_agn"] == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("x64", _DTYPES)
@pytest.mark.parametrize(("torus", "disc", "cos_deg"), _CASES)
def test_sed_xray_is_finite(torus, disc, cos_deg, x64):
    """The X-ray wing is finite in pure float32 as in float64."""
    assert _measure(torus, disc, cos_deg, x64)["sed_xray_finite"]


@pytest.mark.parametrize("x64", _DTYPES)
def test_lopez24_without_an_agn_has_no_corona_and_a_finite_sed(ssp, x64):
    """No AGN in the model: the corona is exactly zero, XRB and hot gas are still emitted."""
    with jax.enable_x64(x64):
        model = _build(ssp, None)
        state = model.predict_state({})
        comp = XRaySEDComponent(config=XRaySEDComponentConfig(model="lopez24"))
        inputs = comp.emitter_inputs(state.derived)
        terms = comp.emission_terms(
            _XP,
            state.wave,
            **{k: np.asarray(v, dtype=state.wave.dtype) for k, v in inputs.items()},
        )
        sed_xray = np.asarray(state.derived["sed_xray"])
    assert np.all(np.asarray(terms["agn"]) == 0.0)
    assert np.all(np.isfinite(sed_xray))
    for name in ("hmxb", "lmxb", "hotgas"):
        assert np.sum(np.asarray(terms[name])) > 0.0, name
    assert np.sum(sed_xray) > 0.0
