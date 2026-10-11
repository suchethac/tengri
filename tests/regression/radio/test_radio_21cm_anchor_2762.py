# SPDX-License-Identifier: BSD-3-Clause
"""Frequency at which the star-formation synchrotron is normalized (#2762).

FIR-radio ratio: q = log10[(TIR / 3.75e12 Hz) / L_nu(nu0)], so

    L_nu(nu0) = L_IR / (3.75e12 Hz x 10^q)

at the frequency nu0 the q was measured at. Bell (2003, ApJ 586, 794, Eq. 1)
measures it at 1.4 GHz; Yun et al. (2001) likewise at 1.4 GHz (Helou et al. 1985
at 1.49 GHz). tengri's ``bell2003`` therefore anchors at 1.4 GHz. pcigale's radio
module (``sed_modules/radio.py``) writes the same relation at lambda = 21 cm
exactly, nu0 = c / 0.21 m = 1.42758 GHz. ``radio.sf.nu_ref = "21cm"`` selects that
anchor; the default stays Bell's.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest
from tests._radio_band import band_l_ir

from tengri import FREE, Fixed, SEDModel
from tengri.components.radio import radio as R
from tengri.components.radio.component import RadioSEDComponentConfig
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.regression_paper

_C_M = 299_792_458.0
_NU_21CM = _C_M / 0.21  # 1.42758 GHz
_C_AA = 2.99792458e18
_L_IR = 1e44  # erg/s
_Q, _ALPHA = 2.58, 0.8


def _l_nu(nu, **kw):
    return float(
        R.radio_sfr_bell2003(np.array([_C_AA / nu]), _L_IR, q_ir=_Q, alpha_sf=_ALPHA, **kw)[0]
    )


def test_21cm_constant_is_c_over_21_cm():
    assert pytest.approx(1.427584e9, rel=1e-6) == R.NU_REF_21CM_HZ


def test_normalization_frequency_follows_the_formula():
    """L_nu(nu0) = L_IR / (3.75e12 x 10^q) at the anchor, whichever anchor is chosen."""
    cal = _L_IR / (3.75e12 * 10.0**_Q)
    assert _l_nu(1.4e9) == pytest.approx(cal, rel=1e-6)
    assert _l_nu(_NU_21CM, nu_ref=_NU_21CM) == pytest.approx(cal, rel=1e-6)
    # Away from its anchor each side is the power law: the default at 21 cm sits below.
    assert _l_nu(_NU_21CM) == pytest.approx(cal * (_NU_21CM / 1.4e9) ** -_ALPHA, rel=1e-6)


def test_default_over_21cm_is_the_constant_power_law_ratio():
    """(1.4 GHz / 1.42758 GHz)^alpha = 0.98451 at alpha = 0.8, at every frequency."""
    nu = np.array([0.15e9, 1.4e9, 5e9, 30e9])
    ratio = [_l_nu(f) / _l_nu(f, nu_ref=_NU_21CM) for f in nu]
    np.testing.assert_allclose(ratio, (1.4e9 / _NU_21CM) ** _ALPHA, rtol=1e-6)
    assert ratio[0] == pytest.approx(0.98451, abs=1e-5)


def test_config_rejects_anchor_for_other_relations():
    with pytest.raises(ConfigError, match="bell2003"):
        RadioSEDComponentConfig(sfr_mode="delvecchio2021", sf_nu_ref=_NU_21CM)
    with pytest.raises(ValueError, match="positive"):
        RadioSEDComponentConfig(sf_nu_ref=-1.0)


def _build(ssp, obs, **sf_extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            dust_emission={"type": "draine_li2014"},
            radio={
                "sf": {"type": "bell2003", "freefree": False, **sf_extra},
                "agn": {"type": "none"},
                "all_params": FREE,
                "q_ir": Fixed(_Q),
                "alpha_sf": Fixed(_ALPHA),
            },
        )


def _radio_at(model, nu):
    state = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
    wave, sed = np.asarray(state.wave), np.asarray(state.derived["sed_radio"])
    keep = sed > 0.0  # the radio block is zero shortward of 1 mm
    value = np.exp(np.interp(np.log(_C_AA / nu), np.log(wave[keep]), np.log(sed[keep])))
    return value, float(state.derived["L_ir"]), band_l_ir(state, "tir")


@pytest.mark.parametrize("spelling", ["21cm", "21 cm", _NU_21CM])
def test_build_spelling_anchors_at_21cm(synthetic_ssp_wide, synthetic_tophat_obs, spelling):
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, nu_ref=spelling)
    got, _, l_tir = _radio_at(model, _NU_21CM)
    # bell2003 defaults to its paper's window: q is on the 8-1000 um TIR (Bell 2003)
    assert got == pytest.approx(l_tir / (3.75e12 * 10.0**_Q), rel=1e-4)


def test_build_default_anchors_at_1p4_ghz(synthetic_ssp_wide, synthetic_tophat_obs):
    got, _, l_tir = _radio_at(_build(synthetic_ssp_wide, synthetic_tophat_obs), 1.4e9)
    assert got == pytest.approx(l_tir / (3.75e12 * 10.0**_Q), rel=1e-4)


def test_build_rejects_bad_spelling(synthetic_ssp_wide, synthetic_tophat_obs):
    with pytest.raises(ValueError, match="nu_ref"):
        _build(synthetic_ssp_wide, synthetic_tophat_obs, nu_ref="1.5GHz")


def test_matches_pcigale_radio_module(synthetic_ssp_wide, synthetic_tophat_obs):
    """Same q and alpha: tengri / pcigale = 1 with the 21 cm anchor, 0.98451 without."""
    pytest.importorskip("pcigale")
    from pcigale.sed import SED
    from pcigale.sed_modules import radio as pcigale_radio

    nu = np.array([0.15e9, 1.4e9, 5e9, 30e9])
    # pcigale feeds its radio module the TOTAL dust luminosity (``dust.luminosity``), so
    # the parity comparison pins ir_window='total'; tengri's default is Bell's 8-1000 um TIR.
    anchored, l_ir, _ = _radio_at(
        _build(synthetic_ssp_wide, synthetic_tophat_obs, nu_ref="21cm", ir_window="total"), nu[0]
    )
    sed = SED()
    sed.add_info("dust.luminosity", l_ir / 1e7, True, unit="W")
    sed.add_info("agn.intrin_Lnu_2500A_30deg", 0.0, True, unit="W/Hz")
    pcigale_radio.Radio(
        name="radio", qir_sf=_Q, alpha_sf=_ALPHA, R_agn=10.0, alpha_agn=0.7
    ).process(sed)
    w = sed.wavelength_grid
    c_nm = _C_M * 1e9
    lnu = sed.luminosities["radio.sf_nonthermal"] * w**2 / c_nm * 1e7  # erg/s/Hz
    pc = np.interp(c_nm / nu, w, lnu)
    default_model = _build(synthetic_ssp_wide, synthetic_tophat_obs, ir_window="total")
    anchored_model = _build(
        synthetic_ssp_wide, synthetic_tophat_obs, nu_ref="21cm", ir_window="total"
    )
    for f, ref in zip(nu, pc, strict=True):
        a, _, _ = _radio_at(anchored_model, f)
        d, _, _ = _radio_at(default_model, f)
        assert a / ref == pytest.approx(1.0, abs=2e-5)
        assert d / ref == pytest.approx((1.4e9 / _NU_21CM) ** _ALPHA, abs=2e-5)
    assert anchored > 0.0
