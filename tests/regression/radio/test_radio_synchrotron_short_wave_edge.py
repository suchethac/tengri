# SPDX-License-Identifier: BSD-3-Clause
"""Star-formation synchrotron extends to 100 um, as in pcigale's radio module.

The non-thermal power law L_nu = L_nu(nu_ref) (nu / nu_ref)^-alpha has no physical
break at 1 mm. pcigale's ``sed_modules/radio.py`` evaluates it on
``np.logspace(5, 11, 1000)`` nm, i.e. down to 100 um, where it is 1.3 % of the CIGALE
SED at 1 mm and 0.7 % at 850 um. Zeroing it shortward of 1 mm (the former edge) removed
that flux and, in the CIGALE reproduction, offset a nebular excess.

Free-free keeps its 1 mm edge: the Murphy et al. (2011) calibration is a radio relation.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest
from tests._radio_band import band_l_ir

from tengri import FREE, Fixed, SEDModel
from tengri.components.radio import radio as R, radio_precompute as RP

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18
_C_M = 299_792_458.0
_NU_21CM = _C_M / 0.21
_L_IR = 1e44  # erg/s
_Q, _ALPHA = 2.58, 0.8
_UM = 1e4  # Angstrom per micron
_SUB_MM_UM = np.array([100.5, 160.0, 250.0, 350.0, 500.0, 850.0, 990.0])


def _power_law(wave_aa, nu_ref=R._NU_REF_BELL2003_HZ, q=_Q, alpha=_ALPHA):
    nu = _C_AA / wave_aa
    return _L_IR / (3.75e12 * 10.0**q) * (nu / nu_ref) ** (-alpha)


@pytest.mark.parametrize("um", _SUB_MM_UM)
def test_bell2003_sub_mm_is_the_power_law(um):
    """Bell (2003) Eq. 1 anchor times (nu/nu_ref)^-alpha at 100-1000 um."""
    wave = np.array([um * _UM])
    got = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
    np.testing.assert_allclose(got, _power_law(wave), rtol=1e-12)


def test_other_sf_modes_share_the_extended_edge():
    wave = np.array([250.0 * _UM, 850.0 * _UM])
    for fn in (R.radio_sfr_delvecchio2021, R.radio_sfr_mccheyne2022):
        out = np.asarray(fn(wave, _L_IR, 10.0, 1.0))
        assert np.all(out > 0.0)
    split = np.asarray(R.radio_sfr_bell2003_split(wave, _L_IR))
    nonthermal_only = np.asarray(R.radio_sfr_bell2003_split(wave, _L_IR, include_thermal=False))
    assert np.all(nonthermal_only > 0.0)
    # The thermal law keeps its 1 mm edge: 850 um carries synchrotron alone.
    assert split[1] == nonthermal_only[1]
    assert split[0] == nonthermal_only[0]


def test_below_100_um_is_zero_and_edge_is_included():
    wave = np.array([5.0e5, 9.9e5, 1.0e6, 1.0e7 - 1.0])
    got = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] > 0.0 and got[3] > 0.0


def test_values_longward_of_1mm_are_unchanged():
    """Frozen from the 1 mm-edge code (origin/main 2876cf562): bit-identical beyond 1 mm."""
    wave = np.array([1.0e7 + 1.0, 3.0e7, 2.14e9, 1.0e10])
    got = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
    np.testing.assert_array_equal(got, _power_law(wave))


def test_free_free_keeps_the_1mm_edge():
    wave = np.array([850.0 * _UM, 1.0e7 + 1.0])
    ff = np.asarray(R.radio_freefree(wave, _L_IR))
    assert ff[0] == 0.0 and ff[1] > 0.0


def test_matches_pcigale_radio_module_below_1mm():
    """Same q, alpha, 21 cm anchor: tengri / pcigale = 1 at every pcigale node >= 100 um."""
    pytest.importorskip("pcigale")
    from pcigale.sed import SED
    from pcigale.sed_modules import radio as pcigale_radio

    sed = SED()
    sed.add_info("dust.luminosity", _L_IR / 1e7, True, unit="W")
    sed.add_info("agn.intrin_Lnu_2500A_30deg", 0.0, True, unit="W/Hz")
    pcigale_radio.Radio(
        name="radio", qir_sf=_Q, alpha_sf=_ALPHA, R_agn=10.0, alpha_agn=0.7
    ).process(sed)
    w_nm = sed.wavelength_grid
    pc = sed.luminosities["radio.sf_nonthermal"] * w_nm**2 / (_C_M * 1e9) * 1e7  # erg/s/Hz
    sub_mm = (w_nm >= 1.0e5) & (w_nm < 1.0e6)
    assert sub_mm.sum() > 100
    ours = np.asarray(
        R.radio_sfr_bell2003(w_nm[sub_mm] * 10.0, _L_IR, q_ir=_Q, alpha_sf=_ALPHA, nu_ref=_NU_21CM)
    )
    np.testing.assert_allclose(ours, pc[sub_mm], rtol=1e-6)
    # 850 and 500 um by interpolation of the pcigale curve
    for um in (500.0, 850.0):
        ref = np.exp(np.interp(np.log(um * 1e3), np.log(w_nm), np.log(pc)))
        mine = float(
            R.radio_sfr_bell2003(
                np.array([um * _UM]), _L_IR, q_ir=_Q, alpha_sf=_ALPHA, nu_ref=_NU_21CM
            )[0]
        )
        assert mine / ref == pytest.approx(1.0, abs=2e-5)


def test_precompute_grid_covers_100_um_and_keeps_old_nodes():
    wave = np.asarray(RP._WAVE_REST)
    assert wave.min() <= 1.0e6
    old = np.logspace(7, 13, 1024, dtype=np.float64)
    assert np.all(np.isin(old, wave))


def _build(ssp, obs, with_radio=True):
    radio = {
        "sf": {"type": "bell2003", "freefree": False},
        "agn": {"type": "none"},
        "all_params": FREE,
        "q_ir": Fixed(_Q),
        "alpha_sf": Fixed(_ALPHA),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            dust_emission={"type": "draine_li2014"},
            **({"radio": radio} if with_radio else {}),
        )


def test_model_sub_mm_synchrotron_and_ir_luminosities_unchanged(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The SED holds the power law at 850 and 500 um; L_IR, the TIR input and sed_dust_ir
    are the dust block's, untouched by the radio block (it is integrated from sed_dust_ir)."""
    with_r = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    without = _build(synthetic_ssp_wide, synthetic_tophat_obs, with_radio=False)
    key = jax.random.PRNGKey(0)
    s1 = with_r.predict_state(with_r.spec.sample(key))
    s0 = without.predict_state(without.spec.sample(key))
    wave = np.asarray(s1.wave)
    sed = np.asarray(s1.derived["sed_radio"])
    l_tir = float(np.asarray(s1.derived["radio_L_ir_input"])) * 3.828e33
    for um in (500.0, 850.0):
        i = int(np.argmin(np.abs(wave - um * _UM)))
        expect = l_tir / (3.75e12 * 10.0**_Q) * (_C_AA / wave[i] / 1.4e9) ** (-_ALPHA)
        assert sed[i] == pytest.approx(expect, rel=1e-3)
    assert sed[wave < 0.99e6].max() == 0.0
    # The radio block extends the wavelength grid; on the shared nodes the dust SED is identical.
    w0 = np.asarray(s0.wave)
    in1 = np.isin(wave, w0)
    assert in1.sum() == w0.size
    np.testing.assert_allclose(
        np.asarray(s1.derived["sed_dust_ir"])[in1],
        np.asarray(s0.derived["sed_dust_ir"]),
        rtol=1e-8,  # grid-dependent normalization, present with or without the edge change
    )
    assert float(s1.derived["L_ir"]) == float(s0.derived["L_ir"])
    assert float(np.asarray(s1.derived["radio_L_ir_input"])) * 3.828e33 == pytest.approx(
        band_l_ir(s1, "tir"), rel=1e-4
    )
