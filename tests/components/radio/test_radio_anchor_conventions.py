# SPDX-License-Identifier: BSD-3-Clause
"""Contract: radio anchor conventions reproduce CIGALE's radio module (#2663).

Two conventions are structural options on the radio group, and both default to the
behavior they had before they existed:

- ``radio={'agn': {'loudness_ref': '2500_30deg'}}`` anchors the AGN loudness on the disc's
  2500 A luminosity at a 30 deg viewing angle (CIGALE's ``R_agn``). ``'4400'`` (default)
  anchors on the 4400 A B-band luminosity (AGNfitter-rX).
- ``radio={'sf': {'nu_ref': '21cm'}}`` normalizes the star-formation synchrotron at 21 cm
  (1.42758 GHz), CIGALE's anchor. The default 1.4 GHz is Bell (2003)'s.

Acceptance numbers are the issue's table: a CIGALE ``R_agn = 10`` disc at 5 GHz is
8.350135e29 erg/s/Hz; the 4400 A anchor gives 1.106943e30.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.radio.radio import NU_REF_21CM_HZ
from tengri.config.exceptions import ConfigError
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.contract

_Q_IR = 2.5
_NU_5GHZ = 5.0e9
_ALPHA_SF = 0.8
#: CIGALE's 5 GHz luminosity of a disc with R_agn = 10 (issue #2663 table).
_CIGALE_L5_R10 = 8.350135e29
#: Today's 4400 A-anchored value for the same R = 10 (issue #2663 table, pinned commit).
_BASELINE_4400_R10 = 1.106943e30


def _composable_agn():
    """Composable AGN with a Schartmann (2005) disc, the reproducer's disc."""
    return {
        "type": "composable",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "agn_torus_frac": Fixed(1.0), "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(11.0),
        "agn_cos_inc": Fixed(float(np.cos(np.radians(30.0)))),
        "all_params": Fixed(DEFAULT),
    }


def _build(ssp, obs, *, sf, agn_radio, agn=None, dusty=False):
    """Build a radio model.

    ``agn`` is the composable AGN block (``None``: no AGN). ``agn_radio`` is the radio
    group's ``'agn'`` sub-dict. ``dusty`` adds the dust screen and IR emission that feed
    the star-formation synchrotron through L_IR.
    """
    dust = (
        {
            "dust_attenuation": {"type": "two_component", "law": "calzetti"},
            "dust_emission": {"type": "draine_li2014"},
        }
        if dusty
        else {"dust_attenuation": {"type": "none"}, "dust_emission": {"type": "none"}}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            agn=agn if agn is not None else {"type": "none"},
            radio={
                "sf": sf,
                "agn": agn_radio,
                "all_params": FREE,
                "q_ir": Fixed(_Q_IR),
                "alpha_sf": Fixed(_ALPHA_SF),
            },
            **dust,
        )


def _state(model):
    return model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))


def _sed_at(state, nu):
    """Log-interpolated radio L_nu [erg/s/Hz] at frequency ``nu`` [Hz]."""
    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_radio"])
    keep = sed > 0.0
    return float(np.exp(np.interp(np.log(C_AA / nu), np.log(wave[keep]), np.log(sed[keep]))))


def _sf_only(nu_ref=None):
    sf = {"type": "bell2003", "freefree": False}
    if nu_ref is not None:
        sf["nu_ref"] = nu_ref
    return sf


def _agn_only(**extra):
    return {"type": "powerlaw", **extra}


@pytest.fixture(scope="module")
def ssp():
    from tests.contract.test_radio_freefree_switch import _synthetic_ssp

    return _synthetic_ssp()


@pytest.fixture
def obs(synthetic_tophat_obs):
    return synthetic_tophat_obs


# (a) defaults ----------------------------------------------------------------------------


def test_defaults_are_bit_exact_for_a_dusty_agn_and_radio_model(ssp, obs):
    """Spelling the defaults out changes nothing: the default is today's behavior."""
    default = _build(
        ssp,
        obs,
        sf=_sf_only(),
        agn_radio={"type": "powerlaw", "radio_loudness": Fixed(0.5)},
        agn=_composable_agn(),
        dusty=True,
    )
    spelled = _build(
        ssp,
        obs,
        sf=_sf_only(nu_ref=1.4e9),
        agn_radio={"type": "powerlaw", "radio_loudness": Fixed(0.5), "loudness_ref": "4400"},
        agn=_composable_agn(),
        dusty=True,
    )
    a = np.asarray(_state(default).derived["sed_radio"])
    b = np.asarray(_state(spelled).derived["sed_radio"])
    assert np.any(a > 0.0)
    np.testing.assert_array_equal(a, b)


# (b) the issue's reproducer: the 2500 A anchor reproduces CIGALE's 5 GHz luminosity -------

#: Jet cutoff factor exp(-nu / 1e13 Hz) at 5 GHz, the one tengri convention CIGALE lacks.
_CUTOFF_5GHZ = float(np.exp(-_NU_5GHZ / 1e13))
#: The issue's table was pinned at 8d7623e2. Current main normalizes the composable disc
#: 0.22 % higher at 2500 A (L_2500 = 8.368879e28 against 8.350177e28), so the absolute
#: CIGALE check carries that drift; the anchor identity below is exact and is the pin.
_ABS_TOL_AT_MAIN = 3e-3


def test_2500_anchor_is_r_times_the_published_2500a_luminosity(ssp, obs):
    model = _build(
        ssp,
        obs,
        sf={"type": "none", "freefree": False},
        agn_radio=_agn_only(radio_loudness=Fixed(1.0), loudness_ref="2500_30deg"),
        agn=_composable_agn(),
    )
    state = _state(model)
    l_2500 = float(state.derived["L_2500_intrinsic"])
    got = _sed_at(state, _NU_5GHZ)
    # CIGALE's R_agn = 10 on the 2500 A anchor, times the jet's exp(-nu/1e13 Hz) cutoff.
    assert got == pytest.approx(10.0 * _CUTOFF_5GHZ * l_2500, rel=1e-4)


def test_2500_anchor_reproduces_cigale_5ghz_luminosity(ssp, obs):
    model = _build(
        ssp,
        obs,
        sf={"type": "none", "freefree": False},
        agn_radio=_agn_only(radio_loudness=Fixed(1.0), loudness_ref="2500_30deg"),
        agn=_composable_agn(),
    )
    got = _sed_at(_state(model), _NU_5GHZ)
    assert got == pytest.approx(_CIGALE_L5_R10, rel=_ABS_TOL_AT_MAIN)


def test_4400_anchor_is_the_issue_baseline(ssp, obs):
    """The same R = 10 on the default 4400 A anchor: 1.106943e30 at the pinned commit."""
    model = _build(
        ssp,
        obs,
        sf={"type": "none", "freefree": False},
        agn_radio=_agn_only(radio_loudness=Fixed(1.0)),
        agn=_composable_agn(),
    )
    state = _state(model)
    l_4400 = float(state.derived["L_4400_intrinsic"])
    got = _sed_at(state, _NU_5GHZ)
    assert got == pytest.approx(10.0 * _CUTOFF_5GHZ * l_4400, rel=1e-4)
    assert got == pytest.approx(_BASELINE_4400_R10, rel=_ABS_TOL_AT_MAIN)


# (c) 21 cm anchor scales the star-forming radio by (1.4 / 1.42758)^alpha ------------------


def test_21cm_anchor_scales_star_forming_radio(ssp, obs):
    default = _build(ssp, obs, sf=_sf_only(), agn_radio={"type": "none"}, dusty=True)
    cigale = _build(ssp, obs, sf=_sf_only(nu_ref="21cm"), agn_radio={"type": "none"}, dusty=True)
    s_default, s_cigale = _state(default), _state(cigale)
    # The default sits 0.98450 below CIGALE's 21 cm normalization at every frequency.
    expected = (1.4e9 / NU_REF_21CM_HZ) ** _ALPHA_SF
    assert expected == pytest.approx(0.98450, abs=2e-5)
    for nu in (0.15e9, 1.4e9, 30e9):
        ratio = _sed_at(s_default, nu) / _sed_at(s_cigale, nu)
        assert ratio == pytest.approx(expected, rel=1e-4)


# (d) refusals -----------------------------------------------------------------------------


def test_unknown_loudness_ref_names_the_menu(ssp, obs):
    with pytest.raises(ValueError, match=r"'4400', '2500_30deg'"):
        _build(
            ssp,
            obs,
            sf=_sf_only(),
            agn_radio=_agn_only(loudness_ref="bogus"),
            agn=_composable_agn(),
        )


def test_2500_anchor_needs_the_composable_disc(ssp, obs):
    """No composable AGN publishes L_2500_intrinsic; the radio must not fall back silently."""
    with pytest.raises(ConfigError, match="composable"):
        _build(
            ssp,
            obs,
            sf={"type": "none", "freefree": False},
            agn_radio=_agn_only(loudness_ref="2500_30deg"),
            agn=None,
        )


def test_nonpositive_nu_ref_names_the_bound(ssp, obs):
    with pytest.raises(ValueError, match="positive"):
        _build(ssp, obs, sf=_sf_only(nu_ref=0.0), agn_radio={"type": "none"})
