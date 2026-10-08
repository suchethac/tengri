# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``radio.sf.ir_window`` reaches the radio block (#2763).

Each q-based model defaults to the IR window of its own paper.

- Grammar wiring: ``radio={'sf': {'ir_window': 'tir'}}`` reaches
  ``model.spec.radio_ir_window`` and the live ``RadioSEDComponentConfig``.
- Default: an absent key resolves per star-formation model (bell2003, bell2003_split,
  delvecchio2021, mccheyne2022: ``'tir'``) and equals the explicit ``'tir'`` bit for
  bit, at L_TIR / (3.75e12 10^q) with L_TIR an independent dense band integral.
- ``'total'``: the q relation on the published ``L_ir`` (CIGALE's convention).
- Physics: the default scales the 1.4 GHz synchrotron by L(8-1000 um) / L_ir against
  ``'total'``.
- Errors: an unknown window raises at the public grammar.
- Round trip: ``ir_window`` survives ``to_groups()``; a default build emits nothing.
- Compile identity: the window is structural, so it changes ``compile_signature()``.

Uses the synthetic SSP of ``test_radio_freefree_switch`` (CI-runnable, no data grids).
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel
from tengri.components.radio.component import RadioSEDComponentConfig
from tengri.components.radio.radio import NU_REF_21CM_HZ
from tengri.parameters.groups import parse_groups
from tengri.radio import radio_sfr_bell2003
from tengri.utils.physics_constants import C_AA
from tests._radio_band import band_l_ir
from tests.contract.test_radio_freefree_switch import _synthetic_ssp

pytestmark = pytest.mark.contract

_RADIO_WAVE_MIN_AA = 1.0e7


@pytest.fixture(scope="module")
def ssp():
    return _synthetic_ssp()


def _build(ssp, obs, ir_window=None, **sf_extra):
    sf = {"type": "bell2003", "freefree": False, **sf_extra}
    if ir_window is not None:
        sf["ir_window"] = ir_window
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
                "sf": sf,
                "agn": {"type": "none"},
                "all_params": FREE,
                "q_ir": Fixed(2.5),
                "alpha_sf": Fixed(0.8),
            },
        )


def _config(model) -> RadioSEDComponentConfig:
    return next(c for n, c in model._component_configs if n == "radio")


def _state(model):
    return model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))


def test_ir_window_reaches_spec_and_component(ssp, synthetic_tophat_obs):
    model = _build(ssp, synthetic_tophat_obs, "fir")
    assert model.spec.radio_ir_window == "fir"
    assert _config(model).ir_window == "fir"


def test_default_is_the_models_own_paper_window(ssp, synthetic_tophat_obs):
    """No key: bell2003 resolves to Bell's 8-1000 um TIR; an explicit 'total' is kept."""
    absent = _build(ssp, synthetic_tophat_obs)
    assert absent.spec.radio_ir_window is None
    assert _config(absent).resolved_ir_window == "tir"
    assert _config(_build(ssp, synthetic_tophat_obs, "total")).resolved_ir_window == "total"


@pytest.mark.parametrize(
    ("sfr_mode", "window"),
    [
        ("bell2003", "tir"),
        ("bell2003_split", "tir"),
        ("delvecchio2021", "tir"),
        ("mccheyne2022", "tir"),
        ("none", "total"),
    ],
)
def test_each_sf_model_resolves_its_papers_window(sfr_mode, window):
    cfg = RadioSEDComponentConfig(sfr_mode=sfr_mode)
    assert cfg.resolved_ir_window == window
    assert RadioSEDComponentConfig(sfr_mode=sfr_mode, ir_window="total").resolved_ir_window == (
        "total"
    )


def test_default_radio_is_the_q_relation_on_the_independent_tir(ssp, synthetic_tophat_obs):
    """Default sed_radio = L_TIR / (3.75e12 10^q) x (nu / 1.4 GHz)^-alpha, L_TIR forward-derived.

    Absent key == explicit 'tir' bit for bit. L_TIR is a dense log-nu resampling of the
    dust SED (tests/_radio_band.py), not the radio block's own cell integral.
    """
    absent, explicit = (
        _state(_build(ssp, synthetic_tophat_obs)),
        _state(_build(ssp, synthetic_tophat_obs, "tir")),
    )
    a = np.asarray(absent.derived["sed_radio"])
    assert np.array_equal(a, np.asarray(explicit.derived["sed_radio"]))
    band = np.asarray(absent.wave) > _RADIO_WAVE_MIN_AA
    expected = np.asarray(radio_sfr_bell2003(absent.wave, band_l_ir(absent, "tir"), 2.5, 0.8))
    # Two integrals of one piecewise-linear SED (edge-exact cell sum vs dense resampling).
    np.testing.assert_allclose(a[band], expected[band], rtol=1e-5)
    assert np.all(a[band] > 0.0)


def test_explicit_total_is_cigales_convention(ssp, synthetic_tophat_obs):
    """ir_window='total': the q relation on the published dust power L_ir, as pcigale does."""
    state = _state(_build(ssp, synthetic_tophat_obs, "total"))
    a = np.asarray(state.derived["sed_radio"])
    band = np.asarray(state.wave) > _RADIO_WAVE_MIN_AA
    expected = np.asarray(radio_sfr_bell2003(state.wave, float(state.derived["L_ir"]), 2.5, 0.8))
    np.testing.assert_allclose(a[band], expected[band], rtol=1e-8)


def test_tir_scales_the_radio_flux_by_the_band_fraction(ssp, synthetic_tophat_obs):
    """sed_radio(default) / sed_radio('total') = L(8-1000 um) / L_ir, band integral independent."""
    tot, tir = (
        _state(_build(ssp, synthetic_tophat_obs, "total")),
        _state(_build(ssp, synthetic_tophat_obs)),
    )
    wave = np.asarray(tot.wave)
    l_tir = band_l_ir(tir, "tir")
    log_l_ir = tot.derived.get("log_L_ir")
    l_ir = float(10.0**log_l_ir) if log_l_ir is not None else float(tot.derived["L_ir"])
    band = wave > _RADIO_WAVE_MIN_AA
    ratio = np.asarray(tir.derived["sed_radio"])[band] / np.asarray(tot.derived["sed_radio"])[band]
    assert 0.5 < l_tir / l_ir < 1.0  # the 8-1000 um band holds only part of the dust power
    np.testing.assert_allclose(ratio, l_tir / l_ir, rtol=2e-3)


def test_unknown_window_raises_at_grammar(ssp, synthetic_tophat_obs):
    with pytest.raises(ValueError, match="ir_window"):
        _build(ssp, synthetic_tophat_obs, "mir")


def test_ir_window_round_trips_and_default_emits_nothing():
    base = dict(sfh={"type": "const", "all_params": FREE}, redshift=Fixed(0.1))
    spec = parse_groups(radio={"sf": {"type": "bell2003", "ir_window": "fir"}}, **base)
    groups = spec.to_groups()
    assert groups["radio"]["sf"]["ir_window"] == "fir"
    assert parse_groups(**groups).radio_ir_window == "fir"
    default_groups = parse_groups(radio={"sf": {"type": "bell2003"}}, **base).to_groups()
    assert "ir_window" not in default_groups["radio"].get("sf", {})
    total = parse_groups(radio={"sf": {"type": "bell2003", "ir_window": "total"}}, **base)
    assert total.to_groups()["radio"]["sf"]["ir_window"] == "total"


def test_ir_window_changes_the_compile_signature(ssp, synthetic_tophat_obs):
    sig_total = _build(ssp, synthetic_tophat_obs).compile_signature()
    sig_tir = _build(ssp, synthetic_tophat_obs, "tir").compile_signature()
    assert sig_total != sig_tir


def test_21cm_anchor_with_tir_window(ssp, synthetic_tophat_obs):
    """nu_ref='21cm' + ir_window='tir': L_nu(c/0.21 m) = L(8-1000 um) / (3.75e12 * 10**q)."""
    state = _state(_build(ssp, synthetic_tophat_obs, "tir", nu_ref="21cm"))
    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_radio"])
    keep = sed > 0.0
    got = np.exp(np.interp(np.log(C_AA / NU_REF_21CM_HZ), np.log(wave[keep]), np.log(sed[keep])))
    assert got == pytest.approx(band_l_ir(state, "tir") / (3.75e12 * 10.0**2.5), rel=2e-3)


def test_combined_sf_settings_round_trip():
    sf = {"type": "bell2003", "nu_ref": "21cm", "ir_window": "tir", "freefree": False}
    base = dict(sfh={"type": "const", "all_params": FREE}, redshift=Fixed(0.1))
    spec = parse_groups(radio={"sf": sf}, **base)
    groups = spec.to_groups()
    again = parse_groups(**groups)
    assert (again.radio_ir_window, again.radio_sf_nu_ref) == ("tir", NU_REF_21CM_HZ)
    assert groups["radio"]["sf"]["ir_window"] == "tir"


def test_published_radio_l_ir_input_follows_the_window(ssp, synthetic_tophat_obs):
    """radio_L_ir_input [Lsun]: the independent 8-1000 um integral for 'tir', L_ir for 'total'."""
    from tengri.utils.physics_constants import L_SUN

    tir = _state(_build(ssp, synthetic_tophat_obs, "tir"))
    got = float(tir.derived["radio_L_ir_input"])
    assert got == pytest.approx(band_l_ir(tir, "tir") / L_SUN, rel=2e-3)
    assert float(tir.derived["radio_log_L_ir_input"]) == pytest.approx(np.log10(got), abs=1e-9)

    tot = _state(_build(ssp, synthetic_tophat_obs, "total"))
    assert float(tot.derived["radio_L_ir_input"]) == pytest.approx(
        float(tot.derived["L_ir"]) / L_SUN, rel=1e-12
    )
    default = _state(_build(ssp, synthetic_tophat_obs))
    assert float(default.derived["radio_L_ir_input"]) == got
