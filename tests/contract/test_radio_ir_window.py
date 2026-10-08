# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``radio.sf.ir_window`` reaches the radio block and defaults to bit-identical (#2763).

- Grammar wiring: ``radio={'sf': {'ir_window': 'tir'}}`` reaches
  ``model.spec.radio_ir_window`` and the live ``RadioSEDComponentConfig``.
- Default: an absent key and the explicit ``'total'`` give a bit-identical
  ``sed_radio``, equal to the q relation evaluated on the published ``L_ir``.
- Physics: ``'tir'`` scales the 1.4 GHz synchrotron by L(8-1000 um) / L_ir, the
  band integral taken independently on a dense interpolation of the dust SED.
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
from tengri.components.radio.component import IR_WINDOWS_AA, RadioSEDComponentConfig
from tengri.components.radio.radio import NU_REF_21CM_HZ
from tengri.parameters.groups import parse_groups
from tengri.radio import radio_sfr_bell2003
from tengri.utils.physics_constants import C_AA
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
    model = _build(ssp, synthetic_tophat_obs, "tir")
    assert model.spec.radio_ir_window == "tir"
    assert _config(model).ir_window == "tir"


def test_default_is_total_and_bit_identical(ssp, synthetic_tophat_obs):
    """Absent key == explicit 'total' bit for bit; both match the q relation."""
    absent = _build(ssp, synthetic_tophat_obs)
    explicit = _build(ssp, synthetic_tophat_obs, "total")
    assert absent.spec.radio_ir_window == "total"
    s_absent, s_explicit = _state(absent), _state(explicit)
    a = np.asarray(s_absent.derived["sed_radio"])
    assert np.array_equal(a, np.asarray(s_explicit.derived["sed_radio"]))

    wave = s_absent.wave
    band = np.asarray(wave) > _RADIO_WAVE_MIN_AA
    # Precedent tolerance of test_radio_freefree_switch (1e-8): the model forms the same
    # quotient through its log companion, so the direct formula agrees to round-off only.
    expected = np.asarray(radio_sfr_bell2003(wave, float(s_absent.derived["L_ir"]), 2.5, 0.8))
    np.testing.assert_allclose(a[band], expected[band], rtol=1e-8)
    assert np.all(a[band] > 0.0)


def test_tir_scales_the_radio_flux_by_the_band_fraction(ssp, synthetic_tophat_obs):
    """sed_radio(tir) / sed_radio(total) = L(8-1000 um) / L_ir, band integral independent."""
    tot, tir = (
        _state(_build(ssp, synthetic_tophat_obs)),
        _state(_build(ssp, synthetic_tophat_obs, "tir")),
    )
    wave = np.asarray(tot.wave)
    sed_ir = np.asarray(tir.derived["sed_dust_ir"])
    # Independent band integral: dense log-nu interpolation, exact window mask.
    lo, hi = IR_WINDOWS_AA["tir"]
    sel = (wave > 1.0e3) & (wave < 5.0e8)
    nu_dense = np.geomspace(C_AA / hi, C_AA / lo, 400001)
    f_dense = np.interp(nu_dense, (C_AA / wave[sel])[::-1], sed_ir[sel][::-1])
    l_tir = float(np.trapezoid(f_dense, nu_dense))
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


def test_ir_window_changes_the_compile_signature(ssp, synthetic_tophat_obs):
    sig_total = _build(ssp, synthetic_tophat_obs).compile_signature()
    sig_tir = _build(ssp, synthetic_tophat_obs, "tir").compile_signature()
    assert sig_total != sig_tir


def test_21cm_anchor_with_tir_window(ssp, synthetic_tophat_obs):
    """nu_ref='21cm' + ir_window='tir': L_nu(c/0.21 m) = L(8-1000 um) / (3.75e12 * 10**q)."""
    state = _state(_build(ssp, synthetic_tophat_obs, "tir", nu_ref="21cm"))
    wave = np.asarray(state.wave)
    sed_ir = np.asarray(state.derived["sed_dust_ir"])
    lo, hi = IR_WINDOWS_AA["tir"]
    sel = (wave > 1.0e3) & (wave < 5.0e8)
    nu_dense = np.geomspace(C_AA / hi, C_AA / lo, 400001)
    f_dense = np.interp(nu_dense, (C_AA / wave[sel])[::-1], sed_ir[sel][::-1])
    l_tir = float(np.trapezoid(f_dense, nu_dense))
    sed = np.asarray(state.derived["sed_radio"])
    keep = sed > 0.0
    got = np.exp(np.interp(np.log(C_AA / NU_REF_21CM_HZ), np.log(wave[keep]), np.log(sed[keep])))
    assert got == pytest.approx(l_tir / (3.75e12 * 10.0**2.5), rel=2e-3)


def test_combined_sf_settings_round_trip():
    sf = {"type": "bell2003", "nu_ref": "21cm", "ir_window": "tir", "freefree": False}
    base = dict(sfh={"type": "const", "all_params": FREE}, redshift=Fixed(0.1))
    spec = parse_groups(radio={"sf": sf}, **base)
    groups = spec.to_groups()
    again = parse_groups(**groups)
    assert (again.radio_ir_window, again.radio_sf_nu_ref) == ("tir", NU_REF_21CM_HZ)
    assert groups["radio"]["sf"]["ir_window"] == "tir"
