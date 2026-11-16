# SPDX-License-Identifier: BSD-3-Clause
"""Engagement evidence (#3505 X1): every gate records its outcome, with a reason."""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.config.exceptions import PrecomputeRefusal
from tengri.forward.precompute.band_table import Decline
from tengri.forward.precompute_report import engagement_reports, precompute_engagement_report

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("bad", [None, "", "   ", 3])
def test_decline_requires_a_reason(bad):
    with pytest.raises((TypeError, ValueError)):
        Decline(bad)


def test_decline_keeps_its_reason():
    assert Decline("why").reason == "why"


def test_feature_top_up_records_a_refusal_as_a_decline_in_the_one_memo(
    monkeypatch, synthetic_tophat_obs, ssp_data_wne
):
    from tengri.forward.precompute.band_table import Decline
    from tengri.inference import fitter

    def refuse(model, cfg):
        raise PrecomputeRefusal("no lines to tabulate")

    monkeypatch.setattr(fitter, "_memoized_approx_clone", refuse)
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    assert fitter._try_feature_top_up(model, ("cfg",)) is None
    assert model.band_table_decline("approx", "feature_precomp") == Decline("no lines to tabulate")


@pytest.mark.parametrize("bug", [RuntimeError("a bug"), ValueError("a shape bug")])
def test_feature_top_up_lets_a_bug_propagate_even_when_it_is_a_value_error(
    monkeypatch, bug, synthetic_tophat_obs, ssp_data_wne
):
    from tengri.inference import fitter

    def broken(model, cfg):
        raise bug

    monkeypatch.setattr(fitter, "_memoized_approx_clone", broken)
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    with pytest.raises(type(bug), match="bug"):
        fitter._try_feature_top_up(model, ("cfg",))
    assert model.band_table_decline("approx", "feature_precomp") is None


def test_the_real_refusal_sites_raise_the_refusal_class(ssp_data_wne, synthetic_tophat_obs):
    """A FeaturePrecomp with no lines to tabulate raises PrecomputeRefusal (a ValueError too)."""
    from tengri.forward.sed_model import FeaturePrecomp

    model = SEDModel.build(
        ssp_data=ssp_data_wne,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "ssp"},
        redshift=Fixed(0.1),
    )
    with pytest.raises(PrecomputeRefusal, match="no emission lines to tabulate"):
        model.with_approx(FeaturePrecomp())


def test_the_decline_is_carried_to_the_fallback_clone(ssp_data_wne, synthetic_tophat_obs):
    from tengri.forward.precompute_report import precompute_engagement_report
    from tengri.inference.fitter import _carry_approx_declines

    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    model.record_decline("approx", "feature_precomp", "why")
    clone = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    assert _carry_approx_declines(model, clone) is clone
    state = precompute_engagement_report(clone).feature_precomp
    assert (state.state, state.reason) == ("declined", "why")


def _eb_model(ssp, obs, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "ssp"},
        redshift=Fixed(0.1),
        approx=approx,
    )


def test_energy_balance_lut_decline_is_recorded_at_the_gate(synthetic_tophat_obs, ssp_data_wne):
    """The gate that declines the LUT records its own reason; the report only reads it."""
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    model._energy_balance_lut(model._build_component_chain())
    recorded = model.band_table_decline("energy_balance", "lut").reason
    assert recorded and "WavePrecomp" in recorded
    report = precompute_engagement_report(model)
    assert report.energy_balance_lut.state == "declined"
    assert report.energy_balance_lut.reason == recorded


def _refusing_clone(monkeypatch, fitter, message):
    def refuse(model, cfg):
        raise PrecomputeRefusal(message)

    monkeypatch.setattr(fitter, "_memoized_approx_clone", refuse)


@pytest.mark.parametrize("bug", [RuntimeError("a bug"), ValueError("a shape bug")])
def test_add_feature_precomp_lets_a_bug_propagate(
    monkeypatch, bug, synthetic_tophat_obs, ssp_data_wne
):
    from tengri.inference import fitter

    monkeypatch.setattr(fitter, "_has_line_adjacent_channel", lambda m: False)
    monkeypatch.setattr(fitter, "_feature_precomp_can_pay", lambda m, **kw: True)

    def broken(model, cfg):
        raise bug

    monkeypatch.setattr(fitter, "_memoized_approx_clone", broken)
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    with pytest.raises(type(bug), match="bug"):
        fitter.Fitter._add_feature_precomp(None, model)


def test_add_feature_precomp_records_a_refusal_and_warns_with_its_reason(
    monkeypatch, synthetic_tophat_obs, ssp_data_wne
):
    from tengri.inference import fitter

    monkeypatch.setattr(fitter, "_has_line_adjacent_channel", lambda m: False)
    monkeypatch.setattr(fitter, "_feature_precomp_can_pay", lambda m, **kw: True)
    _refusing_clone(monkeypatch, fitter, "nothing to tabulate")
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    with pytest.warns(UserWarning, match="nothing to tabulate"):
        out = fitter.Fitter._add_feature_precomp(None, model)
    assert out is model
    assert model.band_table_decline("approx", "feature_precomp") == Decline("nothing to tabulate")


class _SlottedModel:
    __slots__ = ()
    _lut_forward_bias_cache = None

    def predict_photometry(self, params):
        import numpy as np

        return np.array([1.0, 2.0])


class _ReadOnlyModel:
    def predict_photometry(self, params):
        import numpy as np

        return np.array([1.0, 2.0])

    def __setattr__(self, name, value):
        raise RuntimeError("a bug in __setattr__")


def _bias_stub():
    from types import SimpleNamespace

    import numpy as np

    return SimpleNamespace(spec=None, predict_photometry=lambda p: np.array([1.0, 2.0]))


def test_lut_bias_cache_write_refused_by_a_slotted_model_is_best_effort(monkeypatch):
    from tengri.inference import fitter

    monkeypatch.setattr(fitter, "_central_params", lambda spec: {})
    bias = fitter._lut_forward_bias(_bias_stub(), _SlottedModel(), "photometry")
    assert bias.shape == (2,)


def test_lut_bias_cache_write_bug_propagates(monkeypatch):
    from tengri.inference import fitter

    monkeypatch.setattr(fitter, "_central_params", lambda spec: {})
    lut = _ReadOnlyModel()
    with pytest.raises(RuntimeError, match="a bug in __setattr__"):
        fitter._lut_forward_bias(_bias_stub(), lut, "photometry")


def _age_binned_model(ssp, obs):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": [{"law": "calzetti", "window_log_yr": (None, None)}],
            "tau_0": 0.5,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
        approx=None,
    )


def test_a_wave_precomp_refusal_is_a_precompute_refusal(ssp_data_wne, synthetic_tophat_obs):
    """The age_binned refusal at construction is the refusal class, not a bare ValueError."""
    from tengri import WavePrecomp

    model = _age_binned_model(ssp_data_wne, synthetic_tophat_obs)
    with pytest.raises(PrecomputeRefusal, match="age_binned"):
        model.with_approx(WavePrecomp())


def test_batch_resolver_records_a_refusal_under_its_config_slot_and_warns(
    ssp_data_wne, synthetic_tophat_obs
):
    from tengri import WavePrecomp
    from tengri.inference.fitter import _resolve_batch_fit_approx

    model = _age_binned_model(ssp_data_wne, synthetic_tophat_obs)
    with pytest.warns(UserWarning, match="age_binned"):
        out = _resolve_batch_fit_approx(model, WavePrecomp(), "photometry")
    assert out is model
    decline = model.band_table_decline("approx", "wave_precomp")
    assert decline is not None and "age_binned" in decline.reason


@pytest.mark.parametrize("bug", [ValueError("a shape bug"), RuntimeError("a bug")])
def test_batch_resolver_lets_a_plain_error_propagate_and_records_nothing(
    monkeypatch, bug, ssp_data_wne, synthetic_tophat_obs
):
    from tengri import WavePrecomp
    from tengri.inference import fitter

    def broken(model, cfg):
        raise bug

    monkeypatch.setattr(fitter, "_memoized_approx_clone", broken)
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    with pytest.raises(type(bug), match="bug"):
        fitter._resolve_batch_fit_approx(model, WavePrecomp(), "photometry")
    assert model.band_table_decline("approx", "wave_precomp") is None


def test_engagement_report_reads_the_batch_refusal_by_its_approx_key(
    ssp_data_wne, synthetic_tophat_obs
):
    from tengri import WavePrecomp
    from tengri.inference.fitter import _resolve_batch_fit_approx

    model = _age_binned_model(ssp_data_wne, synthetic_tophat_obs)
    with pytest.warns(UserWarning):
        _resolve_batch_fit_approx(model, WavePrecomp(), "photometry")
    state = engagement_reports(model)[0].state_of("approx.wave_precomp")
    assert state.state == "declined"
    assert "age_binned" in state.reason
