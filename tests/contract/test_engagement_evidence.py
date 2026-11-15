# SPDX-License-Identifier: BSD-3-Clause
"""Engagement evidence (#3505 X1): every gate records its outcome, with a reason."""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.config.exceptions import PrecomputeRefusal
from tengri.forward.precompute.band_table import Decline
from tengri.forward.precompute_report import precompute_engagement_report


@pytest.mark.parametrize("bad", [None, "", "   ", 3])
def test_decline_requires_a_reason(bad):
    with pytest.raises((TypeError, ValueError)):
        Decline(bad)


def test_decline_keeps_its_reason():
    assert Decline("why").reason == "why"


class _Model:
    """Stands in for a model: only an attribute bag."""


def test_feature_top_up_records_a_refusal_and_returns_none(monkeypatch):
    from tengri.inference import fitter

    def refuse(model, cfg):
        raise PrecomputeRefusal("no lines to tabulate")

    monkeypatch.setattr(fitter, "_memoized_approx_clone", refuse)
    model = _Model()
    assert fitter._try_feature_top_up(model, ("cfg",)) is None
    assert model._approx_declines == {"feature_precomp": "no lines to tabulate"}


@pytest.mark.parametrize("bug", [RuntimeError("a bug"), ValueError("a shape bug")])
def test_feature_top_up_lets_a_bug_propagate_even_when_it_is_a_value_error(monkeypatch, bug):
    from tengri.inference import fitter

    def broken(model, cfg):
        raise bug

    monkeypatch.setattr(fitter, "_memoized_approx_clone", broken)
    model = _Model()
    with pytest.raises(type(bug), match="bug"):
        fitter._try_feature_top_up(model, ("cfg",))
    assert not hasattr(model, "_approx_declines")


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


def test_report_carries_the_recorded_approx_declines():
    from tengri.forward.precompute_report import _gather_observed_facts

    model = _Model()
    model._approx_declines = {"feature_precomp": "why"}
    assert _gather_observed_facts(model).get("approx_declines") == {"feature_precomp": "why"}


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
    recorded = model._energy_balance_lut_decline
    assert recorded and "WavePrecomp" in recorded
    report = precompute_engagement_report(model)
    assert report.energy_balance_lut.state == "declined"
    assert report.energy_balance_lut.reason == recorded
