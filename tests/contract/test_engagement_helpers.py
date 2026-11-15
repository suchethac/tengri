# SPDX-License-Identifier: BSD-3-Clause
"""Engagement helpers (#3505 X1): assertions read the records a gate left, and nothing else."""

from __future__ import annotations

import re
from unittest.mock import Mock

import pytest

from tengri.forward.precompute_report import (
    PrecomputeEngagementReport,
    PrecomputeState,
    engagement_reports,
)
from tengri.forward.sed_model import WavePrecomp
from tengri.inference.catalog_fitter import CatalogPosterior
from tengri.inference.fitter import Fitter
from tengri.inference.posterior import Posterior
from tests.contract._engagement import assert_declined, assert_engaged
from tests.contract.test_engagement_evidence import _eb_model

pytestmark = pytest.mark.contract


def _post(model) -> Mock:
    """A Posterior stand-in carrying a real model; used only for the dispatch."""
    post = Mock(spec=Posterior)
    post._model = model
    return post


def _catalog(*models) -> Mock:
    """A CatalogPosterior stand-in whose galaxies carry the given real models."""
    catalog = Mock(spec=CatalogPosterior)
    catalog.posteriors = [_post(model) for model in models]
    return catalog


def _lut_built(ssp, obs, approx):
    """A model whose energy-balance LUT gate has run at its build chain."""
    model = _eb_model(ssp, obs, approx=approx)
    model._energy_balance_lut(model._build_component_chain())
    return model


# ── the report: state_of reads what the report holds ─────────────────────────


def test_state_of_reads_the_energy_balance_lut_state(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    report = engagement_reports(model)[0]
    assert report.state_of("energy_balance_lut").state == "declined"
    assert report.state_of("energy_balance_lut").reason == report.energy_balance_lut.reason


def test_state_of_unknown_key_lists_the_known_keys(ssp_data_wne, synthetic_tophat_obs):
    report = engagement_reports(_eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None))[0]
    with pytest.raises(KeyError, match="energy_balance_lut"):
        report.state_of("no_such_mechanism")


def test_state_of_feature_precomp_declined_reads_the_recorded_reason(
    ssp_data_wne, synthetic_tophat_obs
):
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    model.record_decline("approx", "feature_precomp", "no emission lines to tabulate")
    state = engagement_reports(model)[0].state_of("feature_precomp")
    assert state == PrecomputeState(state="declined", reason="no emission lines to tabulate")


def test_state_of_feature_precomp_without_a_top_up_is_never_attempted(
    ssp_data_wne, synthetic_tophat_obs
):
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=WavePrecomp())
    assert engagement_reports(model)[0].state_of("feature_precomp").state == "never_attempted"


def test_state_of_feature_precomp_is_engaged_when_the_top_up_is_live(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    from tengri import DEFAULT, Fixed, SEDModel, Uniform
    from tengri.forward.sed_model import FeaturePrecomp

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-4.0, -1.0)},
        redshift=Fixed(0.1),
        approx=(WavePrecomp(), FeaturePrecomp()),
    )
    assert model.approx.feature_precomp
    assert engagement_reports(model)[0].state_of("feature_precomp") == PrecomputeState("engaged")
    assert_engaged(model, "feature_precomp")


# ── assertions: engaged ──────────────────────────────────────────────────────


def test_assert_engaged_passes_on_an_engaged_lut(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, WavePrecomp())
    assert_engaged(model, "energy_balance_lut", regime="single")


def test_assert_engaged_fails_on_a_declined_mechanism_and_shows_the_reason(
    ssp_data_wne, synthetic_tophat_obs
):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    with pytest.raises(AssertionError, match="WavePrecomp"):
        assert_engaged(model, "energy_balance_lut")


# ── assertions: declined ─────────────────────────────────────────────────────


def test_assert_declined_passes_on_a_matching_reason(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    assert_declined(model, "energy_balance_lut", "WavePrecomp", regime="single")


def test_assert_declined_accepts_a_compiled_pattern(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    assert_declined(model, "energy_balance_lut", re.compile(r"Wave\w+"))


def test_assert_declined_with_a_non_matching_reason_fails_and_shows_the_actual_reason(
    ssp_data_wne, synthetic_tophat_obs
):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    with pytest.raises(AssertionError, match="WavePrecomp"):
        assert_declined(model, "energy_balance_lut", "no such reason")


def test_assert_declined_fails_on_an_engaged_mechanism(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, WavePrecomp())
    with pytest.raises(AssertionError, match="engaged, not declined"):
        assert_declined(model, "energy_balance_lut", "anything")


def test_assert_declined_on_a_feature_decline(ssp_data_wne, synthetic_tophat_obs):
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=None)
    model.record_decline("approx", "feature_precomp", "no emission lines to tabulate")
    assert_declined(model, "feature_precomp", "no emission lines", regime="single")


def test_assert_declined_without_a_reason_is_a_type_error(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    with pytest.raises(TypeError):
        assert_declined(model, "energy_balance_lut")


@pytest.mark.parametrize("bad", [None, "", "   ", 3])
def test_assert_declined_refuses_a_blank_or_non_text_reason(
    ssp_data_wne, synthetic_tophat_obs, bad
):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    with pytest.raises(TypeError):
        assert_declined(model, "energy_balance_lut", bad)


def test_unknown_what_fails_and_lists_the_known_keys(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, WavePrecomp())
    with pytest.raises(AssertionError, match="energy_balance_lut"):
        assert_engaged(model, "no_such_mechanism")


# ── catalog targets: every galaxy must hold ──────────────────────────────────


def test_catalog_failure_names_the_galaxy_index_and_the_recorded_reason(
    ssp_data_wne, synthetic_tophat_obs
):
    engaged = _lut_built(ssp_data_wne, synthetic_tophat_obs, WavePrecomp())
    declined = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    catalog = _catalog(engaged, declined, engaged)
    with pytest.raises(AssertionError, match=r"galaxies \[1\] of 3") as info:
        assert_engaged(catalog, "energy_balance_lut", regime="catalog")
    assert "WavePrecomp" in str(info.value)


def test_catalog_where_every_galaxy_passes_passes(ssp_data_wne, synthetic_tophat_obs):
    engaged = _lut_built(ssp_data_wne, synthetic_tophat_obs, WavePrecomp())
    assert_engaged(_catalog(engaged, engaged), "energy_balance_lut", regime="catalog")


# ── dispatch: which models a target evaluates, and the regime check ──────────


def test_single_model_target_is_its_own_report(ssp_data_wne, synthetic_tophat_obs):
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=WavePrecomp())
    reports = engagement_reports(model, regime="single")
    assert len(reports) == 1
    assert isinstance(reports[0], PrecomputeEngagementReport)


def test_fitter_target_reads_its_model(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    fitter = Mock(spec=Fitter)
    fitter.model = model
    assert_declined(fitter, "energy_balance_lut", "WavePrecomp", regime="single")


def test_posterior_target_reads_its_model_reference(ssp_data_wne, synthetic_tophat_obs):
    model = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    post = Mock(spec=Posterior)
    post._model = model
    assert_declined(post, "energy_balance_lut", "WavePrecomp", regime="single")


def test_posterior_without_a_model_reference_is_refused(ssp_data_wne, synthetic_tophat_obs):
    post = Mock(spec=Posterior)
    post._model = None
    with pytest.raises(ValueError, match="no model reference"):
        engagement_reports(post)


def test_catalog_target_reads_every_galaxy_in_order(ssp_data_wne, synthetic_tophat_obs):
    first = _lut_built(ssp_data_wne, synthetic_tophat_obs, WavePrecomp())
    second = _lut_built(ssp_data_wne, synthetic_tophat_obs, None)
    reports = engagement_reports(_catalog(first, second), regime="catalog")
    assert len(reports) == 2
    assert reports[0].energy_balance_lut.state == "engaged"
    assert reports[1].energy_balance_lut.state == "declined"


def test_regime_mismatch_for_a_single_target_is_a_value_error(ssp_data_wne, synthetic_tophat_obs):
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=WavePrecomp())
    with pytest.raises(ValueError, match="regime"):
        engagement_reports(model, regime="catalog")


def test_regime_mismatch_for_a_catalog_target_is_a_value_error(ssp_data_wne, synthetic_tophat_obs):
    catalog = _catalog(_eb_model(ssp_data_wne, synthetic_tophat_obs, approx=WavePrecomp()))
    with pytest.raises(ValueError, match="regime"):
        assert_engaged(catalog, "energy_balance_lut", regime="single")


def test_unknown_regime_value_is_refused(ssp_data_wne, synthetic_tophat_obs):
    model = _eb_model(ssp_data_wne, synthetic_tophat_obs, approx=WavePrecomp())
    with pytest.raises(ValueError, match="regime must be"):
        engagement_reports(model, regime="batch")


def test_unknown_target_type_is_a_type_error():
    with pytest.raises(TypeError, match="object"):
        engagement_reports(object())
