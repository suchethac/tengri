#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""One failed prerequisite must be reported once, under its own name.

``SEDModel.build(approx=WavePrecomp(...))`` runs four precomputes at
construction: the dust energy-balance LUT, the dust-emission band response, and
the X-ray and radio term responses. Each is wrapped in its own ``try`` so that
one failing does not disable the others or get reported under another's name --
that separation was a deliberate fix.

All four need the same prerequisite, the component chain, and it was built
inside the *first* ``try``. So when the chain build itself failed, three things
happened at once: the failure was announced as an energy-balance failure, the
three later blocks dereferenced a cache attribute the failed build had never
set, and each reported its own ``AttributeError``. One root cause, four
warnings, three of them naming subsystems that had not run and reporting a
missing attribute rather than the reason it was missing.

Falling back was always correct -- the exact per-call path gives the same
numbers -- so nothing was ever wrong in the output. What was wrong was the
report, which sent a reader chasing an attribute name.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

CHAIN_FAILURE = "deliberate chain failure for this test"


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))


def _build(ssp, observation, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "met_logzsol": Fixed(0.0),
        },
        redshift=Fixed(0.5),
        approx=approx,
    )


def _wave_precomp_warnings(recorded):
    return [str(w.message) for w in recorded if "WavePrecomp" in str(w.message)]


def test_a_failed_chain_build_is_reported_once(ssp_data_wne, observation, monkeypatch):
    """The defect. Four precomputes share one prerequisite; it fails once."""
    monkeypatch.setattr(
        SEDModel,
        "_build_component_chain",
        lambda self: (_ for _ in ()).throw(RuntimeError(CHAIN_FAILURE)),
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _build(ssp_data_wne, observation, WavePrecomp())

    messages = _wave_precomp_warnings(recorded)
    assert len(messages) == 1, (
        f"one failed prerequisite produced {len(messages)} WavePrecomp warnings:\n"
        + "\n".join(f"  - {m}" for m in messages)
    )
    assert CHAIN_FAILURE in messages[0], (
        f"the warning does not name the failure that caused it: {messages[0]}"
    )


def test_a_failed_chain_build_is_not_reported_as_a_missing_attribute(
    ssp_data_wne, observation, monkeypatch
):
    """The symptom that sent the reader to the wrong place.

    ``AttributeError: 'SEDModel' object has no attribute
    '_cached_component_chain'`` describes the consequence of the failure, not
    the failure, and names an internal attribute the astronomer has no reason
    to have heard of.
    """
    monkeypatch.setattr(
        SEDModel,
        "_build_component_chain",
        lambda self: (_ for _ in ()).throw(RuntimeError(CHAIN_FAILURE)),
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _build(ssp_data_wne, observation, WavePrecomp())

    joined = "\n".join(_wave_precomp_warnings(recorded))
    assert "AttributeError" not in joined, f"a consequence is reported as a cause:\n{joined}"
    assert "_cached_component_chain" not in joined


def test_a_failed_chain_build_leaves_every_lut_cache_disarmed(
    ssp_data_wne, observation, monkeypatch
):
    """Correctness, not just wording.

    Reporting once is only honest if the dependent precomputes really did not
    run. Each cache must read as absent, so the forward pass takes the exact
    per-call path rather than a half-filled table.
    """
    monkeypatch.setattr(
        SEDModel,
        "_build_component_chain",
        lambda self: (_ for _ in ()).throw(RuntimeError(CHAIN_FAILURE)),
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model = _build(ssp_data_wne, observation, WavePrecomp())

    for attribute in (
        "_energy_balance_lut_cache",
        "_dust_band_response_cache",
        "_xray_term_response_cache",
        "_radio_term_response_cache",
    ):
        assert getattr(model, attribute, None) is None, (
            f"{attribute} survived a failed chain build, so a dependent "
            "precompute ran on a prerequisite that does not exist"
        )


def test_a_healthy_build_warns_about_none_of_this(ssp_data_wne, observation):
    """The control.

    Without this the tests above would pass on a build that warns constantly.
    """
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _build(ssp_data_wne, observation, WavePrecomp())

    unavailable = [m for m in _wave_precomp_warnings(recorded) if "unavailable" in m]
    assert not unavailable, f"a healthy build reported a precompute failure: {unavailable}"
