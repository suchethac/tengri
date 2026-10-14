#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A failed prerequisite of the WavePrecomp precomputes is raised, not reported.

``SEDModel.build(approx=WavePrecomp(...))`` runs its precomputes at
construction. All of them need the component chain, which is built first. A
chain failure is a defect in the build, not an optional absence, so it raises
at build time with its own message. Before #2973 each precompute was wrapped in
a broad ``try`` that warned and fell back to the exact per-call path, and the
chain failure was announced as several unrelated failures (an
``AttributeError`` on a cache the failed build never set). Falling back hid the
defect in the first place, so it is no longer offered.

The control below checks that a healthy build raises and warns about nothing.
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


def test_a_failed_chain_build_raises_its_own_error(ssp_data_wne, observation, monkeypatch):
    """The failure surfaces once, at build, with the message it was raised with."""
    monkeypatch.setattr(
        SEDModel,
        "_build_component_chain",
        lambda self: (_ for _ in ()).throw(RuntimeError(CHAIN_FAILURE)),
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match=CHAIN_FAILURE):
            _build(ssp_data_wne, observation, WavePrecomp())

    assert not _wave_precomp_warnings(recorded), (
        "a failed prerequisite must raise, not fall back with a warning"
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
