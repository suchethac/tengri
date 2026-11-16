# SPDX-License-Identifier: BSD-3-Clause
"""IGM fold engagement evidence (#3505 X1): the ``"auto"`` resolution records its declines."""

from __future__ import annotations

import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

BANDS = ["galex_nuv", "sdss_r"]
PROBE_Z = 1.0


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _igm_model(ssp, observation, igm):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "met_logzsol": Fixed(0.0),
        },
        redshift=Fixed(PROBE_Z),
        igm={"all_params": Fixed(DEFAULT), **igm},
        approx=WavePrecomp(igm_fold="auto"),
    )


def _stellar_fold(model):
    chain = model._build_component_chain()
    return chain[0]._state.igm_fold_resolved


def test_auto_engages_the_exact_fold_and_records_no_decline(ssp_data_wne, observation):
    """A mean IGM at a fixed redshift: the exact fold is built, so nothing is declined."""
    model = _igm_model(ssp_data_wne, observation, {"type": "inoue"})
    assert _stellar_fold(model) == "exact"
    assert model.band_table_decline("igm", "fold") is None


def test_auto_declines_the_exact_fold_with_a_recorded_reason(ssp_data_wne, observation):
    """Patchy reionization has no build-time table: auto resolves to node and says why."""
    model = _igm_model(ssp_data_wne, observation, {"type": "inoue", "patchy": True})
    assert _stellar_fold(model) == "node"
    decline = model.band_table_decline("igm", "fold")
    assert decline is not None and "patchy" in decline.reason
