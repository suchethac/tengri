# SPDX-License-Identifier: BSD-3-Clause
"""Engagement evidence (#3505 X1): every gate records its outcome, with a reason."""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.forward.precompute.band_table import Decline
from tengri.forward.precompute_report import precompute_engagement_report


@pytest.mark.parametrize("bad", [None, "", "   ", 3])
def test_decline_requires_a_reason(bad):
    with pytest.raises((TypeError, ValueError)):
        Decline(bad)


def test_decline_keeps_its_reason():
    assert Decline("why").reason == "why"


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
