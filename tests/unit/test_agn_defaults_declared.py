# SPDX-License-Identifier: BSD-3-Clause
"""AGN block defaults are read off their parameter declarations (ADR-0011).

A literal copy of a declared default drifts silently: the declaration moves and
the block keeps the old number. These tests pin each block-level default to the
declaration it came from, and pin the value the literal used to carry so the
refactor cannot change a result.
"""

from __future__ import annotations

import math

import pytest

from tengri.components.agn import _params
from tengri.components.agn.blocks import runner, torus_screen
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.unit


def test_runner_defaults_come_from_declarations():
    assert declared_default(_params.PARAMS, "agn_theta_torus") == runner.DEFAULT_AGN_THETA_TORUS
    assert declared_default(_params.PARAMS, "agn_oa_skirtor") == runner.DEFAULT_AGN_OA_SKIRTOR
    assert declared_default(_params.PARAMS, "agn_cos_inc") == runner.DEFAULT_AGN_COS_INC
    assert declared_default(_params.PARAMS, "agn_tau_skirtor") == runner.DEFAULT_AGN_TAU_SKIRTOR


def test_runner_defaults_keep_their_values():
    assert runner.DEFAULT_AGN_THETA_TORUS == 30.0
    assert runner.DEFAULT_AGN_OA_SKIRTOR == 40.0
    assert runner.DEFAULT_AGN_TAU_SKIRTOR == 7.0
    assert runner.DEFAULT_AGN_COS_INC == 0.86602540378443864
    assert math.isclose(runner.DEFAULT_AGN_COS_INC, math.cos(math.radians(30.0)), rel_tol=1e-15)


@pytest.mark.parametrize(
    ("torus_block", "expected"),
    [
        ("skirtor", 40.0),  # agn_oa_skirtor
        ("nenkova_agnfitter_2p", 40.0),  # agn_oa_nenkova
        ("unlisted_block", 30.0),  # agn_theta_torus
    ],
)
def test_torus_screen_fallback_angles_match_declarations(torus_block, expected):
    assert torus_screen.polar_follow_opening_angle(torus_block, {}) == expected


def test_torus_screen_fallback_angles_come_from_declarations():
    key_default = {
        "agn_oa_skirtor": declared_default(_params.PARAMS, "agn_oa_skirtor"),
        "agn_oa_nenkova": declared_default(_params.PARAMS, "agn_oa_nenkova"),
    }
    assert key_default == torus_screen._OPENING_ANGLE_DEFAULT
    assert (
        declared_default(_params.PARAMS, "agn_theta_torus") == torus_screen.DEFAULT_AGN_THETA_TORUS
    )
