# SPDX-License-Identifier: BSD-3-Clause
"""Regression: met_mode reachable via nested-dict builder ``stellar={}``.

Issue #311: PR #287 wired five new chemical-evolution modes but they were only
reachable via legacy ``Parameters(met_mode=...)``, contradicting the design
goal that ``SEDModel.build`` is the recommended public surface. The fix adds
a top-level ``stellar={}`` block carrying ``met_mode`` plus per-mode params.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_bug

import tengri
from tengri import FIXED, Fixed


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError:
        pytest.skip("default wNE SSP not available")


def test_two_step(ssp):
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": FIXED},
        stellar={
            "met_mode": "two_step",
            "*": FIXED,
            "logzsol_old": Fixed(-1.0),
            "logzsol_young": Fixed(0.0),
            "step_age_gyr": Fixed(8.0),
        },
        dust={"type": "two_component", "*": FIXED},
    )
    assert m.spec.met_mode == "two_step"
    fixed = m.spec.get_fixed_values()
    assert fixed["met_logzsol_old"] == pytest.approx(-1.0)
    assert fixed["met_logzsol_young"] == pytest.approx(0.0)
    assert fixed["met_step_age_gyr"] == pytest.approx(8.0)


def test_ramp(ssp):
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": FIXED},
        stellar={
            "met_mode": "ramp",
            "*": FIXED,
            "logzsol_0": Fixed(-1.5),
            "logzsol_final": Fixed(0.0),
        },
        dust={"type": "two_component", "*": FIXED},
    )
    assert m.spec.met_mode == "ramp"


def test_unknown_met_mode_raises(ssp):
    with pytest.raises(ValueError, match="Unknown met_mode"):
        tengri.SEDModel.build(
            ssp,
            sfh={"type": "dpl", "*": FIXED},
            stellar={"met_mode": "two_steps"},  # typo
            dust={"type": "two_component", "*": FIXED},
        )


def test_default_no_stellar_block(ssp):
    """Omitting stellar={} preserves the default met_mode='delta'."""
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "*": FIXED},
    )
    assert m.spec.met_mode == "delta"


def test_roundtrip_emits_stellar_block(ssp):
    """to_groups() should emit a stellar block when met_mode is non-default."""
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": FIXED},
        stellar={
            "met_mode": "two_step",
            "*": FIXED,
            "logzsol_old": Fixed(-1.0),
            "logzsol_young": Fixed(0.0),
            "step_age_gyr": Fixed(8.0),
        },
        dust={"type": "two_component", "*": FIXED},
    )
    groups = m.spec.to_groups()
    assert "stellar" in groups
    assert groups["stellar"]["met_mode"] == "two_step"
