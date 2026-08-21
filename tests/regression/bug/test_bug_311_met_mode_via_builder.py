# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the metallicity mode is reachable from the nested-dict builder.

Issue #311: PR #287 wired five chemical-evolution modes that were reachable only
through the legacy ``Parameters(met_mode=...)``, contradicting the design goal
that ``SEDModel.build`` is the recommended public surface. #311 added a
top-level block for them.

That block was ``stellar={'met_mode': ...}``, and #1720 replaced it with
``met={'type': ...}`` — the parallel of ``sfh={'type': ...}``, selecting with
``type`` like every other group. The property #311 established is unchanged and
still asserted here; only the spelling moved.
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
        met={
            "type": "two_step",
            "*": FIXED,
            "logzsol_old": Fixed(-1.0),
            "logzsol_young": Fixed(0.0),
            "step_age_gyr": Fixed(8.0),
        },
        dust_attenuation={"law": "power_law", "type": "two_component", "*": FIXED},
        redshift=Fixed(0.1),
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
        met={
            "type": "ramp",
            "*": FIXED,
            "logzsol_0": Fixed(-1.5),
            "logzsol_final": Fixed(0.0),
        },
        dust_attenuation={"law": "power_law", "type": "two_component", "*": FIXED},
        redshift=Fixed(0.1),
    )
    assert m.spec.met_mode == "ramp"


def test_unknown_met_mode_raises(ssp):
    with pytest.raises(ValueError, match="Unknown metallicity mode"):
        tengri.SEDModel.build(
            ssp,
            sfh={"type": "dpl", "*": FIXED},
            met={"type": "two_steps"},  # typo
            dust_attenuation={"type": "two_component", "*": FIXED},
            redshift=Fixed(0.1),
        )


def test_default_no_met_block(ssp):
    """Omitting met={} preserves the default met_mode='delta'."""
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={"law": "power_law", "type": "two_component", "*": FIXED},
        redshift=Fixed(0.1),
    )
    assert m.spec.met_mode == "delta"


def test_roundtrip_emits_met_block(ssp):
    """to_groups() should emit a met block when the mode is non-default."""
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": FIXED},
        met={
            "type": "two_step",
            "*": FIXED,
            "logzsol_old": Fixed(-1.0),
            "logzsol_young": Fixed(0.0),
            "step_age_gyr": Fixed(8.0),
        },
        dust_attenuation={"law": "power_law", "type": "two_component", "*": FIXED},
        redshift=Fixed(0.1),
    )
    groups = m.spec.to_groups()
    assert "met" in groups
    assert groups["met"]["type"] == "two_step"
