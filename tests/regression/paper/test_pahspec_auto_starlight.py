"""Tests for ``select_pahspec_starlight_auto``.

Nearest-neighbour selection of the best Draine+2021 PAHspec starlight
template from upstream stellar-population parameters
(SPS family, characteristic age, log Z / Z_sun).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_paper
import math

from tengri.components.dust.draine2021_pah import (
    STARLIGHT_PROPERTIES,
    select_pahspec_starlight_auto,
)


@pytest.mark.parametrize(
    ("family", "age_myr", "log_z", "expected"),
    [
        # Exact age + family + Z -> exact-name template.
        ("BC03", 3.0, 0.0, "BC03_Z0.02_3Myr"),
        ("BC03", 10.0, 0.0, "BC03_Z0.02_10Myr"),
        ("BC03", 100.0, 0.0, "BC03_Z0.02_100Myr"),
        ("BC03", 1000.0, 0.0, "BC03_Z0.02_1Gyr"),
        ("BPASS", 3.0, 0.0, "BPASS_Z0.02_3Myr"),
        ("BPASS", 10.0, 0.0, "BPASS_Z0.02_10Myr"),
        # Sub-solar metallicity prefers the matching low-Z template.
        ("BC03", 10.0, -1.7, "BC03_Z0.0004_10Myr"),
        ("BPASS", 10.0, -1.3, "BPASS_Z0.001_10Myr"),
        # Off-grid age within a family snaps to nearest log-age.
        # 6 Myr is geometrically closer to 10 (log d=0.22) than 3 (log d=0.30).
        ("BC03", 6.0, 0.0, "BC03_Z0.02_10Myr"),
        # 500 Myr is closer to 300 (log d=0.22) than 1000 (log d=0.30).
        ("BC03", 500.0, 0.0, "BC03_Z0.02_300Myr"),
    ],
)
def test_exact_and_near_matches(family, age_myr, log_z, expected):
    got = select_pahspec_starlight_auto(
        sps_family=family,
        age_myr=age_myr,
        log_z_solar=log_z,
    )
    assert got == expected


def test_old_population_prefers_m31bulge():
    """Ages well above 5 Gyr should prefer the M31 bulge template
    over the 1 Gyr SSP, since the SSP at 1 Gyr is much bluer than the
    actual integrated old population."""
    # 13 Gyr - clearly old
    got = select_pahspec_starlight_auto(
        sps_family=None,
        age_myr=13_000.0,
        log_z_solar=0.0,
    )
    assert got == "m31bulge"


def test_unknown_family_young_falls_back_to_mmmp():
    """FSPS/MIST/PrSc users with young populations should fall back to
    mMMP (diffuse ISRF), not m31bulge (which is for old populations)."""
    got = select_pahspec_starlight_auto(
        sps_family="FSPS",
        age_myr=10.0,
        log_z_solar=0.0,
    )
    assert got == "mMMP"


def test_unknown_family_old_falls_back_to_m31bulge():
    """FSPS users with very old populations should fall back to
    m31bulge instead of the bluer 1 Gyr SSP."""
    got = select_pahspec_starlight_auto(
        sps_family="FSPS",
        age_myr=10_000.0,
        log_z_solar=0.0,
    )
    assert got == "m31bulge"


def test_none_family_searches_both_ssp_libraries():
    """sps_family=None should consider both BC03 and BPASS SSPs and
    pick the absolute nearest in (age, log Z) space."""
    # 3 Myr at solar -> tied between BC03_Z0.02_3Myr and BPASS_Z0.02_3Myr
    # in (log_age, log_z) distance; either is acceptable.
    got = select_pahspec_starlight_auto(
        sps_family=None,
        age_myr=3.0,
        log_z_solar=0.0,
    )
    assert got in {"BC03_Z0.02_3Myr", "BPASS_Z0.02_3Myr"}


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        select_pahspec_starlight_auto(
            sps_family="BC03",
            age_myr=-1.0,
            log_z_solar=0.0,
        )
    with pytest.raises(ValueError):
        select_pahspec_starlight_auto(
            sps_family="BC03",
            age_myr=float("nan"),
            log_z_solar=0.0,
        )
    with pytest.raises(ValueError):
        select_pahspec_starlight_auto(
            sps_family="BC03",
            age_myr=10.0,
            log_z_solar=float("inf"),
        )


def test_starlight_properties_table_consistent():
    """All keys in STARLIGHT_PROPERTIES must have well-formed metadata."""
    for name, props in STARLIGHT_PROPERTIES.items():
        assert props["kind"] in ("ssp", "diffuse_isrf", "bulge"), name
        if props["kind"] == "ssp":
            assert props["sps_family"] in ("BC03", "BPASS"), name
            assert isinstance(props["age_myr"], float) and props["age_myr"] > 0, name
            assert isinstance(props["log_z_solar"], float), name
            assert math.isfinite(props["log_z_solar"]), name
        else:
            assert props["sps_family"] is None
            assert props["age_myr"] is None
            assert props["log_z_solar"] is None
