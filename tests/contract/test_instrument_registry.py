# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.Instrument premade registry."""

from __future__ import annotations

import dataclasses

import pytest

from tengri import Instrument, list_instruments
from tengri.observation import Photometry

pytestmark = pytest.mark.contract

PREMADE_NAMES = [
    "GALEX",
    "SDSS",
    "TWOMASS",
    "UKIDSS",
    "WISE",
    "SPITZER_IRAC",
    "HERSCHEL",
    "HST_ACS_WFC3",
    "JWST_NIRCam",
]


@pytest.mark.parametrize("name", PREMADE_NAMES)
def test_premade_constructs(name: str) -> None:
    inst = getattr(Instrument, name)()
    assert isinstance(inst, Instrument)
    assert isinstance(inst.photometry, Photometry)
    assert inst.photometry.n_filters >= 2
    assert len(inst.filter_names) == inst.photometry.n_filters
    assert inst.description  # non-empty


def test_observation_round_trip() -> None:
    inst = Instrument.SDSS()
    obs = inst.observation()
    assert obs.photometry is inst.photometry
    assert obs.spectroscopy is None


def test_list_instruments_matches_factories() -> None:
    listing = list_instruments()
    names = {d["name"] for d in listing}
    # 9 premades currently registered.
    assert len(listing) == 9
    assert "JWST_NIRCam" in names
    for entry in listing:
        assert entry["n_bands"] >= 2
        assert entry["description"]


def test_custom_instrument() -> None:
    photo = Photometry.from_names(["sdss_r", "sdss_i"])
    inst = Instrument(name="mini", photometry=photo, description="two-band")
    assert inst.filter_names == ("sdss_r", "sdss_i")
    assert inst.observation().photometry is photo


def test_jwst_nircam_8_wide_bands() -> None:
    inst = Instrument.JWST_NIRCam()
    assert inst.photometry.n_filters == 8
    # All NIRCam wide bands span F070W–F444W in order.
    expected_first_last = ("jwst_f070w", "jwst_f444w")
    assert (inst.filter_names[0], inst.filter_names[-1]) == expected_first_last


def test_frozen_dataclass() -> None:
    inst = Instrument.SDSS()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.name = "renamed"  # type: ignore[misc]
