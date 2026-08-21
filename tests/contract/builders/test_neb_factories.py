# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.neb — nebular emission backend factories."""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract

from tengri import FIXED, Fixed, Uniform, builders, parse_groups
from tengri.parameters.groups import _valid_nebular_types


def test_every_nebular_type_has_a_factory() -> None:
    assert set(builders.neb.available()) == set(_valid_nebular_types())


def test_none_and_ssp_expose_only_wildcard() -> None:
    """Backends that contribute no free params expose just ``wildcard``."""
    for variant in ("none", "ssp"):
        sig = inspect.signature(getattr(builders.neb, variant))
        assert list(sig.parameters) == ["all_params"], variant


def test_cue_signature_lists_canonical_short_params() -> None:
    """Cue's three headline params (logU, logZ_gas, fesc) must surface."""
    sig = inspect.signature(builders.neb.cue)
    params = set(sig.parameters)
    assert {"logU", "logZ_gas", "fesc"}.issubset(params)


def test_cb19_signature_is_a_superset_of_cue() -> None:
    """CB19 adds extra knobs (co, dno, hbfrac, log_nH) on top of Cue's set."""
    cue_params = set(inspect.signature(builders.neb.cue).parameters)
    cb19_params = set(inspect.signature(builders.neb.cb19).parameters)
    assert cue_params.issubset(cb19_params), (
        f"CB19 should be a superset of Cue. Cue-only: {cue_params - cb19_params}"
    )
    extras = cb19_params - cue_params
    assert extras, "CB19 should add at least one extra parameter"


def test_cloudy_borrows_cue_signature() -> None:
    """Cloudy shares ``_NEBULAR_PARAMS`` with Cue; signature must match."""
    cue_params = set(inspect.signature(builders.neb.cue).parameters)
    cloudy_params = set(inspect.signature(builders.neb.cloudy).parameters)
    assert cue_params == cloudy_params


def test_default_call_returns_canonical_shape() -> None:
    assert builders.neb.cue() == {"type": "cue", "all_params": FIXED}
    assert builders.neb.none() == {"type": "none", "all_params": FIXED}


def test_per_param_override_round_trips_to_free() -> None:
    """An explicit Distribution on a nebular param must surface as FREE."""
    spec = parse_groups(
        sfh={"type": "dpl"},
        neb=builders.neb.cue(fesc=Uniform(0.0, 0.5)),
        redshift=Fixed(0.1),
    )
    assert "neb_fesc" in spec.free_params


def test_typo_rejection_lists_valid_names() -> None:
    with pytest.raises(TypeError, match="cue") as exc:
        builders.neb.cue(fes=Uniform(0.0, 0.5))  # typo for 'fesc'
    msg = str(exc.value)
    assert "fes" in msg
    assert "fesc" in msg


def test_ssp_neb_round_trips_without_activating_params() -> None:
    spec = parse_groups(sfh={"type": "dpl"}, neb=builders.neb.ssp(), redshift=Fixed(0.1))
    neb_free = [p for p in spec.free_params if p.startswith("neb_")]
    assert neb_free == []
