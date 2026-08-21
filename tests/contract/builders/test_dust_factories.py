# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.dust — attenuation, and IR emission as a peer group."""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract

from tengri import FIXED, FREE, Fixed, Uniform, builders, parse_groups
from tengri.components.dust.attenuation import DUST_LAWS
from tengri.parameters.groups import _valid_dust_emission_types

# ── Module surface ────────────────────────────────────────────────


def test_top_level_variants() -> None:
    assert builders.dust.available() == ["single_component", "two_component"]


def test_emission_variants_match_parser_enum() -> None:
    # Both sides derive from the registry-backed _valid_dust_emission_types()
    # helper; this test pins the round-trip.
    assert set(builders.dust.emission.available()) == set(_valid_dust_emission_types())


# ── Signatures ────────────────────────────────────────────────────


def test_two_component_signature_carries_settings_and_params() -> None:
    sig = inspect.signature(builders.dust.two_component)
    params = list(sig.parameters)
    assert "all_params" in params
    assert "law_bc" in params
    assert "law_diff" in params
    assert "tau_bc" in params  # short-form param
    assert "tau_diff" in params
    # No tau_v for two_component
    assert "tau_v" not in params
    # `emission` is a PEER group now, not a sub-block, so it is deliberately not
    # a parameter of the attenuation factory. Its own factories live under
    # builders.dust.emission and feed a separate `dust_emission=` keyword.
    assert "emission" not in params


def test_single_component_signature_uses_tau_v() -> None:
    sig = inspect.signature(builders.dust.single_component)
    params = list(sig.parameters)
    assert "tau_v" in params
    assert "tau_bc" not in params  # only in two_component
    assert "tau_diff" not in params
    assert "law_diff" not in params  # only one law for single


def test_emission_dale2014_signature_lists_canonical_short_params() -> None:
    sig = inspect.signature(builders.dust.emission.dale2014)
    params = set(sig.parameters)
    # Headline Dale2014 knob
    assert "alpha_dale" in params


# ── Default-call shape ────────────────────────────────────────────


def test_two_component_requires_a_law() -> None:
    """No default law: attenuation must be named explicitly."""
    with pytest.raises((TypeError, ValueError), match="law"):
        builders.dust.two_component()


def test_two_component_shared_law_carries_law_string() -> None:
    out = builders.dust.two_component(law="calzetti")
    assert out == {
        "type": "two_component",
        "all_params": FIXED,
        "law": "calzetti",
    }


def test_two_component_pair_law_carries_both_strings() -> None:
    out = builders.dust.two_component(law_bc="calzetti", law_diff="power_law")
    assert out == {
        "type": "two_component",
        "all_params": FIXED,
        "law_bc": "calzetti",
        "law_diff": "power_law",
    }


def test_single_component_requires_a_law() -> None:
    with pytest.raises((TypeError, ValueError), match="law"):
        builders.dust.single_component()


def test_single_component_carries_one_law() -> None:
    out = builders.dust.single_component(law="calzetti")
    assert out == {
        "type": "single_component",
        "all_params": FIXED,
        "law": "calzetti",
    }


def test_emission_default_does_not_include_settings() -> None:
    assert builders.dust.emission.dale2014() == {"type": "dale2014", "all_params": FIXED}


# ── Settings validation ───────────────────────────────────────────


def test_unknown_law_raises_with_valid_list() -> None:
    with pytest.raises(ValueError, match="unknown attenuation law") as exc:
        builders.dust.two_component(law="callzeti")  # typo
    msg = str(exc.value)
    assert "calzetti" in msg


def test_every_dust_law_key_is_accepted() -> None:
    """Sanity check: any key in DUST_LAWS is valid."""
    for law in list(DUST_LAWS)[:3]:  # first three suffice
        out = builders.dust.two_component(law=law)
        assert out["law"] == law


# ── Composition: emission is a peer group, not a nested block ──────
#
# The behavior lives in tests/contract/test_dust_split_builders.py, which owns
# the refusal (test_attenuation_builder_raises_on_emission_kwarg), the emission
# dict, and the dust_emission= build path. This file's own contribution is the
# signature assertion above: `emission` is not a parameter of two_component.


# ── Round-trip through parser ─────────────────────────────────────


def test_two_component_free_round_trips_tau_bc_tau_diff() -> None:
    """``*=FREE`` on dust DOES work — verify the parser's wildcard
    pathway for dust (unlike radio/xray) flips the attenuation knobs."""
    spec = parse_groups(
        sfh={"type": "dpl"},
        dust_attenuation=builders.dust.two_component(law="calzetti", _=FREE),
        redshift=Fixed(0.1),
    )
    free_dust = {p for p in spec.free_params if p.startswith("dust_")}
    assert "dust_tau_bc" in free_dust
    assert "dust_tau_diff" in free_dust


def test_single_component_uses_tau_v_not_tau_bc() -> None:
    spec = parse_groups(
        sfh={"type": "dpl"},
        dust_attenuation=builders.dust.single_component(law="calzetti", _=FREE),
        redshift=Fixed(0.1),
    )
    free_dust = {p for p in spec.free_params if p.startswith("dust_")}
    assert "dust_tau_v" in free_dust
    assert "dust_tau_bc" not in free_dust


def test_per_param_override_survives_round_trip() -> None:
    spec = parse_groups(
        sfh={"type": "dpl"},
        dust_attenuation=builders.dust.two_component(law="calzetti", tau_bc=Uniform(0.5, 3.0)),
        redshift=Fixed(0.1),
    )
    assert "dust_tau_bc" in spec.free_params


# ── Validation ────────────────────────────────────────────────────


def test_unknown_kwarg_raises_with_valid_list() -> None:
    with pytest.raises(TypeError, match="two_component") as exc:
        builders.dust.two_component(law="calzetti", tau_bcc=Uniform(0, 2))  # typo
    msg = str(exc.value)
    assert "tau_bcc" in msg
    assert "tau_bc" in msg


def test_invalid_wildcard_rejected() -> None:
    with pytest.raises(ValueError, match="FREE or FIXED"):
        builders.dust.two_component(law="calzetti", _="free")
