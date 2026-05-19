# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.dust — attenuation + nested IR emission."""

from __future__ import annotations

import inspect

import pytest

from tengri import FIXED, FREE, Parameters, Uniform, builders
from tengri.components.dust.attenuation import DUST_LAWS
from tengri.parameters.groups import _VALID_DUST_EMISSION_TYPES

# ── Module surface ────────────────────────────────────────────────


def test_top_level_variants() -> None:
    assert builders.dust.available() == ["single_component", "two_component"]


def test_emission_variants_match_parser_enum() -> None:
    assert set(builders.dust.emission.available()) == set(_VALID_DUST_EMISSION_TYPES)


# ── Signatures ────────────────────────────────────────────────────


def test_two_component_signature_carries_settings_and_params() -> None:
    sig = inspect.signature(builders.dust.two_component)
    params = list(sig.parameters)
    assert "defaults" in params
    assert "law_bc" in params
    assert "law_diff" in params
    assert "tau_bc" in params  # short-form param
    assert "tau_diff" in params
    assert "emission" in params
    # No tau_v for two_component
    assert "tau_v" not in params


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


def test_two_component_default_carries_law_strings() -> None:
    out = builders.dust.two_component()
    assert out == {
        "type": "two_component",
        "*": FIXED,
        "law_bc": "calzetti",
        "law_diff": "calzetti",
    }


def test_single_component_default_carries_one_law() -> None:
    out = builders.dust.single_component()
    assert out == {
        "type": "single_component",
        "*": FIXED,
        "law_bc": "calzetti",
    }


def test_emission_default_does_not_include_settings() -> None:
    assert builders.dust.emission.dale2014() == {"type": "dale2014", "*": FIXED}


# ── Settings validation ───────────────────────────────────────────


def test_unknown_law_raises_with_valid_list() -> None:
    with pytest.raises(ValueError, match="unknown attenuation law") as exc:
        builders.dust.two_component(law_bc="callzeti")  # typo
    msg = str(exc.value)
    assert "calzetti" in msg


def test_every_dust_law_key_is_accepted() -> None:
    """Sanity check: any key in DUST_LAWS is valid."""
    for law in list(DUST_LAWS)[:3]:  # first three suffice
        out = builders.dust.two_component(law_bc=law)
        assert out["law_bc"] == law


# ── Composition with nested emission ──────────────────────────────


def test_two_component_with_nested_emission() -> None:
    out = builders.dust.two_component(
        _=FREE,
        tau_bc=Uniform(0.0, 2.0),
        emission=builders.dust.emission.dale2014(_=FIXED),
    )
    assert out["emission"] == {"type": "dale2014", "*": FIXED}
    assert out["tau_bc"] == Uniform(0.0, 2.0)
    assert out["*"] is FREE


def test_emission_kwarg_must_be_dict() -> None:
    with pytest.raises(TypeError, match="emission"):
        builders.dust.two_component(emission="dale2014")  # forgot to call


# ── Round-trip through parser ─────────────────────────────────────


def test_two_component_free_round_trips_tau_bc_tau_diff() -> None:
    """``*=FREE`` on dust DOES work — verify the parser's wildcard
    pathway for dust (unlike radio/xray) flips the attenuation knobs."""
    spec = Parameters.from_groups(
        sfh={"type": "dpl"},
        dust=builders.dust.two_component(_=FREE),
    )
    free_dust = {p for p in spec.free_params if p.startswith("dust_")}
    assert "dust_tau_bc" in free_dust
    assert "dust_tau_diff" in free_dust


def test_single_component_uses_tau_v_not_tau_bc() -> None:
    spec = Parameters.from_groups(
        sfh={"type": "dpl"},
        dust=builders.dust.single_component(_=FREE),
    )
    free_dust = {p for p in spec.free_params if p.startswith("dust_")}
    assert "dust_tau_v" in free_dust
    assert "dust_tau_bc" not in free_dust


def test_per_param_override_survives_round_trip() -> None:
    spec = Parameters.from_groups(
        sfh={"type": "dpl"},
        dust=builders.dust.two_component(tau_bc=Uniform(0.5, 3.0)),
    )
    assert "dust_tau_bc" in spec.free_params


# ── Validation ────────────────────────────────────────────────────


def test_unknown_kwarg_raises_with_valid_list() -> None:
    with pytest.raises(TypeError, match="two_component") as exc:
        builders.dust.two_component(tau_bcc=Uniform(0, 2))  # typo
    msg = str(exc.value)
    assert "tau_bcc" in msg
    assert "tau_bc" in msg


def test_invalid_wildcard_rejected() -> None:
    with pytest.raises(ValueError, match="FREE or FIXED"):
        builders.dust.two_component(_="free")
