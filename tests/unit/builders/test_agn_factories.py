# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.agn — the composable AGN factory + 5 sub-blocks."""

from __future__ import annotations

import inspect

import pytest

from tengri import FIXED, FREE, Parameters, Uniform, builders
from tengri.parameters.groups import (
    _VALID_AGN_ATTEN_TYPES,
    _VALID_AGN_DISC_TYPES,
    _VALID_AGN_FEII_TYPES,
    _VALID_AGN_LINES_TYPES,
    _VALID_AGN_TORUS_TYPES,
)

# ── Sub-block coverage: every variant has a factory ───────────────


@pytest.mark.parametrize(
    "axis,expected_set",
    [
        ("disc", _VALID_AGN_DISC_TYPES),
        ("torus", _VALID_AGN_TORUS_TYPES),
        ("lines", _VALID_AGN_LINES_TYPES),
        ("feii", _VALID_AGN_FEII_TYPES),
        ("atten", _VALID_AGN_ATTEN_TYPES),
    ],
)
def test_axis_factories_cover_every_variant(axis: str, expected_set: set[str]) -> None:
    mod = getattr(builders.agn, axis)
    assert set(mod.available()) == expected_set


# ── Sub-block default shape ───────────────────────────────────────


def test_disc_default_call() -> None:
    assert builders.agn.disc.multicolor() == {"type": "multicolor", "*": FIXED}


def test_torus_default_call() -> None:
    out = builders.agn.torus.skirtor()
    assert out["type"] == "skirtor"
    assert out["*"] is FIXED


def test_lines_none_default_call() -> None:
    assert builders.agn.lines.none() == {"type": "none", "*": FIXED}


# ── Sub-block signatures: variants within an axis share the same params ──


def test_all_torus_variants_share_signature() -> None:
    """The torus param partition is identical across variants —
    every torus factory must have the same keyword signature."""
    sigs = {
        v: list(inspect.signature(getattr(builders.agn.torus, v)).parameters)
        for v in builders.agn.torus.available()
    }
    reference = sigs["skirtor"]
    for variant, params in sigs.items():
        assert params == reference, f"torus.{variant} drifted: {params}"


def test_torus_signature_lists_canonical_short_params() -> None:
    """Headline torus knobs from ``_AGN_PARTITION`` must surface."""
    sig = inspect.signature(builders.agn.torus.skirtor)
    params = set(sig.parameters)
    assert "tau_skirtor" in params  # SKIRTOR optical depth
    assert "torus_frac" in params


# ── Top-level composable factory ──────────────────────────────────


def test_composable_signature_lists_sub_block_kwargs() -> None:
    sig = inspect.signature(builders.agn.composable)
    params = list(sig.parameters)
    assert params[0] == "_"
    # Sub-blocks come right after the wildcard.
    assert set(params[1:6]) == {"disc", "torus", "lines", "feii", "atten"}


def test_composable_signature_includes_shared_short_params() -> None:
    sig = inspect.signature(builders.agn.composable)
    params = set(sig.parameters)
    # Shared params live in the agn partition.
    assert "log_lbol" in params
    assert "log_mbh" in params


def test_composable_default_call_is_minimal() -> None:
    assert builders.agn.composable() == {"type": "composable", "*": FIXED}


def test_full_composition_produces_grammar_shape() -> None:
    out = builders.agn.composable(
        _=FREE,
        log_lbol=Uniform(43.0, 47.0),
        disc=builders.agn.disc.multicolor(_=FREE),
        torus=builders.agn.torus.skirtor(_=FIXED),
        lines=builders.agn.lines.nlr(),
        feii=builders.agn.feii.none(),
        atten=builders.agn.atten.smc_prevot(),
    )
    assert out["type"] == "composable"
    assert out["*"] is FREE
    assert out["disc"] == {"type": "multicolor", "*": FREE}
    assert out["torus"] == {"type": "skirtor", "*": FIXED}
    assert out["log_lbol"] == Uniform(43.0, 47.0)


# ── Validation ────────────────────────────────────────────────────


def test_sub_block_kwarg_must_be_dict() -> None:
    with pytest.raises(TypeError, match="disc"):
        builders.agn.composable(disc="multicolor")  # forgot to call


def test_unknown_kwarg_raises_with_valid_lists() -> None:
    with pytest.raises(TypeError, match="composable") as exc:
        builders.agn.composable(log_lboll=Uniform(43, 47))  # typo
    msg = str(exc.value)
    assert "log_lboll" in msg
    assert "log_lbol" in msg
    assert "disc" in msg  # sub-block hint in the error


def test_invalid_wildcard_rejected() -> None:
    with pytest.raises(ValueError, match="FREE or FIXED"):
        builders.agn.composable(_="free")


def test_axis_factory_unknown_kwarg_rejected() -> None:
    with pytest.raises(TypeError, match="skirtor"):
        builders.agn.torus.skirtor(tau_skirrtor=Uniform(0, 10))  # typo


# ── Round-trip through parser ─────────────────────────────────────


def test_composable_round_trips_to_free_log_lbol() -> None:
    spec = Parameters.from_groups(
        sfh={"type": "dpl"},
        agn=builders.agn.composable(
            log_lbol=Uniform(43, 47),
            disc=builders.agn.disc.powerlaw(),
            torus=builders.agn.torus.skirtor(),
            lines=builders.agn.lines.nlr(),
            feii=builders.agn.feii.none(),
            atten=builders.agn.atten.smc_prevot(),
        ),
    )
    assert "agn_log_lbol" in spec.free_params


# ── Module-level wiring ───────────────────────────────────────────


def test_agn_in_builders_namespace() -> None:
    assert builders.agn is not None
    for axis in ("disc", "torus", "lines", "feii", "atten"):
        assert getattr(builders.agn, axis) is not None


def test_available_axes_returns_dict_of_lists() -> None:
    axes = builders.agn.available_axes()
    assert set(axes) == {"disc", "torus", "lines", "feii", "atten"}
    for _axis, variants in axes.items():
        assert isinstance(variants, list)
        assert variants == sorted(variants)
        assert len(variants) >= 2
