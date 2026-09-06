# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.agn — the composable AGN factory + 5 sub-blocks."""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract

from tengri import DEFAULT, FREE, Fixed, Uniform, builders, parse_groups
from tengri.components.agn.unified import monolithic_agn_model_names
from tengri.parameters.groups import (
    _VALID_AGN_ATTEN_TYPES,
    _VALID_AGN_BLR_TYPES,
    _VALID_AGN_DISC_TYPES,
    _VALID_AGN_FEII_TYPES,
    _VALID_AGN_NLR_TYPES,
    _VALID_AGN_TORUS_TYPES,
)

# ── Sub-block coverage: every variant has a factory ───────────────


@pytest.mark.parametrize(
    "axis,expected_set",
    [
        ("disc", _VALID_AGN_DISC_TYPES),
        ("torus", _VALID_AGN_TORUS_TYPES),
        ("nlr", _VALID_AGN_NLR_TYPES),
        ("blr", _VALID_AGN_BLR_TYPES),
        ("feii", _VALID_AGN_FEII_TYPES),
        ("atten", _VALID_AGN_ATTEN_TYPES),
    ],
)
def test_axis_factories_cover_every_variant(axis: str, expected_set: set[str]) -> None:
    mod = getattr(builders.agn, axis)
    assert set(mod.available()) == expected_set


# ── Sub-block default shape ───────────────────────────────────────


def test_disc_default_call() -> None:
    assert builders.agn.disc.multicolor() == {"type": "multicolor", "all_params": Fixed(DEFAULT)}


def test_torus_default_call() -> None:
    out = builders.agn.torus.skirtor()
    assert out["type"] == "skirtor"
    assert out["all_params"] == Fixed(DEFAULT)


def test_nlr_none_default_call() -> None:
    assert builders.agn.nlr.none() == {"type": "none", "all_params": Fixed(DEFAULT)}


def test_blr_none_default_call() -> None:
    assert builders.agn.blr.none() == {"type": "none", "all_params": Fixed(DEFAULT)}


# ── Sub-block signatures: variants within an axis share the same params ──


def test_all_torus_variants_share_signature() -> None:
    """The torus param partition is identical across variants —
    every torus factory must have the same keyword signature.

    The count is asserted first because the comparison lives inside the loop
    and is anchored on ``skirtor``. If ``available()`` ever returned just
    ``["skirtor"]``, the loop would compare the reference to itself, pass, and
    prove nothing — which is the failure mode of every "all N agree with a
    reference" test where N can shrink to one. (An *empty* return is already
    caught by the ``sigs["skirtor"]`` lookup raising KeyError.)
    """
    sigs = {
        v: list(inspect.signature(getattr(builders.agn.torus, v)).parameters)
        for v in builders.agn.torus.available()
    }
    assert len(sigs) > 1, (
        f"only {len(sigs)} torus variant(s) discovered — a self-comparison "
        f"cannot detect signature drift"
    )
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
    assert params[0] == "all_params"
    assert params[1] == "other_params", "wildcard synonym comes right after all_params"
    # Sub-blocks come right after the two wildcard spellings.
    assert set(params[2:8]) == {"disc", "torus", "nlr", "blr", "feii", "atten"}


def test_composable_signature_includes_shared_short_params() -> None:
    """The composable factory offers the TRULY shared parameters.

    ``log_mbh`` was pinned here as shared. It is not: every reader is a disc
    block (adaf, kd18, kubota_done, multicolor, relagn, slone_netzer), so the
    ownership model gave it the ``agn.disc`` owner and the composable factory
    correctly stopped offering it at the top level. The assertion moves to the
    disc factory rather than being dropped -- see
    ``test_disc_signature_includes_disc_owned_short_params`` below.
    """
    sig = inspect.signature(builders.agn.composable)
    params = set(sig.parameters)
    # Shared params live in the agn partition.
    assert "log_lbol" in params
    assert "lum_ratio" in params
    assert "log_mbh" not in params


def test_disc_signature_includes_disc_owned_short_params() -> None:
    """The equivalent assertion at ``log_mbh``'s new owner.

    A disc factory that did not offer its own physics would be the same defect
    one level down, so this is the half of the pin that still bites.
    """
    for factory in (builders.agn.disc.multicolor, builders.agn.disc.kd18_agnfitter):
        params = set(inspect.signature(factory).parameters)
        assert "log_mbh" in params, factory.__name__


def test_composable_default_call_is_minimal() -> None:
    assert builders.agn.composable() == {"type": "composable", "all_params": Fixed(DEFAULT)}


def test_full_composition_produces_grammar_shape() -> None:
    out = builders.agn.composable(
        all_params=FREE,
        log_lbol=Uniform(9.42, 13.42),
        disc=builders.agn.disc.multicolor(all_params=FREE),
        torus=builders.agn.torus.skirtor(all_params=Fixed(DEFAULT)),
        nlr=builders.agn.nlr.analytic(),
        blr=builders.agn.blr.analytic(),
        feii=builders.agn.feii.none(),
        atten=builders.agn.atten.smc_prevot(),
    )
    assert out["type"] == "composable"
    # log_lbol is a genuine per-param override (sub-blocks are structural
    # selectors, not parameter entries), so the wildcard is spelled
    # 'other_params' and emitted LAST.
    assert out["other_params"] is FREE
    assert list(out.keys())[-1] == "other_params"
    assert out["disc"] == {"type": "multicolor", "all_params": FREE}
    assert out["torus"] == {"type": "skirtor", "all_params": Fixed(DEFAULT)}
    assert out["log_lbol"] == Uniform(9.42, 13.42)


# ── Validation ────────────────────────────────────────────────────


def test_sub_block_kwarg_must_be_dict() -> None:
    with pytest.raises(TypeError, match="disc"):
        builders.agn.composable(disc="multicolor")  # forgot to call


def test_unknown_kwarg_raises_with_valid_lists() -> None:
    with pytest.raises(TypeError, match="composable") as exc:
        builders.agn.composable(log_lboll=Uniform(9.42, 13.42))  # typo
    msg = str(exc.value)
    assert "log_lboll" in msg
    assert "log_lbol" in msg
    assert "disc" in msg  # sub-block hint in the error


def test_invalid_wildcard_rejected() -> None:
    # _= alias is retired; raises TypeError before checking the value
    with pytest.raises(TypeError, match=r"_=.*retired"):
        builders.agn.composable(_="free")


def test_axis_factory_unknown_kwarg_rejected() -> None:
    with pytest.raises(TypeError, match="skirtor"):
        builders.agn.torus.skirtor(tau_skirrtor=Uniform(0, 10))  # typo


# ── Round-trip through parser ─────────────────────────────────────


def test_composable_round_trips_to_free_log_lbol() -> None:
    spec = parse_groups(
        sfh={"type": "dpl"},
        agn=builders.agn.composable(
            log_lbol=Uniform(9.42, 13.42),
            disc=builders.agn.disc.powerlaw(),
            torus=builders.agn.torus.skirtor(),
            nlr=builders.agn.nlr.analytic(),
            blr=builders.agn.blr.none(),
            feii=builders.agn.feii.none(),
            atten=builders.agn.atten.smc_prevot(),
        ),
        redshift=Fixed(0.1),
    )
    assert "agn_log_lbol" in spec.free_params


# ── Module-level wiring ───────────────────────────────────────────


def test_agn_in_builders_namespace() -> None:
    assert builders.agn is not None
    for axis in ("disc", "torus", "nlr", "blr", "feii", "atten"):
        assert getattr(builders.agn, axis) is not None


def test_available_axes_returns_dict_of_lists() -> None:
    axes = builders.agn.available_axes()
    assert set(axes) == {"disc", "torus", "nlr", "blr", "feii", "atten"}
    for _axis, variants in axes.items():
        assert isinstance(variants, list)
        assert variants == sorted(variants)
        assert len(variants) >= 2


# ── Top-level AGN model factories ─────────────────────────────────


def test_all_13_top_level_models_have_factories() -> None:
    """All 13 non-composable AGN models must be callable factories."""
    top_level_models = sorted(monolithic_agn_model_names())
    for model_name in top_level_models:
        factory = getattr(builders.agn, model_name, None)
        assert factory is not None, f"builders.agn.{model_name} not found"
        assert callable(factory), f"builders.agn.{model_name} is not callable"


def test_available_lists_every_buildable_model() -> None:
    """``available()`` lists exactly the models ``agn={'type': ...}`` accepts.

    These three lists were hand-written and had drifted: they carried
    ``simple`` and ``standard`` -- composable torus block names, not models --
    and omitted ``richards2006`` and ``skirtor_stalevski``, which do build. So
    ``builders.agn.simple()`` produced a config that raised
    ``Unknown AGN model`` at predict time, and the factory for a real model was
    missing. Derived from the registry now, which is the same set R37 validates
    ``agn['type']`` against.
    """
    factories = builders.agn.available()
    assert factories == sorted(factories)
    assert set(factories) == set(monolithic_agn_model_names()) | {"composable"}
    assert "composable" in factories
    non_composable = sorted(monolithic_agn_model_names())
    non_composable = sorted(monolithic_agn_model_names())
    for model_name in non_composable:
        assert model_name in factories, f"{model_name} not in available()"


def test_skirtor_signature_includes_canonical_params() -> None:
    """skirtor (and all top-level models) must expose canonical AGN params."""
    sig = inspect.signature(builders.agn.skirtor)
    params = set(sig.parameters)
    assert "log_lbol" in params
    assert "log_mbh" in params
    assert "tau_skirtor" in params  # torus param


def test_skirtor_default_call_shape() -> None:
    """All top-level models return a dict with type and wildcard."""
    out = builders.agn.skirtor()
    assert out["type"] == "skirtor"
    assert out["all_params"] == Fixed(DEFAULT)


def test_top_level_models_share_signature() -> None:
    """All 13 top-level models must have identical keyword signatures."""
    top_level_models = sorted(monolithic_agn_model_names())
    sigs = {
        m: list(inspect.signature(getattr(builders.agn, m)).parameters) for m in top_level_models
    }
    reference = sigs["multicolor_agn"]
    for model_name, params in sigs.items():
        assert params == reference, (
            f"builders.agn.{model_name} signature drifted from multicolor_agn"
        )


def test_top_level_round_trip_makes_log_lbol_free() -> None:
    """Round-trip through parser must recognize top-level factory params."""
    spec = parse_groups(
        sfh={"type": "dpl"},
        agn=builders.agn.multicolor_agn(log_lbol=Uniform(9.42, 13.42)),
        redshift=Fixed(0.1),
    )
    assert "agn_log_lbol" in spec.free_params
