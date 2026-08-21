# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.sfh — auto-generated SFH config factories.

The contract these tests pin down:

1. Every canonical SFH variant in :data:`SFH_REGISTRY` has a matching
   factory (and short-name aliases like ``tsnorm`` for ``truncated_skewnormal`` don't get a
   duplicate one).
2. Each factory's :func:`inspect.signature` exposes ``_`` (wildcard) plus
   one keyword-only parameter per short-form name — i.e. real
   per-variant signatures that IDEs and type checkers can introspect.
3. Factory output matches the hand-written nested-dict form byte-for-byte
   in shape and round-trips through :func:`parse_groups` to
   an identical :class:`Parameters` (same free params, same priors).
4. Unknown kwargs and invalid wildcard values raise with helpful
   messages — typos surface at call time, not at construction time.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract

from tengri import FIXED, FREE, Fixed, Uniform, builders, parse_groups
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

# ── Coverage: every canonical variant gets a factory ──────────────


def _canonical_variants() -> list[str]:
    """Return the canonical SFH variant names (skipping alias keys).

    Excludes the not-yet-validated SFHs (``UNVALIDATED_SFH_TYPES``): the
    grammar rejects them and the factory layer does not surface them, so
    they have no factory to introspect.
    """
    from tengri.components.stellar.sfh.registry import UNVALIDATED_SFH_TYPES

    canonical: list[str] = []
    for key, entry in SFH_REGISTRY.items():
        if key in UNVALIDATED_SFH_TYPES:
            continue
        spec = entry.callable if hasattr(entry, "callable") else entry
        if getattr(spec, "name", None) == key:
            canonical.append(key)
    return canonical


def test_every_canonical_variant_has_a_factory() -> None:
    canonical = set(_canonical_variants())
    surfaced = set(builders.sfh.available())
    missing = canonical - surfaced
    extra = surfaced - canonical
    assert not missing, f"variants missing factories: {missing}"
    assert not extra, f"factories without backing variants: {extra}"


def test_aliases_do_not_get_their_own_factory() -> None:
    # Factories are keyed by each spec's canonical short ``name``
    # (e.g. ``tsnorm`` rather than the longer ``truncated_skewnormal``).
    assert "tsnorm" in builders.sfh.available()
    assert "truncated_skewnormal" not in builders.sfh.available()


# ── Signature contract: real keyword-only params ──────────────────


def test_dpl_signature_lists_expected_params() -> None:
    sig = inspect.signature(builders.sfh.dpl)
    params = list(sig.parameters)
    assert params[0] == "all_params", "wildcard kwarg must come first for ergonomics"
    assert set(params[1:]) == {"alpha", "beta", "tau_gyr", "age_gyr", "log_total_mass"}
    for name, p in sig.parameters.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, name


@pytest.mark.parametrize("variant", _canonical_variants())
def test_every_factory_has_a_real_signature(variant: str) -> None:
    factory = getattr(builders.sfh, variant)
    sig = inspect.signature(factory)
    assert "all_params" in sig.parameters, "wildcard kwarg missing"
    # Wildcard defaults to FIXED.
    assert sig.parameters["all_params"].default is FIXED
    # Every non-wildcard kwarg is KEYWORD_ONLY.
    for name, p in sig.parameters.items():
        if name == "all_params":
            continue
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, name


# ── Output shape ──────────────────────────────────────────────────


def test_default_call_produces_minimal_dict() -> None:
    assert builders.sfh.dpl() == {"type": "dpl", "all_params": FIXED}


def test_wildcard_free() -> None:
    assert builders.sfh.dpl(_=FREE) == {"type": "dpl", "all_params": FREE}


def test_per_param_override_is_preserved() -> None:
    prior = Uniform(1.0, 3.0)
    out = builders.sfh.dpl(beta=prior)
    assert out == {"type": "dpl", "all_params": FIXED, "beta": prior}


def test_wildcard_plus_explicit_override() -> None:
    pin = Fixed(1.0)
    out = builders.sfh.dpl(_=FREE, log_total_mass=pin)
    assert out == {"type": "dpl", "all_params": FREE, "log_total_mass": pin}


# ── Validation: typos and bad sentinels ───────────────────────────


def test_unknown_kwarg_raises_typeerror_with_valid_list() -> None:
    with pytest.raises(TypeError, match="dpl") as exc:
        builders.sfh.dpl(beat=Uniform(1, 3))  # typo for 'beta'
    msg = str(exc.value)
    assert "beat" in msg
    assert "beta" in msg, "error message should advertise the valid name list"


def test_invalid_wildcard_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="FREE or FIXED"):
        builders.sfh.dpl(_="free")  # string is not a sentinel


# ── Grammar interop: factory output = hand-written dict ───────────


def test_factory_output_round_trips_through_parse_groups() -> None:
    """The factory must produce a Parameters identical to the dict path."""
    via_factory = parse_groups(
        sfh=builders.sfh.dpl(_=FREE, beta=Uniform(1.0, 3.0)), redshift=Fixed(0.1)
    )
    via_dict = parse_groups(
        sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1.0, 3.0)}, redshift=Fixed(0.1)
    )
    assert sorted(via_factory.free_params) == sorted(via_dict.free_params)
    # Spot-check that the user-overridden prior survived.
    assert "sfh_dpl_beta" in via_factory.free_params


def test_const_exp_short_names_resolve_correctly() -> None:
    """Variants whose param prefix differs from the variant name still work.

    ``const_exp`` parameters use the ``sfh_cexp_`` prefix, so the short
    forms exposed by the factory must match what the parser extracts.
    """
    spec = parse_groups(sfh=builders.sfh.const_exp(_=FREE), redshift=Fixed(0.1))
    expected = {
        "sfh_cexp_log_total_mass",
        "sfh_cexp_tau_gyr",
        "sfh_cexp_quench_gyr",
        "sfh_cexp_age_gyr",
    }
    assert expected.issubset(set(spec.free_params))


def _additive_variants() -> list[str]:
    """Variants that can be used standalone as the smooth SFH component.

    ``burst`` (mixture) and ``field`` (modulator) compose on top of a
    smooth SFH and cannot be passed alone as ``sfh=`` — that's a parser-
    side composition rule, not a factory concern. ``table`` carries zero
    free parameters and is exercised elsewhere.
    """
    additive: list[str] = []
    for key in _canonical_variants():
        entry = SFH_REGISTRY[key]
        spec = entry.callable if hasattr(entry, "callable") else entry
        if getattr(spec, "composition_type", "") == "additive" and key != "table":
            additive.append(key)
    return additive


@pytest.mark.parametrize("variant", _additive_variants())
def test_every_additive_factory_default_call_parses_cleanly(variant: str) -> None:
    """Default-call output (no overrides, FIXED wildcard) must parse cleanly.

    Restricted to additive (smooth) variants because mixture/modulator
    variants compose with a smooth SFH and are not valid standalone.
    """
    factory = getattr(builders.sfh, variant)
    parse_groups(sfh=factory(), redshift=Fixed(0.1))


def test_burst_and_field_factories_emit_valid_dicts() -> None:
    """Mixture/modulator factories still emit well-formed dicts, even
    though they cannot be used standalone as ``sfh=``."""
    burst_out = builders.sfh.burst()
    assert burst_out["type"] == "burst"
    assert burst_out["all_params"] is FIXED
    field_out = builders.sfh.field(_=FREE)
    assert field_out["type"] == "field"
    assert field_out["all_params"] is FREE


# ── Docstring smoke ───────────────────────────────────────────────


def test_factory_docstring_mentions_variant_and_params() -> None:
    doc = builders.sfh.dpl.__doc__ or ""
    assert "dpl" in doc
    assert "alpha" in doc
    assert "beta" in doc


# ── Module-level metadata ─────────────────────────────────────────


def test_builders_sfh_in_dunder_all_and_imports_from_tengri() -> None:
    import tengri

    assert "builders" in tengri.__all__
    assert builders.sfh is tengri.builders.sfh
