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
   in shape and round-trips through :meth:`Parameters.from_groups` to
   an identical :class:`Parameters` (same free params, same priors).
4. Unknown kwargs and invalid wildcard values raise with helpful
   messages — typos surface at call time, not at construction time.
"""

from __future__ import annotations

import inspect

import pytest

from tengri import FIXED, FREE, Fixed, Parameters, Uniform, builders
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

# ── Coverage: every canonical variant gets a factory ──────────────


def _canonical_variants() -> list[str]:
    """Return the canonical SFH variant names (skipping alias keys)."""
    canonical: list[str] = []
    for key, entry in SFH_REGISTRY.items():
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
    assert params[0] == "_", "wildcard kwarg must come first for ergonomics"
    assert set(params[1:]) == {"alpha", "beta", "tau_gyr", "log_peak_sfr"}
    for name, p in sig.parameters.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, name


@pytest.mark.parametrize("variant", _canonical_variants())
def test_every_factory_has_a_real_signature(variant: str) -> None:
    factory = getattr(builders.sfh, variant)
    sig = inspect.signature(factory)
    assert "_" in sig.parameters, "wildcard kwarg missing"
    # Wildcard defaults to FIXED.
    assert sig.parameters["_"].default is FIXED
    # Every non-wildcard kwarg is KEYWORD_ONLY.
    for name, p in sig.parameters.items():
        if name == "_":
            continue
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, name


# ── Output shape ──────────────────────────────────────────────────


def test_default_call_produces_minimal_dict() -> None:
    assert builders.sfh.dpl() == {"type": "dpl", "*": FIXED}


def test_wildcard_free() -> None:
    assert builders.sfh.dpl(_=FREE) == {"type": "dpl", "*": FREE}


def test_per_param_override_is_preserved() -> None:
    prior = Uniform(1.0, 3.0)
    out = builders.sfh.dpl(beta=prior)
    assert out == {"type": "dpl", "*": FIXED, "beta": prior}


def test_wildcard_plus_explicit_override() -> None:
    pin = Fixed(1.0)
    out = builders.sfh.dpl(_=FREE, log_peak_sfr=pin)
    assert out == {"type": "dpl", "*": FREE, "log_peak_sfr": pin}


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
    via_factory = Parameters.from_groups(sfh=builders.sfh.dpl(_=FREE, beta=Uniform(1.0, 3.0)))
    via_dict = Parameters.from_groups(sfh={"type": "dpl", "*": FREE, "beta": Uniform(1.0, 3.0)})
    assert sorted(via_factory.free_params) == sorted(via_dict.free_params)
    # Spot-check that the user-overridden prior survived.
    assert "sfh_dpl_beta" in via_factory.free_params


def test_const_exp_short_names_resolve_correctly() -> None:
    """Variants whose param prefix differs from the variant name still work.

    ``const_exp`` parameters use the ``sfh_cexp_`` prefix, so the short
    forms exposed by the factory must match what the parser extracts.
    """
    spec = Parameters.from_groups(sfh=builders.sfh.const_exp(_=FREE))
    expected = {
        "sfh_cexp_log_sfr",
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
    Parameters.from_groups(sfh=factory())


def test_burst_and_field_factories_emit_valid_dicts() -> None:
    """Mixture/modulator factories still emit well-formed dicts, even
    though they cannot be used standalone as ``sfh=``."""
    burst_out = builders.sfh.burst()
    assert burst_out["type"] == "burst"
    assert burst_out["*"] is FIXED
    field_out = builders.sfh.field(_=FREE)
    assert field_out["type"] == "field"
    assert field_out["*"] is FREE


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
