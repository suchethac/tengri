# SPDX-License-Identifier: BSD-3-Clause
"""Regression contracts for the discovery menu and the property catalog.

Fresh-user audit (2026-07) found the public introspection façade had drifted
from the auto-registering component registries:

* ``describe()`` and ``search()`` walked hand-maintained tuples of ``list_*``
  functions that omitted the xray / radio / igm menus and the composable AGN
  blocks — so ``describe("madau")`` (a real IGM model) raised ``KeyError`` and
  ``search("skirtor")`` (a real AGN block) returned nothing, even though the
  models auto-register into their own registries.
* ``list_sfh_models()`` advertised the ``UNVALIDATED_SFH_TYPES`` that the
  builder refuses — a new user picking one off the menu hit a ``ValueError``.
* the menu's "Use:" hint taught the deprecated flat ``Parameters(...)`` form.
* the ``PropertyCatalog`` (``pred.properties``) had no ``.get()``, so the
  quickstart notebook's ``pred.properties.get(name)`` raised ``AttributeError``.

These tests pin the fixes and, via :func:`test_every_listed_model_is_describable`,
guard against the aggregator drift recurring when a new menu is added.
"""

from __future__ import annotations

import re

import jax
import pytest

import tengri
from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sfh.registry import UNVALIDATED_SFH_TYPES

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


# ── describe() / search() cover every menu ──────────────────────────────────


@pytest.mark.parametrize(
    "lister",
    [
        tengri.list_xray_models,
        tengri.list_radio_models,
        tengri.list_igm_models,
    ],
)
def test_describe_finds_xray_radio_igm(lister):
    """Every name a menu advertises must be resolvable by describe()."""
    for name in lister().names():
        rec = tengri.describe(name)  # must not raise KeyError
        assert rec["name"] == name


def test_every_listed_model_is_describable():
    """Systemic guard: nothing in any model menu is invisible to describe().

    This is the durable version of the fix — if a future physics group adds a
    ``list_*`` menu but forgets to wire it into the aggregators, this fails.
    """
    from tengri.registry import _menu_listers

    for lister in _menu_listers():
        for name in lister().names():
            tengri.describe(name)  # KeyError here means the menu drifted again


def test_search_finds_agn_blocks_and_igm():
    """search() walks the composable AGN blocks and the igm menu, not just
    the monolithic AGN model list."""
    assert len(tengri.search("skirtor")) > 0  # skirtor is an AGN *block*
    assert len(tengri.search("madau")) > 0  # madau is an IGM model
    assert len(tengri.search("torus")) > 0  # unchanged behavior still works


def test_search_concept_aliases_reach_the_right_menu():
    """Natural-language concept terms a beginner types don't substring-match the
    terse model short_docs: "star formation" would return only an AGN model
    (its citation title contains the phrase) and none of the SFH models, and
    "dust emission" nothing at all. Concept aliases redirect to the menu that
    actually holds those models."""
    sfh_names = set(tengri.list_sfh_models().names())
    for term in ("star formation", "star-forming"):
        hits = set(tengri.search(term).names())
        assert hits == sfh_names, f"search({term!r}) should reach the SFH menu"
        assert "feltre" not in hits  # the old misleading sole hit is gone

    assert set(tengri.search("dust emission").names()) == set(
        tengri.list_dust_emission_models().names()
    )
    assert set(tengri.search("extinction").names()) == set(tengri.list_dust_laws().names())
    assert set(tengri.search("emission lines").names()) == set(
        tengri.list_nebular_backends().names()
    )

    # a normal substring query is unaffected by the alias layer
    assert "skirtor" in set(tengri.search("skirtor").names())
    assert "madau" in set(tengri.search("madau").names())


def test_describe_discloses_ambiguous_names():
    """A name registered in more than one menu/category (skirtor = disc+torus,
    simple = torus+xray, cue = nlr+nebular) must disclose the other locations
    rather than silently returning only the first match."""
    for name in ("skirtor", "simple", "cue"):
        rec = dict(tengri.describe(name))
        assert "also_registered_as" in rec, f"describe({name!r}) hid its ambiguity"
        assert name in rec["also_registered_as"]

    # an unambiguous name carries no disambiguation note
    assert "also_registered_as" not in dict(tengri.describe("dpl"))


# ── tab-completion (dir) surfaces the recommended-workflow essentials ────────


def test_tab_completion_includes_grammar_essentials():
    """`dir(tengri)` is a curated tab-completion surface (deliberately not the
    full __all__). It must still include the names the *recommended* workflow
    depends on — the grammar sentinels, the recipe/builder entry points, and
    the SSP loader — or a novice exploring `tengri.<TAB>` can't reach them.
    """
    completable = set(dir(tengri))
    essentials = {
        "FREE",
        "FIXED",  # nested-dict grammar sentinels (sfh={'*': FREE})
        "recipes",
        "builders",  # recommended build entry points
        "load_ssp_data",
        "SSPData",  # the first call every user makes + its return type
        "SEDModel",
        "describe",
        "cite",  # already curated — pin them
    }
    missing = essentials - completable
    assert not missing, f"recommended-workflow names not tab-completable: {sorted(missing)}"


def test_every_tab_completion_name_resolves():
    """No curated tab-completion entry may be a dead name (attribute error on
    access) — a broken completion is worse than an omission."""
    for name in dir(tengri):
        assert hasattr(tengri, name), f"dir(tengri) advertises {name!r} but it does not resolve"


# ── list_sfh_models() marks unbuildable types instead of advertising them ────


def test_list_sfh_models_marks_unvalidated():
    """Unvalidated SFH types (which the builder refuses) must not be presented
    as ``production``: they carry ``status='unvalidated'`` so a fresh user is
    not sent into a build-time ValueError by picking one off the menu."""
    rows = {r["name"]: r for r in tengri.list_sfh_models()}
    for name in UNVALIDATED_SFH_TYPES:
        assert name in rows, f"{name} missing from list_sfh_models()"
        assert rows[name]["status"] == "unvalidated", (
            f"{name} must be flagged 'unvalidated', got {rows[name]['status']!r}"
        )


def test_list_sfh_models_production_filter_excludes_unvalidated():
    """Filtering to the buildable set drops every unvalidated type."""
    production = set(tengri.list_sfh_models(status="production").names())
    assert UNVALIDATED_SFH_TYPES.isdisjoint(production), (
        f"production filter leaked unbuildable types: {sorted(UNVALIDATED_SFH_TYPES & production)}"
    )


def _sfh_counts_in(text):
    """Every integer printed on the ``list_sfh_models()`` line of a cheatsheet."""
    line = next(ln for ln in text.splitlines() if "list_sfh_models()" in ln)
    return [int(tok) for tok in re.findall(r"\d+", line)], line


def test_help_headlines_the_buildable_sfh_count_not_the_raw_registry(capsys):
    """``tengri.help()`` must not advertise a count the builder rejects.

    ``summary()`` reports the buildable/total pair, but ``help()`` — the
    surface ``summary()`` itself signposts as the "curated cheatsheet for new
    users" — headlined the raw registry total. A newcomer read "34 SFH
    variants", picked one of the eight :data:`UNVALIDATED_SFH_TYPES`, and hit a
    build-time ``ValueError``. The *first* number on that line is what a reader
    takes as the menu size, so it must be the buildable one.
    """
    n_ok = len(tengri.list_sfh_models(status="production"))
    n_all = len(tengri.list_sfh_models())
    assert n_ok < n_all, (
        "precondition gone: no unvalidated SFH types remain, so this guard "
        "can no longer distinguish the buildable count from the raw total"
    )

    tengri.help()
    numbers, line = _sfh_counts_in(capsys.readouterr().out)

    assert numbers, f"no count on the list_sfh_models line: {line!r}"
    assert numbers[0] == n_ok, (
        f"help() headlines {numbers[0]} SFH models but only {n_ok} build "
        f"(registry holds {n_all}). Line: {line!r}"
    )


def test_summary_and_help_agree_on_the_sfh_counts(capsys):
    """The two front-page surfaces must not contradict each other."""
    tengri.summary()
    summary_out = capsys.readouterr().out
    tengri.help()
    help_numbers, help_line = _sfh_counts_in(capsys.readouterr().out)

    buildable = next(ln for ln in summary_out.splitlines() if "buildable SFH models" in ln)
    total = next(ln for ln in summary_out.splitlines() if "total SFH models" in ln)
    summary_pair = (
        int(re.search(r"\d+", buildable).group()),
        int(re.search(r"\d+", total).group()),
    )

    assert set(summary_pair) <= set(help_numbers), (
        f"summary() reports {summary_pair} for (buildable, total) SFH models but "
        f"help() shows {help_numbers}. Line: {help_line!r}"
    )


def _tutorial_texts():
    """(name, text) for every registered tutorial page."""
    from tengri._tutorials import _TUTORIALS

    assert _TUTORIALS, "no tutorials registered — this guard would vacuously pass"
    return [(name, tut.code) for name, tut in _TUTORIALS.items()]


@pytest.mark.parametrize("name, text", _tutorial_texts())
def test_tutorials_do_not_hardcode_menu_counts(name, text):
    """A hand-written menu size goes stale the moment a model is registered.

    ``design_philosophy`` advertised "12 AGN models / 21 attenuation curves /
    7 IR templates / 4 backends" when the live registries held 1 / 22 / 18 / 5
    — four of five wrong. Counts belong in :func:`tengri.summary`, which
    computes them, never in prose.
    """
    stale = re.findall(r"(tengri\.list_\w+\(\))\s*#\s*(\d+)", text)
    assert not stale, (
        f"tutorial {name!r} hardcodes menu counts {stale}; these rot silently. "
        f"Describe the menu instead and point at tengri.summary() for counts."
    )


@pytest.mark.parametrize("name, text", _tutorial_texts())
def test_tutorials_only_advertise_real_list_functions(name, text):
    """Every ``tengri.list_*()`` a tutorial names must actually exist.

    Same class as the ``list_components()`` hint that advertised
    ``list_stellar_models`` (#1179): a tutorial is a promise the public
    namespace has to keep.
    """
    for call in sorted(set(re.findall(r"tengri\.(list_\w+)\(", text))):
        assert callable(getattr(tengri, call, None)), (
            f"tutorial {name!r} advertises tengri.{call}(), which does not exist"
        )


# ── the structural dust axis is discoverable at all ─────────────────────────


def _valid_dust_types():
    from tengri.parameters.groups import _VALID_DUST_TYPES

    assert _VALID_DUST_TYPES, "no dust types registered — this guard would pass vacuously"
    return sorted(_VALID_DUST_TYPES)


@pytest.mark.parametrize("dust_type", _valid_dust_types())
def test_structural_dust_types_are_discoverable(dust_type):
    """``dust={'type': ...}`` must be reachable from the discovery API.

    Dust is chosen along three axes: structure (``dust={'type': ...}``),
    attenuation curve (``law_bc`` / ``law_diff``), and IR emission
    (``dust={'emission': ...}``). The latter two had menus; the structural
    axis had none — so ``two_component``, the type the shipped recipes
    build with, was named by no ``list_*`` menu, ``describe('two_component')``
    raised ``KeyError`` and ``search('two_component')`` returned nothing. A
    user following the documented discovery path could not find the single
    most consequential dust choice.
    """
    from tengri.registry import _menu_listers

    homes = [ln.__name__ for ln in _menu_listers() if dust_type in set(ln().names())]
    assert homes, f"{dust_type!r} is accepted by the builder but named by no list_* menu"

    assert dict(tengri.describe(dust_type))["name"] == dust_type
    assert dust_type in set(tengri.search(dust_type).names())


def test_dust_model_menu_cannot_drift_from_the_builder():
    """The menu is derived from the validator's own set, both directions.

    Hard-coding the names here later would let the menu advertise a type
    ``SEDModel.build`` rejects (the #1179 failure mode) or hide one it
    accepts (what this test was written for).
    """
    from tengri.parameters.groups import _VALID_DUST_TYPES

    assert set(tengri.list_dust_models().names()) == set(_VALID_DUST_TYPES)


# ── usage hints teach the current SEDModel.build grammar ────────────────────


@pytest.mark.parametrize(
    "lister",
    [
        tengri.list_sfh_models,
        tengri.list_dust_models,
        tengri.list_dust_laws,
        tengri.list_dust_emission_models,
        tengri.list_nebular_backends,
        tengri.list_agn_models,
    ],
)
def test_usage_hints_teach_build_grammar(lister):
    """The 'Use:' hint must point at SEDModel.build, not the deprecated
    flat Parameters(...) escape hatch."""
    hint = lister()[0]["use"]
    assert "SEDModel.build" in hint, hint
    assert "Parameters(" not in hint, hint


# ── PropertyCatalog is dict-like (.get) ─────────────────────────────────────


def test_property_catalog_get(synthetic_ssp_wide, synthetic_tophat_obs):
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED},
        redshift=Fixed(0.1),
    )
    pred = model.predict(model.spec.sample(jax.random.PRNGKey(0)))
    props = pred.properties

    assert hasattr(props, "get")
    # present key returns the same value as __getitem__
    assert props.get("stellar_mass") is not None
    assert props.get("stellar_mass") == props["stellar_mass"]
    # missing key returns the default, never raises
    sentinel = object()
    assert props.get("definitely_not_a_property", sentinel) is sentinel
    assert props.get("definitely_not_a_property") is None
