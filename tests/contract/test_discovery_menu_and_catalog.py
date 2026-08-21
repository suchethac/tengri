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
    # "metallicity" reached only the 5 modes whose short_doc uses the word,
    # silently dropping the gas-regulator and per-bin ones.
    assert set(tengri.search("metallicity").names()) == set(
        tengri.list_metallicity_modes().names()
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


def test_ambiguity_disclosure_fires_exactly_when_a_name_is_shared():
    """Derived both directions, so it cannot rot as menus are added.

    This replaced a literal ``assert "also_registered_as" not in describe("dpl")``.
    That assertion encoded a *bug* as the expectation: ``dpl`` looked
    unambiguous only because the radio AGN variants lived in no menu, and
    registering them (#1276) correctly flipped it. Counting the menus that
    claim each name means the next such name is covered without an edit.
    """
    from collections import Counter

    from tengri.registry import (
        _menu_listers,
        list_components,
        list_filters,
        list_plots,
        list_recipes,
    )

    # The same sources describe() itself consults.
    sources = (*_menu_listers(), list_components, list_filters, list_plots, list_recipes)
    counts: Counter[str] = Counter(name for fn in sources for name in fn().names())
    assert counts, "no names at all — this guard would pass vacuously"

    menu_names = {name for fn in _menu_listers() for name in fn().names()}
    shared = [n for n in menu_names if counts[n] > 1]
    unique = [n for n in menu_names if counts[n] == 1]
    assert shared and unique, f"need both cases to test; got {len(shared)}/{len(unique)}"

    for name in shared:
        rec = dict(tengri.describe(name))
        assert "also_registered_as" in rec, f"describe({name!r}) hid its ambiguity"
        assert name in rec["also_registered_as"]

    for name in unique:
        rec = dict(tengri.describe(name))
        assert "also_registered_as" not in rec, (
            f"describe({name!r}) claims ambiguity but the name is in one menu only"
        )


def test_ambiguity_disclosure_advice_is_executable():
    """The note must hand back a build call, not the name of a lookup helper.

    It used to say "Use the category-specific list (e.g. describe_agn_block
    (name, category=...))" for *every* ambiguity. That was true only while AGN
    blocks were the sole categorized menu: once radio gained sf/agn categories,
    following it for ``dpl`` raised ``KeyError: Unknown AGN block 'dpl' in
    category 'agn'. Known names: []`` — advice that fails the one user it
    exists to help (the #1275 class).
    """
    rec = dict(tengri.describe("dpl"))
    note = rec["also_registered_as"]
    assert "radio={'agn': {'type': 'dpl'}}" in note, note
    assert "describe_agn_block" not in note, note


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
        "FIXED",  # nested-dict grammar sentinels (sfh={'all_params': FREE})
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


# ── the radio sub-block and shock axes are discoverable at all ──────────────


def _radio_block_values():
    from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS

    assert SF_RADIO_MODELS and AGN_RADIO_MODELS, "no radio variants — guard would pass vacuously"
    return [("sf", n) for n in SF_RADIO_MODELS] + [("agn", n) for n in AGN_RADIO_MODELS]


@pytest.mark.parametrize(("category", "value"), _radio_block_values())
def test_radio_subblock_values_are_discoverable(category, value):
    """``radio={'sf'|'agn': {'type': ...}}`` must be reachable from discovery.

    ``list_radio_models()`` covers only the legacy ``radio={'type': ...}`` key.
    The three SF variants and the two AGN variants were accepted by the builder
    but named by no menu at all, so ``describe('bell2003')`` raised ``KeyError``.
    Worse for the AGN pair: ``powerlaw`` and ``dpl`` exist as names in *other*
    menus, so ``describe('dpl')`` confidently answered "Double power-law SFH"
    with an ``sfh={...}`` hint — a wrong answer does more damage than a
    ``KeyError``, and the ambiguity note could not fire because a name must be
    in two menus to be flagged as ambiguous.
    """
    from tengri.registry import _menu_listers

    homes = [ln.__name__ for ln in _menu_listers() if value in set(ln().names())]
    assert homes, f"radio {category} {value!r} is accepted by the builder but named by no menu"
    assert "list_radio_blocks" in homes

    assert dict(tengri.describe(value))["name"] == value
    assert value in set(tengri.search(value).names())


def _valid_shock_types():
    from tengri.parameters.groups import _VALID_SHOCK_TYPES

    assert _VALID_SHOCK_TYPES, "no shock types registered — this guard would pass vacuously"
    return sorted(_VALID_SHOCK_TYPES)


@pytest.mark.parametrize("shock_type", _valid_shock_types())
def test_shock_types_are_discoverable(shock_type):
    """``shock={'type': ...}`` must be reachable from the discovery API."""
    from tengri.registry import _menu_listers

    homes = [ln.__name__ for ln in _menu_listers() if shock_type in set(ln().names())]
    assert homes, f"shock {shock_type!r} is accepted by the builder but named by no list_* menu"
    assert "list_shock_models" in homes

    assert dict(tengri.describe(shock_type))["name"] == shock_type
    assert shock_type in set(tengri.search(shock_type).names())


def test_radio_and_shock_menus_cannot_drift_from_the_builder():
    """Both menus derive their names from the sets the validator checks.

    Hard-coding them here would let a menu advertise a variant
    ``SEDModel.build`` rejects, or hide one it accepts — the #1179 failure
    mode, in both directions.
    """
    from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS
    from tengri.parameters.groups import _VALID_SHOCK_TYPES

    assert set(tengri.list_radio_blocks(category="sf").names()) == set(SF_RADIO_MODELS)
    assert set(tengri.list_radio_blocks(category="agn").names()) == set(AGN_RADIO_MODELS)
    assert set(tengri.list_shock_models().names()) == set(_VALID_SHOCK_TYPES)


def test_radio_block_categories_are_kept_apart():
    """``sf`` and ``agn`` are different axes that share the name ``none``.

    Collapsing them would make ``list_radio_blocks(category='sf')`` advertise
    an AGN-only variant, whose ``use:`` hint would then name the wrong key.
    """
    sf = {r["name"] for r in tengri.list_radio_blocks(category="sf")}
    agn = {r["name"] for r in tengri.list_radio_blocks(category="agn")}
    assert "bell2003" in sf and "bell2003" not in agn
    assert "dpl" in agn and "dpl" not in sf

    for row in tengri.list_radio_blocks():
        assert f"'{row['category']}'" in row["use"], row["use"]


# ── the invariant itself: every axis, not just the ones already fixed ───────


def _grammar_value_sets():
    """Every set of values the build grammar accepts, discovered by introspection.

    The per-axis guards above pin the axes that were *already* found broken.
    That is not the same as pinning the rule, and the difference has cost real
    bugs: #1323 made ``xray_aird`` / ``agn_xray_corona`` / ``radio_powerlaw`` /
    ``radio_dpl`` builder-reachable, and a fully green suite said nothing,
    because no test knew those axes existed. ``met={'type': ...}``
    arrived with no menu at all and nine undiscoverable values for the same
    reason.

    So enumerate the validators rather than listing axes: every module-level
    ``_VALID_*`` collection and every zero-argument ``_valid_*()`` in
    ``parameters.groups``, plus the registry-backed axes that do not follow
    that naming. A new axis added under either convention is covered here the
    day it lands, with no edit to this file.
    """
    import inspect

    from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS
    from tengri.components.stellar.sfh.met_registry import MET_REGISTRY
    from tengri.parameters import groups as G

    found: dict[str, set[str]] = {}
    for attr in dir(G):
        obj = getattr(G, attr)
        if attr.startswith("_VALID_") and isinstance(obj, (set, frozenset)):
            values = obj
        elif (
            attr.startswith("_valid_") and callable(obj) and not inspect.signature(obj).parameters
        ):
            try:
                values = obj()
            except Exception:  # pragma: no cover - a validator needing context
                continue
        else:
            continue
        if values and all(isinstance(v, str) for v in values):
            found[attr] = set(values)

    # Axes validated against a component registry rather than a ``_valid_*``
    # helper, so introspection above cannot see them.
    found["MET_REGISTRY"] = set(MET_REGISTRY)
    found["SF_RADIO_MODELS"] = set(SF_RADIO_MODELS)
    found["AGN_RADIO_MODELS"] = set(AGN_RADIO_MODELS)

    # Anti-vacuity. If the naming convention changes under us, introspection
    # silently returns little or nothing and every assertion below passes for
    # the wrong reason — the failure mode this whole file keeps re-learning.
    assert len(found) >= 12, f"only discovered {sorted(found)} — introspection broke"
    for expected in ("_VALID_DUST_TYPES", "_valid_xray_types", "MET_REGISTRY"):
        assert expected in found, f"{expected} not discovered — introspection broke"
    return found


def _axis_value_pairs():
    return sorted(
        (axis, value) for axis, values in _grammar_value_sets().items() for value in values
    )


@pytest.mark.parametrize(("axis", "value"), _axis_value_pairs())
def test_every_grammar_value_is_named_by_some_menu(axis, value):
    """Anything ``SEDModel.build`` accepts must be findable before it is typed.

    A value the builder takes but no menu names is not merely undocumented: it
    is unreachable by the discovery path the docs teach. Worse, when the name
    also exists on another axis the lookup does not fail — it answers
    confidently about the wrong component (``describe('dpl')`` returned a
    *SFH* model; ``describe('table')`` returned the SFH table, not the
    metallicity mode).
    """
    from tengri.registry import _menu_listers

    homes = [ln.__name__ for ln in _menu_listers() if value in set(ln().names())]
    assert homes, f"{axis}={value!r} is accepted by the builder but named by no list_* menu"


def test_every_menu_derives_its_names_from_the_validator():
    """Menus that re-derive a validator's logic drift; menus that call it cannot.

    Each of these three shipped a docstring promising it could "never drift"
    from the grammar while re-implementing the derivation by hand. All three
    had drifted: the dust-emission copy could not see ``_LAZY_DUST_EMISSION_TYPES``
    (declared inside ``groups.py``), and the xray/radio copies read only the
    legacy ``*_MODELS`` dict while the validator accepts its union with the
    ``SEDModelComponent`` registry.
    """
    from tengri.parameters.groups import (
        _valid_dust_emission_types,
        _valid_radio_types,
        _valid_xray_types,
    )

    for menu, validator in (
        (tengri.list_xray_models, _valid_xray_types),
        (tengri.list_radio_models, _valid_radio_types),
        (tengri.list_dust_emission_models, _valid_dust_emission_types),
        (tengri.list_metallicity_modes, None),
    ):
        if validator is None:
            from tengri.components.stellar.sfh.met_registry import MET_REGISTRY

            assert set(menu().names()) == set(MET_REGISTRY)
            continue
        assert set(menu().names()) == set(validator()), menu.__name__


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
        tengri.list_radio_blocks,
        tengri.list_shock_models,
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
        sfh={"type": "dpl", "all_params": FIXED},
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
