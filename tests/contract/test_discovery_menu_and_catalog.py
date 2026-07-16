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


# ── list_sfh_models() hides unbuildable types by default ────────────────────


def test_list_sfh_models_excludes_unvalidated():
    listed = set(tengri.list_sfh_models().names())
    assert UNVALIDATED_SFH_TYPES.isdisjoint(listed), (
        "list_sfh_models() must not advertise SFH types the builder refuses; "
        f"leaked: {sorted(UNVALIDATED_SFH_TYPES & listed)}"
    )


def test_list_sfh_models_include_unvalidated_flag():
    with_unval = set(tengri.list_sfh_models(include_unvalidated=True).names())
    assert UNVALIDATED_SFH_TYPES.issubset(with_unval)


# ── usage hints teach the current SEDModel.build grammar ────────────────────


@pytest.mark.parametrize(
    "lister",
    [
        tengri.list_sfh_models,
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
