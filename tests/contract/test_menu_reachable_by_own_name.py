# SPDX-License-Identifier: BSD-3-Clause
"""Every menu must answer to its own name.

``search()`` carried hand-written concept synonyms — ``"extinction"`` reaches
:func:`list_dust_laws`, ``"dust emission"`` reaches the dust-emission menu — but
nothing mapped a menu's *own name* onto it. So the surface was incoherent in a
way no user could predict: ``search('dust emission')`` returned 18 rows and
``search('dust emission models')`` returned zero, because adding the noun the
menu is literally called fell off the end of the alias table.

Measured before the fix, 13 of 14 menus were unreachable by their own name:
``dust laws``, ``sfh models``, ``nebular backends``, ``age kernels``,
``inference methods`` — all zero hits.

This is the #1179 invariant (*anything the builder accepts must be findable
before it is typed*) applied to the menus themselves rather than to their
values. The fix derives the aliases from :func:`_menu_listers`, so this test and
the implementation share one source of truth and a new menu is covered the day
it is registered — the drift that hand-written copies caused in #1120 and #1446
cannot recur here.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.registry import _menu_listers, _menu_name_aliases

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

#: The library has 14 menus; well below that the parametrization has collapsed
#: and every assertion below would pass vacuously.
_MIN_MENUS = 10


def _menu_names() -> list[tuple[str, str]]:
    """``(list_fn_name, prose_name)`` for both the plural and singular forms."""
    out: list[tuple[str, str]] = []
    for fn in _menu_listers():
        stem = fn.__name__.removeprefix("list_").replace("_", " ")
        out.append((fn.__name__, stem))
        if stem.endswith("s"):
            out.append((fn.__name__, stem[:-1]))
    return out


def test_the_parametrization_is_not_vacuous():
    assert len(_menu_listers()) >= _MIN_MENUS, (
        f"only {len(_menu_listers())} menus discovered — the guard below is vacuous"
    )


@pytest.mark.parametrize("lister_name,prose", _menu_names())
def test_searching_a_menus_own_name_returns_that_menu(lister_name, prose):
    lister = getattr(tengri, lister_name)
    expected = lister()
    got = tengri.search(prose)
    assert len(got) > 0, (
        f"search({prose!r}) returns nothing, but {lister_name}() is a menu with "
        f"{len(expected)} rows — the menu cannot be found by the name it is called"
    )
    assert [r["name"] for r in got] == [r["name"] for r in expected], (
        f"search({prose!r}) did not redirect to {lister_name}()"
    )


def test_every_menu_is_covered_by_the_derived_alias_table():
    """The table is derived, so a new menu must not need a hand edit."""
    aliases = _menu_name_aliases()
    for fn in _menu_listers():
        stem = fn.__name__.removeprefix("list_").replace("_", " ")
        assert stem in aliases, f"{fn.__name__} has no derived alias for {stem!r}"


class TestHyphenSpellingsResolve:
    """A user types ``x-ray``; the menu is called ``xray``."""

    def test_hyphenated_query_reaches_the_menu(self):
        assert [r["name"] for r in tengri.search("x-ray models")] == [
            r["name"] for r in tengri.list_xray_models()
        ]

    def test_existing_hyphenated_concept_alias_still_works(self):
        """``star-forming`` predates this change and must not regress."""
        assert len(tengri.search("star-forming")) == len(tengri.list_sfh_models())


class TestConceptSynonymsStillWin:
    """Hand-curated intent beats a derived name on a collision."""

    def test_extinction_still_reaches_dust_laws(self):
        assert [r["name"] for r in tengri.search("extinction")] == [
            r["name"] for r in tengri.list_dust_laws()
        ]

    def test_a_non_menu_query_still_falls_through_to_substring_search(self):
        """The redirect must not swallow ordinary searches."""
        hits = tengri.search("torus")
        assert len(hits) > 1
        assert len(hits) < len(tengri.list_agn_blocks())
