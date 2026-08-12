# SPDX-License-Identifier: BSD-3-Clause
"""``describe`` and ``search`` must cover every menu, and not by a hand-written list.

``_menu_listers`` exists so ``describe``/``search``/``list_all`` cannot fall out
of sync when a menu is added. Its docstring says so, and names the two times
they did (#1120, #1446). **It drifted again**, because a guard against a
hand-written list that is itself a hand-written list guards nothing:

* ``list_instruments`` and ``list_known_ssps`` were never added, so
  ``describe('GALEX')`` answered ``Unknown name 'GALEX'`` — for a name
  ``list_instruments()`` prints one line earlier. 30 of 490 advertised names.
* ``search`` built its own union at its own call site and left out
  ``list_recipes``, so **all ten recipes returned zero hits** while
  ``describe`` resolved every one.

That is three hand-written enumerations of "every menu": the tuple, and one
union per call site.

The set is now discovered — every public ``tengri.list_*`` returning rows with
a ``name`` column. Measured at the swap: +2 menus, 0 lost, 460 → 490 rows
walked, **0 new multi-menu names**, so no existing lookup changed its answer.

These tests scan the same way rather than pinning a count, so a menu added
tomorrow is covered without editing this file — which is the whole point.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri.registry import _every_menu_lister, _menu_listers

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _advertised() -> dict[str, str]:
    """{name: menu} for every row any public ``list_*`` menu prints."""
    out: dict[str, str] = {}
    for attr in sorted(n for n in dir(tengri) if n.startswith("list_")):
        fn = getattr(tengri, attr, None)
        if not callable(fn):
            continue
        try:
            rows = fn()
        except Exception:
            continue
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            for row in rows:
                if row.get("name"):
                    out.setdefault(str(row["name"]), attr)
    return out


class TestTheCensus:
    def test_there_are_menus_to_cover(self):
        """A scan finding nothing would pass every test below vacuously."""
        advertised = _advertised()
        assert len(advertised) > 300, (
            f"only {len(advertised)} advertised names found — the scan broke, not the API."
        )

    def test_the_derived_set_is_a_superset_of_the_hand_written_one(self):
        """Deriving must not silently drop a menu someone listed on purpose."""
        derived = {fn.__name__ for fn in _every_menu_lister()}
        hand = {fn.__name__ for fn in _menu_listers()}
        assert hand <= derived, f"deriving lost these menus: {sorted(hand - derived)}"

    def test_the_derived_set_found_menus_the_hand_written_one_missed(self):
        """The converse — otherwise this change was a no-op dressed as a fix."""
        derived = {fn.__name__ for fn in _every_menu_lister()}
        hand = {fn.__name__ for fn in _menu_listers()}
        assert derived - hand, (
            "the derived set adds nothing over the hand-written tuple; either "
            "every menu is now listed by hand (then delete the derivation) or "
            "the discovery predicate stopped matching."
        )

    def test_no_lister_is_walked_twice(self):
        """Two call sites unioned by hand; a duplicate makes describe() report
        a name as living in two menus when it lives in one."""
        names = [fn.__name__ for fn in _every_menu_lister()]
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, f"these listers appear more than once: {dupes}"

    def test_discovery_is_paid_once_not_per_lookup(self):
        """Discovery *calls* every ``list_*`` to check its shape.

        Uncached that was 95 ms — 49% of a 193 ms ``describe()`` — charged
        again on every lookup, which doubled the cost of the surface this
        change exists to improve. Asserted structurally rather than by
        wall-clock: a timing threshold on a 100 ms call is a flake generator,
        and the property that matters is "computed once", not "fast today".
        """
        assert hasattr(_every_menu_lister, "cache_info"), (
            "_every_menu_lister is no longer cached; every describe() and "
            "search() call would re-probe every menu."
        )
        _every_menu_lister()
        before = _every_menu_lister.cache_info().misses
        for _ in range(5):
            _every_menu_lister()
        assert _every_menu_lister.cache_info().misses == before, (
            "repeated calls re-ran discovery, so the cache is not effective."
        )


class TestDescribeCoversEveryMenu:
    def test_every_advertised_name_resolves(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refused = {name: menu for name, menu in _advertised().items() if not _resolves(name)}
        assert not refused, (
            f"{len(refused)} names are printed by a menu and refused by "
            f"describe(): {sorted(refused.items())[:8]}. describe() documents "
            f"itself as 'universal lookup across every menu'."
        )

    @pytest.mark.parametrize("name", ["GALEX", "fsps_prsc_miles_chabrier"])
    def test_the_two_menus_that_were_missing(self, name):
        """Named explicitly so a regression says which menu came loose."""
        assert _resolves(name), f"describe({name!r}) fails; its menu fell out of the derived set."


class TestSearchCoversEveryMenu:
    def test_every_recipe_finds_itself(self):
        """All ten returned zero hits before the derived set."""
        missing = []
        for row in tengri.list_recipes():
            name = row["name"]
            hits = tengri.search(name)
            if not any(str(r.get("name")) == name for r in hits if isinstance(r, dict)):
                missing.append(name)
        assert not missing, f"search() cannot find these recipes: {missing}"

    @pytest.mark.parametrize("name", ["GALEX", "photoz"])
    def test_a_name_from_each_newly_reachable_menu_is_searchable(self, name):
        hits = tengri.search(name)
        assert any(str(r.get("name")) == name for r in hits if isinstance(r, dict)), (
            f"search({name!r}) does not find it"
        )


def _resolves(name: str) -> bool:
    try:
        tengri.describe(name)
    except Exception:
        return False
    return True
