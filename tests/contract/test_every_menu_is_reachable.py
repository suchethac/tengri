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

**Deriving the set was necessary and not sufficient**, because a derivation is
only as wide as the population it scans. Discovery scanned ``dir(tengri)``,
which is curated down to ~30 obvious entry points on purpose, so it inherited
the curation — and the helper in this file scanned ``dir(tengri)`` too, so the
guard and the code shared one blind spot and agreed with each other. Four
name-keyed menus live in ``__all__`` and not in the curated list, and stayed
invisible to both: ``list_parameters`` (358 names), ``list_properties`` (50),
``list_filter_conventions`` (2) and ``list_available_ssps``. That is **410
further advertised names** ``describe()`` refused while calling itself
universal, and three menus ``search()`` returned zero hits from.

The third aggregator was never wired up at all: ``list_all`` walked a
hand-written dict of nine literals and showed **9 of 25** menus.

These tests scan the same way rather than pinning a count, so a menu added
tomorrow is covered without editing this file — which is the whole point. The
population they scan is the union of *both* export lists, per #1608.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri.registry import _every_menu_lister, _menu_listers

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _public_menu_names() -> set[str]:
    """Every public ``list_*`` name, across **both** of tengri's export lists.

    This is the population, and getting it wrong is the whole bug twice over.
    ``dir(tengri)`` is not the public surface: it is a deliberately curated
    ~30-name tab-completion list ("not the 175-item kitchen sink of every
    public symbol", ``src/tengri/__init__.py``). ``__all__`` is the export
    list. Neither contains the other, so any name-based audit of this repo
    must union them — the rule #1608 established after the same blind spot
    made ``check_api_coverage.py`` report 0 missing when 6 were.

    Sweeping ``dir()`` alone hides four name-keyed menus that are in
    ``__all__`` and not curated: ``list_parameters``, ``list_properties``,
    ``list_filter_conventions``, ``list_available_ssps``.
    """
    surface = set(tengri.__all__) | set(dir(tengri))
    return {n for n in surface if n.startswith("list_")}


def _menu_rows() -> dict[str, list[dict]]:
    """{menu: rows} for every public ``list_*`` that prints a name column."""
    out: dict[str, list[dict]] = {}
    for attr in sorted(_public_menu_names()):
        fn = getattr(tengri, attr, None)
        if not callable(fn):
            continue
        try:
            rows = fn()
        except Exception:
            continue
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "name" in rows[0]:
            out[attr] = rows
    return out


def _advertised() -> dict[str, str]:
    """{name: menu} for every row any public ``list_*`` menu prints."""
    out: dict[str, str] = {}
    for attr, rows in _menu_rows().items():
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

    def test_the_population_swept_is_the_export_surface(self):
        """Deriving the set fixed *how* menus are found; this fixes *where*.

        Discovery scanned ``dir(tengri)``, which is curated down to ~30 entry
        points on purpose. So the derivation inherited the curation and four
        name-keyed menus in ``__all__`` stayed invisible — 410 more advertised
        names refused, ``list_parameters`` alone accounting for 358.
        """
        derived = {fn.__name__ for fn in _every_menu_lister()}
        missing = sorted(set(_menu_rows()) - derived)
        assert not missing, (
            f"these public name-keyed menus are not in the census: {missing}. "
            "The sweep is reading one export list; tengri has two."
        )

    def test_the_two_export_lists_really_do_differ(self):
        """Anti-vacuity for the test above.

        If ``dir()`` and ``__all__`` ever agreed, that test would pass without
        exercising the union and would stop guarding anything. It would then
        be the *census* that had quietly narrowed, which is this file's whole
        subject, so say so out loud rather than going green.
        """
        curated = {n for n in dir(tengri) if n.startswith("list_")}
        exported = {n for n in tengri.__all__ if n.startswith("list_")}
        assert exported - curated, (
            "every exported list_* is now curated into dir(); the union above "
            "no longer proves anything. Re-derive the population before "
            "trusting these tests."
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


class TestListAllCoversEveryMenu:
    """The third aggregator, and the one that never walked the tuple at all.

    ``_menu_listers``'s docstring names ``list_all`` as one of the three that
    "all walk this one tuple". It walks a hand-written dict of nine literals,
    so it showed 9 of 25 menus while ``src/tengri/__init__.py`` tells readers
    ``list_all()`` "enumerates every registry live".
    """

    def test_every_menu_has_a_key(self):
        keys = set(tengri.list_all())
        missing = sorted(
            fn.__name__
            for fn in _every_menu_lister()
            if fn.__name__.removeprefix("list_") not in keys
        )
        assert not missing, (
            f"list_all() omits {len(missing)} menus the census walks: {missing}. "
            "It documents itself as enumerating every registry live."
        )

    def test_every_key_is_a_table(self):
        """A dict of tables is the documented return type; keep it one type."""
        wrong = {
            key: type(value).__name__
            for key, value in tengri.list_all().items()
            if not isinstance(value, list)
        }
        assert not wrong, f"these list_all() values are not tables: {wrong}"

    def test_the_keys_that_were_always_there_did_not_move(self):
        """Widening must not rename what nine years of notebooks already read."""
        keys = set(tengri.list_all())
        original = {
            "components",
            "inference_methods",
            "agn_models",
            "dust_laws",
            "dust_emission_models",
            "sfh_models",
            "nebular_backends",
            "filters",
            "plots",
        }
        assert original <= keys, f"list_all() dropped established keys: {sorted(original - keys)}"


def _resolves(name: str) -> bool:
    try:
        tengri.describe(name)
    except Exception:
        return False
    return True
