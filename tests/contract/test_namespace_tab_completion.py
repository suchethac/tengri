# SPDX-License-Identifier: BSD-3-Clause
"""``builders.*`` and ``recipes`` offer only what they mean to offer (#1288).

Every one of these namespaces already declared a correct ``__all__``. But
``__all__`` governs ``from x import *``; it does not filter ``dir()``, and
``dir()`` is what drives tab-completion. So the modules' own imports showed up
as completions beside the physics:

    builders.agn   leaks=['Any','Callable','annotations','make_factory',
                          'recipe_parameters','short_form']

``builders.annotations`` is the ``__future__`` feature object. At the top level
it was one of eight completions — 12% of what a user saw on
``tengri.builders.<TAB>``.

The fix is a module-level ``__dir__`` returning ``__all__``. These tests hold
the two in agreement, so a new import cannot silently re-open the namespace.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.contract

#: Every namespace whose tab-completion surface is contractual.
NAMESPACES = [
    "tengri.builders",
    "tengri.builders.agn",
    "tengri.builders.dust",
    "tengri.builders.igm",
    "tengri.builders.neb",
    "tengri.builders.radio",
    "tengri.builders.sfh",
    "tengri.builders.xray",
    "tengri.recipes",
]

#: Names that must never be completable — typing constructs and internals.
FORBIDDEN = {
    "Any",
    "Callable",
    "Mapping",
    "Sequence",
    "annotations",
    "make_factory",
    "short_form",
    "recipe_parameters",
    "Distribution",
}


@pytest.mark.parametrize("modname", NAMESPACES)
def test_dir_equals_all(modname):
    """Tab-completion and star-import must offer the same names."""
    mod = importlib.import_module(modname)
    public = {n for n in dir(mod) if not n.startswith("_")}
    declared = set(mod.__all__)
    assert public == declared, (
        f"{modname}: dir() and __all__ disagree.\n"
        f"  completable but not exported: {sorted(public - declared)}\n"
        f"  exported but not completable: {sorted(declared - public)}\n"
        "Add the name to __all__, or stop importing it at module scope."
    )


@pytest.mark.parametrize("modname", NAMESPACES)
def test_no_typing_constructs_leak(modname):
    """The specific names that were leaking, asserted by name.

    ``test_dir_equals_all`` would catch these too, but only while ``__all__``
    stays honest. This pins the actual regression.
    """
    mod = importlib.import_module(modname)
    leaked = sorted(FORBIDDEN & {n for n in dir(mod) if not n.startswith("_")})
    assert not leaked, f"{modname} offers implementation details in dir(): {leaked}"


def test_every_namespace_is_actually_populated():
    """Guard the guard: an empty ``__all__`` would satisfy both tests above."""
    for modname in NAMESPACES:
        mod = importlib.import_module(modname)
        assert len(mod.__all__) >= 4, (
            f"{modname}.__all__ has only {len(mod.__all__)} names — if a "
            "namespace emptied out, the equality tests above prove nothing."
        )


def test_recipes_offers_exactly_the_listed_recipes():
    """``tengri.recipes.<TAB>`` and ``list_recipes()`` must agree.

    Before #1288 the namespace additionally offered ``Fixed``, ``Uniform`` and
    ``WavePrecomp`` — re-exported constructors, not recipes.
    """
    import tengri
    import tengri.recipes as R

    listed = {row["name"] for row in tengri.list_recipes()}
    completable = {n for n in dir(R) if not n.startswith("_")}
    assert completable == listed, (
        f"recipes namespace and list_recipes() disagree.\n"
        f"  completable but unlisted: {sorted(completable - listed)}\n"
        f"  listed but not completable: {sorted(listed - completable)}"
    )
