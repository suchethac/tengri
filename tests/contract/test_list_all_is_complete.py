# SPDX-License-Identifier: BSD-3-Clause
""":func:`tengri.list_all` means all — every public menu, or CI fails.

``list_all`` is advertised as the single notebook cell that shows the whole
code. It shipped returning nine of the twenty-one public ``list_*`` menus,
while its own docstring named six of those nine. Whichever of the three
surfaces a reader trusted, two were lying — and the eleven it omitted were not
minor: ``list_dust_models`` holds the ``dust={'type': ...}`` structural choice,
``list_metallicity_modes`` all ten ``stellar={'met_mode': ...}`` options, and
``list_agn_blocks`` forty-nine composable AGN blocks. An astronomer running the
advertised overview concluded tengri had no IGM, radio, X-ray or shock models
(#1724).

Enumerating the fix would rot the same way, so these tests pin the *rule* over a
discovered census: whatever ``tengri`` exports as a public ``list_*`` must appear
in ``list_all``, keyed by convention. Adding a twenty-second menu and forgetting
to wire it in fails here rather than quietly narrowing what "all" means.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.registry import _RegistryTable

pytestmark = pytest.mark.contract

#: Public ``list_*`` callables that are deliberately not menus.
#:
#: ``list_all`` is the aggregator itself. Nothing else is exempt: an entry here
#: is a claim that some public ``list_*`` is not a selectable menu, and it needs
#: the reason written beside it.
_NOT_A_MENU = frozenset({"list_all"})


def _public_menu_functions() -> dict[str, object]:
    """Every public ``list_*`` callable exported from the package."""
    return {
        name: fn
        for name in dir(tengri)
        if name.startswith("list_")
        and name not in _NOT_A_MENU
        and callable(fn := getattr(tengri, name))
    }


def test_census_is_not_empty() -> None:
    """Guard the guard — a census that finds nothing would pass every test below."""
    assert len(_public_menu_functions()) > 15


def test_every_public_menu_appears_in_list_all() -> None:
    """The promise: no public menu is missing from the overview."""
    expected = {name.removeprefix("list_") for name in _public_menu_functions()}
    missing = sorted(expected - set(tengri.list_all()))
    assert not missing, (
        f"list_all() omits {len(missing)} public menu(s): {missing}. "
        "Add the key to _ALL_MENUS in src/tengri/registry.py — it resolves to "
        "tengri.list_<key>, so there is nothing else to edit."
    )


def test_list_all_invents_no_keys() -> None:
    """The converse: every key traces back to a public ``list_*``."""
    expected = {name.removeprefix("list_") for name in _public_menu_functions()}
    extra = sorted(set(tengri.list_all()) - expected)
    assert not extra, f"list_all() returns keys with no public list_* behind them: {extra}"


def test_every_value_is_a_printable_table() -> None:
    """Each entry is a real table, not a bare list or a lazy callable.

    The overview is read by printing it, so a value that does not render as a
    table is as good as absent to the astronomer reading the cell.
    """
    for key, table in tengri.list_all().items():
        assert isinstance(table, _RegistryTable), f"{key!r} is {type(table).__name__}"


def test_menus_are_not_silently_empty() -> None:
    """A wired-but-empty menu reads as "tengri has none of these"."""
    empty = sorted(key for key, table in tengri.list_all().items() if len(table) == 0)
    assert not empty, f"menus wired into list_all() but returning no rows: {empty}"
