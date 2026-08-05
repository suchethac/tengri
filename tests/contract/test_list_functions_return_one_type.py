# SPDX-License-Identifier: BSD-3-Clause
"""Every ``list_*`` returns the same type (#1285).

``list_*`` is the discovery surface, and it used to return four different
things depending on which one you called:

    16 of 21                    _RegistryTable
    list_parameters             list[str]        (350)
    list_known_ssps             dict[str, str]   (21)
    list_filter_conventions     dict[str, str]   (2)
    list_available_ssps         list[dict]       (21)

Same verb, same intent, four shapes for the caller to special-case. The cost
was not only ergonomic: ``list_parameters`` returned bare strings, so the 350
parameters #1264 made discoverable arrived with no description, units or owner
attached — even though the registry stores all three.

``list_all`` is exempt: it returns a mapping *of* menus, which is a different
thing from a menu.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.registry import _RegistryTable

pytestmark = pytest.mark.contract

#: ``list_all`` returns dict[str, _RegistryTable] by design — a map of menus.
EXEMPT = {"list_all"}

LISTERS = sorted(n for n in tengri.__all__ if n.startswith("list_") and n not in EXEMPT)

#: The four that used to be outliers, pinned by name so a revert is loud.
FORMER_OUTLIERS = [
    "list_parameters",
    "list_known_ssps",
    "list_filter_conventions",
    "list_available_ssps",
]


def test_the_census_is_not_empty():
    """Guard the guard: an empty LISTERS would make everything below vacuous."""
    assert len(LISTERS) >= 18, f"only {len(LISTERS)} list_* found — census has rotted"
    for name in FORMER_OUTLIERS:
        assert name in LISTERS, f"{name} vanished from __all__"


@pytest.mark.parametrize("name", LISTERS)
def test_every_lister_returns_a_registry_table(name):
    result = getattr(tengri, name)()
    assert isinstance(result, _RegistryTable), (
        f"{name}() returns {type(result).__name__}, not _RegistryTable. "
        "Every discovery verb must return the same type."
    )


@pytest.mark.parametrize("name", LISTERS)
def test_every_row_has_a_name(name):
    """``.names()`` and ``.to_dict()`` both key on it, so it must be present."""
    result = getattr(tengri, name)()
    if not result:
        pytest.skip(f"{name}() is empty in this build")
    missing = [i for i, row in enumerate(result) if "name" not in row]
    assert not missing, f"{name}() rows {missing[:5]} have no 'name' key"


def test_list_all_is_still_a_map_of_tables():
    """The one deliberate exception, pinned so it is a choice not a leftover."""
    result = tengri.list_all()
    assert isinstance(result, dict)
    assert all(isinstance(v, _RegistryTable) for v in result.values())


# ── the migration accessors ────────────────────────────────────────────────


def test_names_gives_back_the_old_list_parameters_shape():
    names = tengri.list_parameters().names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert names == sorted(names), "list_parameters() must stay sorted"
    assert len(names) > 300, "the registry lost parameters"


def test_list_parameters_rows_carry_the_metadata_that_was_being_discarded():
    """The reason the change is worth its blast radius."""
    rows = tengri.list_parameters(prefix="dust_")
    assert rows, "no dust_ parameters"
    assert set(rows[0]) >= {"name", "description", "units", "owner"}
    described = [r for r in rows if r["description"]]
    assert described, "no dust_ parameter carries a description"


def test_to_dict_gives_back_the_old_mapping_shapes():
    ssps = tengri.list_known_ssps().to_dict("filename")
    assert isinstance(ssps, dict)
    assert ssps["fsps_prsc_miles_chabrier"] == "fsps_prsc_miles_chabrier.h5"

    conv = tengri.list_filter_conventions().to_dict()
    assert isinstance(conv, dict)
    assert "bessell" in conv and "energy" in conv


def test_to_dict_on_an_unknown_column_raises():
    """Returning None values would read as an empty catalog, not a typo."""
    with pytest.raises(KeyError, match="not a column"):
        tengri.list_known_ssps().to_dict("no_such_column")


def test_prefix_filtering_still_works():
    rows = tengri.list_parameters(prefix="radio_")
    assert rows and all(r["name"].startswith("radio_") for r in rows)
    assert tengri.list_parameters(prefix="definitely_not_a_prefix_").names() == []
