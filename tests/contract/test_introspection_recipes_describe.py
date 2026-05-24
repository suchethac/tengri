# SPDX-License-Identifier: BSD-3-Clause
"""Contract: list_recipes / describe_recipe / describe_* introspection (#310).

The introspection surface for recipes and per-kind describe accessors
landed for #310 proposals 1 + 2. These tests pin:

- Every public recipe in :mod:`tengri.recipes.__all__` appears in
  ``list_recipes()`` with a non-empty short_doc.
- ``describe_recipe(name)`` returns a usable block for every listed recipe
  and a clear KeyError for unknown names.
- ``describe_agn_model``, ``describe_dust_law``, ``describe_dust_emission_model``,
  ``describe_sfh_model``, ``describe_nebular_backend``, and
  ``describe_inference_method`` are symmetric with their ``list_*`` siblings.
"""

from __future__ import annotations

import pytest

import tengri

pytestmark = pytest.mark.contract


def test_list_recipes_covers_module_all():
    listed = {entry["name"] for entry in tengri.list_recipes()}
    declared = set(tengri.recipes.__all__)
    assert listed == declared, (
        f"list_recipes() mismatched recipes.__all__: only-in-listed={listed - declared}, "
        f"only-in-all={declared - listed}"
    )


def test_list_recipes_short_doc_non_empty():
    for entry in tengri.list_recipes():
        assert entry["short_doc"], f"recipe {entry['name']!r} has empty short_doc"


@pytest.mark.parametrize("recipe_name", sorted(tengri.recipes.__all__))
def test_describe_recipe_round_trip(recipe_name: str):
    record = tengri.describe_recipe(recipe_name)
    assert record["name"] == recipe_name
    assert record["kind"] == "recipe"
    assert record["docstring"]
    assert "tengri.SEDModel.build" in record["use"]


def test_describe_recipe_unknown_raises_keyerror_with_candidates():
    with pytest.raises(KeyError, match="Unknown recipe"):
        tengri.describe_recipe("bogus_not_a_recipe")


@pytest.mark.parametrize(
    ("list_fn", "describe_fn", "kind"),
    [
        (tengri.list_agn_models, tengri.describe_agn_model, "AGN model"),
        (tengri.list_dust_laws, tengri.describe_dust_law, "dust law"),
        (
            tengri.list_dust_emission_models,
            tengri.describe_dust_emission_model,
            "dust emission",
        ),
        (tengri.list_sfh_models, tengri.describe_sfh_model, "SFH"),
        (tengri.list_nebular_backends, tengri.describe_nebular_backend, "nebular"),
        (
            tengri.list_inference_methods,
            tengri.describe_inference_method,
            "inference",
        ),
    ],
)
def test_describe_fn_symmetric_with_list_fn(list_fn, describe_fn, kind):
    entries = list(list_fn())
    assert entries, f"list_fn for {kind} returned empty"
    # Roundtrip the first entry — must return a dict whose name matches.
    first = entries[0]
    record = describe_fn(first["name"])
    assert record["name"] == first["name"]


@pytest.mark.parametrize(
    "describe_fn",
    [
        tengri.describe_agn_model,
        tengri.describe_dust_law,
        tengri.describe_dust_emission_model,
        tengri.describe_sfh_model,
        tengri.describe_nebular_backend,
        tengri.describe_inference_method,
    ],
)
def test_describe_fn_unknown_raises_keyerror_with_menu(describe_fn):
    with pytest.raises(KeyError, match="Known names"):
        describe_fn("not_a_real_name_xyz_123")
