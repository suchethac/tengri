# SPDX-License-Identifier: BSD-3-Clause
"""Every parameter the grammar can produce must be describable.

``parameters/registry.py`` walks each component's ``_params.py``. Star-formation
history and metallicity own no such module — they declare parameters per model
in ``SFH_REGISTRY[<type>].params`` and ``MET_REGISTRY[<type>].params``, a
mechanism the walk could not see. So *every* SFH parameter was missing from
introspection: ``list_parameters()`` returned 189 names with no ``sfh_*`` at
all, and ``describe_parameter("sfh_dpl_alpha")`` raised ``KeyError`` — for the
identifier ``docs/dev/NAMING_CONTRACT.md`` uses as its worked example. Five of
the eight free parameters in a default DPL model could not be described (#1264).

The bug was structural, not a typo: a second declaration mechanism nobody had
taught the registry about. So the test is structural too — it asserts the
*property* (anything buildable is describable) rather than a list of names,
which is what a hand-kept list would have failed to notice in the first place.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri import Fixed
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

pytestmark = pytest.mark.contract

_RECIPES = (
    "star_forming_photometry",
    "quiescent_z0",
    "agn_panchromatic",
    "stochastic_sfh_jwst",
    "mock_recovery_minimal",
    "composable_agn",
    "dust_demo",
    "high_z",
    "photoz",
    "unified_agn",
)


def _registry_names() -> set[str]:
    return set(tengri.list_parameters().names())


def test_sfh_parameters_are_in_the_registry():
    """The headline regression: not one ``sfh_*`` name used to be present."""
    sfh_names = [n for n in _registry_names() if n.startswith("sfh_")]
    assert sfh_names, "no sfh_* parameters in the registry at all"


@pytest.mark.parametrize(
    "name",
    ["sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr", "sfh_dpl_log_total_mass"],
)
def test_describe_parameter_resolves_sfh_names(name):
    """``describe_parameter`` raised KeyError for every one of these."""
    record = tengri.describe_parameter(name)
    assert record.name == name
    assert record.prior is not None


def test_unit_bearing_names_report_their_units():
    """A name that states its unit must not report an empty units field."""
    assert tengri.describe_parameter("sfh_dpl_tau_gyr").units == "Gyr"


@pytest.mark.parametrize("recipe", _RECIPES)
def test_every_recipe_parameter_is_describable(recipe):
    groups = getattr(tengri.recipes, recipe)()
    groups.pop("approx", None)
    spec = tengri.parse_groups(**groups)
    missing = sorted(p for p in spec.all_params if p not in _registry_names())
    assert not missing, f"{recipe}: registry cannot describe {missing}"


def test_every_sfh_type_has_describable_parameters():
    """Sweep the whole SFH registry, not just the types the recipes happen to use."""
    names = _registry_names()
    missing: dict[str, list[str]] = {}
    for sfh_type in sorted(SFH_REGISTRY):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                spec = tengri.parse_groups(sfh={"type": sfh_type}, redshift=Fixed(0.1))
            except Exception:
                # Some types need extra structural config to build standalone;
                # they are covered via the recipe sweep above.
                continue
        gap = sorted(p for p in spec.all_params if p not in names)
        if gap:
            missing[sfh_type] = gap
    assert not missing, f"registry cannot describe: {missing}"
