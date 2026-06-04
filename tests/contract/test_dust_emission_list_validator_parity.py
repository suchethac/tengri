# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for #495: list helper ⊆ validator accepted set.

Mirrors the AGN registry-drift fix in PR #489 — every name surfaced by
:func:`tengri.list_dust_emission_models` must be a buildable
``dust.emission.type`` value (validated by
:func:`tengri.parameters.groups._valid_dust_emission_types`). Hand-maintained
allowlists drift; deriving from the live registry prevents that.
"""

import pytest

import tengri
from tengri.parameters.groups import _valid_dust_emission_types

pytestmark = pytest.mark.contract


def test_list_dust_emission_models_subset_of_validator():
    listed = {row["name"] for row in tengri.list_dust_emission_models()}
    valid = _valid_dust_emission_types()
    missing = listed - valid
    assert not missing, (
        f"list_dust_emission_models() advertises names the build validator rejects: {missing}. "
        "This is the registry-drift footgun from #495."
    )


@pytest.mark.parametrize("alias", ["dl07", "dl14", "mbb"])
def test_friendly_aliases_resolvable(alias):
    """The short names surfaced by the menu must resolve in the registry."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    assert alias in DUST_EMISSION_MODELS, (
        f"Alias {alias!r} missing from DUST_EMISSION_MODELS — list/validator drift."
    )
