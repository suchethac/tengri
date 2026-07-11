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


def test_every_emission_component_is_both_valid_and_advertised():
    """Reverse-direction guard: every registered component that publishes
    ``sed_dust_ir`` is a genuine selectable model and MUST appear in BOTH the
    grammar validator AND the advertised menu.

    The subset test above only guards ``listed ⊆ valid``, so it cannot catch a
    component model that is *selectable but hidden* (or advertised-but-rejected)
    — exactly the ``energy_balance_split`` drift, where the component was
    grammar-valid yet skipped by the menu. Aliases (dl07/mbb/…) and lazy names
    are validator-only by design and are not required to appear in the menu.
    """
    from tengri.components.sed_model_component import _REGISTRY

    dust_ir_components = {
        name
        for name, cls in _REGISTRY.items()
        # getattr default skips non-emission registry entries (dust-attenuation
        # screens registered for dispatch in #844 lack _outputs_tuple).
        if "sed_dust_ir" in {o.name for o in getattr(cls, "_outputs_tuple", ())}
    }
    valid = _valid_dust_emission_types()
    listed = {row["name"] for row in tengri.list_dust_emission_models()}

    assert dust_ir_components <= valid, (
        f"sed_dust_ir component models rejected by the validator: "
        f"{sorted(dust_ir_components - valid)}"
    )
    assert dust_ir_components <= listed, (
        "sed_dust_ir component models selectable but NOT advertised in the menu "
        f"(hidden-model drift): {sorted(dust_ir_components - listed)}"
    )


@pytest.mark.parametrize("alias", ["dl07", "dl14", "mbb"])
def test_friendly_aliases_resolvable(alias):
    """The short names surfaced by the menu must resolve in the registry."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    assert alias in DUST_EMISSION_MODELS, (
        f"Alias {alias!r} missing from DUST_EMISSION_MODELS — list/validator drift."
    )
