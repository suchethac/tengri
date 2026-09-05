# SPDX-License-Identifier: BSD-3-Clause
"""Behavioral single-dispatch contract for dust emission (ADR-0019).

Proves — at the level of the BUILT COMPONENT CHAIN, not just symbol absence —
that every grammar emission type (and alias) is handled by a registered
``EmissionComponent`` from ``_REGISTRY``, never by a legacy closure. A symbol rename
cannot fake this: it asserts the actual class that will run in the pipeline.

Runs DATA-FREE (no HDF5/SSP): ``build_components`` is structural, so this
enforces in CI where it matters (the #613 data-gated-skip anti-pattern).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

# Every dispatchable grammar emission type + every alias.
_GRAMMAR_TYPES = [
    "greybody",
    "modified_blackbody",
    "casey2012",
    "pah_drude",
    "schreiber2016",
    "schreiber2018",
    "dale2014",
    "dale2014_cigale",
    "draine_li2007",
    "draine_li2014",
    "astrodust",
    "bosa",
    "themis",
    "energy_balance_split",
    # aliases (resolved by _EMISSION_TYPE_ALIASES)
    "dl07",
    "dl07_tabulated",
    "dl14",
    "mbb",
    "draine2021_pah",
]


# The dust-IR re-emission output name(s) a registered emission component publishes.
# EmissionComponent components publish ``sed_dust_ir``; the pre-EmissionComponent
# ``draine2021_pah_ir`` publishes ``L_ir_emission`` (a pre-existing naming
# inconsistency, tracked separately — both are valid registry emission components).
_IR_OUTPUTS = {"sed_dust_ir", "L_ir_emission"}


class _MockSSP:
    ssp_wave = None


def _emission_components(dust_emission_model: str):
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import build_components

    components = build_components(
        ssp_data=_MockSSP(),
        dust_model="two_component",
        dust_emission_model=dust_emission_model,
        use_dust=True,
        nebular_backend=None,
        agn_model=None,
        use_radio=False,
        use_xray=False,
        use_igm=False,
    )
    registry_types = set(_REGISTRY.values())
    return [
        c
        for c in components
        if type(c) in registry_types and (_IR_OUTPUTS & {o.name for o in c.outputs()})
    ]


@pytest.mark.parametrize("emission_type", _GRAMMAR_TYPES)
def test_every_emission_type_dispatches_to_registry_component(emission_type):
    """Each grammar type builds exactly one registered _REGISTRY emission
    component."""
    components = _emission_components(emission_type)
    assert len(components) == 1, (
        f"emission type {emission_type!r} must build exactly one registered "
        f"_REGISTRY emission component; got {[type(c).__name__ for c in components]}"
    )


def test_no_legacy_dispatch_symbol_importable():
    """The legacy dispatch function must be gone (the guard's runtime mirror)."""
    import tengri.components.dust.emission as em

    assert not hasattr(em, "resolve_emission_model"), (
        "resolve_emission_model must be deleted — dispatch is single via _REGISTRY"
    )


def test_energy_balance_split_not_in_load_only_dict():
    """#850: ``energy_balance_split`` has a single dispatch path (the component).

    It must NOT also live in ``DUST_EMISSION_MODELS`` (the load-only loader
    cache). A duplicate entry there resurrects a divergent second path
    (``apply_dust_emission("energy_balance_split")`` -> raw closure with the
    default ``eta_balance``), which the component's ``eta_balance=1.0`` convention
    would silently disagree with. Its canonical dispatch is the
    ``EnergyBalanceSplitIRSEDComponent`` component in ``_REGISTRY``.
    """
    from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
    from tengri.components.sed_model_component import _REGISTRY

    assert "energy_balance_split" not in DUST_EMISSION_MODELS, (
        "energy_balance_split must NOT be registered in DUST_EMISSION_MODELS — "
        "its only dispatch is the _REGISTRY component (single-dispatch, #850)"
    )
    assert "energy_balance_split" in _REGISTRY, (
        "energy_balance_split must remain dispatchable via its _REGISTRY component"
    )
