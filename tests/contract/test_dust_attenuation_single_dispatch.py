# SPDX-License-Identifier: BSD-3-Clause
"""Behavioral single-dispatch contract for dust attenuation (ADR-0019, #844).

Proves — at the level of the BUILT COMPONENT CHAIN — that every ``dust_model``
grammar value is resolved through the ``_REGISTRY`` seam
(``_resolve_registry_component``), never a class hardcoded in
``build_components``. A symbol rename cannot fake this: it asserts the actual
attenuator class that will run in the pipeline is the one registered under that
type name.

Runs DATA-FREE (no HDF5/SSP): ``build_components`` is structural, so this
enforces in CI where it matters (the #613 data-gated-skip anti-pattern).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

# Grammar dust_model value -> the class it must resolve to via _REGISTRY.
_ATTENUATION_TYPES = {
    "two_component": "DustSEDComponent",
    "single_component": "DustAttenuationSEDComponent",
    "wg00": "WG00AttenuationSEDComponent",
}


class _MockSSP:
    ssp_wave = None


def _dust_chain(dust_model: str):
    """Build the component chain for a dust_model and return its attenuator(s).

    An attenuator is any built component whose class is registered in _REGISTRY
    under one of the three dust_model keys.
    """
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import build_components

    atten_classes = {_REGISTRY[k] for k in _ATTENUATION_TYPES}
    components = build_components(
        ssp_data=_MockSSP(),
        dust_model=dust_model,
        dust_emission_model=None,
        use_dust=True,
        nebular_backend=None,
        agn_model=None,
        use_radio=False,
        use_xray=False,
        use_igm=False,
    )
    return [c for c in components if type(c) in atten_classes]


@pytest.mark.parametrize("dust_model", sorted(_ATTENUATION_TYPES))
def test_dust_model_dispatches_to_registry_attenuator(dust_model):
    """Each dust_model builds exactly one attenuator, and it is the _REGISTRY
    class registered under that type name."""
    from tengri.components.sed_model_component import _REGISTRY

    attenuators = _dust_chain(dust_model)
    assert len(attenuators) == 1, (
        f"dust_model {dust_model!r} must build exactly one registered attenuator; "
        f"got {[type(a).__name__ for a in attenuators]}"
    )
    assert type(attenuators[0]) is _REGISTRY[dust_model], (
        f"dust_model {dust_model!r} built {type(attenuators[0]).__name__}, "
        f"expected _REGISTRY[{dust_model!r}] = {_REGISTRY[dust_model].__name__}"
    )
    assert type(attenuators[0]).__name__ == _ATTENUATION_TYPES[dust_model]


def test_attenuators_registered_under_type_names():
    """The three attenuator screens are registered in the unified _REGISTRY."""
    from tengri.components.sed_model_component import _REGISTRY

    for type_name, cls_name in _ATTENUATION_TYPES.items():
        assert type_name in _REGISTRY, f"{type_name!r} missing from _REGISTRY"
        assert _REGISTRY[type_name].__name__ == cls_name


def test_seam_construction_is_bit_identical_to_direct():
    """Physics-equivalence gate: the seam builds the SAME attenuator the old
    hardcoded path did.

    The convergence changes only *how the class is selected* (via _REGISTRY),
    not the class or its config. The attenuators are frozen dataclasses, so
    ``==`` compares class + every field; equality here means ``apply()`` is
    bit-identical between the seam-routed and directly-constructed components.
    (This is the "build both ways, assert bit-exact" gate for a dispatch-only
    refactor that deletes no physics.)
    """
    from tengri.components.dust.component import (
        DustAttenuationSEDComponent,
        DustAttenuationSEDComponentConfig,
    )
    from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig
    from tengri.components.dust.wg00_model import (
        WG00AttenuationSEDComponent,
        WG00AttenuationSEDComponentConfig,
    )
    from tengri.forward.component_factory import _resolve_registry_component

    cases = [
        (
            "two_component",
            DustSEDComponentConfig(law_bc="calzetti", law_diff="calzetti"),
            DustSEDComponent,
        ),
        (
            "single_component",
            DustAttenuationSEDComponentConfig(law="cardelli"),
            DustAttenuationSEDComponent,
        ),
        (
            "wg00",
            WG00AttenuationSEDComponentConfig(dust_curve="smc", geometry="dusty"),
            WG00AttenuationSEDComponent,
        ),
    ]
    for type_name, cfg, cls in cases:
        seam = _resolve_registry_component("dust_attenuation", type_name, config=cfg)
        direct = cls(config=cfg)
        assert seam == direct, (
            f"seam-built {type_name!r} attenuator differs from direct construction "
            f"(dispatch refactor must be bit-identical)"
        )


def test_no_legacy_attenuator_class_in_component_factory():
    """build_components must not import the attenuator CLASSES directly — it
    resolves them from _REGISTRY via the seam (the single-dispatch invariant
    mirrored by tools/check_single_dispatch.py). Only the config dataclasses
    remain imported."""
    import tengri.forward.component_factory as cf

    for cls_name in _ATTENUATION_TYPES.values():
        assert not hasattr(cf, cls_name), (
            f"component_factory must not directly reference {cls_name}; "
            f"dispatch is single via _REGISTRY (#844)"
        )
    # config dataclasses are still needed for parameterization
    assert hasattr(cf, "DustSEDComponentConfig")
    assert hasattr(cf, "WG00AttenuationSEDComponentConfig")
