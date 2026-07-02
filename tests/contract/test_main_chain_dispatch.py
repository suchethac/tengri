# SPDX-License-Identifier: BSD-3-Clause
"""Single-dispatch contract for the radio / X-ray / IGM main-chain components (#845).

These three flag-gated components (``use_radio`` / ``use_xray`` / ``use_igm``)
now resolve through the ``_REGISTRY`` seam (``_resolve_registry_component``)
rather than being hardcoded in ``build_components``. This is a construction-only
convergence: the components' ``apply()`` physics (and their cross-component
reads — L_ir / L_agn_bol / log_mstar / sfr) are unchanged.

Data-free: ``build_components`` is structural.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

# domain/type name -> class name registered under it
_MAIN_CHAIN = {
    "radio": "RadioSEDComponent",
    "xray": "XRaySEDComponent",
    "igm": "IGMSEDComponent",
}


class _MockSSP:
    ssp_wave = None


def _build(**flags):
    from tengri.forward.component_factory import build_components

    return build_components(
        ssp_data=_MockSSP(),
        dust_model="two_component",
        dust_emission_model=None,
        use_dust=False,
        nebular_backend=None,
        agn_model=None,
        use_radio=flags.get("use_radio", False),
        use_xray=flags.get("use_xray", False),
        use_igm=flags.get("use_igm", False),
    )


@pytest.mark.parametrize(
    "name,flag", [("radio", "use_radio"), ("xray", "use_xray"), ("igm", "use_igm")]
)
def test_flag_builds_registry_component(name, flag):
    """Enabling the flag builds exactly the component registered under that name."""
    from tengri.components.sed_model_component import _REGISTRY

    comps = _build(**{flag: True})
    matches = [c for c in comps if type(c) is _REGISTRY[name]]
    assert len(matches) == 1, (
        f"{flag}=True must build exactly one {_MAIN_CHAIN[name]} from _REGISTRY; "
        f"chain = {[type(c).__name__ for c in comps]}"
    )
    assert type(matches[0]).__name__ == _MAIN_CHAIN[name]


def test_registered_under_names():
    from tengri.components.sed_model_component import _REGISTRY

    for name, cls_name in _MAIN_CHAIN.items():
        assert name in _REGISTRY and _REGISTRY[name].__name__ == cls_name


def test_seam_construction_is_bit_identical_to_direct():
    """Physics-equivalence gate: the seam builds the SAME component the old
    hardcoded path did (frozen-dataclass equality => apply() bit-identical)."""
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.radio.component import RadioSEDComponent, RadioSEDComponentConfig
    from tengri.components.xray.component import XRaySEDComponent
    from tengri.forward.component_factory import _resolve_registry_component

    assert _resolve_registry_component(
        "radio", "radio", config=RadioSEDComponentConfig(sfr_mode="bell2003")
    ) == RadioSEDComponent(config=RadioSEDComponentConfig(sfr_mode="bell2003"))
    assert _resolve_registry_component("xray", "xray") == XRaySEDComponent()
    assert _resolve_registry_component("igm", "igm") == IGMSEDComponent()


def test_no_legacy_class_in_component_factory():
    """build_components must not import the component CLASSES directly — it
    resolves them from _REGISTRY (mirrors tools/check_single_dispatch.py)."""
    import tengri.forward.component_factory as cf

    for cls_name in _MAIN_CHAIN.values():
        assert not hasattr(cf, cls_name), (
            f"component_factory must not directly reference {cls_name} (#845)"
        )
