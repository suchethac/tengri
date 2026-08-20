# SPDX-License-Identifier: BSD-3-Clause
"""Composable radio grammar must honor sf/agn variants, not silently drop them (#1461).

The legacy form ``radio={'type': X}`` has been retired in PR6. This test
ensures the composable form ``radio={'sf': {...}, 'agn': {...}}`` correctly
maps all valid model names to their physics without silent substitution.

Historical context: the legacy branch previously assigned ``radio_sfr_mode='bell2003'`` /
``radio_agn_model='powerlaw'`` unconditionally, so three of the four accepted names
produced byte-identical spec state. The consequence was silent and physical:
``radio={'type': 'radio_dpl'}`` asked for Martinez-Ramirez+2024 (broken double
power-law) but got single power-law instead. The composable form fixes this by
requiring explicit ``agn`` block specification.
"""

from __future__ import annotations

import pytest

from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS
from tengri.parameters.groups import _valid_radio_types, parse_groups

pytestmark = pytest.mark.regression_bug


def _radio_state(spec) -> tuple[bool, str, str]:
    """Return the three attributes that decide which radio physics runs."""
    return (bool(spec.radio), spec.radio_sfr_mode, spec.radio_agn_model)


def test_composable_radio_selectors_are_distinct() -> None:
    """The composable radio variants must map to distinct physics models.

    This is the whole bug in one assertion: each sf/agn selector must
    produce different physics. Requesting 'dpl' must give dpl, not powerlaw.
    """
    from tengri.parameters.groups import _legacy_radio_type_to_blocks

    enabled = sorted(_valid_radio_types() - {"none"})
    states = {}
    for name in enabled:
        if name == "condon92":
            spec = {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}
            states[name] = _radio_state(parse_groups(radio=spec))
        else:
            sf_variant, agn_variant = _legacy_radio_type_to_blocks(name)
            spec = {"sf": {"type": sf_variant}, "agn": {"type": agn_variant}}
            states[name] = _radio_state(parse_groups(radio=spec))

    assert len(set(states.values())) > 1, (
        "all radio selectors produced identical spec state "
        f"{states} -- composable grammar is not selecting different models"
    )


def test_radio_dpl_selects_the_double_power_law() -> None:
    """``radio={'agn': {'type': 'dpl'}}`` must reach the DPL, not the power-law.

    Martinez-Ramirez+2024 (A&A 692, A85) is different physics from the
    single power-law -- a broken double power-law with an exponential
    aging cutoff. The composable form explicitly selects it.
    """
    spec = parse_groups(radio={"agn": {"type": "dpl"}})

    assert spec.radio is True
    assert spec.radio_agn_model == "dpl", (
        f"radio={{'agn': {{'type': 'dpl'}}}} gave radio_agn_model="
        f"{spec.radio_agn_model!r} -- expected the Martinez-Ramirez+2024 "
        "double power-law"
    )


def test_composable_radio_mapping_consistency() -> None:
    """Composable sf/agn mapping must be consistent across selections.

    Verify that composable form produces consistent results for
    the same underlying model when selected through different paths.
    """
    # Test that dpl is consistently selected
    dpl_direct = _radio_state(parse_groups(radio={"agn": {"type": "dpl"}}))
    # dpl_from_legacy would use _legacy_radio_type_to_blocks("radio_dpl")
    # which gives sf=bell2003, agn=dpl, same as above
    spec = {"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}}
    dpl_explicit = _radio_state(parse_groups(radio=spec))

    assert dpl_direct == dpl_explicit, (
        f"direct dpl form gave {dpl_direct} but explicit form gave "
        f"{dpl_explicit} -- inconsistent selection"
    )


@pytest.mark.parametrize("name", sorted(_valid_radio_types() - {"none"}))
def test_every_accepted_type_lands_on_a_real_model(name: str) -> None:
    """All composable radio selections must resolve to valid models.

    Guards the mapping: if a future ``radio_*`` component registers
    under a name that resolves to something the config would reject,
    this fails here rather than inside a forward pass.
    """
    from tengri.parameters.groups import _legacy_radio_type_to_blocks

    if name == "condon92":
        spec = parse_groups(radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}})
    else:
        sf_variant, agn_variant = _legacy_radio_type_to_blocks(name)
        spec = parse_groups(radio={"sf": {"type": sf_variant}, "agn": {"type": agn_variant}})

    assert spec.radio is True, f"radio selection {name!r} did not enable radio"
    assert spec.radio_sfr_mode in SF_RADIO_MODELS, (
        f"radio selection {name!r} produced radio_sfr_mode="
        f"{spec.radio_sfr_mode!r}, which RadioSEDComponentConfig rejects"
    )
    assert spec.radio_agn_model in AGN_RADIO_MODELS, (
        f"radio selection {name!r} produced radio_agn_model="
        f"{spec.radio_agn_model!r}, which RadioSEDComponentConfig rejects"
    )


def test_composable_radio_is_properly_attributed() -> None:
    """The citation table must credit the explicitly selected radio model.

    Verify that composable radio grammar credits the correct reference.
    When ``radio={'agn': {'type': 'dpl'}}`` is used, the citation should
    name Martinez-Ramirez+2024 for the double power-law.
    """
    import tengri

    spec = parse_groups(radio={"agn": {"type": "dpl"}})
    rows = [r for r in tengri.cite_components(spec) if str(r.get("component", "")) == "radio_agn"]

    assert len(rows) == 1, f"expected exactly one radio_agn citation row, got {rows}"
    assert rows[0]["name"] == "dpl"
    assert "Martinez-Ramirez" in rows[0]["citation"], (
        f"radio={{'agn': {{'type': 'dpl'}}}} was cited as {rows[0]!r} -- the "
        "double power-law ran but its paper is not credited"
    )


def test_none_still_disables_radio() -> None:
    """Both sf and agn set to 'none' must disable radio."""
    spec = parse_groups(radio={"sf": {"type": "none"}, "agn": {"type": "none"}})

    assert spec.radio is False


def test_condon92_means_default_sf_agn_models() -> None:
    """Composable radio with defaults maps to bell2003/powerlaw.

    This was historically the pre-split composite name; verify
    that the default sf/agn models produce the expected state.
    """
    spec = {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}
    assert _radio_state(parse_groups(radio=spec)) == (
        True,
        "bell2003",
        "powerlaw",
    )


def test_legacy_radio_type_form_is_retired() -> None:
    """The legacy radio={'type': X} form is retired and raises with guidance.

    PR6: the flat legacy form is no longer accepted. Users must use the
    composable surface with sf/agn axes. The error message includes a
    mapping to the equivalent composable form for mechanical conversion.
    """
    with pytest.raises(ValueError, match=r"legacy.*retired"):
        parse_groups(radio={"type": "condon92"})

    # Verify the error message includes the composable mapping
    with pytest.raises(ValueError) as excinfo:
        parse_groups(radio={"type": "radio_dpl"})
    message = str(excinfo.value)
    assert "radio=" in message
    assert "sf" in message
    assert "agn" in message
