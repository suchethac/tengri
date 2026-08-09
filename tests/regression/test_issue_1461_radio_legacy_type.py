# SPDX-License-Identifier: BSD-3-Clause
"""``radio={'type': X}`` must honor X, not validate it and throw it away (#1461).

``_valid_radio_types()`` accepts four names -- ``none``, ``condon92``,
``radio_powerlaw`` and ``radio_dpl`` -- and its docstring states the
SEDModelComponent variants "must be discoverable to users". The legacy
branch of :func:`~tengri.parameters.groups._translate_radio` checked the
name against that set and then assigned ``radio_sfr_mode='bell2003'`` /
``radio_agn_model='powerlaw'`` unconditionally, so three of the four
accepted names produced byte-identical spec state.

The consequence was silent and physical, not cosmetic:
``radio={'type': 'radio_dpl'}`` asks for the Martinez-Ramirez+2024 broken
double power-law with an exponential aging cutoff and got a single
power-law instead. ``radio_agn_model`` is the source of truth all the way
down -- ``component_factory`` passes it to ``RadioSEDComponentConfig`` as
``agn_radio_model``, which dispatches the emission function -- so the
substituted model is what the fit actually runs.

``radio_powerlaw`` is the trap in the middle: it collapses onto the value
that was already the default, so a test written against that one name
passes and certifies the whole axis. The parametrized case below is what
makes the guard load-bearing.
"""

from __future__ import annotations

import pytest

from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS
from tengri.parameters.groups import _valid_radio_types, parse_groups

pytestmark = pytest.mark.regression_bug


def _radio_state(spec) -> tuple[bool, str, str]:
    """Return the three attributes that decide which radio physics runs."""
    return (bool(spec.radio), spec.radio_sfr_mode, spec.radio_agn_model)


def test_legacy_type_names_are_not_all_identical() -> None:
    """The accepted vocabulary must not collapse to one state.

    This is the whole bug in one assertion: every non-``none`` name was
    accepted and every one produced ``(True, 'bell2003', 'powerlaw')``.
    """
    enabled = sorted(_valid_radio_types() - {"none"})
    states = {name: _radio_state(parse_groups(radio={"type": name})) for name in enabled}

    assert len(set(states.values())) > 1, (
        "every accepted radio type produced identical spec state "
        f"{states} -- the 'type' value is validated and then discarded, so "
        "asking for a specific radio model silently runs the default one"
    )


def test_radio_dpl_selects_the_double_power_law() -> None:
    """``radio={'type': 'radio_dpl'}`` must reach the DPL, not the power-law.

    Martinez-Ramirez+2024 (A&A 692, A85) is different physics from the
    single power-law -- a broken double power-law with an exponential
    aging cutoff. Substituting one for the other changes the radio SED
    shape with no warning.
    """
    spec = parse_groups(radio={"type": "radio_dpl"})

    assert spec.radio is True
    assert spec.radio_agn_model == "dpl", (
        f"radio={{'type': 'radio_dpl'}} gave radio_agn_model="
        f"{spec.radio_agn_model!r} -- the Martinez-Ramirez+2024 double "
        "power-law was requested and a single power-law was substituted"
    )


def test_legacy_type_agrees_with_the_composable_form() -> None:
    """The two grammars must not disagree about the same model.

    ``radio={'type': 'radio_dpl'}`` and ``radio={'agn': {'type': 'dpl'}}``
    name the same component through the legacy and the composable
    surface; they must land on the same spec.
    """
    legacy = _radio_state(parse_groups(radio={"type": "radio_dpl"}))
    composable = _radio_state(parse_groups(radio={"agn": {"type": "dpl"}}))

    assert legacy == composable, (
        f"legacy form gave {legacy} but the composable form gave "
        f"{composable} for the same underlying model"
    )


@pytest.mark.parametrize("name", sorted(_valid_radio_types() - {"none"}))
def test_every_accepted_type_lands_on_a_real_model(name: str) -> None:
    """Whatever a name resolves to must exist in the model tuples.

    Guards the derivation itself: if a future ``radio_*`` component
    registers under a name that strips to something the component config
    would reject, this fails here rather than inside a forward pass.
    """
    spec = parse_groups(radio={"type": name})

    assert spec.radio is True, f"radio={{'type': {name!r}}} did not enable radio"
    assert spec.radio_sfr_mode in SF_RADIO_MODELS, (
        f"radio={{'type': {name!r}}} produced radio_sfr_mode="
        f"{spec.radio_sfr_mode!r}, which RadioSEDComponentConfig rejects"
    )
    assert spec.radio_agn_model in AGN_RADIO_MODELS, (
        f"radio={{'type': {name!r}}} produced radio_agn_model="
        f"{spec.radio_agn_model!r}, which RadioSEDComponentConfig rejects"
    )


def test_legacy_type_is_attributed_to_the_model_it_runs() -> None:
    """The citation table must credit the model the legacy name selected.

    #1461 was filed as an attribution problem. Before the fix
    ``radio={'type': 'radio_dpl'}`` emitted a ``radio_agn`` row for
    ``powerlaw``, which carries no citation at all -- so a paper using
    the double power-law credited nobody for it. The row must now name
    Martinez-Ramirez+2024.
    """
    import tengri

    spec = parse_groups(radio={"type": "radio_dpl"})
    rows = [r for r in tengri.cite_components(spec) if str(r.get("component", "")) == "radio_agn"]

    assert len(rows) == 1, f"expected exactly one radio_agn citation row, got {rows}"
    assert rows[0]["name"] == "dpl"
    assert "Martinez-Ramirez" in rows[0]["citation"], (
        f"radio={{'type': 'radio_dpl'}} was cited as {rows[0]!r} -- the "
        "double power-law ran but its paper is not credited"
    )


def test_none_still_disables_radio() -> None:
    """The one name whose behavior must not change."""
    spec = parse_groups(radio={"type": "none"})

    assert spec.radio is False


def test_condon92_remains_the_default_composite() -> None:
    """``condon92`` is the pre-split composite name, not a third AGN model.

    It legitimately means "radio on with the default sf/agn models" --
    two shipped recipes rely on that. Pinned so the fix for the other
    names does not quietly redefine this one.
    """
    assert _radio_state(parse_groups(radio={"type": "condon92"})) == (
        True,
        "bell2003",
        "powerlaw",
    )
