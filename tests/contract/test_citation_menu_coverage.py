# SPDX-License-Identifier: BSD-3-Clause
"""Contract: every physics menu a spec can select from must be citable.

Follow-on to #1265 and #1447. :func:`tengri.cite_components` walks a
hand-maintained list of ``_add(component, name, table_fn)`` calls. Five
menus were never walked at all, so a spec that explicitly requested them
produced no row and no warning:

* ``radio`` (both the star-forming and the AGN slot) -- Bell 2003,
  Delvecchio+2021, Martinez-Ramirez+2024,
* ``shock`` -- MAPPINGS V (Allen+2008, Sutherland & Dopita 2017),
* ``igm`` -- Madau 1995 / Inoue+2014, applied by *default* and therefore
  missing from essentially every fit ever cited,
* ``xray`` -- Yang+2020, Lopez+2024, and
* ``dust_model`` -- **Charlot & Fall 2000**, the model behind the
  recommended ``dust={'type': 'two_component'}`` path.

This is the failure mode that is worse than a crash: the returned table
looks complete, so a user assembling their paper's acknowledgements from
it publishes without the attribution.

The tests below assert *coverage* in both directions. Citing too much is
also a defect here: the slot attributes keep non-``None`` defaults
(``radio_sfr_mode='bell2003'``, ``xray_model='yang20'``) even when the
component is switched off, so a walk that ignores the boolean gate would
credit Bell 2003 in every fit that never computed a radio flux.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.parameters.groups import parse_groups
from tengri.registry import _menu_listers, list_shock_models

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _cited(spec) -> list[dict]:
    return list(tengri.cite_components(spec))


def _citation_for(spec, component: str) -> str | None:
    """The citation string of the first row whose component matches."""
    for row in _cited(spec):
        if row["component"] == component:
            return row["citation"]
    return None


# --------------------------------------------------------------------------
# The five menus that were never walked
# --------------------------------------------------------------------------


def test_radio_slots_are_cited() -> None:
    """Both radio slots carry their own paper and must both appear.

    Block names are not unique across the two categories (``none`` is
    registered in both), so each slot has to be resolved inside its own
    category -- the same trap the composable AGN blocks hit.
    """
    spec = parse_groups(
        sfh={"type": "dpl"},
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}},
    )
    assert "Bell 2003" in (_citation_for(spec, "radio_sf") or ""), (
        "radio.sf=bell2003 was requested but Bell 2003 is not cited"
    )
    assert "Martinez-Ramirez" in (_citation_for(spec, "radio_agn") or ""), (
        "radio.agn=dpl was requested but Martinez-Ramirez+2024 is not cited"
    )


def test_shock_is_cited() -> None:
    """An enabled shock component must cite MAPPINGS V."""
    spec = parse_groups(sfh={"type": "dpl"}, shock={"type": "mappings"})
    citation = _citation_for(spec, "shock") or ""
    assert "Allen" in citation, f"shock enabled but MAPPINGS V not cited (got {citation!r})"


def test_igm_is_cited() -> None:
    """A requested IGM model must be cited."""
    spec = parse_groups(sfh={"type": "dpl"}, igm={"type": "madau"})
    citation = _citation_for(spec, "igm") or ""
    assert "Madau" in citation, f"igm=madau requested but not cited (got {citation!r})"


def test_igm_explicit_is_cited() -> None:
    """When IGM is explicitly enabled, its paper is cited.

    After retiring apply_igm, IGM is OFF by default unless explicitly
    enabled via igm={'type': ...}. When enabled, citations are owed.
    """
    spec = parse_groups(sfh={"type": "dpl"}, igm={"type": "inoue"})
    assert getattr(spec, "apply_igm", False), "igm={'type': 'inoue'} should enable IGM"
    citation = _citation_for(spec, "igm") or ""
    assert "Inoue" in citation, f"IGM enabled but not cited (got {citation!r})"


def test_igm_disabled_not_cited() -> None:
    """When IGM is disabled, no IGM paper is cited."""
    spec = parse_groups(sfh={"type": "dpl"})  # No igm dict -> IGM disabled
    assert not getattr(spec, "apply_igm", False), "default should have IGM disabled"
    citation = _citation_for(spec, "igm") or ""
    assert not citation or "Inoue" not in citation, (
        f"IGM disabled but cited anyway (got {citation!r})"
    )


def test_xray_is_cited() -> None:
    """A requested X-ray model must be cited."""
    spec = parse_groups(sfh={"type": "dpl"}, xray={"type": "yang20"})
    citation = _citation_for(spec, "xray") or ""
    assert "Yang" in citation, f"xray=yang20 requested but not cited (got {citation!r})"


def test_dust_model_is_cited() -> None:
    """The two-component dust model must cite Charlot & Fall 2000.

    ``dust_law_bc`` (the attenuation *curve*) was already walked, but the
    dust *model* -- the birth-cloud + diffuse geometry itself -- was not,
    so the recommended default path dropped its own foundational paper.
    """
    spec = parse_groups(
        sfh={"type": "dpl"},
        dust={"type": "two_component", "law": "calzetti"},
    )
    citation = _citation_for(spec, "dust") or ""
    assert "Charlot" in citation, (
        f"dust={{'type': 'two_component'}} did not cite Charlot & Fall 2000 (got {citation!r})"
    )


# --------------------------------------------------------------------------
# The opposite failure: citing physics that never ran
# --------------------------------------------------------------------------


def test_disabled_components_are_not_cited() -> None:
    """A component that is off must not be credited.

    ``radio_sfr_mode`` / ``xray_model`` / ``shock_norm`` all keep real
    defaults while their gate is ``False``, so a gate-blind walk would
    attach Bell 2003 and Yang+2020 to a plain stellar+dust fit.
    """
    spec = parse_groups(sfh={"type": "dpl"}, dust={"law": "power_law", "type": "two_component"})
    # Premise: the defaults really are non-None while the gates are off.
    assert spec.radio is False
    assert spec.shock is False
    assert spec.xray is False
    assert spec.radio_sfr_mode == "bell2003"
    assert spec.xray_model == "yang20"

    components = {row["component"] for row in _cited(spec)}
    blob = " ".join(row["citation"] for row in _cited(spec))
    assert not any(c.startswith("radio") for c in components), "radio is off but was cited"
    assert "shock" not in components, "shock is off but was cited"
    assert "xray" not in components, "xray is off but was cited"
    assert "Bell 2003" not in blob, "Bell 2003 credited to a fit with no radio emission"
    assert "Yang" not in blob, "Yang+2020 credited to a fit with no X-ray emission"


# --------------------------------------------------------------------------
# Anti-drift: the guard that stops a sixth menu going missing
# --------------------------------------------------------------------------

#: Menus whose selection ``cite_components`` resolves from the spec.
_WALKED = {
    "list_sfh_models",
    "list_agn_models",
    "list_agn_blocks",
    "list_dust_models",
    "list_dust_laws",
    "list_dust_emission_models",
    "list_nebular_backends",
    "list_radio_blocks",
    "list_shock_models",
    "list_igm_models",
    "list_xray_models",
}

#: Menus deliberately not resolved from the spec, each with the reason.
_NOT_WALKED = {
    # Cited from ``Posterior.method`` instead -- inference backend is not a
    # structural field of the spec.
    "list_inference_methods": "cited from Posterior.method, not from the spec",
    # Walking it would double-count. ``radio={'type': X}`` *is* a live
    # surface -- two shipped recipes use ``{'type': 'condon92'}`` -- but it
    # is resolved onto ``radio_sfr_mode`` / ``radio_agn_model`` at parse
    # time (``_legacy_radio_type_to_blocks``), and those two attributes are
    # already walked through ``list_radio_blocks``. There is no surviving
    # ``radio_model`` attribute to read, and nothing is lost: selecting
    # ``radio_dpl`` cites Martinez-Ramirez+2024 via the ``radio_agn`` row.
    #
    # The earlier reason recorded here -- "condon92 is unreachable from any
    # spec" -- was true when #1447 landed and went stale when the legacy
    # ``type`` axis was wired. It is what kept #1461 mis-scoped: the menu
    # entry was reachable all along, and the real defect was that the name
    # was validated and then discarded (#1461).
    "list_radio_models": "resolved onto radio_sfr_mode/radio_agn_model, which are walked",
    # Both kernels are already covered by the stellar component's unconditional
    # ``citations() -> ("dsps",)``. Selecting ``age_kernel='dsps'`` uses DSPS's
    # histogram kernel (Hearin+2021 Eq. 9) and cites DSPS; selecting ``'cic'``
    # uses tengri's own cloud-in-cell kernel (#964), which has no separate
    # reference, and DSPS is still cited for the SSP/MDF machinery around it.
    # So no citation is lost either way and there is nothing kernel-specific to
    # add to the walk.
    "list_age_kernels": "stellar already cites dsps unconditionally; cic is tengri's own (#964)",
    # ``MetModelSpec`` carries no references field at all -- its five fields are
    # ``name``, ``fn``, ``params``, ``settings``, ``internal_param_map`` -- so
    # there is nothing mode-specific for the walk to resolve. All ten modes are
    # parameterizations of the same SSP metallicity axis, and the stellar
    # component's unconditional ``citations() -> ("dsps",)`` already covers the
    # SSP/MDF machinery every one of them runs through. Give a mode its own
    # reference and this entry must move to ``_WALKED``.
    "list_metallicity_modes": "MetModelSpec has no references field; stellar cites dsps",
}


def test_every_menu_is_either_walked_or_documented() -> None:
    """Adding a menu must not silently leave the citation walk behind.

    ``_menu_listers()`` is the canonical set of physics menus; ``describe``,
    ``search`` and ``list_all`` all walk it. ``cite_components`` cannot walk
    it directly because each menu also needs to know *which spec attribute*
    selects it and *which gate* enables it -- neither is derivable from the
    lister. This test supplies the missing link: every lister must be
    classified, so a new menu fails here instead of silently going uncited.
    """
    menus = {fn.__name__ for fn in _menu_listers()}
    classified = _WALKED | set(_NOT_WALKED)

    unclassified = menus - classified
    assert not unclassified, (
        f"new physics menu(s) {sorted(unclassified)} are neither walked by "
        "cite_components nor listed in _NOT_WALKED with a reason -- a fit "
        "using them would cite nobody"
    )
    stale = classified - menus
    assert not stale, f"{sorted(stale)} no longer exist in _menu_listers()"


def test_shock_menu_still_has_exactly_one_selectable_model() -> None:
    """The shock walk assumes the gate implies MAPPINGS V.

    The spec records only ``shock: bool`` plus physics parameters -- there
    is no ``shock_model`` attribute -- so ``cite_components`` maps an
    enabled gate onto the single selectable entry. If a second shock model
    is ever registered that inference stops being sound and a real selector
    has to be threaded through, so fail loudly here rather than cite the
    wrong paper.
    """
    selectable = sorted(e["name"] for e in list_shock_models() if e["name"] != "none")
    assert selectable == ["mappings"], (
        f"shock menu now offers {selectable}; cite_components can no longer "
        "infer the model from the boolean gate alone"
    )
