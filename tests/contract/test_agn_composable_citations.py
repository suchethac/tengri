# SPDX-License-Identifier: BSD-3-Clause
"""Composable-AGN models must cite every AGN block they use.

Fresh-user audit (2026-07): ``cite_components()`` / ``cite(model)`` /
``print_components_bibtex()`` read only ``spec.agn_model`` when walking a
model's components. For a composable AGN model that attribute is the literal
string ``"composable"`` — a registry pseudo-entry whose citation is
``"(no single paper — block recipe of registered tengri AGN blocks)"``. So a
user who built, say, a SKIRTOR torus + Cue NLR got a citation list and a
BibTeX dump that **silently omitted every real AGN paper** (Stalevski 2016 for
SKIRTOR, Nenkova 2008, Buchner 2024, …) and told them to "add manually" — even
though each block's citation is registered and ``cite("skirtor")`` returns it.

The walk now fans out into the six composable AGN block slots (disc, torus,
nlr, blr, feii, attenuation), resolving each within its own category because
block names are not unique across categories (``skirtor`` is both a disc and a
torus, ``qsogen`` and ``grahsp`` span several).
"""

from __future__ import annotations

import contextlib
import io

import pytest

import tengri

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _spec(**agn):
    """A bare Parameters spec (no build, no data files) with AGN blocks set."""
    return tengri.Parameters(mean_sfh_type="dpl", agn_model="composable", **agn)


def _rows(spec):
    return [
        (r["component"], r["name"], r.get("citation", "")) for r in tengri.cite_components(spec)
    ]


def test_composable_agn_fans_out_into_block_citations():
    spec = _spec(
        agn_disc_block="skirtor",
        agn_torus_block="fritz",
        agn_nlr_block="cue",
    )
    rows = _rows(spec)
    by_component = {comp: (name, cit) for comp, name, cit in rows}

    # each active block is cited under its own category, with a real citation
    assert by_component["agn_disc"][0] == "skirtor"
    assert "Stalevski" in by_component["agn_disc"][1]
    assert by_component["agn_torus"][0] == "fritz"
    assert "Fritz" in by_component["agn_torus"][1]
    assert by_component["agn_nlr"][0] == "cue"
    assert "Li" in by_component["agn_nlr"][1]


def test_composable_wrapper_pseudo_citation_never_leaks():
    """The 'composable' registry pseudo-entry has no paper; it must never be
    emitted as a component citation."""
    spec = _spec(agn_disc_block="skirtor", agn_torus_block="skirtor")
    for _comp, name, cit in _rows(spec):
        assert name != "composable"
        assert "no single paper" not in cit


def test_inactive_block_slots_produce_no_rows():
    """A block set to 'none' (or a model with no AGN) must not add a row."""
    spec = _spec(agn_disc_block="skirtor")  # torus/nlr/blr/feii/atten default 'none'
    comps = {comp for comp, _n, _c in _rows(spec)}
    assert "agn_disc" in comps
    assert "agn_torus" not in comps
    assert "agn_nlr" not in comps

    no_agn = tengri.Parameters(mean_sfh_type="dpl")
    assert not any(c.startswith("agn") for c, _n, _cc in _rows(no_agn))


def test_multi_category_block_names_resolve_to_the_right_category():
    """``skirtor`` exists as both a disc and a torus; each slot must resolve
    within its own category (both happen to cite Stalevski, but the walk must
    not collapse them to a single flat name lookup)."""
    spec = _spec(agn_disc_block="skirtor", agn_torus_block="skirtor")
    rows = _rows(spec)
    cats = {comp for comp, _n, _c in rows if comp.startswith("agn")}
    assert cats == {"agn_disc", "agn_torus"}


def test_bibtex_emits_verified_block_keys():
    """print_components_bibtex must emit real BibTeX for AGN blocks whose
    paper is in the bundled registry (skirtor, grahsp, nenkova, multicolor,
    synthesizer, qsogen)."""
    spec = _spec(
        agn_disc_block="grahsp_sbpl",  # -> buchner2024
        agn_torus_block="nenkova",  # -> clumpy_nenkova2008
        agn_nlr_block="synthesizer",  # -> synthesizer
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tengri.print_components_bibtex(spec)
    out = buf.getvalue()
    assert "@article{Buchner_2024" in out
    assert "Nenkova_2008" in out
    assert "Synthesizer" in out or "Lovell" in out
    # the composable wrapper's non-entry must not appear as an AGN bib line
    assert "no bib entry" not in out.split("[agn_disc]")[1].split("%")[0]


#: Generic / toy AGN blocks that legitimately have no paper (a functional form
#: or a graybody, not a published model). They must still get a citation *row*
#: from the walk — just with an empty citation string.
_UNCITED_TOY_BLOCKS = {
    ("disc", "powerlaw"),
    ("torus", "simple"),
    ("torus", "two_temperature"),
}


@pytest.mark.filterwarnings("ignore::tengri.components.agn.blocks.runner.RecipeWarning")
def test_every_agn_block_is_citable():
    """Systemic guard: placing any registered AGN block in its slot yields a
    citation row (never silently dropped by the walk), and every block backed
    by a published model carries a non-empty citation. Prevents a future block
    from being silently uncited when the composable walk fans out."""
    slot_by_category = {
        "disc": "agn_disc_block",
        "torus": "agn_torus_block",
        "nlr": "agn_nlr_block",
        "blr": "agn_blr_block",
        "feii": "agn_feii_block",
        "attenuation": "agn_attenuation_block",
    }
    for category, slot in slot_by_category.items():
        for entry in tengri.list_agn_blocks(category=category):
            name = entry["name"]
            if name == "none":
                continue
            spec = _spec(**{slot: name})
            rows = {r["name"]: r.get("citation", "") for r in tengri.cite_components(spec)}
            assert name in rows, f"{category}/{name} produced no citation row"
            if (category, name) not in _UNCITED_TOY_BLOCKS:
                assert rows[name], f"{category}/{name} has an empty citation"
