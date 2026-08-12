# SPDX-License-Identifier: BSD-3-Clause
"""A reference that exists in references.bib must not print as a comment.

``print_components_bibtex`` heads its output *"Paste into your .bib file"*. On a
stellar+dust+X-ray+radio model, 4 of 11 components printed this instead::

    % [dust] two_component: Charlot & Fall 2000 (ApJ 539, 718)
    %   (no bib entry in tengri.citations — please add manually)

Charlot & Fall 2000 **is** the dust model, and ``@article{Charlot_2000, ...}``
was in ``references.bib`` the whole time — with DOI, ADS URL and a
``registry_key``. So were Bell 2003, Inoue+2014 and Yang+2020. A reader pasted
the output and silently lost them, because a LaTeX comment is not an entry.

The cause was a second map. ``_NAME_TO_BIBKEY`` in ``registry.py`` was a
hand-written component-name -> key dict, while
``tengri.citations.associations`` already carried a ``*_CITATIONS`` table per
subsystem doing the same job for :func:`tengri.collect_citations`. The emitter
consulted only its own copy, and that copy had drifted.

The fix reads both. Measured over every menu row carrying a citation:

    rows with a citation : 117
    resolved BEFORE      : 45
    resolved AFTER       : 69   (+24)

The 24 are not scattered — whole subsystems were dark: every IGM model, eight
dust-emission models, both dust models, shock, and X-ray.

48 rows still resolve to nothing. Those are mostly AGN blocks whose papers are
genuinely absent from ``references.bib``; adding them is bibliography work, not
a mapping fix, and this file does not assert them away.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

import pytest

import tengri
from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
from tengri.citations import cite
from tengri.registry import _NAME_TO_BIBKEY, _association_bibkeys, _bibkeys_for

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

# The four that shipped as comments while sitting in references.bib.
DROPPED = {
    "two_component": "Charlot_2000",
    "bell2003": "Bell_2003",
    "inoue": "Inoue_2014",
    "simple": "Lehmer_2016",
}


def _cited_menu_rows() -> list[tuple[str, str, str]]:
    """(menu, name, citation) for every menu row that claims a reference."""
    rows = []
    for menu in sorted(n for n in dir(tengri) if n.startswith("list_")):
        try:
            table = getattr(tengri, menu)()
        except Exception:
            continue
        for row in table if isinstance(table, list) else []:
            if isinstance(row, dict) and str(row.get("citation", "")).strip():
                rows.append((menu, row["name"], row["citation"]))
    return rows


def _emits_bibtex(keys) -> bool:
    for key in keys:
        try:
            if callable(getattr(cite(key), "to_bibtex", None)):
                return True
        except Exception:
            continue
    return False


@pytest.fixture(scope="module")
def bibtex_text(ssp_data_fsps):
    """BibTeX for a model spanning dust, IGM, X-ray and radio."""
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "wise_w4"]))
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        xray={"type": "simple"},
        radio={"type": "condon92"},
        redshift=Fixed(0.1),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        tengri.print_components_bibtex(model)
    return buffer.getvalue()


class TestTheCensus:
    def test_menus_do_claim_references(self):
        """A census of zero would make every other test here vacuous."""
        assert len(_cited_menu_rows()) > 50, (
            "almost no menu row carries a citation — the scan is broken, or the "
            "citation column was dropped from the registries."
        )

    def test_the_association_tables_are_consulted(self):
        """The union, not one map: association keys must reach the emitter."""
        missed = []
        for _menu, name, _citation in _cited_menu_rows():
            association = _association_bibkeys(name)
            if not association:
                continue
            resolved = _bibkeys_for(name, _NAME_TO_BIBKEY)
            if not set(association) <= set(resolved):
                missed.append((name, association, resolved))
        assert not missed, (
            f"tengri.citations.associations names BibTeX keys for these "
            f"components that the emitter never sees: {missed[:8]}. The "
            f"emitter must read the association tables, not only its own "
            f"hand-written map."
        )

    def test_the_hand_written_map_is_still_needed(self):
        """Not a fallback chain — some names live only in the explicit map.

        ``dpl`` cites Bagpipes and no association table records it, so
        replacing the explicit map with the associations would lose it.
        """
        only_explicit = [
            name
            for name in _NAME_TO_BIBKEY
            if not _association_bibkeys(name) and _emits_bibtex([_NAME_TO_BIBKEY[name]])
        ]
        assert only_explicit, (
            "every explicit mapping is now duplicated in the association "
            "tables; if that is deliberate, delete _NAME_TO_BIBKEY rather than "
            "keeping two copies."
        )


class TestTheFourThatShipped:
    @pytest.mark.parametrize(("name", "bibkey"), sorted(DROPPED.items()))
    def test_the_entry_exists_in_the_bibliography(self, name, bibkey):
        """The premise: these were never missing, only unmapped."""
        keys = _bibkeys_for(name, _NAME_TO_BIBKEY)
        assert keys, f"{name!r} maps to no BibTeX key at all"
        assert _emits_bibtex(keys), (
            f"{name!r} maps to {keys} and none of them yields BibTeX — this "
            f"test's premise (that {bibkey} is in references.bib) is wrong."
        )

    @pytest.mark.parametrize(("name", "bibkey"), sorted(DROPPED.items()))
    def test_it_is_reachable_from_the_component_name(self, name, bibkey):
        keys = _bibkeys_for(name, _NAME_TO_BIBKEY)
        entries = []
        for key in keys:
            try:
                entries.append(cite(key).to_bibtex())
            except Exception:
                continue
        assert any(f"{{{bibkey}," in e for e in entries), (
            f"{name!r} resolves to {keys}, none of which emits @...{{{bibkey}, — "
            f"the reader's .bib would be missing this reference."
        )


class TestTheEmittedDocument:
    def test_nothing_is_dropped_to_a_comment(self, bibtex_text):
        """Every component with a resolvable key emits a real entry."""
        commented = re.findall(r"% \[[^\]]+\] ([^\n:]+): [^\n]*\n%\s+\(no", bibtex_text)
        droppable = [
            name
            for name in (n.strip() for n in commented)
            if _emits_bibtex(_bibkeys_for(name, _NAME_TO_BIBKEY))
        ]
        assert not droppable, (
            f"{droppable} print as LaTeX comments although a BibTeX entry "
            f"exists for them. The header says 'Paste into your .bib file'; a "
            f"comment silently loses the reference."
        )

    def test_the_expected_entries_are_present(self, bibtex_text):
        keys = set(re.findall(r"^@\w+\{([^,]+),", bibtex_text, flags=re.M))
        for expected in ("Charlot_2000", "Bell_2003", "Inoue_2014"):
            assert expected in keys, f"{expected} is not in the emitted BibTeX; got {sorted(keys)}"

    def test_no_duplicate_keys(self, bibtex_text):
        keys = re.findall(r"^@\w+\{([^,]+),", bibtex_text, flags=re.M)
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        assert not dupes, f"duplicate BibTeX keys would collide in a .bib file: {dupes}"

    def test_the_fallback_no_longer_claims_the_entry_is_missing(self, bibtex_text):
        """It is a missing *mapping*; saying otherwise sent readers to the wrong file."""
        assert "no bib entry in tengri.citations" not in bibtex_text, (
            "the fallback still says the entry is missing, which was false for "
            "four references that were in references.bib all along."
        )
