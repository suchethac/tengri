# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_citation_bib_coverage.py`` (#1803).

A guard that passes is indistinguishable from a guard that checks nothing, so
most of this file is about making it *fail*: each check gets a synthetic defect
of the shape it exists to catch, and the two real defects found while writing it
get regression tests against the live tree.

The real defects, both live in ``src/`` before this landed:

* ``components/agn/blr.py`` cited Vanden Berk et al. 2001 with the **correct**
  DOI and the title *"The SDSS Quasar Catalog"*. The bib has *"Composite Quasar
  Spectra from the Sloan Digital Sky Survey"* — verified against the paper
  itself (AJ 122, 549; arXiv astro-ph/0105231).
* ``components/agn/blr_precompute.py`` cited the same paper as *"The SDSS Quasar
  Catalog: IV. Fifth Data Release"* under eprint ``astro-ph/0105488``. That
  title belongs to Schneider et al.; that eprint is not this paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import check_citation_bib_coverage as guard

pytestmark = pytest.mark.contract

VANDEN_BERK_DOI = "10.1086/321167"
VANDEN_BERK_TITLE = "Composite Quasar Spectra from the Sloan Digital Sky Survey"

#: Temple, Hewett & Banerji 2021 as ``references.bib`` stores it. MNRAS is a
#: British journal, so the published title keeps the British spelling.
QSOGEN_PUBLISHED = (
    "QSOgen: a model of the UV-to-submillimetre spectral energy distributions of quasars"
)
#: The same title as a tengri docstring must write it (NAMING_CONTRACT §10).
QSOGEN_AMERICAN = (
    "QSOgen: a model of the UV-to-submillimeter spectral energy distributions of quasars"
)


@pytest.fixture(scope="module")
def scanned():
    """Parse the bib and scan ``src`` once for the whole module."""
    bib_titles, bib_dois = guard._parse_bib(guard.BIB.read_text(encoding="utf-8"))
    cited, titled, placeholders = guard.collect(bib_titles)
    return bib_titles, bib_dois, cited, titled, placeholders


class TestGuardPassesOnCurrentTree:
    """The pinned baseline must describe the tree as it actually is."""

    def test_guard_exits_clean(self):
        assert guard.main([]) == 0

    def test_no_placeholder_identifiers_in_src(self, scanned):
        _, _, _, _, placeholders = scanned
        assert placeholders == []


class TestTitleAgreement:
    """Check 2 — the check a coverage-only guard would not have."""

    def test_disagreeing_title_is_reported(self, monkeypatch):
        monkeypatch.setattr(guard, "BIB_TITLE_DRIFT", frozenset())
        problems = guard.check_titles(
            {"10.1234/abc": [("f.py", 10, "The SDSS Quasar Catalog")]},
            {"10.1234/abc": VANDEN_BERK_TITLE},
        )
        assert len(problems) == 1
        assert "title disagrees" in problems[0]

    def test_matching_title_is_silent(self, monkeypatch):
        monkeypatch.setattr(guard, "BIB_TITLE_DRIFT", frozenset())
        assert (
            guard.check_titles(
                {"10.1234/abc": [("f.py", 10, VANDEN_BERK_TITLE)]},
                {"10.1234/abc": VANDEN_BERK_TITLE},
            )
            == []
        )

    def test_abbreviated_title_is_accepted(self, monkeypatch):
        """Dropping a subtitle is legitimate citation practice, not drift."""
        monkeypatch.setattr(guard, "BIB_TITLE_DRIFT", frozenset())
        assert (
            guard.check_titles(
                {"10.1234/abc": [("f.py", 1, "Dust tori around type II nuclei")]},
                {"10.1234/abc": "Dust tori around type II nuclei. I. Constraints"},
            )
            == []
        )

    def test_british_spelling_is_not_drift(self, monkeypatch):
        """NAMING_CONTRACT §10 forces American prose; the bib keeps the publisher's.

        The live case, not a contrivance: Temple, Hewett & Banerji 2021 (MNRAS
        508, 737; ``10.1093/mnras/stab2586``) appeared in a British journal, so
        ``references.bib`` reproduces the published spelling exactly while every
        docstring quoting it must Americanize. Reporting that as drift would put
        this guard in an unwinnable fight with
        ``tools/check_british_spelling.py``.
        """
        monkeypatch.setattr(guard, "BIB_TITLE_DRIFT", frozenset())
        assert (
            guard.check_titles(
                {"10.1234/abc": [("f.py", 1, QSOGEN_AMERICAN)]},
                {"10.1234/abc": QSOGEN_PUBLISHED},
            )
            == []
        )

    def test_pinned_doi_is_not_reported(self, monkeypatch):
        monkeypatch.setattr(guard, "BIB_TITLE_DRIFT", frozenset({"10.1234/abc"}))
        assert (
            guard.check_titles(
                {"10.1234/abc": [("f.py", 1, "A Completely Different Paper Title")]},
                {"10.1234/abc": VANDEN_BERK_TITLE},
            )
            == []
        )

    def test_stale_title_pin_is_reported(self, monkeypatch):
        """A pin that no longer describes a defect must be removed."""
        monkeypatch.setattr(guard, "BIB_TITLE_DRIFT", frozenset({"10.1234/abc"}))
        problems = guard.check_titles(
            {"10.1234/abc": [("f.py", 1, VANDEN_BERK_TITLE)]},
            {"10.1234/abc": VANDEN_BERK_TITLE},
        )
        assert len(problems) == 1
        assert "stale pin" in problems[0]


class TestCoverage:
    """Check 1 — a DOI in ``src`` must be in the bib."""

    def test_uncurated_doi_is_reported(self, monkeypatch):
        monkeypatch.setattr(guard, "UNCURATED_DOIS", frozenset())
        problems = guard.check_coverage({"10.9999/nope": {"f.py"}}, {"10.1234/abc"})
        assert len(problems) == 1
        assert "not in references.bib" in problems[0]

    def test_curated_doi_is_silent(self, monkeypatch):
        monkeypatch.setattr(guard, "UNCURATED_DOIS", frozenset())
        assert guard.check_coverage({"10.1234/abc": {"f.py"}}, {"10.1234/abc"}) == []

    def test_pin_suppresses_the_report(self, monkeypatch):
        monkeypatch.setattr(guard, "UNCURATED_DOIS", frozenset({"10.9999/nope"}))
        assert guard.check_coverage({"10.9999/nope": {"f.py"}}, {"10.1234/abc"}) == []

    def test_pin_becomes_stale_once_curated(self, monkeypatch):
        """Adding the paper to the bib must force the pin's removal.

        This is what makes the backlog a ratchet rather than a graveyard.
        """
        monkeypatch.setattr(guard, "UNCURATED_DOIS", frozenset({"10.1234/abc"}))
        problems = guard.check_coverage({"10.1234/abc": {"f.py"}}, {"10.1234/abc"})
        assert len(problems) == 1
        assert "stale pin" in problems[0]

    def test_pin_becomes_stale_once_uncited(self, monkeypatch):
        monkeypatch.setattr(guard, "UNCURATED_DOIS", frozenset({"10.9999/gone"}))
        problems = guard.check_coverage({}, {"10.1234/abc"})
        assert len(problems) == 1
        assert "no longer cited" in problems[0]


class TestPlaceholders:
    """Check 3 — a fabricated identifier reads as a real reference."""

    @pytest.mark.parametrize(
        "text",
        [
            "arXiv:2405.xxxxx",
            "arXiv:2503.xxxxx",
            "https://doi.org/10.xxxx/xxxxx",
            "10.1234/xxxxx",
        ],
    )
    def test_placeholder_forms_are_matched(self, text):
        assert guard._PLACEHOLDER.search(text) is not None

    @pytest.mark.parametrize(
        "text",
        ["arXiv:2405.12345", "https://doi.org/10.1086/321167", "10.1093/mnras/stv2794"],
    )
    def test_real_identifiers_are_not_matched(self, text):
        assert guard._PLACEHOLDER.search(text) is None

    def test_reported_with_file_and_line(self):
        problems = guard.check_placeholders([("f.py", 42, "arXiv:2405.xxxxx")])
        assert len(problems) == 1
        assert "f.py:42" in problems[0]


class TestDoiParsing:
    def test_rst_backtick_is_not_part_of_the_doi(self):
        """rST inline markup must not fork one DOI into two backlog entries.

        ``utils/sed_quantities.py`` wraps a DOI in backticks that
        ``components/xray/xray.py`` does not, and a regex that swallowed the
        closing backtick pinned the same paper twice under two spellings.
        """
        found = guard._DOI.findall(
            "see `https://doi.org/10.1051/0004-6361/201936817`_ and "
            "https://doi.org/10.1051/0004-6361/201936817"
        )
        assert {guard._norm_doi(d) for d in found} == {"10.1051/0004-6361/201936817"}

    @pytest.mark.parametrize("suffix", [".", ",", ";", ")", "]", "}"])
    def test_sentence_punctuation_is_stripped(self, suffix):
        doi = guard._DOI.search(f"10.1086/321167{suffix}").group(0)
        assert guard._norm_doi(doi) == VANDEN_BERK_DOI


class TestVandenBerkRegression:
    """The defect that motivated the title check (#1801, #1803)."""

    def test_doi_is_checked_not_pinned(self):
        assert VANDEN_BERK_DOI not in guard.BIB_TITLE_DRIFT

    def test_bib_carries_the_published_title(self, scanned):
        bib_titles, _, _, _, _ = scanned
        assert guard._norm_title(bib_titles[VANDEN_BERK_DOI]) == guard._norm_title(
            VANDEN_BERK_TITLE
        )

    def test_every_citation_agrees_with_the_bib(self, scanned):
        bib_titles, _, _, titled, _ = scanned
        assert VANDEN_BERK_DOI in titled, "no titled citation left to check"
        want = guard._norm_title(bib_titles[VANDEN_BERK_DOI])
        for rel, lineno, raw in titled[VANDEN_BERK_DOI]:
            assert guard._same_paper(guard._norm_title(raw), want), (
                f"{rel}:{lineno} cites {VANDEN_BERK_DOI} as {raw!r}"
            )

    def test_wrong_eprint_is_gone(self):
        """``astro-ph/0105488`` is not this paper; ``astro-ph/0105231`` is."""
        hits = [
            path.relative_to(guard.ROOT).as_posix()
            for path, _ in guard._iter_source_files()
            if "astro-ph/0105488" in path.read_text(encoding="utf-8")
        ]
        assert hits == []
