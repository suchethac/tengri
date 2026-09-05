# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_citation_docs_coverage.py``.

A guard that passes is indistinguishable from a guard that checks nothing, so
most of this file is about making it *fail*: each behavior gets a synthetic
defect of the shape it exists to catch, applied to the live page rather than to
a fixture, so a rewrite of ``docs/citation.md`` cannot quietly strand them.

The defect the guard was written for was live: ``references.bib`` held 105
entries and ``docs/citation.md`` acknowledged about 15. The page also cited the
Ray Tracing Sampler as ``arXiv:2504.20029`` where the record says
``arXiv:2510.25824``.

The subtle check is proximity. "Surname appears somewhere" and "year appears
somewhere" are two facts a page can satisfy without ever citing the paper — the
vacuous-census failure this repo has hit before, where a check counts what is
present instead of what is stated. :func:`test_surname_and_year_must_be_adjacent`
is the regression for that: it leaves both halves on the page and only moves
them apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_citation_docs_coverage as guard

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def bib_text() -> str:
    return guard.BIB_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_text() -> str:
    return guard.DOCS_PATH.read_text(encoding="utf-8")


def missing_keys(bib_text: str, page_text: str) -> list[str]:
    return sorted(entry["registry_key"] for entry in guard.check(bib_text, page_text))


# ---------------------------------------------------------------------------
# The live tree
# ---------------------------------------------------------------------------


def test_every_bib_entry_is_acknowledged(bib_text: str, page_text: str) -> None:
    """The published page credits every paper in the record."""
    assert missing_keys(bib_text, page_text) == []


def test_the_record_is_not_empty(bib_text: str) -> None:
    """A parser that silently returns nothing would make every check vacuous."""
    entries = guard.parse_bib(bib_text)
    assert len(entries) >= 100
    assert all(e["registry_key"] and e["short"] and e["year"] for e in entries)


def test_every_entry_yields_a_usable_surname(bib_text: str) -> None:
    """No ``short`` field degenerates to an empty or year-only name."""
    for entry in guard.parse_bib(bib_text):
        surname = guard.first_surname(entry["short"])
        assert surname, entry["registry_key"]
        assert not surname.isdigit(), entry["registry_key"]
        assert "(" not in surname, entry["registry_key"]


# ---------------------------------------------------------------------------
# Synthetic defects: the guard must fail
# ---------------------------------------------------------------------------


def test_dropping_an_acknowledgement_is_caught(bib_text: str, page_text: str) -> None:
    """Removing DSPS from the page names ``dsps`` and nothing else."""
    mutated = page_text.replace("Hearin et al.\n  (2023)", "REDACTED")
    assert mutated != page_text, "fixture drifted: the DSPS bullet moved"
    assert missing_keys(bib_text, mutated) == ["dsps"]


def test_surname_and_year_must_be_adjacent(bib_text: str, page_text: str) -> None:
    """Both halves stay on the page; only the distance between them changes.

    This is the check that separates a real coverage guard from a word count.
    """
    filler = "padding " * 200
    mutated = page_text.replace("Behroozi (2025)", "Behroozi " + filler + "(2025)")
    assert mutated != page_text, "fixture drifted: the Behroozi bullet moved"
    assert "Behroozi" in mutated and "2025" in mutated
    assert missing_keys(bib_text, mutated) == ["raytrace_behroozi"]


def test_a_new_bib_entry_must_be_acknowledged(bib_text: str, page_text: str) -> None:
    """The guard ratchets: adding to the record obliges the page."""
    mutated_bib = bib_text + (
        "\n@article{Nobody_2099,\n"
        "  author        = {{Nobody}, A.},\n"
        "  title         = {{A paper the page does not credit}},\n"
        "  year          = 2099,\n"
        "  registry_key  = {nobody2099},\n"
        "  category      = {other},\n"
        "  short         = {Nobody et al. (2099)},\n"
        "  role          = {Uncredited},\n"
        "}\n"
    )
    assert missing_keys(mutated_bib, page_text) == ["nobody2099"]


def test_a_short_surname_must_match_on_a_word_boundary(bib_text: str, page_text: str) -> None:
    """``Mor`` must be the whole word, not the opening of ``morphology``."""
    mutated = page_text.replace("Mor & Netzer (2012)", "morphology of Netzer (2012)")
    assert mutated != page_text, "fixture drifted: the Mor & Netzer bullet moved"
    assert missing_keys(bib_text, mutated) == ["mor_netzer2012"]


# ---------------------------------------------------------------------------
# Normalization: the guard must NOT produce false positives
# ---------------------------------------------------------------------------


def test_accents_fold_in_both_directions(bib_text: str, page_text: str) -> None:
    """A page spelling names without accents is still a valid acknowledgement."""
    mutated = (
        page_text.replace("Falcón-Barroso", "Falcon-Barroso")
        .replace("Enßlin", "Ensslin")
        .replace("Tepper-García", "Tepper-Garcia")
        .replace("Martínez-Ramírez", "Martinez-Ramirez")
        .replace("Véron-Cetty", "Veron-Cetty")
    )
    assert mutated != page_text, "fixture drifted: the accented names moved"
    assert missing_keys(bib_text, mutated) == []


def test_a_line_wrap_does_not_break_a_pairing(bib_text: str, page_text: str) -> None:
    """Markdown wraps bullets; a newline between name and year is not a defect."""
    mutated = page_text.replace("Choi et al. (2016)", "Choi et al.\n  (2016)")
    assert mutated != page_text, "fixture drifted: the MIST bullet moved"
    assert missing_keys(bib_text, mutated) == []


@pytest.mark.parametrize(
    ("short", "expected"),
    [
        ("Conroy, Gunn \\& White (2009)", "Conroy"),
        ("Vanden Berk et al. (2001)", "Vanden Berk"),
        ("Le Borgne et al. (2003)", "Le Borgne"),
        ("da Cunha et al. (2013)", "da Cunha"),
        ("Oke \\& Gunn (1983)", "Oke"),
        ("Eldridge, Stanway et al. (2017)", "Eldridge"),
        ("Cooray et al. (2026, Paper I)", "Cooray"),
        ("Falc{\\'o}n-Barroso et al. (2011)", "Falcon-Barroso"),
        ("En{\\ss}lin (2019)", "Ensslin"),
        ("Tepper-Garc{\\'\\i}a (2006)", "Tepper-Garcia"),
        ("Mart{\\'\\i}nez-Ram{\\'\\i}rez et al. (2024)", "Martinez-Ramirez"),
        ("Véron-Cetty, Joly & Véron (2004)", "Véron-Cetty"),
    ],
)
def test_surname_extraction(short: str, expected: str) -> None:
    """Multi-word surnames, TeX accents, and trailing parentheticals."""
    assert guard.first_surname(short) == expected


def test_fold_maps_eszett_and_accents_onto_ascii() -> None:
    """``casefold`` (not ``lower``) is what makes ``ß`` meet ``ss``."""
    assert guard.fold("Enßlin") == guard.fold("Ensslin")
    assert guard.fold("Falcón-Barroso") == guard.fold("Falcon-Barroso")
    assert guard.fold("a\n  b") == "a b"
