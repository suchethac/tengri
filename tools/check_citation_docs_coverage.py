#!/usr/bin/env python3
"""CI guard: every paper in ``references.bib`` must be acknowledged in the docs.

``src/tengri/citations/references.bib`` is the curated record, and
``registry.py`` says so in its own docstring: "The BibTeX file at ``BIB_PATH``
is the single source of truth. Do not hard-code citation data in this module."
The published *acknowledgements* page obeyed no such rule. It was prose, so it
drifted, and nothing could see it drift.

By the time this guard was written the record held **105** entries and
``docs/citation.md`` named about **15** of them. The missing ninety were not
obscure: every isochrone set (MIST, PARSEC, BaSTI, Padova), every stellar
library (MILES, STELIB, BaSeL), all three IMFs, all ten attenuation laws,
thirteen of the fourteen AGN papers, and the entire radio and X-ray blocks were
absent from the page a reader is sent to when they ask how to credit the code.
That is the failure mode that matters here: the people whose work tengri runs on
were uncredited on the page that exists to credit them.

Drift also runs the other way. The page cited the Ray Tracing Sampler as
``arXiv:2504.20029``; the record says ``arXiv:2510.25824`` (Behroozi 2025, *The
Ray Tracing Sampler*). One of the two was wrong for as long as both existed, and
no run went red.

What this checks
----------------

For every entry in the bib, the page must contain the first author's surname
with the publication year **near it** — the ``Surname et al. (YEAR)`` form the
page already uses. Both halves are derived from the bib's own ``short`` and
``year`` fields, so this guard has no list of its own to rot: add an entry to
``references.bib`` and the page is required to acknowledge it on the next run.

Proximity is the point. Checking "surname appears" and "year appears" as two
independent facts would pass a page that mentions Conroy in one section and 2009
in another, which is the vacuous-census failure — a check that counts what is
present rather than what is *stated*. The two must land within
:data:`WINDOW_CHARS` of each other, whitespace collapsed so a line wrap does not
break the pairing.

What this deliberately does **not** check is whether the citation is *correct*.
``check_citation_consistency.py`` compares citations to each other and
``check_citation_bib_coverage.py`` compares those in ``src/`` to the record.
This one asks a narrower question neither can: is the paper credited to its
authors, in public, at all?

Names are compared with TeX accents expanded and folded to ASCII, so the bib's
``Falc{\\'o}n-Barroso`` matches the page's ``Falcón-Barroso`` and ``En{\\ss}lin``
matches ``Enßlin``.

Dependencies: standard library only. The ``lint`` job installs ruff and nothing
else, so this must not import ``yaml`` or ``tengri`` — it parses the .bib
itself.

Usage
-----
    python tools/check_citation_docs_coverage.py           # CI mode
    python tools/check_citation_docs_coverage.py --list    # print every pairing
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import unicodedata

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BIB_PATH = REPO_ROOT / "src" / "tengri" / "citations" / "references.bib"
DOCS_PATH = REPO_ROOT / "docs" / "citation.md"

#: How far after a surname the year may sit and still count as the same
#: citation. "Conroy, Gunn & White (2009)" needs 27; the slack absorbs a
#: parenthetical aside, and is far short of a neighboring bullet.
WINDOW_CHARS = 120

#: TeX control sequences that appear in the ``short`` fields. ``\ss`` is a
#: ligature rather than an accent, so it expands to two letters and is handled
#: before the single-letter accent pattern below.
_TEX_LIGATURES = {r"{\ss}": "ss", r"\&": "&"}

#: ``{\'o}`` -> ``o``, ``{\'\i}`` -> ``i``, ``{\"u}`` -> ``u``. The optional
#: backslash covers dotless-i, written ``\i`` inside the accent group.
_TEX_ACCENT_RE = re.compile(r"""\{\\['`"^~=.]\\?([a-zA-Z])\}""")

#: Splits a ``short`` field at the end of the first author's surname.
#: "Conroy, Gunn & White" -> "Conroy"; "Vanden Berk et al." -> "Vanden Berk".
_AUTHOR_SPLIT_RE = re.compile(r"\s+et\s+al\.?|,\s|\s&\s")

_ENTRY_RE = re.compile(r"^@", re.MULTILINE)
_REGISTRY_KEY_RE = re.compile(r"^\s*registry_key\s*=\s*\{([^}]*)\}", re.MULTILINE)
_SHORT_RE = re.compile(r"^\s*short\s*=\s*\{(.*)\},?\s*$", re.MULTILINE)
_YEAR_RE = re.compile(r"^\s*year\s*=\s*\{?\s*(\d{4})", re.MULTILINE)


def expand_tex(text: str) -> str:
    """Expand the TeX escapes ``references.bib`` uses, then drop stray braces."""
    for tex, plain in _TEX_LIGATURES.items():
        text = text.replace(tex, plain)
    text = _TEX_ACCENT_RE.sub(r"\1", text)
    return text.replace("{", "").replace("}", "")


def fold(text: str) -> str:
    """Fold to accent-free, case-free ASCII for comparison.

    NFKD splits ``é`` into ``e`` plus a combining mark, which is then dropped.
    ``casefold`` (not ``lower``) is what maps ``ß`` onto ``ss``, so the bib's
    expanded ``En{\\ss}lin`` and the page's ``Enßlin`` land on one string.
    Whitespace is collapsed so a wrapped line cannot separate a surname from
    its year.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.casefold())


def first_surname(short: str) -> str:
    """Return the first author's surname from a bib ``short`` field.

    The trailing parenthetical is removed first, so "Cooray et al. (2026, Paper
    I)" does not leak its year into the name.
    """
    without_year = re.sub(r"\s*\([^)]*\)\s*$", "", expand_tex(short)).strip()
    return _AUTHOR_SPLIT_RE.split(without_year)[0].strip()


def parse_bib(text: str) -> list[dict[str, str]]:
    """Extract ``(registry_key, short, year)`` for every entry, stdlib only."""
    entries = []
    starts = [m.start() for m in _ENTRY_RE.finditer(text)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]
        key = _REGISTRY_KEY_RE.search(block)
        short = _SHORT_RE.search(block)
        year = _YEAR_RE.search(block)
        if not (key and short and year):
            continue
        entries.append(
            {
                "registry_key": key.group(1).strip(),
                "short": short.group(1).strip(),
                "year": year.group(1),
            }
        )
    return entries


def is_acknowledged(page_folded: str, surname: str, year: str) -> bool:
    """True when ``surname`` occurs with ``year`` within :data:`WINDOW_CHARS`.

    Word boundaries keep a short surname honest: "Mor" must be the whole word,
    not the opening of "morphology".
    """
    pattern = re.compile(r"\b" + re.escape(fold(surname)) + r"\b")
    return any(
        year in page_folded[match.start() : match.end() + WINDOW_CHARS]
        for match in pattern.finditer(page_folded)
    )


def check(bib_text: str, page_text: str) -> list[dict[str, str]]:
    """Return the entries with no acknowledgement on the page."""
    page_folded = fold(page_text)
    missing = []
    for entry in parse_bib(bib_text):
        surname = first_surname(entry["short"])
        if not is_acknowledged(page_folded, surname, entry["year"]):
            missing.append({**entry, "surname": surname})
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the surname/year pairing derived for every entry",
    )
    args = parser.parse_args(argv)

    bib_text = BIB_PATH.read_text(encoding="utf-8")
    page_text = DOCS_PATH.read_text(encoding="utf-8")
    entries = parse_bib(bib_text)

    if args.list:
        for entry in entries:
            print(f"{entry['registry_key']:28s} {first_surname(entry['short'])} ({entry['year']})")
        return 0

    missing = check(bib_text, page_text)
    docs_rel = DOCS_PATH.relative_to(REPO_ROOT)
    if missing:
        print(f"{len(missing)} of {len(entries)} citations are not acknowledged in {docs_rel}:\n")
        for entry in missing:
            print(f"  {entry['registry_key']:28s} expected: {entry['surname']} ({entry['year']})")
        print(
            f"\nEvery entry in {BIB_PATH.relative_to(REPO_ROOT)} must be credited on that"
            f"\npage. Add a bullet naming the author and year, e.g. "
            f"'{missing[0]['surname']} ({missing[0]['year']})'."
        )
        return 1

    print(f"OK: all {len(entries)} citations in references.bib are acknowledged in {docs_rel}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
