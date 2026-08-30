#!/usr/bin/env python3
"""CI guard: a citation in ``src/`` must agree with ``references.bib``.

``src/tengri/citations/references.bib`` is the repo's curated citation record.
It is good, and it is *correct*: while three docstrings each invented a
different title for the Cue paper (#1801), ``tengri.cite("cue")`` returned the
real one the whole time. Nothing bound docstrings to it. Every consumer of the
bib is a *runtime* path — ``citations/``, ``registry.py`` and two component
precompute modules — so the ``.. [N]`` blocks that **are** the published API
reference (``docs/api/*.rst`` are autodoc stubs) sat one function call away from
an authoritative record they never consulted.

This guard closes that gap offline, with no network call.

Why the sibling guard cannot do it
----------------------------------

``tools/check_citation_consistency.py`` compares citations **to each other** —
one title one bibcode, one bibcode one title, no unresolved markers. That is the
right check for drift, and it caught the Cue titles. It is structurally blind to
a citation that is wrong the same way everywhere: Feltre et al. 2016 was cited
as ``10.1093/mnras/stw2180`` in three places, which resolves to *Gratia &
Fabrycky 2017, "Outer-planet scattering can gently tilt an inner planetary
system"* — perfectly self-consistent, and the correct DOI was already in the
repo. Internal consistency cannot see that. Comparison against the record can.

The two checks are complementary in both directions, which is why this is a
separate tool rather than a fourth check in that one:

* Feltre's DOI is absent from the bib, so :func:`check_coverage` catches it.
* Vanden Berk et al. 2001 is cited in ``components/agn/blr.py`` with the
  **correct** DOI ``10.1086/321167`` and the title *"The SDSS Quasar Catalog"*.
  The DOI is in the bib, so coverage passes; the bib's title is *"Composite
  Quasar Spectra from the Sloan Digital Sky Survey"*, so
  :func:`check_titles` catches it. A coverage-only guard would have shipped it.

Both were live in ``src/`` when this tool was written.

The three checks
----------------

1. **Coverage** — a DOI written in ``src/`` must appear in ``references.bib``.
2. **Title agreement** — when a cited DOI *is* in the bib, the title quoted
   beside it must name the same paper as the bib's title.
3. **No placeholder identifiers** — ``arXiv:2405.xxxxx`` and friends. #1801
   shipped exactly that string in a published notebook. The sibling guard
   rejects *prose* markers ("not found", "TODO: cite"); this one rejects a
   fabricated **identifier**, which reads as a real reference at a glance.

Both backlogs are pinned rather than blocking, the pattern
``check_citation_consistency`` and ``check_zero_hiding_clamps`` already use: the
guard's job is to stop the backlog *growing*, and draining it means checking one
paper at a time against its publisher, which is not a mechanical sweep. A pin
that no longer corresponds to a defect is itself an error, so the lists ratchet
down and cannot quietly become a graveyard.

Spelling
--------

Titles are compared with British spellings folded onto American ones. This is
not sloppiness — it is forced. NAMING_CONTRACT §10 mandates American English in
all prose, so a docstring quotes QSOgen as *"UV-to-submillimeter"*, while the
bib preserves the publisher's *"UV-to-submillimetre"*. Reporting that as drift
would put this guard in a fight with ``tools/check_british_spelling.py`` that
neither could win.

Run ``--baseline`` to print the pin literals for the current tree.

Dependencies: standard library only. The ``lint`` job installs ruff and nothing
else, so this must not import ``yaml`` or ``tengri``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "src" / "tengri" / "citations" / "references.bib"
SRC = ROOT / "src" / "tengri"

#: ``citations/`` is the bib's own plumbing. The single DOI in that tree is a
#: format example in a parameter description (``e.g., "10.1086/512090"``), not a
#: reference to a paper the code implements.
EXCLUDE_DIRS = ("src/tengri/citations",)

#: A DOI as written in prose. The trailing class excludes the punctuation that
#: ends a sentence or closes a parenthetical, which is not part of the name.
#: The backtick matters: rST wraps a DOI in inline literal markup, and letting
#: one trail turns ``10.1051/...817`` and ``10.1051/...817` `` into two distinct
#: DOIs — a phantom second entry in the backlog for a paper cited once.
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;)\]}`]+")

#: A quoted paper title. Straight or curly quotes; >= 15 characters so a short
#: quoted phrase inside a sentence is not mistaken for a title.
_TITLE = re.compile(r'["“]([^"”]{15,300})["”]')

#: A numpydoc/rST reference entry, ``.. [1] ...``, plus indented continuations.
_ENTRY = re.compile(r"^(\s*)\.\.\s+\[[^\]]+\]\s")

#: A fabricated identifier: an eprint or DOI whose digits were never filled in.
#: ``arXiv:2405.xxxxx`` shipped in a published notebook (#1801) and reads as a
#: real reference until someone tries to open it.
_PLACEHOLDER = re.compile(
    r"arxiv:\s*\d{4}\.[x]{3,}"  # arXiv:2405.xxxxx
    r"|arxiv:\s*[x]{4,}"  # arXiv:xxxx.xxxxx
    r"|10\.[x]{4,}"  # 10.xxxx/...
    r"|10\.\d{4,9}/[x]{4,}",  # 10.1234/xxxxx
    re.I,
)

#: British spellings folded onto American ones before a title comparison.
#: See the module docstring: the repo mandates American prose, the bib preserves
#: the publisher's spelling, so the two legitimately differ.
_SPELLING = (
    ("submillimetre", "submillimeter"),
    ("millimetre", "millimeter"),
    ("centre", "center"),
    ("colour", "color"),
    ("ionisation", "ionization"),
    ("ionised", "ionized"),
    ("normalisation", "normalization"),
    ("normalised", "normalized"),
    ("catalogue", "catalog"),
    ("modelling", "modeling"),
    ("behaviour", "behavior"),
    ("favour", "favor"),
    ("analyse", "analyze"),
)

#: DOIs cited in ``src/`` that are not yet in ``references.bib`` (#1803).
#:
#: 107 at the time of writing, across 74 files. Not a mechanical sweep — each
#: needs checking against the publisher — but the shape makes it batchable:
#: 29 appear in more than one file, so they have been copied around, and 73
#: arXiv eprints are already in the bib, so a good fraction are likely the same
#: papers already curated under an eprint rather than a publisher DOI.
#:
#: A DOI may only be added here with a note in the PR saying why it could not be
#: curated instead. Removing one is the goal.
UNCURATED_DOIS = frozenset(
    {
        "10.1046/j.1365-8711.2003.06224.x",  # xray.py +1
        "10.1051/0004-6361/200811368",  # xray.py
        "10.1051/0004-6361/200913298",  # xray_precompute.py
        "10.1051/0004-6361/201321236",  # gp_sfh.py
        "10.1051/0004-6361/201322803",  # attenuation.py
        "10.1051/0004-6361/201323152",  # _shared.py
        "10.1051/0004-6361/201527923",  # _closures.py +1
        "10.1051/0004-6361/201628997",  # themis.py
        "10.1051/0004-6361/201629925",  # schreiber2016_ir.py
        "10.1051/0004-6361/201731036",  # mean_sfh.py
        "10.1051/0004-6361/201936817",  # xray.py +1
        "10.1051/0004-6361/202449801",  # disc.py +2
        "10.1051/0004-6361:20042363",  # disc.py +1
        "10.1051/0004-6361:20066130",  # fritz.py
        "10.1051/0004-6361:20078829",  # mean_sfh.py
        "10.1086/130714",  # _recombination_coeffs.py
        "10.1086/151796",  # disc.py +1
        "10.1086/159815",  # balmer.py
        "10.1086/161102",  # xray.py
        "10.1086/162189",  # dig.py
        "10.1086/162686",  # attenuation.py
        "10.1086/171637",  # attenuation.py
        "10.1086/174330",  # spectral.py
        "10.1086/174348",  # attenuation.py
        "10.1086/191679",  # feii.py +1
        "10.1086/307523",  # fesc_model.py +1
        "10.1086/308056",  # spectral.py
        "10.1086/311810",  # disc.py
        "10.1086/318651",  # attenuation.py
        "10.1086/320357",  # blr_precompute.py
        "10.1086/320360",  # blr.py
        "10.1086/342486",  # attenuation.py +1
        "10.1086/421115",  # disc.py +2
        "10.1086/423885",  # polar_dust.py
        "10.1086/506270",  # blr.py
        "10.1086/506525",  # disc.py +1
        "10.1086/509629",  # radio_precompute.py
        "10.1086/510378",  # drude_profiles.py +4
        "10.1086/519990",  # xray.py
        "10.1086/651008",  # desi.py +1
        "10.1088/0004-637x/685/1/160",  # nenkova_agnfitter.py
        "10.1088/0004-637x/708/1/58",  # attenuation.py
        "10.1088/0004-637x/724/1/559",  # sed_quantities.py
        "10.1088/0004-637x/737/2/67",  # unified.py +2
        "10.1088/0004-637x/745/2/181",  # xray.py
        "10.1088/0004-637x/770/1/57",  # mean_sfh.py
        "10.1088/0067-0049/189/1/15",  # blr.py
        "10.1088/0067-0049/206/1/4",  # _phys.py +1
        "10.1093/mnras/238.3.897",  # disc.py +1
        "10.1093/mnras/264.1.161",  # attenuation.py
        "10.1093/mnras/272.1.41",  # _shared.py
        "10.1093/mnras/273.3.837",  # xray.py
        "10.1093/mnras/283.1.193",  # _nthcomp.py
        "10.1093/mnras/staa1116",  # mean_sfh.py
        "10.1093/mnras/staa1838",  # registry.py
        "10.1093/mnras/staa2620",  # chemical_evolution.py +1
        "10.1093/mnras/stac818",  # dig.py
        "10.1093/mnras/stad1859",  # cb19_precompute.py
        "10.1093/mnras/stad3891",  # cloudy_cb19.py
        "10.1093/mnras/stad478",  # disc.py +2
        "10.1093/mnras/stt2056",  # nlr.py +1
        "10.1093/mnras/stv1950",  # xray.py
        "10.1093/mnras/stv2794",  # nlr_cloudy.py +2
        "10.1093/mnras/stw044",  # feltre_precompute.py
        "10.1103/revmodphys.81.969",  # dig.py
        "10.1111/j.1365-2966.2004.07473.x",  # xray.py +1
        "10.1111/j.1365-2966.2007.12255.x",  # attenuation.py
        "10.1111/j.1365-2966.2008.13535.x",  # _closures.py
        "10.1111/j.1365-2966.2011.18906.x",  # _recombination_coeffs.py
        "10.1111/j.1365-2966.2011.19779.x",  # disc.py
        "10.1111/j.1365-2966.2012.21699.x",  # slone_netzer.py
        "10.1137/0717021",  # dense_basis.py +2
        "10.1146/annurev-astro-081811-125610",  # sfr_window.py
        "10.1146/annurev.astro.41.011802.094840",  # attenuation.py
        "10.1146/annurev.astro.46.060407.145222",  # physics_constants.py
        "10.1214/088342307000000014",  # gp_sfh.py
        "10.22201/ia.01851101p.2019.55.02.14",  # mappings_shock_precompute.py +1
        "10.22201/ia.01851101p.2019.55.02.21",  # shock_model.py
        "10.3390/e23070853",  # native.py +1
        "10.3847/1538-3881/acb212",  # desi.py +1
        "10.3847/1538-4357/833/1/98",  # agn_priors.py
        "10.3847/1538-4357/aa5ffe",  # nonparametric.py
        "10.3847/1538-4357/aa63f0",  # dense_basis.py
        "10.3847/1538-4357/aab0a7",  # sed_model.py
        "10.3847/1538-4357/aad235",  # attenuation.py
        "10.3847/1538-4357/aae386",  # attenuation.py
        "10.3847/1538-4357/aae8e0",  # psd_models.py
        "10.3847/1538-4357/aaf563",  # dense_basis.py
        "10.3847/1538-4357/ab2052",  # dense_basis.py
        "10.3847/1538-4357/ab7cc9",  # fesc_model.py
        "10.3847/1538-4357/ac062c",  # nonparametric.py
        "10.3847/1538-4357/ac1aa7",  # _closures.py
        "10.3847/1538-4357/ac3aca",  # nonparametric.py
        "10.3847/1538-4357/ac4867",  # dsps_wrapper.py
        "10.3847/1538-4357/ac4971",  # xray.py +1
        "10.3847/1538-4357/ac6959",  # _apply.py
        "10.3847/1538-4365/aa6541",  # mappings_photo.py +3
        "10.3847/1538-4365/aa96ad",  # unified.py
        "10.3847/2041-8213/aa6838",  # cat3d_precompute.py
        "10.48550/arxiv.1008.4686",  # noise.py
        "10.5281/zenodo.14140949",  # mappings_photo.py
        "10.7910/dvn/3b6e6s",  # astrodust_hd23.py +1
    }
)

#: DOIs whose bib title and docstring title disagree (#1803).
#:
#: 14 at the time of writing. Some are paraphrase — ApJS 254, 22 is written both
#: "Stellar Population Inference with Prospector" (the real title) and
#: "Prospector: Inferring the Star Formation Histories of Galaxies". Others name
#: an entirely different paper: ``10.1086/589652`` is the MAPPINGS III shock
#: library, cited as "The Distance and Metallicity of the Galaxy M33".
#:
#: A pinned DOI's title is **unchecked** until the pin is removed, so this is a
#: debt list, not an exemption list. Same pin-by-identifier design as
#: ``KNOWN_TITLE_DRIFT`` in ``check_citation_consistency``.
BIB_TITLE_DRIFT = frozenset(
    {
        "10.1051/0004-6361/200912497",  # bib: Analysis of galaxy spectral energy distributi
        "10.1051/0004-6361/201834156",  # bib: CIGALE: a python Code Investigating GALaxy Em
        "10.1086/308197",  # bib: Multiple Scattering in Clumpy Media. II. Galactic Environm
        "10.1086/308692",  # bib: The Dust Content and Opacity of Actively Star-forming Gala
        "10.1086/511055",  # bib: Infrared Emission from Interstellar Dust. IV. The Silicate
        "10.1086/589652",  # bib: The MAPPINGS III Library of Fast Radiative Shock Models
        "10.1088/0004-637x/780/2/172",  # bib: Andromeda's Dust
        "10.1111/j.1365-2966.2012.21455.x",  # bib: Far-infrared spectral energy distributio
        "10.3847/0004-637x/825/1/7",  # bib: The Evolution of Normal Galaxy X-Ray Emission t
        "10.3847/1538-4357/aabf3c",  # bib: Dust Attenuation Curves in the Local Universe: D
        "10.3847/1538-4357/ab133c",  # bib: How to Measure Galaxy Star Formation Histories.
        "10.3847/1538-4357/acc4c2",  # bib: The Astrodust+PAH Model: A Unified Description o
        "10.3847/1538-4365/abef67",  # bib: Stellar Population Inference with Prospector
        "10.3847/2041-8213/adc388",  # bib: Improving Photometric Redshifts of Epoch of Reio
    }
)


def _norm_doi(doi: str) -> str:
    """Lowercase and strip trailing sentence punctuation from a DOI."""
    return doi.rstrip(".,;:)]}").lower()


def _norm_title(title: str) -> str:
    """Casefold a title to letters, digits and single spaces.

    Drops braces (BibTeX capitalization guards), punctuation and line wrapping,
    then folds British spellings, so the comparison sees the words alone.
    """
    text = re.sub(r"[\s\n]+", " ", title).strip(" ,.;:")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[^a-z0-9 ]", "", text.casefold()).strip()
    for british, american in _SPELLING:
        text = text.replace(british, american)
    return re.sub(r" +", " ", text)


def _same_paper(a: str, b: str) -> bool:
    """Two normalized titles naming the same paper.

    A citation legitimately abbreviates — dropping a subtitle after a colon, or
    a series numeral — so prefix containment is the right test. Unrelated papers
    diverge in the first few words and are not matched. Same rule as
    ``check_citation_consistency._same_paper``, deliberately: the two guards
    must not disagree about what counts as one paper.
    """
    return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))


def _iter_source_files():
    """Yield ``(path, repo-relative posix path)`` in a fixed order everywhere.

    ``Path.rglob`` yields in filesystem order, which differs between macOS and
    Linux. A guard whose answer depends on the machine it ran on is not a guard,
    which ``check_citation_consistency`` learned the hard way (its baseline
    passed locally and failed in CI).
    """
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(d) for d in EXCLUDE_DIRS):
            continue
        yield path, rel


def _parse_bib(text: str) -> tuple[dict[str, str], set[str]]:
    """Return ``(doi -> title, {all dois})`` from a BibTeX file.

    Splits on the entry delimiter rather than parsing BibTeX properly: the only
    fields needed are ``doi`` and ``title``, and a real parser would be a
    dependency the ``lint`` job does not have.
    """
    titles: dict[str, str] = {}
    dois: set[str] = set()
    for block in re.split(r"\n@", text):
        found = re.search(r'^\s*doi\s*=\s*[{"]([^}"]+)[}"]', block, re.M | re.I)
        if not found:
            continue
        doi = _norm_doi(found.group(1))
        dois.add(doi)
        title = re.search(r"^\s*title\s*=\s*\{+([^}]+)\}+", block, re.M | re.I)
        if title:
            titles[doi] = title.group(1).strip()
    return titles, dois


def _iter_entries(lines: list[str]):
    """Yield each ``.. [N]`` reference entry as one joined string.

    An entry runs until a blank line, the next entry, or a line that dedents to
    or past the marker — which is how numpydoc continuations are written.
    """
    index = 0
    while index < len(lines):
        match = _ENTRY.match(lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        buffer = [lines[index]]
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor]
            if not line.strip() or _ENTRY.match(line):
                break
            if len(line) - len(line.lstrip()) <= indent:
                break
            buffer.append(line)
            cursor += 1
        yield index + 1, " ".join(buffer)
        index = cursor


def collect(bib_titles: dict[str, str]):
    """Scan ``src`` once and return everything the three checks need.

    Returns
    -------
    cited : dict
        DOI -> set of repo-relative files citing it.
    titled : dict
        DOI -> list of ``(file, line, quoted title)`` from reference entries.
    placeholders : list
        ``(file, line, matched text)`` for fabricated identifiers.
    """
    cited: dict[str, set[str]] = defaultdict(set)
    titled: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    placeholders: list[tuple[str, int, str]] = []

    for path, rel in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        for match in _DOI.finditer(text):
            cited[_norm_doi(match.group(0))].add(rel)

        for lineno, line in enumerate(lines, 1):
            hit = _PLACEHOLDER.search(line)
            if hit:
                placeholders.append((rel, lineno, hit.group(0)))

        for lineno, entry in _iter_entries(lines):
            doi_hit = _DOI.search(entry)
            if not doi_hit:
                continue
            doi = _norm_doi(doi_hit.group(0))
            if doi not in bib_titles:
                continue
            title_hit = _TITLE.search(entry)
            if title_hit:
                titled[doi].append((rel, lineno, title_hit.group(1)))

    return cited, titled, placeholders


def check_coverage(cited, bib_dois) -> list[str]:
    """Every DOI written in ``src`` must appear in ``references.bib``."""
    problems = []
    missing = sorted(d for d in cited if d not in bib_dois and d not in UNCURATED_DOIS)
    for doi in missing:
        where = ", ".join(sorted(cited[doi])[:3])
        problems.append(
            f"DOI not in references.bib: {doi}\n"
            f"    cited in: {where}\n"
            f"    Add the paper to src/tengri/citations/references.bib, or pin it\n"
            f"    in UNCURATED_DOIS with a reason."
        )
    stale = sorted(d for d in UNCURATED_DOIS if d in bib_dois)
    for doi in stale:
        problems.append(
            f"stale pin in UNCURATED_DOIS: {doi} is now in references.bib.\n"
            f"    Remove it from the list — that is what makes the backlog a ratchet."
        )
    unused = sorted(d for d in UNCURATED_DOIS if d not in cited and d not in bib_dois)
    for doi in unused:
        problems.append(
            f"stale pin in UNCURATED_DOIS: {doi} is no longer cited anywhere in src/.\n"
            f"    Remove it from the list."
        )
    return problems


def check_titles(titled, bib_titles) -> list[str]:
    """A cited DOI that is in the bib must carry the bib's title."""
    problems = []
    drifted = set()
    for doi, occurrences in sorted(titled.items()):
        want = _norm_title(bib_titles[doi])
        for rel, lineno, raw in occurrences:
            if _same_paper(_norm_title(raw), want):
                continue
            drifted.add(doi)
            if doi in BIB_TITLE_DRIFT:
                continue
            # Unwrap the citation's line breaks before interpolating: a
            # backslash inside an f-string expression is Python 3.12+ syntax and
            # the package targets 3.11.
            written = re.sub(r"\s+", " ", raw).strip()
            problems.append(
                f"{rel}:{lineno}: title disagrees with references.bib\n"
                f"    doi : {doi}\n"
                f'    src : "{written}"\n'
                f'    bib : "{bib_titles[doi]}"\n'
                f"    Fix the docstring, or pin the DOI in BIB_TITLE_DRIFT."
            )
    stale = sorted(BIB_TITLE_DRIFT - drifted)
    for doi in stale:
        problems.append(
            f"stale pin in BIB_TITLE_DRIFT: {doi} no longer disagrees with the bib.\n"
            f"    Remove it from the list."
        )
    return problems


def check_placeholders(placeholders) -> list[str]:
    """No fabricated identifier may be shipped as a citation."""
    return [
        f"{rel}:{lineno}: placeholder identifier in a citation: {text!r}\n"
        f"    A reference whose digits were never filled in reads as real.\n"
        f"    Resolve it against the publisher, or remove the citation."
        for rel, lineno, text in placeholders
    ]


def _print_baseline(cited, titled, bib_titles, bib_dois) -> None:
    """Print pin literals for the current tree, ready to paste."""
    missing = sorted(d for d in cited if d not in bib_dois)
    print(f"# {len(missing)} uncurated DOIs")
    print("UNCURATED_DOIS = frozenset(\n    {")
    for doi in missing:
        files = sorted(cited[doi])
        # Basename, not the full path: the annotation is a signpost for whoever
        # drains the backlog, and the repo's line limit is 99 columns.
        note = f"  # {Path(files[0]).name}" + (f" +{len(files) - 1}" if len(files) > 1 else "")
        print(f'        "{doi}",{note}')
    print("    }\n)")

    drifted = []
    for doi, occurrences in sorted(titled.items()):
        want = _norm_title(bib_titles[doi])
        if any(not _same_paper(_norm_title(raw), want) for _, _, raw in occurrences):
            drifted.append(doi)
    print(f"\n# {len(drifted)} DOIs whose docstring title disagrees with the bib")
    print("BIB_TITLE_DRIFT = frozenset(\n    {")
    for doi in drifted:
        room = max(12, 92 - len(doi) - len('        "",  # bib: '))
        print(f'        "{doi}",  # bib: {bib_titles[doi][:room].rstrip()}')
    print("    }\n)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="print pin literals for the current tree instead of checking",
    )
    args = parser.parse_args(argv)

    if not BIB.exists():
        print(f"references.bib not found at {BIB}", file=sys.stderr)
        return 1

    bib_titles, bib_dois = _parse_bib(BIB.read_text(encoding="utf-8"))
    cited, titled, placeholders = collect(bib_titles)

    if args.baseline:
        _print_baseline(cited, titled, bib_titles, bib_dois)
        return 0

    problems = (
        check_coverage(cited, bib_dois)
        + check_titles(titled, bib_titles)
        + check_placeholders(placeholders)
    )

    if problems:
        print("Citation/bibliography disagreement:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}\n", file=sys.stderr)
        print(f"{len(problems)} problem(s).", file=sys.stderr)
        return 1

    print(
        f"OK: {len(cited)} DOIs cited in src/, "
        f"{len(bib_dois)} in references.bib, "
        f"{len(UNCURATED_DOIS)} uncurated and {len(BIB_TITLE_DRIFT)} title-drifted "
        f"pinned."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
