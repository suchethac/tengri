#!/usr/bin/env python3
"""CI guard for American-English spelling (NAMING_CONTRACT.md §10).

Flags British spellings in identifiers and prose across the visible source set
(src/, tests/, hand-written docs, gallery examples, and active notebooks). The
project standardized on American English in #819; this guard keeps it that way.

Detection is closed-vocabulary and case/identifier aware: every identifier token
is split into subwords (snake_case, camelCase, and ALL_CAPS aware), so both
``normalise_flux`` and ``MarginalisedLikelihood`` are caught — not only
standalone prose words. A subword is a violation when its American spelling
differs from the British one. An INVARIANT set guards American-invariant words
(``noise``, ``raise``, ``exercise``, ``analyses``-noun, the ``-wise`` suffix,
the matplotlib ``Greys`` colormap) against false positives.

Usage
-----
    python tools/check_british_spelling.py            # scan default roots (CI mode)
    python tools/check_british_spelling.py --fix      # rewrite British -> American
    python tools/check_british_spelling.py --root docs/dev

Exit code 0 if no British spellings are found; non-zero with violations listed
otherwise. ``--fix`` rewrites files in place (case-preserving) and exits 0.

Allowlist
---------
Two escapes, and which one you pick matters:

``ALLOWED_TOKENS``
    Whole tokens that must keep their upstream spelling because the literal has
    to match bytes on disk or an upstream API — the Synthesizer grid HDF5 keys,
    and title-case words distinctive enough to be unambiguous.

``ALLOWED_PHRASES``
    Multi-word phrases, checked positionally. Use this whenever the British word
    is an **ordinary English word** — a verbatim paper title, say. A token entry
    for a common word exempts it repo-wide: a bare ``"modelling"`` added for one
    citation let three unrelated prose uses through and turned the guard's own
    "this word is flagged" test red.

Add new exceptions with a one-line justification — never to silence a genuine
British spelling in tengri's own code.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files scanned: same scope that was converted in #819, plus the repo-root prose
# files (README slipped through the original sweep) and the GitHub-visible
# community files.
#
# The `.github` tree and the root prose files were outside this list until
# 2026-08-18, which is why "Expected behaviour" sat in the bug-report template
# and "labelled" in GOVERNANCE: the rule was repo-wide but the guard's *domain*
# was not, so nothing ever looked. Adding a file to the repo root or to
# `.github/` should mean it is covered by default.
#
# CHANGELOG.md is deliberately NOT scanned. It is a historical record, and many
# of its entries document renames whose *old* name was British-spelled
# (`rest_frame_colour()` -> `rest_frame_color()`). Normalising it rewrites both
# sides of the arrow and turns the entry into `X -> X`, destroying the very
# thing the line exists to record. Measured: --fix produced 23 such entries.
DEFAULT_ROOTS = (
    "src",
    "tests",
    "docs",
    "examples",
    "notebooks",
    ".github",
    "README.md",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "GOVERNANCE.md",
    "CONTRIBUTORS.md",
)
SUFFIXES = (".py", ".md", ".rst")
# Excluded path fragments: generated output and archived notebook trees.
# Matching is exact-on-path-part (see the `path.parts` test below), so each archived
# tree needed its own entry while they sat side by side under `notebooks/`. They are
# now all under `notebooks/archive/`, so the single "archive" part covers every one —
# `archive_2`, `_migrated_galleries`, `_retired`, and `_old_notebooks` were dropped
# rather than left behind as parts no path can match.
EXCLUDE_PARTS = (
    "auto_examples",  # sphinx-gallery generated from examples/
    "_build",  # local sphinx build output under docs/
    "archive",
)

# Rename-ledger docs: they intentionally cite the *old* British names in
# "old -> new" tables, so they are exempt from the scan (repo-relative, POSIX).
EXCLUDE_FILES = frozenset(
    {
        "docs/changelog.md",
        "docs/dev/NAMING_CONTRACT.md",
        # This guard's own test fixtures are British by design.
        "tests/unit/test_check_british_spelling.py",
    }
)

# External data-contract tokens that must keep British spelling (bytes on disk).
ALLOWED_TOKENS = frozenset(
    {
        "ionisation_parameter",  # Synthesizer grid HDF5 axis key
        "log10_specific_ionising_luminosity",  # Synthesizer grid HDF5 dataset key
        "Modelling",  # verbatim paper title — Temple, Hewett & Banerji 2021, MNRAS 508, 737
    }
)

# Verbatim phrases that must keep British spelling, scoped to the phrase.
#
# Use this, NOT ``ALLOWED_TOKENS``, whenever the British word is an ordinary
# English word rather than a data-contract key. A bare ``"modelling"`` entry was
# added here for the Tacchella+2020 title and exempted the word *everywhere* —
# three unrelated prose uses sailed through the guard, and
# ``test_british_words_are_flagged[modelling-modeling]`` went red because the
# guard no longer flagged a word it is supposed to flag. A token allowlist
# cannot express "only inside this citation"; a phrase can.
ALLOWED_PHRASES = (
    # Tacchella, Forbes & Caplar 2020, "Stochastic modelling of star-formation
    # histories II", MNRAS. DOI 10.1093/mnras/staa1838, arXiv:2006.09382.
    # MNRAS is a British journal and paper II's published title uses -lling;
    # paper I's uses -ling. Both are reproduced exactly, per the citation rule
    # in CLAUDE.md ("never write citations from memory").
    "modelling of star-formation histories",
    # Temple, Hewett & Banerji 2021, "QSOgen: a model of the UV-to-submillimetre
    # spectral energy distributions of quasars", MNRAS 508, 737.
    # DOI 10.1093/mnras/stab2586. MNRAS is a British journal and references.bib
    # reproduces the published title exactly, so
    # tools/check_citation_bib_coverage.py folds the spelling rather than
    # reporting it as title drift -- and its test must write the British form to
    # prove the fold works.
    "UV-to-submillimetre spectral energy",
    # The four strings GitHub Actions reports for `needs.<job>.result`, declared
    # once as `GITHUB_RESULTS` in tools/ci_ok.py. An upstream API vocabulary, so
    # it takes the same NAMING_CONTRACT §10 exemption as the Synthesizer HDF5
    # keys -- but "cancelled" is an ordinary English word, so a token entry would
    # exempt it repo-wide and blind the guard to real British prose. That is the
    # exact failure the ALLOWED_TOKENS note above records. Scoped to the tuple
    # instead, which is also why ci_ok.py declares it once and its test derives
    # the values rather than repeating them.
    '"failure", "cancelled", "skipped"',
    # Verbatim citation titles rendered in docs/model_reference
    "Two-metre Sky Survey",
    "modelling framework THEMIS",
    "spectral energy distribution modelling due to bursty",
    "modelling AGN and galaxy SEDs from radio to X-rays",
)


def allowed_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges in ``text`` covered by an :data:`ALLOWED_PHRASES` match.

    Parameters
    ----------
    text : str
        File contents to scan.

    Returns
    -------
    spans : list of (int, int)
        Half-open ``(start, end)`` character offsets, unsorted and possibly
        overlapping.
    """
    spans = []
    for phrase in ALLOWED_PHRASES:
        start = text.find(phrase)
        while start != -1:
            spans.append((start, start + len(phrase)))
            start = text.find(phrase, start + 1)
    return spans


# ---- American-invariant words the rules would otherwise mangle ----
INVARIANT = frozenset(
    {
        # -ise words that are American English, not British variants. The -ise
        # -> -ize rule mangles them: it "corrects" disguise -> disguize.
        "disguise",
        "disguised",
        "disguises",
        "noise",
        "noised",
        "noises",
        "raise",
        "raised",
        "raises",
        "raising",
        "rise",
        "risen",
        "rises",
        "rising",
        "arise",
        "arisen",
        "arises",
        "arising",
        "otherwise",
        "likewise",
        "piecewise",
        "pairwise",
        "pointwise",
        "stepwise",
        "bitwise",
        "elementwise",
        "componentwise",
        "columnwise",
        "rowwise",
        "clockwise",
        "anticlockwise",
        "wise",
        "exercise",
        "exercised",
        "exercises",
        "exercising",
        # Same gap "advisable" had below: the family was listed but its -able
        # derivative was not, so the checker demanded the non-word "exercizable".
        "exercisable",
        # "advise" is -ise in American English too, exactly like the "exercise",
        # "surprise" and "revise" families already listed here. It was simply
        # missing, so the checker demanded the non-word "advized".
        "advise",
        "advised",
        "advises",
        "advising",
        "advisable",
        "precise",
        "concise",
        "surprise",
        "surprised",
        "surprises",
        "surprising",
        "revise",
        "revised",
        "revises",
        "revising",
        "promise",
        "promised",
        "promises",
        "comprise",
        "comprised",
        "comprises",
        "comprising",
        "supervise",
        "supervised",
        "supervising",
        "supervises",
        "advertise",
        "advertised",
        "advertises",
        "advertising",
        "disable",
        "disables",
        "disabled",
        "devise",
        "devised",
        "devises",
        "demise",
        "expertise",
        "paradise",
        "franchise",
        "merchandise",
        "gawiser",
        "improvise",
        "improvised",
        "improvises",
        "improvising",
        "apprise",
        "apprised",
        "surmise",
        "surmised",
        "premise",
        "premises",
        "chastise",
        "enterprise",
        "enterprises",
        "reprise",
        "incise",
        "incised",
        # Missing inflections of verbs already listed above. The -ise -> -ize rule is
        # applied per *word form*, not per lemma, so a family with a gap fails on
        # exactly the absent form and nowhere else: "promise", "promised" and
        # "promises" were all present, so the checker passed on those and demanded
        # the non-word "promizing" for the participle alone. Same shape as the
        # "advise"/"advising" gap noted above, which is why the fix is the family and
        # not the one word that happened to surface.
        "promising",
        "devising",
        "disguising",
        "premised",
        "premising",
        "surmises",
        "surmising",
        "apprises",
        "apprising",
        "enterprising",
        "franchised",
        "franchises",
        "franchising",
        "merchandising",
        "chastised",
        "chastises",
        "chastising",
        "reprised",
        "reprises",
        "incises",
        "incising",
        "analyses",  # plural noun of 'analysis' (invariant) — protect the noun
        "four",
        "hour",
        "hours",
        "your",
        "yours",
        "sour",
        "tour",
        "tours",
        "contour",
        "contours",
        "flour",
        "pour",
        "pours",
        "velour",
        "detour",
        "devour",
        "glamour",
        "our",
        "astres",
        "feltre",
        "acre",
        "acres",
        "genre",
        "genres",
        "ogre",
        "ogres",
        "are",
        "centric",
    }
)

# ---- explicit small families (whitelist-based, safer than blocklist) ----
OUR = {
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "behavioural": "behavioral",
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "colouring": "coloring",
    "colourful": "colorful",
    "colourbar": "colorbar",
    "colourmap": "colormap",
    "colourmaps": "colormaps",
    "multicolour": "multicolor",
    "favour": "favor",
    "favours": "favors",
    "favoured": "favored",
    "favouring": "favoring",
    "favourite": "favorite",
    "favourites": "favorites",
    "flavour": "flavor",
    "flavours": "flavors",
    "flavoured": "flavored",
    "neighbour": "neighbor",
    "neighbours": "neighbors",
    "neighbouring": "neighboring",
    "neighbourhood": "neighborhood",
    "honour": "honor",
    "honours": "honors",
    "honoured": "honored",
    "honouring": "honoring",
    "labour": "labor",
    "labours": "labors",
    "vapour": "vapor",
    "vapours": "vapors",
    "rumour": "rumor",
    "harbour": "harbor",
    "odour": "odor",
    "odours": "odors",
}
TRE = {
    "centre": "center",
    "centres": "centers",
    "centred": "centered",
    "centring": "centering",
    "nanometre": "nanometer",
    "nanometres": "nanometers",
    "micrometre": "micrometer",
    "micrometres": "micrometers",
    "millimetre": "millimeter",
    "millimetres": "millimeters",
    "submillimetre": "submillimeter",
    "submillimetres": "submillimeters",
    "kilometre": "kilometer",
    "kilometres": "kilometers",
    "metre": "meter",
    "metres": "meters",
    "litre": "liter",
    "litres": "liters",
    "fibre": "fiber",
    "fibres": "fibers",
    "calibre": "caliber",
    "theatre": "theater",
    "spectre": "specter",
}
OGUE = {
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "catalogued": "cataloged",
    "cataloguing": "cataloging",
    "analogue": "analog",
    "analogues": "analogs",
    "dialogue": "dialog",
    "dialogues": "dialogs",
}
DLL = {
    "modelling": "modeling",
    "modelled": "modeled",
    "modeller": "modeler",
    "unmodelled": "unmodeled",
    "labelling": "labeling",
    "labelled": "labeled",
    "labeller": "labeler",
    "relabelling": "relabeling",
    "relabelled": "relabeled",
    "signalling": "signaling",
    "signalled": "signaled",
    "cancelling": "canceling",
    "cancelled": "canceled",
    "travelling": "traveling",
    "travelled": "traveled",
    "levelling": "leveling",
    "levelled": "leveled",
    "fuelling": "fueling",
    "fuelled": "fueled",
    "marvelling": "marveling",
    "marvelled": "marveled",
}
GREY = {
    "grey": "gray",
    "greyer": "grayer",
    "greyest": "grayest",
    "greying": "graying",
    "greybody": "graybody",
    "greyscale": "grayscale",
    "greyish": "grayish",  # 'greys'/'Greys' omitted (matplotlib colormap)
}
MISC = {
    "defence": "defense",
    "defences": "defenses",
    "offence": "offense",
    "offences": "offenses",
}

ISE_SUFFIX = [
    ("isations", "izations"),
    ("isation", "ization"),
    ("isabilities", "izabilities"),
    ("isability", "izability"),
    ("isable", "izable"),
    ("isers", "izers"),
    ("iser", "izer"),
    ("ising", "izing"),
    ("ised", "ized"),
    ("ises", "izes"),
    ("ise", "ize"),
]
YSE_SUFFIX = [
    ("ysing", "yzing"),
    ("ysed", "yzed"),
    ("yser", "yzer"),
    ("yses", "yzes"),
    ("yse", "yze"),
]


# Invariant ``-ise`` words, longest first, for the suffix test in ``_is_invariant``.
# Deliberately NOT the whole INVARIANT set: it also protects ``-our`` nouns
# (``our``, ``hour``, ``four``), and matching those as tails would exempt
# ``colour`` from the OUR -> OR rule. ``-yse`` is left out for the same reason —
# ``analyses`` is invariant, but ``catalyses`` must still become ``catalyzes``.
_ISE_TAILS = tuple(
    sorted(
        (w for w in INVARIANT if any(w.endswith(suf) for suf, _ in ISE_SUFFIX) and len(w) >= 4),
        key=len,
        reverse=True,
    )
)


def _is_invariant(word: str) -> bool:
    """True if ``word`` is American-invariant, including prefixed forms.

    The ISE rule is decided by the *tail* of a word, so the exemption has to be
    decided by the tail too. An exact-match table leaves a fresh gap at every
    prefixed and inflected form, which is how this checker came to demand the
    non-words ``advized``, ``promizing`` and ``unsurprizing`` — three one-word
    patches to one missing rule. Matching on the tail covers ``unsurprising``
    from ``surprising`` and ``sunrise`` from ``rise`` without a new entry.
    """
    if word in INVARIANT:
        return True
    return any(word != tail and word.endswith(tail) for tail in _ISE_TAILS)


def to_american(word: str) -> str | None:
    """American spelling of lowercase British ``word``, or None if not British."""
    if _is_invariant(word):
        return None
    for table in (OUR, TRE, OGUE, DLL, GREY, MISC):
        if word in table:
            return table[word]
    for suf, rep in ISE_SUFFIX:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)] + rep
    for suf, rep in YSE_SUFFIX:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)] + rep
    return None


# Full identifier tokens (snake_case stays joined so the allowlist can match).
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Subword split: ALL_CAPS run, Capitalized word, or lowercase run.
SUBWORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+")


def _case_preserve(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def scan_text(text: str):
    """Yield (line, col, british, american) violations for one file's text."""
    spans = allowed_spans(text)
    for tok in TOKEN_RE.finditer(text):
        token = tok.group(0)
        if token in ALLOWED_TOKENS:
            continue
        for sub in SUBWORD_RE.finditer(token):
            american = to_american(sub.group(0).lower())
            if american is None:
                continue
            pos = tok.start() + sub.start()
            if any(lo <= pos < hi for lo, hi in spans):
                continue
            line = text.count("\n", 0, pos) + 1
            col = pos - (text.rfind("\n", 0, pos))
            yield line, col, sub.group(0), american


def fix_text(text: str) -> str:
    """Return ``text`` with every British subword rewritten (case-preserving).

    Honors :data:`ALLOWED_PHRASES` as well as :data:`ALLOWED_TOKENS` — otherwise
    ``--fix`` would silently rewrite the verbatim citation that the scan
    deliberately skips, and the next scan would pass on corrupted text.
    """
    spans = allowed_spans(text)

    def repl_token(m: re.Match) -> str:
        token = m.group(0)
        if token in ALLOWED_TOKENS:
            return token
        if any(lo <= m.start() < hi for lo, hi in spans):
            return token

        def repl_sub(s: re.Match) -> str:
            american = to_american(s.group(0).lower())
            return _case_preserve(s.group(0), american) if american else s.group(0)

        return SUBWORD_RE.sub(repl_sub, token)

    return TOKEN_RE.sub(repl_token, text)


def iter_files(roots):
    for root in roots:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        if root_path.is_file():
            yield root_path
            continue
        for path in sorted(root_path.rglob("*")):
            if path.suffix not in SUFFIXES:
                continue
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            if path.relative_to(REPO_ROOT).as_posix() in EXCLUDE_FILES:
                continue
            # notebooks/: only the top-level + tutorials are in scope.
            if path.parts[len(REPO_ROOT.parts)] == "notebooks":
                rel = path.relative_to(REPO_ROOT / "notebooks")
                if len(rel.parts) > 1 and rel.parts[0] != "tutorials":
                    continue
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Repository-relative path to scan (repeatable). "
        f"Defaults: {', '.join(DEFAULT_ROOTS)}.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite British -> American spellings in place (case-preserving).",
    )
    args = parser.parse_args()
    roots = args.roots or list(DEFAULT_ROOTS)

    if args.fix:
        fixed = 0
        for path in iter_files(roots):
            text = path.read_text(encoding="utf-8", errors="ignore")
            new = fix_text(text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                fixed += 1
        print(f"fixed {fixed} file(s)")
        return 0

    violations: list[tuple[Path, int, int, str, str]] = []
    for path in iter_files(roots):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line, col, british, american in scan_text(text):
            violations.append((path.relative_to(REPO_ROOT), line, col, british, american))

    if not violations:
        print("OK: no British spellings found (NAMING_CONTRACT §10)")
        return 0

    print(f"FAIL: {len(violations)} British spelling(s) found (NAMING_CONTRACT §10)\n")
    for path, line, col, british, american in violations:
        print(f"  {path}:{line}:{col}  {british} -> {american}")
    print(
        "\nFix: use American spelling. If the string is an external data-contract "
        "key that must keep British spelling, add it to ALLOWED_TOKENS in "
        "(or, for a verbatim citation or other prose phrase, ALLOWED_PHRASES in) "
        "tools/check_british_spelling.py with a justification."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
