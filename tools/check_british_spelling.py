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
External data-contract strings keep their upstream spelling because the literal
must match bytes on disk / an upstream API. The only such cases are the
Synthesizer grid HDF5 dataset keys (see ``ALLOWED_TOKENS``). Add new exceptions
there with a one-line justification — never to silence a genuine British
spelling in tengri's own code.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files scanned: same scope that was converted in #819, plus the
# repo-root prose files (README slipped through the original sweep).
DEFAULT_ROOTS = ("src", "tests", "docs", "examples", "notebooks", "README.md", "CONTRIBUTING.md")
SUFFIXES = (".py", ".md", ".rst")
# Excluded path fragments: generated output and archived notebook trees.
EXCLUDE_PARTS = (
    "auto_examples",  # sphinx-gallery generated from examples/
    "_build",  # local sphinx build output under docs/
    "archive",
    "archive_2",
    "_migrated_galleries",
    "_retired",
    "_old_notebooks",
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


def to_american(word: str) -> str | None:
    """American spelling of lowercase British ``word``, or None if not British."""
    if word in INVARIANT:
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
    for tok in TOKEN_RE.finditer(text):
        token = tok.group(0)
        if token in ALLOWED_TOKENS:
            continue
        for sub in SUBWORD_RE.finditer(token):
            american = to_american(sub.group(0).lower())
            if american is None:
                continue
            pos = tok.start() + sub.start()
            line = text.count("\n", 0, pos) + 1
            col = pos - (text.rfind("\n", 0, pos))
            yield line, col, sub.group(0), american


def fix_text(text: str) -> str:
    """Return ``text`` with every British subword rewritten (case-preserving)."""

    def repl_token(m: re.Match) -> str:
        token = m.group(0)
        if token in ALLOWED_TOKENS:
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
        "tools/check_british_spelling.py with a justification."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
