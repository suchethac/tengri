#!/usr/bin/env python3
"""CI guard: tengri implements models, it does not port them.

tengri writes its own implementations of published models and checks them
against the reference codes. It does not port, copy, or adapt anyone else's
source. This guard fails the build when the prose says otherwise.

What to write instead
---------------------
For code, name the model and the check, and drop the provenance verb:
"Implements the same model as Prospector (Johnson et al. 2021 [N]_);
validated against it." Keep every citation.

For data, the word is "repackaged". An SSP grid or template table converted
to a tengri format really is the other project's published data, so say so.

Inside tengri a physics block is a component, not a port.

How it detects
--------------
Bare "port" and "copied" have too many honest uses to grep for directly, so
there are two tiers:

1. ``BANNED_PHRASES`` catches unambiguous claims: "ported from", "copied
   from", "faithful port", and friends.
2. A proximity rule catches the bare word "port" on a line that also names
   one of ``REFERENCE_CODES``. This is what finds "ProSpect port" and
   "SKIRTOR port", where the phrase alone looks innocent.

Every pattern is ``\\b``-anchored on the left. Without that, "ported from"
matches inside "exported from" and "supported from", which is most of this
codebase.

Both tiers also read across a single line break. Docstrings here wrap at 99
columns and markdown at about 79, so "ported" ends one line and "from CIGALE"
starts the next often enough to matter, and a scan that reads one line at a
time sees two innocent halves. Only evidence that actually crosses the break
is reported that way, so nothing is counted twice. A blank line is a
paragraph break rather than a wrap, and for the fuzzier proximity tier a line
ending in a full stop is not a wrap either.

Usage
-----
    python tools/check_reimplementation_language.py           # CI mode
    python tools/check_reimplementation_language.py --root docs

Exits 0 when clean, 1 with the violations listed. There is no ``--fix``,
because the replacement depends on whether the subject is code or data and
only the author knows which.

Allowlist
---------
``EXCLUDE_FILES`` covers the files that have to quote the banned wording: the
rule statements themselves, the licensing audit that exists to ask this exact
question, and files describing an external project's own history rather than
tengri's. ``EXCLUDE_DIRS`` covers upstream copyright headers. Add to either
with a reason. Never add a file to hide a real claim that tengri copied
someone's work.
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ROOTS = (
    "src",
    "tests",
    "docs",
    "examples",
    "notebooks",
    "reproduction",
    "bench",
    "scripts",
    "tools",
    "README.md",
    "NOTICE",
    "CLAUDE.md",
    "data/README.md",
)
SUFFIXES = (".py", ".md", ".rst", ".bib", ".ipynb", "")
# Generated output and frozen archives: not worth gating a PR on.
EXCLUDE_PARTS = (
    "auto_examples",  # sphinx-gallery generated from examples/
    "_build",  # local sphinx build output
    "archive",
    "archive_2",
    "_retired",
    "_old_notebooks",
    "superpowers",  # historical planning records, frozen as written
)

EXCLUDE_FILES = frozenset(
    {
        # The rule statements themselves must quote the banned wording.
        "CLAUDE.md",
        "docs/dev/docstring-standard.md",
        "docs/dev/NAMING_CONTRACT.md",
        "reproduction/CONTRACT.md",
        "tools/check_reimplementation_language.py",
        "tests/unit/test_check_reimplementation_language.py",
        # Asking "is any of this a port?" is this document's entire purpose.
        "docs/dev/audits/upstream-code-licensing.md",
        # BSD-3 was chosen partly to preclude porting GPL code; saying so is policy.
        "docs/adr/0002-license-bsd3.md",
        # RELAGN's nthcomp descends from XSpec donthcomp.f. That is RELAGN's
        # lineage to state, not a claim about tengri's code.
        "src/tengri/components/agn/_nthcomp.py",
        "tests/crossval/test_nthcomp_relagn_crossval.py",
        "scripts/build_nthcomp_templates.py",
        "docs/known_bugs.md",
    }
)

# Directories whose files carry upstream copyright headers verbatim. Changing a
# copyright notice is the copyright holder's call, never a style sweep's.
EXCLUDE_DIRS = ("src/tengri/inference/backends/nested",)

# See the module docstring for why every pattern is \b-anchored on the left.
BANNED_PHRASES = (
    r"\bported\s+from",
    r"\bports\s+from",
    r"\bporting\s+credit",
    r"\bport(?:ed)?\s+(?:in)?to\s+(?:jax|python|tengri)",
    r"\b(?:faithful|direct|exact|straight|line-for-line|bit-faithful)\s+port\b",
    r"\ba\s+port\s+of",
    r"\bis\s+a\s+port\b",
    r"\bcopied\s+from\b",
    r"\badapted\s+from\b",
    r"\btranslated\s+from\b",
    r"\btransliterat(?:ed|ion)\s+(?:of|from)",
)
BANNED_RE = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)

# Reference codes tengri validates against. "port" beside one of these names is
# describing provenance, whatever the surrounding sentence claims.
REFERENCE_CODES = (
    "cigale",
    "prospector",
    "prospect",
    "agnfitter",
    "bagpipes",
    "synthesizer",
    "fsps",
    "dsps",
    "nifty",
    "blackjax",
    "grahsp",
    "relagn",
    "xspec",
    "eazy",
    "fastspecfit",
    "skirtor",
    "mappings",
    "cloudy",
)
PORT_WORD_RE = re.compile(r"\bport(?:s|ed|ing)?\b", re.IGNORECASE)
REFERENCE_RE = re.compile("|".join(REFERENCE_CODES), re.IGNORECASE)
# Network ports and hostnames: "port=3306", "port: 3306", "icg.port.ac.uk".
NETWORK_PORT_RE = re.compile(r"port\s*[=:]|\.port\.|_PORT\b", re.IGNORECASE)


def _scan_line(line: str):
    """Return a reason if this one line states provenance, else None."""
    match = BANNED_RE.search(line)
    if match:
        return f'banned phrase "{match.group(0)}"'
    if NETWORK_PORT_RE.search(line):
        return None
    if PORT_WORD_RE.search(line) and REFERENCE_RE.search(line):
        return '"port" on a line naming a reference code'
    return None


def _scan_wrap(first: str, second: str):
    """Return a reason if a claim straddles the break between two lines.

    Docstrings here wrap at 99 columns and markdown at about 79, so a phrase
    as short as "ported from" lands on a line break often. Read one line at a
    time, "ported" and "from CIGALE" are both innocent.

    Only evidence that actually crosses the break is reported; anything
    contained in one line is the caller's job. A blank line is a paragraph
    break rather than a wrap, and so is a line that ends its sentence.
    """
    if not first.strip() or not second.strip():
        return None
    head, tail = first.rstrip(), second.lstrip()
    joined = f"{head} {tail}"
    boundary = len(head)

    match = BANNED_RE.search(joined)
    if match and match.start() < boundary < match.end():
        return f'banned phrase "{match.group(0)}" wrapped across lines'

    # A finished sentence is not a wrap. This matters only for the proximity
    # rule, whose two halves are ordinary words that co-occur by chance;
    # a split banned phrase cannot span a full stop in the first place.
    if head.endswith((".", "!", "?", ":", ";")) or NETWORK_PORT_RE.search(joined):
        return None
    across = (PORT_WORD_RE.search(head) and REFERENCE_RE.search(tail)) or (
        REFERENCE_RE.search(head) and PORT_WORD_RE.search(tail)
    )
    if across:
        return '"port" wrapped onto a line naming a reference code'
    return None


def scan_text(text: str):
    """Yield (line_no, line, reason) for each violation in one file's text."""
    lines = text.splitlines()
    flagged = set()

    for i, line in enumerate(lines, start=1):
        reason = _scan_line(line)
        if reason:
            flagged.add(i)
            yield i, line.strip(), reason

    for i, (first, second) in enumerate(itertools.pairwise(lines), start=1):
        if i in flagged or i + 1 in flagged:
            continue  # already reported on one of the two lines
        reason = _scan_wrap(first, second)
        if reason:
            yield i, f"{first.strip()} / {second.strip()}", reason


def iter_files(roots):
    for root in roots:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        candidates = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            if rel in EXCLUDE_FILES or rel.startswith(EXCLUDE_DIRS):
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
    args = parser.parse_args()

    violations = []
    for path in iter_files(args.roots or list(DEFAULT_ROOTS)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line, reason in scan_text(text):
            violations.append((path.relative_to(REPO_ROOT), line_no, line, reason))

    if not violations:
        print("OK: no port/copy language found (tengri implements, it does not port)")
        return 0

    print(f"FAIL: {len(violations)} port/copy phrase(s) found\n")
    for path, line_no, line, reason in violations:
        print(f"  {path}:{line_no}  {reason}")
        print(f"      {line[:120]}")
    print(
        "\nFix: describe what tengri does, not where it came from.\n"
        "  code  -> 'Implements the same model as X (Author+Year); validated against it.'\n"
        '  data  -> "X\'s published grid, repackaged in the DSPS HDF5 layout"\n'
        "If the line states an *external* project's own lineage, or is a copyright\n"
        "notice, add the file to EXCLUDE_FILES in this script with a justification."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
