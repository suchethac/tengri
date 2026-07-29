#!/usr/bin/env python3
"""CI guard for reimplementation language (CLAUDE.md "Documentation" section).

tengri independently implements published models and validates them against the
reference codes. It does not port, copy, or adapt anyone else's source. This
guard keeps the prose honest: it fails when tengri's own code, docs, or
notebooks are described as ported/copied/adapted from another codebase.

The rule this enforces
----------------------
- Components and algorithms: *"Implements the same model as Prospector (Johnson
  et al. 2021 [N]_); validated against it."* No provenance verbs, every
  citation kept.
- External template/SSP **data** files converted to a tengri format are
  **repackaged**, with attribution. That is the honest word: the bytes really
  are the reference project's published data, used as matched inputs.
- Internally a physics block is a **component**, never a "port".

Detection is two-tier, because bare "port" and "copied" have many legitimate
uses (network ports, ``deepcopy``, an adapted MCMC step size):

1. ``BANNED_PHRASES`` — unambiguous provenance claims ("ported from",
   "faithful port", "copied from", ...). Flagged anywhere in scope.
2. ``PORT_NEAR_REFERENCE`` — the bare word port/ports/ported/porting on a line
   that also names a reference code. Catches "ProSpect port", "SKIRTOR port".

Usage
-----
    python tools/check_reimplementation_language.py           # CI mode
    python tools/check_reimplementation_language.py --root docs

Exit code 0 when clean, 1 with violations listed otherwise. There is no
``--fix``: the right replacement depends on whether the subject is code (say
"implements") or data (say "repackaged"), and a machine cannot tell.

Allowlist
---------
``EXCLUDE_FILES`` holds files that must state the banned wording: the rule
statements themselves, the licensing audit (whose whole job is to ask the
port-versus-reimplementation question), and files recording an *external*
project's own lineage, which is their history to describe and not ours.
Add new entries with a one-line justification. Never add a file to silence a
genuine claim that tengri copied someone's work.
"""

import argparse
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

# Every pattern is \b-anchored on the left. Without it "ported\s+from" fires
# inside "exported from" / "supported from", which is most of this codebase.
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


def scan_text(text: str):
    """Yield (line_no, line, reason) for each violation in one file's text."""
    for i, line in enumerate(text.splitlines(), start=1):
        match = BANNED_RE.search(line)
        if match:
            yield i, line.strip(), f'banned phrase "{match.group(0)}"'
            continue
        if NETWORK_PORT_RE.search(line):
            continue
        if PORT_WORD_RE.search(line) and REFERENCE_RE.search(line):
            yield i, line.strip(), '"port" on a line naming a reference code'


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
