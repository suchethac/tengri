#!/usr/bin/env python3
"""CI guard: every repo path named in CLAUDE.md must exist.

CLAUDE.md is not documentation. It opens with *"These instructions OVERRIDE any
default behavior and you MUST follow them exactly as written"*, and every agent
that touches this repository reads it before doing anything else. A path in it
that does not resolve is therefore not a typo — it is a standing instruction to
go look in a place that is not there.

The case that prompted this guard: the ``area:examples`` label read *"sphinx-
gallery scripts under ``docs/examples/``"*. There is no ``docs/examples/``. The
gallery sources live at ``examples/`` and render into ``docs/auto_examples/``,
per ``docs/conf.py``::

    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],

Nothing caught it because the existing doc guard, ``check_doc_examples.py``,
resolves *symbols* — names that must import — and says nothing about paths.

Scope is CLAUDE.md alone, deliberately. ``docs/dev/`` is out of scope for the
same reason ``check_doc_examples.py`` exempts it: design notes and parity
audits legitimately name files that were removed or never built. CLAUDE.md
does not get that latitude, because it is binding.

The rule this encodes: **code markup on a path in CLAUDE.md asserts that the
path exists.** A path being *described* rather than cited — one that was
removed, or never existed — is written as plain text. That convention is what
keeps ``EXEMPT`` empty, and an empty exemption list is a materially stronger
guarantee than a populated one: there is no entry to go stale, and no way for
a genuinely wrong instruction to inherit an excuse written for a different
sentence. It was load-bearing immediately. The CLAUDE.md entry announcing this
guard described the very path that motivated it, in backticks, and the guard
failed on its own documentation — the mirror image of the trap recorded in
``check_test_paths_covered.py``, where a guard read its own comments as
evidence and passed. Both failures have the same root: prose about a thing and
a reference to a thing are not distinguishable to a scanner unless the markup
distinguishes them.

Two roots, in order
-------------------
A token resolves if it exists relative to the repo root **or** to
``src/tengri/``. The second is not a fallback bolted on to make the guard pass
— it is the file's own convention: ``parameters/groups.py``,
``utils/physics_constants.py`` and four others are written package-relative
throughout the "Key conventions" and "Package structure" sections. Resolving
against both roots is what lets the guard be strict about the rest.

Dependencies: standard library only. The ``lint`` job installs ruff and
nothing else, so this must not import ``yaml`` or ``tengri``.

Usage
-----
    python tools/check_claude_md_paths.py

Exit code 0 when every named path resolves; 1 otherwise, listing each dead
path with the line it appears on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
PACKAGE_ROOT = REPO_ROOT / "src" / "tengri"

# Paths named in CLAUDE.md that legitimately do not exist. Each needs a reason;
# a bare entry here is how the next dead path gets waved through.
EXEMPT: dict[str, str] = {}

#: Extensions that make a token unambiguously a file path rather than prose.
_EXTENSIONS = "py|md|rst|toml|yml|yaml|h5|ipynb|cfg|txt|json"

#: A backtick span that is *entirely* one repo path: at least one "/", and it
#: either carries an extension above or ends in "/".
#:
#: Matching the whole span, rather than mining tokens out of it, is what keeps
#: the guard quiet enough to be believed. Extracting sub-tokens flagged eleven
#: false positives for every true one: units (`[erg/s/Hz]` -> "erg/s/",
#: `log10(Z/Zsun)` -> "Z/"), machine paths (`~/.cache/tengri_jax_cache` ->
#: "cache/"), and slash-separated prose (`inputs/outputs/optional_inputs`).
#: A guard with that signal-to-noise ratio gets switched off, and a guard that
#: is switched off protects nothing.
_PATH_SPAN = re.compile(rf"^[A-Za-z_][A-Za-z0-9_./-]*/(?:[A-Za-z0-9_.-]+\.(?:{_EXTENSIONS}))?$")

#: Backtick-delimited code spans.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def _is_repo_relative(token: str) -> bool:
    """False for things that are paths but not *this repo's* paths.

    Home-relative (``~/.cache/...``), absolute, and environment-variable paths
    describe the machine, not the checkout, and cannot be resolved here.
    """
    return not token.startswith(("~", "/", "$", "."))


def named_paths() -> dict[str, int]:
    """Every repo-relative path token in CLAUDE.md, mapped to its first line.

    Known limit: a bare basename (``cue.py``, ``torus.py``) is not checked. It
    is a *reference*, not a path — the reader locates it from the package
    layout — and resolving one means guessing which of the matches was meant.
    ``notebook.py`` in the subagent-zombie note is a placeholder, not a file,
    so guessing would produce a false failure on prose that is correct.
    """
    found: dict[str, int] = {}
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        for span in _CODE_SPAN.findall(line):
            token = span.strip()
            if _PATH_SPAN.match(token) and _is_repo_relative(token) and token not in found:
                found[token] = lineno
    return found


def resolve(token: str) -> Path | None:
    """The root a token resolves against, or None if it resolves against none.

    Repo root is tried first so that a token which is genuinely repo-relative
    is never reported as living in the package.
    """
    for root in (REPO_ROOT, PACKAGE_ROOT):
        if (root / token).exists():
            return root
    return None


def main() -> int:
    if not CLAUDE_MD.is_file():
        print(f"ERROR: cannot read {CLAUDE_MD}", file=sys.stderr)
        return 1

    tokens = named_paths()
    dead = sorted(
        (token, lineno)
        for token, lineno in tokens.items()
        if token not in EXEMPT and resolve(token) is None
    )

    if not dead:
        print(f"OK: all {len(tokens)} paths named in CLAUDE.md resolve.")
        return 0

    print("Paths named in CLAUDE.md that do not exist:\n", file=sys.stderr)
    for token, lineno in dead:
        print(f"  CLAUDE.md:{lineno}  {token}", file=sys.stderr)
    print(
        "\nCLAUDE.md is binding on every agent that reads it, so a dead path is a "
        "standing instruction to look in the wrong place. Fix the path, or — if it "
        "genuinely should not exist — record it in EXEMPT in this file with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
