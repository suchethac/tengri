#!/usr/bin/env python3
"""CI guard: every repo path named in docs/dev/verification-protocol.md must exist.

The verification protocol is the file README sends readers to when they ask
"what is safe to use for science?". Its Test File column is the evidence. A
path there that does not resolve is not a typo — it is a citation to a check
that cannot be run.

The case that prompted this guard: the table cited eight test files, and all
eight were absent (test_calzetti.py, test_cue.py, test_inoue_igm.py,
test_skirtor.py, test_charlot_fall.py, test_dsps_roundtrip.py,
test_nifty_vi.py, test_dla.py), alongside a docs/verification.md that was never
written. Every row read PENDING. Meanwhile fifty-six real cross-validation
files sat in tests/crossval/ under different names. The ledger understated the
project for as long as nobody re-typed the filenames (#1725).

``check_claude_md_paths.py`` enforces the same rule for CLAUDE.md and is the
model for this one; ``check_doc_examples.py`` resolves *symbols* and says
nothing about paths. This file is deliberately scoped to the verification
protocol alone. The rest of ``docs/dev/`` keeps its latitude to name removed or
not-yet-built files, because design notes and parity audits legitimately do
that. A ledger of evidence does not get that latitude.

What this guard cannot do
-------------------------
It resolves paths. It does not run tests, and it cannot tell you whether a
cited test executes, passes, or asserts anything meaningful — three of the
files it happily resolves collect zero tests (module-level skips and an
``importorskip``). That is why the protocol carries a ``NOT RUN`` status: a
resolving path and an executed check are different claims, and only the second
is evidence.

Conventions, matching the sibling guard
---------------------------------------
- Only tokens in ``backticks`` are candidates. Prose about a path — one that
  was removed, or never existed — is written as plain text, and that convention
  is what keeps the exemption list empty.
- Only tokens containing ``/`` are treated as paths. A bare basename
  (``cue.py``, ``test_calzetti.py``) is a reference, not a path.
- Tokens containing whitespace are not paths. This file's own status vocabulary
  includes ``PARTIAL (n/m)``, which carries a slash and is not a filename; no
  path in this repository has a space in it.
- Fenced code blocks are skipped: they hold shell commands, not citations.
- A token resolves against the repo root **or** against ``src/tengri/``.

Dependencies: standard library only. The ``lint`` job installs ruff and nothing
else, so this must not import ``yaml`` or ``tengri``.

Usage
-----
    python tools/check_verification_protocol_paths.py

Exit code 0 when every named path resolves; 1 otherwise, listing each dead path
with the line it appears on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "dev" / "verification-protocol.md"

#: Both roots a path may resolve against. ``src/tengri/`` is second because the
#: doc, like CLAUDE.md, writes some paths package-relative.
ROOTS = (REPO_ROOT, REPO_ROOT / "src" / "tengri")

#: Inline-code spans. Non-greedy so adjacent spans on one line stay separate.
_BACKTICKED = re.compile(r"`([^`\n]+)`")

#: A fenced block opener/closer. Content between them is shell, not citation.
_FENCE = re.compile(r"^\s*```")

#: Trailing punctuation Markdown prose leaves attached to a path.
_TRAILING = ",.;:)"


def _candidate_paths(text: str) -> list[tuple[int, str]]:
    """Every backticked, slash-bearing token outside fenced code blocks."""
    found: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for token in _BACKTICKED.findall(line):
            token = token.strip().rstrip(_TRAILING)
            if any(c.isspace() for c in token):
                continue
            if "/" in token and not token.startswith(("http://", "https://")):
                found.append((lineno, token))
    return found


def _resolves(token: str) -> bool:
    return any((root / token).exists() for root in ROOTS)


def main() -> int:
    if not DOC.exists():
        print(f"ERROR: {DOC.relative_to(REPO_ROOT)} not found", file=sys.stderr)
        return 1

    candidates = _candidate_paths(DOC.read_text(encoding="utf-8"))
    dead = [(lineno, token) for lineno, token in candidates if not _resolves(token)]

    if dead:
        rel = DOC.relative_to(REPO_ROOT)
        print(f"{len(dead)} path(s) named in {rel} do not exist:\n", file=sys.stderr)
        for lineno, token in dead:
            print(f"  {rel}:{lineno}: {token}", file=sys.stderr)
        print(
            "\nA path in backticks asserts that the path exists. If you are "
            "describing a path rather than citing one — a file that was removed, "
            "or never written — write it as plain text.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(candidates)} path(s) named in verification-protocol.md all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
