#!/usr/bin/env python3
"""Advisory report: developer docs that name API the package no longer has.

Owner decision: this report is ADVISORY and is never a CI gate. Run it by
hand before tagging a release and fix what it finds; it exits 0 whatever
it finds.

Motivation
----------
``docs/dev/`` is excluded from the published docs and from the guards that
check the public pages, so it can keep teaching API that was deleted
releases ago. ``check_doc_examples.py`` covers the public pages only. This
report covers the developer pages by resolving every backticked identifier
and every dotted ``tengri.*`` name against the live package.

What it reports
---------------
For each name that does not resolve:

``DELETED``
    the name appeared in ``src/`` at some point in history, and
    ``git log -S`` shows the commit that removed it.
``NEVER-EXISTED``
    no commit in ``src/`` ever added the name. Usually prose or a typo.
``IN-SRC-NOT-EXPORTED``
    the name is still in ``src/`` but ``tengri`` does not export it. Often
    a private helper that the docs should not name.

Names are only checked when they look like API: a ``tengri.*`` dotted name,
a dotted name whose head resolves to a tengri object, or a bare name that
is CamelCase or contains an underscore. Single lowercase words and names
headed by third-party modules (``np.``, ``jax.``) are prose and are skipped.

Usage
-----
::

    python tools/report_doc_rot.py > doc_rot.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_DOCS = REPO_ROOT / "docs" / "dev"

# Same pattern as INLINE_CODE in tools/check_doc_examples.py; the tools do not import each other.
INLINE_CODE = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")


def _resolve_step(obj, attr: str):
    """Return ``getattr(obj, attr)`` or None; mirrors check_doc_examples.public_api."""
    try:
        return getattr(obj, attr)
    except AttributeError:
        return None
    except Exception:
        # A deliberate __getattr__ trap for a removed name, not a violation.
        return None


def _walk(tengri, parts: list[str]) -> tuple[bool, str | None]:
    """Walk ``parts`` from the tengri module; return (resolved, first failing part)."""
    obj = tengri
    for part in parts:
        nxt = _resolve_step(obj, part)
        if nxt is None:
            return False, part
        obj = nxt
    return True, None


def check_name(tengri, span: str) -> tuple[str, str | None] | None:
    """Return (status, search_term) for an unresolved API-like name, else None.

    ``status`` is ``None`` when the name resolves or is not API-like.
    """
    text = span[:-2] if span.endswith("()") else span
    if not IDENT.match(text):
        return None
    parts = text.split(".")
    head = parts[0]
    if head == "tengri":
        ok, failing = _walk(tengri, parts[1:])
    elif _resolve_step(tengri, head) is not None:
        ok, failing = _walk(tengri, parts)
    elif len(parts) == 1 and (head[0].isupper() or "_" in head):
        ok, failing = False, head
    else:
        return None
    if ok:
        return None
    return ("unresolved", failing)


def git_status(term: str, cache: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Classify ``term`` by its history in ``src/``; cached per term."""
    if term in cache:
        return cache[term]
    present = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "grep", "-q", "-w", "-e", term, "--", "src"],
        capture_output=True,
    )
    if present.returncode == 0:
        result = ("IN-SRC-NOT-EXPORTED", "")
    else:
        log = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", f"-S{term}", "--format=%h %s", "-1", "--", "src"],
            capture_output=True,
            text=True,
        )
        commit = log.stdout.strip()
        result = ("DELETED", commit) if commit else ("NEVER-EXISTED", "")
    cache[term] = result
    return result


def scan_file(path: Path, tengri, cache: dict) -> list[tuple[int, str, str, str]]:
    """Return (line, span, status, detail) for each finding in one dev page."""
    findings = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        for match in INLINE_CODE.finditer(line):
            span = match.group(1).strip()
            hit = check_name(tengri, span)
            if hit is None:
                continue
            status, detail = git_status(hit[1], cache)
            findings.append((lineno, span, status, detail))
    return findings


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import tengri

    cache: dict[str, tuple[str, str]] = {}
    counts: Counter = Counter()
    checked = 0
    for path in sorted(DEV_DOCS.rglob("*.md")):
        findings = scan_file(path, tengri, cache)
        if not findings:
            continue
        print(path.relative_to(REPO_ROOT).as_posix())
        for lineno, span, status, detail in findings:
            checked += 1
            counts[status] += 1
            suffix = f"  {detail}" if detail else ""
            print(f"  {lineno}: `{span}` {status}{suffix}")
        print()
    print(
        f"{checked} unresolved names: {counts['DELETED']} DELETED, "
        f"{counts['NEVER-EXISTED']} NEVER-EXISTED, "
        f"{counts['IN-SRC-NOT-EXPORTED']} IN-SRC-NOT-EXPORTED"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
