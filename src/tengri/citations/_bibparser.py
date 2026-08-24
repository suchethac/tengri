# SPDX-License-Identifier: BSD-3-Clause
"""Minimal BibTeX parser (stdlib-only).

Supports the subset of BibTeX we need for tengri's citation database:
    @entrytype{key,
      field = {value},
      field = "value",
      field = 2026,
    }

Handles nested braces inside values, multi-line values, comments beginning
with ``%``, and custom (non-standard) fields. Returns a list of dicts:

    [{
        "entry_type": "article",
        "bibtex_key": "Calzetti2000",
        "author": "...",
        "title":  "...",
        ...
    }, ...]

We write a small parser by hand to avoid taking a runtime dependency on
``bibtexparser`` or similar. This file is intentionally small and readable.
"""

from __future__ import annotations

import re


def _strip_comments(text: str) -> str:
    """Remove lines starting with ``%`` (BibTeX line comments)."""
    kept = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _scan_braced(text: str, start: int) -> tuple[str, int]:
    """Return the contents of a ``{...}`` block that starts at ``text[start]=='{'``.

    Handles nested braces. Returns (inner_text, end_index_exclusive_of_closing_brace).
    """
    if text[start] != "{":
        raise ValueError(f"Expected '{{' at position {start}, got {text[start]!r}")
    depth = 0
    out = []
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), i + 1
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    raise ValueError("Unterminated '{' in BibTeX value")


def _scan_quoted(text: str, start: int) -> tuple[str, int]:
    """Return the contents of a ``"..."`` string that starts at ``text[start]=='"'``."""
    if text[start] != '"':
        raise ValueError(f"Expected '\"' at position {start}, got {text[start]!r}")
    out = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise ValueError("Unterminated '\"' in BibTeX value")


_ENTRY_HEAD_RE = re.compile(r"@(\w+)\s*\{", re.UNICODE)
_FIELD_NAME_RE = re.compile(r"(\w+)\s*=\s*", re.UNICODE)


def parse_bibtex(text: str) -> list[dict]:
    """Parse a BibTeX-formatted string.

    Parameters
    ----------
    text : str
        Full contents of a .bib file.

    Returns
    -------
    list of dict
        One dict per entry. Every dict contains ``entry_type`` and
        ``bibtex_key`` plus whatever other fields the entry declares.
    """
    text = _strip_comments(text)
    entries: list[dict] = []

    pos = 0
    while True:
        match = _ENTRY_HEAD_RE.search(text, pos)
        if not match:
            break

        entry_type = match.group(1).lower()
        pos = match.end()  # just after the opening '{'

        # Citation key up to the first comma.
        comma = text.find(",", pos)
        if comma == -1:
            raise ValueError(
                f"Malformed @{entry_type} entry at position {pos}: expected ',' after citation key"
            )
        bibtex_key = text[pos:comma].strip()
        pos = comma + 1

        entry: dict = {"entry_type": entry_type, "bibtex_key": bibtex_key}

        while True:
            # Skip whitespace and stray commas.
            while pos < len(text) and text[pos] in " \t\n\r,":
                pos += 1
            if pos >= len(text):
                raise ValueError(f"Unterminated entry {bibtex_key}")
            if text[pos] == "}":
                pos += 1  # consume the closing brace of the entry
                break

            fmatch = _FIELD_NAME_RE.match(text, pos)
            if not fmatch:
                raise ValueError(
                    f"Expected field name in entry {bibtex_key} at position {pos}: "
                    f"found {text[pos : pos + 30]!r}"
                )
            field = fmatch.group(1).lower()
            pos = fmatch.end()

            if pos >= len(text):
                raise ValueError(f"Unterminated field {field} in entry {bibtex_key}")

            if text[pos] == "{":
                value, pos = _scan_braced(text, pos)
            elif text[pos] == '"':
                value, pos = _scan_quoted(text, pos)
            else:
                # Bare number / word until whitespace or comma or closing brace.
                j = pos
                while j < len(text) and text[j] not in " \t\n\r,}":
                    j += 1
                value, pos = text[pos:j], j

            entry[field] = value.strip()

        entries.append(entry)

    return entries
