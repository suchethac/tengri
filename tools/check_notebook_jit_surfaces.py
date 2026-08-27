#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: notebook code does not pass the non-pytree predict surface to jax.jit/vmap.

Why this exists
---------------

Issue #1255: A published notebook called ``jax.jit(jax.vmap(model.predict))`` — the rich
``Prediction`` and ``PropertyCatalog`` objects are deliberately NOT pytrees, so this raises
at trace time. ruff and ast.parse pass on it, leaving the bug to manifest at notebook
execution time (which never happens on CI — #1256). NAMING_CONTRACT §4b names the ONLY
JIT/vmap-safe prediction surfaces: ``predict_photometry``, ``predict_spectrum``,
``predict_properties``. The rich ``model.predict(params)`` object and its ``.properties``
accessor must never cross jit/vmap.

This guard catches the class at lint time, before notebooks run.

Why existing tooling does not cover it
---------------------------------------

ruff checks style and basic syntax; ast.parse successfully parses `notebooks/*.py`
(they are valid Python). Both tools pass on `jax.jit(jax.vmap(model.predict))` because
§4b's requirement — that the rich prediction surface is NOT a pytree — is a tracing-
semantics contract that no syntax-level tool knows. This guard encodes that contract
as a pattern detection rule in pure stdlib.

What is out of scope, and why
-----------------------------

* ``examples/`` — the gallery CI job executes every example on PRs, so trace errors
  surface there. No need to gate at lint.
* ``docs/`` — generated output, out of scope for all guards.
* ``notebooks/archive/`` — frozen historical record. Like
  ``check_notebook_pairing.py``, this respects the archive as read-only history.

Both exclusions are *directories with a stated reason*, not a list of files. No
allowlist: the fixes are to migrate the notebook to the safe surfaces, and an
allowlist would be a third option that records neither.

Implementation notes
--------------------

The guard finds balanced-parenthesis spans of jax.jit(, jax.vmap(, jax.grad(, jax.value_and_grad(,
bare jit(, bare vmap(, bare grad(, bare value_and_grad(, dropping full-line comments (lines whose
stripped form starts with #) first. Inside each span it flags:

(a) any ``.predict`` not followed by an identifier character (i.e., bare rich accessor:
    ``.predict()``, ``.predict)``, ``.predict,``, etc.)
(b) any ``.predict_<something>`` where something is NOT in {photometry, spectrum, properties}
(c) any ``.properties`` attribute access

Nested spans are flagged once per file:line. The rule is semantic, not syntactic: strings
are NOT parsed (if a string literal somewhere trips it, the tree fails in reality and
the report directs to that real failure).

Dependencies: standard library only. The lint job installs only ruff; it must not import
tengri, jupytext, or anything else.

Exit code 0 when no violations found; 1 otherwise, listing each with file:line and a fix
message.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories excluded from this guard.
_EXCLUDED_PREFIXES = ("examples/", "docs/", "notebooks/archive/", "reproduction/archive/")

#: Safe predict surfaces that may be passed to jax.jit/vmap.
_SAFE_PREDICT_SURFACES = {"photometry", "spectrum", "properties"}


def _tracked_py_files_in_scope() -> list[Path]:
    """Every tracked .py file under notebooks/ and reproduction/, excluding archives.

    Returns absolute paths.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "notebooks/", "reproduction/"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    # Filter to .py files only.
    paths = [
        REPO_ROOT / name
        for name in out.decode("utf-8").split("\0")
        if name and name.endswith(".py")
    ]

    # Filter out excluded prefixes.
    return [
        p
        for p in paths
        if not any(
            p.relative_to(REPO_ROOT).as_posix().startswith(prefix) for prefix in _EXCLUDED_PREFIXES
        )
    ]


def _drop_comments(text: str) -> str:
    """Blanking full-line comments (lines that start with # after stripping whitespace).

    Jupytext markdown cells are comment lines; they must not trip the guard.
    Preserves line structure by replacing comment lines with empty lines (not
    removing them), so position → line_number mappings stay 1:1 to the original.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            lines.append("")  # Preserve line structure, blank the comment
        else:
            lines.append(line)
    return "\n".join(lines)


def _extract_balanced_span(text: str, start_pos: int) -> str:
    """Extract a balanced (...) span starting at start_pos.

    start_pos must be the position of the opening '('. Returns the substring from
    start_pos (inclusive) to the matching ')' (inclusive), or empty string if no
    balanced match found.
    """
    if start_pos >= len(text) or text[start_pos] != "(":
        return ""

    depth = 0
    pos = start_pos
    while pos < len(text):
        char = text[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start_pos : pos + 1]
        elif char == '"' or char == "'":
            # Skip string literals to avoid counting parens inside them.
            quote = char
            pos += 1
            while pos < len(text) and text[pos] != quote:
                if text[pos] == "\\":
                    pos += 2
                else:
                    pos += 1
            # pos now points at the closing quote (or end of text)
        pos += 1

    return ""


def _find_jax_transforms(text: str) -> list[tuple[int, int, str]]:
    """Find all jax.jit, jax.vmap, jax.grad, jax.value_and_grad, and bare forms.

    Finds balanced-parenthesis spans of jax.jit(, jax.vmap(, jax.grad(,
    jax.value_and_grad(, bare jit(, bare vmap(, bare grad(, and bare
    value_and_grad( expressions.

    Returns list of (start_pos, end_pos, span_text) tuples for each transform.
    Handles multi-line spans and avoids duplicates from nested transforms.
    """
    spans = []
    patterns = [
        r"\bjax\.jit\s*\(",
        r"\bjax\.vmap\s*\(",
        r"\bjax\.grad\s*\(",
        r"\bjax\.value_and_grad\s*\(",
        r"\bjit\s*\(",
        r"\bvmap\s*\(",
        r"\bgrad\s*\(",
        r"\bvalue_and_grad\s*\(",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            # The match ends at the '(' character.
            paren_pos = match.end() - 1
            span = _extract_balanced_span(text, paren_pos)
            if span:
                # Anchor the span at its open paren, not at the start of the
                # transform name. ``span`` begins at ``paren_pos``, so this is
                # the coordinate every consumer needs: ``main`` adds an offset
                # *into span_text* to it to locate a violation, and the
                # de-duplication below marks the covered range with it. Keying
                # on ``match.start()`` broke both -- see the two tests named
                # for those symptoms in tests/unit.
                spans.append((paren_pos, paren_pos + len(span), span))

    # De-duplicate nested spans: keep only the outermost span at each position.
    # Sort by (start, -end) so longer spans at the same start come first.
    spans.sort(key=lambda x: (x[0], -x[1]))
    unique = []
    covered_positions = set()
    for start, end, span_text in spans:
        if start not in covered_positions:
            unique.append((start, end, span_text))
            covered_positions.update(range(start, end))

    return unique


def _line_number_at_pos(text: str, pos: int) -> int:
    """Return the 1-indexed line number at position pos in text."""
    return text[:pos].count("\n") + 1


def _check_violations_in_span(
    span_text: str, file_path: Path, start_pos: int, text: str
) -> list[tuple[int, str, str]]:
    """Check for unsafe predict surfaces inside a jax.jit/vmap span.

    Returns list of (line_number, offending_token, excerpt) tuples.
    """
    violations = []

    # Pattern (a): bare .predict (not followed by identifier)
    for match in re.finditer(r"\.predict(?![a-zA-Z0-9_])", span_text):
        line_num = _line_number_at_pos(text, start_pos + match.start())
        token = ".predict"
        start = max(0, match.start() - 20)
        end = min(len(span_text), match.end() + 20)
        excerpt = span_text[start:end].strip()
        violations.append((line_num, token, excerpt))

    # Pattern (b): .predict_<something> where something is not safe
    for match in re.finditer(r"\.predict_([a-zA-Z0-9_]+)", span_text):
        method_name = match.group(1)
        if method_name not in _SAFE_PREDICT_SURFACES:
            line_num = _line_number_at_pos(text, start_pos + match.start())
            token = f".predict_{method_name}"
            start = max(0, match.start() - 20)
            end = min(len(span_text), match.end() + 20)
            excerpt = span_text[start:end].strip()
            violations.append((line_num, token, excerpt))

    # Pattern (c): .properties
    for match in re.finditer(r"\.properties(?![a-zA-Z0-9_])", span_text):
        line_num = _line_number_at_pos(text, start_pos + match.start())
        token = ".properties"
        start = max(0, match.start() - 20)
        end = min(len(span_text), match.end() + 20)
        excerpt = span_text[start:end].strip()
        violations.append((line_num, token, excerpt))

    return violations


def main() -> int:
    problems: list[str] = []

    for file_path in _tracked_py_files_in_scope():
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            rel = file_path.relative_to(REPO_ROOT).as_posix()
            problems.append(f"{rel}: unreadable ({type(exc).__name__})")
            continue

        # Drop comments before searching for jit/vmap spans.
        text_no_comments = _drop_comments(text)

        # Find all jax.jit, jax.vmap, jit, vmap spans.
        spans = _find_jax_transforms(text_no_comments)

        # Check each span for unsafe surfaces.
        for start_pos, _end_pos, span_text in spans:
            violations = _check_violations_in_span(
                span_text, file_path, start_pos, text_no_comments
            )
            rel = file_path.relative_to(REPO_ROOT).as_posix()
            for line_num, token, excerpt in violations:
                problems.append(
                    f"{rel}:{line_num}: {token}\n"
                    f"    {excerpt}\n"
                    f"    §4b: only predict_photometry / predict_spectrum / "
                    f"predict_properties are JIT/vmap-safe; the rich predict()/"
                    f".properties surface is deliberately not a pytree"
                )

    if problems:
        print(f"check_notebook_jit_surfaces: {len(problems)} violation(s) found\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nNotebooks are never executed by PR-time CI (#1256). This guard catches\n"
            "the #1255 class — jax.jit/vmap over the non-pytree rich predict surface —\n"
            "at lint time. The scheduled notebooks.yml execution tier covers runtime\n"
            "errors (other tracing classes, NaN returns, etc.).\n"
            "\nFix: use only predict_photometry, predict_spectrum, or predict_properties\n"
            "inside jax.jit / jax.vmap / jax.grad / jax.value_and_grad spans."
        )
        return 1

    safe_count = len(_tracked_py_files_in_scope())
    print(f"check_notebook_jit_surfaces: OK -- {safe_count} notebook(s) clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
