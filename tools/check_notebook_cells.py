#!/usr/bin/env python3
"""CI guard: no committed notebook may hold a code cell with its newlines gone.

Why this guard exists
---------------------
Five notebooks stored 73 code cells as a single ``source`` entry with every
newline deleted, so the code was one run-on line::

    'import osimport sysimport timeimport warningsos.environ["JAX_PLATFORMS"] ='

Such a cell cannot execute and cannot be read. Nothing noticed for months
(#1820), and every check that might have was looking elsewhere:

- ``check_notebook_renders.py`` covers ``docs/spine/`` only; none of the five
  was in scope.
- ruff excludes ``notebooks/tutorials`` and never reaches ``archive_2``.
- No step parsed notebook code cells as Python.
- The JSON stayed **valid**, so every notebook-aware tool loaded the files
  happily. nbformat permits ``source`` as either a list of lines or one string,
  so a single string is legal — it is the missing newlines *inside* it that are
  the damage, and no schema can see that.

Detection
---------
A code cell is flagged when it contains **no newline** and is at least
:data:`MIN_CHARS` long. Cells opening with an IPython magic (``%``/``!``/``?``)
are skipped, being neither Python nor expected to look like it.

The threshold is measured, not guessed. Across the 469 tracked notebooks the
longest healthy single-line code cell is 90 characters (an ``ssp =
tengri.load_ssp(...)`` with a long grid filename); the shortest of the 73
damaged cells is 148. 120 sits in that gap with room on both sides.

Do not add "and it fails to parse"
----------------------------------
That was the first version of this guard, and it passed 34 of the 73 known-bad
cells -- including every one of the dangerous ones.

A collapsed cell parses whenever its first line ends in a comment, because the
rest of the cell is swallowed by that comment::

    N_WARMUP = 3  # JIT warm-up callsN_REPEAT = 50  # timing callsdef bench_fn(...

That is valid Python. It is also a cell where ``N_REPEAT`` and an entire
function definition have silently ceased to exist -- the notebook runs, raises
nothing, and does almost none of what it says. The cells that *fail* to parse
are the benign half: they announce themselves the first time anyone runs them.

So parse-success is not evidence of health here. It only records whether line
one happened to end in a ``#``.

What this guard cannot do
-------------------------
It cannot repair anything, and it cannot see a collapse that stays under the
length threshold. It says nothing about whether a cell is *correct*.

Recovery, when it fires, is to rebuild the notebook from its jupytext ``.py``
mirror -- CLAUDE.md makes the mirror the source of truth, and all five damaged
notebooks had an intact one.

Dependencies: standard library only, so it runs in the `lint` job.

Usage
-----
    python tools/check_notebook_cells.py

Exit code 0 when every code cell is readable; 1 otherwise, listing each cell.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Longest healthy single-line code cell in the tree: 90 characters. Shortest
#: known collapsed cell: 148. Measured, not guessed -- see the module docstring.
MIN_CHARS = 120

#: Cells opening with these are IPython, not Python.
_MAGIC_PREFIXES = ("%", "!", "?")

#: How much of the offending line to echo back.
_ECHO = 90


def _tracked_notebooks() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in out.decode("utf-8").split("\0") if name]


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _is_collapsed(text: str) -> bool:
    """True when the cell looks like source with its newlines removed."""
    stripped = text.strip()
    if "\n" in stripped or len(stripped) < MIN_CHARS:
        return False
    return not stripped.startswith(_MAGIC_PREFIXES)


def _parses(text: str) -> bool:
    """Whether the cell is valid Python -- reported, never used to decide."""
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


def main() -> int:
    violations: list[tuple[str, int, str]] = []
    unreadable: list[tuple[str, str]] = []
    cells = 0

    for path in _tracked_notebooks():
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            unreadable.append((rel, str(exc)))
            continue
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            cells += 1
            text = _source(cell)
            if _is_collapsed(text):
                stripped = text.strip()
                violations.append((rel, index, stripped[:_ECHO], _parses(stripped)))

    if unreadable:
        print(f"{len(unreadable)} notebook(s) could not be read:\n", file=sys.stderr)
        for rel, message in unreadable:
            print(f"  {rel}: {message}", file=sys.stderr)

    if violations or unreadable:
        if violations:
            print(
                f"\n{len(violations)} code cell(s) have lost their newlines:\n",
                file=sys.stderr,
            )
            for rel, index, excerpt, parses in violations:
                note = "  (parses -- code absorbed into a comment)" if parses else ""
                print(f"  {rel} cell {index}:{note}", file=sys.stderr)
                print(f"      {excerpt}...", file=sys.stderr)
        print(
            "\nA cell like this cannot be executed or read, and the notebook JSON "
            "stays valid,\nso nothing else will report it. Rebuild the notebook "
            "from its jupytext .py mirror:\n"
            "    jupytext --to ipynb --output <notebook>.ipynb <mirror>.py\n",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {cells} code cells across {len(_tracked_notebooks())} notebooks are readable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
