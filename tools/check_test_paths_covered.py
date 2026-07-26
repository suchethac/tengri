#!/usr/bin/env python3
"""CI guard: every test tree must be reachable from a CI job's path list.

The test workflow does not run ``pytest tests/``. It runs an explicit
per-shard path matrix (``tests/contract``, ``tests/components``,
``tests/regression/...``, ``tests/physics``, ``tests/unit``) plus a
label-gated slow tier (``tests/inference``, ``tests/integration``). A test
file that lands in a directory no shard names is collected by a local
``pytest tests/`` run and by nothing else — it passes review, it passes
locally, and it never gates a pull request again.

That is not hypothetical. ``tests/unit`` was emptied by the #148 rehome and
dropped out of the matrix when the shards were rewritten around the
physics-first trees. Ten of the eleven files that later landed there came
from the #1322 inference/prediction-API epic, and one of them had been
failing on ``main`` since #1337 phase 2 removed the ``NotImplementedError``
it asserts. CI was green throughout.

This guard closes the loop: it reads the workflow, unions every path any
job actually passes to pytest, and fails if a directory holding test files
is not underneath one of them.

Dependencies: standard library only. The ``lint`` job installs ruff and
nothing else, so this must not import ``yaml`` or ``tengri``.

Usage
-----
    python tools/check_test_paths_covered.py

Exit code 0 when every test directory is covered; 1 otherwise, listing the
uncovered directories and the files they hold.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# Directories deliberately outside every CI path list. Each needs a reason:
# a bare "we know" entry here is how the next tree goes dark.
EXEMPT: dict[str, str] = {
    "tests/crossval": (
        "cross-validation against bagpipes/FSPS; excluded from every run by "
        "the `-m 'not crossval'` selector and an explicit --ignore, because it "
        "needs peer codes CI does not install"
    ),
    "tests/fixtures": "shared fixture data and helpers, holds no test files",
}

# `tests/${{ matrix.suite }}` in the slow job expands over these.
_MATRIX_VAR = re.compile(r"tests/\$\{\{\s*matrix\.(\w+)\s*\}\}")
_PATH_TOKEN = re.compile(r"(?<!=)\btests/[A-Za-z0-9_][A-Za-z0-9_/]*")
_IGNORE_TOKEN = re.compile(r"--ignore=(tests/[A-Za-z0-9_/]*)")


def _matrix_values(text: str, key: str) -> set[str]:
    """Values a matrix key takes, e.g. ``suite: inference`` -> {"inference"}."""
    return set(re.findall(rf"^\s*-?\s*{key}:\s*([A-Za-z0-9_-]+)\s*$", text, re.MULTILINE))


def strip_comments(text: str) -> str:
    """Drop YAML comments so prose cannot be mistaken for a path declaration.

    This is load-bearing, not tidiness. The first version of this guard
    scanned the raw file and reported the workflow as fully covered even with
    ``tests/unit`` removed from the matrix — because the comment *explaining*
    the shard says "tests/unit". A guard that reads its own documentation as
    evidence fails open, which is the one thing a guard may never do.
    """
    return "\n".join(re.sub(r"#.*$", "", line) for line in text.splitlines())


def covered_prefixes(text: str) -> set[str]:
    """Every ``tests/...`` path some CI job hands to pytest.

    ``--ignore=`` targets are NOT treated as coverage — a directory named
    only to exclude it is not being run. A directory that is both ignored by
    one shard and named by another (``tests/regression/synthesizer_parity``)
    still comes back covered via the shard that names it.
    """
    text = strip_comments(text)
    ignored = set(_IGNORE_TOKEN.findall(text))
    prefixes = {p for p in _PATH_TOKEN.findall(text) if p not in ignored}

    for key in set(_MATRIX_VAR.findall(text)):
        prefixes.update(f"tests/{value}" for value in _matrix_values(text, key))

    # The two bug/ half-shards carry `paths: ""` and compute their file list
    # at runtime from the directory (see the workflow's run step).
    if "bug_half" in text:
        prefixes.add("tests/regression/bug")

    return {p.rstrip("/") for p in prefixes}


def test_directories() -> set[str]:
    """Repo-relative directories that directly contain ``test_*.py`` files."""
    return {
        str(path.parent.relative_to(REPO_ROOT))
        for path in TESTS_DIR.rglob("test_*.py")
        if "__pycache__" not in path.parts
    }


def is_covered(directory: str, prefixes: set[str]) -> bool:
    """True when ``directory`` is at or below one of the covered prefixes."""
    return any(directory == p or directory.startswith(p + "/") for p in prefixes)


def main() -> int:
    if not WORKFLOW.is_file():
        print(f"ERROR: cannot read {WORKFLOW.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    text = WORKFLOW.read_text(encoding="utf-8")
    prefixes = covered_prefixes(text)

    uncovered = sorted(
        directory
        for directory in test_directories()
        if not is_covered(directory, prefixes)
        and not any(directory == e or directory.startswith(e + "/") for e in EXEMPT)
    )

    if not uncovered:
        n_dirs = len(test_directories())
        print(f"OK: all {n_dirs} test directories are covered by a CI path list.")
        return 0

    print("Test directories that NO CI job runs:\n", file=sys.stderr)
    for directory in uncovered:
        files = sorted(p.name for p in (REPO_ROOT / directory).glob("test_*.py"))
        print(f"  {directory}  ({len(files)} file(s))", file=sys.stderr)
        for name in files[:10]:
            print(f"      {name}", file=sys.stderr)
        if len(files) > 10:
            print(f"      ... and {len(files) - 10} more", file=sys.stderr)
    print(
        "\nA local `pytest tests/` run collects these; CI does not. Either add the "
        "directory to a shard's `paths:` in .github/workflows/tests.yml, move the "
        "files under a covered tree, or record a reason in EXEMPT in this file.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
