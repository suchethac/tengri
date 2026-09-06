#!/usr/bin/env python3
"""CI guard: every ``@pytest.mark.slow`` test must be reachable by some CI job.

``tests/conftest.py`` auto-marks two trees ``slow`` by path
(``_SLOW_TREES = ("inference", "integration")``), and the ``slow`` CI job
(``.github/workflows/tests.yml``) runs those two trees on schedule /
workflow_dispatch / the ``run-slow-tests`` label. But an individual test
anywhere else in ``tests/`` can also carry ``@pytest.mark.slow`` directly
(an expensive dense sweep, say), and until this guard existed nothing checked
that the ``slow`` job's matrix actually reaches it.

That silent gap is exactly what happened:
``test_snapped_met_axis_beats_uniform_on_a_dense_sweep``
(``tests/components/nebular/test_nebular_grid_precompute.py``) has carried
``@pytest.mark.slow`` since 2026-07-10. The default pytest addopts
(``-m 'not crossval and not slow and not benchmark'``) deselects it from every
ordinary local run and from the PR-gating tier, and the ``slow`` job's matrix
covered only ``tests/inference`` and ``tests/integration``, so the test never
ran in CI, on any trigger, and sat broken for three weeks after an unrelated
API change (#1796) invalidated its fixture. See #2199 (brief:
``.superpowers/sdd/plan-validation-followups/task-4-brief.md``) for the
fixture fix and the census this guard now enforces going forward.

What this checks
-----------------
1. Read ``_SLOW_TREES`` from ``tests/conftest.py`` (parsed, not evaluated as
   code: the value must already be a literal tuple/list of strings).
2. Read the ``slow`` job's ``strategy.matrix.include`` from
   ``.github/workflows/tests.yml``. Each entry covers either the tree named by
   its ``suite`` field (the ``inference``/``integration`` shape:
   ``pytest tests/<suite> ...``) or, if it carries a ``paths`` field, every
   tree named in that space-separated list (the shape this guard's own third
   entry uses: ``pytest tests/components tests/unit ... -m slow``).
3. Union (1) and (2) into the set of covered top-level trees.
4. Walk every ``tests/test_*.py`` file (recursively) for the literal text
   ``pytest.mark.slow``. For each match, take the file's top-level tree
   (the first path component under ``tests/``) and check it is in the
   covered set from step 3.

A file matches on containing the marker text anywhere (decorator, a class or
module ``pytestmark``, or ``pytest.param(..., marks=pytest.mark.slow)``):
narrower than that missed the class of bug this guard exists for, per the
same lesson ``check_test_markers.py`` already learned about column-anchored
regexes. ``tests/conftest.py`` itself is not a ``test_*.py`` file (it is the
marking machinery, not a test) and is correctly outside this scan.

Dependencies: PyYAML (already pulled transitively via the ``dev`` extra's
``jupytext`` dependency; verified present in the smoke job's venv).

Usage
-----
    python tools/check_slow_marks_reachable.py
    python tools/check_slow_marks_reachable.py --list   # print the full census

Exit code 0 when every slow-marked file is covered; 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CONFTEST_PATH = TESTS_DIR / "conftest.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "tests.yml"

MARKER_TEXT = "pytest.mark.slow"


def read_slow_trees(conftest_path: Path = CONFTEST_PATH) -> tuple[str, ...]:
    """Parse ``_SLOW_TREES`` out of ``tests/conftest.py`` without executing it.

    Returns
    -------
    tuple of str
        The tree names conftest auto-marks ``slow`` by path.

    Raises
    ------
    ValueError
        If no module-level ``_SLOW_TREES = ...`` assignment is found, or its
        value is not a literal tuple/list of strings.
    """
    tree = ast.parse(conftest_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_SLOW_TREES"
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, (tuple, list)) or not all(isinstance(v, str) for v in value):
                raise ValueError(
                    f"{conftest_path}: _SLOW_TREES is not a literal tuple/list of strings"
                )
            return tuple(value)
    raise ValueError(f"{conftest_path}: no module-level '_SLOW_TREES = ...' assignment found")


def read_slow_matrix_trees(workflow_path: Path = WORKFLOW_PATH) -> set[str]:
    """Return every tree the ``slow`` job's matrix ``include`` list reaches.

    An entry with a ``paths`` field (space-separated ``tests/<tree>`` tokens)
    contributes every tree it names; an entry without one contributes the
    single tree named by its ``suite`` field (the ``pytest tests/<suite>``
    shape the ``inference``/``integration`` entries use).

    Raises
    ------
    ValueError
        If the workflow has no ``jobs.slow.strategy.matrix.include`` list.
    """
    doc = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    try:
        include = doc["jobs"]["slow"]["strategy"]["matrix"]["include"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{workflow_path}: no jobs.slow.strategy.matrix.include list found"
        ) from exc
    if not isinstance(include, list) or not include:
        raise ValueError(f"{workflow_path}: jobs.slow.strategy.matrix.include is empty")

    trees: set[str] = set()
    for entry in include:
        paths = entry.get("paths")
        if paths:
            for token in paths.split():
                # "tests/components" -> "components"
                parts = Path(token).parts
                if len(parts) >= 2 and parts[0] == "tests":
                    trees.add(parts[1])
        elif "suite" in entry:
            trees.add(entry["suite"])
    return trees


def find_slow_marked_files(tests_dir: Path = TESTS_DIR) -> list[Path]:
    """Every ``tests/test_*.py`` file (recursive) containing the literal marker."""
    return sorted(
        p for p in tests_dir.rglob("test_*.py") if MARKER_TEXT in p.read_text(encoding="utf-8")
    )


def file_tree(path: Path, tests_dir: Path = TESTS_DIR) -> str:
    """The top-level tree a test file lives under (first path part under tests/)."""
    return path.relative_to(tests_dir).parts[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the full slow-mark census")
    args = parser.parse_args(argv)

    try:
        slow_trees = read_slow_trees()
        matrix_trees = read_slow_matrix_trees()
    except ValueError as exc:
        print(f"check_slow_marks_reachable: FAIL -- {exc}", file=sys.stderr)
        return 1

    covered = set(slow_trees) | matrix_trees
    files = find_slow_marked_files()

    census: list[tuple[Path, str, bool]] = []
    for path in files:
        tree = file_tree(path)
        census.append((path.relative_to(REPO_ROOT), tree, tree in covered))

    if args.list:
        print(f"_SLOW_TREES (tests/conftest.py): {sorted(slow_trees)}")
        print(f"slow-job matrix trees (.github/workflows/tests.yml): {sorted(matrix_trees)}")
        print(f"covered (union): {sorted(covered)}\n")
        for rel, tree, ok in census:
            status = "covered" if ok else "UNREACHABLE"
            print(f"  {status:12s} tree={tree:12s} {rel}")
        return 0

    unreachable = [(rel, tree) for rel, tree, ok in census if not ok]
    if not unreachable:
        print(
            f"check_slow_marks_reachable: OK -- {len(files)} slow-marked file(s), "
            f"all reachable by _SLOW_TREES or the slow job's matrix."
        )
        return 0

    print(
        f"check_slow_marks_reachable: FAIL -- {len(unreachable)} slow-marked file(s) "
        "not reachable by any CI job:\n",
        file=sys.stderr,
    )
    for rel, tree in unreachable:
        print(f"  {rel}  (tree: {tree})", file=sys.stderr)
    print(
        "\nA test carrying @pytest.mark.slow outside tests/inference and "
        "tests/integration never runs unless the slow job's matrix reaches its "
        "tree. Either fold its tree into the slow job's matrix (add it to the "
        "`paths` list of the outside-trees entry in .github/workflows/tests.yml), "
        "or move the test under one of the two auto-marked trees.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
