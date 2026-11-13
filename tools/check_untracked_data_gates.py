#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a test gated on a data file must not depend on a file CI does not have.

A test gated as ``skipif(find_data_str("x.h5") is None)`` is silently skipped on
every CI runner when ``x.h5`` is neither tracked in git nor downloaded by the
workflow. Nothing fails, the report shows a skip, and the test can go stale
unseen. That is how ``tests/regression/bug/test_2460_neb_logu_prior_within_grid_axis.py``
went dark.

Gate calls, matched by the callee's last name:

* ``find_data``, ``find_data_str``, ``data_path``: every string literal among
  the arguments (the names the suite calls them by; see GATE_CALLS).
* ``skipif`` (``pytest.mark.skipif``) and ``skip`` (``pytest.skip``): every
  string literal among the arguments, except the ``reason=`` keyword, which is
  prose.

A string literal is a data file name only when it is a bare path of word
characters, dots, dashes and slashes that ends in one of DATA_SUFFIXES, so a
message with spaces never matches. A module-level string constant that a gate
call names (``_GRID = "data/x.h5"``, then ``skipif(not os.path.exists(_GRID))``)
is resolved to its literal. Names built at run time or bound inside a function
are not followed, so the guard is a floor, not a proof.

A gated name is available on CI when its basename appears in ``git ls-files``
(any directory) or anywhere in ``.github/workflows/*.yml`` (a download step).
Every (test file, data file) pair that is gated on a name not available on CI is
a finding. The findings are recorded in :data:`BASELINE_PATH`, one
``tests/path.py<TAB>file_name`` line each, and that list may only shrink. A new
finding fails the run: track the file, download it in the workflow, or build the
test on synthetic data. A baseline line no longer found fails as stale, so the
list records the suite as it is.

Usage
-----
    python tools/check_untracked_data_gates.py
    python tools/check_untracked_data_gates.py --list            # gated data files
    python tools/check_untracked_data_gates.py --write-baseline  # rewrite the baseline

Exit code 0 when the findings match the baseline; 1 on a new or stale finding;
2 on a usage, git or parse error. Standard library only, so it runs in the
``lint`` job beside the other stdlib guards.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Recorded findings, one ``tests/path.py<TAB>file_name`` line each.
BASELINE_PATH = ROOT / "tools" / "untracked_data_gates_baseline.txt"

#: Where a download step, if any, would name a data file.
WORKFLOW_DIR = ROOT / ".github" / "workflows"

#: Callees whose string arguments are data file names.
GATE_CALLS = frozenset({"find_data", "find_data_str", "data_path"})

#: Callees whose arguments gate a test (the ``reason=`` keyword is excluded).
SKIP_CALLS = frozenset({"skipif", "skip"})

#: Callees that write a file, so a data name they write is made by the test, not shipped.
WRITE_CALLS = frozenset(
    {"write_bytes", "write_text", "save", "savez", "savez_compressed", "open", "touch"}
)

#: Suffixes that mark a string literal as a data file name.
DATA_SUFFIXES = (
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".fits",
    ".dat",
    ".txt",
    ".json",
    ".csv",
    ".tar.gz",
)

_BARE_PATH = re.compile(r"[\w.\-/]+")

Key = tuple[str, str]


def _callee(call: ast.Call) -> str:
    """The last name of the called function: ``f`` for ``f()``, ``skipif`` for ``mark.skipif``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _gate_arguments(call: ast.Call) -> list[ast.expr] | None:
    """The argument expressions that can name a data file, or None if the call is not a gate."""
    name = _callee(call)
    if name in GATE_CALLS:
        return [*call.args, *(kw.value for kw in call.keywords)]
    if name in SKIP_CALLS:
        return [*call.args, *(kw.value for kw in call.keywords if kw.arg != "reason")]
    return None


def _module_strings(tree: ast.Module) -> dict[str, str]:
    """The top-level names bound to a string literal, so a gate that names one is followed."""
    bound: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target: ast.expr | None = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target = stmt.target
        else:
            continue
        value = stmt.value
        is_string = isinstance(value, ast.Constant) and isinstance(value.value, str)
        if isinstance(target, ast.Name) and is_string:
            bound[target.id] = value.value
    return bound


def _strings_in(nodes: Iterable[ast.expr], bound: Mapping[str, str]) -> Iterable[str]:
    """Every string literal under ``nodes``, plus the literal of each module name they use."""
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield sub.value
            elif isinstance(sub, ast.Name) and sub.id in bound:
                yield bound[sub.id]


def _is_data_name(text: str) -> bool:
    """True for a bare path that ends in a data suffix, e.g. ``data/x.h5``.

    A bare suffix (``".h5"``) and a name whose stem starts with ``_`` (a
    deliberately absent ``__absent__.h5``) name no real data file.
    """
    if _BARE_PATH.fullmatch(text) is None or not text.endswith(DATA_SUFFIXES):
        return False
    stem = text.rsplit("/", 1)[-1]
    return not stem.startswith((".", "_"))


def _written_names(tree: ast.Module, bound: Mapping[str, str]) -> set[str]:
    """Data names the module writes itself, so a gate on them is not on a shipped file."""
    written: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee(node) in WRITE_CALLS:
            subjects = [node.func, *node.args]
            written.update(t for t in _strings_in(subjects, bound) if _is_data_name(t))
    return {text.rsplit("/", 1)[-1] for text in written}


def _asserted_calls(tree: ast.Module) -> set[int]:
    """The ids of calls under an ``assert``: a lookup the test checks, not a skip gate."""
    under: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            under.update(id(sub) for sub in ast.walk(node.test) if isinstance(sub, ast.Call))
    return under


def collect_gates(source: str) -> set[str]:
    """The basenames of the data files that gate calls in one test module name."""
    tree = ast.parse(source)
    bound = _module_strings(tree)
    asserted = _asserted_calls(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in asserted:
            continue
        arguments = _gate_arguments(node)
        if arguments is None:
            continue
        for text in _strings_in(arguments, bound):
            if _is_data_name(text):
                found.add(text.rsplit("/", 1)[-1])
    return found - _written_names(tree, bound)


def gate_table(sources: Mapping[str, str]) -> dict[str, set[str]]:
    """Map each gated data file basename to the test files that gate on it."""
    table: dict[str, set[str]] = {}
    for rel, source in sources.items():
        for name in collect_gates(source):
            table.setdefault(name, set()).add(rel)
    return table


def on_ci(name: str, tracked: set[str], workflow_text: str) -> bool:
    """True when the data file is tracked in git or named by a workflow download step."""
    return name in tracked or name in workflow_text


def find_untracked(sources: Mapping[str, str], tracked: set[str], workflow_text: str) -> set[Key]:
    """The (test file, data file) pairs gated on a name CI does not have."""
    return {
        (rel, name)
        for name, rels in gate_table(sources).items()
        if not on_ci(name, tracked, workflow_text)
        for rel in rels
    }


def compare(found: set[Key], recorded: set[Key]) -> list[str]:
    """Describe each finding absent from the baseline and each baseline line no longer found."""
    problems = [f"  untracked gate  {rel}  {name}" for rel, name in sorted(found - recorded)]
    problems += [f"  stale: {rel}  {name}" for rel, name in sorted(recorded - found)]
    return problems


def _read_baseline(path: Path = BASELINE_PATH) -> set[Key]:
    """The recorded findings, comments and blank lines skipped."""
    recorded: set[Key] = set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rel, name = line.split("\t")
        recorded.add((rel, name))
    return recorded


def _format_baseline(found: set[Key]) -> str:
    """The baseline text: a header, then one ``tests/path.py<TAB>file_name`` line per finding."""
    header = (
        "# Untracked data gate baseline (tools/check_untracked_data_gates.py): test files\n"
        "# gated on a data file that CI does not have, one path<TAB>file line each. This list\n"
        "# may only shrink: track or download the file, or rebuild the test on synthetic data,\n"
        "# then rerun with --write-baseline and commit the smaller file.\n"
    )
    body = "".join(f"{rel}\t{name}\n" for rel, name in sorted(found))
    return header + body


def _read_sources(root: Path) -> dict[str, str]:
    """The source text of every test module, keyed by repository-relative posix path."""
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in sorted((root / "tests").rglob("*.py"))
    }


def _tracked_basenames(root: Path) -> set[str]:
    """The basename of every file git tracks, in any directory."""
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    )
    return {path.rsplit("/", 1)[-1] for path in proc.stdout.split("\0") if path}


def _workflow_text(workflows: Path) -> str:
    """The text of every workflow file, where a download step would name a data file."""
    return "\n".join(path.read_text() for path in sorted(workflows.glob("*.yml")))


def _run(args: argparse.Namespace) -> int:
    """Measure, then compare, write or list the gated data files."""
    try:
        sources = _read_sources(ROOT)
        tracked = _tracked_basenames(ROOT)
        workflow_text = _workflow_text(WORKFLOW_DIR)
        found = find_untracked(sources, tracked, workflow_text)
    except (OSError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(f"check_untracked_data_gates: could not measure: {exc}", file=sys.stderr)
        return 2

    if args.write_baseline:
        BASELINE_PATH.write_text(_format_baseline(found))
        print(f"check_untracked_data_gates: wrote {len(found)} entries to {BASELINE_PATH.name}")
        return 0

    if args.list:
        table = gate_table(sources)
        print(f"{len(table)} data files gated on, {len(found)} findings not on CI")
        for name in sorted(table):
            status = "on CI" if on_ci(name, tracked, workflow_text) else "NOT on CI"
            print(f"  {name}\t{len(table[name])}\t{status}")
        return 0

    problems = compare(found, _read_baseline())
    if not problems:
        print(f"check_untracked_data_gates: OK -- {len(found)} findings match the baseline.")
        return 0

    print(f"check_untracked_data_gates: {len(problems)} mismatch(es)\n", file=sys.stderr)
    for line in problems:
        print(line, file=sys.stderr)
    print(
        "\nAn untracked gate is a test that skips on CI for want of a file: track the file,\n"
        "download it in the workflow, or build the test on synthetic data.\n"
        "A stale line means the gate is fixed or gone: shrink the baseline with\n"
        "    python tools/check_untracked_data_gates.py --write-baseline\n"
        "and commit the smaller file.",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the mode flag, then measure, compare, write or list."""
    parser = argparse.ArgumentParser(prog="check_untracked_data_gates.py")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print each gated data file")
    mode.add_argument(
        "--write-baseline", action="store_true", help="rewrite the baseline from current findings"
    )
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
