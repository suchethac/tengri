#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A removed API must not survive where the sweep that removed it did not reach (#1776-#1781).

Removing a public spelling means rewriting every call site. That sweep is done by
hand, and its census is whatever the author happened to grep. When it misses a
tree nothing notices, because the tree is not imported by the test suite: the
stale calls sit there until someone runs the file.

That is not hypothetical. #1720 removed the ``stellar=`` build group, and its
commit message states "Every call site in the repo moved with it -- src, tests,
six notebooks, the gallery example". It touched six files under ``notebooks/``
and none under ``reproduction/``, which has six notebooks of its own. Five of
those six died on the removed spelling and stayed dead, because the only
PR-gated checks over ``reproduction/`` hash the *committed* PNGs -- artifacts of
a run that already happened, which keep matching perfectly while the source that
produced them no longer runs (#1776-#1781).

So the census belongs in a file rather than in one person's grep, and the roots
are the whole repo rather than the part someone remembered. Adding a rule when
you remove a public API is the cost of removing it.

Why the AST and not a regex
---------------------------
Every removed spelling appears legitimately in prose: the changelog names it, the
migration guide tabulates it, and the guard tests pass it on purpose to prove it
raises. A regex cannot tell ``stellar={...}`` in a call from the same characters
in a docstring, and a checker that cries wolf on its own documentation is one
that gets silenced. Matching call *syntax* instead of call *text* removes that
whole class: ``.md`` files are not scanned at all, and a docstring is not a call.
Notebooks are scanned through their code cells, which is where the rot was.

Usage
-----
    python tools/check_removed_apis.py
    python tools/check_removed_apis.py --root reproduction

Exit code 0 if no removed spellings are found; 1 with each one listed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ROOTS = (
    "src",
    "tests",
    "examples",
    "notebooks",
    "reproduction",
    "bench",
    "scripts",
    "tools",
    "docs",
)
#: Executable source only. Prose about a migration is not a call site, and the
#: changelog and migration guide must be free to name what they replaced.
SUFFIXES = (".py", ".ipynb")

EXCLUDE_PARTS = (
    "auto_examples",
    "_build",
    "archive",
    "archive_2",
    "_retired",
    "_old_notebooks",
    "superpowers",
)

#: Files that must call the removed spelling: they are the guards asserting it
#: still raises with its translation. Removing them to satisfy this checker
#: would delete the only proof the translation works.
EXCLUDE_FILES = frozenset(
    {
        "tools/check_removed_apis.py",
        "tests/unit/test_check_removed_apis.py",
        "tests/contract/test_met_group_grammar.py",
        "tests/contract/test_build_routes_unknown_kwargs_to_grammar.py",
        "tests/regression/bug/test_catalog_met_table_advice.py",
    }
)

#: ``list_all()`` is the one lister that still returns a plain ``dict`` (of
#: ``_RegistryTable`` values), so ``.items()`` on it is correct, not the
#: pre-#1574 contract.
_DICT_RETURNING_LISTERS = frozenset({"list_all"})

_DICT_CONTRACT_ATTRS = frozenset({"items", "keys", "values"})


def _lister_name(node: ast.AST) -> str | None:
    """Return the ``list_*`` name if *node* is a call to one, else None."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
    if not isinstance(name, str) or not name.startswith("list_"):
        return None
    if name in _DICT_RETURNING_LISTERS:
        return None
    return name


def scan_tree(tree: ast.AST):
    """Yield (lineno, reason) for each removed spelling used as real syntax."""
    for node in ast.walk(tree):
        # #1720 -- the metallicity group is `met`; `stellar` is gone.
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "stellar":
                    yield node.lineno, "the `stellar=` group was removed in #1720 -- write `met=`"
                # The old group's structural key; every group selects with `type`.
                if isinstance(kw.value, ast.Dict):
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and key.value == "met_mode":
                            yield (
                                node.lineno,
                                "`met_mode` was removed in #1720 -- write `'type':`",
                            )

        # #1574 -- every list_* returns a _RegistryTable, not {label: callable}.
        if isinstance(node, ast.Subscript):
            name = _lister_name(node.value)
            if name and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                yield (
                    node.lineno,
                    f"`{name}()` returns a _RegistryTable since #1574 -- "
                    'string-index `.to_dict("fn")` instead',
                )
        if isinstance(node, ast.Attribute) and node.attr in _DICT_CONTRACT_ATTRS:
            name = _lister_name(node.value)
            if name:
                yield (
                    node.lineno,
                    f"`{name}().{node.attr}()` is the pre-#1574 dict contract -- "
                    'call `.to_dict("fn")` first',
                )


_MAGIC = re.compile(r"^\s*[%!]")


def _parse(source: str) -> ast.AST | None:
    """Parse Python, tolerating IPython magics. None if it still will not parse."""
    cleaned = "\n".join("" if _MAGIC.match(line) else line for line in source.splitlines())
    try:
        return ast.parse(cleaned)
    except SyntaxError:
        return None


def scan_file(path: Path):
    """Yield (lineno, reason) for one .py or .ipynb file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".py":
        tree = _parse(text)
        if tree is not None:
            yield from scan_tree(tree)
        return

    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return
    for n, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        joined = "".join(src) if isinstance(src, list) else src
        tree = _parse(joined)
        if tree is None:
            continue
        for _lineno, reason in scan_tree(tree):
            # Notebook line numbers are per-cell; the cell is the useful locator.
            yield f"cell {n}", reason


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
            if rel in EXCLUDE_FILES:
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
        for where, reason in scan_file(path):
            violations.append((path.relative_to(REPO_ROOT), where, reason))

    if not violations:
        print("OK: no removed API spellings in executable source")
        return 0

    print(f"FAIL: {len(violations)} removed API call site(s)\n")
    for path, where, reason in violations:
        print(f"  {path}:{where}  {reason}")
    print(
        "\nFix: rewrite the call site with the replacement named above.\n"
        "A notebook counts -- reproduction/*.ipynb and its docs/ copy are committed\n"
        "source, and nothing else in CI executes them.\n"
        "If the file must call the removed spelling to prove it raises, add it to\n"
        "EXCLUDE_FILES with a justification."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
