#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a shrink-only ratchet on broad handlers in ``src/tengri`` that drop the exception.

``tools/check_example_silent_failure.py`` makes the gallery keep the exception a broad
``except`` swallowed. The same hazard sits in the package: a catch-all that falls through
(``pass``, ``continue``, ``return None``) turns a broken model into a quiet default, and
nothing downstream says why. Many of the package's handlers are right today, and fixing
them all in one change is not this guard's job. So the existing population is pinned and
may only shrink.

A SITE is a handler that is broad (``except:``, ``except Exception:``,
``except BaseException:``, or a tuple containing one of those) and does not retain the
exception. The two predicates are imported unchanged from ``check_example_silent_failure.py``:
a handler that re-raises, or stores the exception past its scope, is not a site, and a
narrow handler is never a site.

Sites are keyed by ``(relative path, enclosing function qualname)`` and counted, not
numbered, so an unrelated edit above a handler does not move it. Qualnames are dotted
(``Class.method``, ``outer.inner``); a module-level handler is ``<module>``.

Rules, per key, against ``tools/src_silent_fallback_baseline.txt``:

* a count above its baseline fails at each of its sites, with the text "narrow the
  handler to the documented exceptions, or argue it into the baseline";
* a count below its baseline, or a listed key with no sites left, fails as stale and
  names the ``--write-baseline`` rewrite;
* the file may only shrink: ``--write-baseline`` refuses any key whose count would rise.

Ledger lines (``# N -> N-1: reason``) below the header survive a rewrite; the author writes
them in the same change that shrinks the population.

Review rule: a ``# noqa: BLE001`` on a handler is a baseline entry by another name. It hides
the site from ruff and from this guard alike, so the site belongs in the baseline, where the
count stays visible in the diff.

Usage
-----
    python tools/check_src_silent_fallback.py
    python tools/check_src_silent_fallback.py --list
    python tools/check_src_silent_fallback.py --write-baseline

CI: wired in the ``lint`` job beside ``check_lint_ratchet.py ruff``. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Recorded counts, one ``path<TAB>qualname<TAB>count`` line per key.
BASELINE_RELPATH = Path("tools") / "src_silent_fallback_baseline.txt"

#: The fixed header; any other comment line in the baseline is ledger and is kept.
HEADER = (
    "# Silent-fallback ratchet (tools/check_src_silent_fallback.py): broad handlers in",
    "# src/tengri that drop the exception, one path<TAB>qualname<TAB>count line each.",
    "# This list may only shrink: narrow a handler, rerun with --write-baseline, commit.",
)

Key = tuple[str, str]
Sites = dict[Key, list[int]]


def _example_guard_predicates():
    """Return ``(_is_broad, _retains)`` from the example guard, loaded beside this file."""
    path = Path(__file__).resolve().parent / "check_example_silent_failure.py"
    spec = importlib.util.spec_from_file_location("_check_example_silent_failure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._is_broad, module._retains


_is_broad, _retains = _example_guard_predicates()


def _swallowing(handler: ast.ExceptHandler) -> bool:
    """True for a broad handler that neither re-raises nor keeps the exception."""
    return _is_broad(handler) and not _retains(handler)


def _sites_in(tree: ast.AST) -> list[tuple[str, int]]:
    """Return ``(enclosing qualname, lineno)`` for every swallowing handler in *tree*."""
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, (*scope, child.name))
                continue
            if isinstance(child, ast.ExceptHandler) and _swallowing(child):
                found.append((".".join(scope) or "<module>", child.lineno))
            visit(child, scope)

    visit(tree, ())
    return found


def _measure(repo: Path) -> Sites:
    """Map each ``(relative path, qualname)`` to the sorted lines of its swallowing handlers."""
    sites: Sites = {}
    for path in sorted((repo / "src" / "tengri").rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for qualname, lineno in _sites_in(tree):
            sites.setdefault((rel, qualname), []).append(lineno)
    for lines in sites.values():
        lines.sort()
    return sites


def _read_baseline(path: Path) -> dict[Key, int]:
    """The recorded counts, comment and blank lines skipped."""
    recorded: dict[Key, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, qualname, count = line.split("\t")
        recorded[(rel, qualname)] = int(count)
    return recorded


def _ledger(path: Path) -> list[str]:
    """The ledger comment lines of an existing baseline, in file order, header excluded."""
    if not path.exists():
        return []
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#") and line not in HEADER
    ]


def _format_baseline(counts: Mapping[Key, int], ledger: Sequence[str]) -> str:
    """The baseline text: the header, the ledger, then one sorted line per key."""
    body = "".join(
        f"{rel}\t{qualname}\t{count}\n" for (rel, qualname), count in sorted(counts.items())
    )
    return "\n".join([*HEADER, *ledger]) + "\n" + body


def _problems(sites: Sites, recorded: Mapping[Key, int]) -> list[str]:
    """Describe each site above its baseline and each baseline line the code no longer meets."""
    problems: list[str] = []
    for key in sorted(set(sites) | set(recorded)):
        rel, qualname = key
        lines = sites.get(key, [])
        now, base = len(lines), recorded.get(key, 0)
        if now > base:
            problems.extend(
                f"{rel}:{lineno}: broad handler in {qualname} drops the exception "
                f"({now} here, baseline {base}); narrow the handler to the documented "
                "exceptions, or argue it into the baseline"
                for lineno in lines
            )
        elif now < base:
            problems.append(
                f"stale: {rel} {qualname} has {now} swallowing handler(s), baseline says {base}; "
                "rerun with --write-baseline and commit the smaller file"
            )
    return problems


def _grown(sites: Sites, recorded: Mapping[Key, int]) -> list[Key]:
    """The keys whose count is above the recorded one, which a rewrite must not record."""
    return [key for key, lines in sorted(sites.items()) if len(lines) > recorded.get(key, 0)]


def main(argv: Sequence[str] | None = None, repo: Path = REPO) -> int:
    """Check, list or rewrite the baseline for the tree under *repo*; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print the current population")
    mode.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline to the current population (refuses to grow)",
    )
    args = parser.parse_args(argv)

    baseline = repo / BASELINE_RELPATH
    sites = _measure(repo)
    counts = {key: len(lines) for key, lines in sites.items()}
    total = sum(counts.values())
    files = {rel for rel, _ in sites}

    if args.list:
        for (rel, qualname), count in sorted(counts.items()):
            print(f"{rel}\t{qualname}\t{count}")
        print(f"total: {total} swallowing handler(s) in {len(files)} file(s)")
        return 0

    if args.write_baseline:
        grown = _grown(sites, _read_baseline(baseline)) if baseline.exists() else []
        if grown:
            for rel, qualname in grown:
                print(f"refusing to grow: {rel} {qualname}", file=sys.stderr)
            print(
                "the baseline may only shrink; add a line by hand and argue it in review",
                file=sys.stderr,
            )
            return 1
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(_format_baseline(counts, _ledger(baseline)), encoding="utf-8")
        print(f"check_src_silent_fallback: wrote {len(counts)} entries ({total} handlers)")
        return 0

    if not baseline.exists():
        print(
            f"check_src_silent_fallback: no baseline at {BASELINE_RELPATH}; run --write-baseline",
            file=sys.stderr,
        )
        return 2

    problems = _problems(sites, _read_baseline(baseline))
    if not problems:
        print(
            f"check_src_silent_fallback: OK -- {total} swallowing handler(s) in "
            f"{len(files)} file(s); baseline matches."
        )
        return 0

    print(f"check_src_silent_fallback: {len(problems)} problem(s)", file=sys.stderr)
    for line in problems:
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
