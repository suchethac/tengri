#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a shrink-only ratchet on loose tolerances in the test suite.

A test that asserts ``approx(x, rel=0.05)`` or ``np.allclose(a, b, rtol=1e-2)``
accepts a result that is off by up to the stated fraction. A regression of
that size passes silently, and a loose tolerance is the usual way a
conservation law goes unchecked. This guard counts the call sites whose
relative tolerance is a numeric literal of ``1e-2`` or larger, per file, in
``tests/**/*.py``, and holds the counts in :data:`BASELINE_PATH`.

A site is a call whose function name ends in ``approx``, ``assert_allclose``,
``isclose`` or ``allclose`` (bare name or attribute), with a keyword ``rel`` or
``rtol`` whose value is an int or float literal, or a unary sign on one, whose
magnitude is at least ``1e-2``. A value given by a name or an expression is not
a site: the guard reads literals only, so a tolerance hidden behind a constant
is not counted here. Each file's count is the number of such calls, so two
loose calls on one line count twice.

The list may only shrink. A file whose count rises, or a file that is not in
the baseline at all, fails the run. A count that falls below its recorded
figure also fails, as stale, so the list records the suite as it is. An entry
is a debt, not a licence: when a test is tightened, its site leaves the count,
and the baseline is rewritten in the same change. A new loose tolerance is
fixed by tightening it to the precision the code really reaches, or by
measuring that precision and stating it in the test.

Usage
-----
    python tools/check_test_tolerances.py                    # check against the baseline
    python tools/check_test_tolerances.py --list             # print every site
    python tools/check_test_tolerances.py --write-baseline   # rewrite the baseline

Exit code 0 when every count matches its baseline line; 1 when a count is above
or below its baseline or a file is missing from it; 2 when a test file does not
parse. Standard library only, so it runs in the ``lint`` job beside ruff.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent

#: Recorded counts, one ``path<TAB>count`` line per test file.
BASELINE_PATH = ROOT / "tools" / "test_tolerance_baseline.txt"

#: The tree the sites are read from.
TESTS_PATH = ROOT / "tests"

#: A call is a candidate when its function name ends in one of these.
CALL_SUFFIXES = ("approx", "assert_allclose", "isclose", "allclose")

#: The keywords that carry a relative tolerance.
TOLERANCE_KEYWORDS = ("rel", "rtol")

#: A literal tolerance at or above this magnitude is loose.
LOOSE_THRESHOLD = 1e-2


class Site(NamedTuple):
    """One loose tolerance: where it is, and the literal it names."""

    path: str
    line: int
    value: float


def _call_name(node: ast.Call) -> str | None:
    """The bare name of the called function, or None when it is not a name or attribute."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal_magnitude(node: ast.expr) -> float | None:
    """The magnitude of a numeric literal, or of a unary sign on one; None otherwise."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    if not isinstance(node, ast.Constant):
        return None
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return abs(float(value))


def _loose_value(call: ast.Call) -> float | None:
    """The first literal tolerance keyword of ``call`` at or above the threshold, else None."""
    name = _call_name(call)
    if name is None or not name.endswith(CALL_SUFFIXES):
        return None
    for keyword in call.keywords:
        if keyword.arg not in TOLERANCE_KEYWORDS:
            continue
        magnitude = _literal_magnitude(keyword.value)
        if magnitude is not None and magnitude >= LOOSE_THRESHOLD:
            return magnitude
    return None


def sites_in_file(path: Path, root: Path) -> list[Site]:
    """Every loose-tolerance call in one test file, in source order.

    Raises
    ------
    SyntaxError
        If the file does not parse.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            value = _loose_value(node)
            if value is not None:
                found.append(Site(rel, node.lineno, value))
    return sorted(found, key=lambda site: (site.line, site.value))


def collect_sites(tests_dir: Path, root: Path) -> list[Site]:
    """Every loose-tolerance call under ``tests_dir``, files in sorted order."""
    sites: list[Site] = []
    for path in sorted(tests_dir.rglob("*.py")):
        sites.extend(sites_in_file(path, root))
    return sites


def count_by_file(sites: Iterable[Site]) -> Counter[str]:
    """The number of loose sites per test file."""
    counts: Counter[str] = Counter()
    for site in sites:
        counts[site.path] += 1
    return counts


def _read_baseline() -> dict[str, int]:
    """The recorded counts, comments and blank lines skipped."""
    recorded: dict[str, int] = {}
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rel, count = line.split("\t")
        recorded[rel] = int(count)
    return recorded


def _format_baseline(counts: Mapping[str, int]) -> str:
    """The baseline file text: a header, then one ``path<TAB>count`` line per file, sorted."""
    header = (
        "# Test tolerance ratchet baseline (tools/check_test_tolerances.py): loose\n"
        "# relative tolerances (rel= or rtol= at 1e-2 or more) per test file, one\n"
        "# path<TAB>count line each. This list may only shrink: tighten a site, then\n"
        "# rerun with --write-baseline and commit the smaller file.\n"
    )
    body = "".join(f"{rel}\t{count}\n" for rel, count in sorted(counts.items()))
    return header + body


def _compare(counts: Mapping[str, int], recorded: Mapping[str, int]) -> list[str]:
    """Describe every mismatch between the measured counts and the baseline."""
    problems = []
    for rel in sorted(set(counts) | set(recorded)):
        now = counts.get(rel, 0)
        base = recorded.get(rel, 0)
        if now > base:
            problems.append(f"  {rel}  {now}/{base}")
        elif now < base:
            problems.append(f"  stale: {rel}  {now}/{base}")
    return problems


def _list(sites: Sequence[Site], counts: Mapping[str, int]) -> None:
    """Print the total, then every site as ``path:line  rel=value``."""
    print(f"{sum(counts.values())} loose sites in {len(counts)} files")
    for site in sites:
        print(f"  {site.path}:{site.line}  rel={site.value:g}")


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the suite, then compare, write or list."""
    parser = argparse.ArgumentParser(prog="check_test_tolerances.py")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print every loose site")
    mode.add_argument(
        "--write-baseline", action="store_true", help="rewrite the baseline from current counts"
    )
    args = parser.parse_args(argv)

    try:
        sites = collect_sites(TESTS_PATH, ROOT)
    except SyntaxError as exc:
        print(
            f"check_test_tolerances: could not parse {exc.filename}:{exc.lineno}: {exc.msg}",
            file=sys.stderr,
        )
        return 2
    counts = count_by_file(sites)

    if args.write_baseline:
        BASELINE_PATH.write_text(_format_baseline(counts), encoding="utf-8")
        print(
            f"check_test_tolerances: wrote {len(counts)} entries to {BASELINE_PATH.name} "
            f"({sum(counts.values())} sites)"
        )
        return 0

    if args.list:
        _list(sites, counts)
        return 0

    recorded = _read_baseline()
    problems = _compare(counts, recorded)
    if not problems:
        print(
            f"check_test_tolerances: OK -- {sum(counts.values())} loose sites in "
            f"{len(counts)} files; baseline matches ({len(recorded)} entries)."
        )
        return 0

    print(
        f"check_test_tolerances: {len(problems)} mismatch(es) (path now/baseline)\n",
        file=sys.stderr,
    )
    for line in problems:
        print(line, file=sys.stderr)
    print(
        "\nA count above its baseline is a new loose tolerance: tighten it, or measure the\n"
        "precision the code reaches and state it in the test.\n"
        "A stale line means the file has fewer loose sites: shrink the baseline with\n"
        "    python tools/check_test_tolerances.py --write-baseline\n"
        "and commit the smaller file.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
