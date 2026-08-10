#!/usr/bin/env python3
"""CI guard: a warning that reports a computed number must carry it (#1645).

A warn site that formats a quantity into prose and discards the value forces
every consumer to regex-parse the message, and to accept whatever the format
spec rounded it to. ``SFHBeforeBigBangWarning`` rendered ``{frac:.0%}``, so
"69%" meant anything in 0.685-0.695, and the mock builder that needed the
number could not obtain it -- it recorded 0.0 for every galaxy while the
warnings were plainly firing.

The remedy is :func:`tengri.config.exceptions.warn_measured`, which keeps the
rounded prose for humans and attaches the exact values for code.

Detection is AST-based, deliberately. ``warnings.warn(`` and its f-string sit on
different lines in every multi-line call, so a line-based grep over this same
tree reports **zero** hits and looks like a clean codebase.

A site is flagged when its message renders a placeholder with a numeric
precision spec (``.0%``, ``.2f``, ``.4g``, ``.2e``) and the call is a bare
``warnings.warn``. Sites that legitimately report no reusable quantity can move
to ``warn_measured`` with no measurements, or be listed in ``ALLOW``.

Usage
-----
    python tools/check_warning_payloads.py

Exit code 0 on success; non-zero with the offending sites listed otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "tengri"

#: Sites whose rendered number is genuinely not a reusable quantity.
#: Keep this list short and justified; prefer migrating the site.
ALLOW: frozenset[str] = frozenset()

#: Characters that make a format spec numeric-and-rounding.
_ROUNDING = set("%efg")


def _rounded_placeholders(node):
    """Format specs in ``node`` that round a number, e.g. ``.0%``, ``.2f``."""
    specs = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.FormattedValue) or sub.format_spec is None:
            continue
        for piece in ast.walk(sub.format_spec):
            if not (isinstance(piece, ast.Constant) and isinstance(piece.value, str)):
                continue
            spec = piece.value
            if "." in spec and any(c in spec for c in _ROUNDING | {"."}):
                if any(ch.isdigit() for ch in spec):
                    specs.append(spec)
    return specs


def _is_bare_warn(call):
    """True for ``warnings.warn(...)`` / ``warn(...)``, not ``warn_measured``."""
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr == "warn"
    if isinstance(fn, ast.Name):
        return fn.id == "warn"
    return False


def find_violations(root):
    """Warn sites that render a rounded number without carrying it."""
    violations = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_bare_warn(node) or not node.args:
                continue
            specs = _rounded_placeholders(node.args[0])
            if not specs:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if f"{rel}:{node.lineno}" in ALLOW:
                continue
            violations.append((rel, node.lineno, sorted(set(specs))))
    return violations


def main():
    if not SRC.is_dir():
        print(f"FAIL: {SRC} not found — the guard would pass vacuously", file=sys.stderr)
        return 2

    scanned = sum(1 for _ in SRC.rglob("*.py"))
    if scanned < 50:
        print(f"FAIL: only {scanned} files scanned; expected the full package", file=sys.stderr)
        return 2

    violations = find_violations(SRC)
    if not violations:
        print(f"OK: no warn site discards a rounded number ({scanned} files scanned)")
        return 0

    print(f"{len(violations)} warn site(s) render a rounded number but carry no value:\n")
    for rel, line, specs in violations:
        print(f"  {rel}:{line}  renders {specs}")
    print(
        "\nUse tengri.config.exceptions.warn_measured, which keeps the rounded prose\n"
        "and attaches the exact values:\n\n"
        "    from tengri.config.exceptions import warn_measured\n\n"
        "    warn_measured(\n"
        '        f"...forms {frac:.0%} of its mass...",\n'
        "        SomeWarning,\n"
        "        stacklevel=2,          # same number as the warnings.warn it replaces\n"
        "        truncated_fraction=frac,\n"
        "    )\n\n"
        "Read it back with measurements_of(record.message), which returns {} for\n"
        "warnings that carry none — so consumers need not know which sites migrated."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
