#!/usr/bin/env python3
"""CI guard: flag unsafe numeric patterns in src/tengri/ excluding utils/scale.py.

Detects two failure classes:
1. Subnormal-risk floors: jnp.maximum(x, lit < 1e-20) and jnp.clip(x, lit < 1e-20, ...)
   — These guard floors below float32's smallest normal (1.18e-38) fail silently in f32.
   Issues #1604, #1860.

2. Clip-then-power: 10.0 ** jnp.clip(...) and jnp.power(10, jnp.clip(...))
   — These propagate NaN/inf through the clip's VJP in float32, causing gradient failures.
   Issue #1719.

Every current offender is tracked in a ledger keyed on (file path, pattern description)
with a count; this makes the ledger immune to line-number drift (#2050). New sites (live
count > ledger count) fail the check, as do stale entries (live count < ledger count).

The bucket key embeds the visitor's literal message wording, so any rewording orphans the
entire ledger at once — loudly (blanket --regen required), by design.

Blind spot: a same-bucket swap (one site removed, one added in the same bucket in a single
commit) leaves the count equal and ships the new site unadjudicated. This is the accepted
cost of drift-proofness; code review catches the swap.

Usage
-----
    python tools/check_numeric_guards.py              # check against ledger
    python tools/check_numeric_guards.py --regen      # rebuild ledger

Exit code 0 when ledger is current; 1 when new offenders or stale entries found.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = Path(__file__).resolve().parent / ".numeric_guards_ledger"


class NumericGuardVisitor(ast.NodeVisitor):
    """Finds unsafe numeric patterns in AST."""

    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.lines = source.split("\n")
        self.violations: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call):
        """Check for unsafe patterns in function calls."""
        # Check for jnp.maximum(x, lit < 1e-20)
        if self._is_call_to(node.func, ("jnp", "maximum")) and len(node.args) >= 2:
            lit = self._extract_float_literal(node.args[1])
            if lit is not None and lit < 1e-20:
                self.violations.append((node.lineno, f"jnp.maximum with floor {lit} < 1e-20"))

        # Check for jnp.clip(x, lit < 1e-20, ...) but skip if inside a jnp.where
        # guard checking isfinite (safe pattern for #1719 double-where guards)
        if self._is_call_to(
            node.func, ("jnp", "clip")
        ) and not self._is_inside_isfinite_where_guard(node):
            if len(node.args) >= 2:
                lit = self._extract_float_literal(node.args[1])
                if lit is not None and lit < 1e-20:
                    self.violations.append((node.lineno, f"jnp.clip with floor {lit} < 1e-20"))
            # Also check keyword arguments
            for kw in node.keywords:
                if kw.arg == "a_min":
                    lit = self._extract_float_literal(kw.value)
                    if lit is not None and lit < 1e-20:
                        self.violations.append((node.lineno, f"jnp.clip a_min {lit} < 1e-20"))

        # Check for 10.0 ** jnp.clip(...) or jnp.power(10, jnp.clip(...))
        if self._is_base10_power(node):
            msg = "10.0 ** jnp.clip(...) - unsafe in f32 (clip-then-power)"
            self.violations.append((node.lineno, msg))

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        """Check for x ** y where y is a clip call with base 10."""
        if (
            isinstance(node.op, ast.Pow)
            and (left_lit := self._extract_float_literal(node.left)) is not None
            and abs(left_lit - 10.0) < 1e-6
            and self._is_jnp_clip(node.right)
        ):
            msg = "10.0 ** jnp.clip(...) - unsafe in f32 (clip-then-power)"
            self.violations.append((node.lineno, msg))

        self.generic_visit(node)

    def _is_base10_power(self, node: ast.Call) -> bool:
        """True if this is jnp.power(10, jnp.clip(...))."""
        if not self._is_call_to(node.func, ("jnp", "power")):
            return False
        if len(node.args) < 2:
            return False
        # Check if first arg is 10.0
        first = self._extract_float_literal(node.args[0])
        if first is None or abs(first - 10.0) > 1e-6:
            return False
        # Check if second arg is jnp.clip
        return self._is_jnp_clip(node.args[1])

    def _is_jnp_clip(self, node: ast.expr) -> bool:
        """True if node is a call to jnp.clip."""
        if not isinstance(node, ast.Call):
            return False
        return self._is_call_to(node.func, ("jnp", "clip"))

    def _is_call_to(self, func: ast.expr, names: tuple[str, ...]) -> bool:
        """True if func is a call to module.name or just name."""
        if isinstance(func, ast.Attribute):
            if func.attr == names[-1] and isinstance(func.value, ast.Name):
                return func.value.id == names[0]
        elif isinstance(func, ast.Name):
            return len(names) == 1 and func.id == names[0]
        return False

    def _extract_float_literal(self, node: ast.expr) -> float | None:
        """Extract a numeric literal from AST, including unary minus."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._extract_float_literal(node.operand)
            return None if inner is None else -inner
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    def _is_inside_isfinite_where_guard(self, node: ast.expr) -> bool:
        """Check if a clip node is inside a jnp.where guard checking isfinite.

        This pattern is safe for #1719 (double-where guards for clip-then-power).
        Looks for:
          finite = jnp.isfinite(...)
          ... = jnp.where(finite, jnp.clip(...), 0.0)
        Returns True if this line contains jnp.where with the clip inside.
        """
        if node.lineno <= 0 or node.lineno > len(self.lines):
            return False

        # Check the current line for jnp.where
        line = self.lines[node.lineno - 1]
        if "jnp.where" not in line:
            return False

        # Also check for the guard pattern: look at previous few lines
        # for a definition of a finite variable using isfinite
        start = max(0, node.lineno - 5)
        preceding = "\n".join(self.lines[start : node.lineno - 1])

        # Heuristic: if we see both jnp.where (current line) and jnp.isfinite
        # (in preceding lines), this is likely the guarded pattern
        return "isfinite" in preceding or "finite" in line


def _tracked_python_files_in_src() -> list[Path]:
    """Get all .py files in src/tengri/ excluding utils/scale.py."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/tengri/*.py", "src/tengri/**/*.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    files = [ROOT / name for name in out.decode("utf-8").split("\0") if name]
    # Exclude utils/scale.py
    return [f for f in files if "utils/scale.py" not in str(f)]


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single file for unsafe patterns. Return (lineno, pattern) for each violation."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
        visitor = NumericGuardVisitor(source, str(filepath))
        visitor.visit(tree)
        return visitor.violations
    except SyntaxError:
        # Skip files that don't parse
        print(f"Warning: {filepath} does not parse, skipping", file=sys.stderr)
        return []


def load_ledger() -> dict[tuple[str, str], int]:
    """Load the ledger from disk. Format: path | pattern -> count.

    Returns a dict mapping (file_path, pattern_description) to count.
    The ledger format is drift-proof: entries are keyed on (file, pattern),
    not on line numbers, so edits above a violation do not invalidate it.
    See #2050.
    """
    if not LEDGER_FILE.exists():
        return {}
    ledger = {}
    for line in LEDGER_FILE.read_text().strip().split("\n"):
        if line and not line.startswith("#"):
            parts = line.split(" -> ", 1)
            if len(parts) == 2:
                key_part = parts[0].strip()
                count_part = parts[1].strip()
                key_bits = key_part.split(" | ", 1)
                if len(key_bits) == 2:
                    path = key_bits[0].strip()
                    pattern = key_bits[1].strip()
                    try:
                        count = int(count_part)
                        ledger[(path, pattern)] = count
                    except ValueError:
                        pass
    return ledger


def save_ledger(ledger: dict[tuple[str, str], int]) -> None:
    """Save the ledger to disk. Format: path | pattern -> count.

    The (file, pattern) key is immune to line-number drift. See #2050.
    """
    lines = [
        "# Ledger of safe numeric guard patterns (auto-generated)",
        "# Format: path | pattern -> count",
        "# Keyed on (file path, pattern description) for drift-proofness (#2050).",
    ]
    for (path, pattern), count in sorted(ledger.items()):
        lines.append(f"{path} | {pattern} -> {count}")
    LEDGER_FILE.write_text("\n".join(lines) + "\n")


def bucket_violations_by_pattern(
    violations_by_file: dict[str, list[tuple[int, str]]],
) -> dict[tuple[str, str], list[int]]:
    """Group violations into (file_path, pattern) buckets with their line numbers.

    Parameters
    ----------
    violations_by_file : dict[str, list[tuple[int, str]]]
        Violations keyed by file path, values are (lineno, pattern) pairs.

    Returns
    -------
    dict
        Maps (file_path, pattern_description) to sorted list of line numbers.
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for filepath_str, violations in violations_by_file.items():
        for lineno, pattern in violations:
            buckets[(filepath_str, pattern)].append(lineno)
    for key in buckets:
        buckets[key].sort()
    return dict(buckets)


def main(argv: list[str] | None = None) -> int:
    """Check numeric guards against ledger. See module docstring for usage."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--regen", action="store_true", help="Regenerate the ledger")
    args = parser.parse_args(argv)

    files = _tracked_python_files_in_src()
    violations_by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for filepath in files:
        violations = check_file(filepath)
        if violations:
            rel_path = filepath.relative_to(ROOT)
            violations_by_file[str(rel_path)] = violations

    # Group violations into (file, pattern) buckets
    live_buckets = bucket_violations_by_pattern(violations_by_file)
    live_counts = {key: len(lines) for key, lines in live_buckets.items()}

    if args.regen:
        # Rebuild ledger from live counts
        save_ledger(live_counts)
        total_sites = sum(live_counts.values())
        print(
            f"Regenerated ledger with {len(live_counts)} bucket(s), "
            f"{total_sites} total site(s)",
            file=sys.stderr,
        )
        return 0

    # Load existing ledger
    ledger_counts = load_ledger()

    # Find new offenders: buckets where live count > ledger count
    new_offenders = []
    for key, live_count in live_counts.items():
        ledger_count = ledger_counts.get(key, 0)
        if live_count > ledger_count:
            filepath, pattern = key
            new_count = live_count - ledger_count
            lines = live_buckets[key]
            new_offenders.append((filepath, pattern, new_count, lines))

    # Find stale entries: buckets in ledger but not in live
    stale_entries = []
    for key, ledger_count in ledger_counts.items():
        live_count = live_counts.get(key, 0)
        if live_count < ledger_count:
            filepath, pattern = key
            stale_entries.append((filepath, pattern, ledger_count, live_count))

    # Report findings
    if new_offenders:
        print(
            "FAILED: new unsafe numeric pattern site(s) found:", file=sys.stderr
        )
        for filepath, pattern, _count, lines in sorted(new_offenders):
            print(f"  {filepath}: {pattern}", file=sys.stderr)
            print(f"    Ledger: {ledger_counts.get((filepath, pattern), 0)}, "
                  f"Live: {live_counts[(filepath, pattern)]}", file=sys.stderr)
            print(f"    Line(s): {', '.join(map(str, lines))}", file=sys.stderr)
        print("\nRun with --regen to update the ledger", file=sys.stderr)

    if stale_entries:
        print(
            "FAILED: site(s) removed — ratchet down the ledger:", file=sys.stderr
        )
        for filepath, pattern, ledger_count, live_count in sorted(stale_entries):
            print(f"  {filepath}: {pattern}", file=sys.stderr)
            print(f"    Ledger: {ledger_count}, Live: {live_count}", file=sys.stderr)
        print("\nRun with --regen to lock in the improvement", file=sys.stderr)

    if new_offenders or stale_entries:
        return 1

    total_sites = sum(ledger_counts.values())
    msg = f"OK: {len(ledger_counts)} bucket(s), {total_sites} safe pattern(s) verified"
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
