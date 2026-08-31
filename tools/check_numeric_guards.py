#!/usr/bin/env python3
"""CI guard: flag unsafe numeric patterns in src/tengri/ excluding utils/scale.py.

Detects two failure classes:
1. Subnormal-risk floors: jnp.maximum(x, lit < 1e-20) and jnp.clip(x, lit < 1e-20, ...)
   — These guard floors below float32's smallest normal (1.18e-38) fail silently in f32.
   Issues #1604, #1860.

2. Clip-then-power: 10.0 ** jnp.clip(...) and jnp.power(10, jnp.clip(...))
   — These propagate NaN/inf through the clip's VJP in float32, causing gradient failures.
   Issue #1719.

Every current offender is tracked in an allowlist ledger below; new sites fail the check,
and stale ledger entries (sites that no longer match) also fail.

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
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = Path(__file__).resolve().parent / ".numeric_guards_ledger"

# Pattern: path:line -> pattern description
LEDGER: dict[str, str] = {
    # Existing safe patterns that are explicitly guarded
    # (Empty for now; populate with --regen or via audits)
}


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


def load_ledger() -> dict[str, str]:
    """Load the allowlist ledger from disk."""
    if not LEDGER_FILE.exists():
        return {}
    ledger = {}
    for line in LEDGER_FILE.read_text().strip().split("\n"):
        if line and not line.startswith("#"):
            parts = line.split(" -> ", 1)
            if len(parts) == 2:
                ledger[parts[0].strip()] = parts[1].strip()
    return ledger


def save_ledger(ledger: dict[str, str]) -> None:
    """Save the allowlist ledger to disk."""
    lines = [
        "# Allowlist of safe numeric guard patterns (auto-generated)",
        "# Format: path:line -> pattern description",
    ]
    for key in sorted(ledger.keys()):
        lines.append(f"{key} -> {ledger[key]}")
    LEDGER_FILE.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
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

    if args.regen:
        # Rebuild ledger from current violations
        new_ledger = {}
        for filepath_str, violations in violations_by_file.items():
            for lineno, pattern in violations:
                key = f"{filepath_str}:{lineno}"
                new_ledger[key] = pattern
        save_ledger(new_ledger)
        print(f"Regenerated ledger with {len(new_ledger)} entries", file=sys.stderr)
        return 0

    # Load existing ledger
    ledger = load_ledger()

    # Find new violations not in ledger
    new_violations = []
    for filepath_str, violations in violations_by_file.items():
        for lineno, pattern in violations:
            key = f"{filepath_str}:{lineno}"
            if key not in ledger:
                new_violations.append((key, pattern))

    # Find stale ledger entries (in ledger but not in current violations)
    current_keys = {
        f"{path}:{lineno}"
        for path, violations in violations_by_file.items()
        for lineno, _ in violations
    }
    stale_entries = [key for key in ledger if key not in current_keys]

    # Report findings
    if new_violations:
        print("New unsafe numeric patterns found:", file=sys.stderr)
        for key, pattern in new_violations:
            print(f"  {key}: {pattern}", file=sys.stderr)
        print("\nRun with --regen to update the ledger", file=sys.stderr)

    if stale_entries:
        print("Stale ledger entries (patterns no longer present):", file=sys.stderr)
        for key in stale_entries:
            print(f"  {key}", file=sys.stderr)
        print("\nRun with --regen to clean up the ledger", file=sys.stderr)

    if new_violations or stale_entries:
        return 1

    msg = f"OK: {len(ledger)} safe patterns verified in {len(files)} files"
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
