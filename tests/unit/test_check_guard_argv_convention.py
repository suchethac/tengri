# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for guard argv handling convention (issue #2075).

This test verifies that all guard scripts in tools/check_*.py follow the
convention that any main() function accepting parsed arguments must:

1. Accept an optional argv parameter: def main(argv: Sequence[str] | None = None)
   Can be positional or keyword-only (def main(*, argv=None))
2. Forward it to all parse_args() and parse_known_args() calls: parser.parse_args(argv)
   CRITICAL: The argument forwarded MUST be the argv parameter, not a hardcoded literal.
   parse_args([]) is a violation — it bypasses the argv convention.

Out of scope:
- A parse_args() call factored into a helper function called from main
  (the helper is not checked; only direct calls in main are enforced).
- Async main functions are checked the same way.

This ensures that unit tests can call guard.main([]) with explicit argv,
preventing pytest command-line flags from interfering with argparse.

Issue: https://github.com/suchethac/tengri/issues/2075
"""

from __future__ import annotations

import ast
from pathlib import Path


class ParseArgsVisitor(ast.NodeVisitor):
    """Find parse_args and parse_known_args calls in a function."""

    def __init__(self):
        self.has_parse_args = False
        self.parse_args_calls = []

    def visit_Call(self, node: ast.Call) -> None:
        # Check for parse_args() or parse_known_args() calls (method calls only)
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            "parse_args",
            "parse_known_args",
        ):
            self.has_parse_args = True
            self.parse_args_calls.append(node)
        self.generic_visit(node)


def test_check_guards_follow_argv_convention():
    """All guard scripts with parse_args must accept and forward argv."""
    repo_root = Path(__file__).resolve().parents[2]
    tools_dir = repo_root / "tools"

    # Find all check_*.py files
    check_files = sorted(tools_dir.glob("check_*.py"))

    violations = []

    for check_file in check_files:
        try:
            tree = ast.parse(check_file.read_text())
        except SyntaxError as e:
            violations.append(f"{check_file.name}: syntax error: {e}")
            continue

        # Find the main() function (support both sync and async)
        main_func = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
                main_func = node
                break

        if main_func is None:
            # No main() function, skip this file
            continue

        # Check if main() has any parse_args/parse_known_args calls
        visitor = ParseArgsVisitor()
        visitor.visit(main_func)

        if not visitor.has_parse_args:
            # No parse_args calls in main(), exempt from convention
            continue

        # Guard has parse_args calls, enforce the convention
        # 1. Check that main() has argv parameter (positional or keyword-only) with default None
        has_argv_param = False
        argv_is_kwonly = False

        # Check positional arguments
        for arg in main_func.args.args:
            if arg.arg == "argv":
                has_argv_param = True
                # Check that it has a default value of None
                num_defaults = len(main_func.args.defaults)
                num_args = len(main_func.args.args)
                defaults_start = num_args - num_defaults
                arg_index = main_func.args.args.index(arg)

                if arg_index >= defaults_start:
                    default_index = arg_index - defaults_start
                    default = main_func.args.defaults[default_index]
                    # Check if default is None
                    if not isinstance(default, ast.Constant) or default.value is not None:
                        violations.append(f"{check_file.name}: argv parameter default is not None")
                break

        # Also check keyword-only arguments
        if not has_argv_param:
            for i, arg in enumerate(main_func.args.kwonlyargs):
                if arg.arg == "argv":
                    has_argv_param = True
                    argv_is_kwonly = True
                    # Check that it has a default value of None
                    default = main_func.args.kw_defaults[i]
                    if default is None or (
                        not isinstance(default, ast.Constant) or default.value is not None
                    ):
                        violations.append(f"{check_file.name}: argv parameter default is not None")
                    break

        if not has_argv_param:
            violations.append(
                f"{check_file.name}: main() is missing argv parameter (see issue #2075)"
            )
            continue

        # 2. Check that every parse_args/parse_known_args call forwards the argv parameter
        # (not a hardcoded literal)
        for parse_args_call in visitor.parse_args_calls:
            if len(parse_args_call.args) == 0:
                violations.append(f"{check_file.name}: parse_args() called without argv argument")
            elif len(parse_args_call.args) == 1:
                # Check that the argument is the argv parameter, not a literal
                arg = parse_args_call.args[0]
                if not (isinstance(arg, ast.Name) and arg.id == "argv"):
                    violations.append(
                        f"{check_file.name}: parse_args() forwards a literal"
                        " instead of the argv parameter"
                    )

    if violations:
        error_msg = (
            "Guard argv convention violations (issue #2075):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: Every guard with parse_args() must have:\n"
            "  1. def main(argv: Sequence[str] | None = None) -> int:\n"
            "     or: def main(*, argv: Sequence[str] | None = None) -> int:\n"
            "  2. parser.parse_args(argv)  # forward the argv parameter, not a literal\n"
            "\nNote: parse_args([]) is a violation — it bypasses the convention."
        )
        raise AssertionError(error_msg)
