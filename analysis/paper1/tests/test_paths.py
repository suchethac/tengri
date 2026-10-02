# SPDX-License-Identifier: BSD-3-Clause
"""Tests for repo_relative path utility."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add analysis/paper1 to path
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _paths import repo_relative


class TestRepoRelative:
    """Test repo_relative on various path types."""

    def test_inside_repo_absolute_path(self):
        """Absolute path inside repo becomes repo-relative."""
        # Create an absolute path that's inside the repo
        # test_paths.py is at analysis/paper1/tests/test_paths.py, so parent.parent.parent.parent
        # gets us to the worktree root
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        inside_path = repo_root / "analysis" / "paper1" / "test.txt"
        result = repo_relative(inside_path)
        assert result == "analysis/paper1/test.txt"

    def test_worktree_absolute_path(self):
        """Absolute path in a different worktree becomes just the basename."""
        # Simulate a path from a different worktree (outside this repo)
        other_repo_path = Path("/tmp/some_repo/analysis/paper1/test.txt")
        result = repo_relative(other_repo_path)
        # The path is outside the current repo, so it should be the basename
        assert result == "test.txt"

    def test_outside_repo_path(self):
        """Path outside repo becomes just basename."""
        outside_path = Path("/tmp/some_file.txt")
        result = repo_relative(outside_path)
        assert result == "some_file.txt"

    def test_relative_path(self):
        """Relative path gets resolved relative to cwd."""
        relative_path = Path("analysis/paper1/test.txt")
        result = repo_relative(relative_path)
        # Should resolve and become repo-relative
        assert "analysis" in result and "paper1" in result

    def test_string_input(self):
        """String paths are handled like Path objects."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        inside_str = str(repo_root / "analysis" / "paper1" / "test.txt")
        result = repo_relative(inside_str)
        assert result == "analysis/paper1/test.txt"


@pytest.mark.unit
def test_sweep_writers_use_repo_relative():
    """Sweep test: ensure writer modules wrap path str() in repo_relative.

    This test scans the writer modules and verifies that when they construct
    payloads (dicts/lists) that get written to JSON, NPZ, or sidecars with
    paths, they use repo_relative() to wrap any str(), os.fspath(), or
    .resolve() calls on path-like names.

    Paths are identified by names containing: path, dir, file, out, summary,
    results (case-insensitive). The test uses AST walking to find assignments
    to dict keys or list appends whose values are calls to str() or os.fspath()
    on such names, and fails unless wrapped in repo_relative(...).
    """
    import ast

    writer_modules = [
        "fig11_bma.py",
        "config_forward_digest.py",
        "parse_forward_benchmark.py",
        "surviving_mass_census.py",
        "bma_combine.py",
        "fit_one.py",
        "_cell_provenance.py",
    ]

    path_keywords = {"path", "dir", "file", "out", "summary", "results"}

    def is_path_name(name_str: str) -> bool:
        """Check if a name suggests it holds a file path."""
        name_lower = name_str.lower()
        return any(kw in name_lower for kw in path_keywords)

    def extract_name_from_node(node: ast.expr) -> str | None:
        """Extract a name or attribute from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            # For chained calls like Path(...).resolve(), get the outermost attr
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
            elif isinstance(node.func, ast.Name):
                return node.func.id
        return None

    def check_module(module_path: Path) -> list[str]:
        """Scan one module for repo_relative violations.

        Returns a list of violations (line_number, description).
        """
        if not module_path.exists():
            return []

        source = module_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(module_path))
        except SyntaxError:
            return []

        violations = []

        class PathStringChecker(ast.NodeVisitor):
            """Walk the AST looking for str/fspath on path-like names in dicts/lists."""

            def visit_Subscript(self, node: ast.Subscript) -> None:
                """Check dict assignments like d["key"] = str(path)."""
                # This is d[key], need to check if it's an assignment
                # Assignments are handled separately via visit_Assign
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                """Check assignments like d["key"] = str(path)."""
                # Check if target is a subscript (dict/list assignment)
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        # Assignment to dict: d["key"] = value
                        value = node.value
                        if isinstance(value, ast.Call):
                            func_name = None
                            if isinstance(value.func, ast.Name):
                                func_name = value.func.id
                            elif isinstance(value.func, ast.Attribute):
                                func_name = value.func.attr

                            if func_name in ("str", "fspath") and value.args:
                                # Extract the argument
                                arg_name = extract_name_from_node(value.args[0])
                                if arg_name and is_path_name(arg_name):
                                    violations.append(
                                        f"line {node.lineno}: dict assignment "
                                        f"{func_name}({arg_name}) "
                                        f"not wrapped in repo_relative(...)"
                                    )

                self.generic_visit(node)

            def visit_Dict(self, node: ast.Dict) -> None:
                """Check dict literals: {"key": str(path)}."""
                for value in node.values:
                    if isinstance(value, ast.Call):
                        func_name = None
                        if isinstance(value.func, ast.Name):
                            func_name = value.func.id
                        elif isinstance(value.func, ast.Attribute):
                            func_name = value.func.attr

                        if func_name in ("str", "fspath") and value.args:
                            arg_name = extract_name_from_node(value.args[0])
                            if arg_name and is_path_name(arg_name):
                                violations.append(
                                    f"line {value.lineno}: dict literal "
                                    f"{func_name}({arg_name}) "
                                    f"not wrapped in repo_relative(...)"
                                )

                self.generic_visit(node)

        checker = PathStringChecker()
        checker.visit(tree)

        return violations

    # Check each module
    failures = []
    for module_name in writer_modules:
        module_path = ANALYSIS_DIR / module_name
        vios = check_module(module_path)
        if vios:
            failures.append((module_name, vios))

    if failures:
        msg = "Paths stored in output dicts must use repo_relative():\n"
        for module_name, vios in failures:
            msg += f"\n{module_name}:\n"
            for vio in vios:
                msg += f"  {vio}\n"
        pytest.fail(msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
