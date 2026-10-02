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

    This test scans the writer modules mentioned in the spec and verifies
    that when they construct JSON payloads with paths, they route through
    repo_relative.
    """
    writer_modules_with_paths = [
        "surviving_mass_census.py",  # "results_dir" and "merged_from"
        "bma_combine.py",             # "generated_from"
        "fit_one.py",                 # logger calls with paths
        "_cell_provenance.py",        # "results_dir"
    ]

    for module_name in writer_modules_with_paths:
        module_path = ANALYSIS_DIR / module_name
        if not module_path.exists():
            continue

        source = module_path.read_text(encoding="utf-8")

        # Verify the module imports repo_relative if it writes paths
        has_repo_relative_import = "repo_relative" in source or "from ._paths" in source
        assert has_repo_relative_import, (
            f"{module_name} writes paths to artifacts but doesn't import repo_relative"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
