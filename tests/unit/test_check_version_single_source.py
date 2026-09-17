# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for tools/check_version_single_source.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_check_version_matching():
    """Guard exits 0 when pyproject.toml and CITATION.cff versions match."""
    # Import here to avoid issues if the tool doesn't exist
    from tools import check_version_single_source

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create pyproject.toml
        pyproject_content = '''[project]
name = "test-pkg"
version = "1.2.3"
'''
        (tmppath / "pyproject.toml").write_text(pyproject_content)

        # Create CITATION.cff with matching version
        citation_content = '''cff-version: 1.2.0
version: 1.2.3
'''
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Should return 0
        assert check_version_single_source.main(["--root", str(tmppath)]) == 0


def test_check_version_mismatch():
    """Guard exits 1 and reports both versions when they differ."""
    from tools import check_version_single_source

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create pyproject.toml
        pyproject_content = '''[project]
name = "test-pkg"
version = "1.2.3"
'''
        (tmppath / "pyproject.toml").write_text(pyproject_content)

        # Create CITATION.cff with different version
        citation_content = '''cff-version: 1.2.0
version = 1.2.4
'''
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Should return 1
        assert check_version_single_source.main(["--root", str(tmppath)]) == 1


def test_check_version_missing_pyproject():
    """Guard exits 1 when pyproject.toml is missing."""
    from tools import check_version_single_source

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Only create CITATION.cff
        citation_content = '''cff-version: 1.2.0
version: 1.2.3
'''
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Should return 1
        assert check_version_single_source.main(["--root", str(tmppath)]) == 1


def test_check_version_argv_convention():
    """Guard accepts argv parameter per convention."""
    from tools import check_version_single_source

    # Verify that main accepts argv
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        pyproject_content = '''[project]
name = "test-pkg"
version = "1.2.3"
'''
        (tmppath / "pyproject.toml").write_text(pyproject_content)

        citation_content = '''cff-version: 1.2.0
version: 1.2.3
'''
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Call with explicit argv
        result = check_version_single_source.main(["--root", str(tmppath)])
        assert result == 0
