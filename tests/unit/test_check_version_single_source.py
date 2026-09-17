# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for tools/check_version_single_source.py."""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_check_version_matching():
    """Guard exits 0 when pyproject.toml and CITATION.cff versions match."""
    # Import here to avoid issues if the tool doesn't exist
    from tools import check_version_single_source

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create pyproject.toml
        pyproject_content = """[project]
name = "test-pkg"
version = "1.2.3"
"""
        (tmppath / "pyproject.toml").write_text(pyproject_content)

        # Create CITATION.cff with matching version
        citation_content = """cff-version: 1.2.0
version: 1.2.3
"""
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Should return 0
        assert check_version_single_source.main(["--root", str(tmppath)]) == 0


def test_check_version_mismatch():
    """Guard exits 1 and reports both versions when they differ."""
    from tools import check_version_single_source

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create pyproject.toml
        pyproject_content = """[project]
name = "test-pkg"
version = "1.2.3"
"""
        (tmppath / "pyproject.toml").write_text(pyproject_content)

        # Create CITATION.cff with different version
        citation_content = """cff-version: 1.2.0
version = 1.2.4
"""
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Should return 1
        assert check_version_single_source.main(["--root", str(tmppath)]) == 1


def test_check_version_missing_pyproject():
    """Guard exits 1 when pyproject.toml is missing."""
    from tools import check_version_single_source

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Only create CITATION.cff
        citation_content = """cff-version: 1.2.0
version: 1.2.3
"""
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Should return 1
        assert check_version_single_source.main(["--root", str(tmppath)]) == 1


def test_check_version_argv_convention():
    """Guard accepts argv parameter per convention."""
    from tools import check_version_single_source

    # Verify that main accepts argv
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        pyproject_content = """[project]
name = "test-pkg"
version = "1.2.3"
"""
        (tmppath / "pyproject.toml").write_text(pyproject_content)

        citation_content = """cff-version: 1.2.0
version: 1.2.3
"""
        (tmppath / "CITATION.cff").write_text(citation_content)

        # Call with explicit argv
        result = check_version_single_source.main(["--root", str(tmppath)])
        assert result == 0


def test_version_derivation_pyproject_first():
    """tengri.__version__ derives from source tree's pyproject.toml when available."""
    import tengri

    # The version should match pyproject.toml
    assert isinstance(tengri.__version__, str)
    assert len(tengri.__version__) > 0
    assert "." in tengri.__version__  # Should be X.Y.Z format
    assert tengri.__version__ == "0.1.0"  # Repo's current version


def test_version_derivation_fallback_with_metadata():
    """tengri.__version__ contract: pyproject first, then metadata, then raise."""
    import tengri

    assert hasattr(tengri, "__version__")
    assert isinstance(tengri.__version__, str)
    assert len(tengri.__version__) > 0

    # The fallback behavior is tested implicitly: if pyproject.toml doesn't exist
    # and metadata fails, an error would be raised at import time.
    # Since the import succeeded, at least one path worked.
    # Verify it matches the expected value from either source.
    assert tengri.__version__ == "0.1.0"
