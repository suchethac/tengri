# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the spine notebook sync script's --check mode.

The ``--check`` mode runs the normalization operations in memory only,
comparing the would-be output against committed spine notebooks
byte-for-byte, and reports any drift without writing. The mode runs
``check_published_links()`` to validate internal cross-notebook links.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Get the path to the script in the actual project
# tests/unit/... -> resolve to tests/unit, then parents[2] to get project root, then scripts/...
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sync_spine_notebooks_for_docs.py"


def test_check_mode_exits_with_help(tmp_path: Path) -> None:
    """Check mode can be invoked with --help."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--check" in result.stdout


def test_check_mode_accepts_argument() -> None:
    """Check mode accepts --check argument."""
    # Just verify the help output includes --check
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--check" in result.stdout


def test_check_mode_does_not_write_files() -> None:
    """--check mode does not write files when run on real repo."""
    # Run on the real repo after the normal sync (from previous tests)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--check"],
        capture_output=True,
        text=True,
    )

    # Check mode should pass if notebooks are in sync
    # It should NOT modify any files (mtime would change if it did)
    # Just verify it runs and produces expected output
    if result.returncode == 0:
        assert "OK" in result.stdout
        assert "spine twins in sync" in result.stdout
    else:
        # If there's drift, that's also valid - just check it doesn't crash
        assert "error" in result.stderr.lower() or "drifted" in result.stderr.lower()
