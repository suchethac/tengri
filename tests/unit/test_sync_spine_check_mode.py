# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the spine notebook sync script's --check mode.

The ``--check`` mode runs the normalization operations in memory only,
comparing the would-be output against committed spine notebooks
byte-for-byte, and reports any drift without writing. The mode runs
``check_published_links()`` to validate internal cross-notebook links.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Get the path to the script in the actual project
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sync_spine_notebooks_for_docs.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_spine_subset(dest_root: Path) -> str:
    """Copy all spine sources and commits to dest_root.

    Returns the slug of the first notebook for testing.
    """
    # Copy entire notebooks and docs/spine directories
    src_notebooks = _REPO_ROOT / "notebooks"
    dest_notebooks = dest_root / "notebooks"
    dest_notebooks.mkdir(parents=True, exist_ok=True)

    src_spine = _REPO_ROOT / "docs" / "spine"
    dest_spine = dest_root / "docs" / "spine"
    dest_spine.mkdir(parents=True, exist_ok=True)

    # Copy all .py files
    for py_file in src_notebooks.glob("*.py"):
        shutil.copy2(py_file, dest_notebooks / py_file.name)

    # Copy all .ipynb files from spine
    for ipynb_file in src_spine.glob("*.ipynb"):
        shutil.copy2(ipynb_file, dest_spine / ipynb_file.name)

    # Copy docs/conf.py (needed for excluded_spine_slugs)
    src_conf = _REPO_ROOT / "docs" / "conf.py"
    dest_conf = dest_root / "docs" / "conf.py"
    shutil.copy2(src_conf, dest_conf)

    # Also copy experimental subdir if it exists
    src_exp = src_spine / "experimental"
    if src_exp.exists():
        dest_exp = dest_spine / "experimental"
        dest_exp.mkdir(parents=True, exist_ok=True)
        for ipynb_file in src_exp.glob("*.ipynb"):
            shutil.copy2(ipynb_file, dest_exp / ipynb_file.name)

    return "00_quickstart"


def test_check_mode_all_in_sync(tmp_path: Path) -> None:
    """Case (a): in-sync copies exit 0 with 'OK: N spine twins in sync'."""
    slug_1 = _copy_spine_subset(tmp_path)

    # Run check mode on the temporary tree
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--check", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    # Should pass with all notebooks in sync
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK:" in result.stdout
    assert "spine twins in sync" in result.stdout


def test_check_mode_detects_drift(tmp_path: Path) -> None:
    """Case (b): drifted twin detected via .py source change."""
    slug_1 = _copy_spine_subset(tmp_path)

    # Modify one .py source to cause drift
    py_path = tmp_path / "notebooks" / f"{slug_1}.py"
    content = py_path.read_text(encoding="utf-8")
    # Add a marker comment at the start to drift the source
    modified = "# TEST MODIFICATION\n" + content
    py_path.write_text(modified, encoding="utf-8")

    # Run check mode against the modified tree
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--check", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    # Should fail with drift detected
    assert result.returncode == 1
    assert "drifted" in result.stderr.lower()
    # Should name the specific file
    assert f"{slug_1}.ipynb" in result.stderr


def test_check_mode_writes_nothing(tmp_path: Path) -> None:
    """Case (c): check mode does not write to disk."""
    slug_1 = _copy_spine_subset(tmp_path)

    # Get mtime before check
    ipynb_path = tmp_path / "docs" / "spine" / f"{slug_1}.ipynb"
    mtime_before = ipynb_path.stat().st_mtime

    # Run check mode
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--check", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    # Verify mtime didn't change (no writes occurred)
    mtime_after = ipynb_path.stat().st_mtime
    assert mtime_before == mtime_after


def test_check_mode_help() -> None:
    """Check mode can be invoked with --help."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--root" in result.stdout
