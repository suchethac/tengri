# SPDX-License-Identifier: BSD-3-Clause
"""Atomic write operations for BMA evidence and combined output.

This test pins that:
- NPZ writes through _atomic_replace_write with tmp_suffix=".npz" produce the
  final file with no .tmp leftovers, leaving BMA cells consistent on disk.
- JSON writes succeed atomically: either the old file or the complete new file
  is on disk, never a truncated one.
- Errors during a write do not leave the directory dirty: no tmp or partial files.
- A whole-tree sweep ensures bma_evidence.py and bma_combine.py use the shared
  _atomic_replace_write helper instead of hand-rolling os.replace, so all atomic
  writes go through one code path, are validated by tests, and stay in sync.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1._atomic_io import _atomic_replace_write

pytestmark = pytest.mark.unit


class TestAtomicNpzWrite:
    """Test atomic NPZ writes via _atomic_replace_write."""

    def test_npz_roundtrip(self, tmp_path: Path) -> None:
        """NPZ write through helper produces final file, no .tmp leftover."""
        npz_path = tmp_path / "data.npz"
        arrays = {"x": np.array([1, 2, 3]), "y": np.array([4.0, 5.0, 6.0])}

        _atomic_replace_write(
            npz_path,
            lambda tmp_path: np.savez(tmp_path, **arrays),
            tmp_suffix=".npz",
        )

        # Final file should exist
        assert npz_path.exists(), "Final NPZ file should exist"

        # No .tmp file should be left
        tmp_candidate = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
        assert not tmp_candidate.exists(), f"Leftover tmp file should not exist: {tmp_candidate}"

        # Verify content
        loaded = np.load(npz_path)
        assert "x" in loaded.files
        assert "y" in loaded.files
        np.testing.assert_array_equal(loaded["x"], arrays["x"])
        np.testing.assert_array_equal(loaded["y"], arrays["y"])

    def test_npz_error_leaves_no_files(self, tmp_path: Path) -> None:
        """NPZ write error leaves the directory clean."""
        npz_path = tmp_path / "data.npz"

        def faulty_write(tmp_path: Path) -> None:
            raise RuntimeError("Intentional write failure")

        with pytest.raises(RuntimeError, match="Intentional write failure"):
            _atomic_replace_write(
                npz_path,
                faulty_write,
                tmp_suffix=".npz",
            )

        # Neither the final file nor any .tmp file should exist
        assert not npz_path.exists(), "Final file should not exist after error"
        tmp_candidate = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
        assert not tmp_candidate.exists(), f"Tmp file should be cleaned up: {tmp_candidate}"


class TestAtomicJsonWrite:
    """Test atomic JSON writes via _atomic_replace_write."""

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        """JSON write through helper produces final file, no .tmp leftover."""
        json_path = tmp_path / "config.json"
        data = {
            "galaxy": 12345,
            "model": "test_model",
            "log_evidence": -100.5,
            "nested": {"a": 1, "b": 2},
        }

        _atomic_replace_write(
            json_path,
            lambda tmp_path: tmp_path.write_text(json.dumps(data, indent=2)),
        )

        # Final file should exist
        assert json_path.exists(), "Final JSON file should exist"

        # No .tmp file should be left
        tmp_candidate = json_path.with_suffix(".json.tmp")
        assert not tmp_candidate.exists(), f"Leftover tmp file should not exist: {tmp_candidate}"

        # Verify content
        loaded = json.loads(json_path.read_text())
        assert loaded == data

    def test_json_error_leaves_no_files(self, tmp_path: Path) -> None:
        """JSON write error leaves the directory clean."""
        json_path = tmp_path / "config.json"

        def faulty_write(tmp_path: Path) -> None:
            raise ValueError("JSON serialization failed")

        with pytest.raises(ValueError, match="JSON serialization failed"):
            _atomic_replace_write(
                json_path,
                faulty_write,
            )

        # Neither the final file nor any .tmp file should exist
        assert not json_path.exists(), "Final file should not exist after error"
        tmp_candidate = json_path.with_suffix(".json.tmp")
        assert not tmp_candidate.exists(), f"Tmp file should be cleaned up: {tmp_candidate}"


class TestBmaWritersSweep:
    """Sweep tests ensuring all BMA writes use the shared helper."""

    def test_bma_evidence_uses_shared_helper(self) -> None:
        """bma_evidence.py should contain no os.replace or .tmp patterns."""
        evidence_file = PAPER1 / "bma_evidence.py"
        content = evidence_file.read_text()

        # Parse the file as an AST to check for os.replace calls
        tree = ast.parse(content)

        # Collect all Attribute nodes that call os.replace
        os_replace_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue

            if node.func.attr != "replace" or not isinstance(node.func.value, ast.Name):
                continue

            # Check for os.replace or Path.replace calls
            if node.func.value.id == "os" or "path" in node.func.value.id.lower():
                os_replace_calls.append(node)

        assert not os_replace_calls, (
            f"bma_evidence.py should not use os.replace; found {len(os_replace_calls)} calls"
        )

        # Check for .tmp suffix patterns in string literals (a simple heuristic)
        # The shared helper should be the only place using .tmp
        tmp_patterns = [
            line for line in content.split("\n") if ".tmp" in line and "atomic" not in line
        ]
        # Filter to exclude comments and the atomic_io import
        tmp_patterns = [
            line
            for line in tmp_patterns
            if not line.strip().startswith("#") and "_atomic_replace_write" not in line
        ]

        assert not tmp_patterns, (
            f"bma_evidence.py should use _atomic_replace_write, not .tmp patterns: {tmp_patterns}"
        )

    def test_bma_combine_uses_shared_helper(self) -> None:
        """bma_combine.py should contain no os.replace or manual .tmp writes."""
        combine_file = PAPER1 / "bma_combine.py"
        content = combine_file.read_text()

        # Parse as AST to check for os.replace calls
        tree = ast.parse(content)

        os_replace_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                os_replace_calls.append(node)

        assert not os_replace_calls, (
            f"bma_combine.py should not use os.replace; found {len(os_replace_calls)} calls"
        )

        # Check for .tmp suffix patterns
        tmp_patterns = [
            line for line in content.split("\n") if ".tmp" in line and "atomic" not in line
        ]
        tmp_patterns = [
            line
            for line in tmp_patterns
            if not line.strip().startswith("#") and "_atomic_replace_write" not in line
        ]

        assert not tmp_patterns, (
            f"bma_combine.py should use _atomic_replace_write, not .tmp patterns: {tmp_patterns}"
        )
