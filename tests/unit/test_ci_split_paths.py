# SPDX-License-Identifier: BSD-3-Clause
"""Test ci_split_paths script for file partitioning into contiguous slices.

The script partitions a sorted list of paths into N contiguous slices
so that each part is a contiguous block of the list, enabling parallel
CI legs without duplicate or missing files.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


class TestCiSplitPaths:
    """Test ci_split_paths.py behavior."""

    def test_split_into_three_parts_disjoint_and_exhaustive(self, tmp_path: pathlib.Path) -> None:
        """Seven files partitioned into 3 parts should be disjoint and exhaustive."""
        root = tmp_path / "tests"
        root.mkdir()

        # Create 7 test files
        files = [f"test_{i:02d}.py" for i in range(7)]
        for f in files:
            (root / f).touch()

        # Get each part
        parts = {}
        for part_num in range(1, 4):
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/ci_split_paths.py",
                    "--root",
                    str(root),
                    "--part",
                    str(part_num),
                    "--of",
                    "3",
                ],
                capture_output=True,
                text=True,
                cwd=pathlib.Path(__file__).resolve().parents[2],
            )
            assert result.returncode == 0, f"Part {part_num} failed: {result.stderr}"
            lines = result.stdout.strip().split("\n")
            parts[part_num] = [l for l in lines if l]

        # Check that parts are disjoint
        all_returned = set()
        for part_num in range(1, 4):
            part_set = set(parts[part_num])
            assert len(part_set) == len(parts[part_num]), f"Part {part_num} has duplicates"
            assert not (all_returned & part_set), f"Parts {part_num} overlaps with earlier parts"
            all_returned |= part_set

        # Check that union of all parts equals original files
        expected = {str(root / f) for f in files}
        assert all_returned == expected, "Parts do not cover all files"

    def test_empty_part_exits_with_2(self, tmp_path: pathlib.Path) -> None:
        """Requesting a part beyond the file count should exit 2."""
        root = tmp_path / "tests"
        root.mkdir()

        # Create only 2 files but request part 10
        (root / "test_01.py").touch()
        (root / "test_02.py").touch()

        result = subprocess.run(
            [
                sys.executable,
                "tools/ci_split_paths.py",
                "--root",
                str(root),
                "--part",
                "10",
                "--of",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 2, f"Empty part should exit 2, got {result.returncode}"
        assert "empty part" in result.stderr.lower(), (
            f"Error message should mention empty part: {result.stderr}"
        )

    def test_join_flag_outputs_single_line(self, tmp_path: pathlib.Path) -> None:
        """Using --join should output all files on a single space-joined line."""
        root = tmp_path / "tests"
        root.mkdir()

        # Create 3 test files
        for i in range(3):
            (root / f"test_{i:02d}.py").touch()

        result = subprocess.run(
            [
                sys.executable,
                "tools/ci_split_paths.py",
                "--root",
                str(root),
                "--part",
                "1",
                "--of",
                "3",
                "--join",
            ],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 1, f"--join should output one line, got {len(lines)}: {lines}"
        # Should be space-joined
        assert " " in lines[0] or len(lines[0].split()) >= 1

    def test_pattern_flag_filters_files(self, tmp_path: pathlib.Path) -> None:
        """Using --pattern should filter files."""
        root = tmp_path / "tests"
        root.mkdir()

        # Create test files and other files
        (root / "test_01.py").touch()
        (root / "test_02.py").touch()
        (root / "setup.py").touch()
        (root / "conftest.py").touch()

        result = subprocess.run(
            [
                sys.executable,
                "tools/ci_split_paths.py",
                "--root",
                str(root),
                "--part",
                "1",
                "--of",
                "2",
                "--pattern",
                "test_*.py",
            ],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        output = result.stdout.strip()
        # Should only contain test_*.py files, not setup.py or conftest.py
        assert "setup.py" not in output, f"Pattern should exclude setup.py: {output}"
        assert "conftest.py" not in output, f"Pattern should exclude conftest.py: {output}"

    def test_identical_results_across_two_calls(self, tmp_path: pathlib.Path) -> None:
        """Running the same split twice should return identical results."""
        root = tmp_path / "tests"
        root.mkdir()

        # Create 5 test files
        for i in range(5):
            (root / f"test_{i:02d}.py").touch()

        # First run
        result1 = subprocess.run(
            [
                sys.executable,
                "tools/ci_split_paths.py",
                "--root",
                str(root),
                "--part",
                "2",
                "--of",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )

        # Second run
        result2 = subprocess.run(
            [
                sys.executable,
                "tools/ci_split_paths.py",
                "--root",
                str(root),
                "--part",
                "2",
                "--of",
                "3",
            ],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )

        assert result1.stdout == result2.stdout, "Identical calls should return identical output"
