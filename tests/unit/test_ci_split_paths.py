# SPDX-License-Identifier: BSD-3-Clause
"""Test ci_split_paths script for file partitioning into contiguous slices.

The script partitions a sorted list of paths into N contiguous slices
so that each part is a contiguous block of the list, enabling parallel
CI legs without duplicate or missing files.
"""

from __future__ import annotations

import json
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


class TestBugTreeWorkflowSplit:
    """The workflow's bug/ shards must agree with the splitter's part count."""

    REPO = pathlib.Path(__file__).resolve().parents[2]
    BUG_PARTS = ("1", "2", "3")

    def _workflow(self) -> dict:
        import yaml

        return yaml.safe_load((self.REPO / ".github/workflows/tests.yml").read_text())

    def test_every_matrix_has_three_bug_shards(self) -> None:
        wf = self._workflow()
        for job in ("test", "coverage"):
            include = wf["jobs"][job]["strategy"]["matrix"]["include"]
            halves = {e["shard"]: e["bug_half"] for e in include if e.get("bug_half")}
            assert halves == {
                "regression-b1": "1",
                "regression-b2": "2",
                "regression-b3": "3",
            }, job

    def test_split_command_uses_three_parts(self) -> None:
        text = (self.REPO / ".github/workflows/tests.yml").read_text()
        calls = [ln for ln in text.splitlines() if "--root tests/regression/bug" in ln]
        assert len(calls) == 2
        assert all("--of 3" in ln for ln in calls)

    def test_b3_is_wired_like_b2(self) -> None:
        text = (self.REPO / ".github/workflows/tests.yml").read_text()
        assert '"regression-b3":' in text
        assert "matrix.shard == 'regression-b3'" in text

    def test_three_parts_partition_the_real_bug_tree(self) -> None:
        parts = []
        for n in self.BUG_PARTS:
            r = subprocess.run(
                [
                    sys.executable,
                    "tools/ci_split_paths.py",
                    "--root",
                    "tests/regression/bug",
                    "--part",
                    n,
                    "--of",
                    "3",
                ],
                capture_output=True,
                text=True,
                cwd=self.REPO,
            )
            assert r.returncode == 0, r.stderr
            parts.append(r.stdout.split())
        flat = [p for part in parts for p in part]
        assert len(flat) == len(set(flat))
        bug_tree = (self.REPO / "tests/regression/bug").glob("test_*.py")
        assert set(flat) == {str(p.relative_to(self.REPO)) for p in bug_tree}


REPO = pathlib.Path(__file__).resolve().parents[2]
LEDGER = REPO / "tools" / "ci_bug_durations.json"
BUG_ROOT = "tests/regression/bug"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/ci_split_paths.py", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def _parts(root: str, n: int, *extra: str) -> list[list[str]]:
    out = []
    for part in range(1, n + 1):
        res = _run("--root", root, "--part", str(part), "--of", str(n), *extra)
        assert res.returncode == 0, res.stderr
        out.append(res.stdout.split())
    return out


class TestCiSplitByDuration:
    """--durations balances parts by measured seconds (greedy LPT)."""

    def test_real_ledger_parts_disjoint_cover_tree_and_balanced(self) -> None:
        ledger = json.loads(LEDGER.read_text())
        parts = _parts(BUG_ROOT, 3, "--durations", str(LEDGER))
        flat = [p for part in parts for p in part]
        assert len(flat) == len(set(flat)), "parts overlap"
        tree = sorted(
            str(pathlib.Path(BUG_ROOT) / p.name) for p in (REPO / BUG_ROOT).glob("test_*.py")
        )
        assert sorted(flat) == tree
        median = sorted(ledger.values())[len(ledger) // 2]
        sums = [sum(ledger.get(p, median) for p in part) for part in parts]
        print(f"part duration sums (s): {sums}")
        assert max(sums) / min(sums) <= 1.15, sums

    def test_deterministic(self) -> None:
        a = _parts(BUG_ROOT, 3, "--durations", str(LEDGER))
        b = _parts(BUG_ROOT, 3, "--durations", str(LEDGER))
        assert a == b
        assert all(part == sorted(part) for part in a)

    def test_unknown_file_gets_median(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "t"
        root.mkdir()
        for name in ("test_a.py", "test_b.py", "test_c.py", "test_new.py"):
            (root / name).touch()
        ledger = tmp_path / "led.json"
        ledger.write_text(
            json.dumps(
                {
                    str(root / "test_a.py"): 30,
                    str(root / "test_b.py"): 10,
                    str(root / "test_c.py"): 10,
                }
            )
        )
        parts = _parts(str(root), 2, "--durations", str(ledger))
        # median is 10: a(30) -> part 1; new(10), b(10), c(10) -> part 2 (30 vs 30)
        assert [sorted(pathlib.Path(x).name for x in p) for p in parts] == [
            ["test_a.py"],
            ["test_b.py", "test_c.py", "test_new.py"],
        ]

    def test_without_durations_contiguous_unchanged(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "t"
        root.mkdir()
        for i in range(7):
            (root / f"test_{i}.py").touch()
        names = [[pathlib.Path(x).name for x in p] for p in _parts(str(root), 3)]
        assert names == [
            ["test_0.py", "test_1.py", "test_2.py"],
            ["test_3.py", "test_4.py", "test_5.py"],
            ["test_6.py"],
        ]

    def test_workflow_passes_durations_in_both_jobs(self) -> None:
        text = (REPO / ".github" / "workflows" / "tests.yml").read_text()
        splits = [
            ln
            for ln in text.splitlines()
            if "ci_split_paths.py" in ln and f"--root {BUG_ROOT}" in ln
        ]
        assert len(splits) == 2
        for ln in splits:
            assert "--durations tools/ci_bug_durations.json" in ln
