# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for check_numeric_guards.py.

Tests the drift-proof ledger logic (#2050):
1. Count increases (new unsafe site) triggers RED.
2. Count decreases (site removed, improvement) triggers RED with ratchet-down message.
3. Count equal everywhere triggers GREEN.
4. Two violations on one line are counted as 2 (no key collapse).
5. Same content with an inserted comment above a violation produces identical bucket verdict
   (drift-neutrality test).
"""

import importlib.util
import tempfile
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_numeric_guards.py"
_spec = importlib.util.spec_from_file_location("check_numeric_guards", _TOOL)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class TestBucketViolations:
    """Test the bucket_violations_by_pattern function."""

    def test_buckets_group_by_file_and_pattern(self):
        """Violations group into (file, pattern) buckets."""
        violations_by_file = {
            "file1.py": [(10, "pattern_a"), (20, "pattern_a"), (30, "pattern_b")],
            "file2.py": [(15, "pattern_a")],
        }
        result = mod.bucket_violations_by_pattern(violations_by_file)
        assert result[("file1.py", "pattern_a")] == [10, 20]
        assert result[("file1.py", "pattern_b")] == [30]
        assert result[("file2.py", "pattern_a")] == [15]

    def test_line_numbers_are_sorted(self):
        """Line numbers within each bucket are sorted."""
        violations_by_file = {
            "file.py": [(30, "pattern"), (10, "pattern"), (20, "pattern")],
        }
        result = mod.bucket_violations_by_pattern(violations_by_file)
        assert result[("file.py", "pattern")] == [10, 20, 30]

    def test_two_violations_same_line_counted_separately(self):
        """Two violations on the same line with different patterns are distinct."""
        violations_by_file = {
            "file.py": [(10, "pattern_a"), (10, "pattern_b")],
        }
        result = mod.bucket_violations_by_pattern(violations_by_file)
        assert result[("file.py", "pattern_a")] == [10]
        assert result[("file.py", "pattern_b")] == [10]


class TestLedgerIO:
    """Test ledger load and save."""

    def test_save_and_load_roundtrip(self):
        """Ledger survives save and load roundtrip."""
        ledger = {
            ("file1.py", "pattern_a"): 2,
            ("file1.py", "pattern_b"): 1,
            ("file2.py", "pattern_a"): 3,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_file = Path(tmpdir) / "ledger"
            # Temporarily swap ledger file
            orig_ledger_file = mod.LEDGER_FILE
            try:
                mod.LEDGER_FILE = ledger_file
                mod.save_ledger(ledger)
                loaded = mod.load_ledger()
                assert loaded == ledger
            finally:
                mod.LEDGER_FILE = orig_ledger_file

    def test_ledger_format_has_pipe_separator(self):
        """Ledger format uses | to separate path and pattern."""
        ledger = {("file.py", "pattern"): 2}
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_file = Path(tmpdir) / "ledger"
            orig_ledger_file = mod.LEDGER_FILE
            try:
                mod.LEDGER_FILE = ledger_file
                mod.save_ledger(ledger)
                content = ledger_file.read_text()
                assert "file.py | pattern -> 2" in content
            finally:
                mod.LEDGER_FILE = orig_ledger_file


class TestVerdictLogic:
    """Test verdict logic via exit codes: new→1, stale→1, equal→0."""

    def test_new_offenders_exit_1(self, tmp_path, monkeypatch):
        """Live count > ledger count triggers exit 1."""
        src_dir = tmp_path / "src" / "tengri"
        src_dir.mkdir(parents=True)

        # Create a file with 3 violations of the same pattern
        test_file = src_dir / "test_module.py"
        test_file.write_text(
            '"""Test module.\n"""\n'
            "import jax.numpy as jnp\n"
            "x = jnp.maximum(y, 1e-25)  # violation 1\n"
            "z = jnp.maximum(y, 1e-25)  # violation 2\n"
            "w = jnp.maximum(y, 1e-25)  # violation 3\n"
        )

        ledger_file = tmp_path / "ledger"
        ledger_text = "src/tengri/test_module.py | jnp.maximum with floor 1e-25 < 1e-20 -> 2\n"
        ledger_file.write_text(ledger_text)

        orig_root = mod.ROOT
        orig_ledger = mod.LEDGER_FILE
        orig_tracked = mod._tracked_python_files_in_src

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "LEDGER_FILE", ledger_file)
        monkeypatch.setattr(
            mod,
            "_tracked_python_files_in_src",
            lambda: [f for f in src_dir.glob("*.py")],
        )

        try:
            result = mod.main([])
            # Should fail: live has 3 violations, ledger has 2
            assert result == 1
        finally:
            mod.ROOT = orig_root
            mod.LEDGER_FILE = orig_ledger
            mod._tracked_python_files_in_src = orig_tracked

    def test_stale_entries_exit_1(self, tmp_path, monkeypatch):
        """Live count < ledger count triggers exit 1."""
        src_dir = tmp_path / "src" / "tengri"
        src_dir.mkdir(parents=True)

        # Create a file with only 1 violation
        test_file = src_dir / "test_module.py"
        test_file.write_text(
            '"""Test module.\n"""\n'
            "import jax.numpy as jnp\n"
            "x = jnp.maximum(y, 1e-25)  # violation 1\n"
        )

        ledger_file = tmp_path / "ledger"
        ledger_text = "src/tengri/test_module.py | jnp.maximum with floor 1e-25 < 1e-20 -> 2\n"
        ledger_file.write_text(ledger_text)

        orig_root = mod.ROOT
        orig_ledger = mod.LEDGER_FILE
        orig_tracked = mod._tracked_python_files_in_src

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "LEDGER_FILE", ledger_file)
        monkeypatch.setattr(
            mod,
            "_tracked_python_files_in_src",
            lambda: [f for f in src_dir.glob("*.py")],
        )

        try:
            result = mod.main([])
            # Ledger says 2, live has 1 — mismatch triggers fail
            assert result == 1
        finally:
            mod.ROOT = orig_root
            mod.LEDGER_FILE = orig_ledger
            mod._tracked_python_files_in_src = orig_tracked

    def test_equal_counts_exit_0(self, tmp_path, monkeypatch):
        """Live and ledger counts equal triggers exit 0."""
        src_dir = tmp_path / "src" / "tengri"
        src_dir.mkdir(parents=True)

        # Create a file with 2 violations of the same pattern
        test_file = src_dir / "test_module.py"
        test_file.write_text(
            '"""Test module.\n"""\n'
            "import jax.numpy as jnp\n"
            "x = jnp.maximum(y, 1e-25)  # violation 1\n"
            "z = jnp.maximum(y, 1e-25)  # violation 2\n"
        )

        ledger_file = tmp_path / "ledger"
        ledger_text = "src/tengri/test_module.py | jnp.maximum with floor 1e-25 < 1e-20 -> 2\n"
        ledger_file.write_text(ledger_text)

        orig_root = mod.ROOT
        orig_ledger = mod.LEDGER_FILE
        orig_tracked = mod._tracked_python_files_in_src

        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "LEDGER_FILE", ledger_file)
        monkeypatch.setattr(
            mod,
            "_tracked_python_files_in_src",
            lambda: [f for f in src_dir.glob("*.py")],
        )

        try:
            result = mod.main([])
            # Live and ledger both empty — equal counts
            assert result == 0
        finally:
            mod.ROOT = orig_root
            mod.LEDGER_FILE = orig_ledger
            mod._tracked_python_files_in_src = orig_tracked


class TestDriftNeutrality:
    """Test that line-number drift does not affect ledger verdict.

    This is the core property #2050 enforces: if the only change is adding
    or removing lines above a violation (not the violation itself), the
    bucket verdict should remain identical.
    """

    def test_added_comment_above_violation_produces_same_verdict(self):
        """Adding a comment above a violation does not change bucket verdict.

        This simulates the drift-neutrality property: the bucket
        (file.py, "pattern_a") should have count=2 regardless of whether
        there's a comment above the violations.
        """
        # Two violations with their original line numbers
        violations_before = {
            "file.py": [(10, "pattern_a"), (20, "pattern_a")],
        }
        buckets_before = mod.bucket_violations_by_pattern(violations_before)
        counts_before = {key: len(lines) for key, lines in buckets_before.items()}

        # Same violations but with an inserted comment line at line 5
        # (all line numbers >= 10 would shift by +1, but we're grouping by pattern, not line)
        violations_after = {
            "file.py": [(11, "pattern_a"), (21, "pattern_a")],  # lines shifted
        }
        buckets_after = mod.bucket_violations_by_pattern(violations_after)
        counts_after = {key: len(lines) for key, lines in buckets_after.items()}

        # The bucket key is (file, pattern), so both should map to the same bucket
        # and have the same count.
        assert ("file.py", "pattern_a") in counts_before
        assert ("file.py", "pattern_a") in counts_after
        assert counts_before[("file.py", "pattern_a")] == 2
        assert counts_after[("file.py", "pattern_a")] == 2
        # The verdict (pass/fail) depends only on count, not line numbers
        assert counts_before == counts_after


class TestMainFunction:
    """Test the main() entrypoint with argv convention."""

    def test_main_respects_regen_flag(self, tmp_path):
        """main() with --regen flag regenerates the ledger."""
        ledger_file = tmp_path / "ledger"

        orig_ledger_file = mod.LEDGER_FILE
        try:
            mod.LEDGER_FILE = ledger_file
            # Call main with --regen
            result = mod.main(["--regen"])
            assert result == 0
            # Ledger file should now exist and be non-empty
            assert ledger_file.exists()
            content = ledger_file.read_text()
            # Should have at least one bucket (from actual src/ scan)
            assert "src/tengri/" in content
        finally:
            mod.LEDGER_FILE = orig_ledger_file
