# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``tools/check_test_tolerances.py``.

Both polarities: a loose literal the guard must count, and the tolerances it must
leave alone (tight literals, names, expressions). The ratchet itself is tested on
a temporary tree so the repository baseline is never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_test_tolerances as guard

pytestmark = pytest.mark.unit


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _count(tmp_path: Path, source: str) -> int:
    """Loose sites the guard finds in a single test file holding ``source``."""
    tests = tmp_path / "tests"
    _write(tmp_path, "tests/test_sample.py", source)
    return len(guard.collect_sites(tests, tmp_path))


class TestCountedSites:
    """Literal tolerances at or above 1e-2 are sites."""

    def test_approx_rel_literal_is_counted(self, tmp_path):
        assert _count(tmp_path, "x = pytest.approx(1.0, rel=0.05)\n") == 1

    def test_rtol_at_threshold_is_counted(self, tmp_path):
        assert _count(tmp_path, "ok = np.allclose(a, b, rtol=1e-2)\n") == 1

    def test_negative_literal_is_counted(self, tmp_path):
        assert _count(tmp_path, "ok = np.isclose(a, b, rtol=-0.05)\n") == 1

    def test_assert_allclose_is_counted(self, tmp_path):
        assert _count(tmp_path, "np.testing.assert_allclose(a, b, rtol=0.5)\n") == 1

    def test_bare_name_call_is_counted(self, tmp_path):
        assert _count(tmp_path, "ok = isclose(a, b, rel=0.1)\n") == 1

    def test_site_reports_line_and_value(self, tmp_path):
        _write(tmp_path, "tests/test_sample.py", "a = 1\n\nb = approx(2, rel=0.25)\n")
        (site,) = guard.collect_sites(tmp_path / "tests", tmp_path)
        assert site.path == "tests/test_sample.py"
        assert site.line == 3
        assert site.value == pytest.approx(0.25)


class TestIgnoredTolerances:
    """Tight literals, non-literal values and unrelated calls are not sites."""

    def test_tight_literal_is_not_counted(self, tmp_path):
        assert _count(tmp_path, "x = pytest.approx(1.0, rel=1e-6)\n") == 0

    def test_name_value_is_not_counted(self, tmp_path):
        assert _count(tmp_path, "x = pytest.approx(1.0, rel=TOL)\n") == 0

    def test_expression_value_is_not_counted(self, tmp_path):
        assert _count(tmp_path, "x = pytest.approx(1.0, rel=2 * TOL)\n") == 0

    def test_absolute_tolerance_is_not_counted(self, tmp_path):
        assert _count(tmp_path, "x = pytest.approx(1.0, abs=0.5)\n") == 0

    def test_unrelated_call_with_rel_is_not_counted(self, tmp_path):
        assert _count(tmp_path, "x = scale(1.0, rel=0.5)\n") == 0


class TestRatchet:
    """The baseline may only shrink: a rise, a new file or a fall each fail."""

    @pytest.fixture
    def tree(self, tmp_path, monkeypatch):
        """A temporary repository root wired into the guard's module constants."""
        monkeypatch.setattr(guard, "ROOT", tmp_path)
        monkeypatch.setattr(guard, "TESTS_PATH", tmp_path / "tests")
        monkeypatch.setattr(guard, "BASELINE_PATH", tmp_path / "tools" / "baseline.txt")
        _write(tmp_path, "tools/baseline.txt", "# header\n")
        return tmp_path

    def _baseline(self, tree: Path, counts: dict[str, int]) -> None:
        guard.BASELINE_PATH.write_text(guard._format_baseline(counts), encoding="utf-8")

    def test_matching_baseline_passes(self, tree):
        _write(tree, "tests/test_a.py", "x = approx(1.0, rel=0.1)\n")
        self._baseline(tree, {"tests/test_a.py": 1})
        assert guard.main([]) == 0

    def test_new_site_fails(self, tree):
        _write(tree, "tests/test_a.py", "x = approx(1.0, rel=0.1)\ny = approx(2.0, rel=0.1)\n")
        self._baseline(tree, {"tests/test_a.py": 1})
        assert guard.main([]) == 1

    def test_file_missing_from_baseline_fails(self, tree):
        _write(tree, "tests/test_a.py", "x = approx(1.0, rel=0.1)\n")
        self._baseline(tree, {})
        assert guard.main([]) == 1

    def test_stale_entry_fails(self, tree):
        _write(tree, "tests/test_a.py", "x = approx(1.0, rel=1e-6)\n")
        self._baseline(tree, {"tests/test_a.py": 1})
        assert guard.main([]) == 1

    def test_removed_file_with_entry_fails(self, tree):
        _write(tree, "tests/test_a.py", "x = 1\n")
        self._baseline(tree, {"tests/test_gone.py": 2})
        assert guard.main([]) == 1

    def test_write_baseline_then_check_passes(self, tree):
        _write(tree, "tests/test_a.py", "x = approx(1.0, rel=0.1)\n")
        assert guard.main(["--write-baseline"]) == 0
        assert guard.main([]) == 0

    def test_unparseable_test_file_exits_two(self, tree):
        _write(tree, "tests/test_bad.py", "def broken(:\n")
        assert guard.main([]) == 2
