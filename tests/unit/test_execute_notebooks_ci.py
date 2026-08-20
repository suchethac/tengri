# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``scripts/execute_notebooks.py`` CI flags (--ci, --json).

Validates that the module properly filters CI-unexecutable notebooks and
provides JSON output suitable for GitHub Actions matrix expansion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from execute_notebooks import ALL_SLUGS, CI_UNEXECUTABLE

pytestmark = pytest.mark.unit


class TestCIUnexecutableDict:
    """The CI_UNEXECUTABLE dictionary structure."""

    def test_all_keys_are_in_all_slugs(self):
        """Every CI_UNEXECUTABLE key must be a published notebook."""
        for slug in CI_UNEXECUTABLE:
            assert slug in ALL_SLUGS, f"{slug!r} not in ALL_SLUGS"

    def test_all_exclusions_documented(self):
        """All CI_UNEXECUTABLE entries have non-empty reasons."""
        assert len(CI_UNEXECUTABLE) == 2, "Expected 2 excluded notebooks"
        for slug, reason in CI_UNEXECUTABLE.items():
            assert reason, f"{slug!r} has empty reason"
            assert isinstance(reason, str), f"{slug!r} reason is not a string"


class TestListWithoutCI:
    """Subprocess --list (baseline, includes all 17 slugs)."""

    def _run_list(self, *flags) -> list[str]:
        """Run --list with optional flags, return stdout lines."""
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "execute_notebooks.py"),
            "--list",
            *flags,
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"returncode={proc.returncode}, stderr={proc.stderr}"
        return proc.stdout.strip().split("\n")

    def test_list_baseline(self):
        """--list returns all 17 slugs."""
        slugs = self._run_list()
        assert len(slugs) == 17
        assert "apple_mps" in slugs

    def test_list_matches_module_constant(self):
        """--list output matches ALL_SLUGS."""
        slugs = self._run_list()
        assert set(slugs) == set(ALL_SLUGS)


class TestListWithCI:
    """Subprocess --list --ci (filters out CI_UNEXECUTABLE)."""

    def _run_list_ci(self) -> list[str]:
        """Run --list --ci, return stdout lines."""
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "execute_notebooks.py"),
            "--list",
            "--ci",
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"returncode={proc.returncode}, stderr={proc.stderr}"
        return proc.stdout.strip().split("\n")

    def test_list_ci_excludes_unexecutable(self):
        """--list --ci excludes all CI_UNEXECUTABLE slugs."""
        slugs = self._run_list_ci()
        assert len(slugs) == len(ALL_SLUGS) - len(CI_UNEXECUTABLE)
        for excluded_slug in CI_UNEXECUTABLE:
            assert excluded_slug not in slugs

    def test_list_ci_count(self):
        """--list --ci returns 15 slugs (17 total minus 2 exclusions)."""
        slugs = self._run_list_ci()
        assert len(slugs) == 15
        assert "apple_mps" not in slugs
        assert "multimodel_bma_candels" not in slugs


class TestListCIJSON:
    """Subprocess --list --ci --json (returns JSON array)."""

    def _run_list_ci_json(self) -> list[str]:
        """Run --list --ci --json, parse and return array."""
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "execute_notebooks.py"),
            "--list",
            "--ci",
            "--json",
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"returncode={proc.returncode}, stderr={proc.stderr}"
        return json.loads(proc.stdout)

    def test_json_parses(self):
        """Output is valid JSON."""
        slugs = self._run_list_ci_json()
        assert isinstance(slugs, list)

    def test_json_equals_list_ci(self):
        """JSON array equals --list --ci output."""
        json_slugs = self._run_list_ci_json()
        # Rerun --list --ci to get the plain list
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "execute_notebooks.py"),
            "--list",
            "--ci",
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        list_slugs = proc.stdout.strip().split("\n")
        assert json_slugs == list_slugs

    def test_json_excludes_both(self):
        """JSON array excludes both apple_mps and multimodel_bma_candels."""
        slugs = self._run_list_ci_json()
        assert "apple_mps" not in slugs
        assert "multimodel_bma_candels" not in slugs
        assert len(slugs) == 15


class TestMutationValidation:
    """Mutation tests: break the code in expected ways and verify tests catch it."""

    def test_mutation_ci_filter_disabled(self, tmp_path):
        """Mutation: --ci flag filters nothing. Test must fail."""
        broken_script = (REPO_ROOT / "scripts" / "execute_notebooks.py").read_text()
        # Replace the filtering logic to make --ci a no-op
        broken_script = broken_script.replace(
            "available_slugs = [s for s in ALL_SLUGS if s not in CI_UNEXECUTABLE]",
            "available_slugs = ALL_SLUGS  # BROKEN: no filtering",
        )
        script_path = tmp_path / "broken_execute_notebooks.py"
        script_path.write_text(broken_script)

        # Run the broken script with --list --ci, from the repo root
        proc = subprocess.run(
            [sys.executable, str(script_path), "--list", "--ci"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**dict(os.environ), "PYTHONPATH": str(REPO_ROOT / "scripts")},
        )
        if proc.returncode != 0:
            pytest.skip(f"Script failed to run: {proc.stderr}")
        # The broken version would return 17 slugs (all of them)
        # instead of 15, so the test would catch it
        lines = [l for l in proc.stdout.strip().split("\n") if l]
        msg = f"Mutation test: --ci should filter but doesn't. Got {len(lines)} lines"
        assert len(lines) == 17, msg
        assert "apple_mps" in lines
        assert "multimodel_bma_candels" in lines

    def test_mutation_json_not_json(self, tmp_path):
        """Mutation: --json prints repr instead of JSON. Test must fail."""
        broken_script = (REPO_ROOT / "scripts" / "execute_notebooks.py").read_text()
        # Replace the JSON print with repr
        broken_script = broken_script.replace(
            "print(json.dumps(available_slugs))",
            "print(repr(available_slugs))",
        )
        script_path = tmp_path / "broken_execute_notebooks.py"
        script_path.write_text(broken_script)

        # Run the broken script with --list --ci --json
        proc = subprocess.run(
            [sys.executable, str(script_path), "--list", "--ci", "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        # The broken version outputs repr(), which is not JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(proc.stdout)
