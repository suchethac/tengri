# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2784: the local-path guard must see symlink targets.

``tools/check_no_local_paths.py`` walks ``git ls-files`` and skips anything
that is not a regular file. A symlink whose *target* is an absolute home path
(``/Users/<name>/...``) therefore passed, and a broken one was skipped outright.
The target is part of the committed content, so the guard must match it. It
must not follow the link into the file it names.

Marker: contract
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_no_local_paths.py"


def _repo_with_tracked(tmp_path: Path, name: str, link_target: str) -> Path:
    """A throwaway git repo holding the real tool and one tracked symlink."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy(_TOOL, repo / "tools" / "check_no_local_paths.py")
    (repo / name).symlink_to(link_target)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    return repo


def _run_guard(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "tools" / "check_no_local_paths.py")],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "target",
    [
        "/Users/alice/Projects/tengri/data/real.h5",  # target need not exist
        "/home/bob/scratch/model.npz",
    ],
)
def test_symlink_to_absolute_home_path_fails_the_guard(tmp_path, target):
    repo = _repo_with_tracked(tmp_path, "link_to_home", target)
    result = _run_guard(repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "link_to_home" in result.stderr


def test_symlink_to_relative_or_ci_path_passes(tmp_path):
    repo = _repo_with_tracked(tmp_path, "rel_link", "../data/real.h5")
    result = _run_guard(repo)
    assert result.returncode == 0, result.stdout + result.stderr
