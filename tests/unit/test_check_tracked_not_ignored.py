# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``tools/check_tracked_not_ignored.py``.

The guard's whole value rests on one flag. ``git check-ignore`` skips tracked
paths unless it is given ``--no-index``, which is exactly backwards here: drop
the flag and the guard reports a clean repository for every input it exists to
catch, forever, with no way to tell that apart from a genuinely clean tree.

So these tests build throwaway repositories and assert on behavior rather than
on the source text: a tracked-and-ignored file must be found, a negated one
must not, and -- the case that motivated the guard -- an exception inside an
excluded *directory* must be reported, because git cannot re-include it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_tracked_not_ignored.py"
_spec = importlib.util.spec_from_file_location("check_tracked_not_ignored", _TOOL)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _repo(tmp_path: Path, files: dict[str, str], gitignore: str) -> Path:
    """A real git repo with *files* committed and *gitignore* in place."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    (tmp_path / ".gitignore").write_text(gitignore)
    # -f so the fixture can stage paths the ignore file already excludes, which
    # is precisely the state being reproduced.
    subprocess.run(["git", "add", "-f", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def _offenders(repo: Path, monkeypatch) -> list[str]:
    monkeypatch.setattr(guard, "REPO_ROOT", repo)
    return [path for path, _rule in guard._excluded(guard._tracked())]


def test_a_tracked_file_its_gitignore_excludes_is_reported(tmp_path, monkeypatch):
    repo = _repo(tmp_path, {"keep.py": "x = 1\n", "cache/out.npz": "junk"}, "*.npz\n")
    assert _offenders(repo, monkeypatch) == ["cache/out.npz"]


def test_a_negated_file_is_not_reported(tmp_path, monkeypatch):
    """A `!` rule is the repository saying it means to keep the file."""
    repo = _repo(tmp_path, {"data/ref.npz": "junk"}, "*.npz\n!data/ref.npz\n")
    assert _offenders(repo, monkeypatch) == []


def test_an_ordinary_repository_is_clean(tmp_path, monkeypatch):
    """No false positives on files no rule mentions."""
    repo = _repo(tmp_path, {"a.py": "x = 1\n", "docs/b.md": "hi\n"}, "*.log\n")
    assert _offenders(repo, monkeypatch) == []


def test_an_exception_inside_an_excluded_directory_is_still_reported(tmp_path, monkeypatch):
    """Git cannot re-include a file whose parent directory is excluded.

    This is the case that motivated the guard rather than a hypothetical: the
    repository carried `examples/*/data/` beside a `!` line for one committed
    file under it, and the negation was inert -- git never descends into an
    excluded directory to see it. Excluding `dir/**` instead is what makes the
    exception work, and the guard has to keep telling the two apart.
    """
    repo = _repo(tmp_path, {"ex/data/real.txt": "input"}, "ex/*/\n!ex/data/real.txt\n")
    assert _offenders(repo, monkeypatch) == ["ex/data/real.txt"]

    repo2 = _repo(
        tmp_path / "second",
        {"ex/data/real.txt": "input"},
        "ex/*/**\n!ex/data/real.txt\n",
    )
    assert _offenders(repo2, monkeypatch) == []


def test_this_repository_passes():
    """The guard is green here, and stays a real line rather than an aspiration."""
    result = subprocess.run(
        [sys.executable, str(_TOOL)],
        capture_output=True,
        text=True,
        cwd=_TOOL.parent.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_missing_git_checkout_is_not_a_failure(tmp_path, monkeypatch, capsys):
    """An sdist has no .git; the guard must skip, not crash the build."""
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        guard._tracked()
    assert guard.main() == 0
    assert "not a git checkout" in capsys.readouterr().out
