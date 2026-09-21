# SPDX-License-Identifier: BSD-3-Clause
"""A provenance record must describe the code that ran, not the shell's cwd.

``fit_mock_joint.py`` writes a diagnostics sidecar that Section 3 draws on, and
it recorded ``rhat_max``, ``ess_min``, ``divergences`` and wall time -- and
nothing at all about which tree produced them. That is the same gap that makes
``bench_macbook_shapeC_*.json`` unquotable.

The trap these tests exist for is specific and was hit twice in one session. On
this machine a bare ``python`` resolves ``import tengri`` to a checkout in an
unrelated project directory rather than to the worktree the shell is in. A
provenance record built from ``os.getcwd()`` would therefore stamp the
worktree's commit onto numbers produced by different code -- worse than
recording nothing, because it looks like provenance. So the central test puts
the module and the working directory in *different* git repositories and
insists the record follows the module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYSIS_DIR))

from paper1._provenance import code_provenance, provenance_line

pytestmark = pytest.mark.unit


def _git_repo(root: Path, marker: str = "x") -> str:
    """A real repository with one commit; returns its SHA.

    ``marker`` varies the committed content. Two repositories created in the
    same second with identical content, author and message produce byte-
    identical SHAs, and the central test below would then compare a value with
    itself.
    """
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *a], capture_output=True, text=True, check=True
    )
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    (root / "mod_under_test.py").write_text(f"VALUE = 1  # {marker}\n")
    run("add", "-A")
    run("commit", "-q", "-m", "initial")
    return run("rev-parse", "HEAD").stdout.strip()


class _FakeModule:
    """Stands in for an imported package, carrying only what the helper reads."""

    def __init__(self, file: Path | str | None, name="pkg", version=None):
        if file is not None:
            self.__file__ = str(file)
        self.__name__ = name
        if version is not None:
            self.__version__ = version


def test_the_record_follows_the_module_not_the_working_directory(tmp_path, monkeypatch):
    """The defect this file exists for.

    Two repositories. The module lives in one; the process runs in the other.
    A record derived from the cwd reports the wrong commit and looks correct.
    """
    module_repo = tmp_path / "where_the_code_is"
    cwd_repo = tmp_path / "where_the_shell_is"
    module_repo.mkdir()
    cwd_repo.mkdir()
    module_sha = _git_repo(module_repo, marker="module")
    cwd_sha = _git_repo(cwd_repo, marker="cwd")
    assert module_sha != cwd_sha, "the two repos must differ or this proves nothing"

    monkeypatch.chdir(cwd_repo)
    record = code_provenance(_FakeModule(module_repo / "mod_under_test.py"))

    assert record["commit"] == module_sha, (
        f"recorded {record['commit']}, which is the directory the process ran in, "
        f"not the tree the code came from ({module_sha})"
    )
    assert record["commit"] != cwd_sha
    assert Path(record["repo_root"]).resolve() == module_repo.resolve()


def test_the_path_that_was_imported_is_recorded_verbatim(tmp_path):
    """The one fact that contradicts an assumption about which tree ran."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    target = repo / "mod_under_test.py"

    record = code_provenance(_FakeModule(target))

    assert Path(record["path"]).resolve() == target.resolve()


def test_a_dirty_tree_is_reported_as_dirty(tmp_path):
    """A commit identifies a tree only if the tree was clean."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    (repo / "mod_under_test.py").write_text("VALUE = 2  # uncommitted\n")

    record = code_provenance(_FakeModule(repo / "mod_under_test.py"))

    assert record["dirty"] is True
    assert "uncommitted" in provenance_line(record)


def test_a_clean_tree_is_reported_as_clean(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)

    record = code_provenance(_FakeModule(repo / "mod_under_test.py"))

    assert record["dirty"] is False
    assert "uncommitted" not in provenance_line(record)


def test_code_outside_any_repository_reports_no_commit_rather_than_guessing(tmp_path, monkeypatch):
    """Absence must read as absence.

    Run from inside a repository, so a cwd-derived implementation would have a
    plausible SHA to offer. There is no honest commit for this module and the
    record must say so.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    outside = tmp_path / "not_a_repo"
    outside.mkdir()

    monkeypatch.chdir(repo)
    record = code_provenance(_FakeModule(outside / "mod_under_test.py"))

    assert record["commit"] is None, (
        f"invented commit {record['commit']} for code in no repository"
    )
    assert record["repo_root"] is None
    assert record["dirty"] is None
    assert "not quotable" in provenance_line(record)


def test_a_module_with_no_file_is_handled(tmp_path, monkeypatch):
    """Namespace packages and frozen modules have no ``__file__``."""
    monkeypatch.chdir(tmp_path)
    record = code_provenance(_FakeModule(None))

    assert record["path"] is None
    assert record["commit"] is None
    assert "not quotable" in provenance_line(record)


def test_a_worktree_is_resolved_although_its_dot_git_is_a_file(tmp_path):
    """The case this project actually runs in.

    A linked worktree carries a ``.git`` *file*, not a directory. An ancestor
    walk testing ``is_dir()`` would step straight past it and report the main
    checkout's commit -- the wrong branch entirely.
    """
    main = tmp_path / "main"
    main.mkdir()
    _git_repo(main)
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q", "-b", "side", str(wt)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (wt / ".git").is_file(), "fixture assumption: a worktree's .git is a file"

    record = code_provenance(_FakeModule(wt / "mod_under_test.py"))

    assert record["commit"] is not None
    assert Path(record["repo_root"]).resolve() == wt.resolve()
