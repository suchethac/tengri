# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the conftest import-source verification guard (#2170).

The guard refuses to run the suite when the imported ``tengri`` package
lives in a different checkout or worktree than the tests — the signature
of a stale editable-install ``.pth`` in the shared venv.
"""

from pathlib import Path

import pytest
from tests.conftest import _check_tengri_source_tree_match


def _make_dev_tree(root: Path) -> Path:
    """Create a minimal src/tengri layout under root; return __init__.py."""
    src = root / "src" / "tengri"
    src.mkdir(parents=True)
    init = src / "__init__.py"
    init.write_text("")
    return init


@pytest.mark.unit
def test_passes_on_the_real_healthy_tree():
    """With no injection the guard checks the live state and stays silent.

    The guard already ran once at conftest import; this re-runs it
    explicitly so a mismatch fails here with a readable assertion
    rather than only as a collection error.
    """
    _check_tengri_source_tree_match()


@pytest.mark.unit
def test_raises_when_imported_from_a_different_tree(tmp_path):
    """A dev tree whose import resolves elsewhere is refused by name."""
    _make_dev_tree(tmp_path)
    foreign_init = _make_dev_tree(tmp_path / "other_worktree")

    with pytest.raises(RuntimeError, match="wrong source tree") as excinfo:
        _check_tengri_source_tree_match(repo_root=tmp_path, imported_file=foreign_init)

    # The message must name both trees and the remediation.
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert str(foreign_init.resolve()) in message
    assert "pip install -e ." in message


@pytest.mark.unit
def test_silent_when_no_src_directory(tmp_path):
    """No src/tengri beside the tests (installed release): no check."""
    foreign_init = _make_dev_tree(tmp_path / "other_worktree")

    # tmp_path itself has no src/tengri, so even a foreign import is
    # accepted without complaint.
    _check_tengri_source_tree_match(repo_root=tmp_path, imported_file=foreign_init)


@pytest.mark.unit
def test_accepts_a_symlinked_spelling_of_the_same_tree(tmp_path):
    """Paths are compared resolved: a symlink alias is not a mismatch."""
    init = _make_dev_tree(tmp_path)
    alias = tmp_path / "alias_root"
    alias.symlink_to(tmp_path)

    _check_tengri_source_tree_match(repo_root=alias, imported_file=init)
