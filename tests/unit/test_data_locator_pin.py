# SPDX-License-Identifier: BSD-3-Clause
"""Test data locator hermeticity pin (#2329)."""

import pytest

pytestmark = pytest.mark.unit


def test_data_dirs_ancestor_walk_disabled_by_env_var(tmp_path, monkeypatch):
    """With TENGRI_DATA_NO_ANCESTOR_WALK set, ancestor directories are not searched.

    This pins discovery to the repository under test (the package's own repo
    root + its data/), disabling the ancestor walk. Tests run in nested
    worktrees cannot find data from ancestor directories, matching the CI
    environment where such files are absent (#2329).
    """
    from tengri._data_setup import data_dirs

    # Set the env var to disable ancestor walk
    monkeypatch.setenv("TENGRI_DATA_NO_ANCESTOR_WALK", "1")

    # Create a parent/child directory structure with a data file in the parent
    parent = tmp_path / "parent"
    child = tmp_path / "parent" / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "data").mkdir()
    (parent / "data" / "test_file.h5").write_text("dummy")

    # Change to the child directory and check that the parent's data/
    # is NOT found when the pin is set
    monkeypatch.chdir(child)

    result = data_dirs()
    result_strs = [str(d) for d in result]

    # The parent's data/ directory should NOT be in the list when pinned
    assert not any(str(parent / "data") in s for s in result_strs), (
        f"With TENGRI_DATA_NO_ANCESTOR_WALK set, ancestor directory should not be searched. "
        f"Got: {result_strs}"
    )


def test_data_dirs_ancestor_walk_enabled_by_default(tmp_path, monkeypatch):
    """Without TENGRI_DATA_NO_ANCESTOR_WALK set, ancestor directories ARE searched.

    This is the default behavior and documents the pre-fix behavior honestly.
    The fix applied unconditionally in conftest.py, so this test verifies the
    pin's ability to gate the behavior.
    """
    from tengri._data_setup import data_dirs

    # Make sure the env var is NOT set
    monkeypatch.delenv("TENGRI_DATA_NO_ANCESTOR_WALK", raising=False)

    # Create a parent/child directory structure with a data file in the parent
    parent = tmp_path / "parent"
    child = tmp_path / "parent" / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "data").mkdir()
    (parent / "data" / "test_file.h5").write_text("dummy")

    # Change to the child directory and check that the parent's data/
    # IS found when the pin is NOT set
    monkeypatch.chdir(child)

    result = data_dirs()
    result_strs = [str(d) for d in result]

    # The parent's data/ directory SHOULD be in the list when not pinned
    assert any(str(parent / "data") == s for s in result_strs), (
        f"Without TENGRI_DATA_NO_ANCESTOR_WALK, ancestor directory should be searched. "
        f"Expected {parent / 'data'} in {result_strs}"
    )


def test_data_dirs_package_dirs_always_searched_despite_pin(tmp_path, monkeypatch):
    """Package data directories are always searched, even with the pin set.

    The pin disables the ancestor walk but still allows discovery from the
    package's own repository root, which is necessary for the package to
    function.
    """
    from tengri._data_setup import data_dirs, package_data_dirs

    # Set the env var to disable ancestor walk
    monkeypatch.setenv("TENGRI_DATA_NO_ANCESTOR_WALK", "1")

    # Change to a temporary directory far from the package
    monkeypatch.chdir(tmp_path)

    result = data_dirs()

    # The package data directories should still be present
    pkg_dirs = package_data_dirs()
    for pkg_dir in pkg_dirs:
        assert pkg_dir in result, (
            f"Package data directory {pkg_dir} should always be in data_dirs(), "
            f"even with ancestor walk disabled. Got: {result}"
        )
