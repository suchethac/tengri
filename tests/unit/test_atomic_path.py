# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the atomic_path write helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tengri.utils.atomic_io import atomic_path


def test_success_publishes_file_at_final_path(tmp_path):
    final = tmp_path / "result.h5"
    with atomic_path(str(final)) as tmp:
        with open(tmp, "w") as f:
            f.write("payload")
        assert not final.exists()
    assert final.read_text() == "payload"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.h5"]


def test_tmp_name_keeps_suffix(tmp_path):
    final = tmp_path / "result.h5"
    with atomic_path(str(final)) as tmp:
        assert tmp.endswith(".h5")
        assert tmp != str(final)
        assert tmp.startswith(str(tmp_path))
        Path(tmp).write_bytes(b"x")


def test_exception_leaves_no_final_file_and_no_tmp(tmp_path):
    final = tmp_path / "result.pkl"
    with pytest.raises(RuntimeError, match="boom"), atomic_path(str(final)) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"truncated")
        raise RuntimeError("boom")
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []


def test_exception_keeps_previous_content(tmp_path):
    final = tmp_path / "result.pkl"
    final.write_bytes(b"old")
    with pytest.raises(RuntimeError), atomic_path(str(final)) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"new-partial")
        raise RuntimeError("crash")
    assert final.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.pkl"]


def test_h5py_file_round_trips_through_helper(tmp_path):
    h5py = pytest.importorskip("h5py")
    final = tmp_path / "posterior.h5"
    with atomic_path(str(final)) as tmp, h5py.File(tmp, "w") as f:
        f.attrs["method"] = "nuts"
        f.create_dataset("params", data=[1.0, 2.0])
    with h5py.File(final, "r") as f:
        assert f.attrs["method"] == "nuts"
        assert list(f["params"][:]) == [1.0, 2.0]
