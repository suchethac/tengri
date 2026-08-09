# SPDX-License-Identifier: BSD-3-Clause
"""``FitResult.save`` must not silently drop samples it could not write.

The whole sample-serialization block sat under
``contextlib.suppress(Exception)``, so the first unwritable value abandoned
serialization of every entry after it while ``save()`` returned normally.
Measured against the pre-fix code, with ``samples = {"good": ndarray, "bad":
object()}``:

    save() returned normally (no error raised)
      samples group present : True
      datasets written      : ['good']

A **partial** silent write. ``good`` was saved, ``bad` vanished, and nothing
said so. (An earlier probe of mine reported the group as absent entirely — that
was a mistake: it read ``f["samples"]`` when the group lives at
``f["tengri_fitresult"]["samples"]``. The bug is real but narrower than that
first reading suggested, and dict ordering decides how much is lost: an
unwritable first key takes everything after it.)

Still worth fixing, because the loss is invisible in the artifact. A saved fit
missing entries looks exactly like a complete one until someone loads it,
possibly long after the run that produced it is gone. Among the ten blanket
suppressors this is the only one that loses *data* rather than hiding a
diagnostic.

Two properties are pinned below, and the second matters as much as the first:
the caller is told, **and** the writable entries still get written. Trading
silent partial loss for loud total loss would be a worse fix than the bug.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pytest

from tengri.results import FitRecord, FitResult, ResultSerializationError

pytestmark = pytest.mark.regression_bug


def _record() -> FitRecord:
    return FitRecord(
        tengri_version="0",
        python_version="3.12",
        platform="test",
        jax_version=None,
        jax_backend=None,
        timestamp_utc="1970-01-01T00:00:00Z",
    )


class _PartlyWritable:
    """An inner result whose samples h5py can only partly write.

    A bare ``object()`` stands in for anything h5py has no dtype for — the
    realistic case being an inner result that carries a non-array value.
    """

    samples: ClassVar[dict] = {"good": np.arange(5.0), "bad": object()}


class _FullyWritable:
    samples: ClassVar[dict] = {"a": np.arange(3.0), "b": np.linspace(0.0, 1.0, 4)}


@pytest.fixture
def tmp_h5():
    return Path(tempfile.mkdtemp()) / "fit.h5"


# ── the caller is told ───────────────────────────────────────────────────


def test_an_unwritable_entry_raises_instead_of_passing_quietly(tmp_h5):
    res = FitResult(inner=_PartlyWritable(), record=_record(), citation_keys=["a"])
    with pytest.raises(ResultSerializationError, match="could not be written"):
        res.save(str(tmp_h5))


def test_the_error_names_the_entry_that_was_lost(tmp_h5):
    """An error that does not say *what* is missing just moves the guesswork."""
    res = FitResult(inner=_PartlyWritable(), record=_record(), citation_keys=["a"])
    with pytest.raises(ResultSerializationError) as exc:
        res.save(str(tmp_h5))
    assert "bad" in str(exc.value)


# ── and the writable entries survive ────────────────────────────────────


def test_the_writable_entries_are_still_written(tmp_h5):
    """The property that stops this fix being worse than the bug.

    Before, one bad entry cost the whole group. Raising *and* dropping
    everything would trade silent partial loss for loud total loss.
    """
    res = FitResult(inner=_PartlyWritable(), record=_record(), citation_keys=["a"])
    with pytest.raises(ResultSerializationError):
        res.save(str(tmp_h5))

    assert tmp_h5.exists(), "the file should still exist so partial data is recoverable"
    with h5py.File(tmp_h5, "r") as f:
        root = f["tengri_fitresult"]
        assert "samples" in root, "the samples group must be present even on partial failure"
        written = set(root["samples"].keys())
    assert "good" in written, f"the writable entry was lost too; wrote {written}"


def test_the_dropped_names_are_recorded_in_the_file(tmp_h5):
    """Recoverable after the fact, not only in the raised message."""
    res = FitResult(inner=_PartlyWritable(), record=_record(), citation_keys=["a"])
    with pytest.raises(ResultSerializationError):
        res.save(str(tmp_h5))
    with h5py.File(tmp_h5, "r") as f:
        assert "skipped_keys" in f["tengri_fitresult"]["samples"].attrs
        assert "bad" in f["tengri_fitresult"]["samples"].attrs["skipped_keys"]


# ── and loading an incomplete file says so ──────────────────────────────


def test_loading_an_incomplete_file_warns(tmp_h5):
    """The raise at save time is heard once, by whoever ran the fit.

    Whoever loads the file later — another machine, another month — sees a
    samples dict that looks complete. Recording the dropped names without
    reading them back would just move the silence downstream.
    """
    res = FitResult(inner=_PartlyWritable(), record=_record(), citation_keys=["a"])
    with pytest.raises(ResultSerializationError):
        res.save(str(tmp_h5))

    with pytest.warns(UserWarning, match="incomplete save"):
        loaded = FitResult.load(str(tmp_h5))
    assert "bad" not in loaded.inner["samples"], "the entry really is absent"
    assert "good" in loaded.inner["samples"], "and the rest really is there"


def test_loading_a_complete_file_is_silent(tmp_h5):
    """The direction that fails if the warning fired on every load."""
    res = FitResult(inner=_FullyWritable(), record=_record(), citation_keys=["a"])
    res.save(str(tmp_h5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loaded = FitResult.load(str(tmp_h5))
    assert set(loaded.inner["samples"]) == {"a", "b"}


# ── the happy path is untouched ─────────────────────────────────────────


def test_a_fully_writable_result_saves_without_raising(tmp_h5):
    """The direction that fails if the guard became too eager."""
    res = FitResult(inner=_FullyWritable(), record=_record(), citation_keys=["a"])
    res.save(str(tmp_h5))
    with h5py.File(tmp_h5, "r") as f:
        assert set(f["tengri_fitresult"]["samples"].keys()) == {"a", "b"}
        assert "skipped_keys" not in f["tengri_fitresult"]["samples"].attrs
        np.testing.assert_allclose(f["tengri_fitresult"]["samples"]["a"][:], np.arange(3.0))


def test_a_result_with_no_samples_at_all_still_saves(tmp_h5):
    """An inner result exposing neither `samples` nor `to_dict` is legitimate.

    It must not be mistaken for a serialization failure — nothing was lost,
    because there was nothing to write.
    """

    class _Bare:
        pass

    res = FitResult(inner=_Bare(), record=_record(), citation_keys=["a"])
    res.save(str(tmp_h5))
    with h5py.File(tmp_h5, "r") as f:
        assert "samples" in f["tengri_fitresult"]
        assert list(f["tengri_fitresult"]["samples"].keys()) == []
