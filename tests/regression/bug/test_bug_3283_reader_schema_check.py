# SPDX-License-Identifier: BSD-3-Clause
"""Readers must check the schema version and parameter names they are given (#3283).

``FitResult.save`` writes ``schema_version = "v1"`` but ``FitResult.load``
never read it, so a file from an older or foreign writer loaded as if it
matched. ``SBIPosterior.from_file`` unpickled the parameter names and never
compared them to the model the posterior was loaded for, so a posterior
trained on a different parameter set loaded silently.

Each test writes a file with the real writer, tampers with the checked
metadata, and asserts the reader refuses. A clean file must still round-trip.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pytest

from tengri.inference.backends.sbi import SBIPosterior
from tengri.results import FitRecord, FitResult

pytestmark = pytest.mark.regression_bug


class _FakePosterior:
    def sample(self, n, x=None):  # pragma: no cover - not exercised
        raise NotImplementedError

    def log_prob(self, theta, x=None):  # pragma: no cover - not exercised
        raise NotImplementedError


class _Inner:
    samples: ClassVar[dict] = {"param": np.array([1.0, 2.0, 3.0])}


def _save_result(path: Path) -> None:
    rec = FitRecord.capture(random_seed=1)
    FitResult(
        inner=_Inner(),
        record=rec,
        citation_keys=["jax"],
        backend="vi",
    ).save(str(path))


def _set_schema_version(path: Path, value: str) -> None:
    with h5py.File(path, "a") as f:
        f["tengri_fitresult"].attrs["schema_version"] = value


def _drop_schema_version(path: Path) -> None:
    with h5py.File(path, "a") as f:
        del f["tengri_fitresult"].attrs["schema_version"]


# ── results: schema_version is part of the read contract ──────────────


def test_results_clean_file_round_trips(tmp_path):
    path = tmp_path / "result.h5"
    _save_result(path)

    loaded = FitResult.load(str(path))

    assert loaded.backend == "vi"
    assert set(loaded.citation_keys) == {"jax"}
    assert "param" in loaded.inner["samples"]


@pytest.mark.parametrize("bad_version", ["v0", "v2", "1", ""])
def test_results_refuses_foreign_schema_version(tmp_path, bad_version):
    path = tmp_path / "result.h5"
    _save_result(path)
    _set_schema_version(path, bad_version)

    with pytest.raises(ValueError, match=r"result\.h5.*v1"):
        FitResult.load(str(path))


def test_results_refuses_missing_schema_version(tmp_path):
    """A file from before schema_version existed (v0) must be refused, not read."""
    path = tmp_path / "result.h5"
    _save_result(path)
    _drop_schema_version(path)

    with pytest.raises(ValueError, match=r"result\.h5.*v1"):
        FitResult.load(str(path))


# ── SBI: stored parameter names must match the model they load into ──


def test_sbi_matching_param_names_load(tmp_path):
    path = tmp_path / "posterior.pkl"
    SBIPosterior(_FakePosterior(), param_names=["a", "b"]).save(str(path))

    loaded = SBIPosterior.from_file(str(path), expected_param_names=["a", "b"])

    assert loaded.param_names == ["a", "b"]


def test_sbi_foreign_param_names_refused(tmp_path):
    path = tmp_path / "posterior.pkl"
    SBIPosterior(_FakePosterior(), param_names=["a", "b"]).save(str(path))

    with pytest.raises(ValueError, match=r"posterior\.pkl.*\['a', 'b'\].*\['a', 'c'\]"):
        SBIPosterior.from_file(str(path), expected_param_names=["a", "c"])


def test_sbi_reordered_param_names_refused(tmp_path):
    """Theta columns are positional, so the same names in another order are a mismatch."""
    path = tmp_path / "posterior.pkl"
    SBIPosterior(_FakePosterior(), param_names=["a", "b"]).save(str(path))

    with pytest.raises(ValueError, match=r"posterior\.pkl"):
        SBIPosterior.from_file(str(path), expected_param_names=["b", "a"])


def test_sbi_missing_param_names_refused_when_checked(tmp_path):
    """A file that cannot be checked is refused rather than read as if it matched."""
    path = tmp_path / "posterior.pkl"
    path.write_bytes(pickle.dumps({"posterior": _FakePosterior(), "param_names": None}))

    with pytest.raises(ValueError, match=r"posterior\.pkl"):
        SBIPosterior.from_file(str(path), expected_param_names=["a", "b"])
