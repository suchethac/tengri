# SPDX-License-Identifier: BSD-3-Clause
"""``Posterior._save_diagnostics`` dropped entries it had no branch for.

It was an ``if/elif`` chain with no ``else``, and its nested-dict branch handled
only int/float/str. Anything unmatched vanished and ``save()`` returned
normally. Measured against the pre-fix code with::

    {
        "a_float": 1.5,
        "a_str": "ok",
        "a_none": None,
        "a_list_of_str": ["x", "y"],
        "nested": {"inner_float": 2.0, "inner_none": None, "inner_list": [1, 2, 3]},
    }

three of the eight keys were gone from the file::

    top-level dropped : ['a_none']
    nested dropped    : ['inner_list', 'inner_none']

Two distinct causes. ``None`` matched no branch at either level. A list matched
a branch at the top level but not when nested — so the same value saved or
vanished depending on its depth.

This is the sibling of the ``FitResult.save`` bug fixed in #1598, and it is
fixed the same way with one deliberate difference: it **warns** instead of
raising. Diagnostics are metadata about a fit, so losing one should be visible,
but must never cost the samples the fit exists to produce.

The reader was half the bug. It descended one level and read only ``attrs``,
so writing nested arrays without fixing it would have stored values nothing
could read back. Both sides are now recursive, which is what the round-trip
tests here actually pin.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def tmp_h5():
    return Path(tempfile.mkdtemp()) / "diag.h5"


def _write(path, diagnostics):
    with h5py.File(path, "w") as f:
        Posterior._save_diagnostics(f, diagnostics)


def _read(path):
    with h5py.File(path, "r") as f:
        return Posterior._load_diagnostics(f)


# ── the three keys that used to vanish ──────────────────────────────────


def test_a_nested_list_survives_the_round_trip(tmp_h5):
    """Depth must not decide whether a value is saved."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write(tmp_h5, {"nested": {"inner_list": [1, 2, 3]}})
        out = _read(tmp_h5)
    np.testing.assert_allclose(out["nested"]["inner_list"], [1, 2, 3])


def test_a_top_level_list_still_survives(tmp_h5):
    """The direction that already worked, pinned so the recursion did not break it."""
    out_path = tmp_h5
    _write(out_path, {"a_list": [4, 5, 6]})
    out = _read(out_path)
    np.testing.assert_allclose(out["a_list"], [4, 5, 6])


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"a_none": None},
        {"nested": {"inner_none": None}},
    ],
    ids=["top-level", "nested"],
)
def test_an_unwritable_entry_warns_instead_of_vanishing(tmp_h5, diagnostics):
    with pytest.warns(UserWarning, match="could not be written"):
        _write(tmp_h5, diagnostics)


def test_the_dropped_name_is_recorded_in_the_file(tmp_h5):
    """Recoverable after the fact, and dotted so the depth is unambiguous."""
    with pytest.warns(UserWarning):
        _write(tmp_h5, {"nested": {"inner_none": None}})
    with h5py.File(tmp_h5, "r") as f:
        recorded = f["diagnostics"].attrs["skipped_keys"]
    assert "nested.inner_none" in recorded


def test_loading_a_file_with_dropped_keys_warns(tmp_h5):
    """The save-time warning is heard once, by whoever ran the fit."""
    with pytest.warns(UserWarning):
        _write(tmp_h5, {"kept": 1.0, "a_none": None})
    with pytest.warns(UserWarning, match="dropped when"):
        out = _read(tmp_h5)
    assert out["kept"] == 1.0, "the writable entry must still load"


# ── the writable entries must not be collateral damage ──────────────────


def test_one_unwritable_entry_does_not_cost_the_others(tmp_h5):
    """The property that stops the fix being worse than the bug."""
    with pytest.warns(UserWarning):
        _write(
            tmp_h5,
            {
                "a_float": 1.5,
                "a_str": "ok",
                "a_none": None,
                "a_list_of_str": ["x", "y"],
                "nested": {"inner_float": 2.0, "inner_none": None, "inner_list": [1, 2, 3]},
            },
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = _read(tmp_h5)

    assert out["a_float"] == 1.5
    assert out["a_str"] == "ok"
    assert out["nested"]["inner_float"] == 2.0
    np.testing.assert_allclose(out["nested"]["inner_list"], [1, 2, 3])
    assert "a_none" not in out
    assert "inner_none" not in out["nested"]


def test_a_fully_writable_diagnostics_dict_is_silent(tmp_h5):
    """The direction that fails if the warning became eager."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _write(tmp_h5, {"chi2_dof": 1.02, "method": "map", "nested": {"steps": 100.0}})
        out = _read(tmp_h5)
    assert out["chi2_dof"] == pytest.approx(1.02)
    assert out["nested"]["steps"] == pytest.approx(100.0)


def test_skipped_keys_is_not_surfaced_as_a_diagnostic(tmp_h5):
    """Bookkeeping must not masquerade as a fit diagnostic."""
    with pytest.warns(UserWarning):
        _write(tmp_h5, {"a_none": None})
    with pytest.warns(UserWarning):
        out = _read(tmp_h5)
    assert "skipped_keys" not in out


def test_three_level_nesting_round_trips(tmp_h5):
    """The reader recursed one level before; depth is now unbounded."""
    _write(tmp_h5, {"l1": {"l2": {"l3_val": 7.0, "l3_arr": np.arange(3.0)}}})
    out = _read(tmp_h5)
    assert out["l1"]["l2"]["l3_val"] == pytest.approx(7.0)
    np.testing.assert_allclose(out["l1"]["l2"]["l3_arr"], np.arange(3.0))
