# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``scripts/_grid_native_sampling`` (task3 fix round 1, RULING R14).

``native_wavelength_grid``/``place_on_grid`` used to be copy-pasted into three
builder scripts (``build_schreiber2018_grid.py``, ``build_dh02_ce01_grid.py``,
``build_agnfitter_bbb_reference.py``), and one of the three copies had already
drifted: its multi-grid branch returned the raw union with no common-range
restriction, and its ``place_on_grid`` clamped silently via plain
``np.interp`` instead of raising. This file tests the single shared module all
three now import, with synthetic grids that exercise the branches a real
upstream library only exercises by chance (a single shared native grid, in
the case of schreiber2018) or in a way that is easy to get subtly wrong (the
common-range restriction, in the case of dh02_ce01).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _grid_native_sampling import (
    dedupe_last_write_wins,
    native_wavelength_grid,
    place_on_grid,
)

pytestmark = pytest.mark.contract


def test_native_wavelength_grid_verbatim_when_all_rows_share_one_grid() -> None:
    """A single shared native grid is returned verbatim, unmodified."""
    shared = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    rows = [shared, shared.copy(), shared[::-1].copy()]  # order must not matter
    grid, mode = native_wavelength_grid(rows)
    assert mode == "verbatim"
    np.testing.assert_array_equal(grid, shared)


def test_native_wavelength_grid_union_restricted_to_common_range() -> None:
    """Two divergent native grids: union of points within their common range only.

    Regression test for the drift RULING R14 found: an earlier
    ``build_schreiber2018_grid.py`` copy of this function returned the raw
    union with NO common-range restriction, so a row on the narrower grid
    would later need ``place_on_grid`` to extrapolate -- silently clamped by
    plain ``np.interp`` before this fix (see the raise test below).
    """
    coarse = np.array([2.0, 4.0, 6.0, 8.0])  # covers [2, 8]
    fine = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # covers [1, 9], wider
    grid, mode = native_wavelength_grid([coarse, fine])
    assert mode == "union(common-range)"
    # Restricted to the coarse grid's range [2, 8] -- fine's 1.0 and 9.0 are
    # OUTSIDE that common range and must not appear.
    assert grid.min() == 2.0
    assert grid.max() == 8.0
    assert 1.0 not in grid
    assert 9.0 not in grid
    # Every point of the coarse grid that lies in [2, 8] (all of it) is
    # preserved exactly; likewise every in-range fine point.
    for v in coarse:
        assert v in grid
    for v in fine:
        if 2.0 <= v <= 8.0:
            assert v in grid


def test_native_wavelength_grid_no_common_range_raises() -> None:
    """Two native grids that do not overlap at all: no single axis is possible."""
    low = np.array([1.0, 2.0, 3.0])
    high = np.array([10.0, 11.0, 12.0])
    with pytest.raises(RuntimeError, match="no common"):
        native_wavelength_grid([low, high])


def test_place_on_grid_verbatim_returns_native_values_unchanged() -> None:
    """When the row's own (sorted) grid equals the output grid, no interpolation runs."""
    wave = np.array([3.0, 1.0, 2.0])  # deliberately unsorted
    sed = np.array([30.0, 10.0, 20.0])  # matches wave's own order
    grid = np.array([1.0, 2.0, 3.0])
    out = place_on_grid(wave, sed, grid)
    np.testing.assert_array_equal(out, np.array([10.0, 20.0, 30.0]))


def test_place_on_grid_interpolates_within_common_range() -> None:
    """A row lacking some grid points gets them by linear interpolation."""
    wave = np.array([0.0, 2.0, 4.0])
    sed = np.array([0.0, 20.0, 40.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = place_on_grid(wave, sed, grid)
    # Native points (0, 2, 4) reproduced exactly; interpolated points (1, 3)
    # match the linear trend.
    np.testing.assert_allclose(out, np.array([0.0, 10.0, 20.0, 30.0, 40.0]))


def test_place_on_grid_raises_on_extrapolation() -> None:
    """A grid extending past a row's own native coverage raises, never clamps.

    Regression test for RULING R14: the drifted copy of this function used
    plain ``np.interp`` with no bounds check, which silently clamps a
    query point outside the source data to the nearest edge value instead
    of signaling that the row cannot actually populate that point.
    """
    wave = np.array([1.0, 2.0, 3.0])
    sed = np.array([10.0, 20.0, 30.0])
    grid_too_wide = np.array([0.5, 1.0, 2.0, 3.0, 4.0])  # 0.5 and 4.0 are out of range
    with pytest.raises(ValueError, match="extrapolate"):
        place_on_grid(wave, sed, grid_too_wide)


def test_dedupe_last_write_wins_keeps_last_raw_occurrence() -> None:
    """A repeated key keeps the LAST raw-order row, matching a dict-overwrite build.

    Regression test for task3 fix round 1, item 1: DH02_CE01.pickle has
    duplicated ``irlum`` values that AGNfitter-rX's own dict-keyed-by-str
    construction resolves by keeping the last raw row seen, not the first.
    """
    keys = np.array([5.0, 1.0, 3.0, 1.0, 3.0, 1.0])  # 1.0 at [1, 3, 5]; 3.0 at [2, 4]
    unique_keys, kept_raw_index = dedupe_last_write_wins(keys)
    np.testing.assert_array_equal(unique_keys, np.array([1.0, 3.0, 5.0]))
    assert kept_raw_index.tolist() == [5, 4, 0]  # last occurrence of each, ascending by key


def test_dedupe_last_write_wins_no_duplicates_is_identity() -> None:
    """With no repeats, every row survives, sorted ascending by key."""
    keys = np.array([3.0, 1.0, 2.0])
    unique_keys, kept_raw_index = dedupe_last_write_wins(keys)
    np.testing.assert_array_equal(unique_keys, np.array([1.0, 2.0, 3.0]))
    assert kept_raw_index.tolist() == [1, 2, 0]
