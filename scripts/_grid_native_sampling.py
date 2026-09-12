#!/usr/bin/env python3
"""Shared native-wavelength-grid helpers for the AGNfitter-rX template builders.

``build_schreiber2018_grid.py``, ``build_dh02_ce01_grid.py``, and
``build_agnfitter_bbb_reference.py`` each vendor an upstream library whose
templates carry their own per-row native wavelength grid (Schreiber+2018
S17, DH02_CE01) into one output wavelength axis without discarding any
native sample -- the fix for mechanism M-B (``colddust_radio.md`` D1/D2):
resampling every row onto one foreign, coarser axis smeared narrow PAH
features by up to 32% and reached 0.47 dex at grid-edge nodes.

This module used to be duplicated three times (task3 fix round 1, RULING
R14); one of the three copies (``build_schreiber2018_grid.py``'s) had
already drifted from the other two -- its multi-grid branch returned the raw
union with no common-range restriction, and its ``_place_on_grid`` clamped
silently via plain ``np.interp`` instead of raising. A single shared
implementation, imported by all three builders (matching the
``scripts/_agnfitter_download.py`` precedent -- a bare
``from _grid_native_sampling import ...`` works because Python puts a
script's own directory on ``sys.path`` when it is run directly), is the only
guarantee that a fix to one does not leave the others behind.

Also shares :func:`dedupe_last_write_wins`, the tie-break DH02_CE01's builder
and its committed crossval reference must agree on for a duplicated ``irlum``
value (task3 fix round 1, item 1): a single implementation is what makes
"both agree with each other" mean "both agree with upstream", instead of two
independently-written copies quietly agreeing on the same wrong answer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dedupe_last_write_wins", "native_wavelength_grid", "place_on_grid"]


def native_wavelength_grid(wave_rows: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """Return the axis every row should be expressed on, preserving native samples.

    Parameters
    ----------
    wave_rows : list of ndarray
        One per-row wavelength array [Å] (any order; sorted internally).

    Returns
    -------
    grid : ndarray
        Ascending wavelength axis [Å].
    mode : str
        ``"verbatim"`` when every row shares exactly one native grid (the
        returned axis IS that grid, unmodified). ``"union(common-range)"``
        when rows use several distinct native grids: the axis is the sorted
        union of every distinct native grid, restricted to the wavelength
        range every one of them covers, so no row ever needs extrapolation
        (see :func:`place_on_grid`).

    Raises
    ------
    RuntimeError
        If the distinct native grids share no common wavelength range.
    """
    unique_grids: list[np.ndarray] = []
    for w in wave_rows:
        w_sorted = np.sort(np.asarray(w, dtype=np.float64))
        is_new = not any(
            w_sorted.shape == u.shape and np.allclose(w_sorted, u, rtol=1e-10, atol=0.0)
            for u in unique_grids
        )
        if is_new:
            unique_grids.append(w_sorted)
    if len(unique_grids) == 1:
        return unique_grids[0], "verbatim"

    lo = max(g.min() for g in unique_grids)
    hi = min(g.max() for g in unique_grids)
    if lo >= hi:
        raise RuntimeError("Native wavelength grids share no common range; cannot build one axis.")
    pieces = [g[(g >= lo) & (g <= hi)] for g in unique_grids]
    return np.unique(np.concatenate(pieces)), "union(common-range)"


def place_on_grid(wave_row: np.ndarray, sed_row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Express one row's SED on ``grid`` without resampling away native points.

    Returns the row's own tabulated values unchanged when ``wave_row``
    (sorted) already equals ``grid`` (the ``"verbatim"`` case, or a row on
    the ``"union(common-range)"`` axis whose own native grid happens to
    equal it exactly). Otherwise linearly interpolates, which still
    reproduces every point ``grid`` inherited from this row's own native
    grid exactly (``np.interp`` at an exact-match query point returns the
    node value).

    Parameters
    ----------
    wave_row : ndarray
        This row's own native wavelength grid [Å] (any order).
    sed_row : ndarray
        This row's own native SED, same order as ``wave_row``.
    grid : ndarray
        The output axis from :func:`native_wavelength_grid`.

    Returns
    -------
    ndarray
        ``sed_row`` expressed on ``grid``.

    Raises
    ------
    ValueError
        If any point of ``grid`` falls outside
        ``[wave_row.min(), wave_row.max()]`` -- this row cannot populate that
        point without extrapolating, which ``np.interp`` would otherwise do
        silently (clamping to the nearest edge value, #task3 fix round 1
        R14). :func:`native_wavelength_grid`'s common-range restriction
        makes this unreachable through the normal builder flow; the raise
        is the contract, not a workaround for it.
    """
    order = np.argsort(wave_row)
    w = wave_row[order]
    s = sed_row[order]
    if w.shape == grid.shape and np.allclose(w, grid, rtol=1e-10, atol=0.0):
        return s
    if grid[0] < w[0] or grid[-1] > w[-1]:
        raise ValueError(
            f"grid [{grid[0]:.6g}, {grid[-1]:.6g}] extends outside this row's own "
            f"native coverage [{w[0]:.6g}, {w[-1]:.6g}]; would extrapolate silently."
        )
    return np.interp(grid, w, s)


def dedupe_last_write_wins(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate by key, keeping the LAST raw-order occurrence of each value.

    Reproduces a dict built by iterating rows in raw storage order and
    assigning ``dict[key] = row`` on each iteration: a repeated key is
    overwritten by every later occurrence, so the last one survives. This is
    exactly how AGNfitter-rX's own dictionaries are built (e.g.
    ``STARBURSTFdict_4plot[str(irlum)] = ...`` in
    ``MODEL_AGNfitter.py::STARBURST``, keyed by ``str(irlum)`` and built by
    iterating ``range(irlumidx)`` in raw pickle order), so it is the
    tie-break to match whenever a vendored library's rows repeat a key value.
    "First occurrence" (e.g. ``np.unique(sorted_keys, return_index=True)``)
    keeps the WRONG row whenever upstream's own dict-keyed construction would
    have kept the other one (task3 fix round 1, item 1: DH02_CE01 has three
    duplicated ``irlum`` values, each a bit-identical float64 repeat, not
    merely a close one -- so upstream's ``str()``-keyed dict genuinely
    collides on them).

    Parameters
    ----------
    keys : array_like, shape (n,)
        Key values in raw storage order (need not be sorted; may repeat).

    Returns
    -------
    unique_keys : ndarray
        Distinct key values, ascending.
    kept_raw_index : ndarray of int
        For each entry of ``unique_keys``, the raw index into ``keys`` of the
        row that survives -- the LAST raw-order occurrence of that value.
    """
    kept_by_value: dict[float, int] = {}
    for i, value in enumerate(np.asarray(keys, dtype=np.float64)):
        kept_by_value[float(value)] = i
    unique_keys = np.array(sorted(kept_by_value), dtype=np.float64)
    kept_raw_index = np.array([kept_by_value[v] for v in unique_keys], dtype=np.int64)
    return unique_keys, kept_raw_index
