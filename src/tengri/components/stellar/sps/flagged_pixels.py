# SPDX-License-Identifier: BSD-3-Clause
"""Static node windows of the pixels next to a dust-law breakpoint (#3413).

A pixel-moment spectrum table is exact except in the pixels whose rest-frame source range
contains a dust-law breakpoint: there the first-order expansion of the dust factor fails. Those
pixels are flagged at build, and each one keeps a static window of model nodes, padded to a
common length, so the per-draw path can evaluate them exactly on that window.

Everything here is build-time index arithmetic on numpy arrays. The window of a pixel is its
rest-frame source range, widened by ``pad_aa`` on both sides (the lattice window the exact path
uses, at least five sigma of the total broadening).
"""

from __future__ import annotations

import numpy as np


def flagged_pixel_mask(
    edges_rest: np.ndarray,
    breakpoints: np.ndarray,
    pad_aa: np.ndarray | float,
) -> np.ndarray:
    """Which pixels' padded source range contains a breakpoint.

    Parameters
    ----------
    edges_rest : ndarray, shape (n_pix + 1,)
        Rest-frame pixel edges ``wave_obs_edges / (1 + z)`` [Angstrom], increasing.
    breakpoints : ndarray, shape (n_bp,)
        Rest-frame dust-law breakpoints [Angstrom].
    pad_aa : float or ndarray, shape (n_pix,)
        Half-width of the padding on each side of a pixel [Angstrom].

    Returns
    -------
    ndarray of bool, shape (n_pix,)
    """
    edges = np.asarray(edges_rest, dtype=np.float64)
    bps = np.asarray(breakpoints, dtype=np.float64).reshape(-1)
    pad = np.broadcast_to(np.asarray(pad_aa, dtype=np.float64), (edges.size - 1,))
    lo = edges[:-1] - pad
    hi = edges[1:] + pad
    if bps.size == 0:
        return np.zeros(edges.size - 1, dtype=bool)
    # A breakpoint lies in [lo, hi] when the count of breakpoints <= hi exceeds the count < lo.
    below_hi = np.searchsorted(bps, hi, side="right")
    below_lo = np.searchsorted(bps, lo, side="left")
    return below_hi > below_lo


def node_window(wave_nodes: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Indices of the model nodes inside ``[lo, hi]`` [Angstrom], increasing.

    The window must cover the slice of the source that the pixel reads; the caller pads ``lo``
    and ``hi`` by the broadening's reach before calling this.
    """
    w = np.asarray(wave_nodes, dtype=np.float64)
    i0 = int(np.searchsorted(w, lo, side="left"))
    i1 = int(np.searchsorted(w, hi, side="right"))
    return np.arange(i0, i1, dtype=np.int64)


def padded_node_slices(windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad per-pixel node windows to a common length ``L``.

    Parameters
    ----------
    windows : list of ndarray
        One node-index array per flagged pixel (from :func:`node_window`).

    Returns
    -------
    idx : ndarray of int, shape (n_flag, L)
        Node indices; masked entries point at node 0 and carry zero weight.
    mask : ndarray of bool, shape (n_flag, L)
        True where the entry is a real node.
    """
    n_flag = len(windows)
    length = max((w.size for w in windows), default=0)
    idx = np.zeros((n_flag, length), dtype=np.int64)
    mask = np.zeros((n_flag, length), dtype=bool)
    for row, w in enumerate(windows):
        idx[row, : w.size] = w
        mask[row, : w.size] = True
    return idx, mask
