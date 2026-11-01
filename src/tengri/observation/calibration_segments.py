# SPDX-License-Identifier: BSD-3-Clause
"""Camera-segment layout and fiber-fraction window for the calibration multiplier.

Build-time NumPy helpers that turn a user-facing ``calibration_segments`` value
(per-segment pixel counts or a per-pixel segment id) into one integer id per pixel,
validate the ``cal_scale_prior`` pair and the ``cal_scale_window``, and select the
pixels of the mean-multiplier window.
"""

from __future__ import annotations

import numpy as np

#: Effective wavelength of the SDSS r band [Angstrom]; the default window is the
#: camera segment that contains it.
R_BAND_EFFECTIVE_AA = 6231.0

CAL_PRIORS = ("gaussian", "scale_free")


def segment_ids(segments, n_pix: int) -> np.ndarray:
    """Return the per-pixel segment id for pixel counts or an id array.

    Parameters
    ----------
    segments : tuple of int or array_like of int, shape (n_pix,)
        A tuple or list is read as the pixel count of each segment in order; an
        ndarray is read as one id per pixel.
    n_pix : int
        Number of pixels of the grid.

    Returns
    -------
    ndarray of int, shape (n_pix,)
        Segment id of every pixel, in ``range(n_segments)``.

    Raises
    ------
    ValueError
        If a count is below 1, the counts do not sum to ``n_pix``, the id array has
        the wrong shape, or its ids are not the contiguous integers ``0 .. n_seg - 1``.

    Notes
    -----
    Build-time NumPy helper, not JIT-compatible.
    """
    if isinstance(segments, (tuple, list)):
        sizes = np.asarray(segments)
        if sizes.ndim != 1 or sizes.size == 0 or not np.issubdtype(sizes.dtype, np.integer):
            raise ValueError("calibration_segments sizes must be a non-empty tuple of integers")
        if np.any(sizes < 1):
            raise ValueError(f"calibration_segments sizes must be >= 1, got {tuple(segments)}")
        if int(sizes.sum()) != n_pix:
            raise ValueError(
                f"sum(calibration_segments)={int(sizes.sum())} does not match n_pix={n_pix}"
            )
        return np.repeat(np.arange(sizes.size), sizes)
    ids = np.asarray(segments)
    if ids.shape != (n_pix,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"calibration_segments ids must be an integer array of shape ({n_pix},), "
            f"got shape {ids.shape} dtype {ids.dtype}"
        )
    unique = np.unique(ids)
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError(
            f"calibration_segments ids must be the contiguous integers 0..n_seg-1, got {unique}"
        )
    return ids.astype(np.int64)


def check_cal_scale_prior(prior) -> tuple[float, float] | None:
    """Validate ``(log10_center, sigma_dex)`` and return it as floats, or None."""
    if prior is None:
        return None
    try:
        center, sigma = (float(v) for v in prior)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cal_scale_prior must be (log10_center, sigma_dex), got {prior!r}"
        ) from exc
    if not np.isfinite(center) or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"cal_scale_prior needs a finite center and sigma_dex > 0, got ({center}, {sigma})"
        )
    return center, sigma


def window_mask(
    wave: np.ndarray,
    ids: np.ndarray | None,
    window: tuple[float, float] | None,
) -> np.ndarray:
    """Return the 0/1 mask of the mean-multiplier window.

    Parameters
    ----------
    wave : ndarray, shape (n,)
        Observed wavelength grid [Angstrom].
    ids : ndarray of int, shape (n,), or None
        Segment id of every pixel; required when ``window`` is None.
    window : tuple of float or None
        ``(lo, hi)`` [Angstrom]. ``None`` selects every pixel of the segment that
        contains the pixel nearest to the r band's effective wavelength.

    Returns
    -------
    ndarray, shape (n,)
        1.0 inside the window, else 0.0.

    Raises
    ------
    ValueError
        If ``window`` is not ``lo < hi`` inside the grid, selects no pixel, or is
        None while ``ids`` is None.
    """
    if window is None:
        if ids is None:
            raise ValueError("the default window is a segment: pass the segment ids")
        seg = ids[np.argmin(np.abs(wave - R_BAND_EFFECTIVE_AA))]
        return (ids == seg).astype(np.float64)
    lo, hi = (float(v) for v in window)
    if not lo < hi:
        raise ValueError(f"cal_scale_window must satisfy lo < hi, got ({lo}, {hi})")
    if lo < wave.min() or hi > wave.max():
        raise ValueError(
            f"cal_scale_window ({lo}, {hi}) leaves the wavelength grid "
            f"[{wave.min():.1f}, {wave.max():.1f}] Angstrom"
        )
    mask = ((wave >= lo) & (wave <= hi)).astype(np.float64)
    if not mask.any():
        raise ValueError(f"cal_scale_window ({lo}, {hi}) contains no pixel")
    return mask
