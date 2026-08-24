# SPDX-License-Identifier: BSD-3-Clause
"""Wavelength grid construction and interpolation utilities.

Provides functions for building panchromatic wavelength grids that extend
the SSP grid to X-ray and radio wavelengths, and for interpolating SEDs
between grids.

All functions are pure JAX, JIT-compatible.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.utils.grid_interp import resample_template

# Wavelength ranges (Angstrom)
# 0.0413 Angstrom = hc / (300 keV) with hc = 12.398 keV.Angstrom: the hard edge
# is set at 300 keV so the Yang+2020 / X-CIGALE corona exponential cutoff
# (default E_cut = 300 keV) is sampled and visible, rather than clipped at the
# old ~120 keV grid edge where the rollover has barely begun.
XRAY_WAVE_MIN: float = 0.0413  # ~300 keV hard X-ray (matches corona E_cut)
XRAY_WAVE_MAX: float = 100.0  # ~0.12 keV soft X-ray
RADIO_WAVE_MIN: float = 1e5  # 10 μm: overlap with SSP IR tail
RADIO_WAVE_MAX: float = 3e11  # ~1 MHz radio


def make_union_grid(
    *arrays: np.ndarray | jnp.ndarray,
    dedupe_tol_rel: float = 1e-9,
) -> jnp.ndarray:
    """Sorted, deduplicated union of several wavelength grids in Angstrom.

    Each component's native template grid (dust emission, AGN torus, etc.) is
    declared in :mod:`tengri.forward.wavelength_extension`; this helper unions
    them with the SSP grid into the master rest-frame grid that
    ``SEDModel._init_multiwavelength`` exposes as ``state.wave``. The result
    is static, sorted ascending, and contains every input node up to a
    relative floating-point tolerance.

    Parameters
    ----------
    *arrays: array_like
        Wavelength grids in Angstrom. Empty / ``None`` arrays are ignored.
    dedupe_tol_rel: float
        Relative tolerance for collapsing near-coincident points (defaults
        to 1e-9, i.e. one part in a billion). Two values are considered
        equal when their relative gap falls below this tolerance.

    Returns
    -------
    jnp.ndarray
        Sorted, unique wavelength grid in Angstrom.

    Notes
    -----
    The dedupe step is the only knob that affects JIT shape: identical inputs
    always produce identical-shape output, but adding a new component grid
    changes the grid size and forces one re-compile. This is acceptable
    because the union is computed at ``SEDModel.build`` time, not per-call.
    """
    clean: list[np.ndarray] = []
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a, dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size > 0:
            clean.append(arr)

    if not clean:
        return jnp.asarray(np.empty(0, dtype=np.float64))

    merged = np.sort(np.concatenate(clean))
    if dedupe_tol_rel > 0.0 and merged.size > 1:
        # Relative-gap dedupe: keep a point only if its relative jump from
        # the previous kept point exceeds the tolerance.
        keep = np.empty(merged.shape, dtype=bool)
        keep[0] = True
        # Use the larger of the two endpoints for the relative scale so the
        # check is symmetric.
        rel_gap = np.diff(merged) / np.maximum(merged[:-1], 1e-300)
        keep[1:] = rel_gap > dedupe_tol_rel
        merged = merged[keep]
    return jnp.asarray(merged)


def make_panchromatic_grid(
    ssp_wave: np.ndarray | jnp.ndarray,
    extend_xray: bool = True,
    extend_radio: bool = True,
    n_per_decade: int = 20,
) -> jnp.ndarray:
    """Extend SSP wavelength grid with X-ray and radio wings.

    Builds a panchromatic grid by concatenating log-spaced X-ray and/or
    radio wavelengths with the original SSP grid. SSP wavelength points
    are preserved exactly (no resampling) so stellar SED values at those
    points have zero interpolation error.

    Parameters
    ----------
    ssp_wave: array (n_ssp,)
        Base SSP wavelength grid in Angstrom, sorted ascending.
    extend_xray: bool
        If True, prepend log-spaced points from 0.1 Å to the SSP minimum.
    extend_radio: bool
        If True, append log-spaced points from the SSP maximum to 3×10¹¹ Å.
    n_per_decade: int
        Number of log-spaced points per decade in the X-ray and radio wings.

    Returns
    -------
    jnp.ndarray (n_total,)
        Sorted, unique wavelengths in Angstrom. If both flags are False,
        returns ``ssp_wave`` unchanged.
    """
    if not extend_xray and not extend_radio:
        return jnp.asarray(ssp_wave)

    ssp_np = np.asarray(ssp_wave)
    parts = []

    if extend_xray:
        wave_min = XRAY_WAVE_MIN  # hard X-ray edge (~300 keV)
        wave_max = ssp_np[0]  # up to first SSP point (exclusive)
        if wave_min < wave_max:
            n_decades = np.log10(wave_max) - np.log10(wave_min)
            n_pts = max(int(n_decades * n_per_decade), 2)
            xray_wing = np.logspace(np.log10(wave_min), np.log10(wave_max), n_pts, endpoint=False)
            parts.append(xray_wing)

    parts.append(ssp_np)

    if extend_radio:
        wave_min = ssp_np[-1]  # from last SSP point (exclusive)
        wave_max = RADIO_WAVE_MAX
        if wave_min < wave_max:
            n_decades = np.log10(wave_max) - np.log10(wave_min)
            n_pts = max(int(n_decades * n_per_decade), 2)
            radio_wing = np.logspace(np.log10(wave_min), np.log10(wave_max), n_pts, endpoint=True)[
                1:
            ]  # skip first point (== ssp_wave[-1])
            parts.append(radio_wing)

    grid = np.concatenate(parts)
    # Ensure sorted and unique (should already be, but defensive)
    grid = np.unique(grid)
    return jnp.asarray(grid)


def interpolate_sed_to_grid(
    wave_src: jnp.ndarray,
    sed_src: jnp.ndarray,
    wave_target: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate SED to a new wavelength grid in log-log space.

    Uses log-log interpolation which is natural for power-law spectra
    (radio synchrotron, X-ray power laws). Values outside the source
    wavelength range are set to zero (no extrapolation).

    Parameters
    ----------
    wave_src: array (n_src,)
        Source wavelengths (Angstrom), sorted ascending.
    sed_src: array (n_src,)
        SED on source grid (erg/s/Hz or Lsun/Hz).
    wave_target: array (n_tgt,)
        Target wavelengths (Angstrom), sorted ascending.

    Returns
    -------
    array (n_tgt,)
        Interpolated SED. Zero outside source range.
    """
    # Single-sourced on ``resample_template`` so there is one log-log resampler
    # in the codebase, not two. That helper falls back to linear-in-flux on any
    # interval with a non-positive endpoint, which is better than this function's
    # previous ``maximum(sed, 1e-300)`` floor: flooring makes the interval
    # between a zero and a real value a near-vertical geometric ramp, whereas
    # the fallback interpolates it sensibly.
    return resample_template(wave_target, wave_src, sed_src, left=0.0, right=0.0)
