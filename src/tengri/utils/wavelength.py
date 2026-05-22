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

# Wavelength ranges (Angstrom)
XRAY_WAVE_MIN: float = 0.1  # ~120 keV hard X-ray
XRAY_WAVE_MAX: float = 100.0  # ~0.12 keV soft X-ray
RADIO_WAVE_MIN: float = 1e5  # 10 μm — overlap with SSP IR tail
RADIO_WAVE_MAX: float = 3e11  # ~1 MHz radio


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
    ssp_wave : array (n_ssp,)
        Base SSP wavelength grid in Angstrom, sorted ascending.
    extend_xray : bool
        If True, prepend log-spaced points from 0.1 Å to the SSP minimum.
    extend_radio : bool
        If True, append log-spaced points from the SSP maximum to 3×10¹¹ Å.
    n_per_decade : int
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
        wave_min = max(XRAY_WAVE_MIN, 0.1)
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
    wave_src : array (n_src,)
        Source wavelengths (Angstrom), sorted ascending.
    sed_src : array (n_src,)
        SED on source grid (erg/s/Hz or Lsun/Hz).
    wave_target : array (n_tgt,)
        Target wavelengths (Angstrom), sorted ascending.

    Returns
    -------
    array (n_tgt,)
        Interpolated SED. Zero outside source range.
    """
    # Clamp to positive before log (SED can have zeros at grid edges)
    sed_safe = jnp.maximum(sed_src, 1e-300)
    log_wave_src = jnp.log10(wave_src)
    log_sed_src = jnp.log10(sed_safe)
    log_wave_tgt = jnp.log10(wave_target)

    # Interpolate in log-log space
    log_sed_tgt = jnp.interp(log_wave_tgt, log_wave_src, log_sed_src)
    sed_tgt = 10.0**log_sed_tgt

    # Zero outside source range (no extrapolation)
    outside = (wave_target < wave_src[0]) | (wave_target > wave_src[-1])
    sed_tgt = jnp.where(outside, 0.0, sed_tgt)

    return sed_tgt
