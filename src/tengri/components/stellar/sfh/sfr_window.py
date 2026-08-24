# SPDX-License-Identifier: BSD-3-Clause
"""Canonical time-window-averaged SFR helper.

The forward model exposes a single quantity ``_sfr_current`` to all
SFR-driven downstream components (radio, X-ray, nebular Q_H scaling).
Historically this was set to ``sfr[-1]``: the SFR at the boundary of
whichever lookback grid happened to be in scope. That convention silently
returned an "old SFR" rather than a recent one for non-constant SFHs
because the canonical lookback grid runs ``[1 Myr -> 13.8 Gyr]`` and the
last index is the **oldest** bin, not the most recent.

This module provides the canonical replacement: a time-weighted average
of SFR over the most recent ``window_yr`` years (default 10 Myr,
matching the radio / H-alpha / X-ray timescale of Murphy et al. 2011).
"""

from __future__ import annotations

import jax.numpy as jnp


def time_weighted_sfr(
    sfr: jnp.ndarray,
    lbt_grid: jnp.ndarray,
    window_yr: float = 1e7,
) -> jnp.ndarray:
    r"""Time-weighted SFR over the most recent ``window_yr`` years.

    .. math::

        \langle\mathrm{SFR}\rangle_{T}
            = \frac{\sum_i \mathrm{SFR}_i \, \Delta t_i}
                   {\sum_i \Delta t_i}
        \quad \text{for } t_{\rm lbt,i} \leq T

    where :math:`T` = ``window_yr`` and bin widths :math:`\Delta t_i`
    are estimated by ``jnp.gradient(lbt_grid)``.

    Parameters
    ----------
    sfr : array_like, shape (n_grid,)
        SFR(t) on the lookback-time grid [Msun/yr].
    lbt_grid : array_like, shape (n_grid,)
        Lookback-time grid, **ascending** (oldest bin last) [yr].
    window_yr : float, optional
        Time window over which to average [yr]. Default ``1e7`` (10 Myr),
        matching the Murphy+2011 radio-SFR / H-alpha / X-ray timescale.

    Returns
    -------
    scalar ndarray
        Time-weighted SFR over the window [Msun/yr]. Falls back to
        ``sfr[0]`` (the most recent grid point) if no bins fall inside
        the window: e.g. a coarse grid where only the boundary bin
        qualifies.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.where`` masking on a
    fixed-shape array (no boolean indexing).

    **Physical interpretation**: 10 Myr is the canonical SFR-tracer
    timescale for radio free-free / synchrotron, X-ray HMXB, and
    H-alpha-derived SFR; 100 Myr (``window_yr=1e8``) is the standard
    Kennicutt 1998 / FUV-continuum window.

    References
    ----------
    .. [1] Murphy, E. J., et al. 2011, "Calibrating Extinction-free
       Star Formation Rate Diagnostics with 33 GHz Free-free Emission
       in NGC 6946", ApJ, 737, 67.
       https://doi.org/10.1088/0004-637X/737/2/67
       arXiv:1105.4877.
    .. [2] Kennicutt, R. C., & Evans, N. J. 2012, "Star Formation in
       the Milky Way and Nearby Galaxies", ARA&A, 50, 531.
       https://doi.org/10.1146/annurev-astro-081811-125610.
    """
    bin_widths = jnp.gradient(lbt_grid)
    in_window = lbt_grid <= window_yr
    weights = jnp.where(in_window, bin_widths, 0.0)
    weighted_sum = jnp.sum(sfr * weights)
    weight_total = jnp.sum(weights)
    return jnp.where(weight_total > 0.0, weighted_sum / weight_total, sfr[0])


__all__ = ["time_weighted_sfr"]
