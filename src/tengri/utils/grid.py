# SPDX-License-Identifier: BSD-3-Clause
"""Log-age grid construction and conversion utilities.

The GP is defined on a uniform grid in u = log10(t_age / yr),
where t_age is the stellar population age (= lookback time).

Uniform spacing in log-age automatically gives FINER resolution
at recent times (low lookback, young ages) and coarser resolution
at early cosmic times (high lookback, old ages). This matches
the SED sensitivity: UV-emitting young populations change rapidly
on ~Myr timescales, while old populations evolve slowly on ~Gyr
timescales.

Example for 256-point grid:
  At 10 Myr (recent): ~0.4 Myr between points
  At 1 Gyr  (old):    ~38 Myr between points
"""

import jax.numpy as jnp

# Default grid: 1 Myr to ~13.8 Gyr
DEFAULT_N_GRID = 256
DEFAULT_LOG_AGE_MIN = 6.0  # log10(1 Myr / yr)
DEFAULT_LOG_AGE_MAX = 10.14  # log10(13.8 Gyr / yr)


def make_log_age_grid(
    n_grid: int = DEFAULT_N_GRID,
    log_age_min: float = DEFAULT_LOG_AGE_MIN,
    log_age_max: float = DEFAULT_LOG_AGE_MAX,
) -> jnp.ndarray:
    """Create uniform grid in log10(age/yr).

    Default range: 1 Myr to ~13.8 Gyr (approximately the age of the universe).

    Parameters
    ----------
    n_grid : int, optional
        Number of grid points (should be even for FFT efficiency). Default: 256.
    log_age_min : float, optional
        Minimum log10(age/yr) [dimensionless log years]. Default: 6.0 (1 Myr).
    log_age_max : float, optional
        Maximum log10(age/yr) [dimensionless log years]. Default: 10.14 (~13.8 Gyr).

    Returns
    -------
    ndarray, shape (n_grid,)
        Uniform grid in log10(age/yr) [dimensionless log values].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.linspace``.

    This grid is the internal representation for age in GP-based SFH models.
    The log-space parametrization provides better resolution at young ages and
    maps naturally to the logarithmic timescales of stellar evolution.

    This is the single definition. ``components.stellar.sfh.gp_sfh`` re-exports
    it under the same name; the forward model and the inference standardization
    consume it from opposite sides of that boundary, so a second copy could
    desync the two without any test failing (#1402).

    Examples
    --------
    >>> from tengri import make_log_age_grid
    >>> grid = make_log_age_grid(n_grid=64)
    >>> grid.shape
    (64,)
    >>> float(grid[0]), float(grid[-1])
    (6.0, 10.14)
    """
    return jnp.linspace(log_age_min, log_age_max, n_grid)


def log_age_to_age_yr(log_age_grid: jnp.ndarray) -> jnp.ndarray:
    """Convert log10(age/yr) grid to age in years.

    Parameters
    ----------
    log_age_grid : array_like, shape (n_grid,)
        Grid of log10(age/yr) values.

    Returns
    -------
    ndarray, shape (n_grid,)
        Age in years. [yr]
    """
    return 10.0**log_age_grid


def grid_spacing(log_age_grid: jnp.ndarray) -> float:
    """Return the uniform spacing of the log-age grid.

    Parameters
    ----------
    log_age_grid : array_like, shape (n_grid,)
        Uniform grid of log10(age/yr) values.

    Returns
    -------
    float
        Grid spacing in dex. [dex]
    """
    return float(log_age_grid[1] - log_age_grid[0])


def interpolate_to_linear_time(
    log_age_grid: jnp.ndarray, values: jnp.ndarray, n_linear: int = 1000
) -> tuple:
    """Interpolate a quantity from log-age grid to uniform linear time.

    The GP SFH is defined on a log-age grid. When plotted vs linear
    lookback time, the uneven point spacing creates visual artifacts
    (more wiggles at old ages). This function resamples to a uniform
    linear grid for cleaner plotting.

    Parameters
    ----------
    log_age_grid : array, shape (n_grid,)
        Log10(age/yr) grid.
    values : array, shape (n_grid,)
        Values on the log-age grid (e.g., SFR).
    n_linear : int
        Number of points in the output linear grid.

    Returns
    -------
    t_gyr : array, shape (n_linear,)
        Uniform lookback time grid in Gyr.
    values_linear : array, shape (n_linear,)
        Interpolated values on the linear grid.
    """
    age_yr_min = 10.0 ** float(log_age_grid[0])
    age_yr_max = 10.0 ** float(log_age_grid[-1])
    t_linear_yr = jnp.linspace(age_yr_min, age_yr_max, n_linear)
    log_t_linear = jnp.log10(t_linear_yr)
    values_linear = jnp.interp(log_t_linear, log_age_grid, values)
    return t_linear_yr / 1e9, values_linear
