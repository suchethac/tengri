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
DEFAULT_LOG_AGE_MIN = 6.0   # log10(1 Myr / yr)
DEFAULT_LOG_AGE_MAX = 10.14  # log10(13.8 Gyr / yr)


def make_log_age_grid(n_grid: int = DEFAULT_N_GRID,
                      log_age_min: float = DEFAULT_LOG_AGE_MIN,
                      log_age_max: float = DEFAULT_LOG_AGE_MAX) -> jnp.ndarray:
    """Create uniform grid in log10(age/yr).

    Parameters
    ----------
    n_grid : int
        Number of grid points. Should be even for FFT efficiency.
    log_age_min : float
        Minimum log10(age/yr).
    log_age_max : float
        Maximum log10(age/yr).

    Returns
    -------
    array, shape (n_grid,)
        Uniform grid in log10(age/yr).
    """
    return jnp.linspace(log_age_min, log_age_max, n_grid)


def log_age_to_age_yr(log_age_grid: jnp.ndarray) -> jnp.ndarray:
    """Convert log10(age/yr) grid to age in years."""
    return 10.0 ** log_age_grid


def log_age_to_age_gyr(log_age_grid: jnp.ndarray) -> jnp.ndarray:
    """Convert log10(age/yr) grid to age in Gyr."""
    return 10.0 ** (log_age_grid - 9.0)


def grid_spacing(log_age_grid: jnp.ndarray) -> float:
    """Get uniform spacing of the log-age grid (dex)."""
    return float(log_age_grid[1] - log_age_grid[0])
