"""Phenomenological metallicity history models (Bagpipes-compatible).

Provides time-varying metallicity Z(t) functions modelled after the
``chemical_enrichment_history`` module in Bagpipes (Carnall et al. 2018).
Each function returns a ``log10(Z)`` *absolute* array on the SSP age grid,
ready for ``interp_metallicity_evolving`` or DSPS ``compute_dsps_met_table_weights``.

Unlike the gas-regulator model in ``chemical_evolution.py`` (which derives
Z(t) self-consistently from the SFH), these modes let users prescribe Z(t)
directly — useful when physical self-consistency is not required or when
comparing to Bagpipes results.

Modes
-----
- ``two_step``:  Abrupt metallicity change at a lookback time.
- ``psb_two_step``:  Step at the PSB burst age (Leung et al. 2024).
- ``metallicity_bins``:  Per-bin metallicities aligned with continuity SFH bins.
- ``metallicity_bins_continuity``:  Delta-log-Z steps (Dirichlet-coupled Z evolution).

All metallicity inputs/outputs are in **log10(Z) absolute** (not Z/Zsun).
The solar offset ``LOG10_ZSUN`` is applied in the parameter translation layer
(``translate.py``), so these functions receive already-converted values.

References
----------
- Carnall et al. 2018 (MNRAS 480, 4379): Bagpipes
- Leung et al. 2024 (MNRAS 528, 4029): psb_two_step metallicity model
- Leja et al. 2019 (ApJ 876, 3): Continuity SFH / metallicity bins
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def two_step_metallicity(
    ssp_lg_age_gyr: jnp.ndarray,
    log_z_abs_old: float,
    log_z_abs_young: float,
    step_age_gyr: float,
) -> jnp.ndarray:
    """Step-function metallicity history with smooth sigmoid transition.

    Assigns one metallicity to old stars and another to young stars,
    with a smoothed sigmoid transition for JAX differentiability.

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        Log10(age/Gyr) of SSP templates. Lookback time (younger ages first).
    log_z_abs_old : float
        log10(Z) absolute for old stars [dimensionless].
    log_z_abs_young : float
        log10(Z) absolute for young stars [dimensionless].
    step_age_gyr : float
        Lookback time of the metallicity step [Gyr].

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.nn.sigmoid``.

    **Gradient-safe**: yes — sigmoid provides smooth gradients.

    The metallicity history is:

    .. math::

        Z(t) = Z_{\\rm young} + (Z_{\\rm old} - Z_{\\rm young}) \\sigma(x)

    where :math:`\\sigma(x)` is the sigmoid function:

    .. math::

        \\sigma(x) = \\frac{1}{1 + \\exp(-x/w)}

    and :math:`x = \\log_{10}(t_{\\rm age}) - \\log_{10}(t_{\\rm step})`,
    :math:`t_{\\rm age}` is the stellar age [Gyr], :math:`t_{\\rm step}` is
    the step age [Gyr], and :math:`w = 0.02` (width in log-space, ~2%).

    The smooth sigmoid avoids discontinuities that would complicate gradients
    while keeping the transition sharp enough to be physically meaningful.
    """
    log_step = jnp.log10(jnp.maximum(step_age_gyr, 1e-4))
    width = 0.02
    sigmoid = jax.nn.sigmoid((ssp_lg_age_gyr - log_step) / width)
    return log_z_abs_young + (log_z_abs_old - log_z_abs_young) * sigmoid


@jax.jit
def psb_two_step_metallicity(
    ssp_lg_age_gyr: jnp.ndarray,
    log_z_abs_old: float,
    log_z_abs_burst: float,
    burstage_gyr: float,
) -> jnp.ndarray:
    """PSB two-step metallicity: step at the post-starburst burst age.

    Identical to :func:`two_step_metallicity` but semantically tied to
    the PSB SFH model. Stars older than ``burstage`` get ``log_z_abs_old``;
    burst/younger stars get ``log_z_abs_burst`` (Leung et al. 2024).

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        Log10(age/Gyr) of SSP templates [dimensionless].
    log_z_abs_old : float
        log10(Z) absolute for pre-burst stars [dimensionless].
    log_z_abs_burst : float
        log10(Z) absolute for burst stars [dimensionless].
    burstage_gyr : float
        Lookback time of the burst [Gyr].

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — delegates to :func:`two_step_metallicity`.
    """
    return two_step_metallicity(ssp_lg_age_gyr, log_z_abs_old, log_z_abs_burst, burstage_gyr)


@jax.jit
def metallicity_bins_on_ssp_grid(
    ssp_lg_age_gyr: jnp.ndarray,
    bin_edges_log_yr: jnp.ndarray,
    metallicities_abs: jnp.ndarray,
) -> jnp.ndarray:
    """Piecewise-constant metallicity history from time bins.

    Assigns a constant metallicity to each age bin. Designed to pair with
    the continuity SFH model (shared bin edges). SSP ages are mapped to
    bins using lookback-time indexing.

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        Log10(age/Gyr) of SSP templates [dimensionless].
    bin_edges_log_yr : array_like, shape (n_bins+1,)
        Bin edges in log10(age/yr), sorted ascending [dimensionless].
    metallicities_abs : array_like, shape (n_bins,)
        log10(Z) absolute per bin [dimensionless]. Index convention:
        metallicities_abs[0] = youngest bin, metallicities_abs[-1] = oldest bin.

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.searchsorted`` and ``jnp.clip``.

    Bin indexing follows lookback-time convention:

    - ``bin_edges_log_yr`` must be sorted ascending (youngest edge first).
    - ``metallicities_abs[0]`` corresponds to the youngest bin (smallest lookback time).
    - ``metallicities_abs[-1]`` corresponds to the oldest bin (largest lookback time).

    SSP ages outside the bin range are clamped to the nearest edge bin metallicity.

    The metallicity is a step function: :math:`Z(t) = Z_j` for
    :math:`t_j \\leq t < t_{j+1}`, where j is determined by binary search
    on the bin edges.
    """
    ssp_log_yr = ssp_lg_age_gyr + 9.0
    n_bins = metallicities_abs.shape[0]
    bin_idx = jnp.searchsorted(bin_edges_log_yr, ssp_log_yr, side="right") - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    return metallicities_abs[bin_idx]


@jax.jit
def metallicity_bins_continuity_on_ssp_grid(
    ssp_lg_age_gyr: jnp.ndarray,
    bin_edges_log_yr: jnp.ndarray,
    log_z_abs_base: float,
    d_log_z: jnp.ndarray,
) -> jnp.ndarray:
    """Continuity-style Z(t): base metallicity + cumulative delta-log-Z steps.

    Analogous to the continuity SFH, where adjacent bins are coupled
    via delta-log-SFR ratios.  Here, metallicity evolves via delta-log-Z
    steps from the oldest bin toward the youngest::

        Z_oldest     = log_z_abs_base
        Z_{oldest-1} = Z_oldest + d_log_z[0]
        Z_{oldest-2} = Z_{oldest-1} + d_log_z[1]
        ...
        Z_youngest   = Z_oldest + sum(d_log_z)

    Positive ``d_log_z`` values represent enrichment (increasing Z
    toward the present), which is the physically expected direction.

    Parameters
    ----------
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age/Gyr) of SSP templates.
    bin_edges_log_yr : array, shape (n_bins + 1,)
        Bin edges in log10(age/yr), sorted ascending.
    log_z_abs_base : float
        log10(Z) absolute of the oldest bin.
    d_log_z : array, shape (n_bins - 1,)
        Delta-log-Z steps from old to young.  ``d_log_z[0]`` is the
        step from the oldest bin to the second-oldest, etc.

    Returns
    -------
    array, shape (n_age,)
        log10(Z) absolute at each SSP age.
    """
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_log_z)])
    # cumulative[0] = 0 (oldest), cumulative[-1] = sum(d_log_z) (youngest)
    # Reverse so index 0 = youngest bin, index -1 = oldest bin
    # (matching metallicity_bins_on_ssp_grid convention)
    metallicities_abs = log_z_abs_base + cumulative[::-1]
    return metallicity_bins_on_ssp_grid(ssp_lg_age_gyr, bin_edges_log_yr, metallicities_abs)


@jax.jit
def tabulated_metallicity_on_ssp_grid(
    ssp_lg_age_gyr: jnp.ndarray,
    met_log_age_yr: jnp.ndarray,
    met_log_z_abs: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a user-provided Z(t) table onto the SSP age grid.

    For users who have a pre-computed metallicity history (e.g., from
    a hydrodynamical simulation or an external chemical evolution code),
    this function linearly interpolates that history onto the SSP grid.

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        Log10(age/Gyr) of SSP templates [dimensionless].
    met_log_age_yr : array_like, shape (n_table,)
        Log10(age/yr) of the tabulated metallicity history [dimensionless],
        sorted ascending (youngest first).
    met_log_z_abs : array_like, shape (n_table,)
        log10(Z) absolute at each table age [dimensionless].

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age [dimensionless], linearly interpolated
        and clamped to table edge values outside the table range.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.interp`` for linear interpolation.
    """
    ssp_log_yr = ssp_lg_age_gyr + 9.0
    return jnp.interp(ssp_log_yr, met_log_age_yr, met_log_z_abs)
