# SPDX-License-Identifier: BSD-3-Clause
"""Phenomenological metallicity history models (Bagpipes-compatible).

Provides time-varying metallicity Z(t) functions modeled after the
``chemical_enrichment_history`` module in Bagpipes (Carnall et al. 2018).
Each function returns a ``log10(Z)`` *absolute* array on the SSP age grid,
ready for ``interp_metallicity_evolving`` or DSPS ``compute_dsps_met_table_weights``.

Unlike the gas-regulator model in ``chemical_evolution.py`` (which derives
Z(t) self-consistently from the SFH), these modes let users prescribe Z(t)
directly: useful when physical self-consistency is not required or when
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
    **JIT-compatible**: yes, uses ``jax.nn.sigmoid``.

    **Gradient-safe**: yes, sigmoid provides smooth gradients.

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
    **JIT-compatible**: yes, delegates to :func:`two_step_metallicity`.
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
    **JIT-compatible**: yes, uses ``jnp.searchsorted`` and ``jnp.clip``.

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

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives and delegates to
    :func:`metallicity_bins_on_ssp_grid` for final binning.
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
    **JIT-compatible**: yes, uses ``jnp.interp`` for linear interpolation.
    """
    ssp_log_yr = ssp_lg_age_gyr + 9.0
    return jnp.interp(ssp_log_yr, met_log_age_yr, met_log_z_abs)


@jax.jit
def massmap_lin_metallicity(
    ssp_lg_age_gyr: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    sfr_on_ssp: jnp.ndarray,
    log_z_abs_start: float,
    log_z_abs_final: float,
) -> jnp.ndarray:
    """Linear metallicity mapping tied to cumulative stellar mass formed.

    Implements the ProSpect (Bellstedt+2020) massmap_lin model: metallicity
    evolves linearly between Zstart (oldest stars) and Zfinal (present day),
    with the Z(age) ramp driven by the cumulative stellar mass formed
    over time.

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        Log10(age/Gyr) of SSP templates [dimensionless].
    ssp_ages_yr : array_like, shape (n_age,)
        Age of each SSP template in years [yr].
    sfr_on_ssp : array_like, shape (n_age,)
        Star formation rate at each SSP age [Msun/yr].
    log_z_abs_start : float
        log10(Z) absolute at the oldest age (Zstart) [dimensionless].
    log_z_abs_final : float
        log10(Z) absolute at the present day (Zfinal) [dimensionless].

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses JAX primitives for cumulative integration.

    **Gradient-safe**: yes, all operations are differentiable.

    **Physics:**

    The cumulative mass fraction formed (cmf) is defined as:

    .. math::

        \\text{cmf}(t) = \\frac{\\int_0^t \\text{SFR}(t') dt'}{\\int_0^T \\text{SFR}(t') dt'}

    where :math:`t` is lookback time. Metallicity is then linearly mapped:

    .. math::

        Z(t) = Z_{\\rm start} + (Z_{\\rm final} - Z_{\\rm start}) \\cdot \\text{cmf}(t)

    This ties metallicity directly to the mass assembly history, with cmf→0
    at the oldest age (Z→Zstart) and cmf→1 at present (Z→Zfinal).

    References
    ----------
    .. [1] Bellstedt, S., Forbes, D. A., Robotham, A. S. G., et al.
           2020, MNRAS 498, 5581.
           "Galaxy And Mass Assembly (GAMA): the stellar mass content
           of galaxy groups."
    """
    # Input convention: ssp_ages_yr and sfr_on_ssp are in ascending lookback order
    # (youngest/present-day first, oldest last).
    #
    # ProSpect's cmf (cumulative mass fraction) is defined as:
    #   cmf(age) = 1 - (mass formed AFTER this age) / total_mass
    # At present (age=0): cmf = 1 (all mass has been formed in the past)
    # At oldest age: cmf → 0 (almost no mass formed after the oldest time)
    #
    # We compute cumulative sum from present (youngest) backwards to oldest,
    # then convert to cmf.

    # Keep in ascending lookback order (youngest first) for integration
    # dt[i] = time span from ssp_ages_yr[i] to ssp_ages_yr[i+1]
    dt = jnp.abs(jnp.diff(ssp_ages_yr))
    # Average SFR in each interval
    sfr_mid = 0.5 * (sfr_on_ssp[:-1] + sfr_on_ssp[1:])
    # Cumulative sum from youngest (present) backward
    # cumsum_from_youngest[0] = 0 (at present, no mass formed AFTER present)
    # cumsum_from_youngest[n] = total_mass (at oldest, all mass formed after oldest)
    mass_in_intervals = sfr_mid * dt
    cumsum_from_youngest = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass_in_intervals)])

    # Total mass formed over entire history
    total_mass = cumsum_from_youngest[-1]
    total_mass = jnp.maximum(total_mass, 1e-30)

    # CMF = 1 - (mass formed AFTER this age) / total_mass
    # At youngest (present): cmf = 1 - 0 / total_mass = 1
    # At oldest: cmf = 1 - total_mass / total_mass = 0 (approximately)
    cmf = 1.0 - cumsum_from_youngest / total_mass

    # Clamp cmf to [0, 1]
    cmf = jnp.clip(cmf, 0.0, 1.0)

    # Linear interpolation in *linear* Z (ProSpect convention, and the docstring
    # Eq. above): Z(age) = Zstart + (Zfinal - Zstart) * cmf; NOT linear in
    # log Z (that would be a geometric map, ~2x off at the half-mass point).
    z_start = 10.0**log_z_abs_start
    z_final = 10.0**log_z_abs_final
    z_lin = z_start + (z_final - z_start) * cmf
    return jnp.log10(jnp.maximum(z_lin, 1e-30))


@jax.jit
def massmap_box_metallicity(
    ssp_lg_age_gyr: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    sfr_on_ssp: jnp.ndarray,
    log_z_abs_start: float,
    log_z_abs_final: float,
    yield_rho: float = 0.03,
) -> jnp.ndarray:
    """Closed-box chemical evolution metallicity tied to cumulative stellar mass.

    Implements the ProSpect (Bellstedt+2020) massmap_box model: metallicity
    evolves via a Lynden-Bell fixed-yield closed-box model, with the Z(age)
    history driven by the cumulative stellar mass formed. This captures
    self-consistent chemical enrichment with a constant nucleosynthetic yield.

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        Log10(age/Gyr) of SSP templates [dimensionless].
    ssp_ages_yr : array_like, shape (n_age,)
        Age of each SSP template in years [yr].
    sfr_on_ssp : array_like, shape (n_age,)
        Star formation rate at each SSP age [Msun/yr].
    log_z_abs_start : float
        log10(Z) absolute at the oldest age (Zstart) [dimensionless].
    log_z_abs_final : float
        log10(Z) absolute at the present day (Zfinal) [dimensionless].
    yield_rho : float, optional
        Fixed nucleosynthetic yield parameter [dimensionless].
        Default 0.03. Controls the rate of metallicity increase per unit mass.

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses JAX primitives for cumulative integration and logarithm.

    **Gradient-safe**: yes, all operations are differentiable (uses jnp.log with safe clipping).

    **Physics:**

    The Lynden-Bell closed-box model with fixed yield ρ yields:

    .. math::

        Z(t) = Z_{\\rm start} - ρ \\ln[1 - μ_f \\cdot \\text{cmf}(t)]

    where :math:`\\text{cmf}(t)` is the cumulative mass fraction formed
    (0 at oldest age → 1 at present), and :math:`μ_f` is the final gas fraction:

    .. math::

        μ_f = 1 - \\exp\\left( -\\frac{Z_{\\rm final} - Z_{\\rm start}}{ρ} \\right)

    The formula assumes:

    - Zstart and Zfinal are in absolute log10(Z) space
    - Nucleosynthetic yield ρ is constant throughout cosmic time
    - All stellar ejecta immediately enrich the ISM (instantaneous mixing)

    In the small-enrichment limit (Zfinal − Zstart ≪ ρ), the model
    reduces to linear (massmap_lin).

    References
    ----------
    .. [1] Bellstedt, S., Forbes, D. A., Robotham, A. S. G., et al.
           2020, MNRAS 498, 5581.
           "Galaxy And Mass Assembly (GAMA): the stellar mass content
           of galaxy groups."
    .. [2] Robotham, A. S. G., & Obreschkow, D.
           2015, PASA 32, e033.
           "ProSpect: Bayesian SED fitting with application to galaxy
           clustering and SMF estimation."
    """
    # Convert absolute log10(Z) to linear Z for the box-model calculation
    z_start = 10.0**log_z_abs_start
    z_final = 10.0**log_z_abs_final
    # Final gas fraction via yield relation
    mu_final = 1.0 - jnp.exp(-(z_final - z_start) / yield_rho)

    # Compute cumulative mass using same logic as massmap_lin
    dt = jnp.abs(jnp.diff(ssp_ages_yr))
    sfr_mid = 0.5 * (sfr_on_ssp[:-1] + sfr_on_ssp[1:])
    mass_in_intervals = sfr_mid * dt
    cumsum_from_youngest = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass_in_intervals)])

    total_mass = cumsum_from_youngest[-1]
    total_mass = jnp.maximum(total_mass, 1e-30)

    cmf = 1.0 - cumsum_from_youngest / total_mass
    cmf = jnp.clip(cmf, 0.0, 1.0)

    # Apply closed-box model: Z(t) = Zstart - yield * ln(1 - mu_final * cmf)
    # Ensure 1 - mu_final * cmf > 0 to avoid log singularity
    arg = 1.0 - mu_final * cmf
    arg = jnp.clip(arg, 1e-10, 1.0)
    z_abs = z_start - yield_rho * jnp.log(arg)
    # Clamp to [Zstart, Zfinal]
    z_abs = jnp.clip(z_abs, z_start, z_final)
    # Convert back to log10(Z)
    log_z_abs = jnp.log10(z_abs)
    return log_z_abs
