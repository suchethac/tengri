"""Non-parametric star formation history models.

Implements the Continuity (Leja+2019) and Dirichlet (Leja+2017) non-parametric
SFH priors from the Prospector framework. Both describe piecewise-constant
SFR in N lookback-time bins, but differ in how the free parameters are
defined and what priors they imply.

- **Continuity**: free parameters are log-SFR *ratios* between adjacent bins,
  with a Student-t(df=2, scale=0.3) smoothness prior penalizing sharp jumps.
- **Dirichlet**: free parameters are auxiliary Beta(1,1) variables that map
  to mass fractions via stick-breaking, giving a symmetric Dirichlet prior.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.

References
----------
- Leja+2017 (arXiv:1609.09073): Dirichlet SFH prior.
- Leja+2019 (arXiv:1905.11997): Continuity SFH prior.
- Johnson+2021: Prospector implementation.
"""

import jax.numpy as jnp

# Default bin edges in Gyr (8 edges = 7 bins), log-spaced from 30 Myr to 13.7 Gyr.
DEFAULT_BIN_EDGES_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7])
DEFAULT_N_BINS = 7


# ---------------------------------------------------------------------------
# Continuity SFH (Leja+2019)
# ---------------------------------------------------------------------------


def continuity_sfh(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **ratio_kwargs,
) -> jnp.ndarray:
    """Non-parametric SFH with continuity prior (Leja+2019).

    The SFH is piecewise-constant in N age bins. The free parameters
    are log-ratios between adjacent bins: r_j = log10(SFR_j / SFR_{j+1}).
    A Student-t(df=2, scale=0.3) prior on r_j penalizes sharp jumps.

    The total mass normalizes the absolute SFR level.

    Parameters
    ----------
    age_yr : array (n_age,)
        Lookback time grid in years.
    log_total_mass : float
        log10(total stellar mass formed / Msun).
    bin_edges_gyr : array (n_bins+1,) or None
        Bin edges in Gyr. Default: log-spaced from 0 to 13.7 Gyr.
    **ratio_kwargs
        Keyword arguments named ``ratio_0``, ``ratio_1``, ..., ``ratio_{N-2}``
        containing the log10 SFR ratios between adjacent bins.

    Returns
    -------
    array (n_age,)
        SFR at each lookback time (Msun/yr), non-negative.

    Notes
    -----
    r_j > 0 means the younger bin (j) has higher SFR than the older bin (j+1),
    i.e. a rising SFH toward the present. r_j = 0 for all j gives a flat SFH.
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = DEFAULT_BIN_EDGES_GYR

    n_bins = bin_edges_gyr.shape[0] - 1  # len() raises ConcretizationTypeError under JIT

    # Collect ratios from kwargs in order (default 0.0 = flat SFH)
    log_sfr_ratios = jnp.array([ratio_kwargs.get(f"ratio_{i}", 0.0) for i in range(n_bins - 1)])

    # Convert ratios to absolute log-SFR.
    # Oldest bin is the reference (log_sfr = 0). Each younger bin
    # accumulates the sum of ratios from it to the oldest bin:
    #   log_SFR_j = sum(r_k for k = j..N-2)
    log_sfr = jnp.concatenate(
        [
            jnp.cumsum(log_sfr_ratios[::-1])[::-1],
            jnp.array([0.0]),  # oldest bin is reference
        ]
    )

    # Normalize to total mass
    bin_widths_yr = jnp.diff(bin_edges_gyr) * 1e9  # Gyr -> yr
    sfr_unnorm = 10.0**log_sfr
    mass_unnorm = jnp.sum(sfr_unnorm * bin_widths_yr)
    sfr_bins = sfr_unnorm * 10.0**log_total_mass / mass_unnorm

    # Piecewise-constant (step function) — Leja+2019 ApJ 876 3 defines the continuity
    # SFH as step functions, not linearly interpolated.  Use bin EDGES (not centers)
    # for searchsorted so ages near boundaries are assigned to the correct bin.
    bin_edges_yr = bin_edges_gyr * 1e9

    # bin_idx: which bin each age falls in, using left-edge convention [edge_j, edge_{j+1})
    bin_idx = jnp.searchsorted(bin_edges_yr, age_yr, side="right") - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    sfr = sfr_bins[bin_idx]
    return jnp.maximum(sfr, 0.0)


def continuity_prior_logp(
    log_sfr_ratios: jnp.ndarray,
    df: float = 2.0,
    scale: float = 0.3,
) -> jnp.ndarray:
    """Student-t prior on log-SFR ratios (Leja+2019).

    Returns log-probability of the ratios under a Student-t(df, 0, scale)
    distribution. This penalizes sharp jumps in SFR between adjacent bins.

    Parameters
    ----------
    log_sfr_ratios : array (n_bins-1,)
        Log10 SFR ratios between adjacent bins.
    df : float
        Degrees of freedom for the Student-t distribution. Default 2.
    scale : float
        Scale parameter. Default 0.3 dex.

    Returns
    -------
    scalar
        Total log-probability summed over all ratios.
    """
    from jax.scipy.stats import t as student_t

    return jnp.sum(student_t.logpdf(log_sfr_ratios, df, loc=0.0, scale=scale))


# ---------------------------------------------------------------------------
# Dirichlet SFH (Leja+2017)
# ---------------------------------------------------------------------------


def _stick_breaking(z_fractions: jnp.ndarray) -> jnp.ndarray:
    """Convert auxiliary variables to mass fractions via stick-breaking.

    Parameters
    ----------
    z_fractions : array (N-1,)
        Auxiliary variables in (0, 1), each drawn from Beta(1, 1) = Uniform.

    Returns
    -------
    array (N,)
        Mass fractions summing to 1.0.
    """
    # f_0 = z_0
    # f_1 = (1 - z_0) * z_1
    # f_2 = (1 - z_0) * (1 - z_1) * z_2
    # ...
    # f_{N-1} = prod(1 - z_j, j=0..N-2)
    one_minus_z = 1.0 - z_fractions
    # Cumulative product of (1-z_j): [1, (1-z_0), (1-z_0)(1-z_1), ...]
    cumprod = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(one_minus_z)])

    # fractions[j] = cumprod[j] * z[j] for j < N-1
    # fractions[N-1] = cumprod[N-1]
    fractions = jnp.concatenate(
        [
            cumprod[:-1] * z_fractions,
            jnp.array([cumprod[-1]]),
        ]
    )
    return fractions


def dirichlet_sfh(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **z_kwargs,
) -> jnp.ndarray:
    """Non-parametric SFH with Dirichlet prior (Leja+2017).

    Mass fractions are derived from auxiliary variables via stick-breaking:
      f_1 = z_1
      f_2 = (1 - z_1) * z_2
      f_3 = (1 - z_1) * (1 - z_2) * z_3
      ...
      f_N = product(1 - z_j, j=1..N-1)

    When all z_j ~ Beta(1,1) = Uniform(0,1), the mass fractions follow a
    symmetric Dirichlet(1,...,1) distribution.

    Parameters
    ----------
    age_yr : array (n_age,)
        Lookback time grid in years.
    log_total_mass : float
        log10(total stellar mass formed / Msun).
    bin_edges_gyr : array (n_bins+1,) or None
        Bin edges in Gyr. Default: log-spaced from 0 to 13.7 Gyr.
    **z_kwargs
        Keyword arguments named ``z_frac_0``, ``z_frac_1``, ..., ``z_frac_{N-2}``
        containing the auxiliary Beta(1,1) variables in [0, 1].

    Returns
    -------
    array (n_age,)
        SFR at each lookback time (Msun/yr), non-negative.
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = DEFAULT_BIN_EDGES_GYR

    n_bins = bin_edges_gyr.shape[0] - 1  # len() raises ConcretizationTypeError under JIT

    # Collect z_fractions from kwargs in order
    z_fractions = jnp.array([z_kwargs[f"z_frac_{i}"] for i in range(n_bins - 1)])

    # Clip to (epsilon, 1-epsilon) for numerical stability
    z_fractions = jnp.clip(z_fractions, 1e-6, 1.0 - 1e-6)

    # Stick-breaking -> mass fractions
    mass_fracs = _stick_breaking(z_fractions)

    # Convert mass fractions to SFR: SFR_j = M_j / delta_t_j
    bin_widths_yr = jnp.diff(bin_edges_gyr) * 1e9
    total_mass = 10.0**log_total_mass
    sfr_bins = mass_fracs * total_mass / bin_widths_yr

    # Piecewise-constant (step function) — Leja+2019 ApJ 876 3; use bin EDGES
    # (not centers) so ages near boundaries go to the correct bin.
    bin_edges_yr = bin_edges_gyr * 1e9

    bin_idx = jnp.searchsorted(bin_edges_yr, age_yr, side="right") - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    sfr = sfr_bins[bin_idx]
    return jnp.maximum(sfr, 0.0)
