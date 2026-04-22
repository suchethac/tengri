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
- Wang et al. 2024 (arXiv:2401.12198): Prospector-β agebins scheme.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Default bin edges in Gyr (8 edges = 7 bins), log-spaced from 30 Myr to 13.7 Gyr.
DEFAULT_BIN_EDGES_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7])
DEFAULT_N_BINS = 7


# ── Continuity SFH (Leja+2019) ────────────────────────────────────


def continuity_sfh(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **ratio_kwargs,
) -> jnp.ndarray:
    """Non-parametric piecewise-constant SFH with continuity prior (Leja+2019).

    A flexible non-parametric model that divides the age range into N bins
    and parameterizes relative SFR changes between adjacent bins. The continuity
    prior penalizes sharp transitions, promoting smooth SFH evolution.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10 of total stellar mass formed [Msun]. Default 10.0.
    bin_edges_gyr : array_like, shape (n_bins+1,), optional
        Bin edges [Gyr]. Default: 7-edge log-spaced grid from 0 to 13.7 Gyr.
    **ratio_kwargs
        Keyword arguments ``ratio_0``, ``ratio_1``, ..., ``ratio_{N-2}``
        containing log10 SFR ratios between adjacent bins [dimensionless].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The SFH is piecewise-constant (step function) with N bins. Free parameters
    are log-ratios between adjacent bins. The oldest bin is the reference,
    and each younger bin's absolute log-SFR is the cumulative sum of ratios.

    A Student-t(df=2, scale=0.3) prior is applied to each ratio (via
    :func:`continuity_prior_logp`) to penalize sharp jumps.
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


# ── Dirichlet SFH (Leja+2017) ─────────────────────────────────────


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
    """Non-parametric piecewise-constant SFH with symmetric Dirichlet prior (Leja+2017).

    A flexible non-parametric model parameterized by mass fractions in age bins.
    The mass fractions are derived from auxiliary variables via stick-breaking,
    with a natural symmetric Dirichlet(1,...,1) prior on the fractions.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10(total stellar mass formed / Msun). Default: 10.0 (10 Gyr Msun).
    bin_edges_gyr : array_like, shape (n_bins+1,), optional
        Bin edges in Gyr. Default: 7-edge log-spaced grid from 0 to 13.7 Gyr.
    **z_kwargs
        Keyword arguments ``z_frac_0``, ``z_frac_1``, ..., ``z_frac_{N-2}``
        containing the auxiliary Beta(1,1) variables (uniform on [0, 1]).

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The mass fractions are derived from auxiliary variables :math:`z_j \\sim \\mathrm{Beta}(1,1)`
    via stick-breaking:

    .. math::

        f_0 &= z_0 \\\\
        f_1 &= (1 - z_0) z_1 \\\\
        f_2 &= (1 - z_0)(1 - z_1) z_2 \\\\
        &\\ldots \\\\
        f_{N-1} &= \\prod_{j=0}^{N-2} (1 - z_j)

    When all :math:`z_j \\sim \\mathrm{Uniform}(0, 1)`, the mass fractions
    :math:`\\mathbf{f} = (f_0, \\ldots, f_{N-1})` automatically follow
    a symmetric :math:`\\mathrm{Dirichlet}(1, \\ldots, 1)` distribution.

    The SFR in each bin is :math:`\\mathrm{SFR}_j = f_j \\cdot M_{\\star} / \\Delta t_j`,
    where :math:`\\Delta t_j` is the width of bin j.

    References
    ----------
    .. [1] J. Leja et al., "Deriving Physical Properties from Broadband
       Photometry with Prospector: Description of the Model and a Demonstration
       of its Accuracy Using 129 Galaxies in the Local Universe," ApJ, 837, 170
       (2017). arXiv:1609.09073. https://doi.org/10.3847/1538-4357/aa5ffe
    .. [2] J. Leja et al., "How to Measure Galaxy Star Formation Histories.
       II. Nonparametric Models," ApJ, 876, 3 (2019). arXiv:1811.03637.
       https://doi.org/10.3847/1538-4357/ab133c
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


# ── Redshift-aware bin edges (Prospector-β scheme) ─────────────────


def make_agebins_from_zred(
    zred: float,
    n_bins: int = 7,
    cosmo=None,
) -> np.ndarray:
    """Redshift-dependent SFH bin edges (Prospector-β scheme, Wang+2024).

    Constructs bin edges capped at the age of the universe at ``zred`` so
    that no bin extends into the future. For low redshifts the youngest two
    bins are fixed at 30 Myr and 100 Myr; interior bins are log-spaced to
    90% of the universe age; the oldest bin spans 90–100% of the universe age.

    This is a **setup-time utility** — call it when building a
    :class:`~tengri.parameters.Parameters` object, not inside the forward
    model. The returned array is plain NumPy so it can be passed as the
    ``bin_edges_gyr`` argument to :func:`continuity_sfh` or
    :func:`dirichlet_sfh`.

    Parameters
    ----------
    zred : float
        Galaxy redshift. Sets the age of the universe that caps the bins.
    n_bins : int, optional
        Total number of age bins. Default 7 (matches the tengri default).
    cosmo : CosmoParams, optional
        DSPS cosmology parameters. Default: tengri's Planck 2018 cosmology.

    Returns
    -------
    bin_edges_gyr : np.ndarray, shape (n_bins+1,)
        Age bin edges [Gyr], monotonically increasing from 0, capped at age of
        universe at ``zred``.

    Notes
    -----
    **Not JIT-compatible** — uses Python control flow and NumPy. Call once
    at model-construction time, then pass the edges as a static array.

    Ported from Prospector ``zred_to_agebins_pbeta`` (Johnson et al. 2021
    [1]_), with two changes: uses tengri's Planck 2018 cosmology instead of
    WMAP9, and returns edges in Gyr rather than log10(yr).

    References
    ----------
    .. [1] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4295/abef67
    .. [2] S. Wang et al., "Prospector-β," arXiv:2401.12198 (2024).

    Examples
    --------
    >>> edges = make_agebins_from_zred(zred=2.0)
    >>> len(edges)
    8
    >>> bool((edges[:-1] < edges[1:]).all())  # monotone
    True
    """
    from tengri.utils.cosmology import DEFAULT_COSMO, age_at_z

    if cosmo is None:
        cosmo = DEFAULT_COSMO

    t_univ_gyr = float(age_at_z(zred, cosmo=cosmo))

    log_30myr = np.log10(30e6)    # 7.477
    log_100myr = 8.0              # 10^8 yr
    log_90pct = np.log10(0.9 * t_univ_gyr * 1e9)
    log_tuniv = np.log10(t_univ_gyr * 1e9)

    if zred <= 3.0:
        n_middle = n_bins - 3
        if n_middle > 0:
            log_middle = list(np.linspace(log_100myr, log_90pct, n_middle + 1)[1:])
        else:
            log_middle = []
        log_edges = [log_30myr, log_100myr, *log_middle, log_tuniv]
    else:
        log_amin = 6.0  # 1 Myr
        log_edges_inner = list(np.linspace(log_amin, log_90pct, n_bins - 1))
        log_edges = [*log_edges_inner, log_tuniv]

    edges_gyr = np.array([0.0, *[10.0 ** le / 1e9 for le in log_edges]])
    edges_gyr = np.clip(edges_gyr, 0.0, t_univ_gyr)
    edges_gyr = np.maximum.accumulate(edges_gyr)
    return edges_gyr


# ── PSB continuity SFH (Suess+2021) ───────────────────────────────


def psb_continuity_sfh(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    tlast_gyr: float = 0.5,
    tflex_gyr: float = 2.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **ratio_kwargs,
) -> jnp.ndarray:
    """Post-starburst non-parametric SFH with variable quenching epoch (Suess+2021).

    Extends the continuity SFH (Leja+2019) with two additional parameters that
    track the quenching epoch. The oldest N bins have fixed edges and log-SFR
    ratio priors (same as continuity). A flexible zone between ``tlast_gyr``
    and ``tflex_gyr`` captures the transition epoch. The youngest bin spans
    [0, tlast_gyr] and its SFR ratio encodes how recently star formation ceased.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10 of total stellar mass formed [Msun]. Default 10.0.
    tlast_gyr : float, optional
        Lookback time of quenching onset [Gyr]. Sets the youngest bin width.
        Typical range: 0.01 to 1.0 Gyr.
    tflex_gyr : float, optional
        Upper boundary of the flexible zone [Gyr]. Default 2.0.
    bin_edges_gyr : array_like, shape (n_fixed+1,), optional
        Fixed old bin edges [Gyr]. Default: ``DEFAULT_BIN_EDGES_GYR[2:]``
        = [0.3, 1.0, 3.0, 6.0, 13.7] Gyr.
    **ratio_kwargs
        Log-SFR ratios. Convention:
        - ``ratio_young`` — youngest bin vs flex bin (large positive = burst).
        - ``ratio_old_0``, ``ratio_old_1``, ... — ratios among old fixed bins.

    Returns
    -------
    sfr : jnp.ndarray, shape (n_age,)
        Star formation rate [Msun yr^-1], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives; ``tlast_gyr`` and
    ``tflex_gyr`` must be concrete scalars (not traced inside JIT).

    Ported from Prospector ``psb_logsfr_ratios_to_agebins`` and
    ``logsfr_ratios_to_masses_psb`` (Johnson et al. 2021 [1]_), reimplemented
    as a pure JAX step-function SFH compatible with DSPS.

    References
    ----------
    .. [1] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4295/abef67
    .. [2] K. A. Suess et al., "Half-mass Radii for ~7000 Galaxies," ApJ,
       915, 87 (2021). arXiv:2101.03177.
       https://doi.org/10.3847/1538-4357/ac062c

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> t = jnp.logspace(6.0, 10.14, 256)
    >>> sfr = psb_continuity_sfh(t, log_total_mass=10.5, tlast_gyr=0.3,
    ...                          tflex_gyr=2.0, ratio_young=1.0,
    ...                          ratio_old_0=0.2, ratio_old_1=-0.3)
    >>> sfr.shape
    (256,)
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = DEFAULT_BIN_EDGES_GYR[2:]  # [0.3, 1.0, 3.0, 6.0, 13.7]

    n_fixed_bins = bin_edges_gyr.shape[0] - 1

    # Full edge array: [0, tlast, tflex, old_fixed_bins...]
    all_edges_gyr = jnp.concatenate(
        [jnp.array([0.0, tlast_gyr, tflex_gyr]), bin_edges_gyr[1:]]
    )
    n_bins_total = all_edges_gyr.shape[0] - 1

    # Old bins: log-SFR ratios (oldest = reference = 0)
    ratio_young = ratio_kwargs.get("ratio_young", 0.0)
    ratio_old = jnp.array(
        [ratio_kwargs.get(f"ratio_old_{i}", 0.0) for i in range(n_fixed_bins - 1)]
    )
    log_sfr_old = jnp.concatenate(
        [jnp.cumsum(ratio_old[::-1])[::-1], jnp.array([0.0])]
    )

    # Flex bin: same log-SFR as innermost old bin; youngest bin adds ratio_young
    log_sfr_flex = log_sfr_old[0]
    log_sfr_young = log_sfr_flex + ratio_young

    log_sfr_bins = jnp.concatenate(
        [jnp.array([log_sfr_young, log_sfr_flex]), log_sfr_old]
    )

    # Normalize to total mass
    bin_widths_yr = jnp.diff(all_edges_gyr) * 1e9
    sfr_unnorm = 10.0 ** log_sfr_bins
    mass_unnorm = jnp.sum(sfr_unnorm * bin_widths_yr)
    sfr_bins_norm = sfr_unnorm * (10.0 ** log_total_mass) / (mass_unnorm + 1e-30)

    # Piecewise-constant lookup
    bin_edges_yr = all_edges_gyr * 1e9
    bin_idx = jnp.searchsorted(bin_edges_yr, age_yr, side="right") - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins_total - 1)
    sfr = sfr_bins_norm[bin_idx]
    return jnp.maximum(sfr, 0.0)
