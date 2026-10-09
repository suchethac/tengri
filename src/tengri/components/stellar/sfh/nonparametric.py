# SPDX-License-Identifier: BSD-3-Clause
"""Non-parametric star formation history models.

Implements the Continuity (Leja+2019), Dirichlet (Leja+2017), Bursty
Continuity (Tacchella+2022), and ContinuityFlex (Leja+2019) non-parametric
SFH priors from the Prospector framework. All describe piecewise-constant SFR
in N lookback-time bins.

- **Continuity**: free parameters are log-SFR *ratios* between adjacent bins,
  with a Student-t(df=2, scale=0.3) smoothness prior penalizing sharp jumps.
- **Dirichlet**: free parameters are uniform auxiliary variables mapped to
  Beta(1, N-1-j) quantiles, which stick-break into the SFR fractions, giving
  a symmetric Dirichlet(1,...,1) prior on them.
- **Bursty continuity**: same continuity SFH, but young bins use a wider
  Student-t scale (1.0) than old bins (0.3) to permit rapid recent fluctuations.
- **ContinuityFlex**: anchored young/old bins + N flexible intermediate bins
  whose edges are derived from the SFR ratios (constant-mass-per-flex-bin).
- **PSB continuity** (Suess+2021): a youngest bin of width ``tlast``, a
  flexible zone out to ``tflex`` cut into equal-width bins, then fixed old
  bins. Distinct from ContinuityFlex: here the flex ratios set the bins'
  *amplitudes* and the widths are equal, there they set the *widths* and the
  masses are equal.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.

Bin boundaries are asymmetric, deliberately. Ages older than the last edge form
no stars: the bins are the model, and extending the oldest one past its edge
puts mass outside the normalization, which sums bin widths only (#1978). Ages
younger than the first edge take the *youngest* bin's rate, which is the
nearest bin and the only defensible extrapolation; the index arithmetic would
otherwise run off the low end and, JAX indexing being modular, hand back the
OLDEST bin's rate instead. Every ladder these functions build starts at zero
lookback, so the low end is reached only by a caller supplying its own edges.

References
----------

- Leja+2017 (arXiv:1609.09073): Dirichlet SFH prior.
- Leja+2019 (arXiv:1905.11997): Continuity and ContinuityFlex SFH priors.
- Johnson+2021: Prospector implementation.
- Tacchella et al. 2022, ApJ, 926, 134 (arXiv:2102.11954): Bursty continuity.
- Wang et al. 2024 (arXiv:2401.12198): Prospector-β agebins scheme.
- Suess et al. 2022 (ApJ 935, 146; arXiv:2207.02883): PSB flexible-zone SFH.

"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.sfh.mean_sfh import window_weight
from tengri.utils.host_array import device_table, host_array
from tengri.utils.scale import (
    log10_weighted_sum,
    pow10,
    representable_floor,
)

# Default bin edges in Gyr (8 edges = 7 bins), log-spaced from 30 Myr to 13.7 Gyr.
DEFAULT_BIN_EDGES_GYR = host_array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7])
DEFAULT_N_BINS = 7


def _piecewise_constant_sfr(age_yr, bin_edges_yr, sfr_bins, n_bins):
    """Evaluate a binned SFR on an age grid, forming nothing past the last edge.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback times to evaluate [yr].
    bin_edges_yr : array_like, shape (n_bins+1,)
        Bin edges [yr], ascending.
    sfr_bins : array_like, shape (n_bins,)
        SFR in each bin [Msun/yr].
    n_bins : int
        Number of bins. Passed explicitly because ``len`` on a traced array
        raises under JIT.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative, and exactly zero
        for ages beyond ``bin_edges_yr[-1]``.

    Notes
    -----
    **JIT-compatible**: yes, ``jnp`` primitives only.

    Ages *below* the first edge are deliberately clamped into the youngest bin,
    the nearest one. The clamp is load-bearing rather than cosmetic:
    ``searchsorted`` returns 0 there, so the index is -1, and JAX indexing is
    modular, which would silently serve the OLDEST bin's rate to the youngest
    ages. Every ladder built in this module starts at zero lookback, so this is
    reached only when a caller supplies its own ``bin_edges_gyr``.

    Ages *above* the last edge are zeroed, and that is the fix for #1978. They
    used to be clamped into the oldest bin, which extended that bin's SFR to the
    end of the age grid, outside any declared bin and outside the mass
    normalization, which sums bin widths only. With the default ladder ending at
    13.7 Gyr this was a sliver and went unnoticed; once #1975 allowed a ladder
    bounded to the age of the universe, it was 21% of the declared mass forming
    after the model said star formation stopped.
    """
    bin_idx = jnp.searchsorted(bin_edges_yr, age_yr, side="right") - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    sfr = jnp.where(age_yr > bin_edges_yr[-1], 0.0, sfr_bins[bin_idx])
    return jnp.maximum(sfr, 0.0)


def _piecewise_constant_sfr_smooth(age_yr, bin_edges_yr, sfr_bins, n_bins):
    """Evaluate a binned SFR with partial-cell weighting at each bin edge.

    Generalizes :func:`window_weight`'s exact cell-coverage treatment of a
    single boundary to an N-bin ladder: each grid cell's value is the
    coverage-weighted average of every bin's SFR it overlaps, rather than a
    point-sample lookup of whichever bin its own coordinate falls in. A grid
    cell lying entirely inside one bin is unaffected (the one window with
    nonzero coverage there contributes its full SFR, exactly the hard-lookup
    value); only the 1-2 cells straddling a bin edge differ.

    Use this in place of :func:`_piecewise_constant_sfr` where the bin edges
    themselves are free parameters (e.g. ``tlast_gyr`` / ``tflex_gyr`` in
    :func:`psb_continuity` and :func:`psb_continuity_flex`): a hard lookup
    there makes the moving edge a staircase in that parameter, the same
    mechanism `window_weight` exists to remove at a single support boundary
    (#2476). Bins with static edges (:func:`continuity`, :func:`dirichlet`)
    keep the point-sample lookup, which is exact and cheaper when nothing
    moves.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback times to evaluate [yr].
    bin_edges_yr : array_like, shape (n_bins+1,)
        Bin edges [yr], ascending; may be traced (a function of free
        parameters).
    sfr_bins : array_like, shape (n_bins,)
        SFR in each bin [Msun/yr].
    n_bins : int
        Number of bins, a static Python int (the loop below is unrolled at
        trace time, matching :func:`_piecewise_constant_sfr`'s own
        constraint on ``n_bins``).

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes; the Python ``for`` loop over ``n_bins`` unrolls
    at trace time into ``n_bins`` calls to :func:`window_weight`, each
    vectorized over ``age_yr``.

    Ages older than ``bin_edges_yr[-1]`` fall outside every bin's window and
    so are zero, matching :func:`_piecewise_constant_sfr` (#1978); ages
    younger than ``bin_edges_yr[0]`` are covered by the youngest bin's own
    window whenever the ladder starts at zero lookback (every ladder these
    functions build does).
    """
    weights = jnp.stack(
        [window_weight(age_yr, bin_edges_yr[i], bin_edges_yr[i + 1]) for i in range(n_bins)],
        axis=0,
    )
    sfr = jnp.sum(weights * jnp.asarray(sfr_bins)[:, None], axis=0)
    return jnp.maximum(sfr, 0.0)


# ── Continuity SFH (Leja+2019) ────────────────────────────────────


def _normalized_bin_sfr(log_sfr, bin_widths_yr, log_total_mass):
    r"""Scale per-bin log10 SFR so the bins form ``10**log_total_mass`` [Msun].

    Parameters
    ----------
    log_sfr : array_like, shape (n_bins,)
        Unnormalized log10 SFR in each bin [dex]; any additive constant is
        removed by the normalization.
    bin_widths_yr : array_like, shape (n_bins,)
        Bin widths [yr].
    log_total_mass : float
        log10 of the total mass formed [Msun].

    Returns
    -------
    ndarray, shape (n_bins,)
        SFR in each bin [Msun/yr].

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.

    .. math::

        \mathrm{SFR}_i = 10^{\,s_i + m_* - L}, \qquad
        L = \log_{10}\sum_j \Delta t_j \, 10^{s_j}

    where :math:`s_i` is ``log_sfr`` [dex], :math:`\Delta t_j` the bin width
    [yr] and :math:`m_*` is ``log_total_mass`` [dex Msun]. The sum is a base-10
    log-sum-exp, so no bin is exponentiated at its own magnitude: cumulative
    ratios of a few hundred dex (reachable under the untruncated Student-t
    ratio prior) neither overflow to ``inf/inf = NaN`` nor underflow to an
    all-zero history. A zero-width bin is held at the smallest normal width.
    """
    log_w = log_sfr + jnp.log10(jnp.maximum(bin_widths_yr, representable_floor(0.0)))
    return pow10(log_sfr + log_total_mass - log10_weighted_sum(log_w, 1.0))


def _continuity_flex_log_widths(flex_ratios: list, t_flex_yr: float) -> jnp.ndarray:
    r"""log10 flex-bin widths [dex yr] from the ``flex_*`` log ratios.

    Parameters
    ----------
    flex_ratios : list of float or traced scalar
        ``flex_0 ... flex_{N-1}`` [dimensionless]; empty for one flat bin.
    t_flex_yr : float
        Total flex-zone span :math:`T_{\rm flex}` [yr].

    Returns
    -------
    ndarray, shape (N+1,)
        :math:`\log_{10}\Delta t_i = \log_{10}T_{\rm flex} + c_i - L(c)`,
        with :math:`c_i` the cumulative sum of the ratios (:math:`c_0 = 0`) and
        :math:`L` the base-10 log-sum-exp; the widths sum to ``t_flex_yr``.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. Shared by :func:`continuity_flex` and
    :func:`_continuity_flex_edges_yr` so both read one set of edges.
    """
    if flex_ratios:
        c = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.array(flex_ratios))])
    else:
        c = jnp.zeros(1)
    return math.log10(t_flex_yr) + c - log10_weighted_sum(c, 1.0)


def continuity(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **ratio_kwargs,
) -> jnp.ndarray:
    r"""Non-parametric piecewise-constant SFH with continuity prior (Leja+2019).

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

    Raises
    ------
    TypeError
        If any unknown keyword arguments are supplied in ``**ratio_kwargs``.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The SFH is piecewise-constant (step function) with N bins. Free parameters
    are log-ratios between adjacent bins. The oldest bin is the reference,
    and each younger bin's absolute log-SFR is the cumulative sum of ratios.

    A Student-t(df=2, scale=0.3) prior is applied to each ratio (via
    :func:`continuity_prior_logp`) to penalize sharp jumps.

    The bin rates are normalized in log space,

    .. math::

        \mathrm{SFR}_i = 10^{\,s_i + m_* - L}, \qquad
        L = \log_{10}\sum_j \Delta t_j \, 10^{s_j},

    with :math:`s_i` the cumulative log10 SFR [dex], :math:`\Delta t_j` the bin
    widths [yr] and :math:`m_*` = ``log_total_mass`` [dex Msun], so a cumulative
    ratio of several hundred dex stays finite and forms exactly
    ``10**log_total_mass`` (see :func:`_normalized_bin_sfr`). Gradients with
    respect to the ratios and ``log_total_mass`` are finite.
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = device_table(DEFAULT_BIN_EDGES_GYR)

    n_bins = bin_edges_gyr.shape[0] - 1  # len() raises ConcretizationTypeError under JIT

    # Validate that all kwargs are expected ratio_* parameters
    valid_ratio_keys = {f"ratio_{i}" for i in range(n_bins - 1)}
    unknown = sorted(k for k in ratio_kwargs if k not in valid_ratio_keys)
    if unknown:
        raise TypeError(
            f"continuity() got unexpected keyword argument(s) {unknown}; the "
            f"{n_bins}-bin grid accepts 'ratio_0'..'ratio_{n_bins - 2}' only."
        )

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
    sfr_bins = _normalized_bin_sfr(log_sfr, bin_widths_yr, log_total_mass)

    # Piecewise-constant (step function); Leja+2019 ApJ 876 3 defines the continuity
    # SFH as step functions, not linearly interpolated.  Use bin EDGES (not centers)
    # for searchsorted so ages near boundaries are assigned to the correct bin.
    bin_edges_yr = bin_edges_gyr * 1e9

    return _piecewise_constant_sfr(age_yr, bin_edges_yr, sfr_bins, n_bins)


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

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.scipy.stats`` for Student-t density.
    """
    from jax.scipy.stats import t as student_t

    return jnp.sum(student_t.logpdf(log_sfr_ratios, df, loc=0.0, scale=scale))


def bursty_continuity_prior_logp(
    log_sfr_ratios: jnp.ndarray,
    bin_edges_gyr: jnp.ndarray,
    t_split_gyr: float = 1.0,
    scale_young: float = 1.0,
    scale_old: float = 0.3,
    df: float = 2.0,
) -> jnp.ndarray:
    """Compute the bursty-continuity prior log-probability on log-SFR ratios (Tacchella+2022).

    Same prior structure as :func:`continuity_prior_logp` (Leja+2019), but
    applies a wider scale to ratios in young bins (lookback time < ``t_split_gyr``)
    to allow rapid recent SFR fluctuations while keeping old bins smooth.

    Parameters
    ----------
    log_sfr_ratios : array_like, shape (n_bins-1,)
        Log10 SFR ratios between adjacent bins [dimensionless].
        Ratio ``i`` controls the transition from bin ``i+1`` (older) to bin
        ``i`` (younger), following the continuity SFH convention.
    bin_edges_gyr : array_like, shape (n_bins+1,)
        Age bin edges [Gyr], monotonically increasing. Must match the edges
        used to construct the SFH (e.g. ``DEFAULT_BIN_EDGES_GYR``).
    t_split_gyr : float, optional
        Lookback time split [Gyr] separating the bursty (young) regime from
        the smooth (old) regime. Default 1.0 Gyr.
    scale_young : float, optional
        Student-t scale for ratios whose *younger* bin edge is inside the
        bursty regime (``bin_edges_gyr[i+1] < t_split_gyr``). Default 1.0 dex.
    scale_old : float, optional
        Student-t scale for old-regime ratios. Default 0.3 dex (same as the
        standard continuity prior).
    df : float, optional
        Degrees of freedom for both Student-t distributions. Default 2.

    Returns
    -------
    logp : scalar
        Total log-probability [dimensionless] summed over all ratios.

    Notes
    -----
    **JIT-compatible**: yes, ``jnp.where`` selects the scale without branching.
    ``t_split_gyr``, ``scale_young``, ``scale_old``, and ``df`` must be
    concrete (non-traced) scalars.

    **Gradient-safe**: yes, differentiable w.r.t. ``log_sfr_ratios`` everywhere.

    Ratio ``i`` is classified as *young* when ``bin_edges_gyr[i+1] < t_split_gyr``,
    i.e. its younger bin edge lies in the bursty regime. The per-ratio log-prob is:

    .. math::

        \\log p(r_i) = \\log t_{\\nu}\\!\\left(r_i \\mid 0,\\, \\sigma_i\\right),
        \\quad \\sigma_i = \\begin{cases}
            \\sigma_{\\rm young} & \\text{if } t_{i+1} < t_{\\rm split} \\\\
            \\sigma_{\\rm old}   & \\text{otherwise}
        \\end{cases}

    where :math:`t_\\nu` is the Student-t density with :math:`\\nu = \\mathtt{df}`
    degrees of freedom, :math:`t_{i+1}` is ``bin_edges_gyr[i+1]``, and
    :math:`\\sigma_{\\rm young} = 1.0`, :math:`\\sigma_{\\rm old} = 0.3` by default.

    Follows the prior parameterization in Tacchella et al. 2022 [1]_.

    References
    ----------
    .. [1] S. Tacchella et al., "Star Formation Histories from SEDs and Spectra,"
       ApJ, 926, 134 (2022). arXiv:2102.11954.
       https://doi.org/10.3847/1538-4357/ac3aca
    .. [2] J. Leja et al., "How to Measure Galaxy Star Formation Histories.
       II. Nonparametric Models," ApJ, 876, 3 (2019). arXiv:1811.03637.
       https://doi.org/10.3847/1538-4357/ab133c
    """
    from jax.scipy.stats import t as student_t

    # bin_edges_gyr has n_bins+1 edges; ratio i connects bin i (young) to bin i+1 (old).
    # The younger edge of ratio i's young bin is bin_edges_gyr[i+1].
    younger_edges = bin_edges_gyr[1:-1]  # shape (n_bins-1,)
    is_young = younger_edges < t_split_gyr
    scales = jnp.where(is_young, scale_young, scale_old)
    return jnp.sum(student_t.logpdf(log_sfr_ratios, df, loc=0.0, scale=scales))


# ── Dirichlet SFH (Leja+2017) ─────────────────────────────────────


def _stick_breaking(z_fractions: jnp.ndarray) -> jnp.ndarray:
    """Convert auxiliary variables to a simplex vector via stick-breaking.

    Parameters
    ----------
    z_fractions : array (N-1,)
        Auxiliary variables in (0, 1). For a symmetric Dirichlet(1, ..., 1)
        result, element ``j`` must be a Beta(1, N-1-j) variate — see
        :func:`dirichlet`, which maps uniform latents through that quantile
        before calling this helper.

    Returns
    -------
    array (N,)
        Non-negative fractions summing to 1.0.
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


def dirichlet(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **z_kwargs,
) -> jnp.ndarray:
    """Non-parametric piecewise-constant SFH with symmetric Dirichlet prior (Leja+2017).

    A flexible non-parametric model parameterized by the *SFR fractions* in age
    bins. The SFR fractions are derived from auxiliary variables via
    stick-breaking, giving an exactly symmetric Dirichlet(1,...,1) prior on
    them; the bin masses follow as :math:`m_j \\propto f_j \\Delta t_j`.

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
        containing the auxiliary variables :math:`u_j`, uniform on [0, 1]
        [dimensionless]. They are mapped internally to the Beta variates the
        Dirichlet construction requires.

    Raises
    ------
    TypeError
        If any unknown keyword arguments are supplied in ``**z_kwargs``.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives. ``n_bins``
    comes from the static shape of ``bin_edges_gyr``, and the Beta exponents
    are a static NumPy vector built from it.

    Leja et al. 2017 [1]_ (Sect. 2.3 and Appendix) stick-breaks the
    :math:`N` **SFR fractions** from :math:`N-1` auxiliary variables
    :math:`z_j \\sim \\mathrm{Beta}(N-1-j,\\, 1)`. tengri exposes the
    auxiliaries as :math:`u_j \\sim \\mathrm{Uniform}(0, 1)` and applies the
    Beta(1, N-1-j) quantile function (equivalently, :math:`1 - z_j` for the
    Leja :math:`z_j` above):

    .. math::

        v_j = 1 - (1 - u_j)^{1/(N-1-j)}, \\qquad j = 0, \\ldots, N-2

    where :math:`u_j` is the uniform latent [dimensionless] and :math:`v_j`
    the Beta(1, N-1-j) variate [dimensionless]. Stick-breaking then gives the
    SFR fractions

    .. math::

        f_0 &= v_0 \\\\
        f_1 &= (1 - v_0) v_1 \\\\
        f_2 &= (1 - v_0)(1 - v_1) v_2 \\\\
        &\\ldots \\\\
        f_{N-1} &= \\prod_{j=0}^{N-2} (1 - v_j)

    and :math:`\\mathbf{f} = (f_0, \\ldots, f_{N-1})` is then exactly a
    symmetric :math:`\\mathrm{Dirichlet}(1, \\ldots, 1)` vector: every bin has
    mean SFR fraction :math:`1/N` and marginal :math:`\\mathrm{Beta}(1, N-1)`.

    The mass fractions weight :math:`f_j` by the bin widths, and the per-bin
    SFR follows from the mass:

    .. math::

        m_j = \\frac{f_j \\Delta t_j}{\\sum_k f_k \\Delta t_k}, \\qquad
        \\mathrm{SFR}_j = \\frac{m_j M_{\\star}}{\\Delta t_j}
                        = \\frac{f_j M_{\\star}}{\\sum_k f_k \\Delta t_k}

    with :math:`\\Delta t_j` the width of bin :math:`j` [yr],
    :math:`M_{\\star} = 10^{\\mathtt{log\\_total\\_mass}}` [Msun] and
    :math:`\\mathrm{SFR}_j` in [Msun/yr]. So :math:`\\mathrm{SFR}_j \\propto
    f_j`, as the name "SFR fraction" says.

    Implements the same model as Prospector's ``zfrac_to_sfrac`` /
    ``zfrac_to_masses`` (Johnson et al. 2021 [3]_), reached from
    uniform latents rather than Beta-distributed ones.

    References
    ----------
    .. [1] J. Leja et al., "Deriving Physical Properties from Broadband
       Photometry with Prospector: Description of the Model and a Demonstration
       of its Accuracy Using 129 Galaxies in the Local Universe," ApJ, 837, 170
       (2017). arXiv:1609.09073. https://doi.org/10.3847/1538-4357/aa5ffe
    .. [2] J. Leja et al., "How to Measure Galaxy Star Formation Histories.
       II. Nonparametric Models," ApJ, 876, 3 (2019). arXiv:1811.03637.
       https://doi.org/10.3847/1538-4357/ab133c
    .. [3] B. D. Johnson et al., "Stellar Population Inference with Prospector,"
       ApJS, 254, 22 (2021). arXiv:2012.01426.
       https://doi.org/10.3847/1538-4365/abef67
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = device_table(DEFAULT_BIN_EDGES_GYR)

    n_bins = bin_edges_gyr.shape[0] - 1  # len() raises ConcretizationTypeError under JIT

    # Validate that all kwargs are expected z_frac_* parameters
    valid_z_frac_keys = {f"z_frac_{i}" for i in range(n_bins - 1)}
    unknown = sorted(k for k in z_kwargs if k not in valid_z_frac_keys)
    if unknown:
        raise TypeError(
            f"dirichlet() got unexpected keyword argument(s) {unknown}; the "
            f"{n_bins}-bin grid accepts 'z_frac_0'..'z_frac_{n_bins - 2}' only."
        )
    # Collect the uniform latents from kwargs in order
    u_latents = jnp.array([z_kwargs[f"z_frac_{i}"] for i in range(n_bins - 1)])
    u_latents = jnp.clip(u_latents, 0.0, 1.0)

    # Uniform -> Beta(1, N-1-j) via the inverse CDF. The exponents depend only
    # on the static bin count, so they are a concrete NumPy vector.
    beta_exponents = jnp.asarray(1.0 / (n_bins - 1 - np.arange(n_bins - 1, dtype=float)))
    v_betas = 1.0 - (1.0 - u_latents) ** beta_exponents

    # Clip the Beta variates, not the uniform latents, to (epsilon, 1-epsilon):
    # the quantile map compresses the upper end (u = 1 - 1e-6 gives v_0 = 0.9
    # for seven bins), so clipping before it would cap the youngest SFR
    # fraction at 0.9 and cut the Dirichlet support.
    v_betas = jnp.clip(v_betas, 1e-6, 1.0 - 1e-6)

    # Stick-breaking -> SFR fractions (Dirichlet(1,...,1); Leja+2017)
    sfr_fracs = _stick_breaking(v_betas)

    # Mass fractions weight the SFR fractions by the bin widths
    bin_widths_yr = jnp.diff(bin_edges_gyr) * 1e9
    mass_unnorm = sfr_fracs * bin_widths_yr
    mass_fracs = mass_unnorm / jnp.sum(mass_unnorm)

    # Convert mass fractions to SFR: SFR_j = M_j / delta_t_j
    total_mass = 10.0**log_total_mass
    sfr_bins = mass_fracs * total_mass / bin_widths_yr

    # Piecewise-constant (step function); Leja+2019 ApJ 876 3; use bin EDGES
    # (not centers) so ages near boundaries go to the correct bin.
    bin_edges_yr = bin_edges_gyr * 1e9

    return _piecewise_constant_sfr(age_yr, bin_edges_yr, sfr_bins, n_bins)


# ── Redshift-aware bin edges (Prospector-β scheme) ─────────────────


def make_agebins_from_zred(
    zred: float,
    n_bins: int = 7,
    cosmo=None,
) -> np.ndarray:
    """Redshift-dependent SFH bin edges (Prospector-β scheme, Wang+2024).

    Constructs bin edges capped at the age of the universe at ``zred`` so
    that no bin extends into the future. For ``zred <= 3`` the youngest two
    edges are fixed at 30 Myr and 100 Myr, the remaining interior edges are
    log-spaced from 100 Myr to 90% of the universe age, and the oldest bin
    spans 90-100% of the universe age. For ``zred > 3`` the universe is too
    young to hold a 100 Myr bin and still resolve the rest of cosmic time, so
    none of the youngest edges are fixed: the grid is an ``n_bins``-point
    log-spacing from ``10**7.1295 yr`` (13.47 Myr, Prospector-beta's own
    ``amin``) to 90% of the universe age, but ``amin`` only anchors that
    grid -- it is dropped, and the youngest edge is the grid's SECOND point
    -- with the oldest bin again spanning 90-100%.

    This is a **setup-time utility**: call it when building a
    :class:`~tengri.Parameters` object, not inside the forward
    model. The returned array is plain NumPy so it can be passed as the
    ``bin_edges_gyr`` argument to :func:`continuity` or
    :func:`dirichlet`.

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
    **Not JIT-compatible**: uses Python control flow and NumPy. Call once
    at model-construction time, then pass the edges as a static array.

    Implements Prospector ``zred_to_agebins_pbeta``
    (``prospect/models/transforms.py``, Johnson et al. 2021 [1]_), including
    its ``amin = 7.1295`` (log10 yr; 13.47 Myr) grid anchor for ``zred > 3``
    -- itself dropped from the edges, matching that function's own
    ``agelims[0] = 0`` overwrite -- with two changes: uses tengri's Planck
    2018 cosmology instead of WMAP9, and returns edges in Gyr rather than
    log10(yr).

    References
    ----------
    .. [1] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67
    .. [2] S. Wang et al., "Prospector-β," arXiv:2401.12198 (2024).

    Examples
    --------
    >>> edges = make_agebins_from_zred(zred=2.0)
    >>> len(edges)
    8
    >>> bool((edges[:-1] < edges[1:]).all())  # monotone
    True
    """
    from tengri.cosmology import DEFAULT_COSMO, age_at_z

    if cosmo is None:
        cosmo = DEFAULT_COSMO

    t_univ_gyr = float(age_at_z(zred, cosmo=cosmo))

    log_30myr = np.log10(30e6)  # 7.477
    log_100myr = 8.0  # 10^8 yr
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
        # amin (13.47 Myr) anchors the grid but is itself discarded, exactly
        # as Prospector-beta's own ``agelims[0] = 0`` overwrite discards its
        # own first linspace point: the youngest edge here is the SECOND
        # point of an n_bins-point linspace from amin, not amin itself.
        log_amin = 7.1295  # 13.47 Myr; Prospector-beta's own amin (transforms.py)
        log_edges_inner = list(np.linspace(log_amin, log_90pct, n_bins)[1:])
        log_edges = [*log_edges_inner, log_tuniv]

    edges_gyr = np.array([0.0, *[10.0**le / 1e9 for le in log_edges]])
    edges_gyr = np.clip(edges_gyr, 0.0, t_univ_gyr)
    edges_gyr = np.maximum.accumulate(edges_gyr)
    return edges_gyr


# ── PSB continuity SFH (Suess+2021) ───────────────────────────────


def psb_continuity(
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
    and ``tflex_gyr`` captures the transition epoch, resolved into
    ``n_flex`` equal-width bins. The youngest bin spans
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
        = [0.1, 0.3, 1.0, 3.0, 6.0, 13.7] Gyr. The first entry is the boundary
        ``tflex_gyr`` replaces and is discarded; only ``[1:]`` is read, so
        ``tflex_gyr`` must stay below ``bin_edges_gyr[1]`` or the resulting
        ladder is not ascending. :func:`psb_continuity_flex` removes that
        constraint by deriving the fixed bins from ``tflex_gyr``.
    **ratio_kwargs
        Log-SFR ratios [dex]. Every ratio is
        :math:`\\log_{10}(\\mathrm{SFR}_i / \\mathrm{SFR}_{i+1})` for adjacent
        bins ordered youngest to oldest. Convention:

        - ``ratio_young``: youngest bin vs the youngest flex bin (large
          positive = recent burst).
        - ``flex_0``, ``flex_1``, ..., ``flex_{n_flex-2}``: ratios *within* the
          flexible zone. The number of ``flex_*`` keys sets ``n_flex``: N keys
          give N+1 equal-width flex bins. Default: no keys, so ``n_flex = 1``
          and the flexible zone is a single bin, which is the layout this
          model shipped with.
        - ``ratio_old_0``: log10 SFR ratio linking oldest flex bin to youngest
          fixed bin (``n_fixed`` ratios total in the old section). Default 0,
          which reproduces the pre-fix behavior (link pinned to continuous).
        - ``ratio_old_1``, ``ratio_old_2``, ...: log10 SFR ratios among
          adjacent fixed bins (oldest within the fixed section = reference = 0).

    Returns
    -------
    sfr : jnp.ndarray, shape (n_age,)
        Star formation rate [Msun yr^-1], non-negative.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives. ``tlast_gyr`` and
    ``tflex_gyr`` may be traced; the bin *count* is static because it is read
    off the ``flex_*`` keyword names rather than from a numeric argument.

    Implements the same calculation as Prospector ``psb_logsfr_ratios_to_agebins`` and
    ``logsfr_ratios_to_masses_psb`` (Johnson et al. 2021 [1]_), reimplemented
    as a pure JAX step-function SFH compatible with DSPS.

    References
    ----------
    .. [1] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67
    .. [2] K. A. Suess et al., "Half-mass Radii for ~7000 Galaxies," ApJ,
       915, 87 (2021). arXiv:2101.03177.
       https://doi.org/10.3847/1538-4357/ac062c

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> t = jnp.logspace(6.0, 10.14, 256)
    >>> sfr = psb_continuity(
    ...     t,
    ...     log_total_mass=10.5,
    ...     tlast_gyr=0.3,
    ...     tflex_gyr=2.0,
    ...     ratio_young=1.0,
    ...     flex_0=0.1,
    ...     flex_1=-0.2,
    ...     ratio_old_0=0.2,
    ...     ratio_old_1=-0.3,
    ... )
    >>> sfr.shape
    (256,)
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = device_table(DEFAULT_BIN_EDGES_GYR)[2:]  # [0.3, 1.0, 3.0, 6.0, 13.7]

    n_fixed_bins = bin_edges_gyr.shape[0] - 1

    # Number of flexible bins, from the ``flex_*`` ratios the caller supplied:
    # N ratios describe N+1 bins, and no ratios is the single flex bin this
    # model shipped with. Counting kwargs (rather than taking an ``n_flex``
    # argument) keeps the count a Python int, so the bin count stays static
    # under JIT while ``tlast_gyr`` / ``tflex_gyr`` remain traceable.
    n_flex_ratios = sum(1 for k in ratio_kwargs if k.startswith("flex_"))
    n_flex_bins = n_flex_ratios + 1

    # Full edge array: [0, tlast, <n_flex equal-width flex edges>, old_fixed...]
    flex_edges_gyr = jnp.linspace(tlast_gyr, tflex_gyr, n_flex_bins + 1)[1:]
    all_edges_gyr = jnp.concatenate(
        [jnp.array([0.0, tlast_gyr]), flex_edges_gyr, bin_edges_gyr[1:]]
    )
    n_bins_total = all_edges_gyr.shape[0] - 1

    # Old bins: log-SFR ratios (oldest = reference = 0)
    ratio_young = ratio_kwargs.get("ratio_young", 0.0)
    ratio_old = jnp.array([ratio_kwargs.get(f"ratio_old_{i}", 0.0) for i in range(n_fixed_bins)])

    # Separate the link (ratio_old_0) from the adjacent steps (ratio_old_1:)
    ratio_old_link = ratio_old[0]
    ratio_old_adjacent = ratio_old[1:]

    # Build log-SFR values for the fixed section (oldest = reference = 0)
    log_sfr_fixed_section = jnp.concatenate(
        [jnp.cumsum(ratio_old_adjacent[::-1])[::-1], jnp.array([0.0])]
    )

    # The oldest flex bin's log-SFR follows from the link and youngest fixed bin
    log_sfr_oldest_flex = log_sfr_fixed_section[0] + ratio_old_link

    # Flex bins: each ``flex_i`` steps log-SFR from flex bin i to bin i+1.
    # With no ``flex_*`` ratios this collapses to the single flex bin at
    # ``log_sfr_oldest_flex``, bit-identical to the one-flex-bin model.
    flex_ratios = jnp.array([ratio_kwargs.get(f"flex_{i}", 0.0) for i in range(n_flex_ratios)])
    log_sfr_flex = (
        jnp.concatenate([jnp.cumsum(flex_ratios[::-1])[::-1], jnp.array([0.0])])
        + log_sfr_oldest_flex
    )

    # Alias for consistency with downstream code
    log_sfr_old = log_sfr_fixed_section
    log_sfr_young = log_sfr_flex[0] + ratio_young

    log_sfr_bins = jnp.concatenate([jnp.array([log_sfr_young]), log_sfr_flex, log_sfr_old])

    # Normalize to total mass
    bin_widths_yr = jnp.diff(all_edges_gyr) * 1e9
    sfr_bins_norm = _normalized_bin_sfr(log_sfr_bins, bin_widths_yr, log_total_mass)

    # Piecewise-constant lookup. This function is not registered under any
    # SFH type name (`psb_continuity_flex` is the registered generalization,
    # #2184), so it is unreached by any moving free parameter through
    # `SEDModel.build`; the hard lookup, exact at fixed edges, is unchanged.
    bin_edges_yr = all_edges_gyr * 1e9
    return _piecewise_constant_sfr(age_yr, bin_edges_yr, sfr_bins_norm, n_bins_total)


#: Number of fixed old bins :func:`psb_continuity_flex` lays down when no
#: ``bin_edges_gyr`` is given, and the oldest edge [Gyr] it lays them out to.
#: Three equal-width fixed bins spanning ``tflex_gyr`` to 13.7 Gyr, which is the
#: oldest edge every other tengri non-parametric ladder ends at.
PSB_FLEX_DEFAULT_N_FIXED = 3
PSB_FLEX_DEFAULT_MAX_AGE_GYR = 13.7


def psb_continuity_flex(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    tlast_gyr: float = 0.2,
    tflex_gyr: float = 2.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    age_universe_yr: float | None = None,
    **ratio_kwargs,
) -> jnp.ndarray:
    r"""Post-starburst SFH with equal-width fixed old bins.

    :func:`psb_continuity` with one change: the fixed old bins are laid out as
    ``n_fixed`` equal-width intervals spanning ``[tflex_gyr, max_age]``, rather
    than being taken verbatim from ``bin_edges_gyr``. That is what makes the
    ladder ascending for *any* ``tflex_gyr`` below ``max_age``.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10 of total stellar mass formed [Msun]. Default 10.0.
    tlast_gyr : float, optional
        Lookback time of quenching onset [Gyr]; width of the youngest bin.
        Default 0.2.
    tflex_gyr : float, optional
        Boundary between the flexible zone and the fixed old bins [Gyr].
        Default 2.0.
    bin_edges_gyr : array_like, shape (n_fixed+1,), optional
        Supplies only two things here: the **number** of fixed old bins
        (``len - 1``) and the **oldest edge** (``[-1]``, the oldest lookback
        time that forms stars). The interior values are not used, because the
        fixed bins are equal-width by construction. Takes priority over
        ``age_universe_yr`` when both are given. Default: 3 bins out to
        ``age_universe_yr`` (or 13.7 Gyr if that is also absent).
    age_universe_yr : float, optional
        Age of the universe at the model's own redshift [yr]
        (``age_at_z(z)``); the oldest fixed-bin edge, so the fixed section
        never extends past the Big Bang (#2645). Ignored when
        ``bin_edges_gyr`` is given. Not a free parameter: the orchestrator
        injects it from the evaluation redshift, the same way it injects
        ``psb_wild2020``'s ``age_universe_yr``. Default ``None``, which falls
        back to the constant 13.7 Gyr (pre-#2645 behavior).
    **ratio_kwargs
        As :func:`psb_continuity`: ``ratio_young``, ``flex_0`` ...
        ``flex_{n_flex-2}``, and ``ratio_old_0`` ... ``ratio_old_{n_fixed-1}``
        (link + adjacent steps).

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes. ``tflex_gyr`` may be traced; the bin *counts* are
    static (read off the ``flex_*`` keyword names and ``bin_edges_gyr``'s
    length).

    **Why not just reuse** :func:`psb_continuity` **'s ladder.** That function
    splices ``tflex_gyr`` in ahead of ``bin_edges_gyr[1:]``, which requires the
    caller to keep ``tflex_gyr`` below the first fixed edge. With the shipped
    default ladder that first edge is 0.3 Gyr while ``tflex_gyr``'s prior runs
    to 5.0 Gyr, so the edges cross and :func:`jax.numpy.searchsorted` is
    evaluated on a non-ascending array. Deriving the fixed bins from
    ``tflex_gyr`` removes the ordering constraint instead of asking the user to
    respect it.

    Implements the post-starburst-optimized non-parametric SFH of Suess et al.
    2022 [1]_, on the Prospector continuity machinery (Johnson et al. 2021
    [2]_). The step between the oldest flex bin and the youngest fixed bin is
    controlled by ``ratio_old_0`` (the link), with a default of 0 (pinned)
    for backward compatibility (default 0 reproduces pre-#2612 bit-exactly).

    **Fixed section bounded to cosmic time (#2645).** The fixed old bins span
    ``[tflex_gyr, age_universe_yr]`` when ``age_universe_yr`` is supplied,
    instead of a redshift-independent 13.7 Gyr: every other tengri
    non-parametric ladder already scales its oldest edge to ``age_at_z(z)``
    (:func:`make_agebins_from_zred`), and ``psb_wild2020`` already receives
    this same ``age_universe_yr`` injection for its burst DPL. Without it
    (``age_universe_yr=None``, e.g. a bare function call) the fixed section
    falls back to the constant 13.7 Gyr, so a caller at z > 0 that forms
    stars out to the old constant edge describes star formation before the
    Big Bang -- the orchestrator's eager forward path warns
    (:class:`~tengri.components.stellar.component.SFHBeforeBigBangWarning`)
    and truncates that mass.

    **Approximation of Suess et al. 2022, Sect. 3.1.4, in two places.** That
    paper divides the flexible zone into bins of equal *mass* whose edges move,
    and fixes the old-bin edges by a template; both zones here are cut into
    bins of equal *width*, the flexible ones carrying free amplitudes
    (``flex_*``) and the fixed ones laid out from ``tflex_gyr``. The two
    constructions agree only where the history is flat across the zone
    concerned. Valid as a reparameterization of the same three-part shape, not
    as a reproduction of the paper's bin edges.

    References
    ----------
    .. [1] K. A. Suess et al., "Recovering the Star Formation Histories of
       Recently Quenched Galaxies: The Impact of Model and Prior Choices,"
       ApJ, 935, 146 (2022). arXiv:2207.02883.
    .. [2] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> t = jnp.logspace(6.0, 10.14, 256)
    >>> sfr = psb_continuity_flex(
    ...     t,
    ...     log_total_mass=10.0,
    ...     tlast_gyr=0.2,
    ...     tflex_gyr=2.0,
    ...     ratio_young=0.3,
    ...     flex_0=0.1,
    ...     flex_1=-0.2,
    ...     flex_2=0.05,
    ...     flex_3=0.1,
    ...     ratio_old_0=-0.1,
    ...     ratio_old_1=0.2,
    ...     ratio_old_2=-0.15,
    ...     age_universe_yr=13.79e9,
    ... )
    >>> sfr.shape
    (256,)
    """
    if bin_edges_gyr is not None:
        n_fixed = bin_edges_gyr.shape[0] - 1
        max_age_gyr = bin_edges_gyr[-1]
    elif age_universe_yr is not None:
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED
        max_age_gyr = age_universe_yr / 1e9
    else:
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED
        max_age_gyr = PSB_FLEX_DEFAULT_MAX_AGE_GYR
    fixed_edges_gyr = jnp.linspace(tflex_gyr, max_age_gyr, n_fixed + 1)
    return psb_continuity(
        age_yr,
        log_total_mass=log_total_mass,
        tlast_gyr=tlast_gyr,
        tflex_gyr=tflex_gyr,
        bin_edges_gyr=fixed_edges_gyr,
        **ratio_kwargs,
    )


# ── ContinuityFlex SFH (Leja+2019) ────────────────────────────────

#: Narrowest bin, as a fraction of its upper edge, that the SFH integrand resolves.
#: The stellar integrand brackets every bin edge with knots at ``1 +/- 1e-6``
#: (``_inject_edge_knots``); a bin must be wider than the span those knots cover.
_MIN_REL_BIN_WIDTH = 1e-5


def _rates_from_resolved_masses(edges_yr: jnp.ndarray, log_mass: jnp.ndarray) -> jnp.ndarray:
    r"""Per-bin SFR [Msun/yr] that forms each bin's mass over the edges the integrator sees.

    Parameters
    ----------
    edges_yr : array_like, shape (n_bins+1,)
        Ascending bin edges [yr]; ``edges_yr[0] = 0``.
    log_mass : array_like, shape (n_bins,)
        log10 of the mass each bin is declared to form [dex Msun].

    Returns
    -------
    ndarray, shape (n_bins,)
        SFR [Msun/yr] per bin; ``sum(sfr * diff(edges_yr)) = sum(10**log_mass)``.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes; static shapes, ``jnp`` primitives only.

    A bin is *unresolved* when its realized width ``diff(edges_yr)`` is below
    :data:`_MIN_REL_BIN_WIDTH` of its upper edge. Such a bin is at or below the
    float spacing of the edge itself (width exactly zero once it underflows) and
    narrower than the :math:`\pm 10^{-6}` knots the stellar integrand places
    around every edge, so no piecewise-constant rate over it can be summed or
    integrated: ``rate = mass / width`` either overflows or multiplies a zero
    width. Its mass is instead handed to the nearest resolved bin by index (the
    younger on a tie) and spread over that bin's width plus the unresolved
    widths it absorbs, which are contiguous with it. The total formed mass is
    therefore ``sum(10**log_mass)`` for every input, where a rate-clamped
    bookkeeping dropped the mass of every unresolved bin (measured: -11 % at
    flex ratios of 30 dex). The unresolved bin itself is served the absorbing
    bin's rate, so no sample ever reads a zero or overflowed rate. Bins that
    are resolved are never altered.
    """
    n_bins = log_mass.shape[0]
    width = edges_yr[1:] - edges_yr[:-1]
    resolved = width > _MIN_REL_BIN_WIDTH * edges_yr[1:]
    idx = jnp.arange(n_bins)
    dist = jnp.where(resolved[None, :], jnp.abs(idx[:, None] - idx[None, :]), n_bins + 1)
    owner = jnp.argmin(dist, axis=1)  # (n_bins,) resolved bin that absorbs bin i
    member = owner[None, :] == idx[:, None]  # [j, i]: bin i is absorbed by bin j
    # An unresolved row absorbs nothing; keep its own term so no row is empty.
    member = member | (jnp.eye(n_bins, dtype=bool) & ~resolved[:, None])
    log_mass_owner = log10_weighted_sum(
        jnp.broadcast_to(log_mass[None, :], (n_bins, n_bins)), member.astype(log_mass.dtype)
    )
    width_owner = jnp.sum(jnp.where(member, width[None, :], 0.0), axis=1)
    safe_width = jnp.where(resolved, width_owner, 1.0)
    rate_owner = pow10(jnp.where(resolved, log_mass_owner - jnp.log10(safe_width), 0.0))
    return rate_owner[owner]


# Anchor bin edges [t_young_end_gyr, t_old_start_gyr, t_max_gyr].
# ContinuityFlex anchor defaults:
#   young bin [0, 10^7.5 yr] = [0, 31.6 Myr], old bin [10^9.7, 10^10.136 yr] = [5.01, 13.7 Gyr].
CFLEX_DEFAULT_ANCHOR_GYR = host_array([0.0316, 5.012, 13.7])


def continuity_flex(
    age_yr: jnp.ndarray,
    log_total_mass: float = 10.0,
    bin_edges_gyr: jnp.ndarray | None = None,
    **ratio_kwargs,
) -> jnp.ndarray:
    """Non-parametric piecewise-constant SFH with flexible bin edges (ContinuityFlex, Leja+2019).

    Extends the continuity SFH by replacing fixed intermediate bins with N+1
    flex bins whose *widths* are derived from N log-SFR ratio parameters under a
    constant-mass-per-flex-bin constraint. Two anchor bins (young and old) are
    fixed in lookback-time extent; their amplitudes relative to the innermost
    flex bins are set by ``ratio_young`` and ``ratio_old``.

    Parameters
    ----------
    age_yr : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_total_mass : float, optional
        log10 total stellar mass formed [Msun]. Default 10.0.
    bin_edges_gyr : array_like, shape (3,), optional
        Anchor bin edges ``[t_young_end, t_old_start, t_max]`` [Gyr].
        Default: ``[0.0316, 5.012, 13.7]``.
    **ratio_kwargs
        ``ratio_young`` : float
            log10(SFR_young / SFR_flex[0]) [dimensionless]. Default 0.
        ``flex_0``, ``flex_1``, …, ``flex_{N-1}`` : float
            log10 SFR ratios that control flex bin widths [dimensionless]. The
            number of ``flex_*`` keys auto-sets N. Default: N=0 (1 flat flex bin).
        ``ratio_old`` : float
            log10(SFR_old / SFR_flex[N]) [dimensionless]. Default 0.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    Gradients flow through the SFR *amplitudes* (via ``ratio_young``,
    ``flex_*``, ``ratio_old`` and ``log_total_mass``). Flex bin *edge positions*
    depend on ``jnp.searchsorted`` and are not differentiable.

    N ratio parameters (``flex_0``...``flex_{N-1}``) produce N+1 flex time bins.
    The bin widths satisfy:

    .. math::

        \\Delta t_i = \\frac{T_{\\rm flex} \\prod_{j=0}^{i-1} r_j}
                            {\\sum_{k=0}^{N} \\prod_{j=0}^{k-1} r_j},
        \\quad r_j = 10^{(\\text{flex\\_}j)}, \\quad i = 0, \\ldots, N

    where :math:`T_{\\rm flex} = t_{\\rm old} - t_{\\rm young}` [yr] and the
    empty product equals 1. This enforces equal mass per flex bin:

    .. math::

        M_{\\rm bin} = \\frac{10^{m_*}}{N_{\\rm flex} + s_{\\rm young}
            \\Delta t_{\\rm young}/\\Delta t_0 +
            s_{\\rm old} \\Delta t_{\\rm old}/\\Delta t_N}

    where :math:`N_{\\rm flex} = N+1`, :math:`s = 10^{\\rm ratio}`, and the
    per-bin SFR values are:

    .. math::

        {\\rm SFR}_{{\\rm flex},i} = M_{\\rm bin}/\\Delta t_i, \\quad
        {\\rm SFR}_{\\rm young} = s_{\\rm young}\\,M_{\\rm bin}/\\Delta t_0, \\quad
        {\\rm SFR}_{\\rm old} = s_{\\rm old}\\,M_{\\rm bin}/\\Delta t_N.

    All of this is evaluated in log space (``log10_weighted_sum`` over the
    cumulative ratios and the anchor terms), so cumulative ratios of several
    hundred dex stay finite in float64 (about 38 dex in float32).

    **Formed mass is** ``10**log_total_mass`` **at every ratio.** Each bin's mass
    is carried in log space and turned into a rate over the width the integrand
    resolves (:func:`_rates_from_resolved_masses`). A bin narrower than
    ``_MIN_REL_BIN_WIDTH`` (1e-5) of its upper edge -- width exactly zero once
    it underflows -- cannot hold a piecewise-constant rate, so its mass is
    handed to the nearest resolved bin, which spreads it over its own width plus
    the unresolved widths it absorbs. Resolved bins are unchanged, and no rate is
    clipped: an unresolved bin is never the source of an overflowing exponent.
    Equal mass per flex bin then holds exactly only among resolved bins.

    Implements the ContinuityFlex prior of Leja et al. 2019 [1]_ as it is built
    in Prospector (Johnson et al. 2021 [2]_).

    References
    ----------
    .. [1] J. Leja et al., "How to Measure Galaxy Star Formation Histories, I.
       Parametric Models," ApJ, 876, 3 (2019). arXiv:1905.11997.
       https://doi.org/10.3847/1538-4357/ab133c
    .. [2] B. D. Johnson et al., "Stellar Population Inference from the
       Spectral Energy Distributions of Billions of Galaxies," ApJS, 254, 22
       (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> t = jnp.logspace(6.0, 10.14, 256)
    >>> sfr = continuity_flex(
    ...     t,
    ...     log_total_mass=10.5,
    ...     ratio_young=0.5,
    ...     flex_0=0.2,
    ...     flex_1=-0.1,
    ...     flex_2=0.0,
    ...     ratio_old=-0.3,
    ... )
    >>> sfr.shape
    (256,)
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = np.asarray(CFLEX_DEFAULT_ANCHOR_GYR)  # host: float() below

    t_young_end_yr = float(bin_edges_gyr[0]) * 1e9
    t_old_start_yr = float(bin_edges_gyr[1]) * 1e9
    t_max_yr = float(bin_edges_gyr[2]) * 1e9

    # Auto-detect n_flex_ratios from flex_* kwargs
    n_flex_ratios = sum(1 for k in ratio_kwargs if k.startswith("flex_"))

    # Flex bin widths via constant-mass-per-bin constraint, in log space
    log_dt_flex = _continuity_flex_log_widths(
        [ratio_kwargs.get(f"flex_{i}", 0.0) for i in range(n_flex_ratios)],
        t_old_start_yr - t_young_end_yr,
    )
    dt_flex_yr = pow10(log_dt_flex)
    n_flex_bins = log_dt_flex.shape[0]  # N+1 flex bins for N ratios

    ratio_young = ratio_kwargs.get("ratio_young", 0.0)
    ratio_old = ratio_kwargs.get("ratio_old", 0.0)
    log_dt0 = log_dt_flex[0]  # reference width for the young anchor
    log_dtN = log_dt_flex[-1]  # reference width for the old anchor
    dt_young_yr = t_young_end_yr  # young anchor bin: [0, t_young_end]
    dt_old_yr = t_max_yr - t_old_start_yr  # old anchor bin: [t_old_start, t_max]

    # Equal mass per flex bin; anchor masses scale by ratio * dt / dt_reference
    log_denom_mass = log10_weighted_sum(
        jnp.stack(
            [
                jnp.asarray(math.log10(float(n_flex_bins))),
                ratio_young + math.log10(dt_young_yr) - log_dt0,
                ratio_old + math.log10(dt_old_yr) - log_dtN,
            ]
        ),
        1.0,
    )
    log_mbin = log_total_mass - log_denom_mass

    # Per-bin formed mass [dex Msun], youngest to oldest: young, flex[0..N], old.
    # Mass is carried in log space, so a bin whose width underflows keeps a
    # finite mass here; its rate is never formed from the underflowed width.
    log_mass_bins = jnp.concatenate(
        [
            jnp.reshape(ratio_young + log_mbin + math.log10(dt_young_yr) - log_dt0, (1,)),
            jnp.broadcast_to(log_mbin, log_dt_flex.shape),
            jnp.reshape(ratio_old + log_mbin + math.log10(dt_old_yr) - log_dtN, (1,)),
        ]
    )

    # Bin edges: [0, t_young_end, flex_interior..., t_old_start, t_max] (yr)
    flex_interior_edges_yr = t_young_end_yr + jnp.cumsum(dt_flex_yr)
    all_edges_yr = jnp.concatenate(
        [
            jnp.array([0.0, t_young_end_yr]),
            flex_interior_edges_yr,
            jnp.array([t_max_yr]),
        ]
    )
    sfr_bins = _rates_from_resolved_masses(all_edges_yr, log_mass_bins)

    n_bins_total = n_flex_bins + 2  # young + flex bins + old
    return _piecewise_constant_sfr(age_yr, all_edges_yr, sfr_bins, n_bins_total)


def _continuity_flex_edges_yr(sfh_kwargs: dict, bin_edges_gyr=None) -> jnp.ndarray:
    """Lookback-time bin edges [yr] for :func:`continuity_flex` (#765).

    Mirrors the flex-edge derivation inside :func:`continuity_flex` (the same
    constant-mass-per-bin width construction) so the edges can be injected as
    exact knots into the DSPS SFH integrand. Kept as a separate helper rather
    than returned from ``continuity_flex`` to leave that function's SED output
    byte-identical. Returns ``n_flex+3`` ascending edges
    ``[0, t_young_end, flex_interior..., t_max]``; traced-safe (the flex
    interior edges depend on the ``flex_*`` ratio kwargs).
    """
    if bin_edges_gyr is None:
        bin_edges_gyr = np.asarray(CFLEX_DEFAULT_ANCHOR_GYR)  # host: float() below
    t_young_end_yr = float(bin_edges_gyr[0]) * 1e9
    t_old_start_yr = float(bin_edges_gyr[1]) * 1e9
    t_max_yr = float(bin_edges_gyr[2]) * 1e9
    n_flex_ratios = sum(1 for k in sfh_kwargs if k.startswith("flex_"))
    dt_flex_yr = pow10(
        _continuity_flex_log_widths(
            [sfh_kwargs.get(f"flex_{i}", 0.0) for i in range(n_flex_ratios)],
            t_old_start_yr - t_young_end_yr,
        )
    )
    flex_interior_edges_yr = t_young_end_yr + jnp.cumsum(dt_flex_yr)
    return jnp.concatenate(
        [jnp.array([0.0, t_young_end_yr]), flex_interior_edges_yr, jnp.array([t_max_yr])]
    )


def sfh_bin_edges_yr(fn, sfh_kwargs: dict) -> jnp.ndarray | None:
    """Lookback-time bin edges [yr] for a non-parametric SFH callable (#765).

    The piecewise-constant non-parametric SFHs (continuity / dirichlet /
    continuity_flex / psb_continuity_flex) have sharp bin-edge transitions. When
    the SFH is sampled onto a log-spaced integrand grid for DSPS, those edges fall
    *between* grid points, so DSPS interpolates across each step and smears the
    mass distribution; a resolution-insensitive 2-4.5 % optical residual vs
    Prospector (#765). Injecting these exact edges as knots makes the step
    representation exact at any resolution.

    Returns the ascending bin edges in [yr], or ``None`` for callables without
    a known edge set (which then keep the plain dense integrand).

    The post-starburst families are reached through :func:`psb_continuity_flex`:
    both registry entries name that callable, and it derives the whole ladder
    from ``tlast_gyr`` / ``tflex_gyr``. It had no branch here until #2184, so
    both entries were served the plain integrand and paid the smearing this
    function exists to remove: measured on a 256-node lookback grid, 1.1 % on
    the intrinsic SED and 0.0047 dex on the formed mass. :func:`psb_continuity`
    has no branch of its own, because no registry entry names it and its edges
    are not knowable from ``sfh_kwargs`` alone (the caller supplies the fixed
    ladder separately).
    """
    if fn is continuity_flex:
        return _continuity_flex_edges_yr(sfh_kwargs)
    if fn is psb_continuity_flex:
        return _psb_flex_edges_yr(sfh_kwargs)
    if fn is continuity or fn is dirichlet:
        return device_table(DEFAULT_BIN_EDGES_GYR) * 1e9
    return None


def _psb_flex_edges_yr(sfh_kwargs: dict) -> jnp.ndarray:
    """The ladder :func:`psb_continuity_flex` lays down, in [yr].

    Mirrors that function's own construction: ``[0, tlast_gyr]``, then
    ``n_flex`` equal-width flexible edges out to ``tflex_gyr``, then ``n_fixed``
    equal-width fixed edges out to the oldest age. Both counts are read the same
    way the shape function reads them, off the ``flex_*`` keyword names and off
    ``bin_edges_gyr``'s length, so the two cannot drift apart silently. The
    oldest edge itself mirrors the same ``bin_edges_gyr`` / ``age_universe_yr``
    / constant-13.7-Gyr precedence :func:`psb_continuity_flex` applies (#2645),
    so the knots injected into the dense CIC integrand never disagree with the
    bins the shape function actually laid down.
    """
    tlast_gyr = sfh_kwargs.get("tlast_gyr", 0.2)
    tflex_gyr = sfh_kwargs.get("tflex_gyr", 2.0)
    n_flex_bins = sum(1 for k in sfh_kwargs if k.startswith("flex_")) + 1

    bin_edges_gyr = sfh_kwargs.get("bin_edges_gyr")
    age_universe_yr = sfh_kwargs.get("age_universe_yr")
    if bin_edges_gyr is not None:
        bin_edges_gyr = jnp.asarray(bin_edges_gyr)
        n_fixed = bin_edges_gyr.shape[0] - 1
        max_age_gyr = bin_edges_gyr[-1]
    elif age_universe_yr is not None:
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED
        max_age_gyr = age_universe_yr / 1e9
    else:
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED
        max_age_gyr = PSB_FLEX_DEFAULT_MAX_AGE_GYR

    flex_edges_gyr = jnp.linspace(tlast_gyr, tflex_gyr, n_flex_bins + 1)[1:]
    fixed_edges_gyr = jnp.linspace(tflex_gyr, max_age_gyr, n_fixed + 1)[1:]
    return jnp.concatenate([jnp.array([0.0, tlast_gyr]), flex_edges_gyr, fixed_edges_gyr]) * 1e9


def continuity_flex_prior_logp(
    logsfr_ratio_young: float,
    logsfr_ratios: jnp.ndarray,
    logsfr_ratio_old: float,
    df: float = 2.0,
    scale: float = 0.3,
) -> jnp.ndarray:
    """Student-t smoothness prior on all ContinuityFlex log-SFR ratios (Leja+2019).

    Applies an independent Student-t(df, 0, scale) prior to each of the
    ``ratio_young``, flex, and ``ratio_old`` parameters, penalizing large
    deviations from a flat (constant) SFH.

    Parameters
    ----------
    logsfr_ratio_young : float
        log10(SFR_young / SFR_flex[0]) [dimensionless].
    logsfr_ratios : array_like, shape (N,)
        log10 flex bin SFR ratios [dimensionless].
    logsfr_ratio_old : float
        log10(SFR_old / SFR_flex[N]) [dimensionless].
    df : float, optional
        Degrees of freedom. Default 2.
    scale : float, optional
        Scale parameter [dex]. Default 0.3 (same as :func:`continuity_prior_logp`).

    Returns
    -------
    logp : scalar
        Total log-probability [dimensionless], summed over all ratios.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.scipy.stats.t``.
    **Gradient-safe**: yes, differentiable w.r.t. all ratio arguments.

    .. math::

        \\log p = \\sum_{r \\in \\{r_{\\rm young},\\, r_{\\rm flex},\\, r_{\\rm old}\\}}
            \\log t_\\nu(r \\mid 0,\\, \\sigma)

    where :math:`t_\\nu` is the Student-t density with :math:`\\nu = \\mathtt{df}`
    degrees of freedom and :math:`\\sigma = \\mathtt{scale}` [dex].

    References
    ----------
    .. [1] J. Leja et al., "How to Measure Galaxy Star Formation Histories, I.
       Parametric Models," ApJ, 876, 3 (2019). arXiv:1905.11997.
       https://doi.org/10.3847/1538-4357/ab133c
    """
    from jax.scipy.stats import t as student_t

    all_ratios = jnp.concatenate(
        [
            jnp.array([logsfr_ratio_young]),
            jnp.atleast_1d(jnp.asarray(logsfr_ratios)),
            jnp.array([logsfr_ratio_old]),
        ]
    )
    return jnp.sum(student_t.logpdf(all_ratios, df, loc=0.0, scale=scale))
