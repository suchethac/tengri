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
younger than the first edge take the youngest bin's rate, because
``psb_suess2022`` anchors its edges at 0.3 Gyr and relies on it.

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

import jax.numpy as jnp
import numpy as np

# Default bin edges in Gyr (8 edges = 7 bins), log-spaced from 30 Myr to 13.7 Gyr.
DEFAULT_BIN_EDGES_GYR = jnp.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7])
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

    Ages *below* the first edge are deliberately clamped into the youngest bin.
    ``psb_suess2022`` starts its edges at 0.3 Gyr and relies on that.

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


# ── Continuity SFH (Leja+2019) ────────────────────────────────────


def continuity(
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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

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
        bin_edges_gyr = DEFAULT_BIN_EDGES_GYR

    n_bins = bin_edges_gyr.shape[0] - 1  # len() raises ConcretizationTypeError under JIT

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
    that no bin extends into the future. For low redshifts the youngest two
    bins are fixed at 30 Myr and 100 Myr; interior bins are log-spaced to
    90% of the universe age; the oldest bin spans 90–100% of the universe age.

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

    Implements Prospector ``zred_to_agebins_pbeta`` (Johnson et al. 2021
    [1]_), with two changes: uses tengri's Planck 2018 cosmology instead of
    WMAP9, and returns edges in Gyr rather than log10(yr).

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
        log_amin = 6.0  # 1 Myr
        log_edges_inner = list(np.linspace(log_amin, log_90pct, n_bins - 1))
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
        - ``ratio_old_0``, ``ratio_old_1``, ...: ratios among the old fixed
          bins. The ratio between the OLDEST flex bin and the youngest fixed
          bin is pinned at 0 (they share an SFR), so ``n_fixed`` bins take
          ``n_fixed - 1`` of these.

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
        bin_edges_gyr = DEFAULT_BIN_EDGES_GYR[2:]  # [0.3, 1.0, 3.0, 6.0, 13.7]

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
    ratio_old = jnp.array(
        [ratio_kwargs.get(f"ratio_old_{i}", 0.0) for i in range(n_fixed_bins - 1)]
    )
    log_sfr_old = jnp.concatenate([jnp.cumsum(ratio_old[::-1])[::-1], jnp.array([0.0])])

    # Flex bins: the OLDEST flex bin is tied to the innermost old bin (ratio
    # pinned at 0), and each ``flex_i`` steps log-SFR from flex bin i to bin
    # i+1. With no ``flex_*`` ratios this collapses to the single flex bin at
    # ``log_sfr_old[0]``, bit-identical to the one-flex-bin model this replaces.
    flex_ratios = jnp.array([ratio_kwargs.get(f"flex_{i}", 0.0) for i in range(n_flex_ratios)])
    log_sfr_flex = (
        jnp.concatenate([jnp.cumsum(flex_ratios[::-1])[::-1], jnp.array([0.0])]) + log_sfr_old[0]
    )
    log_sfr_young = log_sfr_flex[0] + ratio_young

    log_sfr_bins = jnp.concatenate([jnp.array([log_sfr_young]), log_sfr_flex, log_sfr_old])

    # Normalize to total mass
    bin_widths_yr = jnp.diff(all_edges_gyr) * 1e9
    sfr_unnorm = 10.0**log_sfr_bins
    mass_unnorm = jnp.sum(sfr_unnorm * bin_widths_yr)
    sfr_bins_norm = sfr_unnorm * (10.0**log_total_mass) / (mass_unnorm + 1e-30)

    # Piecewise-constant lookup
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
        fixed bins are equal-width by construction. Default: 3 bins out to
        13.7 Gyr.
    **ratio_kwargs
        As :func:`psb_continuity`: ``ratio_young``, ``flex_0`` ...
        ``flex_{n_flex-2}``, and ``ratio_old_0`` ... ``ratio_old_{n_fixed-2}``.

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
    from 0.5 to 5.0 Gyr, so the edges cross and
    :func:`jax.numpy.searchsorted` is evaluated on a non-ascending array.
    Deriving the fixed bins from ``tflex_gyr`` removes the ordering constraint
    instead of asking the user to respect it.

    Implements the post-starburst-optimized non-parametric SFH of Suess et al.
    2022 [1]_, on the Prospector continuity machinery (Johnson et al. 2021
    [2]_), with the flexible zone resolved into equal-width bins. The step
    between the oldest flex bin and the youngest fixed bin is pinned at 0: the
    two share an SFR.

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
    ... )
    >>> sfr.shape
    (256,)
    """
    if bin_edges_gyr is None:
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED
        max_age_gyr = PSB_FLEX_DEFAULT_MAX_AGE_GYR
    else:
        n_fixed = bin_edges_gyr.shape[0] - 1
        max_age_gyr = bin_edges_gyr[-1]
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

# Anchor bin edges [t_young_end_gyr, t_old_start_gyr, t_max_gyr].
# ContinuityFlex anchor defaults:
#   young bin [0, 10^7.5 yr] = [0, 31.6 Myr], old bin [10^9.7, 10^10.136 yr] = [5.01, 13.7 Gyr].
CFLEX_DEFAULT_ANCHOR_GYR = np.array([0.0316, 5.012, 13.7])


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
        bin_edges_gyr = CFLEX_DEFAULT_ANCHOR_GYR

    t_young_end_yr = float(bin_edges_gyr[0]) * 1e9
    t_old_start_yr = float(bin_edges_gyr[1]) * 1e9
    t_max_yr = float(bin_edges_gyr[2]) * 1e9

    # Auto-detect n_flex_ratios from flex_* kwargs
    n_flex_ratios = sum(1 for k in ratio_kwargs if k.startswith("flex_"))

    # Flex bin widths via constant-mass-per-bin constraint
    if n_flex_ratios > 0:
        flex_ratios_log = jnp.array(
            [ratio_kwargs.get(f"flex_{i}", 0.0) for i in range(n_flex_ratios)]
        )
        sfr_ratios = jnp.power(10.0, flex_ratios_log)
        cumprod_prefixed = jnp.concatenate([jnp.ones(1), jnp.cumprod(sfr_ratios)])
    else:
        cumprod_prefixed = jnp.ones(1)

    n_flex_bins = cumprod_prefixed.shape[0]  # N+1 flex bins for N ratios
    T_flex_yr = t_old_start_yr - t_young_end_yr
    denom_flex = jnp.sum(cumprod_prefixed)
    dt_flex_yr = T_flex_yr * cumprod_prefixed / (denom_flex + 1e-30)

    dt0_yr = dt_flex_yr[0]  # first flex bin width (reference for young anchor)
    dtN_yr = dt_flex_yr[-1]  # last flex bin width (reference for old anchor)
    dt_young_yr = t_young_end_yr  # young anchor bin: [0, t_young_end]
    dt_old_yr = t_max_yr - t_old_start_yr  # old anchor bin: [t_old_start, t_max]

    syoung = jnp.power(10.0, ratio_kwargs.get("ratio_young", 0.0))
    sold = jnp.power(10.0, ratio_kwargs.get("ratio_old", 0.0))

    # Equal mass per flex bin; anchor masses scale by ratio * dt / dt_reference
    denom_mass = (
        float(n_flex_bins)
        + syoung * dt_young_yr / (dt0_yr + 1e-30)
        + sold * dt_old_yr / (dtN_yr + 1e-30)
    )
    mbin = jnp.power(10.0, log_total_mass) / (denom_mass + 1e-30)

    sfr_flex = mbin / (dt_flex_yr + 1e-30)  # (n_flex_bins,)
    sfr_young = syoung * mbin / (dt0_yr + 1e-30)  # scalar
    sfr_old = sold * mbin / (dtN_yr + 1e-30)  # scalar

    # All bins in order from youngest to oldest: young, flex[0..N], old
    sfr_bins = jnp.concatenate([jnp.array([sfr_young]), sfr_flex, jnp.array([sfr_old])])

    # Bin edges: [0, t_young_end, flex_interior..., t_old_start, t_max] (yr)
    flex_interior_edges_yr = t_young_end_yr + jnp.cumsum(dt_flex_yr)
    all_edges_yr = jnp.concatenate(
        [
            jnp.array([0.0, t_young_end_yr]),
            flex_interior_edges_yr,
            jnp.array([t_max_yr]),
        ]
    )

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
        bin_edges_gyr = CFLEX_DEFAULT_ANCHOR_GYR
    t_young_end_yr = float(bin_edges_gyr[0]) * 1e9
    t_old_start_yr = float(bin_edges_gyr[1]) * 1e9
    t_max_yr = float(bin_edges_gyr[2]) * 1e9
    n_flex_ratios = sum(1 for k in sfh_kwargs if k.startswith("flex_"))
    if n_flex_ratios > 0:
        flex_ratios_log = jnp.array(
            [sfh_kwargs.get(f"flex_{i}", 0.0) for i in range(n_flex_ratios)]
        )
        sfr_ratios = jnp.power(10.0, flex_ratios_log)
        cumprod_prefixed = jnp.concatenate([jnp.ones(1), jnp.cumprod(sfr_ratios)])
    else:
        cumprod_prefixed = jnp.ones(1)
    T_flex_yr = t_old_start_yr - t_young_end_yr
    dt_flex_yr = T_flex_yr * cumprod_prefixed / (jnp.sum(cumprod_prefixed) + 1e-30)
    flex_interior_edges_yr = t_young_end_yr + jnp.cumsum(dt_flex_yr)
    return jnp.concatenate(
        [jnp.array([0.0, t_young_end_yr]), flex_interior_edges_yr, jnp.array([t_max_yr])]
    )


def sfh_bin_edges_yr(fn, sfh_kwargs: dict) -> jnp.ndarray | None:
    """Lookback-time bin edges [yr] for a non-parametric SFH callable (#765).

    The piecewise-constant non-parametric SFHs (continuity / dirichlet /
    psb_continuity / continuity_flex) have sharp bin-edge transitions. When the
    SFH is sampled onto a log-spaced integrand grid for DSPS, those edges fall
    *between* grid points, so DSPS interpolates across each step and smears the
    mass distribution; a resolution-insensitive 2-4.5 % optical residual vs
    Prospector (#765). Injecting these exact edges as knots makes the step
    representation exact at any resolution.

    Returns the ascending bin edges in [yr], or ``None`` for callables without
    a known edge set (which then keep the plain dense integrand).
    """
    if fn is continuity_flex:
        return _continuity_flex_edges_yr(sfh_kwargs)
    if fn is psb_continuity:
        tlast_gyr = sfh_kwargs.get("tlast_gyr", 0.5)
        tflex_gyr = sfh_kwargs.get("tflex_gyr", 2.0)
        old_edges_gyr = DEFAULT_BIN_EDGES_GYR[2:]  # [0.3, 1.0, 3.0, 6.0, 13.7]
        return jnp.concatenate([jnp.array([0.0, tlast_gyr, tflex_gyr]), old_edges_gyr[1:]]) * 1e9
    if fn is continuity or fn is dirichlet:
        return DEFAULT_BIN_EDGES_GYR * 1e9
    return None


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
