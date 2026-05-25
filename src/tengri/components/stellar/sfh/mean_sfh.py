# SPDX-License-Identifier: BSD-3-Clause
"""Smooth parametric mean star formation history components.

The GP x(t) has zero mean, so the overall SFH shape comes from these
functions. The full SFH is: SFR(t) = mean(t) * exp(x(t) - K(0)/2),
where -K(0)/2 is the lognormal correction preserving the linear-SFR mean.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.

Models
------
Canonical names (short name alias in parentheses):

- **truncated_skewnormal** (tsnorm): Bellstedt+2020, Robotham+2020 snorm_trunc.
  Most flexible smooth model — 5 params: peak location, width, skew, truncation.
- **skewnormal** (snorm): ``truncated_skewnormal`` without truncation (4 params).
- **gaussian** (norm): ``skewnormal`` with skew=0 (3 params).
- **lognormal** (lnorm): Gaussian in log10(age) space (3 params).
- **dpl** (canonical): Carnall+2018 BAGPIPES parameterization with log_peak_sfr (4 params).
- **double_powerlaw**: Low-level implementation used by ``dpl``.
- **constant** (const): flat SFR between start and end times (3 params).
- **exponential** (exp): declining exponential from start (3 params).
- **delayed_exponential** (dexp): peaks at start + tau (3 params).
- **declining_exponential** (tau): FSPS/bagpipes tau model in lookback time (3 params).
- **triweight_burst**: compact triweight kernel in log-age for burst component.
- **spline**: N-node monotone cubic (PCHIP) spline in log-age space. Nodes are
  static (set at JIT-compile time); SFR values are free parameters. Use directly
  (not via the registry — array node inputs don't fit the scalar-kwarg registry).
- **snorm_burst**: skew-normal SFH + flat recent burst.
- **snorm_trunc_burst** (tsnorm_burst): truncated skew-normal + flat recent burst.

References
----------
- Bellstedt+2020 (arXiv:2005.11917): snorm, snorm_trunc parameterizations.
- Robotham+2020 (arXiv:2002.06980): ProSpect SFH models including spline and burst variants.
- Carnall+2018: double power law (BAGPIPES).
- Zacharegkas+2025 (arXiv:2506.19919): triweight burst model.
"""

import jax
import jax.numpy as jnp

# Maximum age of the universe in years — hardcoded, not fittable.
AGEMAX_YR = 14e9

# Precomputed constant for erfc-based CDF: 1/sqrt(2).
_INV_SQRT2 = 1.0 / jnp.sqrt(2.0)


# ── Shared helpers ────────────────────────────────────────────────


def _clamp_age(t_lookback: jnp.ndarray) -> jnp.ndarray:
    """Clamp lookback time to [1e5, AGEMAX_YR] to avoid numerical issues.

    Parameters
    ----------
    t_lookback : array
        Lookback time (yr).

    Returns
    -------
    array
        Clamped lookback time.
    """
    return jnp.clip(t_lookback, 1e5, AGEMAX_YR)


def _skewed_gaussian_kernel(
    age: jnp.ndarray, peak_lbt: float, width: float, skew: float
) -> jnp.ndarray:
    """Skewed Gaussian kernel shared by snorm, tsnorm, norm.

    Implements the Robotham+2020 / Bellstedt+2020 formulation:
        X = (age - peak_lbt) / width
        Y = X * exp(skew * arcsinh(X))
        kernel = exp(-Y^2 / 2)

    Parameters
    ----------
    age : array
        Lookback time (yr), should be clamped.
    peak_lbt : float
        Peak lookback time (yr).
    width : float
        Width of the Gaussian (yr).
    skew : float
        Skewness parameter. 0 = symmetric, >0 skews toward older ages.

    Returns
    -------
    array
        Unnormalized kernel values.
    """
    x = (age - peak_lbt) / width
    y = x * jnp.exp(skew * jnp.arcsinh(x))
    return jnp.exp(-(y**2) / 2.0)


# ── Smooth SFH models — all take t_lookback (yr), return SFR (Msun/yr)


def truncated_skewnormal(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
    skew: float,
    trunc: float,
) -> jnp.ndarray:
    """Truncated skew-normal star formation history (Bellstedt+2020).

    The most flexible smooth SFH model: a skewed Gaussian kernel multiplied
    by a normal CDF truncation factor that smoothly suppresses star formation
    at recent times (young lookback times).

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR [Msun/yr].
    peak_lbt : float
        Peak lookback time [yr].
    width : float
        Gaussian width parameter [yr]. **SSP grid aliasing**: the forward
        model interpolates ``SFR(t)`` at SSP grid points (a point-sample,
        not a bin-integral), so a ``width`` narrower than the local SSP
        grid spacing at the burst peak produces a non-physical staircase
        as the peak crosses adjacent grid boundaries. Rough minimum:
        ``width ≳ 0.3 Gyr`` at peaks < 2 Gyr, ``≳ 0.6 Gyr`` past 5 Gyr.
        See issue #299. ``SEDModel.build`` emits a
        :class:`SFHBurstAliasingWarning` when the chosen ``width`` is
        too narrow for the SSP grid.
    skew : float
        Skewness parameter [dimensionless]. 0 = symmetric, >0 skews toward older ages.
    trunc : float
        Truncation sharpness [dimensionless]. Larger values produce sharper truncation.
        Typical range: 1-10.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.lax.erfc``.

    **Gradient-safe**: yes — differentiable everywhere except at SFR=0
    (where gradient is ill-defined but finite).

    The SFH is:

    .. math::

        \\mathrm{SFR}(t) = 10^{\\log_{\\rm peak}} \\, K(t) \\, T(t)

    where :math:`K(t)` is the skewed Gaussian kernel and :math:`T(t)` is the
    truncation factor:

    .. math::

        K(t) = \\exp\\left( -\\frac{Y(t)^2}{2} \\right), \\quad
        Y(t) = \\frac{t - t_{\\rm peak}}{w} \\exp(\\gamma \\, \\mathrm{arcsinh}(Y))

    .. math::

        T(t) = \\frac{1}{2} \\mathrm{erfc}\\left( \\frac{t - t_{\\rm peak}}{w \\cdot f_{\\rm trunc} \\sqrt{2}} \\right)  # noqa: E501

    where :math:`w` is width [yr], :math:`\\gamma` is skewness [dimensionless],
    and :math:`f_{\\rm trunc}` is the truncation sharpness [dimensionless].

    **Approximation**: The skewness is implemented via the Robotham+2020
    formulation (arcsinh-based) rather than the classical Owen's lambda
    transformation. This provides better numerical stability and JAX compatibility.

    References
    ----------
    .. [1] S. Bellstedt et al., "Galaxy And Mass Assembly (GAMA): a forensic SED
       reconstruction of the cosmic star formation history and metallicity evolution
       by galaxy type," MNRAS, 498, 5581 (2020). arXiv:2005.11917.
       https://doi.org/10.1093/mnras/staa2620
    .. [2] A. S. G. Robotham et al., "ProSpect: generating spectral energy
       distributions with complex star formation and metallicity histories,"
       MNRAS, 495, 905 (2020). arXiv:2002.06980.
       https://doi.org/10.1093/mnras/staa1116

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import truncated_skewnormal
    >>> t = jnp.linspace(0.0, 13.7e9, 100)
    >>> sfr = truncated_skewnormal(
    ...     t, log_peak_sfr=1.0, peak_lbt=5e9, width=2e9, skew=0.0, trunc=5.0
    ... )
    >>> sfr.shape
    (100,)
    """
    age = _clamp_age(t_lookback)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    # Truncation: 1 - Phi(x) = 0.5 * erfc(x / sqrt(2))
    # Using jax.lax.erfc directly avoids scipy.stats dispatch overhead,
    # producing a simpler XLA graph (~6x faster CDF inside fused kernels).
    x = (age - peak_lbt) / (width * trunc)
    trunc_factor = 0.5 * jax.lax.erfc(x * _INV_SQRT2)
    sfr = peak_sfr * kernel * trunc_factor
    return jnp.maximum(sfr, 0.0)


# Short alias registered in SFH_REGISTRY
tsnorm = truncated_skewnormal


def skewnormal(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
    skew: float,
) -> jnp.ndarray:
    """Skew-normal SFH (Robotham+2020).

    Like truncated_skewnormal but without truncation.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR [Msun/yr].
    peak_lbt : float
        Peak lookback time [yr].
    width : float
        Gaussian width [yr].
    skew : float
        Skewness parameter [dimensionless]. 0 = symmetric, >0 skews toward older ages.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import skewnormal
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = skewnormal(t, log_peak_sfr=1.5, peak_lbt=3e9, width=1e9, skew=1.0)
    >>> sfr.shape
    (64,)
    """
    age = _clamp_age(t_lookback)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    return jnp.maximum(peak_sfr * kernel, 0.0)


# Short alias registered in SFH_REGISTRY
snorm = skewnormal


def gaussian(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
) -> jnp.ndarray:
    """Gaussian (normal) SFH — skewnormal with skew=0.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR [Msun/yr].
    peak_lbt : float
        Peak lookback time [yr].
    width : float
        Gaussian width [yr].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import gaussian
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = gaussian(t, log_peak_sfr=1.5, peak_lbt=3e9, width=1e9)
    >>> sfr.shape
    (64,)
    """
    return skewnormal(t_lookback, log_peak_sfr, peak_lbt, width, skew=0.0)


# Short alias registered in SFH_REGISTRY
norm = gaussian


def lognormal(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
) -> jnp.ndarray:
    """Log-normal SFH — Gaussian in log10(age) space.

    The mode in log10(age) space is peak_lbt, but the peak in linear
    time is shifted: t_peak_linear = peak_lbt * 10^{-w^2 * ln(10)}.
    For w=0.3, this is a ~15% shift toward younger ages. peak_lbt is
    best interpreted as the median lookback time, not the linear peak.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR [Msun/yr].
    peak_lbt : float
        Peak lookback time [yr]. Converted to log10 internally.
    width : float
        Width in log10(age) space [dex].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import lognormal
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = lognormal(t, log_peak_sfr=1.5, peak_lbt=3e9, width=0.3)
    >>> sfr.shape
    (64,)
    """
    age = _clamp_age(t_lookback)
    peak_sfr = 10.0**log_peak_sfr
    log_age = jnp.log10(age)
    log_peak = jnp.log10(jnp.maximum(peak_lbt, 1e5))
    exponent = -0.5 * ((log_age - log_peak) / width) ** 2
    return jnp.maximum(peak_sfr * jnp.exp(exponent), 0.0)


# Short alias registered in SFH_REGISTRY
lnorm = lognormal


def double_powerlaw(
    t_lookback: jnp.ndarray, alpha: float, beta: float, tau: float, norm: float
) -> jnp.ndarray:
    """Double power law star formation history (Carnall+2018, BAGPIPES).

    A flexible two-parameter SFH model that peaks near a characteristic timescale
    and can smoothly transition between rising and declining phases.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    alpha : float
        Falling slope exponent [dimensionless]. Controls the decline from peak
        to present. Larger alpha = steeper decline. Typical range: 0.5-4.
    beta : float
        Rising slope exponent [dimensionless]. Controls the rise from early times
        to peak. Larger beta = steeper rise. Typical range: 0.3-3.
    tau : float
        Turnover timescale [yr]. Approximately when SFR peaks (in cosmic time).
    norm : float
        Normalization factor [Msun/yr]. Note: this controls overall amplitude,
        not stellar mass. :math:`M_\\star = \\int \\mathrm{SFR}(t) \\, dt` is derived.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere for positive tau.

    The double power law SFH is:

    .. math::

        \\mathrm{SFR}(t_{\\rm cosmic}) = \\frac{n}{(t_{\\rm cosmic}/\\tau)^\\alpha + (t_{\\rm cosmic}/\\tau)^{-\\beta}}  # noqa: E501

    where :math:`t_{\\rm cosmic} = t_{\\mathrm{H}} - t_{\\rm lookback}` is cosmic time
    since the Big Bang, :math:`t_{\\mathrm{H}}` is the age of the universe,
    :math:`\\tau` is the turnover timescale, and :math:`\\alpha`, :math:`\\beta`
    are the falling and rising slopes [dimensionless].

    In **cosmic time**: :math:`\\alpha` controls the declining phase (after peak),
    :math:`\\beta` controls the rising phase (before peak).

    In **lookback time** (as plotted): :math:`\\alpha` controls the RIGHT side
    (early universe, large lookback), :math:`\\beta` controls the LEFT side
    (near present, small lookback).

    References
    ----------
    .. [1] A. C. Carnall et al., "Inferring the star formation histories of massive
       quiescent galaxies with BAGPIPES: evidence for multiple quenching mechanisms,"
       MNRAS, 480, 4379 (2018). arXiv:1712.04452.
       https://doi.org/10.1093/mnras/sty2169
    .. [2] P. S. Behroozi, R. H. Wechsler, C. Conroy, "The Average Star Formation
       Histories of Galaxies in Dark Matter Halos from z=0-8," ApJ, 770, 57 (2013).
       arXiv:1207.6105. https://doi.org/10.1088/0004-637X/770/1/57

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import double_powerlaw
    >>> t = jnp.linspace(1e6, 13.7e9, 100)
    >>> sfr = double_powerlaw(t, alpha=1.5, beta=2.0, tau=3e9, norm=10.0)
    >>> sfr.shape
    (100,)
    """
    x = t_lookback / tau
    return norm / (x**alpha + x ** (-beta))


def dpl(
    t_lookback: jnp.ndarray,
    alpha: float,
    beta: float,
    tau: float,
    log_peak_sfr: float,
) -> jnp.ndarray:
    """Double power law with log-peak SFR parameterization (canonical).

    Registry-compatible wrapper around the Carnall+2018 double power law.
    Uses log10(peak_sfr) instead of linear normalization for consistency
    with other parametric SFH models in the registry.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    alpha : float
        Falling slope exponent [dimensionless]. Typical range: 0.5-4.
    beta : float
        Rising slope exponent [dimensionless]. Typical range: 0.3-3.
    tau : float
        Turnover timescale [yr].
    log_peak_sfr : float
        log10 of peak SFR [Msun/yr].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    This function wraps :func:`double_powerlaw` with ``norm = 10**log_peak_sfr``.
    See :func:`double_powerlaw` for physics details.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import dpl
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = dpl(t, alpha=1.5, beta=0.5, tau=3e9, log_peak_sfr=1.5)
    >>> sfr.shape
    (64,)
    """
    peak_sfr = 10.0**log_peak_sfr
    x = t_lookback / tau
    return peak_sfr / (x**alpha + x ** (-beta))


def constant(
    t_lookback: jnp.ndarray,
    log_sfr: float,
    start: float = 0.0,
    end: float = AGEMAX_YR,
) -> jnp.ndarray:
    """Constant SFR between start and end lookback times.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_sfr : float
        log10 of constant SFR [Msun/yr].
    start : float
        Younger lookback boundary [yr]. Default 0 (present). Maps from user-facing
        ``sfh_const_end_gyr``.
    end : float
        Older lookback boundary [yr]. Default AGEMAX_YR. Maps from user-facing
        ``sfh_const_start_gyr``.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], flat between start and end, zero outside.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.where`` and element-wise operations.

    Internal convention: ``start <= t_lookback <= end`` (both in lookback time).
    The user-facing API names are reversed for chronological intuition.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import constant
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = constant(t, log_sfr=1.0, start=5e8, end=5e9)
    >>> sfr.shape
    (64,)
    """
    sfr = 10.0**log_sfr
    mask = (t_lookback >= start) & (t_lookback <= end)
    return jnp.where(mask, sfr, 0.0)


def exponential(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    tau: float,
    start: float = 0.0,
) -> jnp.ndarray:
    """Declining exponential SFH.

    SFR(t) = peak_sfr * exp(-(t - start) / tau) for t >= start, 0 otherwise.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR at start [Msun/yr].
    tau : float
        e-folding timescale [yr].
    start : float
        Start lookback time [yr]. Default 0 (present).

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives for exponential and masking.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import exponential
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = exponential(t, log_peak_sfr=2.0, tau=2e9, start=1e8)
    >>> sfr.shape
    (64,)
    """
    peak_sfr = 10.0**log_peak_sfr
    dt = t_lookback - start
    sfr = peak_sfr * jnp.exp(-dt / tau)
    return jnp.where(dt >= 0, sfr, 0.0)


def delayed_exponential(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    tau: float,
    start: float = 0.0,
) -> jnp.ndarray:
    """Delayed exponential SFH: SFR peaks at start + tau.

    SFR(t) = peak_sfr * (dt/tau) * exp(-dt/tau + 1) for t >= start.
    The +1 in the exponent ensures SFR(start + tau) = peak_sfr.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR [Msun/yr]. Peak occurs at t = start + tau.
    tau : float
        Timescale [yr]. Peak is at start + tau.
    start : float
        Start lookback time [yr]. Default 0 (present).

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import delayed_exponential
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = delayed_exponential(t, log_peak_sfr=2.0, tau=2e9, start=1e8)
    >>> sfr.shape
    (64,)
    """
    peak_sfr = 10.0**log_peak_sfr
    dt = t_lookback - start
    ratio = dt / tau
    sfr = peak_sfr * ratio * jnp.exp(-ratio + 1.0)
    return jnp.where(dt >= 0, jnp.maximum(sfr, 0.0), 0.0)


def declining_exponential(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    tau: float,
    age: float,
) -> jnp.ndarray:
    """Declining tau SFH in lookback time — matches FSPS sfh=1 / bagpipes 'exponential'.

    In cosmic time T, the standard tau model is SFR(T) = peak * exp(-T/tau) for
    0 <= T <= age. Converting T = age - t_lb gives:

        SFR(t_lb) = peak * exp(-(age - t_lb) / tau)  for  0 <= t_lb <= age

    The SFR *increases* going back in lookback time (galaxy formed with highest SFR,
    declining to the present). This is opposite to ``exponential``.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of peak SFR at galaxy formation [Msun/yr] (at t_lb = age).
    tau : float
        e-folding timescale [yr]. Larger tau = slower decline.
    age : float
        Galaxy age [yr] = lookback time of galaxy formation.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives for exponential and masking.
    """
    peak_sfr = 10.0**log_peak_sfr
    dt = age - t_lookback  # cosmic time elapsed since galaxy formation
    sfr = peak_sfr * jnp.exp(-dt / tau)
    return jnp.where((t_lookback >= 0) & (t_lookback <= age), sfr, 0.0)


def constant_then_exponential(
    t_lookback: jnp.ndarray,
    log_sfr: float,
    tau: float,
    quench_age: float,
    age: float,
) -> jnp.ndarray:
    """Constant SFR followed by exponential decline — 'quenching at time T'.

    Constant SFR from formation (age) until quench_age, then exponential
    decline from quench_age to present. In lookback time:

        SFR(t_lb) = SFR_0 * exp(-(quench_age - t_lb) / tau)  for t_lb < quench_age
        SFR(t_lb) = SFR_0                                     for quench_age <= t_lb <= age

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_sfr : float
        log10 of constant SFR before quenching [Msun/yr].
    tau : float
        e-folding decline timescale [yr] after quenching.
    quench_age : float
        Lookback time when quenching began [yr].
    age : float
        Galaxy age [yr] = lookback time of formation. Must be > quench_age.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — uses conditional masking via ``jnp.where``.
    """
    sfr_0 = 10.0**log_sfr
    dt_quench = quench_age - t_lookback
    declining = sfr_0 * jnp.exp(-dt_quench / tau)
    sfr = jnp.where(t_lookback >= quench_age, sfr_0, declining)
    return jnp.where((t_lookback >= 0) & (t_lookback <= age), sfr, 0.0)


def triweight_burst(
    t_lookback: jnp.ndarray,
    log_tpeak_myr: float,
    log_tmax_myr: float,
) -> jnp.ndarray:
    """Triweight burst kernel in log-age space (Zacharegkas+2025).

    A compact-support kernel for modeling starburst episodes. The kernel
    is smooth, has finite support in log-age space, and is designed for
    composition with smooth SFH models via mass-fraction mixing.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_tpeak_myr : float
        log10 of burst peak time [Myr]. Center of the kernel in log-age space.
    log_tmax_myr : float
        log10 of burst duration [Myr]. Controls the kernel half-width.
        The kernel has support roughly ±:math:`3 \times 10^{\\log_{\\mathrm{tmax}}}` Myr.

    Returns
    -------
    ndarray, shape (n_age,)
        Unnormalized burst shape [dimensionless], non-negative. Integrates to ~1.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    This is a **shape-only** function (unitless, not in Msun/yr). The burst
    amplitude is set by the burst mixture fraction in the composition step.

    The triweight kernel in normalized variable :math:`u = x/3` is:

    .. math::

        K(u) = \\frac{35}{96} (1 - u^2)^3, \\quad |u| < 1

    where :math:`x` is the normalized time offset from peak, defined as  # noqa: E501
    :math:`(\\log_{10}(t/\\mathrm{Myr}) - \\log_{10}(t_{\\rm peak}/\\mathrm{Myr})) / \\log_{10}(t_{\\rm max}/\\mathrm{Myr})`.

    The kernel has compact support (finite extent) and is :math:`C^{\\infty}` smooth.
    It is superior to Gaussian kernels for burst modeling because it avoids
    the extended low-level wings that affect neighboring age bins.

    References
    ----------
    .. [1] G. Zacharegkas, A. Hearin, and A. Benson, "Bayesian Posteriors with
       Stellar Population Synthesis on GPUs," The Open Journal of Astrophysics,
       8 (2025). arXiv:2506.19919. https://doi.org/10.33232/001c.151255

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import triweight_burst
    >>> t = jnp.linspace(1e6, 13.7e9, 100)
    >>> kernel = triweight_burst(t, log_tpeak_myr=2.0, log_tmax_myr=1.0)
    >>> kernel.shape
    (100,)
    """
    log_age_myr = jnp.log10(t_lookback / 1e6)  # yr → Myr in log10
    x = (log_age_myr - log_tpeak_myr) / jnp.maximum(log_tmax_myr, 0.01)
    # Triweight kernel: (35/96)(1 - (x/3)^2)^3, compact support |x| < 3
    u = x / 3.0
    kernel = (35.0 / 96.0) * jnp.maximum(1.0 - u**2, 0.0) ** 3
    return kernel


# ── Historical SFH parameterizations ────────────────────────────


def delayed_tau(t_lookback: jnp.ndarray, tau: float, norm: float) -> jnp.ndarray:
    """Delayed-tau SFH: SFR(t) = norm * t * exp(-t/tau). Peaks at t=tau.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    tau : float
        Timescale [yr].
    norm : float
        Normalization factor [Msun/yr].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import delayed_tau
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = delayed_tau(t, tau=2e9, norm=10.0)
    >>> sfr.shape
    (64,)
    """
    return norm * t_lookback * jnp.exp(-t_lookback / tau)


def psb_wild2020(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    age: float,
    tau: float,
    burstage: float,
    alpha: float,
    beta: float,
    fburst: float,
) -> jnp.ndarray:
    """Post-starburst SFH (Wild+2020).

    Two-component model: declining exponential for the old stellar population
    plus a double power law for the recent burst episode. Components are combined
    via mass-fraction weighting: each normalized to unit mass before mixing.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    log_peak_sfr : float
        log10 of overall SFR normalization [Msun/yr].
    age : float
        Galaxy age [yr] = lookback time of old component formation.
        Exponential is active for burstage < t < age.
    tau : float
        e-folding timescale of old exponential component [yr].
    burstage : float
        Lookback time of burst onset [yr]. Burst active for 0 < t < burstage.
    alpha : float
        DPL falling slope [dimensionless] (post-peak in cosmic time).
    beta : float
        DPL rising slope [dimensionless] (pre-peak in cosmic time).
    fburst : float
        Fraction of total stellar mass in burst [dimensionless], range [0, 1].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and logical masking.
    """
    peak_sfr = 10.0**log_peak_sfr

    # --- Old component: declining exponential between burstage and age ---
    t_cosmic_old = age - t_lookback
    sfr_exp = jnp.exp(-t_cosmic_old / tau)
    mask_old = (t_lookback >= burstage) & (t_lookback <= age)
    sfr_exp = jnp.where(mask_old, sfr_exp, 0.0)

    # --- Burst component: DPL in cosmic time, peaks at (age_universe - burstage) ---
    age_universe = AGEMAX_YR
    t_cosmic = age_universe - t_lookback
    tau_burst = age_universe - burstage
    log_ratio = jnp.log(jnp.maximum(t_cosmic, 1.0) / jnp.maximum(tau_burst, 1.0))
    sfr_burst = jnp.exp(-jnp.logaddexp(alpha * log_ratio, -beta * log_ratio))
    mask_burst = t_lookback <= burstage
    sfr_burst = jnp.where(mask_burst, sfr_burst, 0.0)

    # --- Mass-normalize each component using dt-weighted integration ---
    # jnp.gradient gives symmetric finite-difference widths; correct for
    # log-spaced grids (DSPS) where bins span very different linear intervals.
    dt = jnp.gradient(t_lookback)
    m_exp = jnp.sum(sfr_exp * dt) + 1e-30
    m_burst = jnp.sum(sfr_burst * dt) + 1e-30

    sfr = peak_sfr * ((1.0 - fburst) * sfr_exp / m_exp + fburst * sfr_burst / m_burst)
    return jnp.maximum(sfr, 0.0)


def powerlaw(
    t_lookback: jnp.ndarray, alpha: float, norm: float, t_ref: float = 1e8
) -> jnp.ndarray:
    """Power-law SFH: SFR(t) = norm * (t/t_ref)^alpha.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    alpha : float
        Power-law exponent [dimensionless].
    norm : float
        Normalization factor [Msun/yr].
    t_ref : float
        Reference timescale [yr]. Default 1e8.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.
    """
    return norm * (t_lookback / t_ref) ** alpha


def delayed_bq(
    t_lookback: jnp.ndarray,
    tau_main_yr: float,
    age_main_yr: float,
    age_bq_yr: float,
    r_sfr: float,
) -> jnp.ndarray:
    """Delayed-tau SFH with burst or quench episode (Ciesla+2017).

    A delayed exponential SFR that peaks at tau_main_yr, followed by a burst
    or quench at age_bq_yr. Before the burst/quench, the SFR follows the
    delayed-tau form: SFR(t) = t * exp(-t/tau) / tau^2. After age_bq_yr,
    the SFR is constant at r_sfr times the SFR at the burst/quench onset.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    tau_main_yr : float
        e-folding timescale of main component [yr]. Peak occurs at tau_main_yr.
    age_main_yr : float
        Age of the main stellar population / galaxy age [yr]. The delayed tau
        model extends from 0 to age_main_yr.
    age_bq_yr : float
        Age at which burst/quench episode begins [yr].
    r_sfr : float
        Ratio of SFR after/before burst/quench [dimensionless].
        r_sfr < 1 is quenching, r_sfr > 1 is bursting.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere for positive tau_main_yr.

    The SFR is defined as:

    .. math::

        \\mathrm{SFR}(t) = \\begin{cases}
        \\frac{t}{\\tau^2} \\exp(-t/\\tau) & t < t_{\\rm bq} \\\\
        r_{\\rm sfr} \\times \\mathrm{SFR}(t_{\\rm bq}) & t \\geq t_{\\rm bq}
        \\end{cases}

    where :math:`\\tau` is ``tau_main_yr`` [yr], :math:`t_{\\rm bq}` is
    ``age_main_yr - age_bq_yr`` [yr], and :math:`r_{\\rm sfr}` is the
    post-episode SFR ratio [dimensionless].

    **Upstream**: Ported from CIGALE ``sfhdelayedbq.py`` (Boquien et al. 2019 [2]_).

    References
    ----------
    .. [1] L. Ciesla et al., "The SFR-M* main sequence archetypal star-formation
       history and analytical models," A&A, 608, A41 (2017). arXiv:1706.08531.
       https://doi.org/10.1051/0004-6361/201731036
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh.mean_sfh import delayed_bq
    >>> t = jnp.logspace(7, 10.14, 100)
    >>> sfr = delayed_bq(t, tau_main_yr=2e9, age_main_yr=5e9,
    ...     age_bq_yr=500e6, r_sfr=0.1)
    >>> sfr.shape
    (100,)
    """
    t_bq = age_main_yr - age_bq_yr
    t_tau = t_lookback / tau_main_yr
    sfr_delayed = t_tau * jnp.exp(-t_tau) / tau_main_yr
    sfr_at_bq = (t_bq / tau_main_yr) * jnp.exp(-t_bq / tau_main_yr) / tau_main_yr
    sfr_post_bq = r_sfr * sfr_at_bq
    sfr = jnp.where(t_lookback >= t_bq, sfr_post_bq, sfr_delayed)
    return jnp.where((t_lookback >= 0) & (t_lookback <= age_main_yr), sfr, 0.0)


def periodic(
    t_lookback: jnp.ndarray,
    delta_bursts_yr: float,
    tau_bursts_yr: float,
    burst_type: int,
    age_yr: float,
) -> jnp.ndarray:
    """Periodic SFH with regularly-spaced star formation events.

    Regularly-spaced SF events at intervals of delta_bursts_yr, each with
    duration/e-folding timescale tau_bursts_yr. The shape of each event
    depends on burst_type: 0=exponential, 1=delayed, 2=rectangular.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    delta_bursts_yr : float
        Elapsed time between the beginning of each burst [yr].
    tau_bursts_yr : float
        Duration (for rectangular) or e-folding timescale [yr] of each event.
    burst_type : int
        Type of burst event [dimensionless]:
        0 = exponential: exp(-t/tau),
        1 = delayed: (t/tau^2) * exp(-t/tau),
        2 = rectangular: constant 1 for t < tau, 0 otherwise.
    age_yr : float
        Age of the galaxy / maximum time [yr].

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jnp.mod``.

    **Gradient-safe**: yes — differentiable everywhere except at event boundaries
    (discontinuities for rectangular type).

    The SFR is a superposition of multiple burst events spaced delta_bursts_yr
    apart, each following one of three shapes. The burst at event i starts at
    time i * delta_bursts_yr and ends (or decays) within tau_bursts_yr.

    **Upstream**: Ported from CIGALE ``sfhperiodic.py`` (Boquien et al. 2019 [1]_).

    References
    ----------
    .. [1] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh.mean_sfh import periodic
    >>> t = jnp.logspace(6, 10, 100)
    >>> sfr = periodic(t, delta_bursts_yr=50e6, tau_bursts_yr=20e6, burst_type=0, age_yr=1000e6)
    >>> sfr.shape
    (100,)
    """
    tau = tau_bursts_yr

    # Vectorised: burst_starts shape (n_bursts, 1), t shape (1, n_time)
    n_bursts = 100
    burst_starts = jnp.arange(n_bursts) * delta_bursts_yr  # (n_bursts,)
    dt = t_lookback[None, :] - burst_starts[:, None]  # (n_bursts, n_time)

    exp_decay = jnp.exp(-dt / tau)
    delayed_exp = (dt / tau**2) * exp_decay
    rectangular = jnp.where(dt <= tau, 1.0, 0.0)

    burst = (
        jnp.where(burst_type == 0, exp_decay, 0.0)
        + jnp.where(burst_type == 1, delayed_exp, 0.0)
        + jnp.where(burst_type == 2, rectangular, 0.0)
    )

    mask = (dt >= 0) & (t_lookback[None, :] <= age_yr)
    sfr = jnp.sum(jnp.where(mask, burst, 0.0), axis=0)

    return jnp.maximum(sfr, 0.0)


_BUAT08_VELOCITIES = jnp.array(
    [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 220.0, 290.0, 360.0]
)
_BUAT08_A = jnp.array([4.73, 5.28, 5.77, 6.21, 6.62, 6.99, 7.34, 8.74, 10.01, 10.82, 11.35])
_BUAT08_B = jnp.array([-0.11, 0.029, 0.16, 0.29, 0.41, 0.51, 0.61, 0.98, 1.25, 1.36, 1.37])
_BUAT08_C = jnp.array([0.79, 0.68, 0.57, 0.46, 0.36, 0.27, 0.18, -0.20, -0.55, -0.74, -0.85])


def buat08(
    t_lookback: jnp.ndarray,
    velocity_km_s: float,
) -> jnp.ndarray:
    """Chemically-motivated SFH parameterized by galaxy rotational velocity.

    A physically-motivated SFH derived from chemical evolution models.
    The SFR is given by a polynomial in log10-space:

        log10(SFR(t)) = a + b*log10(t) + c*t^0.5

    The coefficients a, b, c are interpolated from Buat+2008 Table 2
    based on the galaxy's rotational velocity.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    velocity_km_s : float
        Rotational velocity of the galaxy [km/s]. Must be between 40 and 360.
        Will be clipped to this range if necessary.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jnp.interp``.

    **Gradient-safe**: yes — differentiable with respect to velocity.

    Coefficients from Buat et al. (2008) Table 2, extended with additional
    data provided by S. Boissier for velocities down to 40 km/s.

    The formula in log time is:

    .. math::

        \\log_{10}(\\mathrm{SFR}(t)) = a(v) + b(v) \\log_{10}(t/\\mathrm{Gyr}) +
        c(v) (t/\\mathrm{Gyr})^{0.5} - 9

    where :math:`v` is the rotational velocity [km/s], and the offset -9 converts
    from galaxy-integrated SFR to local SFR normalization.

    **Upstream**: Ported from CIGALE ``sfh_buat08.py`` (Boquien et al. 2019 [2]_).
    Extended velocity coefficients (40–100 km/s) provided by S. Boissier
    (private communication, referenced in CIGALE source).

    References
    ----------
    .. [1] V. Buat et al., "Star formation history of galaxies from z = 0 to
       z = 0.7. A backward approach to the evolution of star-forming galaxies,"
       A&A, 483, 107 (2008). arXiv:0803.0414.
       https://doi.org/10.1051/0004-6361:20078829
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh.mean_sfh import buat08
    >>> t = jnp.logspace(6, 10.14, 100)
    >>> sfr = buat08(t, velocity_km_s=220.0)
    >>> sfr.shape
    (100,)
    """
    v = jnp.clip(velocity_km_s, 40.0, 360.0)

    a = jnp.interp(v, _BUAT08_VELOCITIES, _BUAT08_A)
    b = jnp.interp(v, _BUAT08_VELOCITIES, _BUAT08_B)
    c = jnp.interp(v, _BUAT08_VELOCITIES, _BUAT08_C)

    t_gyr = jnp.maximum(t_lookback / 1e9, 1e-9)
    log_sfr = a + b * jnp.log10(t_gyr) + c * jnp.sqrt(t_gyr) - 9.0
    sfr = 10.0**log_sfr

    return jnp.maximum(sfr, 0.0)


# ── ProSpect spline SFH (Robotham+2020) ─────────────────────────


def _pchip_slopes(y: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Fritsch-Carlson monotone slopes for PCHIP cubic spline.

    Parameters
    ----------
    y : array_like, shape (n,)
        Values at the n nodes (traced under JIT).
    h : array_like, shape (n-1,)
        Spacings between adjacent nodes (static — must be concrete).

    Returns
    -------
    ndarray, shape (n,)
        Slope at each node satisfying the Fritsch-Carlson monotonicity condition.
    """
    delta = jnp.diff(y) / h  # secant slopes

    # Interior slopes: Fritsch-Carlson weighted harmonic mean
    h0 = h[:-1]
    h1 = h[1:]
    w0 = 2.0 * h1 + h0
    w1 = h1 + 2.0 * h0

    safe_d0 = jnp.where(jnp.abs(delta[:-1]) > 1e-30, delta[:-1], jnp.sign(delta[:-1]) * 1e-30)
    safe_d1 = jnp.where(jnp.abs(delta[1:]) > 1e-30, delta[1:], jnp.sign(delta[1:]) * 1e-30)
    denom = w0 / safe_d0 + w1 / safe_d1
    d_int = jnp.where(delta[:-1] * delta[1:] > 0.0, (w0 + w1) / denom, 0.0)

    # Endpoint slopes: one-sided extrapolation (Moler 2004)
    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    d0 = jnp.where(jnp.sign(d0) * jnp.sign(delta[0]) < 0.0, 0.0, d0)
    d0 = jnp.where(jnp.abs(d0) > 3.0 * jnp.abs(delta[0]), 3.0 * delta[0], d0)

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    dn = jnp.where(jnp.sign(dn) * jnp.sign(delta[-1]) < 0.0, 0.0, dn)
    dn = jnp.where(jnp.abs(dn) > 3.0 * jnp.abs(delta[-1]), 3.0 * delta[-1], dn)

    return jnp.concatenate([jnp.array([d0]), d_int, jnp.array([dn])])


def _pchip_eval(
    x_query: jnp.ndarray,
    x_nodes: jnp.ndarray,
    y_nodes: jnp.ndarray,
    d: jnp.ndarray,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate PCHIP cubic Hermite pieces at query points.

    Parameters
    ----------
    x_query : array_like, shape (n_query,)
        Query positions.
    x_nodes : array_like, shape (n,)
        Node positions (static).
    y_nodes : array_like, shape (n,)
        Node values (traced).
    d : array_like, shape (n,)
        Node slopes from :func:`_pchip_slopes` (traced).
    h : array_like, shape (n-1,)
        Node spacings (static).

    Returns
    -------
    ndarray, shape (n_query,)
        Interpolated values.
    """
    idx = jnp.searchsorted(x_nodes, x_query, side="right") - 1
    idx = jnp.clip(idx, 0, x_nodes.shape[0] - 2)

    x0 = x_nodes[idx]
    hi = h[idx]
    y0 = y_nodes[idx]
    y1 = y_nodes[idx + 1]
    d0 = d[idx]
    d1 = d[idx + 1]

    t = (x_query - x0) / hi
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    return h00 * y0 + h10 * hi * d0 + h01 * y1 + h11 * hi * d1


def spline(
    t_lookback: jnp.ndarray,
    sfr_nodes: jnp.ndarray,
    node_ages_yr: jnp.ndarray,
) -> jnp.ndarray:
    """Monotone cubic spline SFH with free SFR control nodes (Robotham+2020).

    A smooth, continuously varying SFH defined by N control nodes at fixed
    lookback times. SFR values at the nodes are free parameters; PCHIP cubic
    Hermite interpolation (Fritsch-Carlson monotone) in log10(age) space
    ensures a smooth, non-negative SFH between nodes.

    This is the JAX port of ProSpect's ``massfunc_p4`` / ``massfunc_p6``
    (Robotham et al. 2020 [1]_), which use R's ``splinefun(..., method='monoH.FC')``.
    The implementation uses the Fritsch-Carlson (1980) [2]_ algorithm.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time grid [yr].
    sfr_nodes : array_like, shape (n_nodes,)
        SFR at each control node [Msun/yr]. Free parameters. Must be non-negative.
    node_ages_yr : array_like, shape (n_nodes,)
        Lookback times of control nodes [yr]. Must be strictly increasing.
        **Not JIT-traced** — pass as a concrete array constructed before JIT.
        Typical 4-node default: ``[1e5, 2e9, 9e9, 13e9]`` yr.
        Typical 6-node default: ``[1e5, 1e8, 1e9, 5e9, 9e9, 13e9]`` yr.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — ``sfr_nodes`` is fully traced. ``node_ages_yr``
    must be a concrete (non-traced) array; pass it as a Python/NumPy array or
    mark it as static in :func:`jax.jit`.

    The spline evaluates in log10(age) space:

    .. math::

        \\mathrm{SFR}(t) = \\mathcal{S}_{\\mathrm{PCHIP}}
            \\bigl(\\log_{10} t \\;\\big|\\; \\log_{10} t_i,\\; m_i \\bigr)

    where :math:`t_i` are the fixed node ages [yr] and :math:`m_i = \\text{sfr\\_nodes}_i`
    are the free SFR values [Msun/yr]. The PCHIP (Piecewise Cubic Hermite
    Interpolating Polynomial) enforces monotonicity within each segment so
    that the interpolant cannot overshoot between adjacent nodes.

    Outside the range ``[node_ages_yr[0], node_ages_yr[-1]]``, ``jnp.searchsorted``
    clamps to the nearest segment — equivalent to constant extrapolation beyond
    the outermost nodes. Clamp ``node_ages_yr[0]`` to the youngest SSP age (typically
    1e5 yr) to avoid extrapolation into the stellar library edge.

    **Not in the SFH registry**: ``sfr_nodes`` is an array parameter, which does not
    fit the scalar-kwarg registry architecture. Use this function directly with
    ``jax.jit`` or ``jax.grad``, marking ``node_ages_yr`` as static.

    Ported from ProSpect ``massfunc_p4`` / ``massfunc_p6`` (Robotham et al. 2020 [1]_).
    The PCHIP algorithm follows Fritsch & Carlson (1980) [2]_ with endpoint slopes
    from Moler (2004) [3]_.

    References
    ----------
    .. [1] A. S. G. Robotham et al., "ProSpect: generating spectral energy
       distributions with complex star formation and metallicity histories,"
       MNRAS, 495, 905 (2020). arXiv:2002.06980.
       https://doi.org/10.1093/mnras/staa1116
    .. [2] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic
       Interpolation," SIAM J. Numer. Anal., 17(2), 238-246 (1980).
       https://doi.org/10.1137/0717021
    .. [3] C. B. Moler, "Numerical Computing with MATLAB," SIAM, Ch. 3 (2004).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from tengri.components.stellar.sfh.mean_sfh import spline
    >>> node_ages = np.array([1e5, 2e9, 9e9, 13e9])
    >>> sfr_nodes = jnp.array([5.0, 10.0, 3.0, 0.5])
    >>> t = jnp.logspace(5.0, 10.14, 100)
    >>> sfr = spline(t, sfr_nodes, node_ages)
    >>> sfr.shape
    (100,)
    """
    x_nodes = jnp.log10(jnp.maximum(node_ages_yr, 1.0))
    h = jnp.diff(x_nodes)  # static when node_ages_yr is concrete

    x_query = jnp.log10(jnp.maximum(t_lookback, 1.0))
    d = _pchip_slopes(sfr_nodes, h)
    sfr = _pchip_eval(x_query, x_nodes, sfr_nodes, d, h)
    return jnp.maximum(sfr, 0.0)


def snorm_burst(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
    skew: float,
    burst_sfr: float,
    burst_age: float,
) -> jnp.ndarray:
    """Skew-normal SFH with a flat recent burst component (Robotham+2020).

    Adds a constant burst SFR to the skew-normal SFH at lookback times younger
    than ``burst_age``. The burst represents recent (typically <100 Myr) star
    formation activity superimposed on the smooth underlying SFH.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_peak_sfr : float
        log10 of peak SFR of the skew-normal component [Msun/yr].
    peak_lbt : float
        Peak lookback time of the skew-normal component [yr].
    width : float
        Gaussian width of the skew-normal component [yr].
    skew : float
        Skewness parameter [dimensionless]. 0 = symmetric, >0 skews toward older ages.
    burst_sfr : float
        Constant SFR amplitude of the recent burst [Msun/yr]. Set to 0 to disable.
    burst_age : float
        Lookback time below which the burst is active [yr]. Default 1e8 yr (100 Myr).

    Returns
    -------
    ndarray, shape (n_age,)
        Total SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The total SFH is:

    .. math::

        \\mathrm{SFR}(t) = \\mathrm{snorm}(t) + m_{\\rm burst} \\cdot
        \\mathbb{1}[t < t_{\\rm burst}]

    where :math:`\\mathrm{snorm}(t)` is the skew-normal kernel (Robotham+2020 [1]_),
    :math:`m_{\\rm burst}` is the burst amplitude [Msun/yr], and
    :math:`t_{\\rm burst}` is the burst lookback time [yr].

    Ported from ProSpect ``massfunc_snorm_burst`` (Robotham et al. 2020 [1]_).

    References
    ----------
    .. [1] A. S. G. Robotham et al., "ProSpect: generating spectral energy
       distributions with complex star formation and metallicity histories,"
       MNRAS, 495, 905 (2020). arXiv:2002.06980.
       https://doi.org/10.1093/mnras/staa1116

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import snorm_burst
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = snorm_burst(
    ...     t, log_peak_sfr=1.5, peak_lbt=5e9, width=2e9, skew=0.5, burst_sfr=2.0, burst_age=1e8
    ... )
    >>> sfr.shape
    (64,)
    """
    sfr_snorm = skewnormal(t_lookback, log_peak_sfr, peak_lbt, width, skew)
    burst = jnp.where(t_lookback < burst_age, burst_sfr, 0.0)
    return jnp.maximum(sfr_snorm + burst, 0.0)


def snorm_trunc_burst(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
    skew: float,
    trunc: float,
    burst_sfr: float,
    burst_age: float,
) -> jnp.ndarray:
    """Truncated skew-normal SFH with a flat recent burst component (Robotham+2020).

    Adds a constant burst SFR to the truncated skew-normal SFH (tsnorm) at
    lookback times younger than ``burst_age``. Combines the smooth quenching
    truncation of ``truncated_skewnormal`` with a superimposed recent burst.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time grid [yr].
    log_peak_sfr : float
        log10 of peak SFR of the tsnorm component [Msun/yr].
    peak_lbt : float
        Peak lookback time [yr].
    width : float
        Gaussian width [yr].
    skew : float
        Skewness parameter [dimensionless].
    trunc : float
        Truncation sharpness [dimensionless]. Larger values = sharper truncation.
    burst_sfr : float
        Constant burst SFR amplitude [Msun/yr]. Set to 0 to disable.
    burst_age : float
        Lookback time below which the burst is active [yr].

    Returns
    -------
    ndarray, shape (n_age,)
        Total SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The total SFH is:

    .. math::

        \\mathrm{SFR}(t) = \\mathrm{tsnorm}(t) + m_{\\rm burst} \\cdot
        \\mathbb{1}[t < t_{\\rm burst}]

    where :math:`\\mathrm{tsnorm}(t)` is the truncated skew-normal SFH
    (Bellstedt+2020 [2]_) and :math:`m_{\\rm burst}` is the burst amplitude.

    Ported from ProSpect ``massfunc_snorm_burst_trunc`` (Robotham et al. 2020 [1]_).

    References
    ----------
    .. [1] A. S. G. Robotham et al., "ProSpect: generating spectral energy
       distributions with complex star formation and metallicity histories,"
       MNRAS, 495, 905 (2020). arXiv:2002.06980.
       https://doi.org/10.1093/mnras/staa1116
    .. [2] S. Bellstedt et al., "Galaxy And Mass Assembly (GAMA): a forensic
       SED reconstruction of the cosmic star formation history and metallicity
       evolution by galaxy type," MNRAS, 498, 5581 (2020). arXiv:2005.11917.
       https://doi.org/10.1093/mnras/staa2620

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import snorm_trunc_burst
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = snorm_trunc_burst(
    ...     t,
    ...     log_peak_sfr=1.5,
    ...     peak_lbt=5e9,
    ...     width=2e9,
    ...     skew=0.5,
    ...     trunc=2.0,
    ...     burst_sfr=2.0,
    ...     burst_age=1e8,
    ... )
    >>> sfr.shape
    (64,)
    """
    sfr_tsnorm = truncated_skewnormal(t_lookback, log_peak_sfr, peak_lbt, width, skew, trunc)
    burst = jnp.where(t_lookback < burst_age, burst_sfr, 0.0)
    return jnp.maximum(sfr_tsnorm + burst, 0.0)


# Short alias registered in SFH_REGISTRY
tsnorm_burst = snorm_trunc_burst


def top_hat(
    t_lookback: jnp.ndarray,
    amplitude: float,
    t_start: float,
    t_end: float,
    smooth_width: float = 1e8,
) -> jnp.ndarray:
    """Top-hat (constant-window) SFH with smooth sigmoid edges.

    A constant SFR amplitude between t_start and t_end, smoothly tapering
    to zero outside this window via sigmoid functions. Useful for modeling
    bursty episodes or isolated star-forming events.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    amplitude : float
        Constant SFR inside the window [Msun/yr].
    t_start : float
        Older lookback boundary [yr]. Must have t_start > t_end.
    t_end : float
        Younger lookback boundary [yr]. Must have t_end < t_start.
    smooth_width : float, optional
        Width of sigmoid transition region [yr]. Controls gradient smoothness
        at edges. Default 1e8 (100 Myr). Typical range: 1e7 - 1e9.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.nn.sigmoid`` for smooth edges.

    **Gradient-safe**: yes — sigmoid ensures differentiability everywhere.

    The SFH is:

    .. math::

        \\mathrm{SFR}(t) = A \\cdot \\sigma\\left(\\frac{t_{{\\rm start}} - t}{w}\\right)
        \\cdot \\sigma\\left(\\frac{t - t_{{\\rm end}}}{w}\\right)

    where :math:`A` is amplitude [Msun/yr], :math:`t_{{\\rm start}}` and
    :math:`t_{{\\rm end}}` are the window boundaries [yr], :math:`w` is the
    sigmoid smoothing width [yr], and :math:`\\sigma(x) = 1/(1+e^{-x})`
    is the logistic sigmoid function.

    The window is centered around :math:`(t_{{\\rm start}} + t_{{\\rm end}})/2`
    with full-width (at half-maximum) approximately :math:`t_{{\\rm start}} - t_{{\\rm end}}`.

    References
    ----------
    .. [1] Robotham, A. S. G., et al. "ProSpect: generating spectral energy
       distributions with complex star formation and metallicity histories,"
       MNRAS, 495, 905 (2020). arXiv:2002.06980.
       https://doi.org/10.1093/mnras/staa1116

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import top_hat
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = top_hat(t, amplitude=1.0, t_start=5e9, t_end=3e9, smooth_width=1e8)
    >>> sfr.shape
    (64,)
    """
    # Sigmoid functions for smooth edges
    s_start = jax.nn.sigmoid((t_start - t_lookback) / smooth_width)
    s_end = jax.nn.sigmoid((t_lookback - t_end) / smooth_width)
    return amplitude * s_start * s_end


def gaussian_burst(
    t_lookback: jnp.ndarray,
    amplitude: float,
    t_peak: float,
    sigma: float,
) -> jnp.ndarray:
    """Gaussian-in-age burst component (Robotham+2020).

    A parametric burst modeled as a Gaussian envelope in lookback time.
    This form is compatible with composition as an additive component
    on top of any other parametric SFH model.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    amplitude : float
        Peak SFR of the burst [Msun/yr].
    t_peak : float
        Peak lookback time (age of burst) [yr].
    sigma : float
        Standard deviation of the Gaussian envelope [yr]. The FWHM is
        approximately 2.355 * sigma.

    Returns
    -------
    ndarray, shape (n_age,)
        SFR at each lookback time [Msun/yr], non-negative.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives for Gaussian kernel.

    **Gradient-safe**: yes — differentiable everywhere.

    The SFH is:

    .. math::

        \\mathrm{SFR}(t) = A \\exp\\left( -\\frac{(t - t_{{\\rm peak}})^2}{2\\sigma^2} \\right)

    where :math:`A` is amplitude [Msun/yr], :math:`t_{{\\rm peak}}` is the
    burst age [yr], and :math:`\\sigma` is the width parameter [yr].

    **Integration**: The integrated mass in the burst is approximately
    :math:`A \\cdot \\sigma \\cdot \\sqrt{2\\pi}` for a Gaussian with
    sufficient tail coverage.

    References
    ----------
    .. [1] Robotham, A. S. G., et al. "ProSpect: generating spectral energy
       distributions with complex star formation and metallicity histories,"
       MNRAS, 495, 905 (2020). arXiv:2002.06980.
       https://doi.org/10.1093/mnras/staa1116

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.stellar.sfh import gaussian_burst
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = gaussian_burst(t, amplitude=5.0, t_peak=1e9, sigma=1e8)
    >>> sfr.shape
    (64,)
    """
    dt = t_lookback - t_peak
    exponent = -0.5 * (dt / sigma) ** 2
    return amplitude * jnp.exp(exponent)
