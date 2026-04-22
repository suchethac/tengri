"""Smooth parametric mean star formation history components.

The GP x(t) has zero mean, so the overall SFH shape comes from these
functions. The full SFH is: SFR(t) = mean(t) * exp(x(t) - K(0)/2),
where -K(0)/2 is the lognormal correction preserving the linear-SFR mean.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.

Models
------
Canonical names (short name alias in parentheses):

- **truncated_skewnormal_sfh** (tsnorm): Bellstedt+2020, Robotham+2020 snorm_trunc.
  Most flexible smooth model — 5 params: peak location, width, skew, truncation.
- **skewnormal_sfh** (snorm): truncated_skewnormal_sfh without truncation (4 params).
- **gaussian_sfh** (norm): skewnormal_sfh with skew=0 (3 params).
- **lognormal_sfh** (lnorm): Gaussian in log10(age) space (3 params).
- **dpl** (canonical): Carnall+2018 BAGPIPES parameterization with log_peak_sfr (4 params).
- **double_powerlaw**: Low-level implementation used by dpl.
- **constant_sfh** (const): flat SFR between start and end times (3 params).
- **exponential_sfh** (exp): declining exponential from start (3 params).
- **delayed_exponential_sfh** (dexp): peaks at start + tau (3 params).
- **declining_exponential_sfh** (tau): FSPS/bagpipes tau model in lookback time (3 params).
- **triweight_burst**: compact triweight kernel in log-age for burst component.

References
----------
- Bellstedt+2020 (arXiv:2005.11917): snorm, snorm_trunc parameterizations.
- Robotham+2020 (arXiv:2002.06980): ProSpect SFH models.
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


def truncated_skewnormal_sfh(
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
        Gaussian width parameter [yr].
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
    >>> from tengri import truncated_skewnormal_sfh
    >>> t = jnp.linspace(0.0, 13.7e9, 100)
    >>> sfr = truncated_skewnormal_sfh(t, log_peak_sfr=1.0, peak_lbt=5e9,
    ...     width=2e9, skew=0.0, trunc=5.0)
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
tsnorm = truncated_skewnormal_sfh


def skewnormal_sfh(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
    skew: float,
) -> jnp.ndarray:
    """Skew-normal SFH (Robotham+2020).

    Like truncated_skewnormal_sfh but without truncation.

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
    >>> from tengri import skewnormal_sfh
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = skewnormal_sfh(t, log_peak_sfr=1.5, peak_lbt=3e9, width=1e9, skew=1.0)
    >>> sfr.shape
    (64,)
    """
    age = _clamp_age(t_lookback)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    return jnp.maximum(peak_sfr * kernel, 0.0)


# Short alias registered in SFH_REGISTRY
snorm = skewnormal_sfh


def gaussian_sfh(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
) -> jnp.ndarray:
    """Gaussian (normal) SFH — skewnormal_sfh with skew=0.

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
    >>> from tengri import gaussian_sfh
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = gaussian_sfh(t, log_peak_sfr=1.5, peak_lbt=3e9, width=1e9)
    >>> sfr.shape
    (64,)
    """
    return skewnormal_sfh(t_lookback, log_peak_sfr, peak_lbt, width, skew=0.0)


# Short alias registered in SFH_REGISTRY
norm = gaussian_sfh


def lognormal_sfh(
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
    >>> from tengri import lognormal_sfh
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = lognormal_sfh(t, log_peak_sfr=1.5, peak_lbt=3e9, width=0.3)
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
lnorm = lognormal_sfh


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


def constant_sfh(
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
    >>> from tengri import constant_sfh
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = constant_sfh(t, log_sfr=1.0, start=5e8, end=5e9)
    >>> sfr.shape
    (64,)
    """
    sfr = 10.0**log_sfr
    mask = (t_lookback >= start) & (t_lookback <= end)
    return jnp.where(mask, sfr, 0.0)


def exponential_sfh(
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
    >>> from tengri import exponential_sfh
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = exponential_sfh(t, log_peak_sfr=2.0, tau=2e9, start=1e8)
    >>> sfr.shape
    (64,)
    """
    peak_sfr = 10.0**log_peak_sfr
    dt = t_lookback - start
    sfr = peak_sfr * jnp.exp(-dt / tau)
    return jnp.where(dt >= 0, sfr, 0.0)


def delayed_exponential_sfh(
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
    >>> from tengri import delayed_exponential_sfh
    >>> t = jnp.logspace(7, 10.14, 64)
    >>> sfr = delayed_exponential_sfh(t, log_peak_sfr=2.0, tau=2e9, start=1e8)
    >>> sfr.shape
    (64,)
    """
    peak_sfr = 10.0**log_peak_sfr
    dt = t_lookback - start
    ratio = dt / tau
    sfr = peak_sfr * ratio * jnp.exp(-ratio + 1.0)
    return jnp.where(dt >= 0, jnp.maximum(sfr, 0.0), 0.0)


def declining_exponential_sfh(
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
    declining to the present). This is opposite to ``exponential_sfh``.

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


def constant_then_exponential_sfh(
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

    # --- Mass-normalize each component ---
    m_exp = jnp.sum(sfr_exp) + 1e-30
    m_burst = jnp.sum(sfr_burst) + 1e-30

    sfr = peak_sfr * ((1.0 - fburst) * sfr_exp / m_exp + fburst * sfr_burst / m_burst)
    return jnp.maximum(sfr, 0.0)


def powerlaw_sfh(
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
