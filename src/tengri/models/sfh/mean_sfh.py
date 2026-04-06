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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Smooth SFH models — all take t_lookback (yr), return SFR (Msun/yr)
# ---------------------------------------------------------------------------


def truncated_skewnormal_sfh(
    t_lookback: jnp.ndarray,
    log_peak_sfr: float,
    peak_lbt: float,
    width: float,
    skew: float,
    trunc: float,
) -> jnp.ndarray:
    """Truncated skew-normal SFH (Bellstedt+2020, Robotham+2020).

    The most flexible smooth SFH model: a skewed Gaussian kernel
    multiplied by a normal CDF truncation factor that smoothly
    suppresses SFR at recent times.

    SFR(t) = peak_sfr * kernel(t) * (1 - Phi((t - peak) / (width * trunc)))

    Parameters
    ----------
    t_lookback : array
        Lookback time (yr).
    log_peak_sfr : float
        log10 of peak SFR (Msun/yr).
    peak_lbt : float
        Peak lookback time (yr).
    width : float
        Gaussian width (yr).
    skew : float
        Skewness. 0 = symmetric.
    trunc : float
        Truncation sharpness. Larger = more truncation.
        Typical range: 1-10.

    Returns
    -------
    array
        SFR at each lookback time (Msun/yr), non-negative.

    References
    ----------
    Bellstedt+2020 (arXiv:2005.11917), Eq. 2-4.
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


# Alias for backward compatibility and registry
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
    t_lookback : array
        Lookback time (yr).
    log_peak_sfr : float
        log10 of peak SFR (Msun/yr).
    peak_lbt : float
        Peak lookback time (yr).
    width : float
        Gaussian width (yr).
    skew : float
        Skewness. 0 = symmetric.

    Returns
    -------
    array
        SFR (Msun/yr), non-negative.
    """
    age = _clamp_age(t_lookback)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    return jnp.maximum(peak_sfr * kernel, 0.0)


# Alias for backward compatibility and registry
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
    t_lookback : array
        Lookback time (yr).
    log_peak_sfr : float
        log10 of peak SFR (Msun/yr).
    peak_lbt : float
        Peak lookback time (yr).
    width : float
        Gaussian width (yr).

    Returns
    -------
    array
        SFR (Msun/yr), non-negative.
    """
    return skewnormal_sfh(t_lookback, log_peak_sfr, peak_lbt, width, skew=0.0)


# Alias for backward compatibility and registry
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
    For w=0.3, this is a ~15% shift toward younger ages.  peak_lbt is
    best interpreted as the median lookback time, not the linear peak.

    Parameters
    ----------
    t_lookback : array
        Lookback time (yr).
    log_peak_sfr : float
        log10 of peak SFR (Msun/yr).
    peak_lbt : float
        Peak lookback time (yr). Converted to log10 internally.
    width : float
        Width in log10(age) space (dex).

    Returns
    -------
    array
        SFR (Msun/yr), non-negative.
    """
    age = _clamp_age(t_lookback)
    peak_sfr = 10.0**log_peak_sfr
    log_age = jnp.log10(age)
    log_peak = jnp.log10(jnp.maximum(peak_lbt, 1e5))
    exponent = -0.5 * ((log_age - log_peak) / width) ** 2
    return jnp.maximum(peak_sfr * jnp.exp(exponent), 0.0)


# Alias for backward compatibility and registry
lnorm = lognormal_sfh


def double_powerlaw(
    t_lookback: jnp.ndarray, alpha: float, beta: float, tau: float, norm: float
) -> jnp.ndarray:
    """BAGPIPES-style double power law SFH (Carnall+2018, Behroozi+2013).

    SFR(t) = norm / [(t/tau)^alpha + (t/tau)^(-beta)]

    Peaks near t ~ tau.

    In **cosmic time**: alpha controls the falling (declining) phase
    after peak SFR, beta controls the rising phase before peak.

    In **lookback time** plots (what we show): alpha controls the
    RIGHT side (early universe, large lookback), beta controls the
    LEFT side (near present, small lookback).

    Parameters
    ----------
    t_lookback : array
        Lookback time (yr).
    alpha : float
        Falling slope in cosmic time (SFR decline from peak to present).
        Larger alpha = steeper decline. Typical range: 0.5-4.
    beta : float
        Rising slope in cosmic time (SFR rise from early times to peak).
        Larger beta = steeper rise. Typical range: 0.3-3.
    tau : float
        Turnover timescale (yr), approximately when SFR peaks.
    norm : float
        Peak SFR normalization (Msun/yr). NOTE: this is NOT the stellar
        mass. M* = integral of SFR(t) dt is a derived quantity.

    Returns
    -------
    array
        SFR at each lookback time (Msun/yr).
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
    """Double power law with log_peak_sfr parameterization (canonical: dpl).

    Registry-compatible wrapper around the Carnall+2018 DPL.
    Uses log10(peak_sfr) instead of linear norm for consistency
    with other registry models.

    Parameters
    ----------
    t_lookback : array
        Lookback time (yr).
    alpha : float
        Falling slope. Typical range: 0.5-4.
    beta : float
        Rising slope. Typical range: 0.3-3.
    tau : float
        Turnover timescale (yr).
    log_peak_sfr : float
        log10 of peak SFR (Msun/yr).

    Returns
    -------
    array
        SFR (Msun/yr).
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
    t_lookback : array
        Lookback time (yr).
    log_sfr : float
        log10 of constant SFR (Msun/yr).
    start : float
        Start lookback time (yr). Default: 0 (present).
    end : float
        End lookback time (yr). Default: AGEMAX_YR.

    Returns
    -------
    array
        SFR (Msun/yr), flat between start and end, zero outside.
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
    t_lookback : array
        Lookback time (yr).
    log_peak_sfr : float
        log10 of peak SFR at start (Msun/yr).
    tau : float
        e-folding timescale (yr).
    start : float
        Start lookback time (yr). Default: 0 (present).

    Returns
    -------
    array
        SFR (Msun/yr).
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
    t_lookback : array
        Lookback time (yr).
    log_peak_sfr : float
        log10 of peak SFR (Msun/yr). Peak occurs at t = start + tau.
    tau : float
        Timescale (yr). Peak is at start + tau.
    start : float
        Start lookback time (yr). Default: 0 (present).

    Returns
    -------
    array
        SFR (Msun/yr).
    """
    peak_sfr = 10.0**log_peak_sfr
    dt = t_lookback - start
    ratio = dt / tau
    sfr = peak_sfr * ratio * jnp.exp(-ratio + 1.0)
    return jnp.where(dt >= 0, jnp.maximum(sfr, 0.0), 0.0)


def triweight_burst(
    t_lookback: jnp.ndarray,
    log_tpeak_myr: float,
    log_tmax_myr: float,
) -> jnp.ndarray:
    """Triweight burst kernel in log-age space (Zacharegkas+2025).

    A compact-support kernel: K(x) = (35/96)(1 - (x/3)^2)^3 for |x| < 3.
    Applied in log10(age/Myr) space centered at log_tpeak_myr with
    half-width log_tmax_myr.

    This is a *shape-only* function (unitless, integrates to ~1).
    The burst amplitude is set by the burst mixture fraction in the
    composition step.

    Parameters
    ----------
    t_lookback : array
        Lookback time (yr).
    log_tpeak_myr : float
        log10 of burst peak time (Myr). Center of the kernel.
    log_tmax_myr : float
        log10 of burst duration (Myr). Controls kernel width.

    Returns
    -------
    array
        Unnormalized burst shape (non-negative).

    References
    ----------
    Zacharegkas+2025 (arXiv:2506.19919).
    """
    log_age_myr = jnp.log10(t_lookback / 1e6)  # yr → Myr in log10
    x = (log_age_myr - log_tpeak_myr) / jnp.maximum(log_tmax_myr, 0.01)
    # Triweight kernel: (35/96)(1 - (x/3)^2)^3, compact support |x| < 3
    u = x / 3.0
    kernel = (35.0 / 96.0) * jnp.maximum(1.0 - u**2, 0.0) ** 3
    return kernel


# ---------------------------------------------------------------------------
# Legacy functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def delayed_tau(t_lookback: jnp.ndarray, tau: float, norm: float) -> jnp.ndarray:
    """Delayed-tau SFH: SFR(t) = norm * t * exp(-t/tau). Peaks at t=tau."""
    return norm * t_lookback * jnp.exp(-t_lookback / tau)


def powerlaw_sfh(
    t_lookback: jnp.ndarray, alpha: float, norm: float, t_ref: float = 1e8
) -> jnp.ndarray:
    """Power-law SFH: SFR(t) = norm * (t/t_ref)^alpha."""
    return norm * (t_lookback / t_ref) ** alpha
