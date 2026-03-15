"""Smooth parametric mean star formation history components.

The GP x(t) has zero mean, so the overall SFH shape comes from these
functions. The full SFH is: SFR(t) = mean(t) * exp(x(t) - K(0)/2),
where -K(0)/2 is the lognormal correction preserving the linear-SFR mean.

Convention: t_lookback in years, SFR returned in Msun/yr.
All functions are pure JAX and JIT-compatible.
"""

import jax.numpy as jnp


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


def delayed_tau(t_lookback: jnp.ndarray, tau: float, norm: float) -> jnp.ndarray:
    """Delayed-tau SFH: SFR(t) = norm * t * exp(-t/tau). Peaks at t=tau."""
    return norm * t_lookback * jnp.exp(-t_lookback / tau)


def constant_sfh(t_lookback: jnp.ndarray, norm: float) -> jnp.ndarray:
    """Constant SFR."""
    return jnp.broadcast_to(norm, jnp.shape(t_lookback))


def powerlaw_sfh(
    t_lookback: jnp.ndarray, alpha: float, norm: float, t_ref: float = 1e8
) -> jnp.ndarray:
    """Power-law SFH: SFR(t) = norm * (t/t_ref)^alpha."""
    return norm * (t_lookback / t_ref) ** alpha
