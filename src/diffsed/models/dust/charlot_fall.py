"""Charlot & Fall (2000) power-law dust attenuation (backward compatibility).

This module provides the original power-law-only Charlot & Fall functions
used by the fast precomputed path and existing tests. For the generalized
model with pluggable curves and f_obscuration, use ``attenuation.py``.

All new code should import from ``diffsed.models.dust.attenuation`` directly.
"""

import jax
import jax.numpy as jnp

# Re-export from the canonical location
from diffsed.models.dust.attenuation import precompute_dust_age_weights  # noqa: F401


def charlot_fall(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    n_slope: float = -0.7,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
) -> jnp.ndarray:
    """Original Charlot & Fall (2000) with power-law curve.

    Equivalent to ``two_component_dust(..., law_bc="power_law", law_diff="power_law")``.
    """
    wave_ratio = (wavelength / 5500.0) ** n_slope
    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    weight = jax.nn.sigmoid(-(log_age - log_t_birth) / transition_width)
    tau_v_eff = weight * tau_v1 + tau_v2
    tau_lambda = tau_v_eff[:, None] * wave_ratio[None, :]
    return jnp.exp(-tau_lambda)


def charlot_fall_at_wavelengths_fast(
    wavelengths: jnp.ndarray,
    dust_age_weights: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    n_slope: float = -0.7,
) -> jnp.ndarray:
    """Fast path with precomputed age weights (power-law only)."""
    wave_ratio = (wavelengths / 5500.0) ** n_slope
    tau_v_eff = dust_age_weights * tau_v1 + tau_v2
    tau_lambda = tau_v_eff[:, None] * wave_ratio[None, :]
    return jnp.exp(-tau_lambda)


def charlot_fall_at_wavelengths(
    wavelengths: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    n_slope: float = -0.7,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
) -> jnp.ndarray:
    """Evaluate at specific wavelengths (power-law only)."""
    wave_ratio = (wavelengths / 5500.0) ** n_slope
    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    weight = jax.nn.sigmoid(-(log_age - log_t_birth) / transition_width)
    tau_v_eff = weight * tau_v1 + tau_v2
    tau_lambda = tau_v_eff[:, None] * wave_ratio[None, :]
    return jnp.exp(-tau_lambda)


def charlot_fall_hard(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    n_slope: float = -0.7,
    t_birth: float = 1e7,
) -> jnp.ndarray:
    """Step-function variant (for comparison/testing)."""
    wave_ratio = (wavelength / 5500.0) ** n_slope
    tau_young = (tau_v1 + tau_v2) * wave_ratio
    tau_old = tau_v2 * wave_ratio
    is_young = age_grid[:, None] < t_birth
    tau_lambda = jnp.where(is_young, tau_young[None, :], tau_old[None, :])
    return jnp.exp(-tau_lambda)
