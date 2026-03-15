"""Charlot & Fall (2000) two-component dust attenuation.

Separates attenuation into birth-cloud (young stars) and diffuse ISM
(all stars). Uses a smooth sigmoid transition for gradient compatibility.

Reference: Charlot, S. & Fall, S. M. 2000, ApJ, 539, 718
"""

import jax
import jax.numpy as jnp


def charlot_fall(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    n_slope: float = -0.7,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
) -> jnp.ndarray:
    """Smooth Charlot & Fall dust attenuation.

    tau_lambda(t) = [w(t)*tau_v1 + tau_v2] * (lambda/5500)^n

    where w(t) is a sigmoid transitioning from 1 (young) to 0 (old)
    around t_birth. The attenuation factor is exp(-tau_lambda).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid (Angstrom).
    age_grid : array, shape (n_ages,)
        Stellar population ages (yr).
    tau_v1 : float
        V-band optical depth of birth cloud.
    tau_v2 : float
        V-band optical depth of diffuse ISM.
    n_slope : float
        Attenuation curve power-law index. Default -0.7.
    t_birth : float
        Birth cloud dispersal age (yr). Default 1e7 (10 Myr).
    transition_width : float
        Sigmoid width in dex. Default 0.3.

    Returns
    -------
    array, shape (n_ages, n_wave)
        Multiplicative attenuation factor exp(-tau_lambda).
    """
    wave_ratio = (wavelength / 5500.0) ** n_slope

    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    sigmoid_arg = -(log_age - log_t_birth) / transition_width
    weight = jax.nn.sigmoid(sigmoid_arg)

    tau_v_eff = weight * tau_v1 + tau_v2
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
    """Evaluate Charlot & Fall dust at specific wavelengths per filter.

    Unlike charlot_fall() which evaluates on the full wavelength grid,
    this evaluates at a small set of wavelengths (e.g., filter effective
    wavelengths). Used for approximate photometry (Zacharegkas+2025 Eq. 6).

    Parameters
    ----------
    wavelengths : array, shape (n_filters,)
        Wavelengths at which to evaluate dust (rest-frame Angstrom).
    age_grid : array, shape (n_ages,)
        Stellar population ages (yr).
    tau_v1 : float
        V-band optical depth of birth cloud.
    tau_v2 : float
        V-band optical depth of diffuse ISM.
    n_slope : float
        Attenuation curve power-law index. Default -0.7.
    t_birth : float
        Birth cloud dispersal age (yr). Default 1e7.
    transition_width : float
        Sigmoid width in dex. Default 0.3.

    Returns
    -------
    array, shape (n_ages, n_filters)
        Multiplicative attenuation factor exp(-tau_lambda).
    """
    wave_ratio = (wavelengths / 5500.0) ** n_slope  # (n_filters,)

    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    sigmoid_arg = -(log_age - log_t_birth) / transition_width
    weight = jax.nn.sigmoid(sigmoid_arg)  # (n_ages,)

    tau_v_eff = weight * tau_v1 + tau_v2  # (n_ages,)
    tau_lambda = tau_v_eff[:, None] * wave_ratio[None, :]  # (n_ages, n_filters)

    return jnp.exp(-tau_lambda)


def charlot_fall_hard(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    n_slope: float = -0.7,
    t_birth: float = 1e7,
) -> jnp.ndarray:
    """Standard step-function Charlot & Fall (for comparison/testing)."""
    wave_ratio = (wavelength / 5500.0) ** n_slope
    tau_young = (tau_v1 + tau_v2) * wave_ratio
    tau_old = tau_v2 * wave_ratio
    is_young = age_grid[:, None] < t_birth
    tau_lambda = jnp.where(is_young, tau_young[None, :], tau_old[None, :])
    return jnp.exp(-tau_lambda)
