"""Generalized two-component dust attenuation model.

Extends the Charlot & Fall (2000) framework with:
- Pluggable attenuation curves (via laws.py registry)
- Different laws for birth cloud and diffuse ISM
- f_obscuration parameter (Lower 2022 / Zacharegkas 2025)

The two-component structure separates attenuation into:
- Birth cloud: affects young stars (age < t_birth), tau_v1
- Diffuse ISM: affects all stars, tau_v2

The wavelength dependence k(lambda) is now pluggable:
  tau(lambda, age) = [w(age) * tau_v1 * k_bc(lambda) + tau_v2 * k_diff(lambda)]

With f_obscuration (Lower 2022):
  transmission = f_obs + (1 - f_obs) * exp(-tau)

This generalizes the original Charlot & Fall to arbitrary
attenuation curves while preserving the physical birth cloud / ISM
decomposition.

References
----------
- Charlot & Fall 2000, ApJ, 539, 718 (two-component structure)
- Lower et al. 2022, ApJ, 931, 14 (f_obscuration, clumpy geometry)
- Zacharegkas et al. 2025, arXiv:2506.19919 (DSPS implementation)
"""

import jax
import jax.numpy as jnp

from diffsed.models.dust.laws import get_dust_law


def precompute_dust_age_weights(
    age_grid: jnp.ndarray,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
) -> jnp.ndarray:
    """Precompute the age-dependent birth-cloud weight (sigmoid).

    Parameters
    ----------
    age_grid : array, shape (n_ages,)
        Stellar population ages (yr).
    t_birth : float
        Birth cloud dispersal age (yr). Default 1e7 (10 Myr).
    transition_width : float
        Sigmoid width in dex. Default 0.3.

    Returns
    -------
    array, shape (n_ages,)
        Sigmoid weight: 1 for young stars, 0 for old.
    """
    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    sigmoid_arg = -(log_age - log_t_birth) / transition_width
    return jax.nn.sigmoid(sigmoid_arg)


def two_component_dust(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = 0.0,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
    **law_params,
) -> jnp.ndarray:
    """Generalized two-component dust attenuation.

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
    law_bc : str
        Attenuation curve for birth cloud. Default "power_law".
    law_diff : str
        Attenuation curve for diffuse ISM. Default "power_law".
        Set to same as law_bc for uniform curve (the default behavior).
    f_obscuration : float
        Fraction of flux that passes unattenuated [0, 1].
        Accounts for clumpy dust geometry (Lower 2022).
        0 = standard full-slab (default). 1 = no dust.
    t_birth : float
        Birth cloud dispersal age (yr). Default 1e7 (10 Myr).
    transition_width : float
        Sigmoid width in dex. Default 0.3.
    **law_params
        Additional parameters passed to dust law functions:
        n_slope (power_law), dust_bump_strength, dust_delta (kriek_conroy),
        dust_Rv (cardelli), etc.

    Returns
    -------
    array, shape (n_ages, n_wave)
        Multiplicative attenuation factor in [0, 1].
    """
    # Get attenuation curves k(lambda) = A(lambda)/A(V)
    k_bc_fn = get_dust_law(law_bc)
    k_diff_fn = get_dust_law(law_diff)

    k_bc = k_bc_fn(wavelength, **law_params)      # (n_wave,)
    k_diff = k_diff_fn(wavelength, **law_params)   # (n_wave,)

    # Age-dependent weight: sigmoid from 1 (young) to 0 (old)
    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    sigmoid_arg = -(log_age - log_t_birth) / transition_width
    weight = jax.nn.sigmoid(sigmoid_arg)  # (n_ages,)

    # Optical depth: tau(lambda, age) = w(age)*tau_v1*k_bc + tau_v2*k_diff
    tau_lambda = (
        weight[:, None] * tau_v1 * k_bc[None, :]
        + tau_v2 * k_diff[None, :]
    )  # (n_ages, n_wave)

    # Transmission with f_obscuration (Lower 2022)
    transmission = f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)

    return transmission


def two_component_dust_fast(
    wavelengths: jnp.ndarray,
    dust_age_weights: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    """Fast path using precomputed age weights.

    Parameters
    ----------
    wavelengths : array, shape (n_filters,)
        Wavelengths at which to evaluate (rest-frame Angstrom).
    dust_age_weights : array, shape (n_ages,)
        Precomputed sigmoid weights from precompute_dust_age_weights.
    tau_v1 : float
        V-band optical depth of birth cloud.
    tau_v2 : float
        V-band optical depth of diffuse ISM.
    law_bc : str
        Attenuation curve for birth cloud.
    law_diff : str
        Attenuation curve for diffuse ISM.
    f_obscuration : float
        Fraction of unattenuated flux [0, 1].
    **law_params
        Additional parameters for dust law functions.

    Returns
    -------
    array, shape (n_ages, n_filters)
        Multiplicative attenuation factor.
    """
    k_bc_fn = get_dust_law(law_bc)
    k_diff_fn = get_dust_law(law_diff)

    k_bc = k_bc_fn(wavelengths, **law_params)
    k_diff = k_diff_fn(wavelengths, **law_params)

    tau_v_eff_bc = dust_age_weights * tau_v1      # (n_ages,)
    tau_lambda = (
        tau_v_eff_bc[:, None] * k_bc[None, :]
        + tau_v2 * k_diff[None, :]
    )

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)
