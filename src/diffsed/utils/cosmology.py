"""Minimal flat-LCDM cosmology utilities (pure JAX).

These are simple implementations for the forward model. For production
use with real data, consider using astropy.cosmology or jax-cosmo.
"""

import jax
import jax.numpy as jnp

# Planck 2018 defaults
DEFAULT_H0 = 67.4  # km/s/Mpc
DEFAULT_OM0 = 0.315
C_KM_S = 2.998e5  # speed of light in km/s
MPC_TO_CM = 3.0857e24  # Mpc to cm


@jax.jit
def luminosity_distance(z: float, h0: float = DEFAULT_H0, om0: float = DEFAULT_OM0) -> float:
    """Luminosity distance for flat LCDM (cm).

    Uses 100-point Gauss-Legendre quadrature for the comoving integral.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float
        Hubble constant (km/s/Mpc).
    om0 : float
        Matter density parameter.

    Returns
    -------
    float
        Luminosity distance in cm.
    """
    n_quad = 100
    z_grid = jnp.linspace(0.0, z, n_quad + 1)

    ol0 = 1.0 - om0
    e_z = jnp.sqrt(om0 * (1.0 + z_grid) ** 3 + ol0)
    integrand = 1.0 / e_z

    # Trapezoidal rule
    dc = (C_KM_S / h0) * jnp.trapezoid(integrand, z_grid)
    dl = dc * (1.0 + z)
    return dl * MPC_TO_CM


@jax.jit
def age_at_z(z: float, h0: float = DEFAULT_H0, om0: float = DEFAULT_OM0) -> float:
    """Age of universe at redshift z (yr).

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float
        Hubble constant (km/s/Mpc).
    om0 : float
        Matter density parameter.

    Returns
    -------
    float
        Age of universe in years.
    """
    n_quad = 200
    z_max = 30.0
    z_grid = jnp.linspace(z, z_max, n_quad + 1)

    ol0 = 1.0 - om0
    e_z = jnp.sqrt(om0 * (1.0 + z_grid) ** 3 + ol0)
    integrand = 1.0 / ((1.0 + z_grid) * e_z)

    # H0 in 1/s = h0 * 1e5 / (Mpc_in_cm)
    h0_inv_yr = 1.0 / (h0 * 1e5 / (3.0857e24 / 3.156e7))

    return h0_inv_yr * jnp.trapezoid(integrand, z_grid)
