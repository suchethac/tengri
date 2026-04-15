"""Gaussian Process realizations from Power Spectral Density functions.

Two modes:
- Stochastic: generate_gp_fourier() draws random GP realizations (for mocks)
- Deterministic: gp_from_xi() maps a fixed latent vector to a GP (for inference)

The GP is defined on a uniform log-age grid. The IFFT-based generation
implements the NIFTy correlated field model: s = IFFT(sqrt(P) * xi).

Key design: the latent vector xi ~ N(0, I) is the standardized variable
that samplers (geoVI, NUTS) explore. The PSD encodes the prior correlation.
"""

import jax
import jax.numpy as jnp
from jax import random


def make_log_age_grid(
    n_grid: int = 256, log_age_min: float = 6.0, log_age_max: float = 10.14
) -> jnp.ndarray:
    """Create uniform grid in log10(age/yr).

    Default range: 1 Myr to ~13.8 Gyr.

    Parameters
    ----------
    n_grid : int
        Number of grid points (should be even for FFT efficiency).
    log_age_min : float
        Minimum log10(age/yr). Default 6.0 = 1 Myr.
    log_age_max : float
        Maximum log10(age/yr). Default 10.14 ~ 13.8 Gyr.

    Returns
    -------
    array, shape (n_grid,)
        Uniform grid in log10(age/yr).
    """
    return jnp.linspace(log_age_min, log_age_max, n_grid)


def gp_from_xi(xi: jnp.ndarray, sqrt_power: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """Deterministic GP realization from standardized latent vector.

    Implements the IFT correlated field model: s = IFFT(sqrt(P) * xi_hat).
    This is the core function used during inference — the sampler proposes
    xi ~ N(0, I), and this maps it to a correlated GP realization.

    Uses rfft to map real-valued xi to Hermitian-symmetric Fourier
    coefficients. This preserves the correct variance normalization:
    E[|rfft(xi)_k|^2] = N, so with sqrt_power = sqrt(P/dx), we get
    Var[x] = integral P(f) df, as required.

    Parameters
    ----------
    xi : array, shape (n_points,)
        Standardized latent vector (xi ~ N(0, I) under the prior).
    sqrt_power : array, shape (n_freq,)
        Amplitude operator sqrt(P(omega) / d_grid) at rfft frequencies.
        Pre-compute with psd_to_sqrt_power().
    n_points : int
        Number of grid points.

    Returns
    -------
    array, shape (n_points,)
        GP realization on the log-age grid.
    """
    xi_hat = jnp.fft.rfft(xi)
    coeffs = sqrt_power * xi_hat
    return jnp.fft.irfft(coeffs, n=n_points)


def generate_gp_fourier(key: jax.Array, sqrt_power: jnp.ndarray, n_points: int) -> jnp.ndarray:
    """Stochastic GP realization (for mock generation).

    Draws xi ~ N(0, I) and applies the correlated field model.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    sqrt_power : array, shape (n_freq,)
        Amplitude operator at rfft frequencies.
    n_points : int
        Number of grid points.

    Returns
    -------
    array, shape (n_points,)
        GP realization.
    """
    xi = random.normal(key, shape=(n_points,))
    return gp_from_xi(xi, sqrt_power, n_points)


def generate_gp_batch(
    key: jax.Array, sqrt_power: jnp.ndarray, n_points: int, n_realizations: int
) -> jnp.ndarray:
    """Batch of GP realizations via vmap.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    sqrt_power : array, shape (n_freq,)
        Amplitude operator.
    n_points : int
        Grid size.
    n_realizations : int
        Number of independent realizations.

    Returns
    -------
    array, shape (n_realizations, n_points)
        Batch of GP realizations.
    """
    keys = random.split(key, n_realizations)
    return jax.vmap(lambda k: generate_gp_fourier(k, sqrt_power, n_points))(keys)


def compute_sqrt_power_drw(
    n_points: int, d_log_age: float, psd_sigma: float, psd_tau_yr: float, log_age_ref: float = 8.0
) -> jnp.ndarray:
    """Pre-compute the DRW amplitude operator for the log-age grid.

    Converts the DRW PSD from physical frequency (rad/yr) to log-age
    frequency (rad/dex) using the Jacobian correction:

        P_u(q) = P_t(q / (t_ref * ln10)) / (t_ref * ln10)

    where t_ref = 10^log_age_ref is a reference time and q is the
    log-age-space angular frequency.

    Parameters
    ----------
    n_points : int
        Grid size.
    d_log_age : float
        Grid spacing in dex.
    psd_sigma : float
        DRW PSD amplitude.
    psd_tau_yr : float
        DRW damping timescale (yr).
    log_age_ref : float
        Reference log10(age/yr) for Jacobian correction. Default 8.0 (100 Myr).

    Returns
    -------
    array, shape (n_freq,)
        sqrt(P_u(q) / d_log_age) at rfft frequencies.
    """
    from tengri.components.sfh.psd_models import psd_drw, psd_to_sqrt_power

    t_ref = 10.0**log_age_ref
    ln10 = jnp.log(10.0)

    # FFT frequencies in log-age space (rad/dex)
    freqs = jnp.fft.rfftfreq(n_points, d=d_log_age)
    q = 2.0 * jnp.pi * freqs

    # Convert to physical frequency (rad/yr)
    omega_phys = q / (t_ref * ln10)

    # Evaluate PSD in physical domain and apply Jacobian
    p_phys = psd_drw(omega_phys, psd_sigma, psd_tau_yr)
    p_logage = p_phys / (t_ref * ln10)

    return psd_to_sqrt_power(p_logage, d_log_age)
