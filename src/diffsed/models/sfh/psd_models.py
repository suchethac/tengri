"""Power spectral density models for stochastic star formation histories.

All functions are pure JAX and JIT-compatible. Each PSD function takes
frequency arrays and returns power spectral density P(omega).

The DRW (damped random walk / Lorentzian) is the primary model used
in the paper; others are provided for extensibility.

References
----------
- DRW: Munoz+2026 (arXiv:2601.07912)
- Extended Regulator: Tacchella+2020, Caplar & Tacchella 2019
- Flex-PSD: Burnham+2026 (arXiv:2601.20930)
- Matern: generalizes DRW (nu=0.5 recovers DRW)
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Primary: Damped Random Walk
# ---------------------------------------------------------------------------

def psd_drw(omega: jnp.ndarray, sigma_ps: float, tau_ps: float) -> jnp.ndarray:
    """Damped random walk (Lorentzian) power spectral density.

    P(omega) = sigma_PS^2 * tau_PS / (1 + (tau_PS * omega)^2)

    Parameters
    ----------
    omega : array
        Angular frequency (rad / yr).
    sigma_ps : float
        PSD amplitude.
    tau_ps : float
        Characteristic damping timescale (yr).

    Returns
    -------
    array
        Power spectral density at each omega.
    """
    return sigma_ps**2 * tau_ps / (1.0 + (tau_ps * omega) ** 2)


def drw_acf(delta_t: jnp.ndarray, sigma_ps: float, tau_ps: float) -> jnp.ndarray:
    """Analytic autocorrelation function for DRW.

    xi(dt) = (sigma_PS^2 / 2) * exp(-|dt| / tau_PS)

    Parameters
    ----------
    delta_t : array
        Time lag (yr).
    sigma_ps : float
        PSD amplitude.
    tau_ps : float
        Damping timescale (yr).

    Returns
    -------
    array
        Autocovariance at each lag.
    """
    return 0.5 * sigma_ps**2 * jnp.exp(-jnp.abs(delta_t) / tau_ps)


def drw_variance(sigma_ps: float) -> float:
    """Stationary variance of DRW: sigma_x^2 = sigma_PS^2 / 2."""
    return 0.5 * sigma_ps**2


def psd_to_sqrt_power(psd_values: jnp.ndarray, d_grid: float) -> jnp.ndarray:
    """Convert PSD values to amplitude operator sqrt(P / d_grid).

    This is the factor that multiplies the standardized latent vector xi
    in Fourier space: IFFT(sqrt(P/d) * xi_hat).

    Parameters
    ----------
    psd_values : array
        PSD evaluated at rfft frequencies.
    d_grid : float
        Grid spacing (needed for FFT normalization).

    Returns
    -------
    array
        Amplitude operator values.
    """
    return jnp.sqrt(jnp.maximum(psd_values, 1e-30) / d_grid)


# ---------------------------------------------------------------------------
# Alternative PSD models
# ---------------------------------------------------------------------------

def psd_matern(omega: jnp.ndarray, variance: float,
               length_scale: float, nu: float) -> jnp.ndarray:
    """Matern PSD in 1D. Setting nu=0.5 recovers DRW."""
    from jax.scipy.special import gammaln

    lam = 2.0 * nu / length_scale**2
    log_norm = (
        jnp.log(2.0 * jnp.sqrt(jnp.pi))
        + gammaln(nu + 0.5)
        - gammaln(nu)
        + nu * jnp.log(2.0 * nu)
        - 2.0 * nu * jnp.log(length_scale)
    )
    log_spectral = -(nu + 0.5) * jnp.log(lam + omega**2)
    return variance * jnp.exp(log_norm + log_spectral)


def psd_extended_regulator(f: jnp.ndarray, s_reg: float, tau_in: float,
                           tau_eq: float, s_dyn: float,
                           tau_dyn: float) -> jnp.ndarray:
    """Extended regulator PSD (Tacchella+2020). Uses cyclic frequency f."""
    two_pi_f = 2.0 * jnp.pi * f
    regulator = s_reg**2 / (
        (1.0 + (tau_in * two_pi_f) ** 2)
        * (1.0 + (tau_eq * two_pi_f) ** 2)
    )
    dynamical = s_dyn**2 / (1.0 + (tau_dyn * two_pi_f) ** 2)
    return regulator + dynamical
