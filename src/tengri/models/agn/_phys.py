"""Shared physical utility functions for AGN sub-models.

Extracted from disc.py, torus.py, and skirtor.py to eliminate
three identical copies of the Planck function and related helpers.
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------

H_PLANCK: float = 6.62607015e-27  # Planck constant [erg s]
K_BOLTZ: float = 1.380649e-16  # Boltzmann constant [erg K^-1]
C_LIGHT: float = 2.99792458e10  # Speed of light [cm s^-1]
ANGSTROM_CM: float = 1e-8  # Ångström → cm
LSUN_ERG: float = 3.828e33  # Solar luminosity [erg s^-1]


# ---------------------------------------------------------------------------
# Planck function
# ---------------------------------------------------------------------------


def planck_lnu(
    nu: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_nu(T) in erg s^-1 cm^-2 Hz^-1 sr^-1.

    Uses log-space exponent to avoid overflow at low T or high nu.
    Returns 0 where temperature <= 0 (JIT-safe).

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    temperature : float
        Temperature [K].

    Returns
    -------
    array
        B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].
    """
    t_safe = jnp.maximum(temperature, 1.0)
    x = H_PLANCK * nu / (K_BOLTZ * t_safe)
    x_clip = jnp.clip(x, 0.0, 500.0)
    prefactor = 2.0 * H_PLANCK * nu**3 / C_LIGHT**2
    return prefactor / (jnp.exp(x_clip) - 1.0)


# ---------------------------------------------------------------------------
# Wavelength ↔ frequency conversion
# ---------------------------------------------------------------------------


def wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Ångström) to frequency (Hz)."""
    return C_LIGHT / (wavelength_angstrom * ANGSTROM_CM)


# ---------------------------------------------------------------------------
# Line list → SED convolution
# ---------------------------------------------------------------------------


def lines_to_sed(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    wave_obs: jnp.ndarray,
    fwhm_kms: float = 500.0,
) -> jnp.ndarray:
    """Convolve a list of delta-function emission lines onto a wavelength grid.

    Each line is broadened with a Gaussian whose FWHM is ``fwhm_kms`` km/s.
    This is a pure JAX function, JIT-compatible and differentiable.

    Parameters
    ----------
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line centre wavelengths [Angstrom].
    line_luminosities : array, shape (n_lines,)
        Per-line luminosities [Lsun].
    wave_obs : array, shape (n_wave,)
        Output wavelength grid [Angstrom].
    fwhm_kms : float
        Line FWHM in km/s.  Default 500.

    Returns
    -------
    array, shape (n_wave,)
        L_nu on ``wave_obs`` grid [erg s^-1 Hz^-1].
    """
    # sigma = FWHM / (2*sqrt(2*ln2)) in Angstrom at each line centre
    fwhm_aa = line_wavelengths * fwhm_kms / 2.99792458e5  # c in km/s
    sigma_aa = fwhm_aa / 2.3548200450309493  # 2*sqrt(2*ln2)

    # Gaussian profiles: shape (n_wave, n_lines)
    dwave = wave_obs[:, None] - line_wavelengths[None, :]
    profiles = jnp.exp(-0.5 * (dwave / sigma_aa[None, :]) ** 2)

    # Normalise each profile to unit integrated flux
    norm = sigma_aa * jnp.sqrt(2.0 * jnp.pi)  # (n_lines,)
    profiles = profiles / norm[None, :]  # (n_wave, n_lines)

    # Weighted sum -> L_lambda [Lsun/A]
    l_lambda = profiles @ line_luminosities  # (n_wave,)

    # Convert L_lambda [Lsun/A] -> L_nu [erg/s/Hz] via c/lambda^2 factor
    l_nu = l_lambda * LSUN_ERG * wave_obs**2 * ANGSTROM_CM / C_LIGHT
    return l_nu
