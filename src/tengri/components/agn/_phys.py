"""Shared physical utility functions for AGN sub-models.

Extracted from disc.py, torus.py, and skirtor.py to eliminate
three identical copies of the Planck function and related helpers.

Fundamental constants are imported from :mod:`tengri.utils.physics`,
which documents their SI→CGS derivations and CODATA 2018 / IAU 2015 sources.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.physics_constants import (
    AA_TO_CM as ANGSTROM_CM,
    C_CGS as C_LIGHT,
    C_KM_S,
    H_PLANCK,
    K_BOLTZ,
    L_SUN as LSUN_ERG,
)

__all__ = [
    "ANGSTROM_CM",
    "C_LIGHT",
    "H_PLANCK",
    "K_BOLTZ",
    "LSUN_ERG",
    "gaussian_line_profile",
    "lines_to_sed",
    "planck_lnu",
    "ring_area",
    "wavelength_to_nu",
]


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
    # Cast to float64 to prevent nu**3 overflow.  At λ ~ 100 Å,
    # ν ≈ 3×10¹⁷ Hz so ν³ ≈ 2.7×10⁵² — fine in float64 but far
    # beyond float32 max (~3.4×10³⁸).
    nu64 = jnp.asarray(nu, dtype=jnp.float64)
    t_safe = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float64), 1.0)

    # Clamp x = hν/kT to [1e-10, 500].  At x=500, expm1 ≈ 1.4e217
    # (finite in float64).  The clamp avoids both expm1 overflow and
    # division-by-zero, and keeps gradients finite everywhere.
    x = jnp.clip(H_PLANCK * nu64 / (K_BOLTZ * t_safe), 1e-10, 500.0)
    return 2.0 * H_PLANCK * nu64**3 / C_LIGHT**2 / jnp.expm1(x)


# ---------------------------------------------------------------------------
# Wavelength ↔ frequency conversion
# ---------------------------------------------------------------------------


def wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Ångström) to frequency (Hz)."""
    return C_LIGHT / (wavelength_angstrom * ANGSTROM_CM)


# ---------------------------------------------------------------------------
# Gaussian emission-line profile (scalar kernel)
# ---------------------------------------------------------------------------


def gaussian_line_profile(
    wavelength: jnp.ndarray,
    line_center: float,
    fwhm_kms: float,
) -> jnp.ndarray:
    """Normalized Gaussian line profile in wavelength space.

    Returns the profile per unit frequency so that the integral over
    d(nu) equals 1.  Used by NLR and BLR modules via ``jax.vmap`` over
    a line list.

    Parameters
    ----------
    wavelength : array
        Wavelength grid [Angstrom].
    line_center : float
        Line center wavelength [Angstrom].
    fwhm_kms : float
        FWHM in km/s.

    Returns
    -------
    array
        Profile [Hz^-1], normalized so that integral over d(nu) = 1.
    """
    sigma_ang = line_center * (fwhm_kms / C_KM_S) / 2.3548200450309493
    sigma_ang = jnp.maximum(sigma_ang, 0.01)

    phi_lam = jnp.exp(-0.5 * ((wavelength - line_center) / sigma_ang) ** 2) / (
        sigma_ang * jnp.sqrt(2.0 * jnp.pi)
    )

    # Convert per-Angstrom to per-Hz: phi_nu = phi_lam * lam^2 / c
    c_ang = C_LIGHT / ANGSTROM_CM
    phi_nu = phi_lam * wavelength**2 / c_ang

    return phi_nu


# ---------------------------------------------------------------------------
# Disc ring projected area (R&L 1979 Eq 1.6 geometry)
# ---------------------------------------------------------------------------


def ring_area(r_cm: float, dr_cm: float, cos_inc: float) -> float:
    """Projected area of an annular disc ring.

    Encodes ``dL_nu = pi * B_nu * 2*pi*r*dr * cos(i)`` — the hemisphere
    integral of spectral radiance over a flat annular ring
    (Rybicki & Lightman 1979, Eq 1.6).

    Parameters
    ----------
    r_cm : float
        Ring radius [cm].
    dr_cm : float
        Ring width [cm].
    cos_inc : float
        Cosine of inclination angle.

    Returns
    -------
    float
        ``pi * 2*pi * r * dr * max(cos_inc, 0.01)`` [cm^2 sr].
    """
    return jnp.pi * 2.0 * jnp.pi * r_cm * dr_cm * jnp.maximum(cos_inc, 0.01)


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
    from tengri.utils.physics_constants import C_KM_S

    fwhm_aa = line_wavelengths * fwhm_kms / C_KM_S
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
