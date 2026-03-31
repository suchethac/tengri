"""Polar dust extinction and greybody reemission for AGN.

Implements the X-CIGALE polar dust model (Yang et al. 2020, Section 2.2.2):
SMC extinction applied to Type 1 AGN sightlines, with energy-conserving
greybody FIR reemission.

The Type 1/2 boundary uses a smooth sigmoid transition for differentiability.

References
----------
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust)
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
- Pei 1992, ApJ, 395, 130 (SMC parameterization used here)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.models.dust.attenuation import smc as smc_extinction_curve

# Physical constants (CGS / Angstrom-compatible)
_H_PLANCK = 6.62607015e-27  # erg s
_K_BOLTZ = 1.380649e-16  # erg / K
_C_CGS = 2.99792458e10  # cm / s
_C_AA = 2.99792458e18  # Angstrom / s

# SMC R_V from Pei (1992)
_RV_SMC = 2.93

# Default sigmoid sharpness for Type 1/2 boundary
_SIGMOID_SHARPNESS = 20.0


def _type1_mask(
    cos_inc: float,
    opening_angle_deg: float,
    sharpness: float = _SIGMOID_SHARPNESS,
) -> jnp.ndarray:
    """Smooth sigmoid mask: 1 for Type 1 (face-on), 0 for Type 2 (edge-on).

    Parameters
    ----------
    cos_inc : float
        Cosine of inclination angle. 1 = face-on, 0 = edge-on.
    opening_angle_deg : float
        Torus half-opening angle in degrees (measured from equator).
    sharpness : float
        Sigmoid steepness. Higher = sharper transition. Default 20.

    Returns
    -------
    mask : scalar
        Value in [0, 1]. ~1 for Type 1, ~0 for Type 2, ~0.5 at boundary.
    """
    cos_threshold = jnp.cos(jnp.radians(90.0 - opening_angle_deg))
    return jax.nn.sigmoid((cos_inc - cos_threshold) * sharpness)


def polar_dust_extinction(
    l_nu: jnp.ndarray,
    wavelength: jnp.ndarray,
    cos_inc: float,
    opening_angle_deg: float,
    ebv: float,
    law: str = "smc",
    sharpness: float = _SIGMOID_SHARPNESS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply polar dust extinction to AGN luminosity.

    SMC extinction is applied only to Type 1 sightlines (face-on), with a
    smooth sigmoid transition at the Type 1/2 boundary.

    Parameters
    ----------
    l_nu : array, shape (n_wave,)
        Input luminosity density [Lsun/Hz or any consistent unit].
    wavelength : array, shape (n_wave,)
        Wavelength in Angstrom.
    cos_inc : float
        Cosine of inclination. 1 = face-on (Type 1), 0 = edge-on (Type 2).
    opening_angle_deg : float
        Torus half-opening angle in degrees (from equator).
    ebv : float
        Colour excess E(B-V) for the polar dust. 0 = no extinction.
    law : str
        Extinction law name. Currently only ``"smc"`` is supported.
    sharpness : float
        Sigmoid steepness at the Type 1/2 boundary.

    Returns
    -------
    l_nu_attenuated : array, shape (n_wave,)
        Attenuated luminosity density.
    l_absorbed : array, shape (n_wave,)
        Absorbed luminosity density (per wavelength bin). Always >= 0.
    """
    # k(lambda) / k(V), normalized at 5500 A
    k_lambda = smc_extinction_curve(wavelength)

    # A(lambda) = E(B-V) * R_V * k(lambda)
    # Transmission: 10^{-0.4 * A(lambda)} = exp(-0.921 * A(lambda))
    tau_lambda = 0.921 * ebv * _RV_SMC * k_lambda
    extinction_factor = jnp.exp(-tau_lambda)  # fraction transmitted

    # Type 1 mask: 1 for face-on (extinct), 0 for edge-on (no effect)
    mask = _type1_mask(cos_inc, opening_angle_deg, sharpness)

    # Effective transmission: mix between full extinction (Type 1) and
    # no extinction (Type 2)
    effective_transmission = 1.0 - mask * (1.0 - extinction_factor)

    l_nu_attenuated = l_nu * effective_transmission
    l_absorbed = l_nu - l_nu_attenuated

    # Ensure non-negative (numerical safety)
    l_absorbed = jnp.maximum(l_absorbed, 0.0)

    return l_nu_attenuated, l_absorbed


def _planck_nu(wavelength: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Planck function B_nu(T) in CGS units [erg/s/cm^2/Hz/sr].

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    b_nu : array
        Planck function values. Same shape as wavelength.
    """
    nu = _C_AA / wavelength  # Hz
    x = _H_PLANCK * nu / (_K_BOLTZ * temperature)
    # Clip to avoid overflow in exp
    x_safe = jnp.clip(x, 0.0, 500.0)
    return 2.0 * _H_PLANCK * nu**3 / _C_CGS**2 / (jnp.exp(x_safe) - 1.0)


def polar_dust_emission(
    l_absorbed_total: float,
    wavelength: jnp.ndarray,
    temperature: float = 100.0,
    beta: float = 1.6,
    lambda_0: float = 2e6,
) -> jnp.ndarray:
    """Greybody reemission from polar dust.

    Energy-conserving: the integral of the reemitted spectrum equals the
    total absorbed luminosity.

    Parameters
    ----------
    l_absorbed_total : float
        Total absorbed luminosity (scalar, integrated over frequency).
        Same units as input l_nu * delta_nu.
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    temperature : float
        Dust temperature in Kelvin. Default 100 K.
    beta : float
        Dust emissivity index. Default 1.6.
    lambda_0 : float
        Reference wavelength for optical depth in Angstrom.
        Default 2e6 (= 200 um).

    Returns
    -------
    l_nu_reemit : array, shape (n_wave,)
        Reemitted luminosity density [same units as input l_nu].
    """
    # Greybody: L_nu proportional to (1 - exp(-(lambda_0/lambda)^beta)) * B_nu(T)
    opacity_factor = 1.0 - jnp.exp(-((lambda_0 / wavelength) ** beta))
    b_nu = _planck_nu(wavelength, temperature)
    unnormalized = opacity_factor * b_nu

    # Normalize so that integral(L_reemit * dnu) = l_absorbed_total
    # dnu = -c / lambda^2 * dlambda, but we use |dnu|
    # For a wavelength grid, dnu_i ~ c / lambda_i^2 * |dlambda_i|
    nu = _C_AA / wavelength
    # Use trapezoidal spacing; for boundary, replicate nearest interval
    delta_nu = jnp.abs(jnp.diff(nu))
    delta_nu = jnp.concatenate([delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]])

    integral = jnp.sum(unnormalized * delta_nu)
    # Avoid division by zero when integral is tiny (e.g., all wavelengths
    # far from the emission peak)
    safe_integral = jnp.where(integral > 0.0, integral, 1.0)
    norm = l_absorbed_total / safe_integral

    return norm * unnormalized


def polar_dust_total(
    l_nu_disc: jnp.ndarray,
    wavelength: jnp.ndarray,
    cos_inc: float,
    opening_angle_deg: float,
    ebv: float,
    temperature: float = 100.0,
    beta: float = 1.6,
    lambda_0: float = 2e6,
    law: str = "smc",
    sharpness: float = _SIGMOID_SHARPNESS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply polar dust extinction and compute greybody reemission.

    Convenience function combining :func:`polar_dust_extinction` and
    :func:`polar_dust_emission`.

    Parameters
    ----------
    l_nu_disc : array, shape (n_wave,)
        Input AGN disc luminosity density [Lsun/Hz].
    wavelength : array, shape (n_wave,)
        Wavelength in Angstrom.
    cos_inc : float
        Cosine of inclination. 1 = face-on, 0 = edge-on.
    opening_angle_deg : float
        Torus half-opening angle in degrees.
    ebv : float
        Colour excess E(B-V).
    temperature : float
        Polar dust temperature in Kelvin.
    beta : float
        Dust emissivity index.
    lambda_0 : float
        Reference wavelength for optical depth in Angstrom.
    law : str
        Extinction law name.
    sharpness : float
        Sigmoid steepness at the Type 1/2 boundary.

    Returns
    -------
    l_nu_attenuated : array, shape (n_wave,)
        Attenuated disc luminosity.
    l_nu_reemit : array, shape (n_wave,)
        Greybody reemission from polar dust.
    """
    l_nu_attenuated, l_absorbed = polar_dust_extinction(
        l_nu_disc, wavelength, cos_inc, opening_angle_deg, ebv, law, sharpness
    )

    # Total absorbed luminosity: integrate l_absorbed over frequency
    nu = _C_AA / wavelength
    delta_nu = jnp.abs(jnp.diff(nu))
    delta_nu = jnp.concatenate([delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]])
    l_absorbed_total = jnp.sum(l_absorbed * delta_nu)

    l_nu_reemit = polar_dust_emission(l_absorbed_total, wavelength, temperature, beta, lambda_0)

    return l_nu_attenuated, l_nu_reemit
