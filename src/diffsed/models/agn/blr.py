"""Broad Line Region (BLR) emission model.

The BLR is dense gas close to the black hole producing broad permitted
emission lines with FWHM ~ 1000-10000 km/s. The BLR is geometrically
compact and lies within the torus, so it is obscured at high
inclinations (Type 2 AGN).

This module provides an analytic BLR template using broad Gaussian
profiles at the wavelengths of the strongest permitted lines.

All functions are pure JAX and JIT-compilable.

References
----------
- Vanden Berk et al. 2001, AJ, 122, 549 (SDSS composite quasar spectrum)
- Shen et al. 2011, ApJS, 194, 45 (SDSS quasar BLR properties)
"""

import jax.numpy as jnp

# ===================================================================
# Physical constants
# ===================================================================

_C_LIGHT_KMS = 2.99792458e5  # Speed of light [km/s]
_C_LIGHT = 2.99792458e10  # Speed of light [cm s^-1]
_ANGSTROM_CM = 1e-8  # Angstrom -> cm

# ===================================================================
# BLR emission-line template
# ===================================================================

# Key broad emission lines: (rest wavelength [Angstrom], relative strength)
# Relative strengths based on typical Type 1 AGN composite spectra
# (Vanden Berk et al. 2001).
_BLR_LINES = jnp.array([
    # Ly-alpha 1216 (strongest UV line)
    [1216.0, 1.00],
    # NV 1240
    [1240.0, 0.12],
    # SiIV + OIV] 1400
    [1400.0, 0.07],
    # CIV 1549
    [1549.0, 0.50],
    # CIII] 1909
    [1909.0, 0.20],
    # MgII 2800
    [2800.0, 0.35],
    # H-gamma 4340
    [4340.0, 0.05],
    # H-beta 4861
    [4861.0, 0.22],
    # H-alpha 6563
    [6563.0, 0.60],
])

_BLR_LINE_WAVELENGTHS = _BLR_LINES[:, 0]
_BLR_LINE_STRENGTHS = _BLR_LINES[:, 1]

# Default BLR line FWHM [km/s]
_BLR_FWHM_KMS = 5000.0

# Fraction of intercepted luminosity re-emitted as broad lines
# Typical BLR radiative efficiency: ~5-10% of intercepted continuum
_BLR_LINE_EFFICIENCY = 0.08


def _gaussian_line_profile(
    wavelength: jnp.ndarray,
    line_center: float,
    fwhm_kms: float,
) -> jnp.ndarray:
    """Normalized Gaussian line profile in wavelength space.

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
        Profile in units of Hz^-1 (normalized so integral over dnu = 1).
    """
    sigma_ang = line_center * (fwhm_kms / _C_LIGHT_KMS) / 2.3548
    sigma_ang = jnp.maximum(sigma_ang, 0.01)

    phi_lam = jnp.exp(-0.5 * ((wavelength - line_center) / sigma_ang) ** 2) / (
        sigma_ang * jnp.sqrt(2.0 * jnp.pi)
    )

    # Convert per-Angstrom to per-Hz
    c_ang = _C_LIGHT / _ANGSTROM_CM
    phi_nu = phi_lam * wavelength**2 / c_ang

    return phi_nu


def blr_emission(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _BLR_FWHM_KMS,
) -> jnp.ndarray:
    """BLR emission spectrum: broad permitted emission lines.

    The BLR receives ``covering_fraction * L_disc`` and converts
    a fraction into broad emission lines.

    Note: geometric masking by the torus is NOT applied here;
    it must be applied by the caller.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg s^-1].
    covering_fraction : float
        BLR covering fraction (0 to 1). Default 0.1.
    fwhm_kms : float
        Line FWHM [km/s]. Default 5000.

    Returns
    -------
    array, shape (n_wave,)
        BLR L_nu [erg s^-1 Hz^-1] (before torus masking).
    """
    l_intercepted = covering_fraction * l_disc_bol_erg
    l_lines_total = _BLR_LINE_EFFICIENCY * l_intercepted

    # Sum broad Gaussian profiles for each line
    def _single_line(line_data):
        lam_c = line_data[0]
        strength = line_data[1]
        profile = _gaussian_line_profile(wavelength, lam_c, fwhm_kms)
        return strength * l_lines_total * profile

    from jax import vmap
    line_spectra = vmap(_single_line)(_BLR_LINES)
    strength_sum = jnp.sum(_BLR_LINE_STRENGTHS)
    l_nu_blr = jnp.sum(line_spectra, axis=0) / jnp.maximum(strength_sum, 1e-30)

    return l_nu_blr
