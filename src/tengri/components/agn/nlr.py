# SPDX-License-Identifier: BSD-3-Clause
"""Narrow Line Region (NLR) emission model.

The NLR is photoionized gas illuminated by the AGN accretion disc.
It produces nebular-like emission: a power-law continuum plus
forbidden-line emission at key wavelengths.

The NLR emission is isotropic (not masked by the torus) because it
extends on kpc scales beyond the torus opening angle.

For computational efficiency this module uses analytic line profiles
rather than full CLOUDY grids. Each emission line is a Gaussian with
FWHM ~ 500 km/s (narrow lines) placed at the laboratory wavelength.

All functions are pure JAX and JIT-compilable.

References
----------
- Groves et al. 2004, ApJS, 153, 75 (NLR photoionization models)
- Feltre et al. 2016, MNRAS, 456, 3354 (NLR emission-line diagnostics)
- Vijarnwannaluk et al. 2022 (NLR covering fractions)
"""

import jax.numpy as jnp

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile

# ── Physical constants ────────────────────────────────────────────
from tengri.utils.physics_constants import (
    AA_TO_CM as _ANGSTROM_CM,
    C_CGS as _C_LIGHT,
)

# ── NLR emission-line template ────────────────────────────────────

# Key narrow emission lines: (rest wavelength [Angstrom], relative strength)
# Relative strengths are approximate, calibrated to typical Seyfert 2 spectra.
# Normalized so that the sum of line luminosities ~ 1.
_NLR_LINES = jnp.array(
    [
        # [OII] 3727 doublet (blended)
        [3727.0, 0.15],
        # [NeIII] 3869
        [3869.0, 0.04],
        # H-beta 4861 (narrow)
        [4861.0, 0.10],
        # [OIII] 4959 (ratio 5007/4959 = 2.98; Storey & Zeippen 2000)
        [4959.0, 0.117],
        # [OIII] 5007 (strongest NLR line)
        [5007.0, 0.35],
        # [OI] 6300
        [6300.0, 0.03],
        # [NII] 6548
        [6548.0, 0.05],
        # H-alpha 6563 (narrow)
        [6563.0, 0.20],
        # [NII] 6583
        [6583.0, 0.15],
        # [SII] 6716
        [6716.0, 0.06],
        # [SII] 6731
        [6731.0, 0.06],
    ]
)

_NLR_LINE_WAVELENGTHS = _NLR_LINES[:, 0]
_NLR_LINE_STRENGTHS = _NLR_LINES[:, 1]
# Normalize so sum = 1
_NLR_LINE_STRENGTHS_NORM = _NLR_LINE_STRENGTHS / jnp.sum(_NLR_LINE_STRENGTHS)

# Default NLR line FWHM [km/s]
_NLR_FWHM_KMS = 500.0

# Fraction of intercepted luminosity converted to line emission
# (the rest heats dust / continuum). Typical: 5-15%.
_NLR_LINE_EFFICIENCY = 0.10

# Fraction of intercepted luminosity re-emitted as continuum
_NLR_CONTINUUM_EFFICIENCY = 0.02

# Continuum power-law slope (F_nu ~ nu^alpha, typical NLR continuum)
_NLR_CONTINUUM_ALPHA = -1.5


def compute_nlr_sed(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _NLR_FWHM_KMS,
    line_efficiency: float = _NLR_LINE_EFFICIENCY,
) -> jnp.ndarray:
    """NLR emission spectrum: line emission + power-law continuum.

    The NLR receives ``covering_fraction * L_disc`` and re-emits
    a fraction as emission lines and a small continuum component.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg s^-1].
    covering_fraction : float
        NLR covering fraction (0 to 1). Default 0.1.
    fwhm_kms : float
        Line FWHM [km/s]. Default 500.
    line_efficiency : float
        Fraction of intercepted luminosity converted to line emission.
        Default 0.10.

    Returns
    -------
    array, shape (n_wave,)
        NLR L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.
    """
    l_intercepted = covering_fraction * l_disc_bol_erg

    # --- Line emission ---
    l_lines_total = line_efficiency * l_intercepted

    # Sum Gaussian profiles for each line
    def _single_line(line_data):
        """Compute luminosity-weighted Gaussian profile for one NLR emission line."""
        lam_c = line_data[0]
        strength = line_data[1]
        profile = _gaussian_line_profile(wavelength, lam_c, fwhm_kms)
        return strength * l_lines_total * profile

    # vmap over lines
    from jax import vmap

    line_spectra = vmap(_single_line)(_NLR_LINES)
    # Renormalize strengths
    strength_sum = jnp.sum(_NLR_LINE_STRENGTHS)
    l_nu_lines = jnp.sum(line_spectra, axis=0) / jnp.maximum(strength_sum, 1e-30)

    # --- Continuum emission ---
    l_cont_total = _NLR_CONTINUUM_EFFICIENCY * l_intercepted
    nu = _C_LIGHT / (wavelength * _ANGSTROM_CM)
    cont_shape = nu**_NLR_CONTINUUM_ALPHA

    # Normalize continuum
    sort_idx = jnp.argsort(nu)
    integral = jnp.trapezoid(cont_shape[sort_idx], nu[sort_idx])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)
    l_nu_cont = l_cont_total * cont_shape / integral_safe

    return l_nu_lines + l_nu_cont


# ── Richardson+2014 NLR template ──────────────────────────────────

# Emission line wavelengths and normalized fluxes from Richardson+2014 Table 3 'a42'.
# Lines sorted by wavelength [Angstrom], fluxes normalized to Hbeta=1.
# Source: FSPS emline_wavelengths at indices
# [38, 40, 41, 43, 45, 50, 51, 52, 59, 61, 62, 64, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80]
_RICHARDSON_WAVES = jnp.array(
    [
        3727.1180,  # [O II] 3726
        3799.0277,  # Ba-8 3798
        3836.5280,  # Ba-7 3835
        3869.9172,  # [Ne III] 3869
        3890.2127,  # Ba-6 3889
        4102.9514,  # Ba-delta 4101.76A
        4341.7476,  # Ba-gamma 4341
        4364.2938,  # [O III] 4363
        4862.7629,  # Ba-beta 4861
        4960.3702,  # [O III] 4959
        5008.3137,  # [O III] 5007
        5201.7880,  # [N I] 5200
        5756.2941,  # [N II] 5755
        5877.3583,  # He I 5875.64A
        6302.1385,  # [O I] 6300
        6365.6364,  # [O I] 6363
        6549.9587,  # [N II] 6548
        6564.7229,  # Ba-alpha 6563 (H-alpha)
        6585.3687,  # [N II] 6584
        6680.0956,  # He I 6678.15A
        6718.3965,  # [S II] 6716
        6732.7805,  # [S II] 6731
        7137.8656,  # [Ar III] 7135
    ]
)

_RICHARDSON_FLUXES = jnp.array(
    [
        2.96,
        0.06,
        0.1,
        1.0,
        0.2,
        0.25,
        0.48,
        0.13,
        1.0,
        2.87,
        8.53,
        0.07,
        0.02,
        0.1,
        0.33,
        0.09,
        0.79,
        2.86,
        2.13,
        0.03,
        0.77,
        0.65,
        0.19,
    ]
)


def compute_nlr_sed_richardson2014(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _NLR_FWHM_KMS,
    line_efficiency: float = _NLR_LINE_EFFICIENCY,
) -> jnp.ndarray:
    """AGN NLR spectrum using Richardson+2014 Table 3 'a42' line template.

    The narrow-line region (NLR) is photoionized gas illuminated by the AGN
    accretion disc. This function synthesizes the NLR emission spectrum using
    the emission-line template from Richardson et al. (2014), which provides
    AGN-specific line ratios derived from the 'a42' column of Table 3
    (moderate AGN luminosity, intermediate inclination angle).

    The NLR receives ``covering_fraction * L_disc`` and converts a fraction
    into line emission. Each line is modeled as a Gaussian profile at the
    rest-frame wavelength.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg s^-1].
    covering_fraction : float, optional
        NLR covering fraction (0 to 1). Default 0.1.
    fwhm_kms : float, optional
        Line FWHM [km/s]. Default 500.
    line_efficiency : float, optional
        Fraction of intercepted luminosity converted to line emission.
        Default 0.10.

    Returns
    -------
    l_nu : ndarray, shape (n_wave,)
        NLR L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    This function is JIT-compilable (no traced control flow).

    The Richardson+2014 'a42' template uses 23 emission lines normalized to
    H-beta = 1. The strongest line is [O III] 5007 at 8.53× H-beta.
    Implemented via FSPS emission-line indices. Line profiles are Gaussian
    with fixed FWHM (narrow lines, ~500 km/s).

    Ported from Prospector (Johnson et al. 2021 [1]_), which implements
    the same table for AGN NLR modeling.

    References
    ----------
    .. [1] J. C. Richardson, et al., "Optical Spectroscopy of Post-Starburst
       Galaxies," ApJ, 2014. Table 3, column 'a42'.
       https://ui.adsabs.harvard.edu/abs/2014ApJ...782...33R/abstract
    .. [2] B. D. Johnson, et al., "Prospector: Inferring the Star Formation
       Histories of Galaxies from Observed Spectral Energy Distributions,"
       ApJS, 254, 22, 2021. https://doi.org/10.3847/1538-4365/abef67
    """
    l_intercepted = covering_fraction * l_disc_bol_erg

    # --- Line emission ---
    l_lines_total = line_efficiency * l_intercepted

    # Compute the total flux in the template (for normalization)
    flux_sum = jnp.sum(_RICHARDSON_FLUXES)

    # Luminosity per unit flux (normalized to Hbeta)
    l_per_flux = l_lines_total / jnp.maximum(flux_sum, 1e-30)

    # Sum Gaussian profiles for each line
    def _single_line(line_data):
        """Compute luminosity-weighted Gaussian profile for one emission line."""
        wave_c = line_data[0]
        flux_ratio = line_data[1]
        profile = _gaussian_line_profile(wavelength, wave_c, fwhm_kms)
        return flux_ratio * l_per_flux * profile

    # vmap over lines
    from jax import vmap

    line_data = jnp.stack([_RICHARDSON_WAVES, _RICHARDSON_FLUXES], axis=1)
    line_spectra = vmap(_single_line)(line_data)
    l_nu_lines = jnp.sum(line_spectra, axis=0)

    return l_nu_lines
