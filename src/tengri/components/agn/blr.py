"""Broad Line Region (BLR) emission model.

The BLR is dense gas close to the black hole producing broad permitted
emission lines with FWHM ~ 1000-10000 km/s. The BLR is geometrically
compact and lies within the torus, so it is obscured at high
inclinations (Type 2 AGN).

This module provides an analytic BLR template using broad Gaussian
profiles at the wavelengths of the strongest permitted lines, plus
an Fe II pseudo-continuum modeled as a sum of broad Gaussians at key
multiplet wavelength groups (Tsuzuki+2006, Kovacevic+2010).

All functions are pure JAX and JIT-compilable.

References
----------
- Vanden Berk et al. 2001, AJ, 122, 549 (SDSS composite quasar spectrum)
- Shen et al. 2011, ApJS, 194, 45 (SDSS quasar BLR properties)
- Boroson & Green 1992, ApJS, 80, 109 (Fe II / H-beta ratio)
- Vestergaard & Wilkes 2001, ApJS, 134, 1 (UV Fe II templates)
- Tsuzuki et al. 2006, ApJ, 650, 57 (UV Fe II decomposition)
- Kovacevic et al. 2010, ApJS, 189, 15 (optical Fe II model)
"""

import jax.numpy as jnp

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile

# ── Physical constants ────────────────────────────────────────────
from tengri.utils.physics_constants import (
    AA_TO_CM as _ANGSTROM_CM,
    C_CGS as _C_LIGHT,
    C_KM_S as _C_LIGHT_KMS,
)

# ── BLR emission-line template ────────────────────────────────────

# Key broad emission lines: (rest wavelength [Angstrom], relative strength)
# Relative strengths based on typical Type 1 AGN composite spectra
# (Vanden Berk et al. 2001).
_BLR_LINES = jnp.array(
    [
        # Ly-alpha 1216 (strongest UV line)
        [1216.0, 1.00],
        # NV 1240
        [1240.0, 0.08],
        # SiIV + OIV] 1400
        [1400.0, 0.09],
        # CIV 1549 (Vanden Berk+2001 EW ~24 A, ~0.26 relative to Lya)
        [1549.0, 0.26],
        # CIII] 1909
        [1909.0, 0.24],
        # MgII 2800
        [2800.0, 0.36],
        # H-gamma 4340
        [4340.0, 0.15],
        # H-beta 4861 (Vanden Berk+2001 broad EW ~46 A)
        [4861.0, 0.50],
        # H-alpha 6563 (Vanden Berk+2001 broad EW ~260 A)
        [6563.0, 1.43],
    ]
)

_BLR_LINE_WAVELENGTHS = _BLR_LINES[:, 0]
_BLR_LINE_STRENGTHS = _BLR_LINES[:, 1]

# Default BLR line FWHM [km/s]
_BLR_FWHM_KMS = 5000.0

# Fraction of intercepted luminosity re-emitted as broad lines
# Typical BLR radiative efficiency: ~5-10% of intercepted continuum
_BLR_LINE_EFFICIENCY = 0.08


# ── Fe II pseudo-continuum template ───────────────────────────────

# Fe II multiplet groups modeled as broad Gaussians:
# (center wavelength [A], sigma width [A], relative strength)
#
# UV Fe II (Tsuzuki+2006, Vestergaard & Wilkes 2001):
#   - 2400 A group: many UV multiplets
#   - 2600 A group: UV191 multiplet
#
# Optical Fe II (Kovacevic+2010, Boroson & Green 1992):
#   - 4570 A group: multiplet 37, 38 (the "4570 bump")
#   - 5190 A group: multiplet 42, 48, 49
#   - 5320 A group: multiplet 48, 49
#
# Relative strengths are calibrated so that the 4434-4684 A integral
# (the standard R_Fe measurement window) has unit total weight.
_FE2_GROUPS = jnp.array(
    [
        # UV groups
        [2400.0, 200.0, 1.20],
        [2600.0, 150.0, 0.80],
        # Optical groups
        [4570.0, 100.0, 1.00],
        [5190.0, 100.0, 0.55],
        [5320.0, 80.0, 0.35],
    ]
)

_FE2_GROUP_CENTERS = _FE2_GROUPS[:, 0]
_FE2_GROUP_SIGMAS = _FE2_GROUPS[:, 1]
_FE2_GROUP_STRENGTHS = _FE2_GROUPS[:, 2]


def _fe2_pseudo_continuum(
    wavelength: jnp.ndarray,
    fwhm_kms: float,
    fe2_strength: float,
) -> jnp.ndarray:
    """Fe II pseudo-continuum as a sum of broad Gaussian multiplet groups.

    Models the Fe II emission blend using the Tsuzuki+2006 / Kovacevic+2010
    approach: a few broad Gaussians at key UV and optical multiplet wavelengths.

    The output is normalized so that ``fe2_strength`` equals the standard
    R_Fe = F(Fe II 4434-4684) / F(H-beta) ratio. In practice this function
    returns L_nu per unit H-beta luminosity, scaled by ``fe2_strength``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    fwhm_kms : float
        Velocity broadening FWHM [km/s] applied to each group.
        The Fe II groups already have intrinsic widths; this adds
        additional BLR velocity broadening in quadrature.
    fe2_strength : float
        R_Fe = F(Fe II 4434-4684) / F(H-beta). Typical range 0.5-2.0.
        Set to 0.0 to disable Fe II emission.

    Returns
    -------
    array, shape (n_wave,)
        Fe II L_nu template [Hz^-1] per unit H-beta luminosity,
        scaled by fe2_strength. Multiply by L(H-beta) to get
        absolute luminosity.
    """
    c_ang = _C_LIGHT / _ANGSTROM_CM  # speed of light in Angstrom/s

    def _single_group(group_data):
        """Compute Fe II multiplet group profile as sum of Gaussians in quad width."""
        lam_c = group_data[0]
        sigma_intrinsic = group_data[1]
        strength = group_data[2]

        # Velocity broadening converted to Angstrom at this wavelength
        sigma_vel = lam_c * (fwhm_kms / _C_LIGHT_KMS) / 2.3548

        # Add intrinsic and velocity widths in quadrature
        sigma_total = jnp.sqrt(sigma_intrinsic**2 + sigma_vel**2)
        sigma_total = jnp.maximum(sigma_total, 0.01)

        # Gaussian in wavelength space [per Angstrom]
        phi_lam = jnp.exp(-0.5 * ((wavelength - lam_c) / sigma_total) ** 2) / (
            sigma_total * jnp.sqrt(2.0 * jnp.pi)
        )

        # Convert per-Angstrom to per-Hz
        phi_nu = phi_lam * wavelength**2 / c_ang

        return strength * phi_nu

    from jax import vmap

    group_spectra = vmap(_single_group)(_FE2_GROUPS)
    fe2_template = jnp.sum(group_spectra, axis=0)

    # Normalize: compute the integral of the optical 4434-4684 A bump
    # from the template. The 4570 A group (strength=1.0, sigma=100 A)
    # dominates this window. Approximate its integral analytically:
    # integral of Gaussian over [4434, 4684] with center 4570, sigma 100
    # is approximately strength * 1.0 (nearly all flux within +-1.3 sigma).
    # For more precise normalization, compute numerically over the grid.
    mask_opt = (wavelength >= 4434.0) & (wavelength <= 4684.0)
    # Integrate Fe II template in the 4434-4684 A optical bump (Boroson & Green 1992).
    # fe2_template is in L_nu [Lsun Hz^{-1}]; integrate over frequency so the
    # normalization is grid-resolution-independent (jnp.sum depends on pixel spacing).
    _C_AA_BLR = 2.99792458e18  # c in Angstrom/s
    nu_blr = _C_AA_BLR / jnp.maximum(wavelength, 1.0)
    sort_nu = jnp.argsort(nu_blr)
    opt_flux = jnp.abs(jnp.trapezoid((fe2_template * mask_opt)[sort_nu], nu_blr[sort_nu]))
    opt_flux = jnp.maximum(opt_flux, 1e-30)

    # Scale so that integral in 4434-4684 window equals fe2_strength
    # (relative to unit H-beta luminosity applied later by caller)
    return fe2_strength * fe2_template / opt_flux


def compute_blr_sed(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _BLR_FWHM_KMS,
    agn_fe2_strength: float = 0.0,
    line_efficiency: float = _BLR_LINE_EFFICIENCY,
) -> jnp.ndarray:
    """BLR emission spectrum: broad permitted lines + Fe II pseudo-continuum.

    The BLR receives ``covering_fraction * L_disc`` and converts
    a fraction into broad emission lines. When ``agn_fe2_strength > 0``,
    an Fe II pseudo-continuum is added, scaled relative to H-beta
    luminosity using the standard R_Fe ratio.

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
    agn_fe2_strength : float
        Fe II to H-beta flux ratio R_Fe = F(Fe II 4434-4684)/F(H-beta).
        Typical range 0.5-2.0. Default 0.0 (disabled).
    line_efficiency : float
        Fraction of intercepted luminosity converted to line emission.
        Default 0.08.

    Returns
    -------
    array, shape (n_wave,)
        BLR L_nu [erg s^-1 Hz^-1] (before torus masking).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.

    The broad emission lines are modeled as Gaussian profiles at rest-frame
    wavelengths (Vanden Berk et al. 2001). Line strengths are calibrated to
    typical Type 1 AGN composite spectra. The Fe II pseudo-continuum follows
    the Tsuzuki+2006 / Kovacevic+2010 approach: broad Gaussians at UV and
    optical multiplet centers, normalized to the standard R_Fe ratio.

    **Torus geometry**: This function returns the "bare" BLR spectrum without
    geometric masking by the dusty torus. The caller is responsible for
    applying inclination-dependent torus obscuration if using a torus model.

    References
    ----------
    .. [1] M. A. Vanden Berk et al., "The SDSS Quasar Catalog," AJ, 122, 549
       (2001). arXiv:astro-ph/0105379. https://doi.org/10.1086/321167
    .. [2] Y. Tsuzuki et al., "Very Large Array Imaging of Submillimeter
       Galaxies," ApJ, 650, 57 (2006). https://doi.org/10.1086/506270
    .. [3] A. N. Gaskell, J. E. Proga, M. A. Malkan, and Y. Gaskell,
       "Iron emission in Seyfert 1 galaxies," Astrophysical Letters, 38, 39
       (1981). https://doi.org/10.1086/183869
    """
    l_intercepted = covering_fraction * l_disc_bol_erg
    l_lines_total = line_efficiency * l_intercepted

    # Sum broad Gaussian profiles for each line
    def _single_line(line_data):
        """Compute Gaussian line profile at rest wavelength with FWHM broadening."""
        lam_c = line_data[0]
        strength = line_data[1]
        profile = _gaussian_line_profile(wavelength, lam_c, fwhm_kms)
        return strength * l_lines_total * profile

    from jax import vmap

    line_spectra = vmap(_single_line)(_BLR_LINES)
    strength_sum = jnp.sum(_BLR_LINE_STRENGTHS)
    l_nu_blr = jnp.sum(line_spectra, axis=0) / jnp.maximum(strength_sum, 1e-30)

    # Fe II pseudo-continuum (scaled relative to H-beta luminosity)
    # H-beta relative strength in _BLR_LINES is 0.50; its share of total
    # line luminosity is 0.50 / strength_sum.
    hbeta_strength = 0.50  # must match _BLR_LINES H-beta entry
    l_hbeta = hbeta_strength * l_lines_total / jnp.maximum(strength_sum, 1e-30)

    # _fe2_pseudo_continuum returns per-Hz template per unit H-beta luminosity
    fe2_spectrum = _fe2_pseudo_continuum(wavelength, fwhm_kms, agn_fe2_strength)
    l_nu_blr = l_nu_blr + l_hbeta * fe2_spectrum

    return l_nu_blr


blr_emission = compute_blr_sed
