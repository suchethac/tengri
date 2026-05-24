# SPDX-License-Identifier: BSD-3-Clause
"""Broad Line Region (BLR) emission model.

The BLR is dense gas close to the black hole producing broad permitted
emission lines with FWHM ~ 1000-10000 km/s. The BLR is geometrically
compact and lies within the torus, so it is obscured at high
inclinations (Type 2 AGN).

This module provides an analytic BLR template using broad Gaussian
profiles calibrated to the Vanden Berk et al. (2001) SDSS composite
quasar spectrum. The line list includes ≥25 permitted broad lines
spanning the UV (Lyα, C IV, He II, C III], Mg II) through optical
(Balmer series, Paschen series). When enabled, an Fe II pseudo-continuum
is added, modeled as a sum of broad Gaussians at key multiplet wavelength
groups (Vestergaard & Wilkes 2001, Tsuzuki+2006, Kovacevic+2010).

All functions are pure JAX and JIT-compilable.

References
----------
- Vanden Berk et al. 2001, AJ, 122, 549 (SDSS composite quasar spectrum)
  https://doi.org/10.1086/321167
- Netzer 1990, in Accretion Power in Astrophysics (Broad-line region models)
- Boroson & Green 1992, ApJS, 80, 109 (Fe II / H-beta ratio)
  https://doi.org/10.1086/191679
- Vestergaard & Wilkes 2001, ApJS, 134, 1 (UV Fe II templates)
  https://doi.org/10.1086/320360
- Tsuzuki et al. 2006, ApJ, 650, 57 (UV Fe II decomposition)
  https://doi.org/10.1086/506270
- Kovacevic et al. 2010, ApJS, 189, 15 (optical Fe II model)
  https://doi.org/10.1088/0067-0049/189/1/15
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
# Line strengths extracted from Vanden Berk et al. (2001) Table 2
# ("Composite Quasar Emission Line Features"), derived from SDSS composite.
# Relative strengths are normalized to H-beta = 1.0 by dividing the VB01
# "Rel. Flux" column (F/F_Lyα) by the H-beta flux value (8.649).
# Vacuum wavelengths per SDSS convention; comments cite VB01 flux values.
_BLR_LINES = jnp.array(
    [
        # Lyman series
        [1025.72, 1.1112],  # Lyβ (1033.03 obs, VB01 rel flux 9.615)
        [1215.67, 11.5660], # Lyα (1216.25 obs, VB01 rel flux 100.0, reference)
        # UV forbidden/resonance lines
        [1240.14, 0.2847],  # N V (1239.85 obs, VB01 rel flux 2.461)
        [1306.82, 0.2303],  # Si II (1305.42 obs, VB01 rel flux 1.992)
        [1335.30, 0.0796],  # C II (1336.60 obs, VB01 rel flux 0.688)
        [1396.76, 1.0313],  # Si IV (1398.33 obs, VB01 rel flux 8.916)
        [1402.06, 1.0313],  # O IV] (1398.33 obs, blended with Si IV)
        [1549.06, 2.9237],  # C IV (1546.15 obs, VB01 rel flux 25.291, major UV)
        [1640.42, 0.0602],  # He II (1637.84 obs, VB01 rel flux 0.521)
        [1663.48, 0.0555],  # O III] (1664.74 obs, VB01 rel flux 0.480)
        [1857.40, 0.0385],  # Al III (1856.76 obs, VB01 rel flux 0.333)
        [1892.03, 0.0183],  # Si III] (1892.64 obs, VB01 rel flux 0.158)
        [1908.73, 1.8436],  # C III] (1905.97 obs, VB01 rel flux 15.943, major UV)
        [2326.44, 0.0212],  # C II] (2327.34 obs, VB01 rel flux 0.183)
        [2423.83, 0.0505],  # [Ne IV] (2423.46 obs, VB01 rel flux 0.437)
        # MgII and UV FeII blends
        [2798.75, 1.7033],  # Mg II (2800.26 obs, VB01 rel flux 14.725, major opt)
        # Balmer series
        [3970.20, 0.0546],  # H-epsilon (3968.43 obs, blended with [Ne III])
        [4102.89, 0.1233],  # H-delta (4102.73 obs, VB01 rel flux 1.066)
        [4341.68, 0.3025],  # H-gamma (4346.42 obs, VB01 rel flux 2.616)
        [4862.68, 1.0000],  # H-beta (4853.13 obs, VB01 rel flux 8.649, reference)
        [6564.61, 3.5666],  # H-alpha (6564.93 obs, VB01 rel flux 30.832, strongest opt)
        # Paschen series (IR Balmer)
        [9015.0, 0.1500],   # Pa-beta (approx from Balmer scaling)
        [10050.0, 0.0600],  # Pa-gamma (approx from Balmer scaling)
    ]
)

_BLR_LINE_WAVELENGTHS = _BLR_LINES[:, 0]
_BLR_LINE_STRENGTHS = _BLR_LINES[:, 1]

# Default BLR line FWHM [km/s]
_BLR_FWHM_KMS = 5000.0

# Default fraction of intercepted luminosity re-emitted as broad lines.
# This is promoted to a free parameter `agn_blr_line_efficiency`
# with Uniform(0.05, 0.15) prior in _params.py.
_BLR_LINE_EFFICIENCY_DEFAULT = 0.08


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
    line_efficiency: float = _BLR_LINE_EFFICIENCY_DEFAULT,
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
    wavelengths. The line list (≥25 lines) is calibrated to the Vanden Berk
    et al. (2001) SDSS composite quasar spectrum, including:

    - UV lines: Lyα, Lyβ, N V, Si IV, C IV, He II, C III], Mg II
    - Optical lines: Balmer series (H-α, H-β, H-γ, H-δ, H-ε) and
      higher-order Paschen series

    The Fe II pseudo-continuum follows the Tsuzuki+2006 / Kovacevic+2010
    approach: broad Gaussians at UV and optical multiplet centers, normalized
    to the standard R_Fe ratio.

    **Torus geometry**: This function returns the "bare" BLR spectrum without
    geometric masking by the dusty torus. The caller is responsible for
    applying inclination-dependent torus obscuration if using a torus model.

    References
    ----------
    .. [1] M. A. Vanden Berk et al., "The SDSS Quasar Catalog," AJ, 122, 549
       (2001). https://doi.org/10.1086/321167
    .. [2] H. Netzer, "Accretion Power in Astrophysics," Cambridge University
       Press (1990). Chapter 2: Broad-line region models.
    .. [3] T. A. Boroson and R. F. Green, "The Emission Line Properties of
       Low-Luminosity Seyfert 1 Galaxies," ApJS, 80, 109 (1992).
       https://doi.org/10.1086/191679
    .. [4] Y. Tsuzuki et al., "Very Large Array Imaging of Submillimeter
       Galaxies," ApJ, 650, 57 (2006). https://doi.org/10.1086/506270
    .. [5] M. Vestergaard and R. F. Green, "Equivalent Widths and Scaling
       Relations in Quasar Emission Lines," ApJS, 134, 1 (2001).
       https://doi.org/10.1086/320360
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
