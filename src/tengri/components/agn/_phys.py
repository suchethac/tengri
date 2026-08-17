# SPDX-License-Identifier: BSD-3-Clause
"""Shared physical utility functions for AGN sub-models.

Extracted from disc.py, torus.py, and skirtor.py to eliminate
three identical copies of the Planck function and related helpers.

Fundamental constants are imported from :mod:`tengri.utils.physics_constants`,
which documents their SI→CGS derivations and CODATA 2018 / IAU 2015 sources.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.blackbody import planck_bnu_nu as _planck_bnu_nu
from tengri.utils.physics_constants import (
    AA_TO_CM as ANGSTROM_CM,
    C_CGS as C_LIGHT,
    C_KM_S,
    H_PLANCK,
    K_BOLTZ,
    L_SUN,
)

__all__ = [
    "ANGSTROM_CM",
    "C_LIGHT",
    "H_PLANCK",
    "K_BOLTZ",
    "L_SUN",
    "bolometric_integral_nu",
    "compute_l_12um_from_lbol",
    "gaussian_line_profile",
    "lines_to_sed",
    "planck_lnu",
    "ring_area",
    "wavelength_to_nu",
]


# ── Planck function ───────────────────────────────────────────────


def planck_lnu(
    nu: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Compute the Planck blackbody spectral radiance.

    Evaluate the Planck function B_nu(T) at a given frequency and temperature.
    Thin AGN-facing spelling of :func:`tengri.utils.blackbody.planck_bnu_nu`,
    which is the single implementation for the whole tree; do not duplicate it.

    Parameters
    ----------
    nu : array_like, shape (n_freq,)
        Frequency. [Hz]
    temperature : float
        Blackbody temperature. Must be positive. [K]

    Returns
    -------
    ndarray, shape (n_freq,)
        Spectral radiance B_nu(T). [erg/s/cm^2/Hz/sr]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The Planck function is:

    .. math::

        B_\\nu(T) = \\frac{2 h \\nu^3}{c^2} \\frac{1}{e^{h\\nu/k_B T} - 1}

    where :math:`h` is Planck's constant [erg·s], :math:`\\nu` is frequency [Hz],
    :math:`c` is the speed of light [cm/s], :math:`k_B` is Boltzmann's constant
    [erg/K], and :math:`T` is temperature [K].

    **Numerical stability**: The implementation clamps :math:`x = h\\nu/k_B T` to
    the interval [1e-10, 500] to prevent expm1 overflow while keeping gradients
    finite everywhere. Temperature is clamped to [1.0, ∞) K to avoid division
    by zero. Arithmetic is performed in float64 to handle :math:`\\nu^3` at
    UV frequencies (~10^17 Hz) without overflow.

    References
    ----------
    .. [1] M. Planck, "Zur Theorie des Gesetzes der Energieverteilung im
       Normalspektrum," Verhandlungen der Deutschen Physikalischen Gesellschaft,
       Vol. 2, pp. 237-245 (1900).
    """
    return _planck_bnu_nu(nu, temperature)


# ── Wavelength ↔ frequency conversion ─────────────────────────────


def wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength to frequency.

    Parameters
    ----------
    wavelength_angstrom : array_like, shape (n,)
        Wavelength. [Angstrom]

    Returns
    -------
    ndarray, shape (n,)
        Frequency. [Hz]

    Notes
    -----
    **JIT-compatible**: yes.

    Uses the relation :math:`\\nu = c / \\lambda` where :math:`c` is the speed
    of light [cm/s].
    """
    return C_LIGHT / (wavelength_angstrom * ANGSTROM_CM)


# ── Bolometric frequency integral ─────────────────────────────────


def bolometric_integral_nu(
    lnu: jnp.ndarray,
    nu: jnp.ndarray,
    *,
    floor: float | None = None,
) -> jnp.ndarray:
    r"""Trapezoid integral of :math:`L_\nu` over an ascending-sorted frequency grid.

    The one shared implementation of the bolometric-normalization idiom
    used by every tabulated AGN template component (SKIRTOR, Fritz,
    Nenkova, Silva04, CAT3D, qsogen, ...): sort the frequency grid
    ascending (wavelength grids arrive ascending in :math:`\lambda`, i.e.
    *descending* in :math:`\nu`) and integrate.

    .. math::

        L = \int_{\nu_{\min}}^{\nu_{\max}} L_\nu \, d\nu

    where :math:`L_\nu` is the spectral luminosity density [erg/s/Hz],
    :math:`\nu` the frequency [Hz], and :math:`L` the integrated
    luminosity [erg/s].

    Parameters
    ----------
    lnu : array_like, shape (n_wave,)
        Spectral luminosity density on the same grid as ``nu``. [erg/s/Hz]
    nu : array_like, shape (n_wave,)
        Frequency grid, any ordering. [Hz]
    floor : float, optional
        When given, return ``max(|integral|, floor)`` — the safe form used
        as a normalization denominator (templates can be identically zero
        outside their tabulated range). When ``None`` (default), return
        the raw signed integral.

    Returns
    -------
    ndarray, shape ()
        Integrated luminosity [erg/s]; floored absolute value if ``floor``
        is given.

    Notes
    -----
    **JIT-compatible**: yes (``floor`` is a static Python-level branch).

    **Gradient-safe**: yes.

    This is the ``argsort`` formulation and reproduces the historical
    per-component copies bit-for-bit. It is NOT interchangeable bit-for-bit
    with ``tengri.components.dust.emission._physics.integrate_lnu_over_nu``,
    which integrates via the :math:`d\ln\nu = -d\ln\lambda` identity and
    differs in floating-point rounding.
    """
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(lnu[idx_sort], nu[idx_sort])
    if floor is None:
        return integral
    return jnp.maximum(jnp.abs(integral), floor)


# ── Gaussian emission-line profile (scalar kernel) ────────────────


def gaussian_line_profile(
    wavelength: jnp.ndarray,
    line_center: float,
    fwhm_kms: float,
) -> jnp.ndarray:
    """Gaussian emission line profile per unit frequency.

    Evaluate a normalized Gaussian line profile on a wavelength grid, returning
    the profile in frequency space. Used by NLR and BLR modules via ``jax.vmap``
    to broadcast over a line list.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Angstrom]
    line_center : float
        Line center wavelength. [Angstrom]
    fwhm_kms : float
        Full-width half-maximum of the profile. [km/s]

    Returns
    -------
    ndarray, shape (n_wave,)
        Gaussian profile in frequency space, normalized so that
        the integral over d(nu) equals 1. [Hz^-1]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The profile is generated by:

    1. Converting FWHM from km/s to wavelength via :math:`\\Delta\\lambda = \\lambda_0 \\cdot
       (\\text{FWHM} / c)`.
    2. Computing standard deviation :math:`\\sigma_\\lambda = \\text{FWHM} / 2.3548` where
       2.3548 = 2√(2 ln 2).
    3. Evaluating the normalized wavelength Gaussian :math:`\\phi_\\lambda(\\lambda)`.
    4. Transforming to frequency space via the Jacobian :math:`\\phi_\\nu = \\phi_\\lambda
       \\cdot (\\lambda^2 / c)`.

    **Numerical safeguard**: :math:`\\sigma_\\lambda` is clamped to [0.01 Å, ∞)
    to avoid spurious delta-function behavior when FWHM is very small.
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


# ── Disc ring projected area (R&L 1979 Eq 1.6 geometry) ───────────


def ring_area(r_cm: float, dr_cm: float, cos_inc: float) -> float:
    """Projected area of an annular accretion disc ring.

    Compute the projected area (solid angle factor) for an annular ring of
    radius :math:`r` and width :math:`dr` at inclination angle :math:`i`.
    Used to scale the Planck function when integrating disc rings into
    total luminosity.

    Parameters
    ----------
    r_cm : float
        Ring radius. [cm]
    dr_cm : float
        Ring radial width. [cm]
    cos_inc : float
        Cosine of the inclination angle. Must be in (0, 1].

    Returns
    -------
    float
        Projected area factor. [cm^2 sr]

    Notes
    -----
    **JIT-compatible**: yes.

    The geometry is based on Rybicki & Lightman (1979), Eq. 1.6. The luminosity
    from a flat annular ring of radius :math:`r`, width :math:`dr`, and
    temperature :math:`T` is:

    .. math::

        dL_\\nu = \\pi \\cdot B_\\nu(T) \\cdot 2\\pi r \\, dr \\cdot \\cos(i)

    where :math:`B_\\nu(T)` is the Planck function [erg/s/cm^2/Hz/sr],
    :math:`i` is the inclination angle, and the factor :math:`\\pi \\cdot 2\\pi r \\, dr`
    is the projected area of the ring.

    **Numerical safeguard**: :math:`\\cos(i)` is clamped to [0.01, 1.0] to
    prevent zero-area rings at edge-on inclinations while maintaining
    finite derivatives.

    References
    ----------
    .. [1] G. B. Rybicki and A. P. Lightman, "Radiative Processes in
       Astrophysics," John Wiley & Sons (1979). ISBN: 0-471-82759-2
    """
    return jnp.pi * 2.0 * jnp.pi * r_cm * dr_cm * jnp.maximum(cos_inc, 0.01)


# ── Line list → SED convolution ───────────────────────────────────


def compute_l_12um_from_lbol(
    agn_log_lbol: float | jnp.ndarray, f_12: float = 0.07
) -> float | jnp.ndarray:
    r"""Compute rest-frame 12 μm monochromatic luminosity from AGN bolometric luminosity.

    Derives the nuclear 12 μm monochromatic luminosity density using a parametric
    bolometric correction calibrated against AGN SED templates (Krawczyk et al. 2013).
    This correction is used by :func:`~tengri.components.xray.xray.xray_agn_corona_lopez24`
    to compute X-ray luminosity from the α_IRX relation.

    Parameters
    ----------
    agn_log_lbol : float or jnp.ndarray
        AGN bolometric luminosity. [log10(L_sun)]
    f_12 : float, optional
        Bolometric correction fraction L_12μm / L_bol. Default: 0.07.
        [dimensionless]

    Returns
    -------
    float or jnp.ndarray
        Monochromatic luminosity density at rest 12 μm. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    **Bolometric correction calibration** (Krawczyk et al. 2013 [1]_):
    For typical Type 1 AGN, the ratio of 12 μm monochromatic flux to
    bolometric luminosity is approximately constant:

    .. math::

        L_{12\mu\mathrm{m}} = f_{12} \times L_{\mathrm{bol}}

    with :math:`f_{12} \approx 0.07` derived from stacking AGN SED templates
    in the mid-infrared. This is numerically equivalent to

    .. math::

        L_\nu(12\mu\mathrm{m}) = f_{12} \times L_{\mathrm{bol}} / \nu_{12\mu\mathrm{m}}

    where :math:`\nu_{12\mu\mathrm{m}} = c / 12 \mu\mathrm{m}`.

    References
    ----------
    .. [1] C. Krawczyk et al., "The mid-infrared AGN fraction in the XMM-COSMOS
       survey," ApJS, 206, 4 (2013). arXiv:1301.1688.
       https://doi.org/10.1088/0067-0049/206/1/4
    """
    l_bol_erg = 10.0**agn_log_lbol * L_SUN
    # 12 μm = 1.2e5 Å; ν = c / λ [Hz]
    nu_12um = C_LIGHT / (1.2e5 * ANGSTROM_CM)
    l_12um_erg = f_12 * l_bol_erg
    l_12um_erg_hz = l_12um_erg / nu_12um
    return l_12um_erg_hz


def lines_to_sed(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    wave_obs: jnp.ndarray,
    fwhm_kms: float = 500.0,
) -> jnp.ndarray:
    """Convolve emission lines with Gaussian broadening onto a wavelength grid.

    Render a list of delta-function emission lines as broadened Gaussian profiles
    on a wavelength grid. Used by the BLR and NLR modules to compute the
    emission line contribution to the SED.

    Parameters
    ----------
    line_wavelengths : array_like, shape (n_lines,)
        Rest-frame line center wavelengths. [Angstrom]
    line_luminosities : array_like, shape (n_lines,)
        Per-line bolometric luminosities. [Lsun]
    wave_obs : array_like, shape (n_wave,)
        Output wavelength grid. [Angstrom]
    fwhm_kms : float, optional
        Gaussian line FWHM. Default: 500.0 [km/s]

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density at output wavelengths. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable w.r.t. all inputs.

    The algorithm:

    1. Convert FWHM from km/s to wavelength space via :math:`\\Delta\\lambda_i =
       \\lambda_i \\cdot (\\text{FWHM} / c)`.
    2. Compute standard deviations :math:`\\sigma_i = \\Delta\\lambda_i / 2.3548`.
    3. Evaluate normalized Gaussians in wavelength space: :math:`\\phi_\\lambda(\\lambda)`.
    4. Normalize each profile so that :math:`\\int \\phi_\\nu d\\nu = 1`.
    5. Sum the normalized profiles weighted by the line luminosities to get
       :math:`L_\\lambda(\\lambda)` [Lsun/Å].
    6. Convert to frequency space via the Jacobian :math:`L_\\nu = L_\\lambda \\cdot
       (L_\\odot \\cdot c / \\lambda^2)` [erg/s/Hz].

    Each line is broadened independently and superposed additively, so the
    output SED is the sum of all line contributions plus any continuum that
    may be added separately.
    """
    from tengri.utils.physics_constants import C_KM_S

    fwhm_aa = line_wavelengths * fwhm_kms / C_KM_S
    sigma_aa = fwhm_aa / 2.3548200450309493  # 2*sqrt(2*ln2)

    # Gaussian profiles: shape (n_wave, n_lines)
    dwave = wave_obs[:, None] - line_wavelengths[None, :]
    profiles = jnp.exp(-0.5 * (dwave / sigma_aa[None, :]) ** 2)

    # Normalize each profile to unit integrated flux
    norm = sigma_aa * jnp.sqrt(2.0 * jnp.pi)  # (n_lines,)
    profiles = profiles / norm[None, :]  # (n_wave, n_lines)

    # Weighted sum -> L_lambda [Lsun/A]
    l_lambda = profiles @ line_luminosities  # (n_wave,)

    # Convert L_lambda [Lsun/A] -> L_nu [erg/s/Hz] via c/lambda^2 factor
    l_nu = l_lambda * L_SUN * wave_obs**2 * ANGSTROM_CM / C_LIGHT
    return l_nu
