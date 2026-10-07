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
from tengri.utils.scale import representable_floor

__all__ = [
    "ANGSTROM_CM",
    "COS_INC_ISOTROPIC_REFERENCE",
    "C_LIGHT",
    "H_PLANCK",
    "K_BOLTZ",
    "L_SUN",
    "TWO_FACES",
    "bolometric_integral_nu",
    "gaussian_line_profile",
    "lines_to_sed",
    "log10_nu_lnu_at",
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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

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
        When given, return ``max(|integral|, floor)``, the safe form used
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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

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

#: Faces of a thin disc: the angle-integrated power of an annulus is this times
#: :math:`\sigma T^4\,2\pi r\,dr`.
TWO_FACES: float = 2.0
#: cos i at which the line-of-sight power of a ``2 cos i`` disc equals its angle-integrated
#: power (:math:`2 \cos i = 1`).
COS_INC_ISOTROPIC_REFERENCE: float = 0.5


def ring_area(r_cm: float, dr_cm: float, cos_inc: float) -> float:
    r"""Isotropic-equivalent projected area of an annular accretion-disc ring.

    A flat, optically thick ring of radius :math:`r` and width :math:`dr` radiates the
    intensity :math:`B_\nu(T)` from each face, so a distant observer at inclination
    :math:`i` receives :math:`F_\nu = B_\nu\,(2\pi r\,dr \cos i)/d^2`. The luminosity
    density the observer assigns to the ring assuming isotropy is :math:`4\pi d^2 F_\nu`.

    Parameters
    ----------
    r_cm : float
        Ring radius. [cm]
    dr_cm : float
        Ring radial width. [cm]
    cos_inc : float
        Cosine of the inclination angle, in :math:`[0, 1]`. [dimensionless]

    Returns
    -------
    float
        Projected area factor that multiplies :math:`B_\nu`. [cm^2 sr]

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, linear in ``cos_inc``.

    .. math::

        dL_\nu(i) = 4\pi\, B_\nu(T)\; 2\pi r\,dr\; \cos i
                  = 2\cos i\; dD_\nu ,

    where :math:`dD_\nu = 2\pi B_\nu \, 2\pi r\,dr` is the angle-integrated (two-face)
    luminosity density of the ring: :math:`\int dD_\nu\,d\nu = 2\sigma T^4\, 2\pi r\,dr`.
    The mean of :math:`2\cos i` over :math:`\cos i \in [0, 1]` is 1, so the average of the
    ring power over directions is its two-face power, and at :math:`\cos i = 0.5` the
    line-of-sight power equals it. The factor is exactly zero at an edge-on view; there is no
    floor, so the derivative with respect to ``cos_inc`` is the constant :math:`2\,dD_\nu`.

    References
    ----------
    .. [1] G. B. Rybicki and A. P. Lightman, "Radiative Processes in
       Astrophysics," John Wiley & Sons (1979). ISBN: 0-471-82759-2
    .. [2] A. Kubota and C. Done, "A physical model of the broad-band continuum of AGN and
       its implications for the UV/X relation and optical variability," MNRAS, 480, 1247
       (2018), Sect. 2.1. arXiv:1804.00171.
    """
    return 4.0 * jnp.pi * 2.0 * jnp.pi * r_cm * dr_cm * cos_inc


# ── Line list → SED convolution ───────────────────────────────────


def log10_nu_lnu_at(
    wave: jnp.ndarray,
    lnu: jnp.ndarray,
    wavelength_aa: float,
    log10_scale: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    r"""``log10`` of :math:`\nu L_\nu` at one wavelength, by log-log interpolation.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength grid, ascending [Angstrom].
    lnu : array_like, shape (n_wave,)
        Spectral luminosity density sampled on ``wave`` [erg/s/Hz], possibly
        carried at a reference scale (see ``log10_scale``).
    wavelength_aa : float
        Wavelength at which to evaluate :math:`\nu L_\nu` [Angstrom].
    log10_scale : float, optional
        ``log10`` of the factor by which the true spectrum exceeds ``lnu`` [dex].
        Added in log space so a float32 spectrum evaluated at a low reference
        luminosity never forms the true ~1e45 erg/s linear value. Default 0.

    Returns
    -------
    ndarray, scalar
        :math:`\log_{10}[\nu L_\nu(\lambda)/(\mathrm{erg\,s^{-1}})]` [dex];
        ``-inf`` where the spectrum is zero at that wavelength.

    Notes
    -----
    **JIT/grad/vmap-safe.** The interpolation is linear in
    :math:`(\log\lambda, \log L_\nu)`, exact for a power law, and the
    wavelength is clamped to the grid ends by ``jnp.interp``.
    """
    wave = jnp.asarray(wave)
    lnu = jnp.asarray(lnu)
    floor = representable_floor(1e-30)
    log_lnu = jnp.interp(
        jnp.log10(wavelength_aa), jnp.log10(wave), jnp.log10(jnp.maximum(lnu, floor))
    )
    log_nu = jnp.log10(C_LIGHT / (wavelength_aa * ANGSTROM_CM))
    nonzero = log_lnu > jnp.log10(floor) + 1.0
    return jnp.where(nonzero, log_lnu + log_nu + log10_scale, -jnp.inf)


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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable w.r.t. all inputs.

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
