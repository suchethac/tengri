# SPDX-License-Identifier: BSD-3-Clause
"""Polar dust extinction and graybody reemission for AGN.

Implements the X-CIGALE polar dust model (Yang et al. 2020, Section 2.2.2):
bi-conical polar dust with viewing-angle-independent absorption and
energy-conserving graybody FIR reemission. SMC extinction curve applied
to observer-frame disc attenuation (Type 1 sightlines only); absorption
is geometry-independent (Yang+2020 §2.2.2).

The Type 1/2 boundary uses a smooth sigmoid transition for differentiability.

References
----------

- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust § 2.2.2)
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
- Pei 1992, ApJ, 395, 130 (SMC parameterization used here)

"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.components.agn._phys import planck_lnu, wavelength_to_nu
from tengri.components.dust.attenuation import smc as smc_extinction_curve

# Physical constants (CGS / Angstrom-compatible)
from tengri.utils.physics_constants import C_AA as _C_AA

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
    cos_inc: float
        Cosine of inclination angle. 1 = face-on, 0 = edge-on.
    opening_angle_deg: float
        Torus half-opening angle in degrees (measured from equator).
    sharpness: float
        Sigmoid steepness. Higher = sharper transition. Default 20.

    Returns
    -------
    mask: scalar
        Value in [0, 1]. ~1 for Type 1, ~0 for Type 2, ~0.5 at boundary.
    """
    cos_threshold = jnp.cos(jnp.radians(90.0 - opening_angle_deg))
    return jax.nn.sigmoid((cos_inc - cos_threshold) * sharpness)


def calzetti2000_extinction_curve(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Calzetti et al. (2000) dust extinction curve.

    Piecewise polynomial extinction law valid for 0.12–2.2 um (1200–22000 A).
    Widely used for star-forming galaxies and AGN optical/UV extinction.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    k_lambda: ndarray, shape (n_wave,)
        Extinction coefficient k(lambda) = A(lambda) / E(B-V).
        [dimensionless]

    Notes
    -----
    Calzetti et al. (2000) parameterization:

    For :math:`\\lambda < 630` nm (6300 A):

    .. math::

        k(\\lambda) = 2.659 \\times (-2.156 + 1509/\\lambda_{\\rm nm}
                    - 0.198 \\times 10^6/\\lambda_{\\rm nm}^2
                    + 0.011 \\times 10^9/\\lambda_{\\rm nm}^3) + 4.05

    For :math:`\\lambda \\geq 630` nm:

    .. math::

        k(\\lambda) = 2.659 \\times (-1.857 + 1040/\\lambda_{\\rm nm}) + 4.05

    where :math:`\\lambda_{\\rm nm}` is wavelength in nanometers.

    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Gradient-safe**: yes, fully differentiable.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``k_ext()``
    (Boquien et al. 2019 [2]_); validated against its output.

    References
    ----------
    .. [1] D. Calzetti et al., "The Dust Content and Opacity of Actively
       Star-forming Galaxies," ApJ, 533, 682 (2000).
       https://doi.org/10.1086/308692
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Convert Angstrom to nanometers
    wave_nm = wavelength / 10.0

    # Split into two regimes
    x_short = 1.0 / wave_nm  # 1/lambda for short wavelengths

    # Short wavelength (lambda < 630 nm, x > 1/630)
    short_part = (
        2.659 * (-2.156 + 1509.0 * x_short - 0.198e6 * x_short**2 + 0.011e9 * x_short**3) + 4.05
    )

    # Long wavelength (lambda >= 630 nm)
    long_part = 2.659 * (-1.857 + 1040.0 * x_short) + 4.05

    # Select based on wavelength
    k_lambda = jnp.where(wave_nm < 630.0, short_part, long_part)

    return k_lambda


def gaskell2004_extinction_curve(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Gaskell et al. (2004) dust extinction curve.

    Polynomial extinction law parameterized in inverse wavelength space.
    Developed from observations of AGN and suitable for AGN UV/optical
    extinction studies.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    k_lambda: ndarray, shape (n_wave,)
        Extinction coefficient k(lambda) = A(lambda) / E(B-V).
        [dimensionless]

    Notes
    -----
    Gaskell et al. (2004) parameterization in terms of :math:`x = 1000/\\lambda`
    (where :math:`\\lambda` is in nm):

    For :math:`x < 3.69` (wavelength > ~271 nm, 2710 A):

    .. math::

        A(\\lambda) / A_V = -0.8175 + 1.5848 x - 0.3774 x^2 + 0.0296 x^3

    For :math:`x \\geq 3.69`:

    .. math::

        A(\\lambda) / A_V = 1.3468 + 0.0087 x

    The result is then divided by :math:`A_B / A_V = 1.182` to convert from
    A(λ)/A_V to k(λ) = A(λ) / E(B-V).

    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Gradient-safe**: yes, fully differentiable.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``k_ext()``
    (Boquien et al. 2019 [2]_); validated against its output.

    References
    ----------
    .. [1] C. M. Gaskell et al., "A Redetermination of the Reddening of
       AGNs," ApJ, 616, 147 (2004).
       https://doi.org/10.1086/423885
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Convert Angstrom to nanometers
    wave_nm = wavelength / 10.0

    # x = 1000 / lambda_nm
    x = 1000.0 / wave_nm

    # A(lambda) / A_V from polynomial
    a_av_short = -0.8175 + 1.5848 * x - 0.3774 * x**2 + 0.0296 * x**3
    a_av_long = 1.3468 + 0.0087 * x

    # Select regime
    a_av = jnp.where(x < 3.69, a_av_short, a_av_long)

    # Convert A(lambda)/A_V to k(lambda) = A(lambda)/E(B-V)
    # Using CIGALE's convention: k = a_av / 0.182
    # This ensures positivity for all wavelengths
    k_lambda = jnp.maximum(a_av / 0.182, 0.0)

    return k_lambda


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

    Extinction is applied only to Type 1 sightlines (face-on), with a
    smooth sigmoid transition at the Type 1/2 boundary.

    Parameters
    ----------
    l_nu: array, shape (n_wave,)
        Input luminosity density [Lsun/Hz or any consistent unit].
    wavelength: array, shape (n_wave,)
        Wavelength in Angstrom.
    cos_inc: float
        Cosine of inclination. 1 = face-on (Type 1), 0 = edge-on (Type 2).
        [dimensionless, 0–1]
    opening_angle_deg: float
        Torus half-opening angle in degrees (from equator). [degrees]
    ebv: float
        Color excess E(B-V) for the polar dust. 0 = no extinction.
        [dimensionless, mag]
    law: str
        Extinction law name: ``"smc"`` (Pei 1992), ``"calzetti"`` (Calzetti
        et al. 2000), or ``"gaskell"`` (Gaskell et al. 2004).
        Default: ``"smc"``.
    sharpness: float
        Sigmoid steepness at the Type 1/2 boundary. [dimensionless]

    Returns
    -------
    l_nu_attenuated: array, shape (n_wave,)
        Observer-frame disc luminosity after Type-1-masked attenuation.
        Same units as input l_nu. Type-2 sightlines are unchanged (mask ≈ 0).
    l_absorbed: array, shape (n_wave,)
        Absorbed luminosity density (per wavelength bin): bi-conical dust
        absorbs a fraction (1 - exp(-tau_lambda)) regardless of viewing angle.
        Always >= 0. Same units as input l_nu.

    Notes
    -----
    **Absorption vs. Attenuation:**

    - ``l_absorbed`` (geometry-independent) is the disc photon fraction intercepted
      by the bi-conical polar dust (Yang+2020 §2.2.2). This drives the
      graybody FIR reemission and is viewed isotropically.
    - ``l_nu_attenuated`` (Type-1-masked) is the observer-frame disc after
      passing through the near-cone geometry. Only face-on sightlines see
      attenuation; edge-on sightlines (Type 2) have the disc already screened
      by the equatorial torus (handled upstream), so ``l_nu_attenuated = l_nu``.

    **JIT-compatible**: yes, uses ``jnp`` primitives and smooth sigmoid.
    """
    # Select extinction law
    if law == "smc":
        k_lambda = smc_extinction_curve(wavelength)
        r_v = _RV_SMC
    elif law == "calzetti":
        k_lambda = calzetti2000_extinction_curve(wavelength)
        r_v = 1.0  # Calzetti normalized to E(B-V) directly
    elif law == "gaskell":
        k_lambda = gaskell2004_extinction_curve(wavelength)
        r_v = 1.0  # Gaskell normalized to E(B-V) directly
    else:
        # Fallback to SMC
        k_lambda = smc_extinction_curve(wavelength)
        r_v = _RV_SMC

    # A(lambda) = E(B-V) * R_V * k(lambda)
    # Transmission: 10^{-0.4 * A(lambda)} = exp(-0.921 * A(lambda))
    tau_lambda = 0.921 * ebv * r_v * k_lambda
    extinction_factor = jnp.exp(-tau_lambda)  # fraction transmitted

    # Polar-dust absorption is geometry-independent: the bi-conical dust always
    # intercepts the same disc-photon fraction (set by E(B-V)) regardless of
    # observer viewing angle. Re-emission is isotropic, so observers at any
    # inclination see the FIR bump.
    l_absorbed = jnp.maximum(l_nu * (1.0 - extinction_factor), 0.0)

    # Type 1 mask: 1 for face-on (extinct), 0 for edge-on (no effect)
    mask = _type1_mask(cos_inc, opening_angle_deg, sharpness)

    # Observed disc attenuation is gated by Type-1 mask: only face-on sight-lines
    # look through the near polar cone. Type-2 sightlines have the disc already
    # screened by the equatorial torus (handled upstream), so we leave l_nu
    # unchanged here.
    effective_transmission = 1.0 - mask * (1.0 - extinction_factor)
    l_nu_attenuated = l_nu * effective_transmission

    return l_nu_attenuated, l_absorbed


def polar_dust_emission(
    l_absorbed_total: float,
    wavelength: jnp.ndarray,
    temperature: float = 100.0,
    beta: float = 1.6,
    lambda_0: float = 2e6,
) -> jnp.ndarray:
    """Graybody reemission from polar dust.

    Energy-conserving: the integral of the reemitted spectrum equals the
    total absorbed luminosity.

    Parameters
    ----------
    l_absorbed_total: float
        Total absorbed luminosity (scalar, integrated over frequency).
        Same units as input l_nu * delta_nu.
    wavelength: array, shape (n_wave,)
        Wavelength grid [Angstrom].
    temperature: float
        Dust temperature [K]. Default 100.
    beta: float
        Dust emissivity index [dimensionless]. Default 1.6.
    lambda_0: float
        Reference wavelength for optical depth [Angstrom].
        Default 2e6 (= 200 um).

    Returns
    -------
    l_nu_reemit: array, shape (n_wave,)
        Reemitted luminosity density [same units as input l_absorbed_total].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives only.

    **Gradient-safe**: yes, fully differentiable.
    """
    # Graybody: L_nu proportional to (1 - exp(-(lambda_0/lambda)^beta)) * B_nu(T)
    opacity_factor = 1.0 - jnp.exp(-((lambda_0 / wavelength) ** beta))
    b_nu = planck_lnu(wavelength_to_nu(wavelength), temperature)
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


def anisotropic_polar_luminosity(
    l_nu_disk: jnp.ndarray,
    wavelength: jnp.ndarray,
    opening_angle_deg: float,
    extinction_factor: jnp.ndarray,
) -> float:
    """Total extincted luminosity with anisotropic disc geometry.

    Computes the bolometric luminosity accounting for the anisotropic emission
    pattern of the disc as seen through the polar dust. The disc emission varies
    with inclination angle θ as L(θ,λ) ∝ A(λ) · cosθ · (1 + 2cosθ), and the
    observable luminosity is averaged over a solid angle.

    This models the viewing-angle dependent attenuation for an anisotropic
    accretion disc viewed through a clumpy torus with opening angle.

    Parameters
    ----------
    l_nu_disk: array, shape (n_wave,)
        Intrinsic disc luminosity density [erg/s/Hz].
    wavelength: array, shape (n_wave,)
        Wavelength in Angstrom.
    opening_angle_deg: float
        Torus half-opening angle in degrees (from equator). [degrees]
    extinction_factor: array, shape (n_wave,)
        Wavelength-dependent transmission through polar dust
        (i.e., exp(-tau_lambda) from :func:`polar_dust_extinction`).
        [dimensionless, 0–1]

    Returns
    -------
    l_total: float
        Total extincted luminosity integrated over frequency and solid angle
        [erg/s].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives only.

    The anisotropic geometry factor is derived from CIGALE's SKIRTOR module.
    For a given opening angle (related to the torus geometry), the average
    over viewing angles of the anisotropic factor is:

    .. math::

        \\langle f_{\\rm aniso} \\rangle = \\frac{7}{18} - \\frac{\\sin^2 \\Phi}{6}
                                          - \\frac{2 \\sin^3 \\Phi}{9}

    where :math:`\\Phi` is the torus half-opening angle in degrees.

    The extincted luminosity is then:

    .. math::

        L_{\\rm ext} = f_{\\rm aniso} \\int L_\\nu (1 - A_\\nu) \\, d\\nu

    This accounts for both the varying disc brightness with inclination and
    the wavelength-dependent extinction through the polar dust.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``agn_lnu_ir``
    function (Boquien et al. 2019 [2]_); validated against its output.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
       torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
       arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # Compute anisotropic geometry factor
    sin_oa = jnp.sin(jnp.radians(opening_angle_deg))
    aniso_factor = 7.0 / 18.0 - sin_oa**2 / 6.0 - (2.0 / 9.0) * sin_oa**3

    # ABSORBED disc flux per unit frequency: (1 - transmission) × L_nu.
    # ``extinction_factor`` is the wavelength-dependent transmission
    # (exp(-tau_lambda)); the polar dust absorbs the complementary
    # fraction. Matches CIGALE skirtor2016.py:368:
    # ``l_ext = ... × np.trapz(AGN1.disk * (1.0 - ext_fac), x=AGN1.wl)``.
    l_nu_absorbed = l_nu_disk * (1.0 - extinction_factor)

    # Integrate over frequency: convert wavelength integral to frequency integral
    # dnu = -c/lambda^2 dlambda, so |dnu| = c/lambda^2 |dlambda|
    nu = _C_AA / wavelength
    # Use trapezoidal rule on frequency grid (descending order)
    l_total = jnp.trapezoid(l_nu_absorbed[::-1], nu[::-1])

    # Apply anisotropic geometry factor
    l_total = aniso_factor * l_total

    return jnp.maximum(l_total, 0.0)


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
    """Apply polar dust extinction and compute graybody reemission.

    Convenience function combining :func:`polar_dust_extinction` and
    :func:`polar_dust_emission`.

    Parameters
    ----------
    l_nu_disc: array, shape (n_wave,)
        Input AGN disc luminosity density.  Unit-agnostic: output units
        match input (e.g. erg/s/Hz in → erg/s/Hz out).
    wavelength: array, shape (n_wave,)
        Wavelength in Angstrom.
    cos_inc: float
        Cosine of inclination. 1 = face-on, 0 = edge-on.
    opening_angle_deg: float
        Torus half-opening angle in degrees.
    ebv: float
        Color excess E(B-V).
    temperature: float
        Polar dust temperature in Kelvin.
    beta: float
        Dust emissivity index.
    lambda_0: float
        Reference wavelength for optical depth in Angstrom.
    law: str
        Extinction law name.
    sharpness: float
        Sigmoid steepness at the Type 1/2 boundary.

    Returns
    -------
    l_nu_attenuated: array, shape (n_wave,)
        Attenuated disc luminosity (same units as input).
    l_nu_reemit: array, shape (n_wave,)
        Graybody reemission from polar dust (same units as input).
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
