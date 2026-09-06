# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation model for tengri.

This module provides a generalized two-component dust attenuation framework
(Charlot & Fall 2000) with pluggable attenuation curves and clumpy dust
geometry (f_obscuration, Lower 2022).

Structure
---------
Two physical components with independent attenuation curves:

- **Birth cloud**: affects young stars (age < t_birth), optical depth tau_v1
- **Diffuse ISM**: affects all stars, optical depth tau_v2

The wavelength dependence k(lambda) is pluggable per component::

    tau(lambda, age) = w(age) * tau_v1 * k_bc(lambda) + tau_v2 * k_diff(lambda)
    transmission = f_obs + (1 - f_obs) * exp(-tau)

where w(age) is a smooth sigmoid transition at t_birth.

Available Attenuation Curves
----------------------------

- **power_law**: (lambda/5500)^n, original Charlot & Fall (2000)
- **calzetti**: Calzetti et al. (2000) starburst polynomial, R_V=4.05
- **leitherer02**: Leitherer et al. (2002) UV extension of Calzetti (970-1800 A)
- **kriek_conroy**: Calzetti + UV bump + slope delta, the Prospector default
- **noll09**: Noll et al. (2009) modified Calzetti+L02: (base+bump)*slope
- **salim_sbl18**: Salim+2018 modified Calzetti+L02: base*slope+bump
- **smc**: Pei (1992, ApJ 395 130) SMC Bar, steep UV, no 2175A bump
- **cardelli**: Cardelli et al. (1989) MW curve with free R_V
- **li08**: Li et al. (2008) Eq. (1) four-coefficient curve (continuum + FUV rise + 2175 Å bump)
- **salim**: Salim et al. (2018) modified Calzetti (= DSPS default)
- **tea**: Haskell et al. (2024) TEA 3-param empirical (NIHAO-SKIRT bump-slope correlation)
- **narayanan_z**: Narayanan et al. (2018) redshift-dependent Kriek-Conroy (MUFASA RT)
- **conroy2010**: Conroy+2010 mixed MW + power-law (FSPS dust_type=1)
- **vw07_bc**: Wild+2007 birth cloud power-law (n=-1.3)
- **vw07_diff**: Wild+2007 diffuse ISM power-law (n=-0.7)

Dust Geometries (Witt & Gordon 2000)
-------------------------------------

- **wg00_shell**: Foreground screen, standard exp(-tau*k)
- **wg00_cloudy**: Homogeneous dust-star mix (slab), grayer than screen
- **wg00_dusty**: Clumpy two-phase medium (Natta & Panagia 1984), grayest

References
----------

- Calzetti et al. 2000, ApJ, 533, 682
- Cardelli et al. 1989, ApJ, 345, 245
- Charlot & Fall 2000, ApJ, 539, 718
- Conroy, White & Gunn 2010, ApJ, 708, 58
- Gordon et al. 2003, ApJ, 594, 279
- Haskell et al. 2024, arXiv:2401.11007
- Hobson & Padman 1993, MNRAS, 264, 161
- Kriek & Conroy 2013, ApJL, 775, L16
- Leitherer et al. 2002, ApJS, 140, 303
- Li et al. 2008, ApJ, 685, 1046
- Lower et al. 2022, ApJ, 931, 14
- Narayanan et al. 2018, ApJ, 869, 70
- Natta & Panagia 1984, ApJ, 287, 228
- Noll et al. 2009, A&A, 507, 1793
- Salim, Boquien & Lee 2018, ApJ, 859, 11
- Wild et al. 2007, MNRAS, 381, 543
- Witt & Gordon 2000, ApJ, 528, 799
- Zacharegkas et al. 2025, arXiv:2506.19919

"""

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.dust.laws._registry import (
    _HEADLINE_LAWS as _HEADLINE_LAWS,
    DUST_LAWS as DUST_LAWS,
    DustLawRegistryEntry as DustLawRegistryEntry,
    _calzetti_kprime_unnormalized as _calzetti_kprime_unnormalized,
    _calzetti_l02_kprime as _calzetti_l02_kprime,
    _drude_profile as _drude_profile,
    law_kwarg_names as law_kwarg_names,
    list_laws as list_laws,
    register_dust_law as register_dust_law,
    reject_unread_law_kwargs as reject_unread_law_kwargs,
    resolve_dust_law as resolve_dust_law,
    select_law_kwargs as select_law_kwargs,
)
from tengri.utils.physics_constants import V_BAND_ANGSTROM

# ── Attenuation curves ────────────────────────────────────────────


@register_dust_law(
    "power_law",
    citation="Charlot & Fall 2000 (ApJ 539, 718)",
    short_doc="Generic power-law attenuation",
)
def power_law(
    wavelength: jnp.ndarray,
    n_slope: float = -0.7,
) -> jnp.ndarray:
    r"""Power-law dust attenuation curve following Charlot & Fall (2000).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    n_slope : float, optional
        Power-law slope. Default: -0.7 (standard Charlot & Fall). [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), where k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The attenuation is:

    .. math::

        k(\lambda) = \left(\frac{\lambda}{5500 \, \text{\AA}}\right)^n

    where :math:`n = -0.7` produces the standard Charlot & Fall (2000) wavelength
    dependence. Negative slopes make dust redder (stronger attenuation at short wavelengths).

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    return (wavelength / V_BAND_ANGSTROM) ** n_slope


@register_dust_law(
    "vw07_bc",
    citation="Wild et al. 2007 (MNRAS 381, 543)",
    short_doc="Wild+07 birth cloud (n=-1.3, steep UV)",
)
def vw07_bc(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    r"""Wild+2007 birth cloud: power-law with n = -1.3 (steep UV).

    Steeper than the diffuse ISM curve, reflecting the denser dust
    geometry around young stellar populations.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), where k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The attenuation is:

    .. math::

        k(\lambda) = \left(\frac{\lambda}{5500 \, \text{\AA}}\right)^{-1.3}

    :math:`n = -1.3` is the Charlot & Fall (2000) [2]_ birth-cloud slope and is
    a constant OF this law, not a parameter of it: the signature declares no
    ``n_slope``, so ``dust_attenuation={'law': 'vw07_bc', 'slope': ...}`` raises
    rather than being accepted and discarded (#2185). Use ``power_law``, which
    is the same curve with the slope free.

    References
    ----------
    .. [1] V. Wild, G. Kauffmann, T. Heckman, S. Charlot, G. Lemson,
       J. Brinchmann, T. Reichard, and A. Pasquali, "Bursty stellar populations
       and obscured active galactic nuclei in galaxy bulges," MNRAS, 381, 543
       (2007). https://doi.org/10.1111/j.1365-2966.2007.12256.x

    .. [2] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of
       Starlight by Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    return (wavelength / V_BAND_ANGSTROM) ** (-1.3)


@register_dust_law(
    "vw07_diff",
    citation="Wild et al. 2007 (MNRAS 381, 543)",
    short_doc="Wild+07 diffuse ISM (n=-0.7, standard)",
)
def vw07_diff(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    r"""Wild+2007 diffuse ISM: power-law with n = -0.7 (standard CF00).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), where k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The attenuation is:

    .. math::

        k(\lambda) = \left(\frac{\lambda}{5500 \, \text{\AA}}\right)^{-0.7}

    :math:`n = -0.7` is the Charlot & Fall (2000) [2]_ effective absorption
    curve and is a constant OF this law, not a parameter of it: the signature
    declares no ``n_slope``, so
    ``dust_attenuation={'law': 'vw07_diff', 'slope': ...}`` raises rather than
    being accepted and discarded (#2185). Use ``power_law``, which is the same
    curve with the slope free and the same -0.7 default.

    References
    ----------
    .. [1] V. Wild, G. Kauffmann, T. Heckman, S. Charlot, G. Lemson,
       J. Brinchmann, T. Reichard, and A. Pasquali, "Bursty stellar populations
       and obscured active galactic nuclei in galaxy bulges," MNRAS, 381, 543
       (2007). https://doi.org/10.1111/j.1365-2966.2007.12256.x

    .. [2] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of
       Starlight by Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    return (wavelength / V_BAND_ANGSTROM) ** (-0.7)


@register_dust_law(
    "calzetti",
    citation="Calzetti et al. 2000 (ApJ 533, 682)",
    short_doc="Calzetti et al. starburst attenuation (R_V=4.05)",
)
def calzetti(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    r"""Calzetti et al. (2000) starburst attenuation curve.

    R_V = 4.05 (fixed). Valid: 0.12 - 2.2 μm. Widely used for star-forming
    galaxies.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), where k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Uses piecewise polynomials in :math:`x = 1/\lambda` [μm⁻¹]:

    .. math::

        k'(\lambda) = \begin{cases}
        2.659(-2.156 + 1.509x - 0.198x^2 + 0.011x^3) + R_V & \lambda < 0.63 \, \mu{\rm m} \\
        2.659(-1.857 + 1.040x) + R_V & \lambda \geq 0.63 \, \mu{\rm m}
        \end{cases}

    then normalized: :math:`k(\lambda) = k'(\lambda) / R_V` with :math:`R_V = 4.05`.

    :math:`R_V = 4.05 \pm 0.80` is the value Calzetti et al. (2000) [1]_ measure
    for the starburst sample, and the piecewise polynomial above is fitted at it,
    so R_V is a constant OF this law rather than a parameter of it. The signature
    declares no ``dust_Rv``, so
    ``dust_attenuation={'law': 'calzetti', 'Rv': ...}`` raises rather than being
    accepted and discarded (#2185). Use ``cardelli`` or ``conroy2010`` for a free
    R_V.

    References
    ----------
    .. [1] S. Calzetti et al., "The Dust Content and Opacity of Star-Forming
       Galaxies," ApJ, 533, 682 (2000).
       https://doi.org/10.1086/308692
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    k_ir = 2.659 * (-1.857 + 1.040 * x)
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3)

    rv = 4.05
    k_prime = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    # Polynomial is extrapolated through the FUV (< 1200 Å) to keep the
    # dust attenuation defined across the full SED range: users
    # modeling galaxies where Lyman-continuum dust attenuation matters
    # need the curve there. (CIGALE's ``a_vs_ebv`` clips at 912 Å on
    # the assumption that H ionization handles those photons separately;
    # tengri leaves the choice to the user.)
    k = (k_prime + rv) / rv
    # Normalize by k(5500) to ensure k(5500) = 1.0 (#1731: pre-fix value 0.999479)
    # Compute k(5500): at 5500 Å (0.55 μm), use UV formula (< 0.63)
    x_5500 = 1.0 / 0.55
    k_uv_5500 = 2.659 * (-2.156 + 1.509 * x_5500 - 0.198 * x_5500**2 + 0.011 * x_5500**3)
    k_5500 = (k_uv_5500 + rv) / rv
    return jnp.clip(k / k_5500, 0.0)


@register_dust_law(
    "reddy15",
    citation="Reddy et al. 2015 (ApJ 806, 259)",
    short_doc="Reddy et al. MOSDEF attenuation (R_V=2.505)",
)
def reddy15(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    r"""Reddy et al. (2015) MOSDEF dust attenuation curve.

    High-redshift (z~1.4-2.6) attenuation curve derived from MOSDEF galaxies
    using Balmer decrements. R_V = 2.505 (lower than Calzetti). Valid:
    0.15 - 2.85 μm. Similar to SMC at long wavelengths.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), where k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Uses piecewise polynomials in :math:`x = 1/\lambda` [μm⁻¹]:

    .. math::

        k(\lambda) = \begin{cases}
        -5.726 + 4.004 x - 0.525 x^2 + 0.029 x^3 + R_V & 0.15 \leq \lambda < 0.60 \, \mu{\rm m} \\
        -2.672 - 0.010 x + 1.532 x^2 - 0.412 x^3 + R_V & 0.60 \leq \lambda \leq 2.85 \, \mu{\rm m}
        \end{cases}

    then normalized: :math:`k(\lambda) = k(\lambda) / R_V` with :math:`R_V = 2.505`.
    The normalization ensures :math:`k(5500 \, \text{\AA}) = 1`.

    **Approximation**: The polynomial form is valid over 0.15–2.85 μm. Extrapolation
    beyond this range follows the functional form but is not empirically constrained
    (Reddy et al. 2015, Section 3.6.1).

    References
    ----------
    .. [1] N. A. Reddy, M. Kriek, A. E. Shapley, W. R. Freeman, B. Siana,
       A. L. Coil, B. Mobasher, S. H. Price, R. L. Sanders, and I. Shivaei,
       "The MOSDEF Survey: Measurements of Balmer Decrements and the Dust
       Attenuation Curve at Redshifts z ~ 1.4–2.6," ApJ, 806, 259 (2015).
       arXiv:1504.02782. https://doi.org/10.1088/0004-637X/806/2/259
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    # Low-wavelength segment (0.15 <= lambda < 0.60 um)
    k_low = -5.726 + 4.004 * x - 0.525 * x**2 + 0.029 * x**3

    # High-wavelength segment (0.60 <= lambda <= 2.85 um)
    k_high = -2.672 - 0.010 * x + 1.532 * x**2 - 0.412 * x**3

    rv = 2.505
    k_prime = jnp.where(wave_um < 0.60, k_low, k_high) + rv
    k = k_prime / rv
    # Normalize by k(5500) to ensure k(5500) = 1.0 (#1731: pre-fix value 0.997113)
    # Compute k(5500): at 5500 Å (0.55 μm), use low-wavelength segment (< 0.60)
    x_5500 = 1.0 / 0.55
    k_5500_unnorm = -5.726 + 4.004 * x_5500 - 0.525 * x_5500**2 + 0.029 * x_5500**3
    k_5500 = (k_5500_unnorm + rv) / rv
    return jnp.clip(k / k_5500, 0.0)


@register_dust_law(
    "kriek_conroy",
    citation="Kriek & Conroy 2013 (ApJL 775, L16)",
    short_doc="Kriek & Conroy modified Calzetti + bump + slope",
)
def kriek_conroy(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 1.0,
    dust_delta: float = 0.0,
) -> jnp.ndarray:
    r"""Kriek & Conroy (2013) modified Calzetti + UV bump + slope delta.

    Default attenuation law in Prospector, applied there through FSPS
    (``dust_type=4``). Reproduces FSPS' construction: a Calzetti baseline
    tilted by a power law in :math:`\lambda`, plus a 2175 Å Drude bump
    whose amplitude is **coupled to the slope** via Kriek & Conroy (2013)
    Eqn 3, with the bump divided by :math:`R_V = 4.05`.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_bump_strength : float
        Multiplier on the KC13-derived bump amplitude
        :math:`E_b = 0.85 - 1.9\,\delta`. [dimensionless] Default: 1.0,
        which reproduces FSPS ``dust_type=4`` exactly; set to 0.0 to
        remove the bump, or scale to weaken/strengthen it.
    dust_delta : float
        Power-law slope modification :math:`\delta`. [dimensionless]
        Default: 0.0. Steeper (more negative) :math:`\delta` gives a
        stronger bump, per KC13 Eqn 3.

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), where k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Following FSPS ``attn_curve.f90`` (``dust_type=4``), the unnormalized
    curve is:

    .. math::

        k'(\lambda) = \left[ k_{\rm Calz}(\lambda)
        + \frac{E_b}{R_V}\,D(\lambda; \lambda_0 = 2175\,\text{\AA}) \right]
        \left(\frac{\lambda}{5500\,\text{\AA}}\right)^{\delta},
        \qquad E_b = 0.85 - 1.9\,\delta

    with :math:`R_V = 4.05`, :math:`k_{\rm Calz}` the Calzetti et al.
    (2000) curve, and :math:`D` the unit-peak Drude profile. The result
    is renormalized to :math:`k(5500\,\text{\AA}) = 1`. The
    ``dust_bump_strength`` multiplier scales :math:`E_b`.

    **Upstream**: follows the FSPS ``dust_type=4`` branch (Conroy, Gunn &
    White 2009); the slope–bump coupling is Kriek & Conroy (2013) Eqn 3.

    References
    ----------
    .. [1] M. Kriek and C. Conroy, "The Dust Attenuation Law in Distant Galaxies:
       Evidence for Variation with Spectral Type," ApJL, 775, L16 (2013).
       https://doi.org/10.1088/2041-8205/775/1/L16
    .. [2] C. Conroy, J. E. Gunn, M. White, "The Propagation of
       Uncertainties in Stellar Population Synthesis Modeling. I.,"
       ApJ, 699, 486 (2009). https://doi.org/10.1088/0004-637X/699/1/486
    """
    R_V = 4.05
    wave_um = wavelength / 1e4
    e_b = dust_bump_strength * (0.85 - 1.9 * dust_delta)  # KC13 Eqn 3

    # Use unnormalized Calzetti base to avoid double normalization (#1731).
    # _calzetti_kprime_unnormalized returns k'(λ) without R_V or k(5500) normalization.
    k_calz_kprime = _calzetti_kprime_unnormalized(wavelength)
    k_calz = k_calz_kprime / R_V
    slope_mod = (wavelength / V_BAND_ANGSTROM) ** dust_delta
    k_unnorm = (k_calz + e_b * _drude_profile(wave_um) / R_V) * slope_mod

    # Normalize to k(5500 Å) = 1 (slope_mod = 1 at V band).
    v_band_um = V_BAND_ANGSTROM / 1e4
    k_calz_kprime_v = _calzetti_kprime_unnormalized(jnp.array([V_BAND_ANGSTROM]))[0]
    k_calz_v = k_calz_kprime_v / R_V
    bump_v = e_b * _drude_profile(jnp.array([v_band_um]))[0] / R_V
    k_at_v = k_calz_v + bump_v

    k = k_unnorm / k_at_v
    return jnp.clip(k, 0.0)


def _pei92_curve(
    wavelength: jnp.ndarray,
    lam_i: jnp.ndarray,
    a_i: jnp.ndarray,
    b_i: jnp.ndarray,
    n_i: jnp.ndarray,
    R_V: float,
) -> jnp.ndarray:
    r"""Pei (1992, ApJ, 395, 130) generalized Drude profile sum.

    Computes A(lambda)/A(V) normalized to k(5500 A) = 1. Fully continuous,
    no piecewise boundaries. Used for SMC and LMC extinction curves.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    lam_i : array_like, shape (n_components,)
        Central wavelengths (in μm) of Drude components. [μm]
    a_i : array_like, shape (n_components,)
        Amplitudes of Drude components. [dimensionless]
    b_i : array_like, shape (n_components,)
        Denominatornormalization coefficients. [dimensionless]
    n_i : array_like, shape (n_components,)
        Power-law exponents for profile broadening. [dimensionless]
    R_V : float
        Total-to-selective extinction ratio. [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized extinction curve k(λ). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The extinction function is:

    .. math::

        \xi(\lambda) = \sum_i \frac{a_i}{(\lambda/\lambda_i)^{n_i} + (\lambda_i/\lambda)^{n_i} + b_i}

    Normalized by :math:`\xi(V) / R_V` where V = 5500 Å.

    References
    ----------
    .. [1] P. G. Pei, "Interstellar Dust from the Ultraviolet to the Infrared,"
       ApJ, 395, 130 (1992).
       https://doi.org/10.1086/171637
    """
    wave_um = wavelength / 1e4  # (n_wave,)
    # xi(lambda) = sum_i a_i / ((lam/lam_i)^n_i + (lam_i/lam)^n_i + b_i)
    #
    # The Drude component axis is TRAILING, so the reduction is axis=-1 and this
    # holds for a wavelength grid of any rank. Spelling it the other way round
    # (``wave_um[None, :]`` against ``lam_i[:, None]``) pins the input to rank 1:
    # it works on the exact path, which passes a 1-D grid, and raises under
    # WavePrecomp, which evaluates the curve on a (sub-band, age, filter) grid
    # where the 6-component axis collides with the filter axis. Every other
    # attenuation law is elementwise in wavelength and so broadcast at any rank;
    # only the Pei-92 pair sums over a component axis, which is why smc and lmc
    # alone were unusable on the path every fitter resolves approx="auto" to.
    ratio = wave_um[..., None] / lam_i  # (..., n_components)
    denom = ratio**n_i + ratio ** (-n_i) + b_i
    xi = jnp.sum(a_i / denom, axis=-1)  # (...)

    # Pei 1992 Drude sum gives extinction proportional to tau(lambda).
    # A(lambda)/A(V) = xi(lambda) / xi(V). Normalize to k(5500 A) = 1.
    wave_v = 0.55  # um
    ratio_v = wave_v / lam_i
    xi_v = jnp.sum(a_i / (ratio_v**n_i + ratio_v ** (-n_i) + b_i))
    return jnp.clip(xi / xi_v, 0.0)


# Pei 1992 Table 4: SMC Bar (6 components, R_V = 2.93, no 2175 A bump)
_SMC_LAM = jnp.array([0.042, 0.08, 0.22, 9.7, 18.0, 25.0])
_SMC_A = jnp.array([185.0, 27.0, 0.005, 0.010, 0.012, 0.030])
_SMC_B = jnp.array([90.0, 5.50, -1.95, -1.95, -1.80, 0.00])
_SMC_N = jnp.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0])
_SMC_RV = 2.93

# Pei 1992 Table 4: LMC (6 components, R_V = 3.16, weak 2175 A bump)
_LMC_LAM = jnp.array([0.046, 0.08, 0.22, 9.7, 18.0, 25.0])
_LMC_A = jnp.array([175.0, 19.0, 0.023, 0.005, 0.006, 0.020])
_LMC_B = jnp.array([90.0, 5.50, -1.95, -1.95, -1.80, 0.00])
_LMC_N = jnp.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0])
_LMC_RV = 3.16


@register_dust_law(
    "smc",
    citation="Pei 1992 (ApJ 395, 130) SMC Bar",
    short_doc="SMC extinction (Pei 1992, no 2175 bump)",
)
def smc(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    """SMC Bar extinction curve (Pei 1992, ApJ, 395, 130).

    Steep UV rise, NO 2175 Å bump. Common at high redshift. R_V = 2.93.
    Uses generalized Drude profile sum: fully continuous, no piecewise
    boundaries.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized extinction curve k(λ). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    From Pei (1992) Table 4: Small Magellanic Cloud Bar parameters,
    6 Drude components with R_V = 2.93 (relatively gray).

    References
    ----------
    P. G. Pei, "Interstellar Dust from the Ultraviolet to the Infrared,"
    ApJ, 395, 130 (1992).
    """
    return _pei92_curve(wavelength, _SMC_LAM, _SMC_A, _SMC_B, _SMC_N, _SMC_RV)


@register_dust_law(
    "lmc", citation="Pei 1992 (ApJ 395, 130) LMC", short_doc="LMC extinction (Pei 1992, weak bump)"
)
def lmc(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    """LMC average extinction curve (Pei 1992, ApJ, 395, 130).

    Weak 2175 Å bump, intermediate between MW and SMC. R_V = 3.16.
    Uses generalized Drude profile sum: fully continuous.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized extinction curve k(λ). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    From Pei (1992) Table 4: Large Magellanic Cloud average parameters,
    6 Drude components with R_V = 3.16. Shows weak 2175 Å bump feature.

    References
    ----------
    P. G. Pei, "Interstellar Dust from the Ultraviolet to the Infrared,"
    ApJ, 395, 130 (1992).
    """
    return _pei92_curve(wavelength, _LMC_LAM, _LMC_A, _LMC_B, _LMC_N, _LMC_RV)


@register_dust_law(
    "prevot_smc",
    citation="Prevot et al. 1984 (A&A 132, 389)",
    short_doc="Prevot et al. SMC extinction for AGN",
)
def prevot_smc(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    r"""Prevot et al. (1984) SMC extinction law for AGN obscuration.

    Analytic SMC extinction curve from the UV to near-infrared, used in
    AGNfitter for AGN disc reddening. The analytic Prevot+1984 fit
    :math:`k_{\rm raw}(\lambda) = 1.39\,\lambda_{\mu m}^{-1.2} - 0.38`
    evaluates to :math:`k_{\rm raw}(0.55\,\mu m) \approx 2.468` at V; this
    function returns the V-band-normalized *shape*
    :math:`k(\lambda) = k_{\rm raw}(\lambda)/k_{\rm raw}(V)` (so ``k(V)=1``),
    matching the convention used by ``cardelli`` and the rest of the
    ``components.dust.attenuation`` registry. The measured SMC ratio
    :math:`R_V = 2.72` (Prevot+1984) is applied *separately* by
    :func:`tengri.components.agn.reddening.redden_disc` as
    :math:`A(\lambda) = k(\lambda)\,R_V\,E(B-V)`. AGNfitter's ``BBBred_Prevot``
    instead applies the bare :math:`k_{\rm raw}` (its ``function_prevot``
    declares ``RV=2.72`` but does not use the argument), so its effective
    ratio is 2.468 and its ``EBVbbb`` maps onto tengri's as
    :math:`E(B-V)_{\rm tengri} \approx E(B-V)_{\rm AGNfitter} / 1.102`.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Extinction curve :math:`k(\lambda) = A(\lambda)/A(V)` normalized
        to ``k(5500 A) = 1``. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    .. math::

        k_{\rm raw}(\lambda) = 1.39 \, \lambda_{\mu m}^{-1.2} - 0.38,
        \qquad
        k(\lambda) = k_{\rm raw}(\lambda) / k_{\rm raw}(0.55\,\mu m)

    where :math:`\lambda_{\mu m}` is wavelength in micrometers and
    :math:`k_{\rm raw}(0.55) \approx 2.468`.

    For wavelengths below 62 Å, reddening is ramped to zero using a smooth
    sigmoid (no hard discontinuity) to suppress extinction in the X-ray regime
    where dust is ineffective.

    **Gradient-compatible**: yes, enables optimization of extinction parameters.

    References
    ----------
    .. [1] M. Prevot et al., "The Ultraviolet Extinction Curve in the Small
       Magellanic Cloud from 1200 Å to 3200 Å," A&A, 132, 389 (1984).
    .. [2] J. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC approach to
       fitting spectral energy distributions of Active Galactic Nuclei,"
       ApJ, 863, 56 (2018). arXiv:1808.04989.
       https://doi.org/10.3847/1538-4357/aad235
    """
    # Convert wavelength from Å to μm
    wavelength_um = wavelength / 1e4
    # Raw Prevot+1984 form: A(lambda)/E(B-V) with lambda in μm. The
    # published fit is only calibrated for 1200–3200 Å (UV); beyond
    # ~5–10 μm the analytic form goes negative because of the constant
    # offset (k_raw → -0.38 as λ → ∞). Clamp to zero to enforce the
    # physical constraint k(λ) ≥ 0: extrapolation past the calibrated
    # range gets no attenuation rather than negative attenuation, in
    # keeping with how Synthesizer extrapolates the Calzetti grid.
    k_raw = jnp.maximum(1.39 * jnp.power(wavelength_um, -1.2) - 0.38, 0.0)
    # Normalize to k(V) = 1 (V band = 5500 Å = 0.55 μm) so that
    # the result is A(lambda)/A(V), matching tengri's dust-law convention.
    k_v_raw = 1.39 * (0.55) ** (-1.2) - 0.38  # ≈ 2.4683
    k_norm = k_raw / k_v_raw

    # Suppress extinction for lambda < 62 A (X-ray regime, E > 200 eV)
    # where dust is ineffective. Use a smooth sigmoid ramp to avoid discontinuities.
    # Sigmoid: f(x) = 1 / (1 + exp(-slope * (x - x_0)))
    # At x = 62 A: f = 0.5 (half suppression)
    # At x = 20 A: f ~ 0 (full suppression)
    # At x = 100 A: f ~ 1 (no suppression)
    lambda_xray_edge = 62.0  # Å
    sigmoid_slope = 0.5  # steeper transition for more aggressive ramp-down
    ramp_factor = 1.0 / (1.0 + jnp.exp(-sigmoid_slope * (wavelength - lambda_xray_edge)))

    return k_norm * ramp_factor


@register_dust_law(
    "cardelli",
    citation="Cardelli et al. 1989 (ApJ 345, 245) MW extinction",
    short_doc="Cardelli et al. MW extinction (parameterized R_V)",
)
def cardelli(
    wavelength: jnp.ndarray,
    dust_Rv: float = 3.1,
) -> jnp.ndarray:
    r"""Cardelli, Clayton & Mathis (1989) MW extinction with free R_V.

    Detailed piecewise fit to Milky Way extinction spanning UV to IR.
    R_V parameterization allows flexibility for different dust types.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_Rv : float
        Total-to-selective extinction ratio. [dimensionless] Default: 3.1.

    Returns
    -------
    ndarray, shape (n_wave,)
        Extinction curve A(λ)/A(V) normalized to k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Uses piecewise polynomials in infrared, optical, UV, and far-UV regimes.
    Parameterized by :math:`x = 1/\lambda` [μm⁻¹] with :math:`a(x)` and :math:`b(x)`
    coefficients fitted to extinction curves.

    References
    ----------
    .. [1] D. E. Cardelli, G. C. Clayton, and J. S. Mathis, "The Relationship
       between Infrared, Optical, and Ultraviolet Extinction," ApJ, 345, 245 (1989).
       https://doi.org/10.1086/167900
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    # IR: 0.3 <= x <= 1.1
    a_ir = 0.574 * x**1.61
    b_ir = -0.527 * x**1.61

    # Optical: 1.1 <= x <= 3.3
    y = x - 1.82
    a_opt = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b_opt = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    # UV: 3.3 <= x <= 8.0
    f_a = jnp.where(x >= 5.9, -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3, 0.0)
    f_b = jnp.where(x >= 5.9, 0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3, 0.0)
    a_uv = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + f_a
    b_uv = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + f_b

    # Far-UV: 8.0 <= x <= 10.0 (CCM89 Table 4)
    y_fuv = jnp.clip(x, 8.0, 10.0) - 8.0
    a_fuv = -1.073 - 0.628 * y_fuv + 0.137 * y_fuv**2 - 0.070 * y_fuv**3
    b_fuv = 13.670 + 4.257 * y_fuv - 0.420 * y_fuv**2 + 0.374 * y_fuv**3

    a = jnp.where(
        x < 1.1,
        a_ir,
        jnp.where(x < 3.3, a_opt, jnp.where(x < 8.0, a_uv, a_fuv)),
    )
    b = jnp.where(
        x < 1.1,
        b_ir,
        jnp.where(x < 3.3, b_opt, jnp.where(x < 8.0, b_uv, b_fuv)),
    )

    k = a + b / dust_Rv
    # Normalize by k(5500) to ensure k(5500) = 1.0 (#1731: pre-fix default dust_Rv=3.1 value 0.998850)
    # Compute k(5500) for the current dust_Rv value
    x_5500 = 1.0 / 0.55  # 5500 Å = 0.55 μm
    a_5500 = 1.0 + 0.17699 * (x_5500 - 1.82) - 0.50447 * (x_5500 - 1.82) ** 2
    b_5500 = 1.41338 * (x_5500 - 1.82) + 2.28305 * (x_5500 - 1.82) ** 2
    k_5500 = a_5500 + b_5500 / dust_Rv
    return jnp.clip(k / k_5500, 0.0)


@register_dust_law(
    "li08",
    citation="Li et al. 2008 (ApJ 685, 1046)",
    short_doc="Li et al. analytical flexible attenuation curve",
)
def li08(
    wavelength: jnp.ndarray,
    dust_c1: float = 6.0,
    dust_c2: float = 4.0,
    dust_c3: float = 2.0,
    dust_c4: float = 0.04,
) -> jnp.ndarray:
    """Li et al. (2008) analytical dust attenuation/extinction curve.

    A flexible 4-parameter analytical model that can reproduce MW, LMC,
    SMC, and Calzetti-like curves through a single functional form.
    The three terms represent a UV/optical continuum, a far-UV rise,
    and the 2175 Angstrom UV bump respectively.

    The normalized attenuation is (Li et al. 2008, Eq. 1)::

        A_lam/A_V = c1 / [(lam/0.08)^c2 + (0.08/lam)^c2 + c3]
                  + 233 * [1 - c1/(6.88^c2 + 0.145^c2 + c3) - c4/4.60]
                    / [(lam/0.046)^2 + (0.046/lam)^2 + 90]
                  + c4 / [(lam/0.2175)^2 + (0.2175/lam)^2 - 1.95]

    where lam is the wavelength in micron and c1-c4 are dimensionless.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_c1 : float
        Continuum amplitude. Controls overall UV-optical shape.
    dust_c2 : float
        Continuum curvature. Higher values produce steeper UV rises.
    dust_c3 : float
        Continuum offset. Shifts the overall curve level.
    dust_c4 : float
        UV bump amplitude at 2175 Angstrom. Set to 0 for bump-free.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda), normalized to k(5500 A) = 1.

    Notes
    -----
    Approximate presets for common curves (Markov et al. 2023, 2025):

    - **MW-like**: c1~6.0, c2~4.0, c3~2.0, c4~0.04
    - **SMC-like**: c1~5.0, c2~5.5, c3~1.5, c4~0.0
    - **Calzetti-like**: c1~3.5, c2~2.5, c3~3.0, c4~0.0

    References
    ----------
    Li, A., Liang, S. L., Kann, D. A., et al. 2008, ApJ, 685, 1046
    Markov, V., Gallerani, S., Pallottini, A., et al. 2023, A&A, 679, A12
    Markov, V., Gallerani, S., Pallottini, A., et al. 2025, A&A (arXiv:2504.12378)
    """
    lam = wavelength / 1e4  # Angstrom -> micron

    # Term 1: UV/optical continuum
    t1 = dust_c1 / ((lam / 0.08) ** dust_c2 + (0.08 / lam) ** dust_c2 + dust_c3)

    # Normalization constant for term 2 (ensures A_V continuity)
    norm_c1 = dust_c1 / (6.88**dust_c2 + 0.145**dust_c2 + dust_c3)
    norm_c4 = dust_c4 / 4.60

    # Term 2: Far-UV rise
    t2 = 233.0 * (1.0 - norm_c1 - norm_c4) / ((lam / 0.046) ** 2 + (0.046 / lam) ** 2 + 90.0)

    # Term 3: 2175 Angstrom UV bump
    t3 = dust_c4 / ((lam / 0.2175) ** 2 + (0.2175 / lam) ** 2 - 1.95)

    a_lam_over_av = t1 + t2 + t3

    # Evaluate at V-band (5500 A = 0.55 um) for normalization to k(5500)=1
    lam_v = 0.55
    t1_v = dust_c1 / ((lam_v / 0.08) ** dust_c2 + (0.08 / lam_v) ** dust_c2 + dust_c3)
    t2_v = 233.0 * (1.0 - norm_c1 - norm_c4) / ((lam_v / 0.046) ** 2 + (0.046 / lam_v) ** 2 + 90.0)
    t3_v = dust_c4 / ((lam_v / 0.2175) ** 2 + (0.2175 / lam_v) ** 2 - 1.95)
    a_v_over_av = t1_v + t2_v + t3_v

    # Return k(lambda) = A_lambda/A_V normalized so k(5500 A) = 1
    return jnp.clip(a_lam_over_av / a_v_over_av, 0.0)


@register_dust_law(
    "salim",
    citation="Salim et al. 2018 (ApJ 859, 11)",
    short_doc="Salim et al. modified Calzetti (DSPS default)",
)
def salim(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
) -> jnp.ndarray:
    """Salim et al. (2018) modified Calzetti law (DSPS/Zacharegkas+2025 default).

    Same functional form as :func:`kriek_conroy`; aliased here as the default
    Zacharegkas+2025 diffuse ISM attenuation law.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelengths. [Angstrom]
    dust_bump_strength : float
        Amplitude of the 2175 Angstrom UV bump. [dimensionless]
    dust_delta : float
        Power-law tilt relative to Calzetti slope. [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(lambda) normalized to k(5500 A) = 1.
        [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, uses only jnp primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    This is the default attenuation law in DSPS and Zacharegkas+2025
    simulations. It combines Calzetti et al. (2000) optical/NIR continuum
    with Leitherer et al. (2002) UV extension, plus a variable 2175 Å bump
    and power-law tilt.
    """
    return kriek_conroy(
        wavelength,
        dust_bump_strength=dust_bump_strength,
        dust_delta=dust_delta,
    )


@register_dust_law(
    "leitherer02",
    citation="Leitherer et al. 2002 (ApJS 140, 303)",
    short_doc="Leitherer et al. UV starburst attenuation",
)
def leitherer02(
    wavelength: jnp.ndarray,
) -> jnp.ndarray:
    r"""Leitherer et al. (2002) UV starburst attenuation curve.

    Far-UV extension of the Calzetti (2000) law, valid 970-1800 Å.
    Uses R_V = 4.05 (same as Calzetti). For wavelengths outside the
    L02 range, falls back to Calzetti (2000).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(λ), normalized to k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The reddening curve k'(λ) = A(λ)/E(B−V) follows:

    .. math::

        k'(\lambda) = 5.472 + 0.671x - 9.218 \times 10^{-3} x^2 + 2.620 \times 10^{-3} x^3

    where :math:`x = 1/\lambda` [μm⁻¹], valid for 970–1800 Å (0.097–0.18 μm).
    The final k(λ) = k'(λ) / R_V with R_V = 4.05.

    References
    ----------
    .. [1] C. Leitherer et al., "Global Far-Ultraviolet (912–1800 Å) Properties of
       Star-forming Galaxies," ApJS, 140, 303 (2002).
       https://doi.org/10.1086/342486
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um
    rv = 4.05

    # L02 polynomial (valid 0.097-0.18 um)
    k_l02 = 5.472 + 0.671 * x - 9.218e-3 * x**2 + 2.620e-3 * x**3

    # Calzetti UV polynomial (valid 0.12-0.63 um)
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + rv

    # Calzetti IR polynomial (valid 0.63-2.2 um)
    k_ir = 2.659 * (-1.857 + 1.040 * x) + rv

    # Use L02 up to 1800 Å (full L02 range), Calzetti above.
    # Compare in Å (exact for clean integers) rather than wave_um <= 0.18:
    # the latter is platform-dependent at the boundary because 1800/1e4 is
    # not exactly representable in float64.
    k_calz = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    k_prime = jnp.where(wavelength <= 1800.0, k_l02, k_calz)

    # The L02 polynomial is extended through the FUV to keep dust
    # attenuation defined across the full UV grid. CIGALE clips at
    # 912 Å on the assumption that H ionization handles Lyman-continuum
    # photons separately; tengri leaves the choice to the user.
    k = k_prime / rv
    # Normalize by k(5500) to ensure k(5500) = 1.0 (#1731: pre-fix value 0.999479)
    # Compute k(5500): at 5500 Å (0.55 μm), use UV formula (< 0.63)
    x_5500 = 1.0 / 0.55
    k_uv_5500 = 2.659 * (-2.156 + 1.509 * x_5500 - 0.198 * x_5500**2 + 0.011 * x_5500**3) + rv
    k_5500 = k_uv_5500 / rv
    return jnp.clip(k / k_5500, 0.0)


@register_dust_law(
    "noll09",
    citation="Noll et al. 2009 (A&A 507, 1793)",
    short_doc="Noll et al. modified Calzetti + L02 + bump + slope",
)
def noll09(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
    dust_bump_x0: float = 0.2175,
    dust_bump_gamma: float = 0.035,
) -> jnp.ndarray:
    r"""Noll et al. (2009) modified Calzetti + L02 with UV bump + slope delta.

    This is the ``N09`` model from the ``dust_attenuation`` package.
    Uses Leitherer (2002) for λ < 1500 Å and Calzetti (2000) above.
    The modification order is: **(base + bump) × power_law**.

    This differs from ``kriek_conroy`` which does NOT use L02 and applies
    the bump AFTER the slope: ``base × power_law + bump``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_bump_strength : float
        Amplitude of 2175 Å UV bump (E_b). [dimensionless] Default: 0.0 (no bump).
    dust_delta : float
        Power-law slope modification. [dimensionless] Default: 0.0 (pure Calzetti+L02).
    dust_bump_x0 : float
        Central wavelength of UV bump. [μm] Default: 0.2175.
    dust_bump_gamma : float
        FWHM of UV bump. [μm] Default: 0.035.

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(λ) = k'(λ) / R_V with R_V = 4.05. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The attenuation is:

    .. math::

        k(\lambda) = [k_{\rm L02+C00}(\lambda) + E_b D(\lambda; \lambda_0, \gamma)]
        \times \left(\frac{\lambda}{5500 \, \text{\AA}}\right)^\delta / R_V

    Normalization follows the ``dust_attenuation`` package convention: k(λ) = k'(λ)/R_V
    where R_V = 4.05 is fixed. This means k(V) is NOT exactly 1.0 when bump or slope
    modifications are applied.

    References
    ----------
    .. [1] S. Noll, S. Pierini, B. Coles, et al., "On the link between
       galaxy morphology and supermassive black holes in the nearby Universe,"
       A&A, 507, 1793 (2009).
       https://doi.org/10.1051/0004-6361/200912497
    """
    wave_um = wavelength / 1e4
    rv = 4.05

    # Base k'(lambda) = A(lambda)/E(B-V): L02 below 0.15 um, Calzetti above
    k_base = _calzetti_l02_kprime(wavelength)

    # UV bump (Drude profile)
    bump = dust_bump_strength * _drude_profile(wave_um, x0=dust_bump_x0, gamma=dust_bump_gamma)

    # Power law slope modification: (lambda / 0.55 um)^delta
    slope_mod = (wave_um / 0.55) ** dust_delta

    # N09 order: (base + bump) * slope_mod
    k_prime = (k_base + bump) * slope_mod

    # Normalize by fixed Rv (package convention, NOT by k_prime(V))
    k = k_prime / rv
    # Normalize by k(5500) to ensure k(5500) = 1.0 (#1731: pre-fix value 0.999479)
    # Compute k(5500): k_base(5500) uses UV formula since 5500 Å < 0.63 μm
    wave_5500 = jnp.asarray(5500.0)
    k_base_5500 = _calzetti_l02_kprime(wave_5500)
    bump_5500 = dust_bump_strength * _drude_profile(
        jnp.asarray(0.55), x0=dust_bump_x0, gamma=dust_bump_gamma
    )
    k_prime_5500 = (k_base_5500 + bump_5500) * 1.0  # slope_mod(5500) = 1 for any dust_delta
    k_5500 = k_prime_5500 / rv
    return jnp.clip(k / k_5500, 0.0)


@register_dust_law(
    "salim_sbl18",
    citation="Salim et al. 2018 (ApJ 859, 11)",
    short_doc="Salim, Boquien & Lee modified Calzetti + L02",
)
def salim_sbl18(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
    dust_bump_x0: float = 0.2175,
    dust_bump_gamma: float = 0.035,
) -> jnp.ndarray:
    r"""Salim, Boquien & Lee (2018) modified Calzetti + L02 with UV bump + slope.

    This is the ``SBL18`` model from the ``dust_attenuation`` package.
    Uses Leitherer (2002) for λ < 1500 Å and Calzetti (2000) above.
    The modification order is: **(base × power_law) + bump**.

    This differs from ``noll09`` which applies: ``(base + bump) × power_law``.
    The SBL18 order is identical to ``kriek_conroy``, but SBL18 additionally
    uses L02 in the far-UV.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_bump_strength : float
        Amplitude of 2175 Å UV bump (E_b). [dimensionless] Default: 0.0 (no bump).
    dust_delta : float
        Power-law slope modification. [dimensionless] Default: 0.0 (pure Calzetti+L02).
    dust_bump_x0 : float
        Central wavelength of UV bump. [μm] Default: 0.2175.
    dust_bump_gamma : float
        FWHM of UV bump. [μm] Default: 0.035.

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(λ) = k'(λ) / R_V with R_V = 4.05. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The attenuation is:

    .. math::

        k(\lambda) = \left[k_{\rm L02+C00}(\lambda) \times \left(\frac{\lambda}{5500 \, \text{\AA}}\right)^\delta
        + E_b D(\lambda; \lambda_0, \gamma)\right] / R_V

    References
    ----------
    .. [1] S. Salim, M. Boquien, and J. C. Lee, "CANDELS: Constraining the AGN
       Contribution to the Star Formation Rate Density at z > 1,"
       ApJ, 859, 11 (2018).
       https://doi.org/10.3847/1538-4357/aabf3c
    """
    wave_um = wavelength / 1e4
    rv = 4.05

    # Base k'(lambda): L02 below 0.15 um, Calzetti above
    k_base = _calzetti_l02_kprime(wavelength)

    # UV bump (Drude profile)
    bump = dust_bump_strength * _drude_profile(wave_um, x0=dust_bump_x0, gamma=dust_bump_gamma)

    # Power law slope modification
    slope_mod = (wave_um / 0.55) ** dust_delta

    # SBL18 order: (base * slope_mod) + bump
    k_prime = k_base * slope_mod + bump

    # Normalize by fixed Rv (package convention)
    k = k_prime / rv
    # Normalize by k(5500) to ensure k(5500) = 1.0 (#1731: pre-fix value 0.999479)
    # Compute k(5500): k_base(5500) uses UV formula since 5500 Å < 0.63 μm
    wave_5500 = jnp.asarray(5500.0)
    k_base_5500 = _calzetti_l02_kprime(wave_5500)
    bump_5500 = dust_bump_strength * _drude_profile(
        jnp.asarray(0.55), x0=dust_bump_x0, gamma=dust_bump_gamma
    )
    k_prime_5500 = k_base_5500 * 1.0 + bump_5500  # slope_mod(5500) = 1 for any dust_delta
    k_5500 = k_prime_5500 / rv
    return jnp.clip(k / k_5500, 0.0)


@register_dust_law(
    "tea",
    citation="Haskell et al. 2024 (arXiv:2401.11007)",
    short_doc="TEA empirical attenuation from NIHAO-SKIRT simulations",
)
def tea(
    wavelength: jnp.ndarray,
    dust_delta: float = -0.2,
    dust_tea_scatter: float = 0.0,
) -> jnp.ndarray:
    r"""TEA attenuation curve (Haskell+2024, NIHAO-SKIRT).

    Three-parameter empirical attenuation with physically motivated
    bump-slope correlation from radiative transfer simulations. The
    functional form is identical to Kriek & Conroy (2013), but E_b is
    derived from delta via a tight relation calibrated on NIHAO-SKIRT.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_delta : float
        Power-law slope modification. [dimensionless] Default: -0.2.
        Steeper (more negative) = weaker bump.
    dust_tea_scatter : float
        Scatter in E_b around the median relation. [dex] Default: 0.0 (median).

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(λ), normalized to k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The bump amplitude is derived from the slope via:

    .. math::

        E_b = 2.5 \times \exp(3.5 \times \delta) \times 10^{\text{scatter}}

    then uses Kriek & Conroy (2013) functional form with the derived E_b.

    References
    ----------
    .. [1] A. Haskell, C. L. Steinhardt, C. Conselice, et al., "The Evolution
       of the Dust Attenuation Curve with Redshift from SIMULATIONS,"
       arXiv:2401.11007 (2024).
       https://arxiv.org/abs/2401.11007
    """
    eb = 2.5 * jnp.exp(3.5 * dust_delta) * 10.0**dust_tea_scatter
    return kriek_conroy(wavelength, dust_delta=dust_delta, dust_bump_strength=eb)


# The three tables below are the output of
# ``scripts/fit_narayanan2018_medians.py``, which fits the Kriek & Conroy
# (2013) form to ``data/attenuation/narayanan2018_median_curves.dat`` -- the
# Narayanan et al. (2018) published medians, repackaged with attribution. Rerun
# that script to reproduce every digit; it also writes
# ``data/attenuation/narayanan2018_kc13_fits.json``, which
# ``tests/regression/bug/test_bug_2199_narayanan_z_redshift.py`` reads back to
# check this hand-copy against it.

#: Redshifts at which Narayanan et al. (2018) publish a median attenuation curve.
_NARAYANAN_Z_NODES = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

#: Kriek & Conroy (2013) slope :math:`\delta` fitted to each published median.
#: Full precision, so that this table and the JSON the script writes are the
#: same numbers and a golden pinned against either holds against both.
_NARAYANAN_DELTA = jnp.array(
    [
        -0.5555790140809852,
        -0.2910618326689639,
        -0.4655455976558615,
        -0.38570378510780945,
        -0.32995141823672564,
        0.13785066855951653,
        0.07950863548914745,
    ]
)

#: Multiplier on the KC13 bump amplitude fitted to each published median. The
#: implied :math:`E_b = m\,(0.85 - 1.9\,\delta)` is 6.36, 2.77, 5.08, 5.36,
#: 5.37, 1.98, 1.96 at z = 0 to 6.
_NARAYANAN_BUMP_STRENGTH = jnp.array(
    [
        3.336277843127103,
        1.9767528352381496,
        2.926977165512936,
        3.389079602445048,
        3.6339238083055148,
        3.3729209110204086,
        2.809597222823232,
    ]
)


@register_dust_law(
    "narayanan_z",
    citation="Narayanan et al. 2018 (ApJ 869, 70)",
    short_doc="Narayanan et al. z-dependent attenuation (MUFASA)",
)
def narayanan_z(
    wavelength: jnp.ndarray,
    dust_delta: float = -0.2,
    dust_bump_strength: float = 1.0,
    redshift: float = 0.0,
) -> jnp.ndarray:
    r"""Narayanan+2018 redshift-dependent attenuation.

    Evaluates the Kriek & Conroy (2013) curve [2]_ at slope and bump values
    interpolated in redshift from a table fitted to the median attenuation
    curves Narayanan et al. (2018) [1]_ publish for their 25 Mpc MUFASA
    cosmological radiative-transfer run. The curve gets grayer with redshift,
    which is the trend the paper reports in its Section 5.1.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_delta : float
        Power-law slope modification :math:`\delta`. [dimensionless]
        Default: -0.2, a **sentinel**, not a published value: left at it, the
        slope comes from the fitted table instead. Any other value is used as
        given, at every redshift.
    dust_bump_strength : float
        Multiplier on the KC13 bump amplitude :math:`0.85 - 1.9\,\delta`.
        [dimensionless] Default: 1.0, the same sentinel arrangement as
        ``dust_delta``.
    redshift : float
        Galaxy redshift. [dimensionless] Default: 0.0. Supplied by the model,
        not by the ``dust_attenuation`` group.

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(λ), normalized to k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, including with respect to ``redshift``: the table
    is read with ``jnp.interp``, which is piecewise linear and so
    differentiable away from the seven nodes.

    With both shape parameters left at their sentinels,

    .. math::

        k(\lambda, z) = k_{\rm KC13}\bigl(\lambda;\ \delta(z),\ m(z)\bigr),

    where :math:`\delta(z)` and :math:`m(z)` are linear interpolations of the
    module tables ``_NARAYANAN_DELTA`` and ``_NARAYANAN_BUMP_STRENGTH`` in
    :math:`z` [dimensionless], :math:`k_{\rm KC13}` is :func:`kriek_conroy`,
    and :math:`m` multiplies that curve's bump amplitude
    :math:`E_b = 0.85 - 1.9\,\delta`. Outside :math:`0 \le z \le 6` the table
    is held at its end node rather than extrapolated.

    **Approximation**: the Kriek & Conroy form fitted to published median
    curves, not a formula the paper states. Narayanan et al. (2018) give the
    redshift dependence as median curves (their Figure 9) and publish them at
    the data URL below; ``scripts/fit_narayanan2018_medians.py`` fits three
    parameters per redshift (:math:`\delta`, the bump multiplier and a
    normalization) to the repackaged copy at
    ``data/attenuation/narayanan2018_median_curves.dat``, and every number in
    the tables above comes from that fit. Residual rms over the fit window
    1250 Å to 1 μm is 0.018, 0.024, 0.014, 0.011, 0.012, 0.019 and 0.010 at
    z = 0 to 6, against curves of order unity there, so the form reproduces the
    medians to a few percent and no better. Valid over
    :math:`0 \le z \le 6` and 1250 Å to 1 μm; blueward of 1250 Å the Calzetti
    (2000) baseline the KC13 form tilts is itself an extrapolation, and the
    fit excludes that region.

    The published curves are normalized by the 3000 Å optical depth rather than
    by A_V, so each fit carries a free normalization; that factor is *not*
    applied here, because this function returns k(5500 Å) = 1 like every other
    registered law and the overall depth is the model's ``dust_tau_v``.

    References
    ----------
    .. [1] D. Narayanan, C. Conroy, R. Davé, B. D. Johnson and G. Popping,
       "A Theory for the Variation of Dust Attenuation Laws in Galaxies,"
       ApJ, 869, 70 (2018). arXiv:1805.06905.
       https://doi.org/10.3847/1538-4357/aaed25
       Median curves: https://bitbucket.org/desika/narayanan_attenuation_laws/
    .. [2] M. Kriek and C. Conroy, "The Dust Attenuation Law in Distant
       Galaxies: Evidence for Variation with Spectral Type," ApJL, 775, L16
       (2013). https://doi.org/10.1088/2041-8205/775/1/L16
    """
    # ``jnp.interp`` holds the end node outside the tabulated range, which is
    # the clip to 0 <= z <= 6 the fit range calls for; no separate clip.
    z = jnp.asarray(redshift)
    delta_table = jnp.interp(z, _NARAYANAN_Z_NODES, _NARAYANAN_DELTA)
    bump_table = jnp.interp(z, _NARAYANAN_Z_NODES, _NARAYANAN_BUMP_STRENGTH)
    # Use tolerance comparison (not ==) to avoid JIT-unsafe float equality on traced values.
    delta_z = jnp.where(jnp.abs(dust_delta - (-0.2)) < 1e-6, delta_table, dust_delta)
    bump_z = jnp.where(jnp.abs(dust_bump_strength - 1.0) < 1e-6, bump_table, dust_bump_strength)
    return kriek_conroy(wavelength, dust_delta=delta_z, dust_bump_strength=bump_z)


@register_dust_law(
    "conroy2010",
    citation="Conroy et al. 2010 (ApJ 708, 58)",
    short_doc="Conroy+10 mixed MW + power-law (FSPS default)",
)
def conroy2010(
    wavelength: jnp.ndarray,
    dust_Rv: float = 3.1,
    n_slope: float = -0.7,
) -> jnp.ndarray:
    r"""Conroy+2010 mixed MW + power-law attenuation (FSPS dust_type=1).

    Milky Way (Cardelli 1989) curve dominates at UV wavelengths, power-law dominates
    in the infrared. A smooth sigmoid blend ensures differentiability.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_Rv : float
        Total-to-selective extinction ratio for the MW component. [dimensionless] Default: 3.1.
    n_slope : float
        Power-law index for the long-wavelength component. [dimensionless] Default: -0.7.

    Returns
    -------
    ndarray, shape (n_wave,)
        Attenuation curve k(λ), normalized to k(5500 Å) = 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    A smooth sigmoid transition function weights between MW (short wavelength)
    and power-law (long wavelength) components at the V-band (5500 Å):

    .. math::

        k(\lambda) = [w(\lambda) \, k_{\rm MW}(\lambda) + (1-w(\lambda)) \, k_{\rm PL}(\lambda)] / k_{\rm MW}(5500 \, \text{\AA})

    where :math:`w(\lambda) = \sigma(\log_{10}(\lambda/5500 \, \text{\AA}) / 0.05)` is a sigmoid.

    References
    ----------
    .. [1] C. Conroy, R. H. White, and J. S. Gunn, "Recovering the Intergalactic
       Dust from Galaxies with z < 1," ApJ, 708, 58 (2010).
       https://doi.org/10.1088/0004-637X/708/1/58
    """
    k_mw = cardelli(wavelength, dust_Rv=dust_Rv)
    k_pl = power_law(wavelength, n_slope=n_slope)
    # Smooth sigmoid blend: MW dominates UV, power-law dominates IR
    x = jnp.log10(wavelength / V_BAND_ANGSTROM)
    blend = jax.nn.sigmoid(x / 0.05)
    k_raw = (1.0 - blend) * k_mw + blend * k_pl
    # Normalize to k(V-band) = 1
    lam_v = jnp.array(V_BAND_ANGSTROM)
    x_v = jnp.log10(lam_v / V_BAND_ANGSTROM)
    blend_v = jax.nn.sigmoid(x_v / 0.05)
    k_v = (1.0 - blend_v) * cardelli(lam_v[None], dust_Rv=dust_Rv)[0] + blend_v * 1.0
    return jnp.clip(k_raw / k_v, 0.0)


# ── Two-component dust model ──────────────────────────────────────


from tengri.components.dust._apply import (
    _TWO_COMPONENT_LAW_PARAMS as _TWO_COMPONENT_LAW_PARAMS,
    TWO_COMPONENT_OVERRIDE_KEYS as TWO_COMPONENT_OVERRIDE_KEYS,
    apply_lyman_cutoff as apply_lyman_cutoff,
    precompute_dust_age_mask as precompute_dust_age_mask,
    precompute_dust_age_weights as precompute_dust_age_weights,
    resolve_bc_diff_law_params as resolve_bc_diff_law_params,
    single_component_dust as single_component_dust,
    single_component_dust_fast as single_component_dust_fast,
    two_component_dust as two_component_dust,
    two_component_dust_fast as two_component_dust_fast,
    two_component_dust_separable as two_component_dust_separable,
)


def wg00_shell(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "cardelli",
    **law_params,
) -> jnp.ndarray:
    r"""Witt & Gordon (2000) SHELL geometry: foreground screen.

    The simplest geometry: a uniform dust slab in front of all stars.
    Transmission is the standard Beer-Lambert law.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    tau_v : float
        V-band optical depth (at 5500 Å). [dimensionless]
    law : str
        Underlying extinction curve name. Default: "cardelli" (MW).
    **law_params
        Passed to the extinction curve function (e.g., ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission T(λ) in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    The transmission is:

    .. math::

        T(\lambda) = \exp[-\tau_V k(\lambda)]

    This is identical to ``single_component_dust`` with ``f_obscuration=0``
    and is included for completeness alongside the CLOUDY and DUSTY models.

    References
    ----------
    .. [1] A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media.
       I. Point Source Embedded in a Clumpy Medium," ApJ, 528, 799 (2000).
       https://doi.org/10.1086/308197
    """
    reject_unread_law_kwargs(law_params, (law,), "wg00_shell")
    k = resolve_dust_law(law)(wavelength, **select_law_kwargs(law, law_params))
    return jnp.exp(-tau_v * k)


def wg00_cloudy(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "cardelli",
    **law_params,
) -> jnp.ndarray:
    r"""Witt & Gordon (2000) CLOUDY dust geometry: homogeneous dust-star mix.

    Stars and dust are uniformly mixed throughout a slab of total V-band optical depth.
    The analytic solution integrates radiative transfer, producing a wavelength-dependent
    transmission that is grayer (less wavelength-dependent) than a foreground screen.
    Realistic for galaxies with well-mixed ISM (e.g., starburst regions).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    tau_v : float
        Total V-band optical depth through the slab. [dimensionless]
    law : str, optional
        Underlying attenuation curve name, from ``DUST_LAWS`` registry. Default: "cardelli" (MW).
    **law_params
        Keyword arguments passed to the attenuation curve function (e.g., ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission T(λ) ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives, with numerically stable
    Taylor expansion for small optical depth.

    **Gradient-safe**: yes, differentiable everywhere via smooth blending between exact
    and Taylor regimes.

    The transmission for a homogeneous slab is:

    .. math::

        T(\lambda) = \frac{1 - \exp[-\tau_V \, k(\lambda)]}{\tau_V \, k(\lambda)}

    where :math:`k(\lambda)` is the normalized attenuation curve. This follows the solution
    of radiative transfer in a uniform dust-star slab (Natta & Panagia 1984, Section 3.1;
    Calzetti et al. 1994).

    **Limiting behavior**: At low optical depth (:math:`\tau_V k \ll 1`), :math:`T \to 1`
    (transparent). At high optical depth, :math:`T \approx 1/(\tau_V k)`, producing
    a grayer (less wavelength-dependent) effective attenuation than the foreground screen
    because stars near the observer-facing side suffer less extinction.

    **Numerical stability**: Uses a Taylor expansion (correct to order :math:`\tau^3`)
    for :math:`\tau_V k < 10^{-4}` to avoid division-by-zero, preserving gradients throughout.

    **Approximation**: The analytic solution assumes pure absorption (zero scattering).
    Witt & Gordon (2000) Monte Carlo simulations including scattering find the effective
    attenuation is slightly grayer still. The analytic form captures the dominant geometric
    effect and is widely used in SED fitting codes (e.g., Synthesizer, CIGALE).

    References
    ----------
    .. [1] A. Natta and N. Panagia, "Extinction in inhomogeneous clouds," ApJ, 287, 228 (1984).
       https://doi.org/10.1086/162686

    .. [2] D. Calzetti, A. L. Kinney, and T. Storchi-Bergmann, "Dust Extinction of the Stellar
       Continua in Starburst Galaxies: The Ultraviolet and Optical Extinction Law,"
       ApJ, 429, 582 (1994). https://doi.org/10.1086/174348

    .. [3] A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media. II. Galactic
       Environments," ApJ, 528, 799 (2000). Section 3.2, "homogeneous" model.
       https://doi.org/10.1086/308197
    """
    reject_unread_law_kwargs(law_params, (law,), "wg00_cloudy")
    k = resolve_dust_law(law)(wavelength, **select_law_kwargs(law, law_params))
    tau_k = tau_v * k

    # Numerically stable: for small tau_k, use Taylor expansion
    # (1 - exp(-x)) / x -> 1 - x/2 + x^2/6 - ... for x -> 0
    # Switch at |x| < 1e-4 to avoid loss of precision
    # Use jnp.maximum (not jnp.where) so the gradient of ratio w.r.t. tau_k stays
    # connected when tau_k is small: jnp.where with a constant fallback gives zero
    # gradient in the masked branch, producing dead gradients near tau_k=0.
    ratio = (1.0 - jnp.exp(-jnp.maximum(tau_k, 1e-10))) / jnp.maximum(tau_k, 1e-10)
    # Taylor expansion for small tau_k: correct gradient throughout [0, 1e-4]
    taylor = 1.0 - tau_k / 2.0 + tau_k**2 / 6.0
    return jnp.where(tau_k > 1e-4, ratio, taylor)


def wg00_dusty(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "cardelli",
    n_clumps: float = 10.0,
    **law_params,
) -> jnp.ndarray:
    r"""Witt & Gordon (2000) DUSTY geometry: clumpy two-phase medium.

    The ISM is modeled as ``n_clumps`` identical clumps, each with
    optical depth ``tau_clump = tau_V / n_clumps``, distributed along
    random sightlines (Natta & Panagia 1984; Hobson & Padman 1993).
    The probability of a photon traversing N clumps follows a Poisson
    distribution, giving the mean transmission.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    tau_v : float
        Total V-band optical depth (= ``n_clumps × tau_clump``). [dimensionless]
    law : str
        Underlying extinction curve name. Default: "cardelli" (MW).
    n_clumps : float
        Mean number of clumps along a sightline. [dimensionless] Default: 10.0.
        Higher values approach the homogeneous limit. Typical range: 1–40.
    **law_params
        Passed to the extinction curve function (e.g., ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission T(λ) in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    The transmission is:

    .. math::

        T(\lambda) = \exp[-n_{\rm clumps} (1 - \exp[-\tau_{\rm clump} \, k(\lambda)])]

    where :math:`\tau_{\rm clump} = \tau_V / n_{\rm clumps}`.

    This produces the *grayest* (least wavelength-dependent) effective attenuation
    of the three WG00 geometries because photons preferentially escape through
    low-column channels between clumps.

    **Limiting behavior:**

    - :math:`n_{\rm clumps} \to \infty` (fixed :math:`\tau_V`): recovers the homogeneous slab.
    - :math:`n_{\rm clumps} = 1`: single clump with Poisson averaging.
    - :math:`\tau_V = 0`: T = 1 (transparent), regardless of n_clumps.

    References
    ----------
    .. [1] A. Natta and N. Panagia, "Extinction in Inhomogeneous Clouds,"
       ApJ, 287, 228 (1984). https://doi.org/10.1086/162686

    .. [2] M. P. Hobson and L. Padman, "A Probabilistic Approach to Extinction
       in Irregular Media," MNRAS, 264, 161 (1993).
       https://doi.org/10.1093/mnras/264.1.161

    .. [3] A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media.
       I. Point Source Embedded in a Clumpy Medium," ApJ, 528, 799 (2000).
       https://doi.org/10.1086/308197
    """
    reject_unread_law_kwargs(law_params, (law,), "wg00_dusty")
    k = resolve_dust_law(law)(wavelength, **select_law_kwargs(law, law_params))
    tau_clump = tau_v / jnp.maximum(n_clumps, 1e-10)
    return jnp.exp(-n_clumps * (1.0 - jnp.exp(-tau_clump * k)))


# ── Dust-to-gas ratio scaling ─────────────────────────────────────


def dust_to_gas_scaling_remy_ruyer(logzsol: float) -> float:
    r"""Metallicity-dependent dust-to-gas ratio scaling (Rémy-Ruyer+2014).

    Returns multiplicative factor for dust optical depths relative to solar.
    Broken power law: linear above 0.1 Z_sun, quadratic below.

    Parameters
    ----------
    logzsol : float
        log10(Z / Z_sun). [dimensionless]

    Returns
    -------
    float
        D/G ratio relative to solar (1.0 at solar metallicity). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The scaling is a broken power law:

    .. math::

        \text{D/G} = \begin{cases}
        (Z/Z_{\odot})^2 \times 0.1 & (Z/Z_{\odot}) \leq 0.1 \\
        (Z/Z_{\odot}) & (Z/Z_{\odot}) > 0.1
        \end{cases}

    References
    ----------
    .. [1] S. Rémy-Ruyer, I. Miville-Deschênes, T. Siebel, et al.,
       "Dust and Gas Relationship in Nearby Galaxies,"
       A&A, 563, A31 (2014).
       https://doi.org/10.1051/0004-6361/201322803
    """
    z_ratio = 10.0**logzsol
    scaling = jnp.where(
        z_ratio > 0.1,
        z_ratio,
        0.1 * (z_ratio / 0.1) ** 2.0,
    )
    return scaling


# ── Grain model dust attenuation laws (WD01, D03, HD23) ──────────
# Precomputed at module import from dust_extinction (astropy-affiliated).
# Stored as numpy constants; JAX functions use jnp.interp for JIT compatibility.


def _precompute_grain_curve(model_cls: type, submodel: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a grain model and return wavelength [Å] + k(λ) arrays.

    Parameters
    ----------
    model_cls : type
        Class from ``dust_extinction.grain_models``.
    submodel : str
        Submodel key (e.g. ``"SMCBar"``, ``"MWRV31"``).

    Returns
    -------
    wave_aa : ndarray, shape (n,)
        Wavelength array [Å], sorted ascending.
    k_norm : ndarray, shape (n,)
        Normalized attenuation curve k(λ), k(5500 Å) = 1 [dimensionless].

    Notes
    -----
    **JIT-compatible**: no, uses astropy units at call time.
    Call at module level (import time), not inside JIT.
    """
    from dust_extinction.grain_models import WD01  # noqa: F401 (ensures package present)

    m = model_cls(submodel)
    data_x = np.asarray(m.data_x, dtype=np.float64)
    data_axav = np.asarray(m.data_axav, dtype=np.float64)
    # data_x is in 1/μm (wavenumber); valid where > 0
    mask = data_x > 0
    wave_aa = 1e4 / data_x[mask]
    k = data_axav[mask]
    order = np.argsort(wave_aa)
    wave_aa, k = wave_aa[order], k[order]
    # Normalize to k(V-band) = 1 for consistency with other tengri dust laws
    k_at_5500 = float(np.interp(V_BAND_ANGSTROM, wave_aa, k))
    return wave_aa, k / k_at_5500


def _make_grain_law(wave_aa: np.ndarray, k_norm: np.ndarray):
    """Return a JIT-compatible dust law that interpolates a precomputed curve.

    Parameters
    ----------
    wave_aa : ndarray, shape (n,)
        Precomputed wavelength grid [Å].
    k_norm : ndarray, shape (n,)
        Precomputed k(λ) values, k(5500 Å) = 1 [dimensionless].

    Returns
    -------
    callable
        Function ``(wavelength) -> k(λ)`` compatible with
        ``@register_dust_law``.
    """
    _wave = jnp.asarray(wave_aa)
    _k = jnp.asarray(k_norm)

    def _law(wavelength: jnp.ndarray) -> jnp.ndarray:
        """Interpolate precomputed dust attenuation curve at given wavelengths."""
        return jnp.maximum(jnp.interp(wavelength, _wave, _k), 0.0)

    return _law


# Precompute at module import (numpy, once).
try:
    from dust_extinction.grain_models import D03, HD23, WD01

    _WD01_SMCBAR = _precompute_grain_curve(WD01, "SMCBar")
    _WD01_MWRV31 = _precompute_grain_curve(WD01, "MWRV31")
    _D03_MWRV31 = _precompute_grain_curve(D03, "MWRV31")
    _HD23_MWRV31 = _precompute_grain_curve(HD23, "MWRV31")
    _GRAIN_MODELS_AVAILABLE = True
except ImportError:
    _GRAIN_MODELS_AVAILABLE = False


def _grain_law_unavailable(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Fallback when dust-extinction package is not installed."""
    raise ImportError(
        "dust-extinction package required for grain model dust laws. "
        "Install with: pip install dust-extinction"
    )


@register_dust_law(
    "wd01_smcbar",
    citation="Weingartner & Draine 2001 (ApJ 548, 296)",
    short_doc="WD01 SMC Bar grain model attenuation",
)
def wd01_smcbar(wavelength: jnp.ndarray) -> jnp.ndarray:
    r"""Weingartner & Draine (2001) SMC Bar grain model attenuation curve.

    Physically motivated dust grain-size + composition distribution for SMC-like
    dust: steep UV rise, no 2175 Å bump. Most relevant for high-redshift galaxies.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid [Å].

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), k(5500 Å) = 1 [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.interp`` on precomputed curve.

    Precomputed from ``dust_extinction.grain_models.WD01("SMCBar")`` at module
    import time. Data cover ~100–10\ :sup:`7` Å. Values outside the tabulated
    range are held constant at boundary values.

    Upstream credit: grain model data from Weingartner & Draine (2001) [1]_,
    accessed via the ``dust-extinction`` astropy-affiliated package.
    Independent implementation in synthesizer (Wilkins et al. 2025 [2]_).

    References
    ----------
    .. [1] J. C. Weingartner & B. T. Draine, "Dust Grain-Size Distributions
       and Extinction in the Milky Way, LMC, and SMC," ApJ, 548, 296 (2001).
       arXiv:astro-ph/0008146. https://doi.org/10.1086/318651
    .. [2] C. C. Lovell et al. 2025, Open J. Astrophys. 8,
       "Synthesizer: a Software Package for Synthetic Astronomical Observables,"
       doi:10.33232/001c.145766; W. J. Roper et al. 2026, JOSS 11, 9436,
       doi:10.21105/joss.09436 (cite both Synthesizer papers).
    """
    if not _GRAIN_MODELS_AVAILABLE:
        return _grain_law_unavailable(wavelength)
    return _make_grain_law(*_WD01_SMCBAR)(wavelength)


@register_dust_law(
    "wd01_mwrv31",
    citation="Weingartner & Draine 2001 (ApJ 548, 296)",
    short_doc="WD01 MW R_V=3.1 grain model attenuation",
)
def wd01_mwrv31(wavelength: jnp.ndarray) -> jnp.ndarray:
    r"""Weingartner & Draine (2001) MW R_V=3.1 grain model attenuation curve.

    Physically motivated dust grain-size + composition distribution for Milky
    Way diffuse ISM dust: prominent 2175 Å bump, R_V = 3.1.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid [Å].

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), k(5500 Å) = 1 [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.interp`` on precomputed curve.

    Upstream credit: grain model data from Weingartner & Draine (2001) [1]_,
    accessed via the ``dust-extinction`` astropy-affiliated package.
    Independent implementation in synthesizer (Wilkins et al. 2025 [2]_).

    References
    ----------
    .. [1] J. C. Weingartner & B. T. Draine, "Dust Grain-Size Distributions
       and Extinction in the Milky Way, LMC, and SMC," ApJ, 548, 296 (2001).
       arXiv:astro-ph/0008146. https://doi.org/10.1086/318651
    .. [2] C. C. Lovell et al. 2025, Open J. Astrophys. 8,
       "Synthesizer: a Software Package for Synthetic Astronomical Observables,"
       doi:10.33232/001c.145766; W. J. Roper et al. 2026, JOSS 11, 9436,
       doi:10.21105/joss.09436 (cite both Synthesizer papers).
    """
    if not _GRAIN_MODELS_AVAILABLE:
        return _grain_law_unavailable(wavelength)
    return _make_grain_law(*_WD01_MWRV31)(wavelength)


@register_dust_law(
    "d03_mwrv31",
    citation="Draine 2003 (ARA&A 41, 241)",
    short_doc="Draine 2003 MW R_V=3.1 grain model attenuation",
)
def d03_mwrv31(wavelength: jnp.ndarray) -> jnp.ndarray:
    r"""Draine (2003) MW R_V=3.1 updated grain model attenuation curve.

    Updated Milky Way grain model incorporating revised PAH properties and
    emission efficiencies relative to WD01. R_V = 3.1.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid [Å].

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), k(5500 Å) = 1 [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.interp`` on precomputed curve.

    Upstream credit: grain model data from Draine (2003) [1]_,
    accessed via the ``dust-extinction`` astropy-affiliated package.
    Independent implementation in synthesizer (Wilkins et al. 2025 [2]_).

    References
    ----------
    .. [1] B. T. Draine, "Interstellar Dust Grains," ARA&A, 41, 241 (2003).
       arXiv:astro-ph/0304489. https://doi.org/10.1146/annurev.astro.41.011802.094840
    .. [2] C. C. Lovell et al. 2025, Open J. Astrophys. 8,
       "Synthesizer: a Software Package for Synthetic Astronomical Observables,"
       doi:10.33232/001c.145766; W. J. Roper et al. 2026, JOSS 11, 9436,
       doi:10.21105/joss.09436 (cite both Synthesizer papers).
    """
    if not _GRAIN_MODELS_AVAILABLE:
        return _grain_law_unavailable(wavelength)
    return _make_grain_law(*_D03_MWRV31)(wavelength)


@register_dust_law(
    "hd23_mwrv31",
    citation="Hensley & Draine 2023 (ApJ 948, 55)",
    short_doc="Hensley & Draine 2023 astrodust+PAH grain model",
)
def hd23_mwrv31(wavelength: jnp.ndarray) -> jnp.ndarray:
    r"""Hensley & Draine (2023) astrodust+PAH MW R_V=3.1 grain model.

    State-of-the-art MW grain model combining astrodust grains with PAH
    emission. R_V = 3.1. Supersedes WD01/D03 for the diffuse MW ISM.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid [Å].

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized attenuation curve k(λ), k(5500 Å) = 1 [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.interp`` on precomputed curve.

    Upstream credit: grain model data from Hensley & Draine (2023) [1]_,
    accessed via the ``dust-extinction`` astropy-affiliated package.
    Independent implementation in synthesizer (Wilkins et al. 2025 [2]_).

    References
    ----------
    .. [1] B. S. Hensley & B. T. Draine, "The Astrodust+PAH Model: A Unified
       Description of the Dust of the Diffuse Milky Way," ApJ, 948, 55 (2023).
       arXiv:2208.12365. https://doi.org/10.3847/1538-4357/acc4c2
    .. [2] C. C. Lovell et al. 2025, Open J. Astrophys. 8,
       "Synthesizer: a Software Package for Synthetic Astronomical Observables,"
       doi:10.33232/001c.145766; W. J. Roper et al. 2026, JOSS 11, 9436,
       doi:10.21105/joss.09436 (cite both Synthesizer papers).
    """
    if not _GRAIN_MODELS_AVAILABLE:
        return _grain_law_unavailable(wavelength)
    return _make_grain_law(*_HD23_MWRV31)(wavelength)
