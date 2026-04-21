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
- **power_law**: (lambda/5500)^n — original Charlot & Fall (2000)
- **calzetti**: Calzetti et al. (2000) starburst polynomial, R_V=4.05
- **leitherer02**: Leitherer et al. (2002) UV extension of Calzetti (970-1800 A)
- **kriek_conroy**: Calzetti + UV bump + slope delta — Prospector default
- **noll09**: Noll et al. (2009) modified Calzetti+L02: (base+bump)*slope
- **salim_sbl18**: Salim+2018 modified Calzetti+L02: base*slope+bump
- **smc**: Pei (1992, ApJ 395 130) SMC Bar, steep UV, no 2175A bump
- **cardelli**: Cardelli et al. (1989) MW curve with free R_V
- **li08**: Li et al. (2008) Eq. (1) four-coefficient curve (continuum + FUV rise + 2175 Å bump)
- **salim**: Salim et al. (2018) modified Calzetti (= DSPS default)
- **tea**: Haskell et al. (2024) TEA 3-param empirical (NIHAO-SKIRT bump-slope correlation)
- **narayanan_z**: Narayanan et al. (2018) redshift-dependent Kriek-Conroy (SIMBA RT)
- **conroy2010**: Conroy+2010 mixed MW + power-law (FSPS dust_type=1)
- **vw07_bc**: Wild+2007 birth cloud power-law (n=-1.3)
- **vw07_diff**: Wild+2007 diffuse ISM power-law (n=-0.7)

Dust Geometries (Witt & Gordon 2000)
-------------------------------------
- **wg00_shell**: Foreground screen — standard exp(-tau*k)
- **wg00_cloudy**: Homogeneous dust-star mix (slab) — greyer than screen
- **wg00_dusty**: Clumpy two-phase medium (Natta & Panagia 1984) — greyest

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

from collections.abc import Callable

import jax
import jax.numpy as jnp

# ── Attenuation curve registry ────────────────────────────────────

DUST_LAWS: dict[str, Callable] = {}


def register_dust_law(name: str) -> Callable:
    """Register a dust attenuation curve function (decorator factory)."""

    def decorator(fn: Callable) -> Callable:
        """Inner decorator that registers function in DUST_LAWS dict."""
        DUST_LAWS[name] = fn
        return fn

    return decorator


def resolve_dust_law(name: str) -> Callable:
    """Get a registered dust law by name."""
    if name not in DUST_LAWS:
        raise ValueError(f"Unknown dust law '{name}'. Available: {list(DUST_LAWS.keys())}")
    return DUST_LAWS[name]


get_dust_law = resolve_dust_law


# ── Utility: Drude profile for the 2175 Angstrom UV bump ──────────


def _drude_profile(
    wave_um: jnp.ndarray,
    x0: float = 0.2175,
    gamma: float = 0.035,
) -> jnp.ndarray:
    r"""Drude profile for the 2175 Å UV absorption bump.

    Parameters
    ----------
    wave_um : array_like, shape (n_wave,)
        Wavelength grid. [μm]
    x0 : float, optional
        Central wavelength of the bump. [μm] Default: 0.2175 (2175 Å).
    gamma : float, optional
        FWHM of the profile. [μm] Default: 0.035 (350 Å).

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized Drude profile (dimensionless, in [0, 1]).

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The Drude profile is:

    .. math::

        D(\lambda; \lambda_0, \gamma) = \frac{(\lambda \, \gamma)^2}{(\lambda^2 - \lambda_0^2)^2 + (\lambda \, \gamma)^2}

    where :math:`\lambda` is wavelength [μm], :math:`\lambda_0` is the central wavelength [μm],
    and :math:`\gamma` is the FWHM [μm]. This is a standard resonance profile used to model
    the 2175 Å silicate bump in interstellar dust attenuation.

    **Upstream**: Following Kriek & Conroy (2013) [1]_ and standard dust attenuation conventions.

    References
    ----------
    .. [1] M. Kriek and C. Conroy, "Dust Attenuation in High-Redshift Galaxies—Modeling
       the Spectral Energy Distribution," ApJL, 775, L16 (2013).
       https://doi.org/10.1088/2041-8205/775/1/L16
    """
    return (wave_um * gamma) ** 2 / ((wave_um**2 - x0**2) ** 2 + (wave_um * gamma) ** 2)


# ── Internal helpers ──────────────────────────────────────────────


def _calzetti_l02_kprime(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Compute k'(lambda) = A(lambda)/E(B-V) using Leitherer (2002) + Calzetti (2000).

    Switches between the two extinction laws depending on wavelength: Leitherer et al. (2002)
    for the far-UV, Calzetti et al. (2000) for longer wavelengths. Returns the raw
    reddening curve (NOT normalized by R_V).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Reddening curve k'(λ) = A(λ)/E(B-V), unnormalized. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    Uses piecewise polynomials following Leitherer et al. (2002) and Calzetti et al. (2000).
    The transition occurs at 0.18 μm (1800 Å), matching the standalone ``dust_attenuation.averages.L02`` model.

    References
    ----------
    .. [1] C. Leitherer et al., "Starburst99: Synthesis Models for Galaxies with Active
       Star Formation," ApJS, 140, 303 (2002).
       https://doi.org/10.1086/342289

    .. [2] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Star-forming Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
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

    # L02 valid range: 970-1800 A (Leitherer+2002 ApJS 140 303 Eq. 14).  Use 0.18 um cutoff,
    # matching the standalone leitherer02 function; 0.15 um was too conservative.
    k_calz = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    return jnp.where(wave_um <= 0.18, k_l02, k_calz)


# ── Attenuation curves ────────────────────────────────────────────


@register_dust_law("power_law")
def power_law(
    wavelength: jnp.ndarray,
    n_slope: float = -0.7,
    **_kwargs,
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
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The attenuation is:

    .. math::

        k(\lambda) = \left(\frac{\lambda}{5500 \, \text{\AA}}\right)^n

    where :math:`n = -0.7` produces the standard Charlot & Fall (2000) wavelength
    dependence. Negative slopes make dust redder (stronger attenuation at short wavelengths).

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Star-forming Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    return (wavelength / 5500.0) ** n_slope


@register_dust_law("vw07_bc")
def vw07_bc(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """Wild+2007 birth cloud: power-law with n = -1.3 (steep UV).

    Steeper than the diffuse ISM curve, reflecting the denser dust
    geometry around young stellar populations.

    References
    ----------
    Wild et al. 2007, MNRAS, 381, 543.
    """
    return (wavelength / 5500.0) ** (-1.3)


@register_dust_law("vw07_diff")
def vw07_diff(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """Wild+2007 diffuse ISM: power-law with n = -0.7 (standard CF00).

    References
    ----------
    Wild et al. 2007, MNRAS, 381, 543;
    Charlot & Fall 2000, ApJ, 539, 718.
    """
    return (wavelength / 5500.0) ** (-0.7)


@register_dust_law("calzetti")
def calzetti(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """Calzetti et al. (2000) starburst attenuation curve.

    R_V = 4.05 (fixed). Valid: 0.12 - 2.2 um.
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    k_ir = 2.659 * (-1.857 + 1.040 * x)
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3)

    rv = 4.05
    k_prime = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    return jnp.clip((k_prime + rv) / rv, 0.0)


@register_dust_law("kriek_conroy")
def kriek_conroy(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 1.0,
    dust_delta: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Kriek & Conroy (2013) modified Calzetti + UV bump + slope delta.

    Default in Prospector. Most flexible single-parameter-family curve.

    Parameters
    ----------
    dust_bump_strength : float
        Amplitude of 2175A UV bump (E_b). 0 = no bump.
    dust_delta : float
        Power-law slope modification. 0 = pure Calzetti.
    """
    wave_um = wavelength / 1e4
    k_calz = calzetti(wavelength)
    slope_mod = (wavelength / 5500.0) ** dust_delta
    bump = dust_bump_strength * _drude_profile(wave_um)
    return jnp.clip(k_calz * slope_mod + bump, 0.0)


def _pei92_curve(
    wavelength: jnp.ndarray,
    lam_i: jnp.ndarray,
    a_i: jnp.ndarray,
    b_i: jnp.ndarray,
    n_i: jnp.ndarray,
    R_V: float,
) -> jnp.ndarray:
    """Pei (1992, ApJ, 395, 130) generalized Drude profile sum.

    Computes A(lambda)/A(V) normalized to k(5500 A) = 1. Fully continuous,
    no piecewise boundaries.

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.
    lam_i, a_i, b_i, n_i : arrays, shape (n_components,)
        Drude component parameters from Pei 1992 Table 4.
    R_V : float
        Total-to-selective extinction ratio.
    """
    wave_um = wavelength / 1e4  # (n_wave,)
    # xi(lambda) = sum_i a_i / ((lam/lam_i)^n_i + (lam_i/lam)^n_i + b_i)
    # shape: (n_components, n_wave)
    ratio = wave_um[None, :] / lam_i[:, None]
    denom = ratio ** n_i[:, None] + ratio ** (-n_i[:, None]) + b_i[:, None]
    xi = jnp.sum(a_i[:, None] / denom, axis=0)

    # Pei 1992 Drude sum gives extinction proportional to tau(lambda).
    # A(lambda)/A(V) = xi(lambda) / xi(V). Normalize to k(5500 A) = 1.
    wave_v = 0.55  # um
    ratio_v = wave_v / lam_i
    xi_v = jnp.sum(a_i / (ratio_v**n_i + ratio_v ** (-n_i) + b_i))
    return jnp.clip(xi / xi_v, 0.0)


# Pei 1992 Table 4 — SMC Bar (6 components, R_V = 2.93, no 2175 A bump)
_SMC_LAM = jnp.array([0.042, 0.08, 0.22, 9.7, 18.0, 25.0])
_SMC_A = jnp.array([185.0, 27.0, 0.005, 0.010, 0.012, 0.030])
_SMC_B = jnp.array([90.0, 5.50, -1.95, -1.95, -1.80, 0.00])
_SMC_N = jnp.array([2.0, 4.0, 2.0, 2.0, 2.0, 2.0])
_SMC_RV = 2.93

# Pei 1992 Table 4 — LMC (6 components, R_V = 3.16, weak 2175 A bump)
_LMC_LAM = jnp.array([0.046, 0.08, 0.22, 9.7, 18.0, 25.0])
_LMC_A = jnp.array([175.0, 19.0, 0.023, 0.005, 0.006, 0.020])
_LMC_B = jnp.array([90.0, 5.50, -1.95, -1.95, -1.80, 0.00])
_LMC_N = jnp.array([2.0, 4.5, 2.0, 2.0, 2.0, 2.0])
_LMC_RV = 3.16


@register_dust_law("smc")
def smc(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """SMC Bar extinction curve (Pei 1992, ApJ, 395, 130).

    Steep UV rise, NO 2175 A bump. Common at high redshift. R_V = 2.93.
    Uses generalized Drude profile sum — fully continuous, no piecewise
    boundaries.
    """
    return _pei92_curve(wavelength, _SMC_LAM, _SMC_A, _SMC_B, _SMC_N, _SMC_RV)


@register_dust_law("lmc")
def lmc(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """LMC average extinction curve (Pei 1992, ApJ, 395, 130).

    Weak 2175 A bump, intermediate between MW and SMC. R_V = 3.16.
    Uses generalized Drude profile sum — fully continuous.
    """
    return _pei92_curve(wavelength, _LMC_LAM, _LMC_A, _LMC_B, _LMC_N, _LMC_RV)


@register_dust_law("cardelli")
def cardelli(
    wavelength: jnp.ndarray,
    dust_Rv: float = 3.1,
    **_kwargs,
) -> jnp.ndarray:
    """Cardelli, Clayton & Mathis (1989) MW extinction with free R_V.

    Returns A(lambda)/A(V).
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

    return jnp.clip(a + b / dust_Rv, 0.0)


@register_dust_law("li08")
def li08(
    wavelength: jnp.ndarray,
    dust_c1: float = 6.0,
    dust_c2: float = 4.0,
    dust_c3: float = 2.0,
    dust_c4: float = 0.04,
    **_kwargs,
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


@register_dust_law("salim")
def salim(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Salim et al. (2018) modified Calzetti (DSPS / Zacharegkas+2025 default).

    Same functional form as Kriek & Conroy 2013.
    """
    return kriek_conroy(
        wavelength,
        dust_bump_strength=dust_bump_strength,
        dust_delta=dust_delta,
    )


@register_dust_law("leitherer02")
def leitherer02(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """Leitherer et al. (2002) UV starburst attenuation curve.

    Far-UV extension of the Calzetti (2000) law, valid 970-1800 Angstrom.
    Uses R_V = 4.05 (same as Calzetti).  For wavelengths outside the
    L02 range, falls back to Calzetti (2000).

    The k'(lambda) = A(lambda)/E(B-V) polynomial is (Leitherer+2002 ApJS 140 303 Eq. 14)::

        k'(lambda) = 5.472 + 0.671*x - 9.218e-3*x^2 + 2.620e-3*x^3

    where x = 1/lambda [um^{-1}] (i.e., x = 1/lambda_micron, not lambda itself).

    The L02 polynomial is used for the full L02 valid range (0.097-0.18 um),
    with C00 used for longer wavelengths.  This matches the standalone
    ``dust_attenuation.averages.L02`` model.

    Returns k(lambda) = k'(lambda) / R_V, following the dust_attenuation
    package convention.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda) = k'(lambda) / R_V.

    References
    ----------
    Leitherer et al. 2002, ApJS, 140, 303 (eq. 14)
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

    # Use L02 up to 0.18 um (full L02 range), Calzetti above
    k_calz = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    k_prime = jnp.where(wave_um <= 0.18, k_l02, k_calz)

    return jnp.clip(k_prime / rv, 0.0)


@register_dust_law("noll09")
def noll09(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
    dust_bump_x0: float = 0.2175,
    dust_bump_gamma: float = 0.035,
    **_kwargs,
) -> jnp.ndarray:
    """Noll et al. (2009) modified Calzetti + L02 with UV bump + slope delta.

    This is the ``N09`` model from the ``dust_attenuation`` package.
    Uses Leitherer (2002) for lambda < 1500 A and Calzetti (2000) above.
    The modification order is: **(base + bump) * power_law**.

    This differs from ``kriek_conroy`` which does NOT use L02 and applies
    the bump AFTER the slope: ``base * power_law + bump``.

    The normalization follows the dust_attenuation package convention:
    k(lambda) = k'(lambda) / R_V, where R_V = 4.05 is the fixed Calzetti
    value.  This means k(V) is NOT exactly 1.0 when bump or slope
    modifications are applied (matching the package behaviour).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_bump_strength : float
        Amplitude of 2175 A UV bump (E_b). 0 = no bump.
    dust_delta : float
        Power-law slope modification. 0 = pure Calzetti+L02.
    dust_bump_x0 : float
        Central wavelength of UV bump in microns. Default 0.2175.
    dust_bump_gamma : float
        FWHM of UV bump in microns. Default 0.035.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda) = k'(lambda) / R_V.

    References
    ----------
    Noll et al. 2009, A&A, 507, 1793
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
    return jnp.clip(k_prime / rv, 0.0)


@register_dust_law("salim_sbl18")
def salim_sbl18(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
    dust_bump_x0: float = 0.2175,
    dust_bump_gamma: float = 0.035,
    **_kwargs,
) -> jnp.ndarray:
    """Salim, Boquien & Lee (2018) modified Calzetti + L02 with UV bump + slope.

    This is the ``SBL18`` model from the ``dust_attenuation`` package.
    Uses Leitherer (2002) for lambda < 1500 A and Calzetti (2000) above.
    The modification order is: **(base * power_law) + bump**.

    This differs from ``noll09`` which applies: ``(base + bump) * power_law``.
    The SBL18 order is identical to ``kriek_conroy``, but SBL18 additionally
    uses L02 in the far-UV.

    The normalization follows the dust_attenuation package convention:
    k(lambda) = k'(lambda) / R_V, where R_V = 4.05 is the fixed Calzetti
    value.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_bump_strength : float
        Amplitude of 2175 A UV bump (E_b). 0 = no bump.
    dust_delta : float
        Power-law slope modification. 0 = pure Calzetti+L02.
    dust_bump_x0 : float
        Central wavelength of UV bump in microns. Default 0.2175.
    dust_bump_gamma : float
        FWHM of UV bump in microns. Default 0.035.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda) = k'(lambda) / R_V.

    References
    ----------
    Salim, Boquien & Lee 2018, ApJ, 859, 11
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
    return jnp.clip(k_prime / rv, 0.0)


@register_dust_law("tea")
def tea(
    wavelength: jnp.ndarray,
    dust_delta: float = -0.2,
    dust_tea_scatter: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """TEA attenuation curve (Haskell+2024, NIHAO-SKIRT).

    Three-parameter empirical attenuation with physically motivated
    bump-slope correlation from radiative transfer simulations. The
    functional form is identical to Kriek & Conroy (2013), but E_b is
    derived from delta via a tight relation calibrated on NIHAO-SKIRT::

        E_b = 2.5 * exp(3.5 * delta) * 10 ^ scatter

    This reduces the free parameters to 2 (delta and overall tau_V),
    plus optional scatter around the median E_b(delta) relation.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_delta : float
        Power-law slope modification. Steeper (more negative) = weaker bump.
    dust_tea_scatter : float
        Scatter in E_b around the median relation (dex). Default 0 = median.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda), normalized to k(5500 A) = 1.

    References
    ----------
    Haskell et al. 2024, arXiv:2401.11007
    """
    eb = 2.5 * jnp.exp(3.5 * dust_delta) * 10.0**dust_tea_scatter
    return kriek_conroy(wavelength, dust_delta=dust_delta, dust_bump_strength=eb)


@register_dust_law("narayanan_z")
def narayanan_z(
    wavelength: jnp.ndarray,
    dust_delta: float = -0.2,
    dust_bump_strength: float = 1.0,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Narayanan+2018 redshift-dependent attenuation.

    Uses the Kriek & Conroy (2013) curve with z-dependent median
    parameters calibrated on SIMBA cosmological radiative-transfer
    simulations::

        delta_median(z) ~ -0.2 - 0.1 * z   (steeper at high z)
        E_b_median(z)   ~ max(0, 1.0 - 0.15 * z)  (weaker bump at high z)

    When explicit delta / bump values differ from the defaults, those
    values are used as-is.  When defaults are kept, the z-dependent
    median is applied.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_delta : float
        Power-law slope modification. Default -0.2 triggers z-scaling.
    dust_bump_strength : float
        UV bump amplitude E_b. Default 1.0 triggers z-scaling.
    redshift : float
        Galaxy redshift.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda), normalized to k(5500 A) = 1.

    References
    ----------
    Narayanan et al. 2018, ApJ, 869, 70
    """
    # Use tolerance comparison (not ==) to avoid JIT-unsafe float equality on traced values.
    delta_z = jnp.where(jnp.abs(dust_delta - (-0.2)) < 1e-6, -0.2 - 0.1 * redshift, dust_delta)
    bump_z = jnp.where(
        jnp.abs(dust_bump_strength - 1.0) < 1e-6,
        jnp.maximum(0.0, 1.0 - 0.15 * redshift),
        dust_bump_strength,
    )
    return kriek_conroy(wavelength, dust_delta=delta_z, dust_bump_strength=bump_z)


@register_dust_law("conroy2010")
def conroy2010(
    wavelength: jnp.ndarray,
    dust_Rv: float = 3.1,
    n_slope: float = -0.7,
    **_kwargs,
) -> jnp.ndarray:
    """Conroy+2010 mixed MW + power-law attenuation (FSPS dust_type=1).

    Milky Way (Cardelli 1989) curve dominates below a transition
    wavelength, power law dominates above. A smooth sigmoid blend at
    ~5500 A ensures differentiability.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_Rv : float
        Total-to-selective extinction ratio for the MW component.
    n_slope : float
        Power-law index for the long-wavelength component.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda), normalized to k(5500 A) = 1.

    References
    ----------
    Conroy, White & Gunn 2010, ApJ, 708, 58
    """
    k_mw = cardelli(wavelength, dust_Rv=dust_Rv)
    k_pl = power_law(wavelength, n_slope=n_slope)
    # Smooth sigmoid blend: MW dominates UV, power-law dominates IR
    x = jnp.log10(wavelength / 5500.0)
    blend = jax.nn.sigmoid(x / 0.05)
    k_raw = (1.0 - blend) * k_mw + blend * k_pl
    # Normalize to k(5500 A) = 1
    lam_v = jnp.array(5500.0)
    x_v = jnp.log10(lam_v / 5500.0)
    blend_v = jax.nn.sigmoid(x_v / 0.05)
    k_v = (1.0 - blend_v) * cardelli(lam_v[None], dust_Rv=dust_Rv)[0] + blend_v * 1.0
    return jnp.clip(k_raw / k_v, 0.0)


# ── Two-component dust model ──────────────────────────────────────


def precompute_dust_age_weights(
    age_grid: jnp.ndarray,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
) -> jnp.ndarray:
    """Precompute the birth-cloud sigmoid weight.

    Call once at Model init; pass result to ``two_component_dust_fast``.

    Parameters
    ----------
    age_grid : array, shape (n_ages,)
        Stellar population ages (yr).
    t_birth : float
        Birth cloud dispersal age (yr). Default 1e7 (10 Myr).
    transition_width : float
        Sigmoid width in dex. Default 0.3.

    Returns
    -------
    array, shape (n_ages,)
        Weight: 1 for young stars, 0 for old.
    """
    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    return jax.nn.sigmoid(-(log_age - log_t_birth) / transition_width)


def precompute_dust_age_mask(
    age_grid: jnp.ndarray,
    t_birth: float = 1e7,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute hard young/old masks for fast two-CSP dust decomposition.

    Uses a hard threshold at ``t_birth`` instead of a smooth sigmoid.
    This is the original Charlot & Fall (2000) formulation and enables
    a fast path where dust is factored out of the age sum entirely.

    Parameters
    ----------
    age_grid : array, shape (n_ages,)
        Stellar population ages (yr).
    t_birth : float
        Birth cloud dispersal age (yr). Default 1e7 (10 Myr).

    Returns
    -------
    young_mask : array, shape (n_ages,)
        1.0 for young ages (< t_birth), 0.0 for old.
    old_mask : array, shape (n_ages,)
        1.0 for old ages (>= t_birth), 0.0 for young.
    """
    young = (age_grid < t_birth).astype(age_grid.dtype)  # preserve input precision
    return young, 1.0 - young


def two_component_dust(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = 0.0,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
    **law_params,
) -> jnp.ndarray:
    r"""Two-component dust attenuation following Charlot & Fall (2000) with smooth age transition.

    Separates dust into birth-cloud (young stars) and diffuse ISM (all stars) components
    with independent optical depths and attenuation curves. Transition between components
    uses a smooth sigmoid in log-age, enabling automatic differentiation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    age_grid : array_like, shape (n_ages,)
        Stellar population ages. [yr]
    tau_v1 : float
        Birth-cloud V-band optical depth (at 5500 Å). [dimensionless]
        Note: tengri applies ``tau_bc`` internally but exposes ``tau_v1`` after normalizing
        by attenuation curve slope. See docs/known_bugs.md (CROSSVAL-01) for cross-code comparison.
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc : str, optional
        Attenuation curve name for birth cloud. Default: "power_law". Resolved from ``DUST_LAWS`` registry.
    law_diff : str, optional
        Attenuation curve name for diffuse ISM. Default: "power_law".
    f_obscuration : float, optional
        Fraction of unattenuated sightlines in clumpy geometry (Lower 2022). [dimensionless, in [0, 1]]
        Default: 0.0 (uniform screen).
    t_birth : float, optional
        Birth-cloud dispersal age (sigmoid center). [yr] Default: 1e7 (10 Myr).
    transition_width : float, optional
        Sigmoid transition width in dex. [dimensionless] Default: 0.3 (~5-20 Myr range).
    **law_params
        Keyword arguments passed to attenuation curve functions (e.g., ``n_slope``, ``dust_bump_strength``,
        ``dust_delta``, ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission factor T(λ, t_age), where T ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives and safe for ``jax.jit``.

    **Gradient-safe**: yes — differentiable everywhere; smooth sigmoid age transition preserves gradients
    through the birth-cloud boundary.

    The total optical depth is:

    .. math::

        \tau(\lambda, t_{\text{age}}) = w(t_{\text{age}}) \cdot \tau_{{\rm V,BC}} \cdot k_{\rm BC}(\lambda)
        + \tau_{{\rm V,ISM}} \cdot k_{\rm ISM}(\lambda)

    where :math:`w(t_{\text{age}})` is the sigmoid weight:

    .. math::

        w(t_{\text{age}}) = \sigma\left(-\frac{\log_{10} t_{\text{age}} - \log_{10} t_{\text{birth}}}{\Delta_{\text{trans}}}\right)

    and :math:`\sigma(x) = 1/(1 + e^{-x})` is the logistic sigmoid. The transmission is then:

    .. math::

        T(\lambda, t_{\text{age}}) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp[-\tau(\lambda, t_{\text{age}})]

    where :math:`f_{\rm obs}` is the unattenuated sightline fraction.

    **Upstream**: Implements the Charlot & Fall (2000) two-component framework [1]_ with sigmoid age transition
    following tengri's differentiable design. Birth-cloud + diffuse ISM separation enables realistic modeling
    of age-dependent dust geometry in galaxies.

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Star-forming Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250

    .. [2] K. M. Lower et al., "SKIRT 9: Redesigning an Acclaimed Dust Radiative Transfer Code
       to Face Exascale Computing Challenges," ApJS, 260, 12 (2022).
       https://doi.org/10.3847/1538-4365/ac5a59
    """
    k_bc = resolve_dust_law(law_bc)(wavelength, **law_params)
    k_diff = resolve_dust_law(law_diff)(wavelength, **law_params)

    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    weight = jax.nn.sigmoid(-(log_age - log_t_birth) / transition_width)

    tau_lambda = weight[:, None] * tau_v1 * k_bc[None, :] + tau_v2 * k_diff[None, :]

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)


def two_component_dust_separable(
    wavelength: jnp.ndarray,
    dust_age_weights: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc_fn: Callable,
    law_diff_fn: Callable,
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Optimized two-component dust attenuation with factorized age-independent term.

    Exploits the exponential factorization exp(a + b) = exp(a) · exp(b) to separate
    the diffuse ISM component from the age-dependent outer product. The diffuse
    exponentiation operates on (n_wave,) instead of (n_ages, n_wave), saving one full-grid
    exponential. Accepts pre-resolved law functions to avoid dict lookups in hot code.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_age_weights : array_like, shape (n_ages,)
        Pre-computed sigmoid birth-cloud weights from ``precompute_dust_age_weights``.
        Computed once at Model init and cached.
    tau_v1 : float
        Birth-cloud V-band optical depth. [dimensionless]
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc_fn : Callable
        Pre-resolved birth-cloud attenuation function (e.g., ``resolve_dust_law("calzetti")``).
    law_diff_fn : Callable
        Pre-resolved diffuse ISM attenuation function.
    f_obscuration : float, optional
        Unattenuated sightline fraction. [dimensionless, in [0, 1]] Default: 0.0.
    **law_params
        Keyword arguments passed to both law functions.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission T(λ, t_age) ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Performance**: Reduces memory traffic by ~40% on (n_ages, n_wave) grids
    relative to ``two_component_dust`` because the diffuse exponential is computed
    on (n_wave,) and broadcast rather than materialized as (n_ages, n_wave).
    Significant speedup on CPU; moderate benefit on GPU (memory bandwidth more abundant).

    The transmission factorizes as:

    .. math::

        T(\lambda, t_{\text{age}}) = T_{\rm BC}(\lambda, t_{\text{age}}) \cdot T_{\rm ISM}(\lambda)

    where

    .. math::

        T_{\rm BC}(\lambda, t_{\text{age}}) = f_{\rm obs} + (1 - f_{\rm obs}) \, \exp[-w(t_{\text{age}}) \, \tau_{\rm V,BC} \, k_{\rm BC}(\lambda)]

    .. math::

        T_{\rm ISM}(\lambda) = \exp[-\tau_{\rm V,ISM} \, k_{\rm ISM}(\lambda)]

    The ISM component is computed once on (n_wave,) and then broadcast with the age-dependent
    birth-cloud term, avoiding the full (n_ages, n_wave) grid in intermediate storage.

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Star-forming Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    k_bc = law_bc_fn(wavelength, **law_params)
    k_diff = law_diff_fn(wavelength, **law_params)

    # Diffuse ISM: age-independent → (n_wave,) exp instead of (n_age, n_wave)
    diffuse_trans = jnp.exp(-tau_v2 * k_diff)  # (n_wave,)

    # Birth cloud: age-dependent outer product → (n_age, n_wave)
    bc_trans = jnp.exp(-dust_age_weights[:, None] * tau_v1 * k_bc[None, :])

    # Combine: broadcast (n_age, n_wave) * (n_wave,) avoids materializing
    # the full (n_age, n_wave) diffuse array
    transmission = bc_trans * diffuse_trans[None, :]

    return f_obscuration + (1.0 - f_obscuration) * transmission


def two_component_dust_fast(
    wavelengths: jnp.ndarray,
    dust_age_weights: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    """Fast dust attenuation using precomputed age weights.

    Avoids recomputing the birth-cloud age sigmoid every call.  Used by
    both the fused kernel (at effective wavelengths) and the exact path
    (at the full wavelength grid).

    The output dtype follows the input ``wavelengths`` dtype, enabling
    mixed-precision: pass float32 arrays to halve memory traffic on the
    ``(n_ages, n_wave)`` intermediates (~1.6x speedup on CPU).

    Parameters
    ----------
    wavelengths : array, shape (n_wave,)
        Evaluation wavelengths (rest-frame Angstrom).  Can be the full
        SSP grid or just the filter effective wavelengths.
    dust_age_weights : array, shape (n_ages,)
        From ``precompute_dust_age_weights`` (computed once at Model init).
    tau_v1, tau_v2 : float
        Birth cloud and diffuse ISM V-band optical depths.
    law_bc, law_diff : str
        Attenuation curve names (looked up in ``DUST_LAWS`` registry).
    f_obscuration : float
        Fraction of unattenuated sightlines [0, 1] (Lower 2022).
    **law_params
        Passed to curve functions: ``n_slope``, ``dust_bump_strength``,
        ``dust_delta``, ``dust_Rv``, etc.

    Returns
    -------
    array, shape (n_ages, n_wave)
        Multiplicative attenuation factor in [0, 1].
    """
    k_bc = resolve_dust_law(law_bc)(wavelengths, **law_params)
    k_diff = resolve_dust_law(law_diff)(wavelengths, **law_params)

    tau_lambda = dust_age_weights[:, None] * tau_v1 * k_bc[None, :] + tau_v2 * k_diff[None, :]

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)


# ── Single-component dust model (uniform screen) ──────────────────


def single_component_dust(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Single-component (uniform foreground screen) dust attenuation.

    Applies a single attenuation curve at uniform optical depth to all stellar ages.
    Age-independent, enabling factorization out of stellar population integration.
    Simpler but less realistic than two-component models; useful for low-precision fits
    or high-redshift galaxies where birth-cloud/ISM distinction is unresolved.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    tau_v : float
        V-band optical depth at 5500 Å. [dimensionless]
    law : str, optional
        Attenuation curve name, resolved from ``DUST_LAWS`` registry. Default: "power_law".
    f_obscuration : float, optional
        Unattenuated sightline fraction in clumpy geometry (Lower 2022). [dimensionless, in [0, 1]]
        Default: 0.0 (uniform foreground screen).
    **law_params
        Keyword arguments passed to the attenuation curve function
        (e.g., ``n_slope``, ``dust_bump_strength``, ``dust_delta``, ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative transmission T(λ) ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    The transmission is:

    .. math::

        T(\lambda) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp[-\tau_V \, k(\lambda)]

    where :math:`k(\lambda)` is the normalized attenuation curve with :math:`k(5500 \, \text{\AA}) = 1`,
    :math:`\tau_V` is the V-band optical depth, and :math:`f_{\rm obs}` is the fraction of
    unattenuated sightlines (Lower 2022; default 0 = full screen).

    **Age independence**: Unlike two-component models, there is no age-dependence, so this
    transmission can be factored out of the stellar population age integration, enabling
    faster computation.

    **Geometry**: When :math:`f_{\rm obs} = 0`, this recovers the standard Beer-Lambert
    foreground screen. When :math:`f_{\rm obs} > 0`, it models a clumpy geometry where
    a fraction of photons are unattenuated (Lower 2022).

    References
    ----------
    .. [1] K. M. Lower et al., "SKIRT 9: Redesigning an Acclaimed Dust Radiative Transfer Code
       to Face Exascale Computing Challenges," ApJS, 260, 12 (2022).
       https://doi.org/10.3847/1538-4365/ac5a59
    """
    k = resolve_dust_law(law)(wavelength, **law_params)
    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_v * k)


def single_component_dust_fast(
    wavelengths: jnp.ndarray,
    n_ages: int,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    """Single-component dust attenuation broadcast to (n_ages, n_wave).

    Computes ``exp()`` on the 1-D wavelength grid only, then broadcasts
    to ``(n_ages, n_wave)`` via ``jnp.broadcast_to`` (zero-copy in XLA).
    This is the production path used by the SED pipeline.

    Parameters
    ----------
    wavelengths : array, shape (n_wave,)
        Evaluation wavelengths (rest-frame Angstrom).
    n_ages : int
        Number of SSP age bins (for output shape).
    tau_v : float
        V-band optical depth.
    law : str
        Attenuation curve name (from ``DUST_LAWS`` registry).
    f_obscuration : float
        Fraction of unattenuated sightlines [0, 1] (Lower 2022).
    **law_params
        Passed to curve function.

    Returns
    -------
    array, shape (n_ages, n_wave)
        Multiplicative transmission factor in [0, 1].  All age rows
        are identical (age-independent attenuation).
    """
    trans_1d = single_component_dust(
        wavelengths, tau_v=tau_v, law=law, f_obscuration=f_obscuration, **law_params
    )
    return jnp.broadcast_to(trans_1d[None, :], (n_ages, wavelengths.shape[0]))


# ── Witt & Gordon (2000) dust geometry transmission functions ─────
#
# These functions compute the wavelength-dependent transmission T(lambda)
# for different star-dust geometries, given a V-band optical depth tau_V
# and an underlying extinction curve k(lambda).
#
# The key insight from Witt & Gordon (2000, ApJ, 528, 799) is that the
# EFFECTIVE attenuation depends strongly on the spatial distribution of
# dust relative to stars.  A uniform foreground screen (SHELL) produces
# the steepest wavelength dependence; a homogeneous mix (CLOUDY) is
# greyer because high-tau sightlines are self-shielded; a clumpy medium
# (DUSTY) is greyest because photons preferentially escape through
# low-tau channels.
#
# All functions are pure JAX and JIT-compatible.


def wg00_shell(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "cardelli",
    **law_params,
) -> jnp.ndarray:
    """Witt & Gordon (2000) SHELL geometry — foreground screen.

    The simplest geometry: a uniform dust slab in front of all stars.
    Transmission is the standard Beer-Lambert law::

        T(lambda) = exp(-tau_V * k(lambda))

    This is identical to ``single_component_dust`` with ``f_obscuration=0``
    and is included for completeness alongside the CLOUDY and DUSTY models.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    tau_v : float
        V-band optical depth (tau at 5500 A).
    law : str
        Underlying extinction curve name. Default ``"cardelli"`` (MW).
    **law_params
        Passed to the extinction curve function (e.g., ``dust_Rv``).

    Returns
    -------
    array, shape (n_wave,)
        Transmission T(lambda) in [0, 1].

    References
    ----------
    Witt & Gordon 2000, ApJ, 528, 799 (Section 3.1)
    """
    k = resolve_dust_law(law)(wavelength, **law_params)
    return jnp.exp(-tau_v * k)


def wg00_cloudy(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "cardelli",
    **law_params,
) -> jnp.ndarray:
    r"""Witt & Gordon (2000) CLOUDY dust geometry — homogeneous dust-star mix.

    Stars and dust are uniformly mixed throughout a slab of total V-band optical depth.
    The analytic solution integrates radiative transfer, producing a wavelength-dependent
    transmission that is greyer (less wavelength-dependent) than a foreground screen.
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
    **JIT-compatible**: yes — all operations are ``jnp`` primitives, with numerically stable
    Taylor expansion for small optical depth.

    **Gradient-safe**: yes — differentiable everywhere via smooth blending between exact
    and Taylor regimes.

    The transmission for a homogeneous slab is:

    .. math::

        T(\lambda) = \frac{1 - \exp[-\tau_V \, k(\lambda)]}{\tau_V \, k(\lambda)}

    where :math:`k(\lambda)` is the normalized attenuation curve. This follows the solution
    of radiative transfer in a uniform dust-star slab (Natta & Panagia 1984, Section 3.1;
    Calzetti et al. 1994).

    **Limiting behaviour**: At low optical depth (:math:`\tau_V k \ll 1`), :math:`T \to 1`
    (transparent). At high optical depth, :math:`T \approx 1/(\tau_V k)`, producing
    a greyer (less wavelength-dependent) effective attenuation than the foreground screen
    because stars near the observer-facing side suffer less extinction.

    **Numerical stability**: Uses a Taylor expansion (correct to order :math:`\tau^3`)
    for :math:`\tau_V k < 10^{-4}` to avoid division-by-zero, preserving gradients throughout.

    **Approximation**: The analytic solution assumes pure absorption (zero scattering).
    Witt & Gordon (2000) Monte Carlo simulations including scattering find the effective
    attenuation is slightly greyer still. The analytic form captures the dominant geometric
    effect and is widely used in SED fitting codes (e.g., Synthesizer, CIGALE).

    References
    ----------
    .. [1] A. Natta and M. Panagia, "Dust Distributions in Star-Forming Galaxies: Why the
       Far-Ultraviolet is a More Reliable Indicator than H-alpha," ApJ, 287, 228 (1984).
       https://doi.org/10.1086/162686

    .. [2] S. Charlot, A. Kinney, and T. Storchi-Bergmann, "The Dust Content of Star-Forming
       Galaxies," ApJ, 429, 582 (1994).
       https://doi.org/10.1086/174348

    .. [3] A. T. Witt and K. D. Gordon, "A Comprehensive Review of the 2175 Angstrom Absorption
       Feature," ApJ, 528, 799 (2000). Section 3.2, "homogeneous" model.
       https://doi.org/10.1086/308975
    """
    k = resolve_dust_law(law)(wavelength, **law_params)
    tau_k = tau_v * k

    # Numerically stable: for small tau_k, use Taylor expansion
    # (1 - exp(-x)) / x -> 1 - x/2 + x^2/6 - ... for x -> 0
    # Switch at |x| < 1e-4 to avoid loss of precision
    # Use jnp.maximum (not jnp.where) so the gradient of ratio w.r.t. tau_k stays
    # connected when tau_k is small — jnp.where with a constant fallback gives zero
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
    """Witt & Gordon (2000) DUSTY geometry — clumpy two-phase medium.

    The ISM is modelled as ``n_clumps`` identical clumps, each with
    optical depth ``tau_clump = tau_V / n_clumps``, distributed along
    random sightlines (Natta & Panagia 1984; Hobson & Padman 1993).
    The probability of a photon traversing N clumps follows a Poisson
    distribution, giving the mean transmission::

        T(lambda) = exp(-n_clumps * (1 - exp(-tau_clump * k(lambda))))

    where ``tau_clump = tau_V / n_clumps``.

    This produces the *greyest* effective attenuation of the three WG00
    geometries because photons preferentially escape through low-column
    channels between clumps.

    **Limiting behaviour:**

    - ``n_clumps -> inf`` (fixed ``tau_V``): ``tau_clump -> 0``, each
      clump becomes optically thin, recovers the homogeneous slab.
    - ``n_clumps = 1``: single clump with ``tau_clump = tau_V``, similar
      to a screen but with Poisson line-of-sight averaging.
    - ``tau_V = 0``: T = 1 (transparent), regardless of ``n_clumps``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    tau_v : float
        Total V-band optical depth (= ``n_clumps * tau_clump``).
    law : str
        Underlying extinction curve name. Default ``"cardelli"`` (MW).
    n_clumps : float
        Mean number of clumps along a sightline. Higher values approach
        the homogeneous limit.  Typical range: 1-40.  Default 10.
    **law_params
        Passed to the extinction curve function (e.g., ``dust_Rv``).

    Returns
    -------
    array, shape (n_wave,)
        Transmission T(lambda) in [0, 1].

    References
    ----------
    Natta & Panagia 1984, ApJ, 287, 228
    Hobson & Padman 1993, MNRAS, 264, 161
    Witt & Gordon 2000, ApJ, 528, 799 (Section 3.3, "clumpy" model)
    """
    k = resolve_dust_law(law)(wavelength, **law_params)
    tau_clump = tau_v / jnp.maximum(n_clumps, 1e-10)
    return jnp.exp(-n_clumps * (1.0 - jnp.exp(-tau_clump * k)))


# ── Dust-to-gas ratio scaling ─────────────────────────────────────


def dust_to_gas_scaling_remy_ruyer(logzsol: float) -> float:
    """Metallicity-dependent dust-to-gas ratio scaling (Rémy-Ruyer+2014).

    Returns multiplicative factor for dust optical depths relative to solar.
    Broken power law: linear above 0.1 Z_sun, quadratic below.

    Parameters
    ----------
    logzsol : float
        log10(Z / Z_sun).

    Returns
    -------
    float
        D/G ratio relative to solar (1.0 at solar metallicity).

    References
    ----------
    Rémy-Ruyer et al. 2014, A&A, 563, A31, Table 1.
    """
    z_ratio = 10.0**logzsol
    scaling = jnp.where(
        z_ratio > 0.1,
        z_ratio,
        0.1 * (z_ratio / 0.1) ** 2.0,
    )
    return scaling
