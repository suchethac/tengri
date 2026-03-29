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
- **smc**: Gordon et al. (2003) SMC Bar, steep UV, no 2175A bump
- **cardelli**: Cardelli et al. (1989) MW curve with free R_V
- **li08**: Li et al. (2008) parametric 3-slope + UV bump (reproduces MW/SMC/Calzetti)
- **salim**: Salim et al. (2018) modified Calzetti (= DSPS default)
- **tea**: Haskell et al. (2024) TEA 3-param empirical (NIHAO-SKIRT bump-slope correlation)
- **narayanan_z**: Narayanan et al. (2018) redshift-dependent Kriek-Conroy (SIMBA RT)
- **conroy2010**: Conroy+2010 mixed MW + power-law (FSPS dust_type=1)

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
- Li et al. 2008, MNRAS, 385, 1903
- Lower et al. 2022, ApJ, 931, 14
- Narayanan et al. 2018, ApJ, 869, 70
- Natta & Panagia 1984, ApJ, 287, 228
- Noll et al. 2009, A&A, 507, 1793
- Salim, Boquien & Lee 2018, ApJ, 859, 11
- Witt & Gordon 2000, ApJ, 528, 799
- Zacharegkas et al. 2025, arXiv:2506.19919
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

# ===================================================================
# Attenuation curve registry
# ===================================================================

DUST_LAWS: dict[str, Callable] = {}


def register_dust_law(name: str) -> Callable:
    """Register a dust attenuation curve function (decorator factory)."""

    def decorator(fn: Callable) -> Callable:
        DUST_LAWS[name] = fn
        return fn

    return decorator


def get_dust_law(name: str) -> Callable:
    """Get a registered dust law by name."""
    if name not in DUST_LAWS:
        raise ValueError(f"Unknown dust law '{name}'. Available: {list(DUST_LAWS.keys())}")
    return DUST_LAWS[name]


# ===================================================================
# Utility: Drude profile for the 2175 Angstrom UV bump
# ===================================================================


def _drude_profile(
    wave_um: jnp.ndarray,
    x0: float = 0.2175,
    gamma: float = 0.035,
) -> jnp.ndarray:
    """Drude profile for the 2175 Angstrom UV bump.

    D(lambda) = (lambda * gamma)^2 / ((lambda^2 - x0^2)^2 + (lambda*gamma)^2)
    """
    return (wave_um * gamma) ** 2 / ((wave_um**2 - x0**2) ** 2 + (wave_um * gamma) ** 2)


# ===================================================================
# Internal helpers
# ===================================================================


def _calzetti_l02_kprime(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Compute k'(lambda) = A(lambda)/E(B-V) using L02 + C00.

    Returns the raw reddening curve (NOT normalized by R_V).
    Uses Leitherer (2002) for lambda <= 1500 A, Calzetti (2000) above.

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.

    Returns
    -------
    array
        k'(lambda) reddening curve.
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

    # Use L02 below 0.15 um, Calzetti above
    k_calz = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    return jnp.where(wave_um <= 0.15, k_l02, k_calz)


# ===================================================================
# Attenuation curves
# ===================================================================


@register_dust_law("power_law")
def power_law(
    wavelength: jnp.ndarray,
    n_slope: float = -0.7,
    **_kwargs,
) -> jnp.ndarray:
    """Simple power-law: k(lambda) = (lambda/5500)^n.

    Original Charlot & Fall (2000) wavelength dependence.
    """
    return (wavelength / 5500.0) ** n_slope


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
    dust_UV_slope: float = -1.0,
    dust_OPT_slope: float = -1.3,
    dust_FUV_slope: float = -1.8,
    dust_bump_strength: float = 1.0,
    **_kwargs,
) -> jnp.ndarray:
    """Li et al. (2008) parametric 3-slope attenuation + UV bump.

    A flexible single functional form with independent power-law slopes
    in three wavelength regimes (FUV, UV-optical, optical-NIR) joined by
    smooth sigmoid transitions, plus a Drude UV bump at 2175 Angstrom.
    Can reproduce MW, SMC, LMC, and Calzetti curves as special cases.

    The unnormalized curve is::

        k_raw(lambda) = (lambda/1500)^FUV_slope  * sigma_FUV(lambda)
                      + (lambda/1500)^UV_slope   * sigma_UV(lambda)
                      + (lambda/6000)^OPT_slope  * sigma_OPT(lambda)
                      + bump * D(lambda, 2175A)

    where sigma are sigmoid weighting functions for smooth transitions,
    and the result is normalized so that k(5500 A) = 1.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    dust_UV_slope : float
        Power-law slope in the UV regime (1500-6000 A).
        MW ~ -1.0, SMC ~ -1.2, Calzetti ~ -0.75.
    dust_OPT_slope : float
        Power-law slope in the optical-NIR regime (> 6000 A).
        MW ~ -1.3, SMC ~ -1.6, Calzetti ~ -1.05.
    dust_FUV_slope : float
        Power-law slope in the FUV regime (< 1500 A).
        MW ~ -1.8, SMC ~ -2.4, Calzetti ~ -1.4.
    dust_bump_strength : float
        Amplitude of the 2175 A UV bump. MW ~ 1.0, SMC/Calzetti ~ 0.0.

    Returns
    -------
    array, shape (n_wave,)
        Attenuation curve k(lambda), normalized to k(5500 A) = 1.

    Notes
    -----
    Presets for common curves:

    - **MW**: UV_slope=-1.0, OPT_slope=-1.3, FUV_slope=-1.8, bump=1.0
    - **SMC**: UV_slope=-1.2, OPT_slope=-1.6, FUV_slope=-2.4, bump=0.0
    - **LMC**: UV_slope=-1.1, OPT_slope=-1.4, FUV_slope=-2.0, bump=0.5
    - **Calzetti**: UV_slope=-0.75, OPT_slope=-1.05, FUV_slope=-1.4, bump=0.0

    References
    ----------
    Li et al. 2008, MNRAS, 385, 1903
    """
    wave_um = wavelength / 1e4

    # Pivot wavelengths in Angstrom
    lam_fuv = 1500.0
    lam_opt = 6000.0

    # Smooth sigmoid transitions (width ~ 0.1 in log-lambda for
    # differentiability; steepness 20 gives ~95% transition over
    # factor-of-1.3 in wavelength)
    steepness = 20.0
    log_wave = jnp.log10(wavelength)
    log_fuv = jnp.log10(lam_fuv)
    log_opt = jnp.log10(lam_opt)

    # w_fuv ~ 1 for lambda << 1500, ~ 0 for lambda >> 1500
    w_fuv = jax.nn.sigmoid(-steepness * (log_wave - log_fuv))
    # w_opt ~ 1 for lambda >> 6000, ~ 0 for lambda << 6000
    w_opt = jax.nn.sigmoid(steepness * (log_wave - log_opt))
    # w_uv fills the middle
    w_uv = 1.0 - w_fuv - w_opt

    # Three power-law segments
    k_fuv = (wavelength / lam_fuv) ** dust_FUV_slope
    k_uv = (wavelength / lam_fuv) ** dust_UV_slope
    k_opt = (wavelength / lam_opt) ** dust_OPT_slope

    # Weighted combination
    k_raw = w_fuv * k_fuv + w_uv * k_uv + w_opt * k_opt

    # UV bump via Drude profile at 2175 A
    bump = dust_bump_strength * _drude_profile(wave_um)
    k_raw = k_raw + bump

    # Normalize to k(V) = 1 at 5500 A
    # Evaluate at 5500 A using the same formula
    lam_v = 5500.0
    log_v = jnp.log10(lam_v)
    w_fuv_v = jax.nn.sigmoid(-steepness * (log_v - log_fuv))
    w_opt_v = jax.nn.sigmoid(steepness * (log_v - log_opt))
    w_uv_v = 1.0 - w_fuv_v - w_opt_v

    k_v = (
        w_fuv_v * (lam_v / lam_fuv) ** dust_FUV_slope
        + w_uv_v * (lam_v / lam_fuv) ** dust_UV_slope
        + w_opt_v * (lam_v / lam_opt) ** dust_OPT_slope
        + dust_bump_strength * _drude_profile(jnp.array(lam_v / 1e4))
    )

    return jnp.clip(k_raw / k_v, 0.0)


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

    The k'(lambda) = A(lambda)/E(B-V) polynomial is::

        k'(lambda) = 5.472 + 0.671/x - 9.218e-3/x^2 + 2.620e-3/x^3

    where x = lambda in microns.

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
    delta_z = jnp.where(dust_delta == -0.2, -0.2 - 0.1 * redshift, dust_delta)
    bump_z = jnp.where(
        dust_bump_strength == 1.0,
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


# ===================================================================
# Two-component dust model
# ===================================================================


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
    young = (age_grid < t_birth).astype(jnp.float64)
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
    """Generalized two-component dust attenuation.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid (Angstrom).
    age_grid : array, shape (n_ages,)
        Stellar population ages (yr).
    tau_v1 : float
        V-band optical depth of birth cloud.
    tau_v2 : float
        V-band optical depth of diffuse ISM.
    law_bc : str
        Attenuation curve name for birth cloud.
    law_diff : str
        Attenuation curve name for diffuse ISM.
    f_obscuration : float
        Fraction of unattenuated sightlines [0, 1] (Lower 2022).
    t_birth : float
        Birth cloud dispersal age (yr).
    transition_width : float
        Sigmoid width in dex.
    **law_params
        Passed to curve functions: n_slope, dust_bump_strength,
        dust_delta, dust_Rv, etc.

    Returns
    -------
    array, shape (n_ages, n_wave)
        Multiplicative attenuation factor in [0, 1].
    """
    k_bc = get_dust_law(law_bc)(wavelength, **law_params)
    k_diff = get_dust_law(law_diff)(wavelength, **law_params)

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
    """Optimized dust attenuation: separates age-dependent and age-independent terms.

    Exploits exp(a + b) = exp(a) * exp(b) to factor the diffuse ISM
    component out of the (n_age, n_wave) broadcast.  The diffuse ``exp()``
    operates on (n_wave,) instead of (n_age, n_wave), saving one full-grid
    exponentiation.  Accepts pre-resolved law functions (no dict lookup).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid (Angstrom).
    dust_age_weights : array, shape (n_ages,)
        From ``precompute_dust_age_weights`` (precomputed at Model init).
    tau_v1, tau_v2 : float
        Birth cloud and diffuse ISM optical depths.
    law_bc_fn, law_diff_fn : callable
        Pre-resolved dust law functions (e.g. ``get_dust_law("calzetti")``).
    f_obscuration : float
        Unattenuated fraction [0, 1].
    **law_params
        Passed to law functions.

    Returns
    -------
    array, shape (n_ages, n_wave)
        Multiplicative attenuation factor in [0, 1].
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
    k_bc = get_dust_law(law_bc)(wavelengths, **law_params)
    k_diff = get_dust_law(law_diff)(wavelengths, **law_params)

    tau_lambda = dust_age_weights[:, None] * tau_v1 * k_bc[None, :] + tau_v2 * k_diff[None, :]

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)


# ===================================================================
# Single-component dust model (uniform screen)
# ===================================================================


def single_component_dust(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    """Single-component (uniform screen) dust attenuation.

    Applies one attenuation curve at one optical depth to all stellar
    ages identically.  Faster than the two-component model because there
    is no age-dependent birth-cloud term.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid (Angstrom).
    tau_v : float
        V-band optical depth.
    law : str
        Attenuation curve name (from ``DUST_LAWS`` registry).
    f_obscuration : float
        Fraction of unattenuated sightlines [0, 1] (Lower 2022).
    **law_params
        Passed to curve function: ``n_slope``, ``dust_bump_strength``,
        ``dust_delta``, ``dust_Rv``, etc.

    Returns
    -------
    array, shape (n_wave,)
        Multiplicative transmission factor in [0, 1].
    """
    k = get_dust_law(law)(wavelength, **law_params)
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
    return jnp.broadcast_to(trans_1d[None, :], (n_ages, len(wavelengths)))


# ===================================================================
# Witt & Gordon (2000) dust geometry transmission functions
# ===================================================================
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
    k = get_dust_law(law)(wavelength, **law_params)
    return jnp.exp(-tau_v * k)


def wg00_cloudy(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "cardelli",
    **law_params,
) -> jnp.ndarray:
    """Witt & Gordon (2000) CLOUDY geometry — homogeneous dust-star mix.

    Stars and dust are uniformly mixed throughout a slab of total
    V-band optical depth ``tau_V``.  The analytic solution for a
    homogeneous slab (e.g., Natta & Panagia 1984; Calzetti et al. 1994)
    integrates the radiative transfer along the slab::

        T(lambda) = (1 - exp(-tau_V * k(lambda))) / (tau_V * k(lambda))

    At low optical depth (tau*k -> 0), T -> 1 (transparent).
    At high optical depth, T -> 1/(tau*k), producing a *greyer* curve
    than the foreground screen because stars near the observer's side
    of the slab suffer less extinction.

    A numerically stable implementation is used to avoid division by
    zero when tau*k is very small.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    tau_v : float
        Total V-band optical depth through the slab.
    law : str
        Underlying extinction curve name. Default ``"cardelli"`` (MW).
    **law_params
        Passed to the extinction curve function (e.g., ``dust_Rv``).

    Returns
    -------
    array, shape (n_wave,)
        Transmission T(lambda) in [0, 1].

    Notes
    -----
    The slab formula is the zero-scattering (absorption-only) solution.
    WG00 Monte Carlo simulations include scattering, which makes the
    effective attenuation slightly greyer still.  The analytic form here
    captures the dominant geometric effect and is widely used in SED
    fitting codes (e.g., Synthesizer, CIGALE).

    References
    ----------
    Natta & Panagia 1984, ApJ, 287, 228
    Calzetti, Kinney & Storchi-Bergmann 1994, ApJ, 429, 582
    Witt & Gordon 2000, ApJ, 528, 799 (Section 3.2, "homogeneous" model)
    """
    k = get_dust_law(law)(wavelength, **law_params)
    tau_k = tau_v * k

    # Numerically stable: for small tau_k, use Taylor expansion
    # (1 - exp(-x)) / x -> 1 - x/2 + x^2/6 - ... for x -> 0
    # Switch at |x| < 1e-4 to avoid loss of precision
    safe_tau_k = jnp.where(tau_k > 1e-10, tau_k, 1.0)
    ratio = (1.0 - jnp.exp(-safe_tau_k)) / safe_tau_k
    # Taylor expansion for small tau_k: 1 - tau_k/2
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
    k = get_dust_law(law)(wavelength, **law_params)
    tau_clump = tau_v / jnp.maximum(n_clumps, 1e-10)
    return jnp.exp(-n_clumps * (1.0 - jnp.exp(-tau_clump * k)))
