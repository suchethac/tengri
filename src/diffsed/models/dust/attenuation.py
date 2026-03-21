"""Dust attenuation model for diffsed.

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
- **kriek_conroy**: Calzetti + UV bump + slope delta — Prospector default
- **smc**: Gordon et al. (2003) SMC Bar, steep UV, no 2175A bump
- **cardelli**: Cardelli et al. (1989) MW curve with free R_V
- **li08**: Li et al. (2008) parametric 3-slope + UV bump (reproduces MW/SMC/Calzetti)
- **salim**: Salim et al. (2018) modified Calzetti (= DSPS default)

References
----------
- Calzetti et al. 2000, ApJ, 533, 682
- Cardelli et al. 1989, ApJ, 345, 245
- Charlot & Fall 2000, ApJ, 539, 718
- Gordon et al. 2003, ApJ, 594, 279
- Kriek & Conroy 2013, ApJL, 775, L16
- Li et al. 2008, MNRAS, 385, 1903
- Lower et al. 2022, ApJ, 931, 14
- Salim et al. 2018, ApJ, 859, 11
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
        raise ValueError(
            f"Unknown dust law '{name}'. Available: {list(DUST_LAWS.keys())}"
        )
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
    return (
        (wave_um * gamma) ** 2
        / ((wave_um**2 - x0**2) ** 2 + (wave_um * gamma) ** 2)
    )


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


@register_dust_law("smc")
def smc(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """SMC Bar extinction curve (Gordon+2003 / Pei 1992).

    Steep UV rise, NO 2175A bump. Common at high redshift.
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    k_uv = 1.0 + 0.2 * (x - 3.33) + 0.25 * (x - 3.33) ** 2
    k_opt = 1.0 + 1.39 * (x - 1.82)
    k_ir = 0.6 * x**1.7

    k = jnp.where(
        wave_um < 0.3, k_uv,
        jnp.where(wave_um < 1.0, k_opt, k_ir),
    )
    return jnp.clip(k, 0.0)


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
    a_opt = (1.0 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3
             + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6
             + 0.32999 * y**7)
    b_opt = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3
             - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6
             - 2.09002 * y**7)

    # UV: 3.3 <= x <= 8.0
    f_a = jnp.where(x >= 5.9,
                     -0.04473 * (x - 5.9)**2 - 0.009779 * (x - 5.9)**3, 0.0)
    f_b = jnp.where(x >= 5.9,
                     0.2130 * (x - 5.9)**2 + 0.1207 * (x - 5.9)**3, 0.0)
    a_uv = 1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341) + f_a
    b_uv = -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263) + f_b

    # Far-UV: 8.0 <= x <= 10.0 (CCM89 Table 4)
    y_fuv = jnp.clip(x, 8.0, 10.0) - 8.0
    a_fuv = -1.073 - 0.628 * y_fuv + 0.137 * y_fuv**2 - 0.070 * y_fuv**3
    b_fuv = 13.670 + 4.257 * y_fuv - 0.420 * y_fuv**2 + 0.374 * y_fuv**3

    a = jnp.where(
        x < 1.1, a_ir,
        jnp.where(x < 3.3, a_opt, jnp.where(x < 8.0, a_uv, a_fuv)),
    )
    b = jnp.where(
        x < 1.1, b_ir,
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

    k_v = (w_fuv_v * (lam_v / lam_fuv) ** dust_FUV_slope
           + w_uv_v * (lam_v / lam_fuv) ** dust_UV_slope
           + w_opt_v * (lam_v / lam_opt) ** dust_OPT_slope
           + dust_bump_strength * _drude_profile(jnp.array(lam_v / 1e4)))

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

    tau_lambda = (
        weight[:, None] * tau_v1 * k_bc[None, :]
        + tau_v2 * k_diff[None, :]
    )

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
    """Fast path using precomputed age weights.

    Parameters
    ----------
    wavelengths : array, shape (n_filters,)
        Evaluation wavelengths (rest-frame Angstrom).
    dust_age_weights : array, shape (n_ages,)
        From ``precompute_dust_age_weights``.
    tau_v1, tau_v2 : float
        Birth cloud and diffuse ISM optical depths.
    law_bc, law_diff : str
        Attenuation curve names.
    f_obscuration : float
        Unattenuated fraction [0, 1].
    **law_params
        Curve parameters.

    Returns
    -------
    array, shape (n_ages, n_filters)
    """
    k_bc = get_dust_law(law_bc)(wavelengths, **law_params)
    k_diff = get_dust_law(law_diff)(wavelengths, **law_params)

    tau_lambda = (
        dust_age_weights[:, None] * tau_v1 * k_bc[None, :]
        + tau_v2 * k_diff[None, :]
    )

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)
