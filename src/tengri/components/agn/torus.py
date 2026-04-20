"""Toy torus models for AGN infrared emission.

.. warning::
    These are **toy models** using 1-2 temperature modified blackbodies.
    They are NOT radiative transfer results and should NOT be used for
    science.  For production work, use the SKIRTOR templates in
    ``tengri.components.agn.skirtor`` (tabulated from 3D Monte Carlo RT).

Two toy models are provided for testing and fast prototyping:

1. **simple_torus** — single-temperature modified blackbody
   with silicate opacity. 2 free parameters.
2. **two_temperature_torus** — hot + warm dust components. 4 free params.

Both return specific luminosity L_nu in erg/s/Hz. All functions are pure
JAX and JIT-compilable.

References
----------
- Nenkova et al. 2008, ApJ, 685, 147 (CLUMPY torus)
- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR)
- Draine 2003, ARA&A, 41, 241 (silicate opacity)
"""

import warnings

import jax.numpy as jnp

from tengri.components.agn._phys import (
    LSUN_ERG as _LSUN_ERG,
    planck_lnu as _planck_lnu,
    wavelength_to_nu as _wavelength_to_nu,
)

# ── Physical constants (CGS) ──────────────────────────────────────

_MICRON_ANGSTROM = 1e4  # Micron -> Angstrom

# Silicate feature wavelength
_LAMBDA_SI = 9.7 * _MICRON_ANGSTROM  # 9.7 um in Angstrom


# ── Model 1: Simple hot blackbody torus ───────────────────────────


def simple_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float = 0.5,
    agn_T_torus: float = 1000.0,
    agn_tau_torus: float = 5.0,
    agn_tau_beta: float = 1.5,
    **_kwargs,
) -> jnp.ndarray:
    """Simple single-temperature dust torus with silicate opacity.

    L_nu = L_bol * f_torus * B_nu(T_torus) / B_int * (1 - exp(-tau * (9.7um/lam)^beta))

    where B_int normalizes the modified blackbody to integrate to
    L_bol * f_torus.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_torus_frac : float
        Fraction of L_bol re-emitted by torus (covering factor).
        Typical range: 0.1 to 0.9. Default 0.5.
    agn_T_torus : float
        Torus dust temperature [K].
        Typical range: 500 to 1500. Default 1000.
    agn_tau_torus : float
        Optical depth at 9.7 um silicate feature.
        Typical range: 1 to 10. Default 5.
    agn_tau_beta : float
        Power-law index for opacity wavelength dependence.
        Typical range: 1.0 to 2.0. Default 1.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].
    """
    warnings.warn(
        "simple_torus is a toy model (single-temperature MBB, not radiative transfer) "
        "and should NOT be used for science. Use skirtor_analytic from "
        "tengri.components.agn.skirtor for production work.",
        DeprecationWarning,
        stacklevel=2,
    )
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    nu = _wavelength_to_nu(wavelength)

    # Blackbody emission
    b_nu = _planck_lnu(nu, agn_T_torus)

    # Silicate opacity: tau(lambda) = tau_torus * (9.7um / lambda)^beta
    opacity = 1.0 - jnp.exp(
        -agn_tau_torus * (_LAMBDA_SI / jnp.maximum(wavelength, 1.0)) ** agn_tau_beta
    )

    # Modified blackbody shape
    shape = b_nu * opacity

    # Normalize to L_bol * f_torus
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg


# ── Model 2: Two-temperature torus (SKIRTOR-inspired) ─────────────


def two_temperature_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float = 0.5,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_tau_torus: float = 5.0,
    agn_tau_beta: float = 1.5,
    **_kwargs,
) -> jnp.ndarray:
    """Two-temperature dust torus (hot sublimation + warm outer torus).

    Inspired by SKIRTOR clumpy torus models. The emission is a mixture
    of two modified blackbodies:

        L_nu = f_hot * BB(T_hot) + (1 - f_hot) * BB(T_warm)

    both modified by the same silicate opacity profile.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_torus_frac : float
        Fraction of L_bol re-emitted by torus. Default 0.5.
    agn_T_hot : float
        Hot dust temperature [K], near sublimation.
        Typical range: 1000 to 1500. Default 1200.
    agn_T_warm : float
        Warm dust temperature [K], outer torus.
        Typical range: 200 to 800. Default 300.
    agn_frac_hot : float
        Luminosity fraction in hot component (0 to 1). Default 0.3.
    agn_tau_torus : float
        Optical depth at 9.7 um. Default 5.
    agn_tau_beta : float
        Opacity power-law index. Default 1.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].
    """
    warnings.warn(
        "two_temperature_torus is a toy model (two-temperature MBB, not radiative transfer) "
        "and should NOT be used for science. Use skirtor_analytic from "
        "tengri.components.agn.skirtor for production work.",
        DeprecationWarning,
        stacklevel=2,
    )
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    nu = _wavelength_to_nu(wavelength)

    # Two blackbody components
    b_hot = _planck_lnu(nu, agn_T_hot)
    b_warm = _planck_lnu(nu, agn_T_warm)

    # Silicate opacity
    opacity = 1.0 - jnp.exp(
        -agn_tau_torus * (_LAMBDA_SI / jnp.maximum(wavelength, 1.0)) ** agn_tau_beta
    )

    # Weighted mixture with opacity
    shape = (agn_frac_hot * b_hot + (1.0 - agn_frac_hot) * b_warm) * opacity

    # Normalize
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg
