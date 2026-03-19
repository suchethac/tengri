"""Dust torus models for AGN infrared emission.

Two models are provided:

1. **Simple hot blackbody torus** — single-temperature modified blackbody
   with silicate opacity. 2 free parameters.
2. **Two-temperature torus** — hot + warm dust components inspired by
   SKIRTOR clumpy torus geometry. 4 free parameters.

Both return specific luminosity L_nu in Lsun/Hz. All functions are pure
JAX and JIT-compilable.

Physical picture: The torus intercepts a fraction of the disc luminosity
(the covering factor) and re-emits it thermally in the IR. The silicate
feature at 9.7 um produces opacity-dependent absorption/emission.

References
----------
- Nenkova et al. 2008, ApJ, 685, 147 (CLUMPY torus)
- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR)
- Draine 2003, ARA&A, 41, 241 (silicate opacity)
"""

import jax.numpy as jnp

# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27  # Planck constant [erg s]
_K_BOLTZ = 1.380649e-16  # Boltzmann constant [erg K^-1]
_C_LIGHT = 2.99792458e10  # Speed of light [cm s^-1]
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]
_ANGSTROM_CM = 1e-8  # Angstrom -> cm
_MICRON_ANGSTROM = 1e4  # Micron -> Angstrom

# Silicate feature wavelength
_LAMBDA_SI = 9.7 * _MICRON_ANGSTROM  # 9.7 um in Angstrom


# ===================================================================
# Planck function (numerically stable)
# ===================================================================


def _planck_lnu(
    nu: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    temperature : float
        Temperature [K].

    Returns
    -------
    array
        B_nu(T).
    """
    t_safe = jnp.maximum(temperature, 1.0)
    x = _H_PLANCK * nu / (_K_BOLTZ * t_safe)
    x_clip = jnp.clip(x, 0.0, 500.0)
    prefactor = 2.0 * _H_PLANCK * nu**3 / _C_LIGHT**2
    return prefactor / (jnp.exp(x_clip) - 1.0)


def _wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    return _C_LIGHT / (wavelength_angstrom * _ANGSTROM_CM)


# ===================================================================
# Model 1: Simple hot blackbody torus
# ===================================================================


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
        Specific luminosity L_nu [Lsun Hz^-1].
    """
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
    return l_nu_erg / _LSUN_ERG


# ===================================================================
# Model 2: Two-temperature torus (SKIRTOR-inspired)
# ===================================================================


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
        Specific luminosity L_nu [Lsun Hz^-1].
    """
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
    return l_nu_erg / _LSUN_ERG
