"""Dust attenuation curve library.

Each law is a pure JAX function that returns k(lambda) = A(lambda)/A(V),
the attenuation curve normalized at V-band (5500 Angstrom).

All functions take wavelength in Angstrom and return k(lambda).

Laws
----
- power_law: Simple power law (current default)
- calzetti: Calzetti et al. (2000) starburst curve
- kriek_conroy: Kriek & Conroy (2013) — Calzetti + UV bump + delta
- smc: Gordon et al. (2003) SMC Bar curve
- cardelli: Cardelli et al. (1989) MW curve with free R_V
- salim: Salim et al. (2018) modified Calzetti + bump (DSPS default)

References
----------
- Calzetti et al. 2000, ApJ, 533, 682
- Cardelli et al. 1989, ApJ, 345, 245
- Gordon et al. 2003, ApJ, 594, 279
- Kriek & Conroy 2013, ApJL, 775, L16
- Salim et al. 2018, ApJ, 859, 11
"""

from typing import Callable

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Drude profile (shared utility for UV bump)
# ---------------------------------------------------------------------------

def _drude_profile(
    wave_um: jnp.ndarray,
    x0: float = 0.2175,
    gamma: float = 0.035,
) -> jnp.ndarray:
    """Drude profile for the 2175 Angstrom UV bump.

    D(lambda) = (lambda * gamma)^2 / ((lambda^2 - x0^2)^2 + (lambda*gamma)^2)

    Parameters
    ----------
    wave_um : array
        Wavelength in microns.
    x0 : float
        Central wavelength in microns (default 0.2175 um = 2175 A).
    gamma : float
        Width in microns (default 0.035 um).

    Returns
    -------
    array
        Drude profile values.
    """
    return (
        (wave_um * gamma) ** 2
        / ((wave_um**2 - x0**2) ** 2 + (wave_um * gamma) ** 2)
    )


# ---------------------------------------------------------------------------
# 1. Power Law (current default)
# ---------------------------------------------------------------------------

@register_dust_law("power_law")
def power_law(
    wavelength: jnp.ndarray,
    n_slope: float = -0.7,
    **_kwargs,
) -> jnp.ndarray:
    """Simple power-law attenuation: k(lambda) = (lambda/5500)^n.

    This is the original Charlot & Fall (2000) wavelength dependence.
    """
    return (wavelength / 5500.0) ** n_slope


# ---------------------------------------------------------------------------
# 2. Calzetti et al. (2000)
# ---------------------------------------------------------------------------

@register_dust_law("calzetti")
def calzetti(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """Calzetti et al. (2000) starburst attenuation curve.

    k(lambda) from Table 3 of Calzetti et al. 2000.
    R_V = 4.05 (fixed).
    Valid range: 0.12 - 2.2 microns.
    """
    wave_um = wavelength / 1e4  # Angstrom -> micron
    x = 1.0 / wave_um  # inverse microns

    # Two polynomial regimes
    # 0.63 <= lambda <= 2.2 um
    k_ir = 2.659 * (-1.857 + 1.040 * x)
    # 0.12 <= lambda < 0.63 um
    k_uv = 2.659 * (
        -2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3
    )

    rv = 4.05
    # k'(lambda) + R_V gives total, then normalize to A(lambda)/A(V)
    k_prime = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    k_normalized = (k_prime + rv) / rv

    # Clip: outside valid range, extrapolate gently
    return jnp.clip(k_normalized, 0.0)


# ---------------------------------------------------------------------------
# 3. Kriek & Conroy (2013)
# ---------------------------------------------------------------------------

@register_dust_law("kriek_conroy")
def kriek_conroy(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 1.0,
    dust_delta: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Kriek & Conroy (2013) modified Calzetti curve.

    Calzetti base + UV bump (Drude at 2175 A) + power-law slope modification.
    This is the default in Prospector.

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.
    dust_bump_strength : float
        Amplitude of 2175 A UV bump (E_b). 0 = no bump. Default 1.0.
    dust_delta : float
        Power-law slope modification. 0 = pure Calzetti. Default 0.0.
    """
    wave_um = wavelength / 1e4

    # Start with Calzetti
    k_calz = calzetti(wavelength)

    # Add slope modification: multiply by (lambda/5500A)^delta
    slope_mod = (wavelength / 5500.0) ** dust_delta

    # Add UV bump: Drude profile at 2175 A
    bump = dust_bump_strength * _drude_profile(wave_um)

    return jnp.clip(k_calz * slope_mod + bump, 0.0)


# ---------------------------------------------------------------------------
# 4. SMC (Gordon et al. 2003)
# ---------------------------------------------------------------------------

@register_dust_law("smc")
def smc(
    wavelength: jnp.ndarray,
    **_kwargs,
) -> jnp.ndarray:
    """SMC Bar extinction curve (Pei 1992 parameterization).

    Steep UV rise, NO 2175 Angstrom bump.
    Common at high redshift.
    Normalized to A(lambda)/A(V).
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    # Piecewise fit from Gordon et al. 2003 / Pei 1992
    # UV portion (lambda < 0.3 um, x > 3.33)
    k_uv = 1.0 + 0.2 * (x - 3.33) + 0.25 * (x - 3.33) ** 2

    # Optical (0.3 - 1.0 um)
    # Linear interpolation anchored at A(V)=1 at 5500A
    k_opt = 1.0 + 1.39 * (x - 1.82)

    # IR (lambda > 1.0 um)
    k_ir = 0.6 * x**1.7

    k = jnp.where(
        wave_um < 0.3,
        k_uv,
        jnp.where(wave_um < 1.0, k_opt, k_ir),
    )
    return jnp.clip(k, 0.0)


# ---------------------------------------------------------------------------
# 5. Cardelli et al. (1989) MW
# ---------------------------------------------------------------------------

@register_dust_law("cardelli")
def cardelli(
    wavelength: jnp.ndarray,
    dust_Rv: float = 3.1,
    **_kwargs,
) -> jnp.ndarray:
    """Cardelli, Clayton & Mathis (1989) MW extinction with variable R_V.

    Returns A(lambda)/A(V).

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.
    dust_Rv : float
        Total-to-selective extinction ratio. Default 3.1 (standard MW).
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    # Infrared: 0.3 <= x <= 1.1 (lambda 0.91 - 3.3 um)
    a_ir = 0.574 * x**1.61
    b_ir = -0.527 * x**1.61

    # Optical/NIR: 1.1 <= x <= 3.3 (lambda 0.3 - 0.91 um)
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

    # UV: 3.3 <= x <= 8.0 (lambda 0.125 - 0.3 um)
    f_a = jnp.where(
        x >= 5.9,
        -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3,
        0.0,
    )
    f_b = jnp.where(
        x >= 5.9,
        0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3,
        0.0,
    )
    a_uv = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + f_a
    b_uv = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + f_b

    # Select regime
    a = jnp.where(
        x < 1.1,
        a_ir,
        jnp.where(x < 3.3, a_opt, a_uv),
    )
    b = jnp.where(
        x < 1.1,
        b_ir,
        jnp.where(x < 3.3, b_opt, b_uv),
    )

    # A(lambda)/A(V) = a(x) + b(x)/R_V
    return jnp.clip(a + b / dust_Rv, 0.0)


# ---------------------------------------------------------------------------
# 6. Salim et al. (2018) — modified Calzetti + bump (DSPS default)
# ---------------------------------------------------------------------------

@register_dust_law("salim")
def salim(
    wavelength: jnp.ndarray,
    dust_bump_strength: float = 0.0,
    dust_delta: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Salim et al. (2018) modified Calzetti attenuation curve.

    Same functional form as Kriek & Conroy but with the Salim+2018
    parameterization used by DSPS/Zacharegkas+2025.

    A(lambda) = (A_V / 4.05) * [k_calzetti(lambda) * (lambda/V)^delta + D(lambda)]

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.
    dust_bump_strength : float
        UV bump amplitude E_b. Default 0.0.
    dust_delta : float
        Power-law slope modification. Default 0.0.
    """
    # Identical to kriek_conroy in practice
    return kriek_conroy(
        wavelength,
        dust_bump_strength=dust_bump_strength,
        dust_delta=dust_delta,
    )
