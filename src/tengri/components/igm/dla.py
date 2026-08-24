# SPDX-License-Identifier: BSD-3-Clause
"""Damped Lyman-alpha (DLA) absorption from individual foreground absorbers.

Computes transmission through a DLA system with known neutral hydrogen
column density N_HI, temperature T, and turbulent broadening b_turb.
The absorption profile is a Voigt function centered on Ly-alpha at the
absorber redshift.

This is distinct from the *statistical* DLA contribution in ``igm.py``
(Inoue+2014), which models the mean opacity from all DLA systems along
a random sightline.  This module handles a *specific* absorber: e.g.,
when a spectrum shows a clear damped trough.

Physics
-------
The Ly-alpha scattering cross-section is:

    σ(ν) = (√π e² f) / (m_e c Δν_D) × H(a, x)

where H(a, x) is the Voigt–Hjerting function, a = Γ/(4π Δν_D) is the
damping parameter, x = (ν − ν₀)/Δν_D is the dimensionless frequency
offset, and Δν_D is the thermal + turbulent Doppler width.

The Voigt function approximation follows Tepper-García (2006, MNRAS 369,
2025), with a first-order asymmetry correction from Lee (2013).

All functions are pure JAX (JIT-compilable, differentiable).

Parameters
----------

- ``dla_log_n_hi``: log10(N_HI / cm⁻²), typically 19–22.
- ``dla_z``: redshift of the absorber.
- ``dla_temp``: gas temperature [K], typically fixed at 10⁴ K.
- ``dla_b_turb``: turbulent broadening [km/s], typically 0–30.

References
----------

- Tepper-García 2006, MNRAS, 369, 2025 (Voigt approximation)
- Lee 2013, ApJ, 773, 120 (asymmetry correction)
- Dijkstra 2014, PASA, 31, e040 (Ly-alpha cross-section review)

"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.utils.physics_constants import (
    C_CGS,
    E_CHARGE_ESU,
    K_BOLTZ,
    M_ELECTRON,
    M_PROTON,
)

# ── Ly-alpha atomic data ──────────────────────────────────────────

_F_LYA: float = 0.4162
"""Oscillator strength of the Ly-alpha (2p → 1s) transition."""

_A_LYA: float = 6.265e8
"""Einstein A coefficient for Ly-alpha [s⁻¹]."""

_WL_LYA: float = 1215.6701
"""Ly-alpha rest wavelength [Å]."""

_NU_LYA: float = C_CGS / (_WL_LYA * 1e-8)
"""Ly-alpha rest frequency [Hz]."""

_K_LYA: float = _F_LYA * jnp.sqrt(jnp.pi) * E_CHARGE_ESU**2 / (M_ELECTRON * C_CGS)
"""Cross-section prefactor K = √π e² f / (m_e c) [cm² Hz]."""


# ── Doppler width ─────────────────────────────────────────────────


def _deltanu_doppler(temp: float, b_turb_kms: float) -> float:
    """Thermal + turbulent Doppler frequency width [Hz].

    Parameters
    ----------
    temp: float
        Gas temperature [K].
    b_turb_kms: float
        Turbulent broadening parameter [km/s].

    Returns
    -------
    float
        Δν_D [Hz].
    """
    b_turb_cgs = b_turb_kms * 1e5  # km/s → cm/s
    v_doppler = jnp.sqrt(2.0 * K_BOLTZ * temp / M_PROTON + b_turb_cgs**2)
    return _NU_LYA * v_doppler / C_CGS


# ── Voigt profile (Tepper-García 2006) ────────────────────────────


def _voigt_tepper_garcia(
    x: jnp.ndarray,
    a: float,
) -> jnp.ndarray:
    """Voigt–Hjerting function H(a, x) via Tepper-García (2006).

    Accurate to < 10⁻⁴ relative error for all (a, x) relevant to
    DLA absorption (a ~ 10⁻⁴ to 10⁻²).

    Parameters
    ----------
    x: array
        Dimensionless frequency offset (ν − ν₀) / Δν_D.
    a: float
        Voigt damping parameter Γ / (4π Δν_D).

    Returns
    -------
    array
        H(a, x) ≈ exp(-x²) + correction term.
    """
    x2 = x**2
    z = (x2 - 0.855) / (x2 + 3.42)
    # Safe gradient: pre-compute x2 under where mask to avoid huge gradients
    # through 1/(x2 + 1e-30) when x2 → 0 in the masked-off branch.
    x2_safe = jnp.where(z > 0.0, x2 + 1e-30, 1.0)
    q = jnp.where(
        z > 0.0,
        z
        * (1.0 + 21.0 / x2_safe)
        * a
        / (jnp.pi * (x2 + 1.0))
        * (0.1117 + z * (4.421 + z * (5.674 * z - 9.207))),
        0.0,
    )
    return jnp.sqrt(jnp.pi) * q + jnp.exp(-x2)


# ── Cross-section ─────────────────────────────────────────────────


def _sigma_lya(
    x: jnp.ndarray,
    temp: float,
    b_turb_kms: float,
) -> jnp.ndarray:
    """Ly-alpha scattering cross-section σ(x) [cm²].

    Includes the Tepper-García (2006) Voigt profile and the first-order
    asymmetry correction from Lee (2013).

    Parameters
    ----------
    x: array
        Dimensionless frequency offset.
    temp: float
        Gas temperature [K].
    b_turb_kms: float
        Turbulent broadening [km/s].

    Returns
    -------
    array
        σ [cm²] at each frequency offset.
    """
    dnu = _deltanu_doppler(temp, b_turb_kms)
    a = _A_LYA / (4.0 * jnp.pi * dnu)
    sigma = _K_LYA / dnu * _voigt_tepper_garcia(x, a)
    # Lee (2013) asymmetry correction
    sigma = sigma * (1.0 - 1.792 * x * dnu / _NU_LYA)
    return sigma


# ── Public API ────────────────────────────────────────────────────


@jax.jit
def dla_transmission(
    wave_rest: jnp.ndarray,
    log_n_hi: float,
    temp: float = 1e4,
    b_turb_kms: float = 0.0,
) -> jnp.ndarray:
    """Transmission through a DLA absorber at rest-frame wavelengths.

    Parameters
    ----------
    wave_rest: array, shape (n_wave,)
        Rest-frame wavelength [Å] in the absorber frame
        (i.e., already de-redshifted to the absorber's rest frame).
    log_n_hi: float
        log10(N_HI / cm⁻²). Typical DLA: 20.3–22.
    temp: float
        Gas temperature [K]. Default 10⁴ K (warm neutral medium).
    b_turb_kms: float
        Turbulent broadening [km/s]. Default 0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission T = exp(−τ) [dimensionless, 0–1].

    Notes
    -----
    **JIT-compatible**: yes, pure JAX function with ``@jax.jit`` decorator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import dla_transmission
    >>> wave = jnp.linspace(1100.0, 1300.0, 200)
    >>> T = dla_transmission(wave, log_n_hi=21.0)
    >>> T.shape
    (200,)
    """
    nu = C_CGS / (wave_rest * 1e-8)
    dnu = _deltanu_doppler(temp, b_turb_kms)
    x = (nu - _NU_LYA) / dnu
    sigma = _sigma_lya(x, temp, b_turb_kms)
    n_hi = 10.0**log_n_hi
    tau = n_hi * sigma
    return jnp.exp(-jnp.clip(tau, min=0.0))


@jax.jit
def dla_transmission_obs(
    wave_obs: jnp.ndarray,
    z_dla: float,
    log_n_hi: float,
    temp: float = 1e4,
    b_turb_kms: float = 0.0,
) -> jnp.ndarray:
    """Transmission through a DLA absorber at observed wavelengths.

    Convenience wrapper that de-redshifts ``wave_obs`` to the absorber
    rest frame before computing the Voigt absorption.

    Parameters
    ----------
    wave_obs: array, shape (n_wave,)
        Observed-frame wavelength [Å].
    z_dla: float
        Redshift of the DLA absorber.
    log_n_hi: float
        log10(N_HI / cm⁻²).
    temp: float
        Gas temperature [K]. Default 10⁴ K.
    b_turb_kms: float
        Turbulent broadening [km/s]. Default 0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission T = exp(−τ) [dimensionless, 0–1].

    Notes
    -----
    **JIT-compatible**: yes, pure JAX function with ``@jax.jit`` decorator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import dla_transmission_obs
    >>> wave_obs = jnp.linspace(1100.0 * 1.5, 1300.0 * 1.5, 200)
    >>> T = dla_transmission_obs(wave_obs, z_dla=0.5, log_n_hi=21.0)
    >>> T.shape
    (200,)
    """
    wave_rest = wave_obs / (1.0 + z_dla)
    return dla_transmission(wave_rest, log_n_hi, temp, b_turb_kms)
