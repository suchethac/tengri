# SPDX-License-Identifier: BSD-3-Clause
"""Shared physics utilities for dust IR emission models.

Pure, dependency-light helpers used by every emission closure: the Planck
function, the da Cunha et al. (2013) CMB heating correction, and the
energy-balance absorbed-luminosity integrals. This module is a leaf: it
imports only ``jax.numpy`` and physical constants, so closure modules
(``analytic/``, ``templates/``) and the ``emission`` facade can all import
from it without an import cycle.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.blackbody import planck_bnu_wave as _planck_bnu_wave
from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_AA as _C_AA_PER_S,
    C_CGS as _C_CGS,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZMANN,
)

# ── Utility: frequency integral ───────────────────────────────────


def integrate_lnu_over_nu(
    L_nu: jnp.ndarray,
    wave_aa: jnp.ndarray,
) -> jnp.ndarray:
    r"""JIT-friendly trapezoid of :math:`\int L_\nu \, d\nu`.

    Uses the identity
    :math:`\int L_\nu \, d\nu = \int (\nu L_\nu)\, d\ln\nu` and the
    transformation :math:`d\ln\nu = -d\ln\lambda`, with the two sign
    flips canceling for an increasing-:math:`\lambda` grid:

    .. math::

        \int_{\nu_{\min}}^{\nu_{\max}} L_\nu \, d\nu
        \;=\; \int_{\lambda_{\min}}^{\lambda_{\max}}
              (\nu L_\nu)\, d\ln\lambda \, .

    Parameters
    ----------
    L_nu: array_like, shape ``(..., n_wave_aa)``
        :math:`L_\nu` in [erg/s/Hz] (or any per-Hz unit).
    wave_aa: array_like, shape ``(n_wave_aa,)``
        Wavelength grid in Angstrom; strictly increasing.

    Returns
    -------
    ndarray, shape ``(...,)``
        :math:`\int L_\nu \, d\nu` in [erg/s] (or the matching unit).

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp.trapezoid`` over a static
    axis.  **Gradient-safe**: yes.

    The single canonical implementation: consolidated from bit-identical
    per-module copies in ``draine2021_pah.py`` and ``astrodust_hd23.py``
    (2026-07).
    """
    nu = _C_AA_PER_S / wave_aa
    nu_lnu = nu * L_nu
    return jnp.trapezoid(nu_lnu, jnp.log(wave_aa), axis=-1)


# ── Utility: Planck function ──────────────────────────────────────


def planck_bnu(
    wavelength_aa: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    r"""Planck function B_ν(T) evaluated at given wavelengths.

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Wavelength grid. [Å]
    temperature: float
        Blackbody temperature. [K]

    Returns
    -------
    ndarray, shape (n_wave,)
        Planck brightness. [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The Planck function is:

    .. math::

        B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{\exp(h\nu / k_B T) - 1}

    where :math:`\nu = c / \lambda` is the frequency, :math:`h` is Planck's constant,
    :math:`k_B` is Boltzmann's constant, and :math:`c` is the speed of light.

    Delegates to :func:`tengri.utils.blackbody.planck_bnu_wave`, the single
    implementation shared with the AGN closures. See it for the numerical
    stability contract.
    """
    return _planck_bnu_wave(wavelength_aa, temperature)


# ── CMB heating correction (da Cunha+2013) ────────────────────────

_T_CMB_0 = 2.725  # CMB temperature at z=0 (K)


def cmb_corrected_temperature(
    T_dust: float,
    redshift: float,
    beta_ir: float = 1.6,
) -> float:
    r"""Effective dust temperature including CMB heating.

    At high redshift the CMB sets a temperature floor on dust grains.
    The effective equilibrium temperature is (da Cunha et al. 2013).

    Parameters
    ----------
    T_dust: float
        Intrinsic dust temperature (what the galaxy would have at z=0 in isolation). [K]
    redshift: float
        Source redshift. [dimensionless]
    beta_ir: float
        Dust emissivity index. [dimensionless] Default: 1.6.

    Returns
    -------
    float
        Effective dust temperature including CMB heating. [K]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The effective temperature is:

    .. math::

        T_{\rm eff} = \left[T_{\rm dust}^{4+\beta} + T_{\rm CMB}(z)^{4+\beta}
        - T_{\rm CMB}(z=0)^{4+\beta}\right]^{1/(4+\beta)}

    where :math:`T_{\rm CMB}(z) = T_{\rm CMB,0} (1 + z)` with :math:`T_{\rm CMB,0} = 2.725` K.

    """
    exponent = 4.0 + beta_ir
    T_cmb_z = _T_CMB_0 * (1.0 + redshift)
    # Clamp T_dust to positive values before raising to a fractional exponent.
    # Negative T_dust (possible during unconstrained sampling) would give NaN.
    T_dust_safe = jnp.maximum(T_dust, 1.0)
    inner = jnp.maximum(T_dust_safe**exponent + T_cmb_z**exponent - _T_CMB_0**exponent, 0.0)
    T_eff = inner ** (1.0 / exponent)
    return T_eff


def cmb_contrast_factor(
    wavelength_aa: jnp.ndarray,
    T_eff: float,
    redshift: float,
) -> jnp.ndarray:
    r"""Flux suppression factor from observing dust against the CMB.

    The observed flux is reduced because the galaxy's dust emission is
    measured against the CMB background (da Cunha et al. 2013).

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Wavelength grid. [Å]
    T_eff: float
        CMB-corrected effective dust temperature. [K]
    redshift: float
        Source redshift. [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative contrast factor in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The contrast factor is:

    .. math::

        C(\lambda) = 1 - \frac{B_\nu(T_{\rm CMB}(z))}{B_\nu(T_{\rm eff})}

    Since :math:`T_{\rm eff} > T_{\rm CMB}(z)`, we have :math:`0 \leq C(\lambda) \leq 1`.

    """
    T_cmb_z = _T_CMB_0 * (1.0 + redshift)

    # Compute the Planck ratio B_nu(T_cmb)/B_nu(T_eff) stably.
    # Since both share the same nu^3 prefactor, the ratio simplifies to
    #   (exp(x_eff) - 1) / (exp(x_cmb) - 1)
    # where x = h*nu/(k*T).  For x >> 1 this approaches exp(x_eff - x_cmb)
    # which is safe because x_cmb > x_eff (T_eff > T_cmb).
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    x_eff = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_eff), 0.0, 500.0)
    x_cmb = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_cmb_z), 0.0, 500.0)

    # Ratio = (exp(x_eff) - 1) / (exp(x_cmb) - 1)
    # Use log-space: log(ratio) = log(expm1(x_eff)) - log(expm1(x_cmb))
    # For large x, expm1(x) ~ exp(x), so log(expm1(x)) ~ x.
    log_expm1_eff = jnp.where(
        x_eff > 30.0, x_eff, jnp.log(jnp.expm1(jnp.clip(x_eff, 1e-10, 30.0)))
    )
    log_expm1_cmb = jnp.where(
        x_cmb > 30.0, x_cmb, jnp.log(jnp.expm1(jnp.clip(x_cmb, 1e-10, 30.0)))
    )

    # B_cmb/B_eff = exp(log_expm1_eff - log_expm1_cmb)
    # Since T_eff >= T_cmb, x_cmb >= x_eff, so the exponent is <= 0
    # and the ratio is in [0, 1].
    log_ratio = log_expm1_eff - log_expm1_cmb
    ratio = jnp.exp(jnp.clip(log_ratio, -100.0, 0.0))

    return jnp.clip(1.0 - ratio, 0.0, 1.0)


# ── Energy balance ────────────────────────────────────────────────


def compute_absorbed_luminosity(
    wavelength_aa: jnp.ndarray,
    L_nu_intrinsic: jnp.ndarray,
    transmission: jnp.ndarray,
) -> float:
    r"""Compute total luminosity absorbed by dust.

    Integrates (1 - transmission) × L_nu over frequency to get the total
    absorbed energy, which must be re-emitted in the IR (energy balance).

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å] Must be sorted ascending.
    L_nu_intrinsic: array_like, shape (n_wave,)
        Intrinsic (dust-free) luminosity density. [Lsun Hz⁻¹]
    transmission: array_like, shape (n_wave,)
        Dust transmission fraction in [0, 1]. For age-dependent models
        this should be the SFH-weighted effective transmission.

    Returns
    -------
    float
        Total absorbed luminosity. [Lsun]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The absorbed luminosity is:

    .. math::

        L_{\rm absorbed} = \int [1 - T(\lambda)] L_\nu(\lambda) d\nu

    where the integral is over frequency (ν is descending as λ is ascending).

    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # descending (since wave is ascending)

    absorbed_Lnu = (1.0 - transmission) * L_nu_intrinsic

    # Integrate over frequency: nu is descending, negate for positive result
    return -jnp.trapezoid(absorbed_Lnu, nu)


def compute_absorbed_luminosity_from_tau(
    wavelength_aa: jnp.ndarray,
    L_nu_intrinsic: jnp.ndarray,
    tau_lambda: jnp.ndarray,
) -> float:
    r"""Compute total absorbed luminosity from optical depth.

    Convenience wrapper when you have τ(λ) rather than transmission T(λ).

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å] Must be sorted ascending.
    L_nu_intrinsic: array_like, shape (n_wave,)
        Intrinsic luminosity density. [Lsun Hz⁻¹]
    tau_lambda: array_like, shape (n_wave,)
        Optical depth as a function of wavelength. [dimensionless]

    Returns
    -------
    float
        Total absorbed luminosity. [Lsun]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Internally converts τ(λ) to transmission via T(λ) = exp(−τ(λ))
    then calls ``compute_absorbed_luminosity``.

    """
    transmission = jnp.exp(-tau_lambda)
    return compute_absorbed_luminosity(wavelength_aa, L_nu_intrinsic, transmission)
