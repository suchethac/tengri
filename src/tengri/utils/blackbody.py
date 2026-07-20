# SPDX-License-Identifier: BSD-3-Clause
"""The Planck blackbody function.

The single implementation of :math:`B_\\nu(T)` for the whole tree. It lives in
``utils/`` rather than inside a component because two independent subsystems
need it — the AGN torus/polar-dust closures and the dust IR emission closures —
and ``utils/`` is the layer below both. Importing it from either component
would invert the layering and couple the two.

Both historical spellings delegate here: :func:`tengri.components.agn._phys.planck_lnu`
(frequency argument) and :func:`tengri.components.dust.emission._physics.planck_bnu`
(wavelength argument).
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_CGS as _C_CGS,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
)

__all__ = ["planck_bnu_nu", "planck_bnu_wave"]

# x = h*nu/(k_B*T) is clamped to this interval.  At x = 500, expm1(x) ~ 1.4e217
# — finite in float64.  The clamp avoids both expm1 overflow and division by
# zero, and keeps gradients finite everywhere.
_X_MIN: float = 1e-10
_X_MAX: float = 500.0

# Temperature floor [K].  Guards 1/T at T -> 0.  Below this the Planck function
# is ~1e-221 (i.e. numerically zero) at every wavelength tengri models, so the
# floor is unobservable in value and only pins the gradient to zero.
_T_MIN: float = 1.0


def planck_bnu_nu(nu: jnp.ndarray, temperature: float) -> jnp.ndarray:
    r"""Planck blackbody spectral radiance at a given frequency.

    Parameters
    ----------
    nu : array_like, shape (n_freq,)
        Frequency. [Hz]
    temperature : float
        Blackbody temperature. Must be positive; clamped to >= 1 K. [K]

    Returns
    -------
    ndarray, shape (n_freq,)
        Spectral radiance :math:`B_\nu(T)`. [erg/s/cm^2/Hz/sr]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives. Safe under
    ``grad`` and ``vmap``.

    .. math::

        B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{e^{h\nu/k_B T} - 1}

    where :math:`h` is Planck's constant [erg s], :math:`\nu` is frequency [Hz],
    :math:`c` is the speed of light [cm/s], :math:`k_B` is Boltzmann's constant
    [erg/K], and :math:`T` is temperature [K].

    **Numerical stability**: arithmetic is forced to float64 because
    :math:`\nu^3` overflows float32 — at :math:`\lambda \sim 100` Angstrom,
    :math:`\nu \approx 3\times10^{17}` Hz so :math:`\nu^3 \approx 3\times10^{52}`,
    far beyond the float32 maximum of :math:`3.4\times10^{38}`. JAX weak-type
    promotion keeps float32 arrays float32 even when combined with Python float
    scalars, so the cast must be explicit even with x64 enabled globally.
    :math:`x = h\nu/k_B T` is clamped to [1e-10, 500] and :math:`T` to
    [1 K, inf).

    References
    ----------
    .. [1] M. Planck, "Zur Theorie des Gesetzes der Energieverteilung im
       Normalspektrum," Verhandlungen der Deutschen Physikalischen Gesellschaft,
       Vol. 2, pp. 237-245 (1900).
    """
    nu64 = jnp.asarray(nu, dtype=jnp.float64)
    t_safe = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float64), _T_MIN)
    x = jnp.clip(_H_PLANCK * nu64 / (_K_BOLTZ * t_safe), _X_MIN, _X_MAX)
    return 2.0 * _H_PLANCK * nu64**3 / _C_CGS**2 / jnp.expm1(x)


def planck_bnu_wave(wavelength_aa: jnp.ndarray, temperature: float) -> jnp.ndarray:
    r"""Planck blackbody spectral radiance at a given wavelength.

    Wavelength-argument spelling of :func:`planck_bnu_nu`, via
    :math:`\nu = c/\lambda`.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Angstrom]
    temperature : float
        Blackbody temperature. Must be positive; clamped to >= 1 K. [K]

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral radiance :math:`B_\nu(T)`. [erg/s/cm^2/Hz/sr]

    Notes
    -----
    **JIT-compatible**: yes. See :func:`planck_bnu_nu` for the equation and the
    numerical-stability contract.
    """
    wavelength_cm = jnp.asarray(wavelength_aa, dtype=jnp.float64) * _AA_TO_CM
    return planck_bnu_nu(_C_CGS / wavelength_cm, temperature)
