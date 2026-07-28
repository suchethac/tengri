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
from tengri.utils.scale import max_finite_exponent

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

    **Numerical stability**: the prefactor is grouped as
    :math:`2h\nu(\nu/c)^2` so :math:`\nu^3` is never formed. Written out, that
    intermediate reaches :math:`\approx 3\times10^{52}` at
    :math:`\lambda \sim 100` Angstrom — far beyond the float32 maximum of
    :math:`3.4\times10^{38}` — even though :math:`B_\nu` itself peaks around
    1e-11 and is representable. This grouping caps the largest intermediate at
    :math:`\sim 10^{12}` and is identical in float64 to ~4e-16 relative.

    Arithmetic follows ``jnp.result_type(float)`` (float64 under x64, float32
    without it) rather than being forced to float64: a hard cast is silently
    truncated under ``jax.enable_x64(False)``, so it would protect nothing in
    the configuration that needs it. JAX weak-type promotion keeps float32
    arrays float32 even when combined with Python float scalars, so the cast
    must still be explicit.

    :math:`x = h\nu/k_B T` is clamped to ``[1e-10, min(500,
    max_finite_exponent())]`` and :math:`T` to [1 K, inf). The upper clamp
    follows the dtype because ``expm1`` overflows above :math:`x \approx 88.7`
    in float32; a saturated denominator gives the correct forward limit
    (:math:`B_\nu \to 0`) but a NaN *gradient* (#1206).

    References
    ----------
    .. [1] M. Planck, "Zur Theorie des Gesetzes der Energieverteilung im
       Normalspektrum," Verhandlungen der Deutschen Physikalischen Gesellschaft,
       Vol. 2, pp. 237-245 (1900).
    """
    # Work at the session's working precision. ``result_type(float)`` is
    # float64 under x64 and float32 without it; a hard ``dtype=jnp.float64``
    # is silently truncated back to float32 under ``jax.enable_x64(False)``
    # (JAX warns and carries on), so it protected nothing in exactly the
    # configuration it was written for (#1206).
    dtype = jnp.result_type(float)
    nu_w = jnp.asarray(nu, dtype=dtype)
    t_safe = jnp.maximum(jnp.asarray(temperature, dtype=dtype), _T_MIN)

    # ``_X_MAX`` is 500, and expm1(500) = 1.4e217 — finite in float64 but inf
    # in float32, which overflows above x ~ 88.7. A saturated denominator
    # still gives the right forward value (B_nu -> 0, the Wien-tail limit) but
    # its gradient is inf/inf = NaN, so a fit would fail where the forward pass
    # looked healthy. Cap at the dtype's own limit; float64 is unaffected.
    # Grouped as ``(h/k)·nu / T``, NOT ``h·nu / (k·T)`` — associativity, but the
    # reverse pass is not associative in float32 (#1439). Division's derivative
    # w.r.t. its denominator is ``-g·A/den**2``. Spelled ``h·nu / (k·T)`` the
    # denominator is ``k·T``, so that intermediate is ``-g·(h·nu)/(k·T)**2`` —
    # measured 2e40 for a disc ring, past float32's 3.4e38 — and the small ``k``
    # that would bring it back into range is only applied *afterwards*, by which
    # point it is ``inf``. With ``k`` folded into the numerator the denominator is
    # just ``T``, the same intermediate is ``-g·(h·nu/k)/T**2``, and nothing
    # leaves range. Measured on one ring: gradient ``inf`` -> 2.7565e+24, the
    # float64 answer.
    #
    # Same regrouping idea as ``nu·(nu/c)**2`` below, one level down: there it
    # keeps a *forward* intermediate in range, here a *backward* one.
    x = jnp.clip(
        (_H_PLANCK / _K_BOLTZ) * nu_w / t_safe, _X_MIN, min(_X_MAX, max_finite_exponent())
    )

    # Algebraically 2h·nu³/c², but never forming nu³: at λ ~ 100 Å that
    # intermediate is ~2.7e52, far past float32's 3.4e38, while B_nu itself
    # peaks at ~8e-12 and is perfectly representable. Grouping as nu·(nu/c)²
    # caps the largest intermediate at ~1e12; identical in float64 to ~4e-16.
    #
    # ``1/expm1(x)`` is spelled ``exp(-x) / -expm1(-x)`` — the same number
    # (``1/(e^x - 1) == e^-x/(1 - e^-x)``), but with a denominator that cannot
    # overflow (#1439). Division's derivative needs the denominator *squared*:
    # ``expm1(x)**2`` passes float32's 3.4e38 once x > ~44, so with a large
    # incoming cotangent the reverse pass formed ``inf/inf`` and returned NaN —
    # while the forward value stayed perfectly healthy, because a saturated
    # denominator still gives the right Wien-tail limit. The clamp above bounds
    # x by ``expm1``'s *forward* overflow (~88.7 in float32); the derivative
    # needs half that, so the guard could not have covered it.
    #
    # The rewritten denominator ``1 - e^-x`` lives in (0, 1], so its square is
    # bounded by 1 at every x and in every dtype — the failure mode is removed
    # rather than bounded. Accuracy is preserved at both ends: for small x,
    # ``-expm1(-x) -> x`` is exactly what ``expm1`` exists to compute.
    #
    # Measured: float64 gradients bit-identical and values within 1.8e-16; in
    # float32 the Wien-tail gradient goes from a silent ``-0.0`` to the correct
    # value (x=44: -7.78e-08, x=60: -8.76e-15, x=87: -1.65e-26), each matching
    # float64.
    return 2.0 * _H_PLANCK * nu_w * (nu_w / _C_CGS) ** 2 * jnp.exp(-x) / -jnp.expm1(-x)


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
    wavelength_cm = jnp.asarray(wavelength_aa, dtype=jnp.result_type(float)) * _AA_TO_CM
    return planck_bnu_nu(_C_CGS / wavelength_cm, temperature)
