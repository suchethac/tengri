# SPDX-License-Identifier: BSD-3-Clause
"""The Planck blackbody function.

The single implementation of :math:`B_\\nu(T)` for the whole tree. It lives in
``utils/`` rather than inside a component because two independent subsystems
need it (the AGN torus/polar-dust closures and the dust IR emission closures)
and ``utils/`` is the layer below both. Importing it from either component
would invert the layering and couple the two.

Both historical spellings delegate here: ``tengri.components.agn._phys.planck_lnu``
(frequency argument) and ``tengri.components.dust.emission._physics.planck_bnu``
(wavelength argument).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_CGS as _C_CGS,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
)

__all__ = ["planck_bnu_nu", "planck_bnu_wave"]

# x = h*nu/(k_B*T) is clamped to this interval.  The ceiling is a plain
# constant, not a dtype-dependent one: the denominator below is ``-expm1(-x)``,
# which lives in (0, 1] and cannot overflow at any x in any dtype, while
# ``exp(-x)`` underflows to exactly 0.0: the true Wien limit.  Measured
# identical values AND gradients at x = 40, 60, 87, 90, 150, 400 in both
# dtypes with and without a dtype-aware cap, so the cap was inert (#1439).
# The floor is what still earns its keep: it bounds 1/x as x -> 0.
_X_MIN: float = 1e-10
_X_MAX: float = 500.0

# Temperature floor [K].  Guards 1/T at T -> 0.  Below this the Planck function
# is ~1e-221 (i.e. numerically zero) at every wavelength tengri models, so the
# floor is unobservable in value and only pins the gradient to zero.
_T_MIN: float = 1.0


@jax.custom_jvp
def _planck_core(nu_w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    r"""``2h nu (nu/c)**2 * e**-x / (1 - e**-x)``: the smooth part of :math:`B_\nu`.

    Split out from :func:`planck_bnu_nu` so the explicit derivative rule below
    covers *only* this, and the clamps on ``x`` and ``T`` stay ordinary
    ``jnp`` ops that autodiff handles exactly as it always has. Re-deriving a
    clamp's derivative by hand would mean matching ``lax.max``'s tie-breaking
    (it splits the tangent evenly at equality); a second source of truth that
    can disagree with the first, and did: masking on ``T > _T_MIN`` severed the
    derivative at exactly ``T = _T_MIN`` where the primal passes it through.

    Parameters
    ----------
    nu_w : array_like, shape (n_freq,)
        Frequency, already cast to the working dtype. [Hz]
    x : array_like, shape (n_freq,)
        Already-clamped exponent :math:`h\nu/k_B T`. [dimensionless]

    Returns
    -------
    ndarray, shape (n_freq,)
        Spectral radiance :math:`B_\nu`. [erg/s/cm^2/Hz/sr]
    """
    # Algebraically 2h·nu³/c², but never forming nu³: at λ ~ 100 Å that
    # intermediate is ~2.7e52, far past float32's 3.4e38, while B_nu itself
    # peaks at ~8e-12 and is perfectly representable. Grouping as nu·(nu/c)²
    # caps the largest intermediate at ~1e12; identical in float64 to ~4e-16.
    #
    # ``1/expm1(x)`` is spelled ``exp(-x) / -expm1(-x)``: the same number
    # (``1/(e^x - 1) == e^-x/(1 - e^-x)``), but with a denominator that cannot
    # overflow (#1439). Division's derivative needs the denominator *squared*:
    # ``expm1(x)**2`` passes float32's 3.4e38 once x > ~44, so with a large
    # incoming cotangent the reverse pass formed ``inf/inf`` and returned NaN,
    # while the forward value stayed perfectly healthy, because a saturated
    # denominator still gives the right Wien-tail limit. A dtype-aware clamp
    # used to sit above, sized on ``expm1``'s *forward* overflow (~88.7 in
    # float32): but the derivative breaks at half that, so no setting of that
    # clamp could ever have covered this. That is why the fix is the rewrite
    # here and not a tighter bound; with it in place the clamp measured inert
    # and was removed.
    #
    # The rewritten denominator ``1 - e^-x`` lives in (0, 1], so its square is
    # bounded by 1 at every x and in every dtype: the failure mode is removed
    # rather than bounded. Accuracy is preserved at both ends: for small x,
    # ``-expm1(-x) -> x`` is exactly what ``expm1`` exists to compute.
    #
    # Measured: float64 gradients bit-identical and values within 1.8e-16; in
    # float32 the Wien-tail gradient goes from a silent ``-0.0`` to the correct
    # value (x=44: -7.78e-08, x=60: -8.76e-15, x=87: -1.65e-26), each matching
    # float64.
    return 2.0 * _H_PLANCK * nu_w * (nu_w / _C_CGS) ** 2 * jnp.exp(-x) / -jnp.expm1(-x)


@_planck_core.defjvp
def _planck_core_jvp(
    primals: tuple[jnp.ndarray, jnp.ndarray],
    tangents: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Analytic tangent for :func:`_planck_core`, grouped to stay in float32.

    .. math::

        \frac{\partial B_\nu}{\partial \nu}\bigg|_x = \frac{3 B_\nu}{\nu},
        \qquad
        \frac{\partial B_\nu}{\partial x} = -\frac{B_\nu}{1 - e^{-x}},

    with :math:`B_\nu` in [erg/s/cm^2/Hz/sr], :math:`\nu` in [Hz] and
    :math:`x = h\nu/k_B T` dimensionless. The first is the :math:`\nu^3`
    prefactor's own derivative at fixed :math:`x`; the second follows from
    :math:`\frac{d}{dx}\frac{e^{-x}}{1-e^{-x}} = -\frac{e^{-x}}{(1-e^{-x})^2}`.
    Callers reach :math:`\partial B_\nu/\partial T` and the rest of
    :math:`\partial B_\nu/\partial \nu` through the chain rule on ``x``, which
    autodiff still supplies.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes, in both autodiff modes.

    Why a rule at all. Autodiff differentiates the primal's division by
    :math:`1-e^{-x}` with the quotient rule, which needs that denominator
    *squared*. With a caller's ~1e30 cotangent arriving before the ~3.97e-13
    prefactor, :math:`g/(1-e^{-x})^2` reaches 4.8e39 (past float32's 3.4e38)
    for a true answer of 1.9e27. Stating the derivative means the square is
    never formed by autodiff at all, which is what source-level regrouping
    could not achieve: three groupings were measured and all three still
    overflowed, because XLA reassociates.

    Why ``custom_jvp`` and not the ``custom_vjp`` #1439 originally prescribed.
    Three reasons, in order of what they cost:

    1. A ``custom_vjp`` is **opaque to forward mode**: ``jvp`` raises
       ``TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp
       function``. Forward mode already computes this gradient *correctly* in
       pure float32 (measured 2.761298e+16, matching float64, where reverse
       mode returned ``inf``), and geoVI and ``inference/preconditioning.py``
       both differentiate forward. The ``custom_vjp`` spelling would trade a
       silent NaN in one mode for a hard error in a mode that works; the same
       regression this branch already had to undo for
       ``tengri.components.stellar.component._mass_scale_lnu``.
    2. A ``custom_vjp`` would have to reduce ``g * dB/dT`` back to the
       temperature's shape by hand, because ``temperature`` is a scalar per
       ring under ``vmap``. That hand-written reduction was the stated reason
       #1439 stopped short of landing a fix. Transposing a ``custom_jvp``
       derives it instead, so the risk does not arise.
    3. One rule serves both modes, so the two cannot disagree.

    ``optimization_barrier`` supplies the ordering guarantee that
    ``custom_vjp``'s opacity would have: a ``custom_jvp``'s transpose is
    inlined into the backward jaxpr, and without it XLA is free to reassociate
    the cotangent back inside the coefficient and re-form the overflow. See
    ``tengri.components.stellar.component._mass_scale_lnu``, where the
    same pairing is measured.
    """
    nu_w, x = primals
    d_nu, d_x = tangents
    b_nu = _planck_core(nu_w, x)

    # ``B/(1-e^-x)`` is the whole point: it is the only place a squared
    # denominator appears, and forming it HERE: one value, ~2e-3 at the
    # failing disc ring, before any cotangent exists: is what keeps it in
    # range. ``B/nu`` is likewise spelled with one power of nu removed rather
    # than divided out, so nu never reaches a denominator: at nu = 0 the naive
    # ``3*b_nu/nu`` is 0/0 = NaN where the true derivative is 0.
    b_over_d = b_nu / -jnp.expm1(-x)
    b_over_nu = 2.0 * _H_PLANCK * (nu_w / _C_CGS) ** 2 * jnp.exp(-x) / -jnp.expm1(-x)

    d_b_d_nu, d_b_d_x = jax.lax.optimization_barrier((3.0 * b_over_nu, -b_over_d))
    return b_nu, d_b_d_nu * d_nu + d_b_d_x * d_x


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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives. Safe under
    ``grad`` and ``vmap``.

    .. math::

        B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{e^{h\nu/k_B T} - 1}

    where :math:`h` is Planck's constant [erg s], :math:`\nu` is frequency [Hz],
    :math:`c` is the speed of light [cm/s], :math:`k_B` is Boltzmann's constant
    [erg/K], and :math:`T` is temperature [K].

    **Numerical stability**: the prefactor is grouped as
    :math:`2h\nu(\nu/c)^2` so :math:`\nu^3` is never formed. Written out, that
    intermediate reaches :math:`\approx 3\times10^{52}` at
    :math:`\lambda \sim 100` Angstrom; far beyond the float32 maximum of
    :math:`3.4\times10^{38}`: even though :math:`B_\nu` itself peaks around
    1e-11 and is representable. This grouping caps the largest intermediate at
    :math:`\sim 10^{12}` and is identical in float64 to ~4e-16 relative.

    Arithmetic follows ``jnp.result_type(float)`` (float64 under x64, float32
    without it) rather than being forced to float64: a hard cast is silently
    truncated under ``jax.enable_x64(False)``, so it would protect nothing in
    the configuration that needs it. JAX weak-type promotion keeps float32
    arrays float32 even when combined with Python float scalars, so the cast
    must still be explicit.

    :math:`x = h\nu/k_B T` is clamped to ``[1e-10, 500]`` and :math:`T` to
    [1 K, inf). The reciprocal is spelled :math:`e^{-x}/(1 - e^{-x})` rather
    than :math:`1/(e^x - 1)`; the two are the same number, but the former's
    denominator lies in (0, 1], so neither it nor the *square* of it that the
    reverse pass forms can overflow in any dtype. The exponent is likewise
    grouped :math:`(h/k_B)\,\nu/T` rather than :math:`h\nu/(k_B T)`, which
    keeps the reverse pass's :math:`1/T^2` in range where :math:`1/(k_B T)^2`
    was not. Both rewrites are exact identities and leave float64 gradients
    bit-identical; they exist because a guard sized for where the *value*
    breaks is off by a square root from where the *derivative* breaks
    (#1206, #1439).

    Those two rewrites are not sufficient on their own, because the last
    overflow is not inside this function at all. A caller multiplies
    :math:`B_\nu` by a ring area or a template normalization ~1e30, so the
    reverse pass arrives carrying a cotangent that large and forms
    :math:`g/(1-e^{-x})^2`; 4.8e39 at a real disc ring: *before* the tiny
    :math:`2h\nu(\nu/c)^2` prefactor (3.97e-13) can bring it back. The correct
    answer, 1.9e27, is perfectly representable; only the intermediate is not,
    and the two factors live on opposite sides of the function boundary, so no
    rewrite of this expression can reach it. Three source-level regroupings
    were measured and all three still overflowed, because XLA reassociates.

    The derivative of the smooth part is therefore supplied explicitly, as the
    :func:`jax.custom_jvp` rule on :func:`_planck_core`. Defining it means the
    quotient rule's :math:`(1-e^{-x})^2` is never formed by autodiff at all.
    The two clamps stay outside that rule, as ordinary ``jnp`` ops, so their
    derivatives are unchanged.
    """
    # Work at the session's working precision. ``result_type(float)`` is
    # float64 under x64 and float32 without it; a hard ``dtype=jnp.float64``
    # is silently truncated back to float32 under ``jax.enable_x64(False)``
    # (JAX warns and carries on), so it protected nothing in exactly the
    # configuration it was written for (#1206).
    dtype = jnp.result_type(float)
    nu_w = jnp.asarray(nu, dtype=dtype)
    t_safe = jnp.maximum(jnp.asarray(temperature, dtype=dtype), _T_MIN)

    # Grouped as ``(h/k)·nu / T``, NOT ``h·nu / (k·T)``: associativity, but the
    # reverse pass is not associative in float32 (#1439). Division's derivative
    # w.r.t. its denominator is ``-g·A/den**2``. Spelled ``h·nu / (k·T)`` the
    # denominator is ``k·T``, so that intermediate is ``-g·(h·nu)/(k·T)**2``,
    # measured 2e40 for a disc ring, past float32's 3.4e38, and the small ``k``
    # that would bring it back into range is only applied *afterwards*, by which
    # point it is ``inf``. With ``k`` folded into the numerator the denominator is
    # just ``T``, the same intermediate is ``-g·(h·nu/k)/T**2``, and nothing
    # leaves range. Measured on one ring: gradient ``inf`` -> 2.7565e+24, the
    # float64 answer.
    #
    # Same regrouping idea as ``nu·(nu/c)**2`` in the core, one level down:
    # there it keeps a *forward* intermediate in range, here a *backward* one.
    x = jnp.clip((_H_PLANCK / _K_BOLTZ) * nu_w / t_safe, _X_MIN, _X_MAX)
    return _planck_core(nu_w, x)


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
