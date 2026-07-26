# SPDX-License-Identifier: BSD-3-Clause
"""Range-safe application of large log10 scale factors (float32 feasibility).

Physical scales in this code (``mass_scale`` ~ 1e42 erg/s, ``d_L²`` ~ 1e56 cm²,
``flux_scale`` ~ 1e-58) fall outside the float32 window ``[1.18e-38, 3.40e38]``.
This module applies such a scale to an array by carrying it as a ``log10``
offset and peak-normalizing the array, so no out-of-range intermediate is
materialized. In float64 the result equals the naive product to machine
precision; in float32 it stays finite whenever the *net* magnitude is in range.
See issue #1186.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

LN10 = 2.302585092994046
LOG10_4PI = float(jnp.log10(4.0 * jnp.pi))  # ~1.09921


def pow10(x):
    """10**x, computed as ``exp(x·ln10)`` to preserve the input dtype.

    Parameters
    ----------
    x : array_like
        Exponent [dimensionless].

    Returns
    -------
    ndarray
        ``10**x`` in the dtype of ``x``.

    Notes
    -----
    JIT/grad/vmap-safe. ``jnp.power(10.0, x)`` would promote to float64 under
    weak typing; ``exp(x·ln10)`` does not.
    """
    return jnp.exp(x * LN10)


def apply_log10_scale(arr, log10_scale):
    """Return ``arr * 10**log10_scale`` without out-of-range intermediates.

    Parameters
    ----------
    arr : array_like
        Values to scale (any magnitude within the dtype range).
    log10_scale : array_like, scalar
        Base-10 log of the multiplicative factor [dimensionless]. May be far
        outside the dtype range (e.g. -58); only the *net* result must be
        representable.

    Returns
    -------
    ndarray
        ``arr * 10**log10_scale``, equal to the naive product within ~1e-12 in
        float64 and finite in float32 when ``max|arr| * 10**log10_scale`` lies in
        the dtype's normal range.

    Notes
    -----
    JIT/grad/vmap-safe. Factors ``arr`` by its peak so the exponentiated scale
    is applied to an O(1) array; the peak's decades are folded into the exponent.

    The peak is held under ``stop_gradient`` (#1415). It is a pure factorization
    constant — ``(arr/p) * 10**(s + log10 p)`` equals ``arr * 10**s`` for *any*
    ``p`` — so its derivative contributions cancel analytically. Left free, they
    are two separate autodiff paths (through the numerator, and through
    ``peak -> net -> pow10``) that must cancel numerically instead. They do in
    float64, but in float32 one side underflows while the other survives, the
    cancellation fails, and what is left is an uncancelled term the size of the
    main one — gradients exactly **2x** too large. Measured on a photometry fit:
    ``d(neg_log_posterior)/d(sfh_delayed_log_total_mass)`` was ``-5915.16``
    against a true ``-2957.58``. Stopping the gradient leaves the single correct
    path, ``d out/d arr = 10**log10_scale``, and float32 then tracks float64 to
    ~1e-7.

    Float64 **forward** values are untouched — ``stop_gradient`` is a no-op on the
    forward pass. Float64 **gradients** move by at most a few ulp: measured
    bit-identical where there is one scale seam (stellar, stellar+dust) and
    ``<= 1.5e-15`` relative where there are several (stellar+dust IR+AGN). That is
    the residue of a cancellation which was only ever exact to rounding, and it is
    three orders inside the ``rtol <= 1e-12`` no-behavioral-change bar for #1206.
    """
    # initial=0.0 makes the peak of a zero-size array 0 (max over empty has no
    # identity and raises); the where() below then maps it to 1, so an empty
    # arr passes through as an empty result instead of a trace-time error.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(arr), initial=0.0))
    usable = peak > 0
    safe_peak = jnp.where(usable, peak, jnp.ones_like(peak))
    # With no peak to fold in, the exponent would collapse to the raw
    # ``log10_scale``. That is fine in float64, but a dust-luminosity scale
    # (~43 dex) overflows float32, and ``0 * inf`` is NaN — so an all-zero
    # array would scale to NaN rather than to zero (#1206). Zeroing the
    # exponent keeps the identity ``0 * 10**s == 0`` at every scale, and costs
    # nothing when there is a peak.
    net = jnp.where(usable, log10_scale + jnp.log10(safe_peak), jnp.zeros_like(peak))
    return (arr / safe_peak) * pow10(net)


def log10_add(log_a, log_b, *, sign_a=1.0, sign_b=1.0):
    """Return ``log10|s_a·10**log_a + s_b·10**log_b|`` without leaving log space.

    A signed base-10 ``logaddexp``. Log-domain contracts (``log_nion``,
    ``log_L_ir``) are exact under multiplication but not under addition, so a
    seam that sums two such quantities would otherwise have to exponentiate
    both — reintroducing the very out-of-range intermediate the log form
    exists to avoid.

    Parameters
    ----------
    log_a, log_b : array_like
        Base-10 log magnitudes [dex]. ``-inf`` denotes an exactly zero term.
    sign_a, sign_b : array_like, optional
        Signs of the two terms (+1.0 or -1.0). Default +1.0. Cancellation
        between opposite signs is resolved at the precision of the larger
        term, as in any signed sum.

    Returns
    -------
    ndarray
        ``log10`` of the magnitude of the sum [dex]; ``-inf`` when the terms
        are both zero or cancel exactly.

    Notes
    -----
    JIT/grad/vmap-safe. Factors out the larger exponent so the exponentiated
    terms are O(1); the where-dummy keeps the backward pass free of NaN when
    the sum vanishes.
    """
    larger = jnp.maximum(log_a, log_b)
    usable = jnp.isfinite(larger)
    offset = jnp.where(usable, larger, 0.0)
    total = sign_a * pow10(log_a - offset) + sign_b * pow10(log_b - offset)
    magnitude = jnp.abs(total)
    positive = usable & (magnitude > 0)
    safe = jnp.where(positive, magnitude, 1.0)
    return jnp.where(positive, offset + jnp.log10(safe), -jnp.inf)


def max_finite_exponent() -> float:
    """Largest ``x`` for which ``exp(x)``/``expm1(x)`` stays finite at working precision.

    Returns 500.0 under float64 — the value the Planck clamps used before —
    and ~87.7 under float32, whose ``exp`` overflows above x ~ 88.7.

    An overflowing exponential is not merely a forward-value problem. A
    saturated ``inf`` denominator still gives the correct limit (the Wien tail
    tends to zero), but its *gradient* is ``inf/inf`` — NaN — so a fit would
    fail where the forward pass looked fine. Capping at the dtype's own limit
    keeps both finite.

    Physically free for a blackbody: x = 88 already puts the Wien tail at
    ``e**-88 ~ 6e-39`` of the peak, far below anything measurable (#1206).

    Returns
    -------
    float
        Clamp ceiling [dimensionless]; a static Python float, safe as a
        ``jnp.clip`` bound under JIT.

    Notes
    -----
    **JIT-compatible**: yes — resolved at trace time from
    ``jnp.result_type(float)``, so it is a compile-time constant.
    """
    import math

    return float(min(500.0, math.log(float(jnp.finfo(jnp.result_type(float)).max)) - 1.0))
