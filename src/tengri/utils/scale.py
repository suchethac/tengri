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
    """
    # initial=0.0 makes the peak of a zero-size array 0 (max over empty has no
    # identity and raises); the where() below then maps it to 1, so an empty
    # arr passes through as an empty result instead of a trace-time error.
    peak = jnp.max(jnp.abs(arr), initial=0.0)
    peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    net = log10_scale + jnp.log10(peak)
    return (arr / peak) * pow10(net)
