# SPDX-License-Identifier: BSD-3-Clause
"""Monotone PCHIP read of the SSP tables across metallicity, at an undeclared scatter (#3434).

Owner ruling (2026-11-15, "true PCHIP on the spectra"): with no declared metallicity
scatter, every SSP table is read across its metallicity nodes by a monotone
Fritsch-Carlson PCHIP in log Z, column by column, and only then contracted with the
age weights. This replaces the signed cubic-Hermite weights of #3434, which overshoot
monotone data. The declared-scatter (lognormal kernel) branch is not touched here.

The read is node-exact (bit for bit), never overshoots monotone data, is
differentiable in log Z, and matches :class:`scipy.interpolate.PchipInterpolator`
column by column. Outside the grid the query is held at the edge node.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax import lax

from tengri.utils.grid_interp import _pchip_slopes, pchip_interp_1d, pchip_interp_local

__all__ = [
    "met_collapsed",
    "met_collapsed_hermite",
    "met_weights",
    "native_slope_flux",
    "native_slope_scope",
    "pchip_node_slopes",
    "read_met",
    "read_met_hermite",
    "read_met_hermite_per_age",
    "read_met_per_age",
]

#: Wavelength columns per tangent call. The PCHIP tangents are eager, so a whole-grid call
#: holds about ten cubes of the slope shape at once (a 600 MB transient on the wNE grid,
#: #3541). Each column is an independent map along the last axis, so the chunked result is
#: bit-identical to the whole-grid one.
_SLOPE_CHUNKS = 16

#: Per-thread stack of open :func:`native_slope_scope` memos, innermost last.
_SLOPE_STATE = threading.local()


def _open_scopes() -> list[dict]:
    """The calling thread's stack of open slope scopes."""
    if not hasattr(_SLOPE_STATE, "scopes"):
        _SLOPE_STATE.scopes = []
    return _SLOPE_STATE.scopes


def _clamp_to_grid(log_z, x):
    """Hold the query at the node hull with a where, not a clip (no halved slope)."""
    return jnp.where(log_z < x[0], x[0], jnp.where(log_z > x[-1], x[-1], log_z))


def _pin_to_node(val, table, x, zc):
    """Make the read bit-exact at a node without changing its derivative.

    The cubic is exact at a node in exact arithmetic but its scaled evaluation can
    miss the node value by one ulp. The correction ``exact - val`` is exact by
    Sterbenz's lemma (the two are within a factor of two), so ``val + correction``
    returns the node value bit for bit. It is applied under ``stop_gradient``, so the
    derivative in log Z is the cubic's own, which is continuous at the node.
    """
    k = jnp.argmin(jnp.abs(x - zc))
    hit = x[k] == zc
    correction = jnp.where(hit, table[k] - val, jnp.zeros_like(val))
    return val + jax.lax.stop_gradient(correction)


def read_met(table, ssp_lgmet, log_z):
    """Read a metallicity-leading table at one log Z by monotone PCHIP.

    Parameters
    ----------
    table : array_like, shape (n_met, ...)
        Node values, metallicity on the leading axis. Any trailing shape.
    ssp_lgmet : array_like, shape (n_met,)
        Ascending metallicity nodes, absolute log10(Z).
    log_z : scalar, possibly traced
        Query log10(Z), same units as ``ssp_lgmet``.

    Returns
    -------
    ndarray, shape table.shape[1:]
        The PCHIP read of every trailing column at ``log_z``.

    Notes
    -----
    For ``n_met >= 4`` the 4-node stencil :func:`pchip_interp_local` is used; it
    gives the global PCHIP tangents exactly. For ``n_met`` 2 or 3 the column-wise
    :func:`pchip_interp_1d` is used. ``n_met == 1`` returns the single node.
    **JIT/vmap/grad-safe**: yes.
    """
    table = jnp.asarray(table)
    x = jnp.asarray(ssp_lgmet)
    n = x.shape[0]
    if n == 1:
        return table[0]
    zc = _clamp_to_grid(jnp.asarray(log_z, dtype=x.dtype), x)
    if n >= 4:
        val = pchip_interp_local(x, table, zc)
    else:
        flat = table.reshape(n, -1)
        cols = jax.vmap(lambda col: pchip_interp_1d(x, col, zc[None])[0], in_axes=1)(flat)
        val = cols.reshape(table.shape[1:])
    return _pin_to_node(val, table, x, zc)


def read_met_per_age(table, ssp_lgmet, log_z_age, *, age_axis=1):
    """Read a table at one log Z per age: ``F[age, ...] = PCHIP_Z(T[:, age, ...])(z_age)``.

    Parameters
    ----------
    table : array_like, shape (n_met, ..., n_age, ...)
        Metallicity on axis 0; the age axis is ``age_axis``.
    ssp_lgmet : array_like, shape (n_met,)
        Ascending metallicity nodes.
    log_z_age : array_like, shape (n_age,)
        Query log10(Z) for each age.
    age_axis : int, optional
        Position of the age axis in ``table`` (>= 1). Default 1.

    Returns
    -------
    ndarray, shape table.shape[1:]
        Each age column read at its own log Z; the age axis stays at ``age_axis - 1``.
    """
    moved = jnp.moveaxis(jnp.asarray(table), age_axis, 1)  # (n_met, n_age, ...)
    per_age = jax.vmap(lambda col, z: read_met(col, ssp_lgmet, z), in_axes=(1, 0))(
        moved, jnp.asarray(log_z_age)
    )  # (n_age, ...)
    return jnp.moveaxis(per_age, 0, age_axis - 1)


def met_collapsed(table, ssp_lgmet, log_z, *, age_axis=1):
    """Met-collapsed table ``[1, ...]`` for a scalar or per-age log Z.

    The leading axis has length one with met weights ``[1.0]``, so the existing
    ``"ma,ma...->a..."`` contractions run unchanged with the age weights alone.
    A scalar ``log_z`` reads every age at the same Z; an array reads age ``a`` at
    ``log_z[a]``.
    """
    if jnp.ndim(log_z) == 0:
        return read_met(table, ssp_lgmet, log_z)[None]
    return read_met_per_age(table, ssp_lgmet, log_z, age_axis=age_axis)[None]


def exact_table(table, joint, lgmet, met_z, *, age_axis=1):
    """The table to contract with ``joint``: exact at the galaxy's met_z (#3434, phase 2).

    A joint with the full metallicity axis (``n_met`` rows, a declared scatter or
    a single node) contracts the table as stored. A joint with one row is the
    undeclared age marginal, and the table is PCHIP-read over metallicity at
    ``met_z`` and collapsed to ``[1, ...]`` first. Static on the joint's shape.
    """
    if joint.shape[0] == lgmet.shape[0]:
        return table
    return met_collapsed(table, lgmet, met_z, age_axis=age_axis)


def met_weights(log_z, ssp_lgmet):
    """Linear met-weight surrogate of the PCHIP read: ``W_m(z)`` with ``read = W @ table``.

    Obtained by reading the identity table, so ``W(z)`` is one-hot at every node and
    sums to one (PCHIP reproduces constants). It is used only where an
    ``(n_met, n_age)`` joint weight is needed for a consumer that has not yet moved
    to the exact read (#3434 phase 2). It is NOT the exact PCHIP read between nodes,
    because PCHIP is non-linear in the node values. A scalar ``log_z`` gives
    ``(n_met,)``; an array of per-age values gives ``(n_met, n_age)``.
    """
    x = jnp.asarray(ssp_lgmet)
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    if jnp.ndim(log_z) == 0:
        raw = read_met(eye, x, log_z)
        return raw / jnp.sum(raw)
    raw = jax.vmap(lambda z: read_met(eye, x, z))(jnp.asarray(log_z))
    return (raw / jnp.sum(raw, axis=1, keepdims=True)).T


def pchip_node_slopes(table, ssp_lgmet):
    """Native PCHIP tangents along the metallicity axis (#3541).

    The tangents depend only on the node values, not on the query Z, so they are
    computed once on the native grid and carried through each table's linear build.
    Same formula as :func:`read_met`: :func:`pchip_interp_local` uses the global
    Fritsch-Carlson tangents for ``n_met >= 4``, :func:`pchip_interp_1d` for 2 or 3,
    and the secant for 2. ``n_met == 1`` gives zeros.
    """
    table = jnp.asarray(table)
    x = jnp.asarray(ssp_lgmet)
    if x.shape[0] == 1:
        return jnp.zeros_like(table)
    n_wave = table.shape[-1]
    k = min(_SLOPE_CHUNKS, n_wave)
    if k <= 1:
        return _pchip_slopes(x, table)
    # Fixed-width windows: every call has one shape, so the eager ops compile once. The last
    # window is clamped to the grid end and trimmed to the columns it owns.
    width = -(-n_wave // k)
    pieces = []
    for i in range(k):
        lo = i * width
        if lo >= n_wave:
            break
        hi = min(lo + width, n_wave)
        start = min(lo, n_wave - width)
        window = _pchip_slopes(x, lax.dynamic_slice_in_dim(table, start, width, axis=-1))
        pieces.append(window[..., lo - start : hi - start])
    return jnp.concatenate(pieces, axis=-1)


@contextmanager
def native_slope_scope():
    """Compute each SSP grid's native PCHIP tangents once while this scope is open (#3541).

    Reentrant: a scope opened inside an open one shares its memo, so a whole model build
    takes the tangents once per grid, and the memo (one slope cube per grid) is released
    when the outermost scope exits. Outside any scope nothing is cached.
    """
    scopes = _open_scopes()
    if scopes:
        yield
        return
    scopes.append({})
    try:
        yield
    finally:
        scopes.pop()


def native_slope_flux(ssp_flux, ssp_lgmet):
    """Native PCHIP tangents of an SSP flux cube, shared within an open scope (#3541).

    The one place the native slopes are formed. Inside :func:`native_slope_scope` the cube is
    keyed on the identity of the flux and grid objects it was built from, so a second request
    for the same grid returns the first result instead of recomputing it.
    """
    scopes = _open_scopes()
    if not scopes:
        return pchip_node_slopes(ssp_flux, ssp_lgmet)
    memo = scopes[-1]
    hit = memo.get(id(ssp_flux))
    if hit is not None and hit[0] is ssp_flux and hit[1] is ssp_lgmet:
        return hit[2]
    slopes = pchip_node_slopes(ssp_flux, ssp_lgmet)
    memo[id(ssp_flux)] = (ssp_flux, ssp_lgmet, slopes)
    return slopes


def read_met_hermite(values, slopes, ssp_lgmet, log_z):
    """Cubic Hermite read of (node values, native slopes) at one log Z (#3541).

    Equals :func:`read_met` on the table the slopes came from, to round-off: the
    Hermite basis in the bracketing cell, clamped outside the grid as :func:`read_met`
    clamps, and node-exact. Linear in ``values`` and in ``slopes``, so a table and its
    slope sibling can be built by the same linear code.
    """
    values = jnp.asarray(values)
    slopes = jnp.asarray(slopes)
    x = jnp.asarray(ssp_lgmet)
    n = x.shape[0]
    if n == 1:
        return values[0]
    zc = _clamp_to_grid(jnp.asarray(log_z, dtype=x.dtype), x)
    i = jnp.clip(jnp.searchsorted(x, zc) - 1, 0, n - 2)
    h = x[i + 1] - x[i]
    t = (zc - x[i]) / h
    t2 = t * t
    t3 = t2 * t
    val = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * values[i]
        + (t3 - 2.0 * t2 + t) * h * slopes[i]
        + (-2.0 * t3 + 3.0 * t2) * values[i + 1]
        + (t3 - t2) * h * slopes[i + 1]
    )
    return _pin_to_node(val, values, x, zc)


def read_met_hermite_per_age(values, slopes, ssp_lgmet, log_z_age, *, age_axis=1):
    """Per-age :func:`read_met_hermite`, the twin of :func:`read_met_per_age` (#3541)."""
    mv = jnp.moveaxis(jnp.asarray(values), age_axis, 1)
    ms = jnp.moveaxis(jnp.asarray(slopes), age_axis, 1)
    per_age = jax.vmap(lambda v, s, z: read_met_hermite(v, s, ssp_lgmet, z), in_axes=(1, 1, 0))(
        mv, ms, jnp.asarray(log_z_age)
    )
    return jnp.moveaxis(per_age, 0, age_axis - 1)


def met_collapsed_hermite(values, slopes, ssp_lgmet, log_z, *, age_axis=1):
    """Hermite twin of :func:`met_collapsed`: ``[1, ...]`` for scalar or per-age log Z."""
    if jnp.ndim(log_z) == 0:
        return read_met_hermite(values, slopes, ssp_lgmet, log_z)[None]
    return read_met_hermite_per_age(values, slopes, ssp_lgmet, log_z, age_axis=age_axis)[None]
