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

import jax
import jax.numpy as jnp

from tengri.utils.grid_interp import pchip_interp_1d, pchip_interp_local

__all__ = ["met_collapsed", "met_weights", "read_met", "read_met_per_age"]


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
