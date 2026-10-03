# SPDX-License-Identifier: BSD-3-Clause
"""Redshift-tabulated build-time responses, read at the evaluation redshift.

A fast-path response (dust-IR band response, radio and X-ray term response,
the energy-balance LUT of a redshift-reading attenuation law) is a function of
the redshift the model is *evaluated* at, not of any redshift known when the
table is built. Under ``WavePrecomp(catalog_z_range=...)`` the spec redshift is
a placeholder ``Fixed`` while every evaluation runs at its own runtime z; under
a free redshift every sample runs at its own z. A table baked at the spec value
is then wrong by 50-300 % in the bands where the filter samples a steep part of
the template (WISE W3/W4).

The fix is to tabulate the response over the model's redshift range once, and
interpolate at the traced evaluation redshift. This module is the one place that
knows the table layout, so the producer (``SEDModel``) and the consumers (the
emitter components, the dust component) cannot drift apart.

Layout: a mapping ``{"ln1pz": (n_z,), "values": (n_z, ...)}``. The nodes are
uniform in :math:`\\ln(1+z)`, which resolves the low-z end of a wide prior 10-20x
better than uniform-z nodes at the same count, because every quantity here
depends on :math:`1+z` (filter wavelengths are divided by it). A model whose
redshift is a single ``Fixed`` value gets a one-node table, and reading it is
exactly ``values[0]``, bit-for-bit the pre-table constant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "build_z_table",
    "interp_z_table",
    "ln1pz_nodes",
    "single_node",
    "z_bracket",
]

#: Keys of a tabulated response.
NODES_KEY = "ln1pz"
VALUES_KEY = "values"


def ln1pz_nodes(z_lo: float, z_hi: float, n_z: int, dtype=None) -> jnp.ndarray:
    """Nodes uniform in ``ln(1+z)`` spanning ``[z_lo, z_hi]``.

    Parameters
    ----------
    z_lo, z_hi : float
        Redshift range, ``-1 < z_lo < z_hi``. [dimensionless]
    n_z : int
        Node count, at least 2.
    dtype : dtype, optional
        Node dtype. Defaults to the JAX default float.

    Returns
    -------
    ndarray, shape (n_z,)
        ``ln(1+z)`` at each node. [dimensionless]
    """
    if n_z < 2:
        raise ValueError(f"a redshift range needs at least 2 nodes, got n_z={n_z}")
    if not (-1.0 < z_lo < z_hi):
        raise ValueError(f"need -1 < z_lo < z_hi for a redshift table, got ({z_lo}, {z_hi})")
    lo = jnp.log1p(jnp.asarray(z_lo, dtype=dtype))
    hi = jnp.log1p(jnp.asarray(z_hi, dtype=dtype))
    return jnp.linspace(lo, hi, int(n_z))


def single_node(z: Any, dtype=None) -> jnp.ndarray:
    """One-node ``ln(1+z)`` axis for a model whose redshift is a single value."""
    return jnp.log1p(jnp.atleast_1d(jnp.asarray(z, dtype=dtype)))


def build_z_table(
    fn: Callable[[jnp.ndarray], Any], nodes: jnp.ndarray, z_values: Any = None
) -> dict:
    """Tabulate ``fn(z)`` at every node.

    ``fn`` is jitted once and called per node, so the build costs one compile
    plus ``n_z`` cheap executions rather than ``n_z`` eager op-by-op passes.

    Parameters
    ----------
    fn : callable
        ``fn(z) -> pytree of arrays`` of fixed shapes, jit-safe in ``z``.
    nodes : ndarray, shape (n_z,)
        ``ln(1+z)`` axis.
    z_values : array_like, shape (n_z,), optional
        The redshifts to call ``fn`` at. Defaults to ``expm1(nodes)``; a
        one-node table for a ``Fixed`` redshift passes the spec value so the
        table holds ``fn(z)`` for that exact ``z`` rather than one round trip
        through ``log1p``/``expm1``.

    Returns
    -------
    dict
        ``{"ln1pz": nodes, "values": pytree stacked to (n_z, ...)}``.
    """
    zs = jnp.expm1(nodes) if z_values is None else jnp.atleast_1d(jnp.asarray(z_values))
    jitted = jax.jit(fn)
    rows = [jitted(zs[i]) for i in range(zs.shape[0])]
    values = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)
    return {NODES_KEY: nodes, VALUES_KEY: values}


def z_bracket(nodes: jnp.ndarray, z: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Lower bracketing node and the two linear weights for ``z``.

    Linear in ``ln(1+z)``. A ``z`` outside the table holds the end node (the
    weights are clipped), so the interpolant is continuous and differentiable
    on the table and flat beyond it.

    Parameters
    ----------
    nodes : ndarray, shape (n_z,)
        Ascending ``ln(1+z)`` axis, ``n_z >= 2``. Need not be uniform: a table
        over a law with redshift breakpoints carries a node on each.
    z : scalar
        Evaluation redshift. [dimensionless]

    Returns
    -------
    i0 : ndarray, shape (), int32
        Lower node index in ``[0, n_z - 2]``.
    weights : ndarray, shape (2,)
        Weights on nodes ``i0`` and ``i0 + 1``, summing to 1.

    Notes
    -----
    **JIT/grad/vmap-safe.** ``i0`` is piecewise constant; the derivative in
    ``z`` flows through the weights.
    """
    x = jnp.log1p(jnp.asarray(z, dtype=nodes.dtype))
    n = nodes.shape[0]
    i0 = jnp.clip(jnp.searchsorted(nodes, x, side="right") - 1, 0, n - 2).astype(jnp.int32)
    lo, hi = nodes[i0], nodes[i0 + 1]
    t = jnp.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return i0, jnp.stack([1.0 - t, t])


def interp_z_table(table: Mapping[str, jnp.ndarray], z: Any) -> jnp.ndarray:
    """Read a tabulated response at the evaluation redshift ``z``.

    Parameters
    ----------
    table : mapping
        From :func:`build_z_table`.
    z : scalar
        Evaluation redshift, the merged params' ``redshift``. [dimensionless]

    Returns
    -------
    ndarray
        The response at ``z``, shaped like one row of ``table["values"]``. For a
        one-node table this is ``values[0]`` regardless of ``z``.
    """
    values = table[VALUES_KEY]
    nodes = table[NODES_KEY]
    if nodes.shape[0] == 1:
        return values[0]
    i0, w = z_bracket(nodes, z)
    rows = jax.lax.dynamic_slice_in_dim(values, i0, 2, axis=0)
    return jnp.tensordot(w.astype(values.dtype), rows, axes=1)
