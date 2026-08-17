# SPDX-License-Identifier: BSD-3-Clause
"""Differentiable N-dimensional triweight histograms in JAX.

Vendored from diffsky (Hearin et al.), commit 2024-xx:
https://github.com/ArgonneCPAC/diffsky/blob/main/diffsky/diffndhist.py

Provides smooth, differentiable histogram operations using the triweight
kernel — the same kernel used throughout tengri for grid interpolation
(see ``utils/interpolation.py``).  While interpolation maps a query point
to a grid value, histogramming maps a cloud of data points to bin counts.

Two public functions:

- ``tw_ndhist``: differentiable N-dim histogram (bin counts)
- ``tw_ndhist_weighted``: differentiable weighted sum within bins

Adapted to tengri style: ``jnp.where`` instead of ``lax.cond``.

References
----------

- Hearin et al. 2023, MNRAS, 521, 1741 (triweight kernel / DSPS)

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap

__all__ = [
    "tw_ndhist",
    "tw_ndhist_weighted",
]


@jax.jit
def _tw_cuml_kern(x: float, m: float, h: float) -> float:
    """Triweight kernel CDF (scalar).

    Identical to ``interpolation.tw_cuml_kern`` but kept self-contained
    so this module has zero internal dependencies.
    """
    z = (x - m) / h
    z2 = z * z
    val = (
        z * (35.0 / 96.0 + z2 * (-35.0 / 864.0 + z2 * (7.0 / 2592.0 + z2 * (-5.0 / 69984.0))))
        + 0.5
    )
    val = jnp.where(z < -3.0, 0.0, val)
    val = jnp.where(z > 3.0, 1.0, val)
    return val


@jax.jit
def _tw_bin_weight_kern(x: float, sig: float, lo: float, hi: float) -> float:
    """Triweight kernel integrated across one bin boundary (scalar)."""
    return _tw_cuml_kern(x, lo, sig) - _tw_cuml_kern(x, hi, sig)


_tw_bin_weight_kern_vmap = jax.jit(vmap(_tw_bin_weight_kern, in_axes=(0, 0, 0, 0)))


@jax.jit
def _tw_ndhist_kern(nddata, ndsig, ndlo, ndhi):
    """Weight of one N-dim point in one N-dim bin."""
    return jnp.prod(_tw_bin_weight_kern_vmap(nddata, ndsig, ndlo, ndhi))


_tw_ndhist_kern_vmap = jax.jit(vmap(_tw_ndhist_kern, in_axes=(0, 0, None, None)))


@jax.jit
def _tw_ndhist_sumkern(nddata, ndsig, ndlo, ndhi):
    """Sum contributions from all points to one bin."""
    return jnp.sum(_tw_ndhist_kern_vmap(nddata, ndsig, ndlo, ndhi))


_tw_ndhist_vmap = jax.jit(vmap(_tw_ndhist_sumkern, in_axes=(None, None, 0, 0)))


@jax.jit
def tw_ndhist(
    nddata: jnp.ndarray,
    ndsig: jnp.ndarray,
    ndbins_lo: jnp.ndarray,
    ndbins_hi: jnp.ndarray,
) -> jnp.ndarray:
    """N-dimensional differentiable histogram with arbitrary bins.

    Parameters
    ----------
    nddata : array, shape (npts, ndim)
        Data points in N-dimensional space.
    ndsig : array, shape (npts, ndim)
        Triweight scatter for each point in each dimension.
    ndbins_lo : array, shape (nbins, ndim)
        Lower bound of each bin in each dimension.
    ndbins_hi : array, shape (nbins, ndim)
        Upper bound of each bin in each dimension.

    Returns
    -------
    array, shape (nbins,)
        Smooth histogram counts.
    """
    return _tw_ndhist_vmap(nddata, ndsig, ndbins_lo, ndbins_hi)


@jax.jit
def _tw_ndhist_weighted_sum_kern(nddata, ndsig, y, ndlo, ndhi):
    """Weight * y for one point in one bin."""
    w = jnp.prod(_tw_bin_weight_kern_vmap(nddata, ndsig, ndlo, ndhi))
    return w * y


_tw_ndhist_weighted_sum_vmap = jax.jit(
    vmap(_tw_ndhist_weighted_sum_kern, in_axes=(0, 0, 0, None, None))
)


@jax.jit
def _tw_ndhist_weighted_kern(nddata, ndsig, y, ndlo, ndhi):
    """Summed weighted contributions from all points to one bin."""
    return jnp.sum(_tw_ndhist_weighted_sum_vmap(nddata, ndsig, y, ndlo, ndhi))


_tw_ndhist_weighted_vmap = jax.jit(
    vmap(_tw_ndhist_weighted_kern, in_axes=(None, None, None, 0, 0))
)


@jax.jit
def tw_ndhist_weighted(
    nddata: jnp.ndarray,
    ndsig: jnp.ndarray,
    ydata: jnp.ndarray,
    ndbins_lo: jnp.ndarray,
    ndbins_hi: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable weighted histogram: sum of ydata within N-dim bins.

    Together with ``tw_ndhist``, computes conditional averages
    ``<y | x0, x1, ..., xN>`` by dividing weighted sums by counts.

    Parameters
    ----------
    nddata : array, shape (npts, ndim)
        Data points in N-dimensional space.
    ndsig : array, shape (npts, ndim)
        Triweight scatter for each point in each dimension.
    ydata : array, shape (npts,)
        Quantity to sum within each bin.
    ndbins_lo : array, shape (nbins, ndim)
        Lower bound of each bin in each dimension.
    ndbins_hi : array, shape (nbins, ndim)
        Upper bound of each bin in each dimension.

    Returns
    -------
    array, shape (nbins,)
        Weighted histogram (sum of ydata per bin).
    """
    return _tw_ndhist_weighted_vmap(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
