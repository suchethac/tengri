# SPDX-License-Identifier: BSD-3-Clause
"""The engine-independent sampling problem and its metric layer.

A :class:`FlatTarget` is a log density on one unconstrained vector with an iid N(0, 1)
prior, which is the one space every sampler in :mod:`tengri.inference.sampling.driver`
runs in. :class:`LatentLayout` is the one description of how that vector is assembled
from named blocks (population-shared latents, per-galaxy latents, the stochastic-SFH latent
block) and how it maps back to physical values.

The dense-metric layer lives here as well: :func:`pilot_metric` turns pilot draws into a
whitening factor and :meth:`FlatTarget.whiten` reparametrizes any target by it, so a dense
metric works for any sampler, not only for those with a dense adaptation of their own.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.utils.transforms import to_bounded

#: Weight ``a`` of the diagonal target in the shrunk pilot covariance
#: ``Sigma = (1 - a) S + a diag(S)``. It keeps the factor positive definite when the
#: pilot has few effective draws, at the cost of a little bias toward the diagonal.
DENSE_SHRINKAGE = 0.1


@dataclass(frozen=True)
class LatentBlock:
    """One named block of the flat vector.

    Parameters
    ----------
    group : str
        Sub-dictionary the block lives in; ``""`` for the top level.
    name : str
        Block name within its group.
    shape : tuple of int
        Shape of the block; ``()`` for a scalar.
    bounds : tuple of float or None
        ``(lo, hi)`` physical bounds when the block is a bounded latent mapped by
        :func:`~tengri.utils.transforms.to_bounded`.
    """

    group: str
    name: str
    shape: tuple[int, ...] = ()
    bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class LatentLayout:
    """How a flat latent vector is assembled from named blocks.

    Notes
    -----
    JIT/grad/vmap-compatible: the layout is static and hashable; ``ravel`` and ``unravel``
    are pure array code. The flat order is :func:`jax.flatten_util.ravel_pytree`'s order on
    the nested template, which sorts keys, so a layout built by :meth:`hierarchical` keeps
    the order the hierarchical seam always used.
    """

    blocks: tuple[LatentBlock, ...]

    @classmethod
    def vector(cls, n_dim: int) -> LatentLayout:
        """Return the layout of a plain vector of ``n_dim`` latents."""
        return cls((LatentBlock("", "x", (int(n_dim),)),))

    @classmethod
    def hierarchical(
        cls,
        free_names: tuple[str, ...],
        n_galaxies: int,
        n_grid: int = 0,
        *,
        sigma_bounds: tuple[float, float],
        tau_bounds: tuple[float, float],
    ) -> LatentLayout:
        """Return the layout of a population fit.

        Parameters
        ----------
        free_names : tuple of str
            Per-galaxy free parameter names, one ``(n_galaxies,)`` block each.
        n_galaxies : int
            Galaxies in the catalog.
        n_grid : int
            Stochastic-SFH latent count per galaxy; 0 for a smooth SFH.
        sigma_bounds, tau_bounds : tuple of float
            Bounds of the population PSD amplitude [dex] and timescale [Myr].
        """
        blocks = [
            LatentBlock("", "psd_sigma_u", (), tuple(sigma_bounds)),
            LatentBlock("", "psd_tau_u", (), tuple(tau_bounds)),
            *(LatentBlock("gal", name, (int(n_galaxies),)) for name in free_names),
        ]
        if n_grid:
            blocks.append(LatentBlock("", "gal_xi", (int(n_galaxies), int(n_grid))))
        return cls(tuple(blocks))

    @property
    def n_dim(self) -> int:
        """Length of the flat vector."""
        total = 0
        for block in self.blocks:
            size = 1
            for extent in block.shape:
                size *= extent
            total += size
        return total

    def template(self) -> dict:
        """Return the nested zero pytree this layout flattens."""
        tree: dict = {}
        for block in self.blocks:
            target = tree.setdefault(block.group, {}) if block.group else tree
            target[block.name] = jnp.zeros(block.shape)
        return tree

    def ravel(self, tree: dict) -> jax.Array:
        """Flatten a nested pytree of latents to the flat vector."""
        return ravel_pytree(tree)[0]

    def unravel(self, flat: jax.Array) -> dict:
        """Return the nested pytree of latents held by ``flat``."""
        return _unraveler(self)(flat)

    def to_physical(self, flat: jax.Array, pushforwards: dict[str, Callable] | None = None):
        """Map the latents to physical values.

        Parameters
        ----------
        flat : ndarray, shape (D,)
            Flat latent vector.
        pushforwards : dict of callable, optional
            ``name -> f`` applied to the per-galaxy blocks of that name; a bounded block is
            mapped by its bounds, and a block with neither is returned unchanged.

        Returns
        -------
        dict
            ``name -> physical value`` for every block.
        """
        tree = self.unravel(flat)
        out = {}
        for block in self.blocks:
            value = tree[block.group][block.name] if block.group else tree[block.name]
            if block.bounds is not None:
                value = to_bounded(value, *block.bounds)
            elif pushforwards and block.name in pushforwards:
                value = pushforwards[block.name](value)
            out[block.name] = value
        return out


@functools.cache
def _unraveler(layout: LatentLayout) -> Callable:
    """Return the cached ``unravel`` of a layout's template."""
    return ravel_pytree(layout.template())[1]


def masked_covariance(draws, mask):
    """Sample covariance of the rows of ``draws`` selected by ``mask``, unbiased.

    Rows outside ``mask`` are excluded even when non-finite, so a divergent or NaN pilot
    draw cannot reach the covariance. With fewer than two selected rows the result is NaN,
    which the caller's finiteness check then refuses.

    Parameters
    ----------
    draws : ndarray, shape (n, D)
        Pilot positions in the sampled coordinates.
    mask : ndarray, shape (n,), bool
        True for the rows that may enter the covariance.

    Returns
    -------
    ndarray, shape (D, D)
        Covariance of the selected rows.
    """
    weight = mask.astype(draws.dtype)
    count = jnp.sum(weight)
    kept = jnp.where(mask[:, None], draws, 0.0)
    mean = jnp.where(
        count > 0.5,
        jnp.sum(weight[:, None] * kept, axis=0) / jnp.maximum(count, 1.0),
        jnp.nan,
    )
    centered = jnp.where(mask[:, None], draws - mean, 0.0)
    return jnp.where(
        count > 1.5,
        (centered.T @ centered) / jnp.maximum(count - 1.0, 1.0),
        jnp.nan,
    )


def pilot_metric(draws, mask, shrinkage: float = DENSE_SHRINKAGE):
    """Return the whitening factor of a pilot run and whether it fell back.

    The factor is the lower Cholesky ``L`` of ``Sigma = (1 - a) S + a diag(S)``, ``S`` the
    covariance of the usable pilot rows and ``a`` the shrinkage. A pilot with fewer than
    ``D + 1`` usable rows, or a ``Sigma`` that is not positive definite (a stuck pilot has
    ``S = 0``), cannot define a metric, and the identity is returned with
    ``fallback_used`` True so the caller keeps the pilot's own draws and reports it.

    Parameters
    ----------
    draws : ndarray, shape (n, D)
        Pilot positions in the sampled coordinates.
    mask : ndarray, shape (n,), bool
        Rows that may enter the covariance (finite and not divergent).
    shrinkage : float
        Weight ``a`` of the diagonal target.

    Returns
    -------
    factor : ndarray, shape (D, D)
        Lower Cholesky factor, or the identity on fallback.
    fallback_used : bool ndarray
        True when the identity replaced the factor.

    Notes
    -----
    JIT/vmap-compatible: no data-dependent Python control flow.
    """
    dim = draws.shape[-1]
    cov = masked_covariance(draws, mask).reshape(dim, dim)
    shrunk = (1.0 - shrinkage) * cov + shrinkage * jnp.diag(jnp.diag(cov))
    factor = jnp.linalg.cholesky(0.5 * (shrunk + shrunk.T))
    ok = (jnp.sum(mask) >= dim + 1) & jnp.all(jnp.isfinite(factor))
    return jnp.where(ok, factor, jnp.eye(dim, dtype=factor.dtype)), ~ok


def pilot_whitening(draws, mask, shrinkage: float = DENSE_SHRINKAGE):
    """Return the center, factor and fallback flag of a pilot run.

    Parameters
    ----------
    draws : ndarray, shape (n, D)
        Pilot positions in the sampled coordinates.
    mask : ndarray, shape (n,), bool
        Rows that may enter the moments.
    shrinkage : float
        See :func:`pilot_metric`.

    Returns
    -------
    mu : ndarray, shape (D,)
        Mean of the usable rows; zero on fallback.
    factor : ndarray, shape (D, D)
        Lower Cholesky factor from :func:`pilot_metric`, the identity on fallback.
    fallback_used : bool ndarray
        True when the identity replaced the factor.
    """
    factor, fallback = pilot_metric(draws, mask, shrinkage)
    weight = mask.astype(draws.dtype)
    mean = jnp.sum(weight[:, None] * jnp.where(mask[:, None], draws, 0.0), axis=0)
    mean = mean / jnp.maximum(jnp.sum(weight), 1.0)
    return jnp.where(fallback, 0.0, mean), factor, fallback


@functools.cache
def whiten_logdensity(logdensity):
    """Return ``f(z, (mu, L, item_args))`` evaluating ``logdensity(mu + L z, item_args)``.

    The returned function has a stable identity per ``logdensity``, so it can sit in a
    static jit slot while ``mu`` and ``L`` travel as traced data (per galaxy under vmap).
    """

    def whitened_logdensity(z, args):
        mu, factor, inner = args
        return logdensity(mu + factor @ z, inner)

    return whitened_logdensity


@dataclass(frozen=True)
class FlatTarget:
    """A sampling problem on one unconstrained vector with an iid N(0, 1) prior.

    Parameters
    ----------
    logdensity : callable
        ``logdensity(x, item_args) -> scalar``, the log posterior. A module-level function
        (stable identity), with the data in ``item_args`` so one compiled program serves
        every item.
    layout : LatentLayout
        How ``x`` is assembled.
    init : ndarray, shape (D,)
        Starting position.
    item_args : pytree or None
        Arrays traced through ``logdensity``.
    log_likelihood : callable or None
        ``log_likelihood(x, item_args)``, for samplers that treat the prior themselves
        (elliptical slice, nested, tempered SMC). When None it is derived as
        ``logdensity + 0.5 |x|^2``, valid when ``prior_is_standard``.
    prior_is_standard : bool
        True when the prior is exactly iid N(0, 1) in ``x``. False after whitening, and for
        a partially centered field prior.
    """

    logdensity: Callable[[Any, Any], Any]
    layout: LatentLayout
    init: Any
    item_args: Any = None
    log_likelihood: Callable | None = None
    prior_is_standard: bool = True
    extras: dict = field(default_factory=dict, compare=False)

    @property
    def n_dim(self) -> int:
        """Length of the flat vector."""
        return int(self.init.shape[0])

    def likelihood(self) -> Callable:
        """Return ``log_likelihood(x, item_args)``, deriving it under the standard prior."""
        if self.log_likelihood is not None:
            return self.log_likelihood
        if not self.prior_is_standard:
            raise ValueError("a target with a non-standard prior needs an explicit log_likelihood")
        return _derived_likelihood(self.logdensity)

    def whiten(self, mu, factor):
        """Return the target reparametrized by ``x = mu + L z`` and the back transform.

        Parameters
        ----------
        mu : ndarray, shape (D,)
            Center of the new coordinates.
        factor : ndarray, shape (D, D)
            Lower-triangular ``L``; ``L L^T`` is the pilot covariance.

        Returns
        -------
        whitened : FlatTarget
            Target in ``z``; its prior is no longer standard.
        back : callable
            ``back(z_draws) -> x_draws``, rows mapped by ``mu + L z``.
        """
        z0 = jax.scipy.linalg.solve_triangular(factor, self.init - mu, lower=True)
        whitened = FlatTarget(
            logdensity=whiten_logdensity(self.logdensity),
            layout=self.layout,
            init=z0,
            item_args=(mu, factor, self.item_args),
            prior_is_standard=False,
            extras=dict(self.extras),
        )

        def back(draws):
            return mu + draws @ factor.T

        return whitened, back


@functools.cache
def _derived_likelihood(logdensity):
    def log_likelihood(x, args):
        return logdensity(x, args) + 0.5 * jnp.sum(x**2)

    return log_likelihood
