# SPDX-License-Identifier: BSD-3-Clause
"""Recover the centered SFH field from stored non-centered posterior samples."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.components.stellar.sfh.registry import compute_field_gp

__all__ = ["centered_fields"]


def centered_fields(xi, psd_sigma_dex, psd_tau_yr, log_age_grid, centering=1.0):
    r"""SFH log-modulation implied by stored latents and hyperparameters.

    tengri stores the field in non-centered coordinates: the posterior samples
    carry :math:`\xi \sim \mathcal{N}(0, I)` together with
    :math:`(\sigma, \tau)`, and the field itself is the deterministic image

    .. math:: m = \operatorname{gp\_x}(\xi, \sigma, \tau) - K(0)/2

    with :math:`\operatorname{gp\_x}` the correlated modulation and
    :math:`K(0)/2 = (\sigma \ln 10)^2 / 2` the lognormal bias correction, both
    in natural-log units. The star formation rate is modulated by
    :math:`\exp(m)`.

    Both terms are required. :math:`\sigma` enters the likelihood twice -- once
    inside the covariance and once through this bias correction -- so returning
    ``gp_x`` alone omits a term that grows quadratically with :math:`\sigma`,
    biasing burstier populations more than smooth ones.

    Parameters
    ----------
    xi : array_like, shape (..., n)
        Standard normal latents, as stored under ``Posterior.samples["psd_xi"]``.
        Leading axes are mapped over.
    psd_sigma_dex : array_like, shape (...)
        Modulation amplitude [dex], broadcasting against ``xi``'s leading axes.
    psd_tau_yr : array_like, shape (...)
        Damping timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age / yr)`` nodes, monotone.
    centering : float, optional
        The ``sfh={'field_centering': a}`` the fit was run with, in ``[0, 1]``
        [dimensionless]. Default ``1.0`` (non-centered), which is what stored
        latents mean unless the model set otherwise. **Pass the model's value**:
        reconstructing an ``a < 1`` fit with the default silently applies a
        different map to the samples, which is the exact drift this function's
        delegation to ``compute_field_gp`` exists to prevent (#1355).

    Returns
    -------
    m : ndarray, shape (..., n)
        Centered field [natural-log units].

    Notes
    -----
    **JIT/grad/vmap compatible**: yes.

    This delegates to :func:`~tengri.components.stellar.sfh.registry.compute_field_gp`
    -- the same function the forward model calls -- rather than reimplementing
    the map. Two implementations of one transform is how a reconstruction
    silently stops matching the fit that produced it.
    """
    xi = jnp.asarray(xi)
    log_age_grid = jnp.asarray(log_age_grid)
    n_grid = log_age_grid.shape[0]

    # The trailing axis of ``xi`` MUST be the field grid. Without this check a
    # mismatch does not raise: ``xi.reshape(-1, n_grid)`` happily redistributes
    # the elements, and the error surfaces far downstream as an unrelated vmap
    # complaint about ``sigma`` having the wrong length -- which is how a
    # 256-latent posterior paired with a 16-point grid was first seen.
    if xi.ndim == 0 or xi.shape[-1] != n_grid:
        raise ValueError(
            f"xi's trailing axis is {xi.shape[-1] if xi.ndim else '(scalar)'} but "
            f"log_age_grid has {n_grid} points; they must match, since the trailing "
            f"axis of xi IS the field grid. Pass the same n_grid the model was built "
            f"with (SEDModel.build(..., n_grid=N), default 256), a posterior's "
            f"'psd_xi' has shape (n_samples, n_grid)."
        )

    d_log_age = float(log_age_grid[1] - log_age_grid[0])

    def one(xi_1d, sigma, tau):
        gp_x, k0_half = compute_field_gp(
            xi_1d,
            sigma,
            tau,
            n_grid,
            d_log_age,
            log_age_grid=log_age_grid,
            centering=centering,
        )
        return gp_x - k0_half

    sigma = jnp.broadcast_to(jnp.asarray(psd_sigma_dex), xi.shape[:-1])
    tau = jnp.broadcast_to(jnp.asarray(psd_tau_yr), xi.shape[:-1])

    flat_xi = xi.reshape(-1, n_grid)
    flat_out = jax.vmap(one)(flat_xi, sigma.reshape(-1), tau.reshape(-1))
    return flat_out.reshape(xi.shape)
