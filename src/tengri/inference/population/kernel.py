# SPDX-License-Identifier: BSD-3-Clause
"""Exact O(n) Gaussian log-density for a damped-random-walk field."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["ou_logpdf"]


def ou_logpdf(m, mean, psd_sigma_dex, psd_tau_yr, times_yr):
    r"""Log-density of a DRW field under its own prior, in ``O(n)``.

    The damped random walk is a first-order Markov (Ornstein-Uhlenbeck) process,
    so its joint density factorizes into a chain of univariate conditionals and
    its precision matrix is tridiagonal. Evaluating it therefore costs ``O(n)``
    with no Cholesky and no matrix storage:

    .. math::

        \log p(m) = \log \mathcal{N}\!\left(m_0;\ \mu,\ v\right)
          + \sum_{i=1}^{n-1} \log \mathcal{N}\!\left(
              m_i;\ \mu + \rho_i (m_{i-1} - \mu),\ v\,(1 - \rho_i^2)\right)

    with :math:`v = (\sigma \ln 10)^2` the marginal variance [natural-log units],
    :math:`\rho_i = \exp(-|t_i - t_{i-1}| / \tau)` the lag-one correlation
    [dimensionless], :math:`\mu` the mean [natural-log units], :math:`t_i` the
    physical times [yr] and :math:`\tau` the damping timescale [yr].

    This is exact, not an approximation: it is the same density a dense
    multivariate normal with :math:`K_{ij} = v \exp(-|t_i - t_j| / \tau)` returns.

    Parameters
    ----------
    m: array_like, shape (n,)
        Field values [natural-log units], ordered to match ``times_yr``.
    mean: float
        Mean of the field [natural-log units], broadcast over the grid.
    psd_sigma_dex: float
        Modulation amplitude [dex].
    psd_tau_yr: float
        Damping timescale [yr].
    times_yr: array_like, shape (n,)
        Physical times [yr], same order as ``m``.

    Returns
    -------
    logpdf: ndarray, shape ()
        Log-density [nats].

    Notes
    -----
    **JIT/grad/vmap compatible**: yes, in every argument. **O(n)** time, **O(n)**
    memory.

    Order matters only through the consecutive differences of ``times_yr``; the
    density is invariant to reversing both ``m`` and ``times_yr`` together.
    """
    m = jnp.asarray(m)
    times_yr = jnp.asarray(times_yr)
    var = (jnp.asarray(psd_sigma_dex) * jnp.log(10.0)) ** 2
    resid = m - mean

    head = -0.5 * (resid[0] ** 2 / var + jnp.log(2.0 * jnp.pi * var))

    dt = jnp.abs(jnp.diff(times_yr))
    rho = jnp.exp(-dt / jnp.asarray(psd_tau_yr))
    cond_var = var * (1.0 - rho**2)
    innov = resid[1:] - rho * resid[:-1]
    tail = -0.5 * jnp.sum(innov**2 / cond_var + jnp.log(2.0 * jnp.pi * cond_var))

    return head + tail
