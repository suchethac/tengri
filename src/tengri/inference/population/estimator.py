# SPDX-License-Identifier: BSD-3-Clause
"""Two-step hierarchical estimators for the shared SFH PSD block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.inference.population.kernel import ou_logpdf

__all__ = ["ESSSummary", "SharedGrid", "effective_sample_size", "shared_log_posterior"]


class ESSSummary(NamedTuple):
    """Effective sample size diagnostic for importance weights.

    Attributes
    ----------
    at_mode: ndarray, shape (N,)
        ESS at the posterior mode [dimensionless], the primary diagnostic.
    min_high_mass: ndarray, shape (N,)
        Minimum ESS over nodes carrying the top 99% of posterior mass
        [dimensionless]. Use this to detect degeneracy in the tails.
    """

    at_mode: jnp.ndarray
    min_high_mass: jnp.ndarray


def _field_mean(sigma_dex):
    return -0.5 * (jnp.asarray(sigma_dex) * jnp.log(10.0)) ** 2


#: Beyond this, ``exp(-|t_i - t_j| / tau)`` underflows to 1 and ``ou_logpdf`` returns NaN.
_TAU_NAN_THRESHOLD_YR = 1.0e20
_AGE_UNIVERSE_YR = 1.38e10


def _validate_grid_bounds(sigma_bounds, tau_bounds_yr):
    """Reject bounds whose grid would be NaN instead of an error (#1585).

    Both axes are positive by construction and the quadrature runs in
    ``log(tau)``, so a non-positive, equal, or inverted bound yields an
    intact-looking all-NaN grid and raises nothing. ``log_prior`` stays finite
    over that grid under the default ``"log_uniform"`` prior, so no downstream
    field reports the corruption either.

    Parameters
    ----------
    sigma_bounds: tuple of float
        ``(lo, hi)`` amplitude support [dex].
    tau_bounds_yr: tuple of float
        ``(lo, hi)`` timescale support [yr].

    Raises
    ------
    ValueError
        If either bound pair is non-positive or not strictly increasing, or if
        the upper tau bound exceeds the ``ou_logpdf`` underflow threshold.
    """
    sigma_lower, sigma_upper = (float(b) for b in sigma_bounds)
    tau_lower, tau_upper = (float(b) for b in tau_bounds_yr)

    if tau_upper > _TAU_NAN_THRESHOLD_YR:
        raise ValueError(
            f"tau_bounds_yr[1]={tau_upper:.2e} yr exceeds {_TAU_NAN_THRESHOLD_YR:.2e} yr "
            f"(the underflow threshold where ou_logpdf returns NaN). "
            f"The age of the universe is {_AGE_UNIVERSE_YR:.2e} yr; bounds this large "
            f"suggest a units error (e.g., Myr where years are expected)."
        )
    if tau_lower <= 0.0:
        raise ValueError(
            f"tau_bounds_yr[0]={tau_lower:.3g} yr must be positive. The tau grid is "
            f"geometric and the quadrature runs in log(tau), so a non-positive lower "
            f"bound returns an all-NaN grid rather than raising -- and log_prior stays "
            f"finite over it, so nothing downstream reports the corruption. A "
            f"correlation timescale has no negative branch. Bounds centered on a truth "
            f"and scaled symmetrically (as in a prior-breadth sweep) cross zero once "
            f"the half-width exceeds the center; clip them at a small positive value."
        )
    if tau_lower >= tau_upper:
        raise ValueError(
            f"tau_bounds_yr must be strictly increasing, got "
            f"({tau_lower:.3g}, {tau_upper:.3g}) yr. The node spacing d_log_tau would be "
            f"zero or negative, making every log_volume entry -inf or NaN while the tau "
            f"nodes themselves stay finite."
        )
    if sigma_lower <= 0.0:
        raise ValueError(
            f"sigma_bounds[0]={sigma_lower:.3g} dex must be positive. sigma is a "
            f"modulation amplitude that reaches the DRW kernel only through "
            f"var=(sigma*ln10)^2, which is even -- a negative node returns a density "
            f"bit-identical to its positive mirror, so quadrature there double-counts "
            f"the amplitude axis on unphysical support. sigma=0 divides by that "
            f"variance and yields NaN. Use a small positive lower bound (e.g. 0.01)."
        )
    if sigma_lower >= sigma_upper:
        raise ValueError(
            f"sigma_bounds must be strictly increasing, got "
            f"({sigma_lower:.3g}, {sigma_upper:.3g}) dex. The node spacing d_sigma would "
            f"be zero or negative, making every log_volume entry -inf or NaN."
        )


@dataclass(frozen=True)
class SharedGrid:
    """Quadrature grid over the shared ``(sigma, tau)`` block.

    The grid spans ``(sigma, log tau)`` coordinates: sigma is uniform in
    amplitude [dex], and tau is sampled geometrically. The implied prior is
    uniform in ``(sigma, log tau)`` space, i.e., **log-uniform in tau**, this
    is deliberate. Quadrature weights account for the differential volume
    element in log-tau space, so ``log_volume`` is a constant (the integral of
    ``log(d_sigma) + log(d_log_tau)`` over all nodes).

    Attributes
    ----------
    sigma: ndarray, shape (A,)
        Amplitude nodes [dex].
    tau_yr: ndarray, shape (B,)
        Timescale nodes [yr].
    log_prior: ndarray, shape (A * B,)
        Log prior density at each node [nats], C-ordered so node ``a * B + b``
        is ``(sigma[a], tau_yr[b])``.
    log_volume: ndarray, shape (A * B,)
        Log quadrature weight of each node [nats], representing the differential
        volume element in ``(sigma, log tau)`` space.
    """

    sigma: jnp.ndarray
    tau_yr: jnp.ndarray
    log_prior: jnp.ndarray
    log_volume: jnp.ndarray

    @classmethod
    def uniform(cls, *, sigma_bounds, tau_bounds_yr, n_sigma, n_tau, tau_prior="log_uniform"):
        r"""Grid uniform in ``sigma`` and log-uniform in ``tau``.

        Creates a quadrature grid in ``(sigma, log tau)`` space. The implied
        prior is uniform over both dimensions. Tau nodes are spaced
        geometrically to ensure log-uniform coverage; the quadrature weights
        account for the Jacobian of the coordinate transformation.

        **Important:** Keep ``tau_bounds_yr`` within physically meaningful
        ranges (the age of the universe is 1.38e10 yr). At tau >= 1e20 yr,
        the exponential kernel ``exp(-|t_i - t_j| / tau)`` underflows to 1,
        causing ``ou_logpdf`` to return NaN. This typically signals a units
        error (e.g., Myr passed where years are expected). Bounds that exceed
        1e20 yr will raise ``ValueError``.

        **Both axes must be positive and strictly increasing.** sigma is an
        amplitude [dex] and tau a correlation timescale [yr]; neither has a
        negative branch. Bounds that violate this are rejected rather than
        propagated, because the grid is geometric in tau and the quadrature runs
        in ``log(tau)``: a non-positive or inverted bound produces an
        intact-looking, entirely NaN grid and raises nothing on its own. Note
        that ``log_prior`` stays finite over such a grid under the default
        ``"log_uniform"`` prior, so it cannot be used to detect the problem
        (issue #1585).

        Parameters
        ----------
        sigma_bounds: tuple of float
            ``(lo, hi)`` amplitude support [dex].
        tau_bounds_yr: tuple of float
            ``(lo, hi)`` timescale support [yr].
        n_sigma, n_tau: int
            Node counts.
        tau_prior: {"log_uniform", "uniform"}, optional
            Prior shape in tau. Default ``"log_uniform"`` (flat in log tau).
            Use ``"uniform"`` for a prior flat in tau itself.

            **This must match the prior the interim fits ran under.**
            ``shared_log_posterior`` uses ``log_prior`` twice: as the interim
            pushforward inside ``p_0``, and as the shared prior on the final
            posterior. The first is not a modeling choice -- it is a fact about
            how the per-galaxy draws were generated, and getting it wrong makes
            B2 divide by the wrong density. Fitting tau with
            ``Uniform(10, 500)`` [Myr] and scoring on a ``"log_uniform"`` grid
            mismatches the two by a factor proportional to tau, which is 50x
            end-to-end on that range.

            Neither the single-draw truth test nor the toy can catch this: with
            one draw per galaxy ``p_0`` is a per-galaxy constant that cancels in
            the normalization, and the toy generates and scores under the same
            prior. It only shows up with real multi-draw interim posteriors.

        Returns
        -------
        grid: SharedGrid

        Raises
        ------
        ValueError
            If either bound pair is non-positive or not strictly increasing, if the
            upper tau bound exceeds 1e20 yr, or if ``tau_prior`` is unrecognized.
        """
        _validate_grid_bounds(sigma_bounds, tau_bounds_yr)

        if tau_prior not in ("log_uniform", "uniform"):
            raise ValueError(
                f"tau_prior must be 'log_uniform' or 'uniform', got {tau_prior!r}. "
                "It must match the prior the INTERIM fits actually ran under, or "
                "the B2 pushforward correction divides by the wrong density."
            )

        sigma = jnp.linspace(sigma_bounds[0], sigma_bounds[1], n_sigma)
        tau = jnp.geomspace(tau_bounds_yr[0], tau_bounds_yr[1], n_tau)
        d_sigma = (sigma_bounds[1] - sigma_bounds[0]) / n_sigma
        d_log_tau = (jnp.log(tau_bounds_yr[1]) - jnp.log(tau_bounds_yr[0])) / n_tau
        n_nodes = n_sigma * n_tau

        # Quadrature runs in (sigma, log tau), so a node's weight must carry the
        # Jacobian of whichever prior is being represented:
        #     int f(tau) pi(tau) dtau = int f(tau) pi(tau) tau dlog(tau)
        # A log-uniform prior (pi ∝ 1/tau) cancels the tau and leaves flat
        # weights. A linear-uniform prior (pi = const) does not -- its node
        # weight is proportional to tau, spanning a factor of 50 across a
        # 10-500 Myr grid.
        if tau_prior == "uniform":
            log_w = jnp.log(jnp.tile(tau, n_sigma))
        else:
            log_w = jnp.zeros((n_nodes,))
        log_prior = log_w - jax.scipy.special.logsumexp(log_w)
        log_volume = jnp.full((n_nodes,), jnp.log(d_sigma) + jnp.log(d_log_tau))
        return cls(sigma=sigma, tau_yr=tau, log_prior=log_prior, log_volume=log_volume)

    @property
    def nodes(self):
        """``(A * B, 2)`` array of ``(sigma [dex], tau [yr])`` pairs."""
        a = jnp.repeat(self.sigma, self.tau_yr.size)
        b = jnp.tile(self.tau_yr, self.sigma.size)
        return jnp.stack([a, b], axis=-1)


def effective_sample_size(log_weights):
    r"""Kish effective sample size of log-domain importance weights.

    .. math:: \mathrm{ESS} = \frac{\left(\sum_k w_k\right)^2}{\sum_k w_k^2}

    Parameters
    ----------
    log_weights: array_like, shape (..., K)
        Unnormalized log weights [nats]; the reduction is over the last axis.

    Returns
    -------
    ess: ndarray, shape (...)
        Effective number of draws [dimensionless], in ``[1, K]``.
    """
    lw = jnp.asarray(log_weights)
    lw = lw - jnp.max(lw, axis=-1, keepdims=True)
    w = jnp.exp(lw)
    return jnp.sum(w, axis=-1) ** 2 / jnp.sum(w**2, axis=-1)


#: Grid nodes evaluated per streaming chunk. Peak memory is
#: ``node_chunk * N * K * 8`` bytes, independent of the grid size.
_DEFAULT_NODE_CHUNK = 128


def _node_logpdf_chunk(fields, times_yr, nodes):
    """``(C, N, K)`` field log-density for one chunk of ``C`` grid nodes."""

    def one_node(node):
        sigma, tau = node[0], node[1]
        mean = _field_mean(sigma)
        per_sample = jax.vmap(lambda m: ou_logpdf(m, mean, sigma, tau, times_yr))
        return jax.vmap(per_sample)(fields)

    return jax.lax.map(one_node, nodes)


def _chunk_nodes(nodes, coef, node_chunk):
    """Split nodes and their log-coefficients into equal chunks, padding with -inf.

    Returns ``(nodes_chunked, coef_chunked, n_nodes)`` with shapes
    ``(n_chunks, C, 2)`` and ``(n_chunks, C)``. Padded slots carry a
    coefficient of ``-inf`` so they contribute exactly zero to any
    ``logsumexp`` over nodes, and their per-node outputs are sliced away.
    """
    n_nodes = nodes.shape[0]
    chunk = int(min(node_chunk, n_nodes))
    n_chunks = -(-n_nodes // chunk)
    pad = n_chunks * chunk - n_nodes
    if pad:
        # Repeat a real node so ou_logpdf never sees a degenerate (0, 0) pair;
        # the -inf coefficient is what actually neutralizes the padding.
        nodes = jnp.concatenate([nodes, jnp.repeat(nodes[-1:], pad, axis=0)], axis=0)
        coef = jnp.concatenate([coef, jnp.full((pad,), -jnp.inf)], axis=0)
    return nodes.reshape(n_chunks, chunk, 2), coef.reshape(n_chunks, chunk), n_nodes


def shared_log_posterior(fields, times_yr, grid, *, method="b2", node_chunk=_DEFAULT_NODE_CHUNK):
    r"""Shared ``(sigma, tau)`` log-posterior from per-galaxy interim samples.

    The population factorizes given the shared block, so

    .. math::

        \log p(\sigma, \tau \mid \{d\}) = \log p(\sigma, \tau)
          + \sum_{i=1}^{N} \log\!\left[\frac{1}{K}\sum_{k}
            \frac{\mathcal{N}\big(m_i^{(k)};\ \mu(\sigma),\ K(\sigma,\tau)\big)}
                 {p_0\big(m_i^{(k)}\big)}\right]

    where :math:`m_i^{(k)}` are interim posterior draws of the centered field
    [natural-log units] and :math:`p_0` is the interim pushforward prior,
    evaluated by the same quadrature as the numerator so one grid serves both.

    Parameters
    ----------
    fields: array_like, shape (N, K, n)
        Interim centered-field draws [natural-log units].
    times_yr: array_like, shape (n,)
        Physical times [yr].
    grid: SharedGrid
        Quadrature grid.
    method: {"b2", "b1"}, optional
        ``"b2"`` (default) is the reweighting estimator. ``"b1"`` is the
        marginal-posterior product, retained as an independent cross-check
        whose error mode is different; see Notes.
    node_chunk: int, optional
        Grid nodes evaluated per streaming chunk [count]. Default 128. Peak
        memory is ``node_chunk * N * K * 8`` bytes and does **not** grow with
        the grid, so this is the knob that decouples population size from
        memory; see Notes.

    Returns
    -------
    log_posterior: ndarray, shape (A * B,)
        Unnormalized log-posterior [nats] on ``grid.nodes``.
    ess: ESSSummary
        Per-galaxy effective sample size diagnostic. Gate on ``ess.at_mode``
        (ESS at the posterior mode). Inspect ``ess.min_high_mass`` (minimum
        ESS over nodes carrying the top 99% of posterior mass) to detect
        degeneracy in the tails.

    Notes
    -----
    **JIT/vmap compatible**: yes. Cost is ``O(A B N K n)`` with no matrix
    factorization, because :func:`ou_logpdf` exploits the Markov structure.

    **Streaming, not materialized.** The ``(A B, N, K)`` node/galaxy/sample
    log-density table is never held whole. At a 60x60 grid with ``N = 256``
    and ``K = 500`` it would be 3.7 GB, and the importance weights are a
    second array of the same shape -- 7.4 GB live, which is what OOM-killed
    earlier sweeps. Instead ``log_p0`` is accumulated by a running-max
    ``logsumexp`` over node chunks (carrying only ``(N, K)``), and a second
    pass reduces each chunk to its ``(C, N)`` contribution. Peak memory is
    set by ``node_chunk``, so **population size is not memory-bounded**. The
    price is evaluating the table twice; it is the cheap stage, and buys the
    ceiling off the expensive one.

    ``"b2"`` fails by importance-weight degeneracy, which the returned
    ``ess.at_mode`` measures directly. ``"b1"`` fails by compounding
    density-estimation bias in the tails, which is *not* observable from
    inside the estimator -- multiplying N kernel density estimates whose
    tails err in a common direction shifts the result without widening it.
    Prefer ``"b2"``; use ``"b1"`` only to disagree with it.
    """
    fields = jnp.asarray(fields)
    times_yr = jnp.asarray(times_yr)
    if method not in ("b1", "b2"):
        raise ValueError(
            f"method must be 'b2' (production) or 'b1' (cross-check), got {method!r}."
        )

    n_samples = fields.shape[1]
    nodes_c, coef_c, n_nodes = _chunk_nodes(
        grid.nodes, grid.log_prior + grid.log_volume, node_chunk
    )

    # Pass 1 -- the interim pushforward prior p_0(m), on the same quadrature as
    # the numerator, accumulated by a running-max logsumexp over node chunks.
    # Carrying (running_max, running_sum) of shape (N, K) is what keeps the
    # (G, N, K) table off the heap.
    def _p0_step(carry, xs):
        run_max, run_sum = carry
        nodes_i, coef_i = xs
        t = _node_logpdf_chunk(fields, times_yr, nodes_i) + coef_i[:, None, None]
        chunk_max = jnp.max(t, axis=0)
        new_max = jnp.maximum(run_max, chunk_max)
        new_sum = run_sum * jnp.exp(run_max - new_max) + jnp.sum(jnp.exp(t - new_max), axis=0)
        return (new_max, new_sum), None

    carry_shape = (fields.shape[0], n_samples)
    init = (jnp.full(carry_shape, -jnp.inf), jnp.zeros(carry_shape))
    (p0_max, p0_sum), _ = jax.lax.scan(_p0_step, init, (nodes_c, coef_c))
    log_p0 = p0_max + jnp.log(p0_sum)  # (N, K)

    # Pass 2 -- reduce each chunk over samples. Only (C, N) survives per chunk,
    # so the importance weights are never materialized across the whole grid.
    def _weight_step(nodes_i):
        t = _node_logpdf_chunk(fields, times_yr, nodes_i)  # (C, N, K)
        lw = t - log_p0[None, :, :]
        b2 = jax.scipy.special.logsumexp(lw, axis=-1) - jnp.log(n_samples)
        # b1: the marginal-posterior product drops the pushforward correction
        # and uses the interim marginal directly. Deliberately a different
        # estimator, so it fails differently; see Notes.
        b1 = jax.scipy.special.logsumexp(t, axis=-1) - jnp.log(n_samples)
        return b2, b1, effective_sample_size(lw)

    b2_c, b1_c, ess_c = jax.lax.map(_weight_step, nodes_c)
    per_galaxy = (b2_c if method == "b2" else b1_c).reshape(-1, fields.shape[0])[:n_nodes]
    ess_per_node = ess_c.reshape(-1, fields.shape[0])[:n_nodes]  # (G, N)

    log_posterior = grid.log_prior + jnp.sum(per_galaxy, axis=-1)
    best = jnp.argmax(log_posterior)
    # Recompute the single best node rather than retaining every node's weights.
    log_w_best = _node_logpdf_chunk(fields, times_yr, grid.nodes[best][None, :])[0] - log_p0
    ess_at_mode = effective_sample_size(log_w_best)

    # ESS at minimum over nodes carrying top 99% posterior mass.
    p_normalized = jnp.exp(log_posterior - jax.scipy.special.logsumexp(log_posterior))
    cumsum_sorted = jnp.cumsum(jnp.sort(p_normalized)[::-1])
    high_mass_threshold = jnp.searchsorted(cumsum_sorted, 0.99)
    high_mass_mask = p_normalized >= jnp.sort(p_normalized)[-(high_mass_threshold + 1)]
    ess_min_high_mass = jnp.min(jnp.where(high_mass_mask[:, None], ess_per_node, jnp.inf), axis=0)

    return log_posterior, ESSSummary(at_mode=ess_at_mode, min_high_mass=ess_min_high_mass)
