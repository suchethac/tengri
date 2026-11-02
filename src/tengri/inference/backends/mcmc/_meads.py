"""MEADS warmup with an ensemble re-whitening phase, for the catalog ``"ghmc"`` sampler.

BlackJAX's ``meads_adaptation(...).run`` is one monolithic scan that cannot be paused, so the
two-phase schedule below is written out here. The per-step rule is Algorithm 3 of Hoffman and
Sountsov [1]_ (K folds, fold ``t mod K`` frozen, fold ``k`` tuned on fold ``k - 1``'s
statistics, chains reshuffled across folds every K steps). The two-phase schedule, the carried
adaptation state, and the redrawn momenta are those of the production MEADS runs of the DESI
dwarf fits (R. Wechsler and collaborators, private communication, 2026).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

#: Prior weight of Stan's covariance regularization [dimensionless].
_STAN_PRIOR_DRAWS = 5.0

#: Diagonal target of Stan's covariance regularization, in the units of the sampled space.
_STAN_DIAGONAL = 1.0e-3


class _WarmCarry(NamedTuple):
    """Warmup carry: chain states, iteration count, and the last per-fold parameters."""

    states: object
    t: jax.Array
    step_size: jax.Array
    scale: jax.Array
    alpha: jax.Array
    delta: jax.Array


def ensemble_metric(positions):
    """Return the mean and Cholesky factor of the Stan-regularized ensemble covariance.

    Parameters
    ----------
    positions : ndarray, shape (n, D)
        Positions of every chain over the second half of the first warmup phase [latent units].

    Returns
    -------
    center : ndarray, shape (D,)
        Mean position [latent units].
    factor : ndarray, shape (D, D)
        Lower Cholesky factor of the regularized covariance; non-finite when the regularized
        covariance is not positive definite.

    Notes
    -----
    **JIT/vmap-compatible**; not differentiated. With :math:`S` the sample covariance of
    :math:`n` positions,

    .. math:: \\Sigma = \\frac{n}{n+5} S + \\frac{5}{n+5}\\, 10^{-3} I,

    which is Stan's shrinkage of a windowed metric estimate [2]_. It is the estimator's
    shrinkage toward a small isotropic covariance, not a floor: it vanishes as :math:`n`
    grows and leaves the well-sampled directions' variances unchanged to O(1/n).

    References
    ----------
    .. [2] Stan Development Team, Stan Reference Manual, "HMC algorithm parameters:
       Euclidean metric" (regularized windowed covariance estimate).
    """
    n, dim = positions.shape
    center = positions.mean(axis=0)
    cov = jnp.cov(positions, rowvar=False).reshape(dim, dim)
    shrunk = n / (n + _STAN_PRIOR_DRAWS) * cov
    shrunk = shrunk + _STAN_PRIOR_DRAWS / (n + _STAN_PRIOR_DRAWS) * _STAN_DIAGONAL * jnp.eye(
        dim, dtype=positions.dtype
    )
    return center, jnp.linalg.cholesky(0.5 * (shrunk + shrunk.T))


def _warm_step_fn(ld: Callable, n_folds: int, n_chains: int, multiplier: float, slowdown: float):
    """Build one MEADS adaptation step (Algorithm 3) for the log-density ``ld``."""
    import blackjax.mcmc.ghmc as ghmc
    from blackjax.adaptation.meads_adaptation import maximum_eigenvalue

    kernel = ghmc.build_kernel()
    n_per = n_chains // n_folds

    def step(carry: _WarmCarry, key):
        states, t = carry.states, carry.t
        keys = jax.random.split(key, n_chains + 1)
        chain_keys, shuffle_key = keys[:n_chains], keys[n_chains]
        dim = states.position.shape[-1]
        fold_pos = states.position.reshape((n_folds, n_per, dim))
        fold_grad = states.logdensity_grad.reshape((n_folds, n_per, dim))
        fold_scale = fold_pos.std(axis=1)
        own_eps = jax.vmap(
            lambda g: jnp.minimum(multiplier / jnp.sqrt(maximum_eigenvalue(g)), 1.0)
        )(fold_grad * fold_scale[:, None, :])
        eps = jnp.roll(own_eps, 1)
        scale = jnp.roll(fold_scale, 1, axis=0)

        def damping(pos_k, eps_k):
            centered = pos_k - pos_k.mean(axis=0)
            gamma = jnp.maximum(
                1.0 / jnp.sqrt(maximum_eigenvalue(centered)), slowdown / ((t + 1) * eps_k)
            )
            alpha_k = 1.0 - jnp.exp(-2.0 * eps_k * gamma)
            return alpha_k, alpha_k / 2.0

        alpha, delta = jax.vmap(damping)(fold_pos / fold_scale[:, None, :], eps)
        per_chain = [jnp.repeat(v, n_per, axis=0) for v in (eps, scale, alpha, delta)]
        new, _ = jax.vmap(kernel, in_axes=(0, 0, None, 0, 0, 0, 0))(
            chain_keys, states, ld, *per_chain
        )
        if n_folds > 1:
            frozen = jnp.repeat(jnp.arange(n_folds) == t % n_folds, n_per)
            new = jax.tree.map(
                lambda nv, ov: jnp.where(
                    frozen.reshape(frozen.shape + (1,) * (nv.ndim - 1)), ov, nv
                ),
                new,
                states,
            )
            perm = jax.random.permutation(shuffle_key, n_chains)
            new = jax.lax.cond(
                (t + 1) % n_folds == 0,
                lambda s: jax.tree.map(lambda v: v[perm], s),
                lambda s: s,
                new,
            )
        return _WarmCarry(new, t + 1, eps, scale, alpha, delta), new.position

    return step


def _init_states(positions, ld, key):
    """Chain states at ``positions`` with fresh momenta, slices, log densities and gradients."""
    import blackjax.mcmc.ghmc as ghmc

    keys = jax.random.split(key, positions.shape[0])
    return jax.vmap(lambda p, k: ghmc.init(p, ld, k))(positions, keys)


def meads_rewhitened_warmup(
    ld: Callable,
    ensemble_init,
    warmup_key,
    n_warmup: int,
    n_folds: int,
    rewhiten_at: float,
    step_size_multiplier: float = 0.5,
    damping_slowdown: float = 1.0,
):
    """Run MEADS warmup, re-whitening the coordinates once from the ensemble.

    Parameters
    ----------
    ld : callable
        ``log_p(position)`` in the starting coordinates, with a non-finite value mapped to
        ``-inf`` and a zero gradient there.
    ensemble_init : ndarray, shape (n_chains, D)
        Starting positions [latent units].
    warmup_key : PRNGKey
        Key of the adaptation.
    n_warmup : int
        Total adaptation steps (static).
    n_folds : int
        MEADS folds K (static); ``n_chains`` is a multiple of it.
    rewhiten_at : float
        Fraction of ``n_warmup`` after which the coordinates are re-estimated from the ensemble
        (static); ``0`` disables the re-whitening.
    step_size_multiplier, damping_slowdown : float
        Algorithm 3 constants [dimensionless].

    Returns
    -------
    states : GHMCState
        Chain states in the final coordinates.
    center : ndarray, shape (D,)
        Origin of the final coordinates in the starting coordinates [latent units].
    factor : ndarray, shape (D, D)
        ``position = center + factor @ w``; the identity when ``rewhiten_at`` is 0.
    parameters : dict
        Fold-averaged ``step_size``, ``momentum_inverse_scale`` (diagonal, in the final
        coordinates), ``alpha`` and ``delta``.
    ok : ndarray, shape (), bool
        False when the ensemble covariance could not be factorized; the caller reports it.

    Notes
    -----
    **JIT-compatible**, not differentiated. At step ``n_warmup * rewhiten_at`` the center and
    factor come from :func:`ensemble_metric` over all chains' positions in the second half of
    the first phase; every chain is mapped to :math:`w = L^{-1}(x - c)`, its log density,
    gradient, momentum and slice are redrawn, and the adaptation continues with the iteration
    count, fold scales, step sizes and damping carried over. The log density in :math:`w` is
    that in :math:`x` plus the constant :math:`\\log|\\det L|`.

    References
    ----------
    .. [1] M. D. Hoffman and P. Sountsov, "Tuning-Free Generalized Hamiltonian Monte Carlo",
       AISTATS, PMLR 151:7799-7813, 2022. https://proceedings.mlr.press/v151/hoffman22a.html
    """
    n_chains, dim = ensemble_init.shape
    n_rw = round(rewhiten_at * n_warmup) if rewhiten_at else 0
    key_init, key_adapt, key_redraw = jax.random.split(warmup_key, 3)
    keys = jax.random.split(key_adapt, n_warmup)
    states = _init_states(ensemble_init, ld, key_init)
    carry = _WarmCarry(
        states,
        jnp.zeros((), jnp.int32),
        jnp.ones((n_folds,), ensemble_init.dtype),
        jnp.ones((n_folds, dim), ensemble_init.dtype),
        jnp.ones((n_folds,), ensemble_init.dtype),
        jnp.ones((n_folds,), ensemble_init.dtype),
    )
    step = _warm_step_fn(ld, n_folds, n_chains, step_size_multiplier, damping_slowdown)
    identity = jnp.eye(dim, dtype=ensemble_init.dtype)
    center = jnp.zeros((dim,), ensemble_init.dtype)
    factor = identity
    ok = jnp.ones((), bool)

    if n_rw:
        half = n_rw // 2
        carry, _ = jax.lax.scan(step, carry, keys[:half])
        carry, stacked = jax.lax.scan(step, carry, keys[half:n_rw])
        center, factor = ensemble_metric(stacked.reshape(-1, dim))
        ok = jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.isfinite(center))
        factor = jnp.where(ok, factor, identity)
        center = jnp.where(ok, center, 0.0)
        whitened = jax.scipy.linalg.solve_triangular(
            factor, (carry.states.position - center).T, lower=True
        ).T

        def ld_w(w):
            return ld(center + factor @ w)

        step = _warm_step_fn(ld_w, n_folds, n_chains, step_size_multiplier, damping_slowdown)
        carry = carry._replace(states=_init_states(whitened, ld_w, key_redraw))
        keys = keys[n_rw:]
    carry, _ = jax.lax.scan(step, carry, keys)
    parameters = {
        "step_size": carry.step_size.mean(),
        "momentum_inverse_scale": carry.scale.mean(axis=0),
        "alpha": carry.alpha.mean(),
        "delta": carry.delta.mean(),
    }
    return carry.states, center, factor, parameters, ok
