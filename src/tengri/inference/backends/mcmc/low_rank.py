# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-`L` HMC with a low-rank mass matrix learned from warmup.

``inference/preconditioning.py``'s own module docstring states the gap this
fills, and states it as a limitation of the two options that exist today:

    "cond ~ 1e5 at the MAP. **A diagonal mass matrix cannot cover that, and a
    dense one estimated from warmup draws is both noisy and memory-hungry.**"

tengri has had two answers and no third. ``window_adaptation`` estimates either
a diagonal (cheap, blind to correlation) or a dense ``(D, D)`` Welford
covariance (the 20+ GB warmup spike that made ``dense_mass_matrix=False`` the
policy above D = 8, #319). ``blackjax.window_adaptation_low_rank`` is the
middle term: a rank-`k` correction to a diagonal, ``M^-1 = diag(s) (I + U (L -
I) U^T) diag(s)``, fitted by minimizing a Fisher divergence over the warmup
draws **and their gradients** rather than by Welford covariance over draws
alone.

**Why it is worth a measurement under a speed-first brief.** It changes nothing
about the sampling program's shape: the kernel is ``blackjax.hmc``, unchanged,
so the compiled scan is the same fixed-length leapfrog loop it already was and
the compile cost does not move. Whatever it buys, it buys at the same
per-draw cost -- unlike a learned trajectory length, which
``bench/reports/2026-08-31_catalog_preconditioning.md`` measured as a net
negative at 1.6x the wall clock.

**What it is not.** It is a *fixed* metric, adapted once from warmup. It does
not address the position-dependence that ``preconditioning.py``'s closing
paragraph names -- "one posterior standard deviation away the whitened
stiffness runs 3.7e2 to 1.7e5". It composes with the analytic
``J^T N^-1 J + I`` (that is a change of variables, this is a mass matrix
inside it), so the two can be measured together and separately, and the honest
question is whether the low-rank estimate adds anything once the analytic
metric is already there.

References
----------
.. [1] A. Seyboldt, "Preconditioning for Hamiltonian Monte Carlo with low-rank
   mass matrices," as implemented in nutpie and in
   ``blackjax.adaptation.low_rank_adaptation``. BlackJAX 1.6.2.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import (
    _maybe_map_init,
    _mean_params,
    _vmap_samples_to_physical,
)
from tengri.inference.backends.mcmc._shared import (
    _get_flat_logdensity,
    _hmc_chain_scan,
    _hmc_low_rank_full_scan,
)
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)

#: Default rank of the correction to the diagonal mass matrix.
#:
#: BlackJAX's own default, and nutpie's. Left unchanged rather than tuned:
#: tengri's photometric posteriors are D = 3-9, where ``max_rank=10`` is a
#: **full-rank** correction and the knob cannot bind. It becomes a real choice
#: only on the D = 75 stochastic-field posterior, which is not measured here.
DEFAULT_MAX_RANK: int = 10


def run_hmc_low_rank(
    context,
    *,
    key,
    init_from=None,
    n_warmup=1000,
    n_burnin=0,
    n_samples=1000,
    n_chains=1,
    n_leapfrog_steps=10,
    max_rank: int = DEFAULT_MAX_RANK,
    target_accept_rate=0.85,
    precondition: bool | float | None = None,
    verbose=True,
):
    """Fixed-`L` HMC whose mass matrix is a low-rank correction to a diagonal.

    Identical to ``run_hmc`` except for the warmup: the inverse mass matrix is
    a ``LowRankInverseMassMatrix`` pytree from
    ``blackjax.window_adaptation_low_rank`` instead of a ``(D,)`` or ``(D, D)``
    array from ``blackjax.window_adaptation``. The sampling kernel, the
    trajectory length and the compiled scan are unchanged, so a head-to-head
    against ``mcmc_hmc`` at the same ``n_leapfrog_steps`` isolates the mass
    matrix and nothing else.

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference target; see ADR-0010.
    key : PRNGKey
        Random key.
    init_from : optional
        Initial parameters, or ``None`` to run a MAP first.
    n_warmup : int, optional
        Window adaptation steps. Default 1000.
    n_burnin : int, optional
        Post-warmup draws discarded, Python-side. Default 0.
    n_samples : int, optional
        Kept draws per chain. Default 1000.
    n_chains : int, optional
        Independent chains, run sequentially from jittered starts and sharing
        one adaptation. Default 1.
    n_leapfrog_steps : int, optional
        Leapfrog steps per proposal. Default 10, matching the value
        ``bench/reports/2026-08-31_catalog_preconditioning.md`` measured
        against ChEES's learned length.
    max_rank : int, optional
        Rank of the correction to the diagonal. Default
        :data:`DEFAULT_MAX_RANK`. At D <= ``max_rank`` this is a full-rank
        correction and the knob does nothing, which is the regime every
        photometric fixture here sits in.
    target_accept_rate : float, optional
        Dual-averaging target [dimensionless]. Default 0.85, HMC's, so the row
        is comparable with ``mcmc_hmc``.
    precondition : bool, float, or None, optional
        Analytic ``J^T N^-1 J + I`` whitening strength. Composes with the
        low-rank mass matrix: the metric is a change of variables, the mass
        matrix lives inside it.
    verbose : bool, optional
        Log progress. Default True.

    Returns
    -------
    Posterior
        Draws in physical parameters. ``diagnostics`` carries ``n_divergent``,
        ``step_size``, ``max_rank`` and the preconditioning record. The inverse
        mass matrix is **not** reported: it is a pytree, not a number, and a
        float cast on it raises.

    Raises
    ------
    ImportError
        If blackjax is not installed.
    ValueError
        If ``max_rank`` is not a positive integer.

    Notes
    -----
    No adaptation caching. ``window_adaptation_low_rank`` returns a pytree
    rather than an array, and the shared adaptation cache is keyed and
    round-tripped for arrays; caching it is a correctness question this
    backend does not need answered to be measured, and a wrong cache hit is
    the failure mode that would be hardest to see.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for low-rank HMC: pip install blackjax") from None

    if not isinstance(max_rank, int) or max_rank < 1:
        raise ValueError(
            f"max_rank must be a positive int, got {max_rank!r}. It is the rank of "
            "the correction to the diagonal mass matrix; at max_rank >= D the "
            "correction is full rank."
        )

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    fitter = context.fitter

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)
    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat

    n_dim = len(init_flat)
    n_chains = max(1, int(n_chains))
    n_iter = int(n_burnin) + int(n_samples)

    if verbose:
        logger.info(
            "HMC (low-rank mass, max_rank=%d): %d parameters, %d warmup, "
            "%d burn-in, %d samples, %d leapfrog/step",
            max_rank,
            n_dim,
            n_warmup,
            n_burnin,
            n_samples,
            n_leapfrog_steps,
        )

    t0 = time.time()

    key, warmup_key = jax.random.split(key)
    key, chain_key = jax.random.split(key)
    chain_keys = jax.random.split(chain_key, n_chains * n_iter)
    chain_keys = chain_keys.reshape(n_chains, n_iter, 2)

    jitter_key, key = jax.random.split(key)
    starts = init_flat[None, :] + 1e-3 * jax.random.normal(jitter_key, shape=(n_chains, n_dim))

    # ONE warmup, shared by every chain, exactly as ``run_hmc`` does. Giving
    # each chain its own would be defensible on its own terms and would make
    # the head-to-head against ``mcmc_hmc`` cost ``n_chains`` times more warmup
    # on one side only -- and a comparison that differs in two things measures
    # neither (``bench/reports/2026-08-31_catalog_preconditioning.md``,
    # Finding 5).
    with compile_timer(
        "hmc_low_rank_full_scan",
        fitter.compile_signature(),
        method="mcmc_hmc_lowrank",
    ):
        pos, div, step_size, inv_mass_matrix = _hmc_low_rank_full_scan(
            starts[0],
            warmup_key,
            chain_keys[0],
            log_posterior_flat_2arg,
            data_args,
            int(n_warmup),
            int(n_leapfrog_steps),
            int(max_rank),
            float(target_accept_rate),
        )
        jax.block_until_ready(pos)
    positions_per_chain = [pos]
    divergent_per_chain = [div]

    for c in range(1, n_chains):
        state = blackjax.mcmc.hmc.init(starts[c], lambda p: log_posterior_flat_2arg(p, data_args))
        with compile_timer(
            "hmc_low_rank_chain_scan",
            fitter.compile_signature(),
            method="mcmc_hmc_lowrank",
        ):
            pos, div = _hmc_chain_scan(
                state,
                chain_keys[c],
                log_posterior_flat_2arg,
                data_args,
                step_size,
                inv_mass_matrix,
                int(n_leapfrog_steps),
            )
            jax.block_until_ready(pos)
        positions_per_chain.append(pos)
        divergent_per_chain.append(div)

    positions = jnp.stack(positions_per_chain)
    divergent = jnp.stack(divergent_per_chain)
    if n_burnin > 0:
        positions = positions[:, n_burnin:]
        divergent = divergent[:, n_burnin:]
    positions = positions.reshape(-1, positions.shape[-1])
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    positions = problem.restore(positions)
    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  Low-rank HMC complete in %.1fs. Divergences: %d/%d",
            wall_time,
            n_divergent,
            n_samples * n_chains,
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="HMC low-rank mass (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_leapfrog_steps": n_leapfrog_steps,
            "max_rank": int(max_rank),
            "n_divergent": n_divergent,
            "step_size": float(step_size),
            "preconditioned": bool(problem.enabled),
            "metric_condition": problem.metric_condition,
            "whitened_condition": problem.whitened_condition,
        },
        loss_history=None,
        _model=context.model,
    )
