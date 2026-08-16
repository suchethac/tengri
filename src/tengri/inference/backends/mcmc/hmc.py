# SPDX-License-Identifier: BSD-3-Clause
"""Standard Hamiltonian Monte Carlo via BlackJAX.

Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import logging
import time
import warnings

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
from tengri.inference.backends.mcmc._shared import (
    _get_cached_adaptation,
    _get_flat_logdensity,
    _hmc_chain_scan,
    _hmc_warmup_only,
    _parallel_chains,
    _sequential_chains,
    _set_cached_adaptation,
    _vmap_chains,
)
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.utils.compile_log import compile_timer

logger = logging.getLogger(__name__)


def _resolve_chain_runner(chain_method, n_chains):
    """Pick the multi-chain executor: sequential, vmap, or parallel.

    - ``"sequential"``: loop the single-chain scan (peak memory = one chain);
      runs on modest RAM, wall ~ ``n_chains`` × one chain.
    - ``"vmap"``: SIMD-batch the chains (peak ~ ``n_chains`` × one chain);
      one device, wall ~ ``n_chains`` × one chain but a single kernel.
    - ``"parallel"``: one chain per device via ``jax.pmap`` (~one chain's wall
      with enough forced host devices); falls back to vmap with a warning when
      too few devices are visible, so a fit never fails for want of ``XLA_FLAGS``.

    """
    if chain_method == "sequential":
        return _sequential_chains
    if chain_method == "vmap":
        return _vmap_chains
    if chain_method == "parallel":
        if jax.device_count() >= n_chains:
            return _parallel_chains
        warnings.warn(
            f"chain_method='parallel' needs >= {n_chains} JAX devices, found "
            f"{jax.device_count()}; falling back to vmap. On CPU set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={n_chains} before importing "
            f"jax / tengri to enable true parallel chains.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _vmap_chains
    raise ValueError(
        f"chain_method must be 'sequential', 'vmap', or 'parallel', got {chain_method!r}"
    )


def run_hmc(
    context,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
    n_leapfrog_steps=10,
    target_accept_rate=0.85,
    dense_mass_matrix=None,
    chain_method="vmap",
    precondition: bool | float | None = None,
    verbose=True,
):
    """HMC sampling via BlackJAX.

    Standard Hamiltonian Monte Carlo with fixed trajectory length.
    Predictable cost per step (no tree building), making it faster
    than NUTS per sample when the geometry is well-conditioned.

    Parameters
    ----------
    n_warmup : int
        Warmup/adaptation steps (tunes step size and mass matrix).
    n_burnin : int
        Post-warmup burn-in steps (discarded). Discarded Python-side
        rather than inside JIT, so changing this does NOT trigger a
        recompile when ``n_burnin + n_samples`` is unchanged.
    n_samples : int
        Posterior samples per chain to collect.
    n_chains : int, default 1
        Number of independent HMC chains, sharing one adapted step size and
        mass matrix. Warmup is adapted once; the chains then sample from
        jittered starts, honored on the **first** call as well as cached ones.
        Final posterior has ``n_chains * n_samples`` samples. Under the default
        ``chain_method="vmap"`` the chains are SIMD-batched, so wall scales ~
        linearly with ``n_chains`` on CPU.
    n_leapfrog_steps : int
        Number of leapfrog integration steps per proposal.
    target_accept_rate : float
        Target acceptance rate for step size adaptation.
    dense_mass_matrix : bool or None, default None
        ``None`` (auto) switches to diagonal at D >= 8, the same policy NUTS
        uses (:func:`~tengri.inference.backends.mcmc.nuts._resolve_dense_mass_matrix`,
        #319). ``True`` / ``False`` force the choice.

        This used to default to ``True``, which combined with the
        ``n_dim <= 30`` cap below meant HMC ran a **dense** mass matrix over
        the whole D = 8-30 band — exactly the band where NUTS deliberately
        switches to diagonal to dodge the 20+ GB warmup spike. Measured
        consequence: ``mcmc_hmc`` at D = 9 peaked at 13.47 GB and was
        SIGKILLed, while ``mcmc_nuts`` at the same D was already diagonal
        (#1413, #1454). The high-D advisory could not catch it either — it
        fires above D = 30, by which point HMC has *stopped* using dense.
    chain_method : {"vmap", "sequential", "parallel"}, default "vmap"
        How ``n_chains > 1`` chains are executed.

        - ``"vmap"`` (default): SIMD-batch the chains into one kernel. Peak
          memory ~ ``n_chains`` × one chain — can OOM a dense-mass fit on
          modest RAM.
        - ``"sequential"``: loop the single-chain scan (compiled once, reused).
          **Peak memory = one chain**, so it runs on cheap hardware; wall ~
          ``n_chains`` × one chain. Prefer this when RAM is the constraint.
        - ``"parallel"``: one chain per device via ``jax.pmap`` (~one chain's
          wall) — on CPU needs
          ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set before
          importing jax; falls back to ``"vmap"`` with a warning if fewer than
          ``n_chains`` devices are visible.

    precondition : bool, float or None, default None
        Sample in metric-whitened coordinates (#1301): the metric is built
        analytically at the initial point and the chain samples ``H(A zeta)``
        with ``A A^T = G^-alpha``, draws mapped back exactly — the posterior is
        unchanged, only the integrator's geometry. **Opt-in** (#1397): ``None``
        (default) and ``False`` are off; ``True`` uses
        :data:`~tengri.inference.preconditioning.DEFAULT_WHITENING_STRENGTH`, and
        a float in ``[0, 1]`` sets the strength (``1.0`` is full whitening).
        Full whitening amplifies a misspecified metric without bound (#1442).
        See :mod:`tengri.inference.preconditioning`.
    verbose : bool
        Print progress.
    """
    try:
        import blackjax
    except ImportError:
        raise ImportError("blackjax required for HMC: pip install blackjax") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    # ``_shared.py`` helpers still take a Fitter; reach through
    # the context until they migrate.
    context = InferenceContext.from_target(context)
    fitter = context.fitter

    init_params, key = _maybe_map_init(fitter, key, init_from, verbose)

    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        fitter,
        init_params,
    )

    # Metric preconditioning (#1301) — see ``run_nuts`` for the rationale. Linear change
    # of variables, so the posterior is untouched; draws are mapped back below.
    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    log_posterior_flat_2arg, init_flat = problem.logdensity, problem.init_flat

    n_dim = len(init_flat)

    # One definition of the policy, shared with NUTS, rather than a second
    # heuristic that drifts from it (#1454). The `n_dim <= 30` cap stays as a
    # backstop for an explicit `dense_mass_matrix=True` on a large problem.
    from tengri.inference.backends.mcmc.nuts import (
        _maybe_warn_high_memory_nuts,
        _resolve_dense_mass_matrix,
    )

    use_dense = _resolve_dense_mass_matrix(dense_mass_matrix, n_dim) and n_dim <= 30
    # HMC carries the same O(D^2) warmup cost as NUTS, so it gets the same
    # warning. It never had one: the shared high-D advisory keys on method name
    # and fires above D = 30, where HMC has already fallen back to diagonal.
    _maybe_warn_high_memory_nuts(n_dim, use_dense, getattr(fitter, "spec", None))

    if verbose:
        burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
        logger.info(
            "HMC: %d parameters, %d warmup%s, %d samples, %d leapfrog/step",
            n_dim,
            n_warmup,
            burnin_msg,
            n_samples,
            n_leapfrog_steps,
        )

    t0 = time.time()

    # n_warmup, n_leapfrog_steps and target_accept_rate belong in the key: they
    # *produce* the adaptation, so leaving them out makes those knobs silently
    # inert on a model that already holds an entry.
    adapt_key = (
        "hmc",
        not use_dense,
        int(n_warmup),
        int(n_leapfrog_steps),
        float(target_accept_rate),
        problem.cache_key,
    )
    cached = _get_cached_adaptation(fitter, adapt_key)

    def ld_1arg(pos):
        return log_posterior_flat_2arg(pos, data_args)

    # ── Warmup: adapt (step_size, inverse_mass_matrix) once, then cache. ──
    # Split from sampling so the FIRST call honors n_chains too (previously a
    # fresh multi-chain run silently sampled a single chain and mislabelled it).
    #
    # The warmup split is hoisted out of the ``else`` so both branches advance
    # the key identically. Cache presence is invisible to the caller and must
    # not steer the RNG stream, or two identical ``fit`` calls with one ``key``
    # return different chains. ``warmup_key`` is unused on the cached path.
    key, warmup_key = jax.random.split(key)

    if cached is not None:
        parameters = cached
        if verbose:
            logger.info(
                "  Reusing cached warmup (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )
    else:
        with compile_timer("hmc_warmup", fitter.compile_signature(), method="mcmc_hmc"):
            step_size, inv_mass_matrix = _hmc_warmup_only(
                init_flat,
                warmup_key,
                log_posterior_flat_2arg,
                data_args,
                n_warmup,
                n_leapfrog_steps,
                use_dense,
                target_accept_rate,
            )
            jax.block_until_ready(step_size)
        parameters = {"step_size": step_size, "inverse_mass_matrix": inv_mass_matrix}
        _set_cached_adaptation(fitter, adapt_key, parameters)
        if verbose:
            logger.info(
                "  Warmup complete (%.1fs). Step size: %.4f",
                time.time() - t0,
                float(parameters["step_size"]),
            )

    # ── Sampling: single- or multi-chain, honored on first AND cached calls. ──
    key, chain_key = jax.random.split(key)
    if n_chains > 1:

        def _init(p):
            return blackjax.mcmc.hmc.init(p, ld_1arg)

        def _scan(s, ks):
            return _hmc_chain_scan(
                s,
                ks,
                log_posterior_flat_2arg,
                data_args,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
                n_leapfrog_steps,
            )

        chain_runner = _resolve_chain_runner(chain_method, n_chains)
        with compile_timer("hmc_chain_scan_vmap", fitter.compile_signature(), method="mcmc_hmc"):
            positions, divergent = chain_runner(
                _init,
                _scan,
                init_flat=init_flat,
                chain_key=chain_key,
                n_chains=n_chains,
                n_iter=n_burnin + n_samples,
                n_burnin=n_burnin,
            )
            jax.block_until_ready(positions)
        _multichain_burnin_done = True
    else:
        state = blackjax.mcmc.hmc.init(init_flat, ld_1arg)
        chain_keys = jax.random.split(chain_key, n_burnin + n_samples)
        with compile_timer("hmc_chain_scan", fitter.compile_signature(), method="mcmc_hmc"):
            positions, divergent = _hmc_chain_scan(
                state,
                chain_keys,
                log_posterior_flat_2arg,
                data_args,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
                n_leapfrog_steps,
            )
            jax.block_until_ready(positions)
        _multichain_burnin_done = False

    # Burnin discard happens Python-side. Multichain branch already
    # discarded per-chain before flattening; skip the global slice there.
    if n_burnin > 0 and not _multichain_burnin_done:
        positions = positions[n_burnin:]
        divergent = divergent[n_burnin:]
    n_divergent = int(jnp.sum(divergent))

    wall_time = time.time() - t0

    # Leave the whitened coordinates before the draws are read as parameters.
    positions = problem.restore(positions)

    samples_phys = _vmap_samples_to_physical(positions, unravel_fn, context.to_physical)
    best_params = _mean_params(samples_phys)

    if verbose:
        logger.info(
            "  HMC complete in %.1fs. Divergences: %d/%d", wall_time, n_divergent, n_samples
        )

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="HMC (BlackJAX)",
        wall_time_s=wall_time,
        diagnostics={
            "n_warmup": n_warmup,
            "n_burnin": n_burnin,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_leapfrog_steps": n_leapfrog_steps,
            "n_divergent": n_divergent,
            "step_size": float(parameters["step_size"]),
        },
        loss_history=None,
        _model=context.model,
    )
