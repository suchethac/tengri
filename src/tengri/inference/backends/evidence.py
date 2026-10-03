# SPDX-License-Identifier: BSD-3-Clause
"""Nested Slice Sampling for Bayesian evidence (log Z).

Extracted from fitter.py.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from tengri.inference._model_cache import _default_owner as _model_cache_owner

_N_VOLUME_REALIZATIONS = 100  # simulated prior-volume sequences for H and log Z scatter


def _nss_information_and_error(log_w, logL, n_live, num_delete):
    r"""Information H and evidence error of a nested-sampling run.

    Returns the information H of the run, the standard deviation of
    :math:`\log Z` that follows from it, and the effective live count used.

    Parameters
    ----------
    log_w : ndarray, shape (n_dead, K)
        Unnormalized log weights :math:`\log(L_i\,\Delta X_i)`, one column per
        simulated sequence of prior volumes (K realizations).
    logL : ndarray, shape (n_dead,)
        Log-likelihood of each dead point, in nats.
    n_live : int
        Number of live points.
    num_delete : int
        Points removed per iteration.

    Returns
    -------
    H : float
        Information (Kullback-Leibler divergence of posterior from prior), nats.
    log_evidence_err : float
        Standard deviation of :math:`\log Z`, nats.
    n_eff : float
        Effective live count, dimensionless.

    Raises
    ------
    ValueError
        If ``num_delete`` is not in [1, n_live), or if H is negative by more
        than round-off (1e-6 nats).

    Notes
    -----
    For realization k the posterior weights are
    :math:`p_{ik} = \exp(\log w_{ik} - \log Z_k)` with
    :math:`\log Z_k = \log\sum_i \exp(\log w_{ik})`, and

    .. math::

        H = \frac{1}{K}\sum_{k=1}^{K}\Bigl(\sum_i p_{ik}\log L_i - \log Z_k\Bigr).

    Removing ``num_delete`` = :math:`m` points per iteration from a live set of
    :math:`n` shrinks :math:`\ln X` at the j-th removal by a mean of
    :math:`1/(n-j)` with variance :math:`1/(n-j)^2`. Reaching the posterior
    bulk takes about H divided by the mean shrinkage per removal, so

    .. math::

        n_\mathrm{eff} = \frac{\sum_{j=0}^{m-1} (n-j)^{-1}}
                              {\sum_{j=0}^{m-1} (n-j)^{-2}},
        \qquad
        \sigma(\log Z) = \sqrt{H / n_\mathrm{eff}}.

    For :math:`m = 1` this is :math:`n_\mathrm{eff} = n` and Skilling's
    :math:`\sqrt{H/n}` [1]_. The result is leading order in :math:`1/n`: it
    treats the log prior volume at the posterior bulk as Gaussian.

    References
    ----------
    .. [1] Skilling, J. (2006). Nested sampling for general Bayesian
           computation. Bayesian Analysis, 1(4), 833-859.
           doi:10.1214/06-BA127
    """
    if num_delete < 1 or num_delete >= n_live:
        raise ValueError(
            f"num_delete ({num_delete}) must satisfy 1 <= num_delete < n_live ({n_live})"
        )
    log_w = np.asarray(log_w, dtype=np.float64)
    logL = np.asarray(logL, dtype=np.float64)

    log_z_k = logsumexp(log_w, axis=0)
    p = np.exp(log_w - log_z_k)
    # p == 0 where a weight underflows; 0 * (-inf) would be NaN, the term is 0.
    h_k = np.sum(np.where(p > 0, p * logL[:, None], 0.0), axis=0) - log_z_k
    information = float(np.mean(h_k))

    if information < -1e-6:
        raise ValueError(
            f"information H = {information:.3e} nats is negative beyond round-off "
            f"(log_w shape {log_w.shape}, logL shape {logL.shape}, "
            f"log_w range [{log_w.min():.2f}, {log_w.max():.2f}])"
        )
    if information < 0.0:
        information = 0.0

    removal = n_live - np.arange(num_delete, dtype=np.float64)
    # Reciprocal once, then square it (the float32 guard bans 1/x**2 spellings
    # tree-wide; removal is an O(n_live) count, but the idiom is the rule).
    inv_removal = 1.0 / removal
    n_eff = float(np.sum(inv_removal) / np.sum(inv_removal**2))
    return information, float(np.sqrt(information / n_eff)), n_eff


def _resolve_nss_settings(preset, n_live, num_delete, log_evidence_tol, max_shrinkage):
    """Expand preset name to tuple of settings, with explicit kwargs overriding.

    Parameters
    ----------
    preset : str or None
        Preset name: "fast", "accurate", or None (defaults to "accurate").
    n_live : int or None
        Number of live points (overrides preset if not None).
    num_delete : int or None
        Points to replace per iteration (overrides preset if not None).
    log_evidence_tol : float or None
        Termination tolerance on log(Z_remaining) (overrides preset if not None).
    max_shrinkage : int or None
        Maximum shrinkage steps (overrides preset if not None).

    Returns
    -------
    tuple of (int, int, float, int)
        Resolved (n_live, num_delete, log_evidence_tol, max_shrinkage).

    Raises
    ------
    ValueError
        If preset is not in ("fast", "accurate", None).
    """
    presets = {
        "fast": (100, 20, -2.0, 10),
        "accurate": (500, 50, -3.0, 20),
        None: (500, 50, -3.0, 20),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}. Valid options: 'fast', 'accurate', or None.")

    default_n_live, default_num_delete, default_log_tol, default_shrink = presets[preset]

    return (
        n_live if n_live is not None else default_n_live,
        num_delete if num_delete is not None else default_num_delete,
        log_evidence_tol if log_evidence_tol is not None else default_log_tol,
        max_shrinkage if max_shrinkage is not None else default_shrink,
    )


def _get_nss_fns(
    fitter,
    *,
    num_inner_steps,
    num_delete,
    max_steps,
    max_shrinkage,
):
    """Return (init_jit, step_jit) cached on the model.

    Both functions accept ``data_args`` as a *traced* JAX argument so that the
    compiled XLA program is generic; it is reused for every galaxy that shares
    the same model dimensionality and data shape, with zero recompilation.

    The functions are keyed by model configuration and stored in the model-level
    WeakKeyDictionary cache so they are garbage-collected with the model.
    """
    cache_key = (
        fitter._engine_cache_key(),
        "nss",
        num_inner_steps,
        num_delete,
        max_steps,
        max_shrinkage,
    )
    cache = _model_cache_owner.get_or_compile_model(fitter.model).setdefault("nss_fns", {})

    if cache_key not in cache:
        from tengri.inference.backends.nested.nss import as_top_level_api

        logprior_fn = fitter._build_logprior_fn()
        loglikelihood_fn = fitter._get_or_build_loglikelihood_fn()

        # Build algo *inside* the JIT-traced function so that data_args is
        # abstract (traced) rather than a Python constant.  JAX traces these
        # once; subsequent calls with the same Python function object and the
        # same abstract shapes hit the Python-level JIT cache without retrace.
        def _step_with_data(key, state, data_args):
            """Advance nested sampling state by one iteration with the given data."""

            def _loglik(params):
                """Evaluate likelihood for given parameters."""
                return loglikelihood_fn(params, data_args)

            _algo = as_top_level_api(
                logprior_fn,
                _loglik,
                num_inner_steps,
                num_delete=num_delete,
                max_steps=max_steps,
                max_shrinkage=max_shrinkage,
            )
            return _algo.step(key, state)

        def _init_with_data(particles, data_args):
            """Initialize nested sampling state with particles and data."""

            def _loglik(params):
                """Evaluate likelihood for given parameters."""
                return loglikelihood_fn(params, data_args)

            _algo = as_top_level_api(
                logprior_fn,
                _loglik,
                num_inner_steps,
                num_delete=num_delete,
                max_steps=max_steps,
                max_shrinkage=max_shrinkage,
            )
            return _algo.init(particles)

        cache[cache_key] = (jax.jit(_init_with_data), jax.jit(_step_with_data))

    return cache[cache_key]


def run_nss(
    context,
    *,
    key,
    init_from=None,
    preset=None,
    n_live=None,
    num_delete=None,
    num_inner_steps=None,
    log_evidence_tol=None,
    max_iterations=10000,
    n_posterior_samples=1000,
    max_steps=10,
    max_shrinkage=None,
    verbose=True,
):
    """Nested Slice Sampling for Bayesian evidence computation.

    Uses Hit-and-Run Slice Sampling (HRSS) as the inner kernel.
    Based on Yallup, Kroupa & Handley (2026, arXiv:2601.23252).

    Restricted to parametric (non-stochastic) SFH models where D ≲ 30.

    Parameters
    ----------
    preset : str or None
        Preset configuration for evidence accuracy vs. speed. Options:

        - ``"fast"``: n_live=100, num_delete=20, log_evidence_tol=-2.0,
          max_shrinkage=10. Evidence scatter σ_logZ ≈ 0.3–0.45 nats; suitable
          when Δlog Z between models ≳ 1. Wall time ~2–3× faster than "accurate".
        - ``"accurate"``: n_live=500, num_delete=50, log_evidence_tol=-3.0,
          max_shrinkage=20 (default). Evidence scatter σ_logZ ≈ 0.15–0.2 nats;
          calibrated reference.
        - ``None``: equivalent to ``"accurate"``.

        Explicit non-None arguments (n_live, num_delete, etc.) always override
        preset values.
    n_live : int or None
        Number of live points. Overrides preset if not None.
    num_delete : int or None
        Points to replace per iteration. Overrides preset if not None.
    num_inner_steps : int or None
        HRSS walk length per replacement. Defaults to D.
    log_evidence_tol : float or None
        Terminate when log(Z_remaining) - log(Z_accumulated) < this.
        Overrides preset if not None.
    max_iterations : int
        Safety limit on iterations.
    n_posterior_samples : int
        Number of posterior samples to draw after convergence.
    max_steps : int
        Maximum stepping-out steps in slice sampling.
    max_shrinkage : int or None
        Maximum shrinking steps in slice sampling. Overrides preset if not None.
        Default behavior (when None + no preset override) is 20 to limit XLA
        graph size, since each shrinkage step is compiled into ``vmap(lax.while_loop)``
        body, and ``max_shrinkage=100`` caused 20 GB+ JIT compilation memory.
    verbose : bool
        Print progress.

    Notes
    -----
    **Preset rationale**: Evidence scatter scales as σ_logZ ≈ √(H/n_eff) where
    H is the information and n_eff is the effective live count accounting for
    batch deletion (n_eff = n_live for single-point deletions). "fast" (n=100,
    k=20, n_eff≈89.8) trades ~2–3× wall time for σ_logZ ≈ 0.3–0.45 nats, fine
    for BMA when Δlog Z ≳ 1. "accurate" (n=500, k=50, n_eff≈474.6) provides
    σ_logZ ≈ 0.15–0.2 nats. The required live set must satisfy n_live > D
    (number of free parameters); a guard checks this after preset resolution.

    **Cross-galaxy cache reuse**

    The compiled XLA step function is cached on the ``SEDModel`` object via
    the default ``tengri.inference._model_cache.ModelCacheOwner``.  ``data_args`` is
    passed as a *traced* JAX value (not a compile-time constant), so the same
    compiled program is reused for every galaxy that shares the same model
    dimensionality and photometric band layout.  The cache is keyed on
    ``(model_cache_key, num_inner_steps, num_delete, max_steps, max_shrinkage)``.

    **XLA compilation size**: each ``lax.while_loop`` shrinkage step adds nodes
    to the compiled XLA graph when ``jax.vmap`` batches it over ``num_delete``
    particles.  Increase max_shrinkage only if acceptance rates fall below ~0.5.

    Cold compile (~10–15 s) happens once per model configuration; subsequent
    galaxies pay only the per-step XLA execution time.

    JIT/grad/vmap: the step body is fully JIT-compatible.
    """
    from tengri.inference.backends.nested.base import NSInfo as _NSInfo
    from tengri.inference.backends.nested.utils import (
        ess as ns_ess,
        log_weights as ns_log_weights,
        sample as ns_sample,
    )
    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    # ``_get_nss_fns`` and the spec/fixed-values dict still live on the Fitter.
    fitter = context.fitter

    if context.spec.stochastic:
        raise ValueError(
            "NSS not supported for stochastic SFH models (D~137). "
            "Use 'vi' or 'mcmc_raytrace' instead."
        )

    D = len(context.free_names)
    n_live, num_delete, log_evidence_tol, max_shrinkage = _resolve_nss_settings(
        preset, n_live, num_delete, log_evidence_tol, max_shrinkage
    )

    if n_live <= D:
        raise ValueError(
            f"n_live ({n_live}) must be > D ({D}, number of free parameters). "
            f"Nested sampling needs strictly more live points than dimensions."
        )

    if num_inner_steps is None:
        num_inner_steps = D

    if verbose:
        print(
            f"NSS: {D} params, {n_live} live, {num_delete} del/iter, {num_inner_steps} HRSS steps"
        )

    init_jit, step_jit = _get_nss_fns(
        fitter,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    data_args = context.data_args

    key, init_key = jax.random.split(key)
    all_samples = context.spec.sample_batch(init_key, n_live)
    particles = {name: all_samples[name] for name in context.free_names}
    live = init_jit(particles, data_args)

    # Collect only dead particles, not the full NSInfo (update_info is MCMC internals
    # of the replacement step, 3-4× larger than particles but unused by ns_sample/ns_ess).
    dead_particles_list = []
    n_iter = 0
    t0 = time.time()

    while True:
        key, subkey = jax.random.split(key)
        live, dead = step_jit(subkey, live, data_args)
        dead_particles_list.append(dead.particles)
        n_iter += 1

        logZ_est = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
        remaining = float(live.integrator.logZ_live - live.integrator.logZ)

        if verbose and n_iter % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  NSS iter {n_iter}: log Z ≈ {logZ_est:.2f}, "
                f"n_dead={n_iter * num_delete}, "
                f"elapsed={elapsed:.1f}s"
            )

        if remaining < log_evidence_tol:
            break
        if n_iter >= max_iterations:
            if verbose:
                print("  NSS: max iterations reached")
            break

    wall_time = time.time() - t0
    logZ = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))

    all_ps = [*dead_particles_list, live.particles]
    del dead_particles_list
    final_particles = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_ps)
    ns_run = _NSInfo(final_particles, None)

    key, sample_key = jax.random.split(key)
    resampled = ns_sample(sample_key, ns_run, n_posterior_samples)

    key, ess_key = jax.random.split(key)
    ess_val = float(ns_ess(ess_key, ns_run))

    key, weight_key = jax.random.split(key)
    information_nats, log_evidence_err, n_live_effective = _nss_information_and_error(
        np.asarray(ns_log_weights(weight_key, ns_run, shape=_N_VOLUME_REALIZATIONS)),
        np.asarray(ns_run.particles.loglikelihood),
        n_live,
        num_delete,
    )

    samples_phys = {name: resampled.position[name] for name in context.free_names}
    for name, val in fitter._fixed_values.items():
        samples_phys[name] = jnp.full(n_posterior_samples, val)

    best_params = {k: jnp.median(v, axis=0) for k, v in samples_phys.items()}

    if verbose:
        print(f"  NSS complete in {wall_time:.1f}s. log Z = {logZ:.2f}, ESS = {ess_val:.0f}")

    return Posterior(
        samples=samples_phys,
        params=best_params,
        method="NSS (Yallup+2026)",
        wall_time_s=wall_time,
        diagnostics={
            "n_live": n_live,
            "num_delete": num_delete,
            "num_inner_steps": num_inner_steps,
            "n_iterations": n_iter,
            "n_dead": n_iter * num_delete,
            "log_evidence": logZ,
            "log_evidence_err": log_evidence_err,
            "information_nats": information_nats,
            "n_live_effective": n_live_effective,
            "ess": ess_val,
        },
        log_evidence=logZ,
        _model=context.model,
    )
