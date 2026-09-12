# SPDX-License-Identifier: BSD-3-Clause
"""MAP optimization, Laplace approximation, and Pathfinder.

Extracted from fitter.py. Migrated to the :class:`InferenceContext`
Protocol, accepts either an ``InferenceContext`` or a ``Fitter`` (the
latter is normalized internally during the multi-PR migration window).
See ADR-0010.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize as _jax_minimize

from tengri.inference._model_cache import _default_owner as _model_cache_owner
from tengri.inference.context import InferenceContext
from tengri.inference.likelihoods.gaussian import inv_noise_std

logger = logging.getLogger(__name__)

_OPTAX_OPTIMIZERS = {"adam", "adamw", "sgd"}
_SCIPY_OPTIMIZERS = {"lbfgs", "lbfgs_scipy"}
_ALL_OPTIMIZERS = _OPTAX_OPTIMIZERS | _SCIPY_OPTIMIZERS

# Short-form name aliases used in fitter and tests
_JAXOPT_SOLVERS = _SCIPY_OPTIMIZERS
_QUASI_NEWTON = _SCIPY_OPTIMIZERS


def _publish_map_init_cache(context, posterior):
    """Share an explicit MAP result with the sampler's MAP-init cache.

    ``_maybe_map_init`` seeds HMC/NUTS/VI from a cached MAP point, but only ever
    saw the short MAP *it* ran itself. An explicit ``fit(method='map')`` wrote
    nothing there, so the ordinary two-step workflow (optimize, then sample)
    silently paid for a **second** MAP: a full user-configured run, followed by
    another 1000-step one inside the sampler, on the same model and the same
    data.

    Publishing here closes that. The entry is stamped with
    ``tengri.inference._sample_utils._data_fingerprint``, the same guard
    the sampler's own writes use, so it can never seed a different target
    (issue #1529), and a user's MAP is typically the better starting point
    anyway, being run at their chosen ``n_steps`` / ``n_restarts`` rather than
    the init default.

    Returns ``posterior`` unchanged so call sites can wrap a return directly.
    """
    from tengri.inference._sample_utils import _data_fingerprint

    params = getattr(posterior, "params", None)
    if not params:
        return posterior
    try:
        mc = _model_cache_owner.get_or_compile_model(context.model)
        mc["map_params_physical"] = {k: jnp.asarray(v) for k, v in params.items()}
        mc["map_data_fingerprint"] = _data_fingerprint(context.fitter)
    except (AttributeError, TypeError, ValueError) as exc:
        # A cache is an optimization and must not fail a completed fit -- but it
        # says so out loud. A blanket ``except Exception: pass`` here is the
        # shape that hid an UnboundLocalError in the hybrid-photometry builder
        # for weeks, silently costing a 107x larger compiled graph.
        logger.debug("MAP-init cache not published: %s: %s", type(exc).__name__, exc)
    return posterior


def _reject_nonfinite_map(params: dict) -> None:
    """Refuse to hand a non-finite MAP point downstream (#1397).

    A MAP estimate containing NaN or inf is not a usable answer, and it is not a
    private problem either: every MCMC and VI backend takes ``init_from`` from here,
    so one bad point poisons whatever runs next. In #1397 it produced a NUTS run whose
    every parameter was NaN and whose ``rhat()`` was NaN, which silently disarms a
    notebook's ``max_rhat < 1.01`` assertion, because NaN compares false against any
    threshold. Nothing raised.

    Any non-finite coordinate is rejected, not only an all-NaN point: the optimizer
    has already failed by then, and a partially-NaN start is no more samplable than a
    fully-NaN one.

    Parameters
    ----------
    params : dict
        Parameter name -> value. Values may be scalars or arrays (field latents
        arrive as vectors).

    Raises
    ------
    ValueError
        If any value contains NaN or inf. The message names the offending parameters.
    """
    import numpy as _np

    bad = sorted(
        name
        for name, value in params.items()
        if not bool(_np.all(_np.isfinite(_np.asarray(value, dtype=float))))
    )
    if not bad:
        return
    raise ValueError(
        f"MAP optimization produced a non-finite estimate for {bad}, the fit did not "
        f"converge (a run ending in loss=nan is the usual cause). Handing this on as "
        f"an initialization gives a posterior of NaN with NaN R-hat, which raises "
        f"nothing and silently disarms any convergence check. Inspect data scaling "
        f"and prior bounds, or pass an explicit finite `init_from`."
    )


def _build_optax_optimizer(optimizer, learning_rate):
    """Build an optax optimizer from a string name or return as-is."""
    try:
        import optax
    except ImportError:
        raise ImportError("optax required for MAP: pip install optax") from None

    if isinstance(optimizer, str):
        opt_builders = {
            "adam": lambda: optax.adam(learning_rate),
            "adamw": lambda: optax.adamw(learning_rate),
            "sgd": lambda: optax.sgd(learning_rate, momentum=0.9),
        }
        if optimizer not in opt_builders:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Use {sorted(_ALL_OPTIMIZERS)} or pass an optax optimizer."
            )
        name = optimizer.upper()
        return opt_builders[optimizer](), name

    return optimizer, "custom"


def _build_jaxopt_solver(optimizer, loss_fn, *, maxiter, tol):
    """Build a jaxopt solver from a string name.

    Used by ``_fit_batch_vmap_map`` for batched ``jax.vmap(solver.run)``
    optimization.  For single-galaxy optimization, use ``_run_map_scipy``
    which avoids all JAX compilation for the optimizer.

    Parameters
    ----------
    optimizer : str
        One of ``_QUASI_NEWTON``.
    loss_fn : callable
        ``(params, data_args) -> scalar`` loss function.
    maxiter : int
        Maximum number of solver iterations.
    tol : float
        Gradient norm tolerance for convergence.

    Returns
    -------
    solver : jaxopt solver
        Configured solver instance.
    name : str
        Display name for diagnostics.
    """
    try:
        import jaxopt
    except ImportError:
        raise ImportError("jaxopt required for quasi-Newton MAP: pip install jaxopt") from None

    builders = {
        "lbfgs": lambda: jaxopt.LBFGS(
            fun=loss_fn,
            maxiter=maxiter,
            tol=tol,
            jit=True,
        ),
    }

    display_names = {
        "lbfgs": "L-BFGS",
    }

    return builders[optimizer](), display_names[optimizer]


_SCAN_BATCH = 50


def _get_or_build_map_fns(model, loss_fn, optimizer, learning_rate):
    """Return cached (scan_batch, single_step, opt, opt_name).

    ``scan_batch`` runs ``_SCAN_BATCH`` steps inside ``jax.lax.scan``
    (one XLA dispatch, zero per-step Python allocation).
    ``single_step`` is for the final remainder steps.
    """
    import optax

    cache_key = optimizer if isinstance(optimizer, str) else "custom"
    step_key = (id(loss_fn), cache_key, learning_rate)

    cache = _model_cache_owner.get_or_compile_model(model).setdefault("map_step", {})

    if step_key in cache:
        return cache[step_key]

    opt, opt_name = _build_optax_optimizer(optimizer, learning_rate)

    @jax.jit
    def scan_batch(params, ostate, d_args):
        """Execute _SCAN_BATCH optimizer steps in a single jax.lax.scan."""

        def _step_body(carry, _):
            """Execute one optimizer step: compute gradient, update state and params."""
            params, ostate = carry
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, d_args))(params)
            updates, new_ostate = opt.update(grads, ostate, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_ostate), loss

        (params, ostate), losses = jax.lax.scan(
            _step_body,
            (params, ostate),
            None,
            length=_SCAN_BATCH,
        )
        return params, ostate, losses

    @jax.jit
    def single_step(params, ostate, d_args):
        """Perform one JIT-compiled gradient update, returning updated params and loss."""
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, d_args))(params)
        updates, new_ostate = opt.update(grads, ostate, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_ostate, loss

    cache[step_key] = (scan_batch, single_step, opt, opt_name)
    return scan_batch, single_step, opt, opt_name


# ── scipy quasi-Newton path (single-galaxy, no JAX compilation for optimizer)


def _run_map_scipy(
    context: InferenceContext,
    *,
    init_params,
    grad_fn,
    loss_fn,
    data_args,
    optimizer,
    n_steps,
    tol,
    verbose,
    verbose_steps=False,
    print_every=50,
):
    """MAP optimization via scipy quasi-Newton solvers.

    Uses scipy's L-BFGS-B / BFGS, pure Fortran optimizer with zero
    JAX compilation.  Only the forward model + gradient evaluation is
    JIT-compiled (via the cached ``grad_fn``).

    This is ~100x faster cold-start than jaxopt's ``solver.run(jit=True)``,
    which compiles the entire while_loop + line-search into one monolithic
    XLA program.
    """
    from scipy.optimize import minimize as scipy_minimize

    from tengri.inference.posterior import Posterior

    scipy_method = {"lbfgs": "L-BFGS-B", "lbfgs_scipy": "L-BFGS-B"}[optimizer]
    opt_name = {"lbfgs": "L-BFGS", "lbfgs_scipy": "L-BFGS"}[optimizer]

    # Flatten params to 1D array for scipy
    init_flat, unravel_fn = ravel_pytree(init_params)

    # Zero free parameters (every parameter Fixed): scipy's L-BFGS-B refuses to
    # call the objective on an empty x0 -- confirmed by direct test, it returns
    # a placeholder ``OptimizeResult(fun=0.0, nit=0)`` no matter what the
    # objective would report, silently discarding the real loss (#lbfgs-default).
    # Evaluate the (single, fixed) point directly instead of asking scipy to
    # "optimize" nothing. jax.scipy.optimize.minimize does not share this bug
    # (verified: it returns the true value), so only this scipy path needs it.
    if init_flat.size == 0:
        t0 = time.time()
        final_loss_val, grad_val = grad_fn(init_params, data_args)
        jax.block_until_ready((final_loss_val, grad_val))
        wall_time = time.time() - t0
        _reject_nonfinite_map(init_params)
        best_params_physical = context.to_physical(init_params)
        final_loss = float(final_loss_val)
        if verbose:
            print(
                f"  MAP ({opt_name}) complete in {wall_time:.1f}s, "
                f"0 free parameters (nothing to optimize), loss={final_loss:.6f}"
            )
        return Posterior(
            samples=None,
            params=best_params_physical,
            method=f"MAP ({opt_name})",
            wall_time_s=wall_time,
            diagnostics={
                "n_steps": 0,
                "n_evals": 1,
                "final_loss": final_loss,
                "optimizer": opt_name,
                "converged": True,
                "grad_norm": 0.0,
            },
            loss_history=jnp.asarray([final_loss]),
            _model=context.model,
        )

    # Warmup: ensure grad_fn is compiled before timing
    _warmup = grad_fn(init_params, data_args)
    jax.block_until_ready(_warmup)

    n_evals = [0]
    # Per-iteration objective value, so ``Posterior.loss_history`` is a real
    # curve rather than the single final number scipy's OptimizeResult
    # reports. ``losses[0]`` is seeded with the value at the initial point
    # (scipy's ``callback`` only fires *after* each completed iteration, so
    # without this the curve would start one step late); every entry after
    # that is one `callback(xk)` call, i.e. one iterate. D is small here
    # (single-galaxy MAP), so the extra loss_fn eval per iteration is cheap.
    losses = [float(loss_fn(init_params, data_args))]

    def objective(flat_params_np):
        """Objective and gradient for scipy optimizer: (loss, grad_flat)."""
        params = unravel_fn(jnp.asarray(flat_params_np))
        val, grad = grad_fn(params, data_args)
        flat_grad, _ = ravel_pytree(grad)
        n_evals[0] += 1
        return float(val), np.asarray(flat_grad, dtype=np.float64)

    def callback(xk):
        """Record (and, if requested, print) the loss at each iterate."""
        params = unravel_fn(jnp.asarray(xk))
        loss_val = float(loss_fn(params, data_args))
        losses.append(loss_val)
        if verbose_steps:
            step = len(losses) - 1
            if step % print_every == 0:
                print(f"    step {step}: loss={loss_val:.6f}")

    t0 = time.time()
    result = scipy_minimize(
        objective,
        np.asarray(init_flat, dtype=np.float64),
        method=scipy_method,
        jac=True,
        callback=callback,
        options={"maxiter": n_steps, "gtol": tol},
    )
    wall_time = time.time() - t0

    best_params = unravel_fn(jnp.asarray(result.x))
    _reject_nonfinite_map(best_params)
    best_params_physical = context.to_physical(best_params)
    final_loss = float(result.fun)
    converged = result.success

    if verbose:
        cv_msg = " (converged)" if converged else ""
        print(
            f"  MAP ({opt_name}) complete in {wall_time:.1f}s, "
            f"{result.nit} iters, {n_evals[0]} evals{cv_msg}, "
            f"loss={final_loss:.6f}"
        )

    grad_norm = float(jnp.linalg.norm(jnp.asarray(result.jac)))

    # ``losses`` always has at least the seeded initial-point value (above),
    # even when scipy converges in zero iterations and ``callback`` never
    # fires, so no ``if losses else`` fallback is needed here.
    loss_hist = jnp.asarray(losses)

    return Posterior(
        samples=None,
        params=best_params_physical,
        method=f"MAP ({opt_name})",
        wall_time_s=wall_time,
        diagnostics={
            "n_steps": result.nit,
            "n_evals": n_evals[0],
            "final_loss": final_loss,
            "optimizer": opt_name,
            "converged": converged,
            "grad_norm": grad_norm,
        },
        loss_history=loss_hist,
        _model=context.model,
    )


def _best_finite_restart(final_losses) -> int:
    """Index of the lowest **finite** loss among multi-start restarts (#1397).

    Parameters
    ----------
    final_losses : array_like, shape (n_restarts,)
        Each restart's loss at its last step [dimensionless].

    Returns
    -------
    int
        Index of the best restart that produced a usable loss.

    Raises
    ------
    ValueError
        If no restart finished finite, naming the count and what to change.

    Notes
    -----
    ``jnp.argmin`` cannot be used here: every comparison against NaN is false,
    so it returns index 0 whenever the vector contains a NaN. Multi-start exists
    precisely because some inits diverge, so "keep the best" was keeping the
    worst -- on ``recipes.mock_recovery_minimal()`` it discarded a converged
    restart at loss 4.635 in favor of a NaN one. ``-inf`` is rejected for the
    same reason in reverse: a plain ``argmin`` would rank it the best fit
    possible.

    Not JIT-safe by design, reads concrete values to decide, and runs once per
    fit after ``block_until_ready``.
    """
    losses = np.asarray(final_losses, dtype=float)
    finite = np.isfinite(losses)
    if not finite.any():
        raise ValueError(
            f"all {losses.size} MAP restarts diverged to a non-finite loss "
            "(NaN or inf), so there is no starting point to return. The "
            "optimizer stepped outside the model's valid range from every "
            "initialization: try a smaller learning_rate=, fewer n_steps=, or "
            "a larger n_restarts= so at least one init survives."
        )
    return int(np.argmin(np.where(finite, losses, np.inf)))


def _run_map_multistart(context, *, key, n_restarts, n_steps, learning_rate, optimizer, verbose):
    """Run ``n_restarts`` independent ADAM optimizations in parallel; keep the best.

    Each restart starts from its own prior-sampled init (``key`` split
    ``n_restarts`` ways) and runs ``n_steps`` optax updates inside a single
    ``jax.lax.scan``; ``jax.vmap`` batches the restarts. The restart with the
    lowest final loss wins. Returns a MAP :class:`Posterior` identical in shape
    to the single-start path (its ``loss_history`` is the winning restart's).
    """
    import optax

    from tengri.inference.posterior import Posterior

    loss_fn = context.neg_log_posterior_fn
    data_args = context.data_args
    keys = jax.random.split(key, n_restarts)
    inits = jax.vmap(lambda k: context.initial_params(k, init_from=None))(keys)
    opt, opt_name = _build_optax_optimizer(optimizer, learning_rate)

    # Both the restart inits (parameters) and ``data_args`` (the data) are
    # threaded as runtime arguments (never closure-captured) so the compiled
    # kernel is reused across galaxies/datasets instead of baking the data in as
    # a constant (which would recompile per dataset). ``in_axes=(0, None)`` maps
    # over the restart axis while broadcasting the shared data.
    def _optimize_one(p0, d_args):
        ostate = opt.init(p0)

        def _step(carry, _):
            params, ostate = carry
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, d_args))(params)
            updates, ostate = opt.update(grads, ostate, params)
            return (optax.apply_updates(params, updates), ostate), loss

        (params, _), losses = jax.lax.scan(_step, (p0, ostate), None, length=n_steps)
        return params, losses

    def _build_restarts():
        return jax.jit(
            lambda batched_inits, d_args: jax.vmap(_optimize_one, in_axes=(0, None))(
                batched_inits, d_args
            )
        )

    # Memoize the vmapped restart kernel so repeated same-config
    # ``run("map", n_restarts>1)`` calls (e.g. a sequential per-galaxy catalog
    # loop) reuse the compiled executable instead of recompiling from a fresh
    # closure that misses jax.jit's cache. ``loss_fn`` is structurally cached and
    # the inits + data thread as arguments (see above), so the key need only
    # carry the optimizer config; a non-string optimizer we cannot fingerprint
    # disables the memo (build fresh, never risk a stale kernel).
    _fitter = getattr(context, "fitter", None)
    if (
        isinstance(optimizer, str)
        and _fitter is not None
        and hasattr(_fitter, "_memo_batch_kernel")
    ):
        _run_restarts = _fitter._memo_batch_kernel(
            "_map_multistart_kernel_cache",
            (
                "map_multistart",
                _fitter.compile_signature(),
                optimizer,
                int(n_steps),
                float(learning_rate),
                int(n_restarts),
            ),
            _build_restarts,
        )
    else:
        _run_restarts = _build_restarts()
    t0 = time.time()
    params_b, losses_b = _run_restarts(inits, data_args)
    jax.block_until_ready(losses_b)
    final_losses = losses_b[:, -1]
    best = _best_finite_restart(final_losses)
    best_params = jax.tree.map(lambda x: x[best], params_b)
    _reject_nonfinite_map(best_params)
    best_losses = losses_b[best]
    wall_time = time.time() - t0
    final_loss = float(best_losses[-1])

    if verbose:
        n_diverged = int(np.sum(~np.isfinite(np.asarray(final_losses, dtype=float))))
        finite_losses = np.asarray(final_losses, dtype=float)
        worst = float(np.max(finite_losses[np.isfinite(finite_losses)]))
        diverged_note = f"; {n_diverged} diverged" if n_diverged else ""
        print(
            f"  MAP ({opt_name} x{n_restarts} restarts) complete in {wall_time:.1f}s, "
            f"best loss={final_loss:.4f} "
            f"(restart {best}; worst finite={worst:.1f}{diverged_note})"
        )

    return Posterior(
        samples=None,
        params=context.to_physical(best_params),
        method=f"MAP ({opt_name}, {n_restarts} restarts)",
        wall_time_s=wall_time,
        diagnostics={
            "n_steps": int(n_steps),
            "final_loss": final_loss,
            "optimizer": opt_name,
            "n_restarts": int(n_restarts),
        },
        loss_history=best_losses,
        _model=context.model,
    )


def _run_map_multistart_scipy(
    context, *, key, n_restarts, n_steps, tol, optimizer, verbose, loss_fn, data_args
):
    """Run ``n_restarts`` scipy L-BFGS-B optimizations one after another; keep the best.

    The multistart that :func:`_maybe_map_init` seeds every sampler with.
    Measured 2026-09-12 on notebooks/07's tsnorm photometry model: the vmapped
    JAX-BFGS multistart (:func:`_run_map_multistart_qn`) returned
    neg-log-posterior **182.5** with ``converged=False`` from eight prior
    draws, where one scipy L-BFGS-B start reaches **10.2** and the old Adam
    multistart 27.6 -- a seed bad enough to send the notebook's HMC from
    R-hat 1.000 to 1.37 and its joint fit past a 3000 s timeout. scipy's
    L-BFGS-B is not JAX-traceable, so the restarts cannot be vmapped; at
    ~0.4 s per start on a D <= 12 photometry fit a sequential loop is cheap,
    and unlike JAX BFGS every start actually converges. Each restart draws
    its own init from a split of ``key``; the lowest finite final loss wins
    (:func:`_best_finite_restart` semantics: a start that diverged to a
    non-finite loss never wins).
    """
    from tengri.inference.posterior import Posterior

    keys = jax.random.split(key, n_restarts)
    results = []
    t0 = time.time()
    for k in keys:
        init_params = context.initial_params(k, init_from=None)
        results.append(
            _run_map_scipy(
                context,
                init_params=init_params,
                grad_fn=context.grad_fn,
                loss_fn=loss_fn,
                data_args=data_args,
                optimizer=optimizer,
                n_steps=n_steps,
                tol=tol,
                verbose=False,
            )
        )
    final_losses = jnp.asarray([float(r.diagnostics["final_loss"]) for r in results])
    best = int(_best_finite_restart(final_losses))
    winner = results[best]
    wall_time = time.time() - t0
    if verbose:
        finite = np.asarray(final_losses, dtype=float)
        n_diverged = int(np.sum(~np.isfinite(finite)))
        worst = float(np.max(finite[np.isfinite(finite)]))
        diverged_note = f"; {n_diverged} diverged" if n_diverged else ""
        print(
            f"  MAP (L-BFGS x{n_restarts} restarts, scipy) complete in {wall_time:.1f}s, "
            f"best loss={float(final_losses[best]):.4f} (restart {best}; "
            f"worst finite={worst:.1f}{diverged_note})"
        )
    diagnostics = dict(winner.diagnostics)
    diagnostics.update(
        {"n_restarts": int(n_restarts), "backend": "scipy_lbfgsb_sequential", "restart": best}
    )
    return Posterior(
        samples=None,
        params=winner.params,
        method=f"MAP (L-BFGS, {n_restarts} restarts)",
        wall_time_s=wall_time,
        diagnostics=diagnostics,
        loss_history=winner.loss_history,
        _model=context.model,
    )


def _run_map_multistart_qn(context, *, key, n_restarts, n_steps, tol, optimizer, verbose):
    """Run ``n_restarts`` independent JAX BFGS optimizations via vmap; keep the best.

    The quasi-Newton sibling of :func:`_run_map_multistart`. scipy's L-BFGS-B
    (:func:`_run_map_scipy`, the ``n_restarts=1`` path) is not JAX-traceable, so
    it cannot be vmapped over restarts; this function instead uses
    :func:`jax.scipy.optimize.minimize` (``method="BFGS"``), which is pure JAX
    and therefore vmappable, at the cost of BFGS's :math:`O(D^2)` dense inverse
    Hessian versus L-BFGS-B's limited-memory one -- acceptable here since a
    single restart still runs on the flat parameter vector, whose dimension is
    the model's free-parameter count, not the data size. Requires no optional
    dependency (``jax.scipy`` ships with ``jax`` itself), unlike a jaxopt- or
    optax-based alternative.

    Each restart starts from its own prior-sampled init (``key`` split
    ``n_restarts`` ways); the restart with the lowest **finite** final loss
    wins, via :func:`_best_finite_restart` -- a restart that reports
    ``success=False`` but a finite ``fun`` still competes on ``fun``, since
    BFGS's own convergence flag is conservative (e.g. line-search stalling
    near the optimum) and not a reliable finite/non-finite signal on its own.
    Returns a MAP :class:`~tengri.inference.posterior.Posterior` identical in
    shape to :func:`_run_map_multistart`, except ``loss_history`` is just
    ``[initial, final]`` for the winning restart: JAX BFGS runs inside
    ``jax.vmap(jax.jit(...))`` with no Python-level callback, so no
    per-iteration trace is available (unlike :func:`_run_map_scipy`, which
    records one value per scipy iteration via ``callback=``).
    """
    from tengri.inference.posterior import Posterior

    loss_fn = context.neg_log_posterior_fn
    data_args = context.data_args
    keys = jax.random.split(key, n_restarts)
    inits = jax.vmap(lambda k: context.initial_params(k, init_from=None))(keys)

    def _optimize_one(p0, d_args):
        flat0, unravel_fn = ravel_pytree(p0)

        def fun(flat):
            return loss_fn(unravel_fn(flat), d_args)

        result = _jax_minimize(fun, flat0, method="BFGS", tol=tol, options={"maxiter": n_steps})
        return unravel_fn(result.x), result.fun, result.success, result.nit

    def _build_restarts():
        return jax.jit(
            lambda batched_inits, d_args: jax.vmap(_optimize_one, in_axes=(0, None))(
                batched_inits, d_args
            )
        )

    # Memoize like ``_run_map_multistart`` (see its comment): a non-string
    # optimizer cannot reach this function (the ``run_map`` gate below only
    # routes strings in ``_SCIPY_OPTIMIZERS`` here), so the memo is always safe
    # to use when a fitter is available.
    # No memo: this path is opt-in (``optimizer="bfgs_jax"``) since the
    # sequential scipy multistart replaced it as the L-BFGS default, and a memo
    # keyed on the compile signature must not itself be a Fitter attribute.
    _run_restarts = _build_restarts()

    t0 = time.time()
    params_b, fun_b, success_b, nit_b = _run_restarts(inits, data_args)
    jax.block_until_ready(fun_b)
    final_losses = fun_b
    best = _best_finite_restart(final_losses)
    best_params = jax.tree.map(lambda x: x[best], params_b)
    _reject_nonfinite_map(best_params)
    wall_time = time.time() - t0
    final_loss = float(final_losses[best])
    # No per-iteration trace: ``jax.scipy.optimize.minimize`` runs inside
    # ``jax.vmap(jax.jit(...))`` with no Python-level callback (see the
    # docstring). ``[initial, final]`` at least distinguishes "did the winning
    # restart improve" from ``_run_map_scipy``'s single-value fallback, at the
    # cost of one extra (cheap, D-sized) loss_fn eval on the winning restart's
    # own starting point.
    initial_loss = float(loss_fn(jax.tree.map(lambda x: x[best], inits), data_args))
    opt_name = "L-BFGS"

    if verbose:
        n_diverged = int(np.sum(~np.isfinite(np.asarray(final_losses, dtype=float))))
        finite_losses = np.asarray(final_losses, dtype=float)
        worst = float(np.max(finite_losses[np.isfinite(finite_losses)]))
        diverged_note = f"; {n_diverged} diverged" if n_diverged else ""
        print(
            f"  MAP ({opt_name} x{n_restarts} restarts, JAX BFGS) complete in "
            f"{wall_time:.1f}s, best loss={final_loss:.4f} "
            f"(restart {best}; worst finite={worst:.1f}{diverged_note})"
        )

    return Posterior(
        samples=None,
        params=context.to_physical(best_params),
        method=f"MAP ({opt_name}, {n_restarts} restarts)",
        wall_time_s=wall_time,
        diagnostics={
            "n_steps": int(nit_b[best]),
            "final_loss": final_loss,
            "optimizer": opt_name,
            "backend": "jax_bfgs_vmap",
            "n_restarts": int(n_restarts),
            "converged": bool(success_b[best]),
        },
        loss_history=jnp.asarray([initial_loss, final_loss]),
        _model=context.model,
    )


def run_map(
    context,
    *,
    key,
    init_from=None,
    n_steps=500,
    learning_rate=0.02,
    optimizer="lbfgs",
    n_restarts=1,
    early_stopping=True,
    patience=100,
    rtol=1e-5,
    tol=1e-5,
    verbose=True,
    verbose_steps=False,
    print_every=200,
):
    """MAP optimization via quasi-Newton solvers or gradient descent.

    Parameters
    ----------
    n_steps : int
        Maximum number of optimization steps. Quasi-Newton (``optimizer``
        one of ``_SCIPY_OPTIMIZERS``): passed through as ``maxiter`` --
        scipy's ``options={"maxiter": n_steps}`` on the single-start
        (``n_restarts=1``) path, ``jax.scipy.optimize.minimize``'s
        ``options={"maxiter": n_steps}`` on the vmapped multi-start
        (``n_restarts>1``) path. Optax (``"adam"``/``"sgd"``/``"adamw"``/a
        custom optax optimizer): the number of gradient steps actually taken,
        subject to early stopping.
    learning_rate : float
        Optax step size. Ignored for quasi-Newton solvers, which have no
        learning rate (BFGS/L-BFGS-B choose their own step via line search).
    optimizer : str or optax optimizer
        ``"lbfgs"`` (default, alias ``"lbfgs_scipy"``): quasi-Newton MAP.
        **Two different implementations share this name, chosen by
        ``n_restarts``, because scipy is not JAX-traceable and so cannot be
        vmapped**: ``n_restarts=1`` (or an explicit ``init_from``) runs
        scipy's L-BFGS-B (:func:`_run_map_scipy`) with a Wolfe line search and
        zero JAX compilation for the optimizer itself; ``n_restarts>1`` runs
        ``jax.scipy.optimize.minimize(method="BFGS")`` for every restart
        inside one ``jax.vmap`` (:func:`_run_map_multistart_qn`), so the
        restarts execute as a single compiled kernel instead of a Python
        loop. Both converge to the same optimum on a smooth posterior;
        neither needs an optional dependency (``jax.scipy`` ships with
        ``jax``, scipy ships transitively via ``jax``'s own dependencies).
        Optax: ``"adam"``, ``"sgd"``, ``"adamw"``, or a pre-built optax
        optimizer -- first-order, vmappable at any ``n_restarts``
        (:func:`_run_map_multistart`), but converges less reliably to a true
        optimum than either L-BFGS path (see module Notes).
    n_restarts : int
        Number of independent prior-sampled starts to run and keep the
        best-loss result. ``1`` (default) for quasi-Newton uses the faster
        single-start scipy path; ``>1`` switches quasi-Newton to the vmapped
        JAX BFGS path (see ``optimizer`` above). Ignored (treated as ``1``)
        when ``init_from`` is given: an explicit start is always honored
        as-is, never multiplied into restarts.
    early_stopping : bool
        Stop if loss doesn't improve (optax only). Quasi-Newton solvers use
        their own convergence test (``tol`` below) and always run to
        convergence or ``n_steps``, whichever comes first.
    patience : int
        Steps to wait for improvement before stopping (optax only; ignored
        for quasi-Newton).
    rtol : float
        Relative tolerance for early stopping (optax only; ignored for
        quasi-Newton).
    tol : float
        Gradient norm tolerance for convergence (quasi-Newton only): scipy's
        ``gtol`` on the single-start path, ``jax.scipy.optimize.minimize``'s
        ``tol`` on the vmapped multi-start path. Ignored for optax, which has
        no analogous stopping criterion (see ``rtol``/``patience`` instead).
    verbose : bool
        Print progress summary.
    verbose_steps : bool
        Print per-step loss (single-start quasi-Newton only; no effect on
        optax or the vmapped multi-start quasi-Newton path, neither of which
        supports a Python-level per-step callback).
    print_every : int
        Print interval (optax and single-start quasi-Newton verbose_steps).

    Notes
    -----
    Why ``"lbfgs"`` is the default: on the ctl-dpl mock recovery fixture
    (D=8, 14 bands), Adam with the population default (``n_restarts=8``,
    ``n_steps=800``) reached a negative log posterior of 6.33 and had not
    converged (a 300-step single run reached 7.88, a 100-step run 115);
    single-start scipy L-BFGS-B reached 6.0008 from one start in under a
    second. On a second fixture the Adam point had a *negative* Hessian
    eigenvalue -- not a minimum. Every downstream consumer of a MAP point
    (NUTS/HMC warm-start via ``_maybe_map_init``, the Laplace approximation,
    preconditioning metrics) is better served by a converged optimum than by
    a fixed gradient-step budget that may or may not have reached one.
    """
    from tengri.inference.posterior import Posterior

    # Normalize: dispatcher passes an InferenceContext for migrated
    # backends, but internal callsites (``Fitter._run_map``) may still
    # pass a raw Fitter during the migration window, accept both.
    context = InferenceContext.from_target(context)
    loss_fn = context.neg_log_posterior_fn
    data_args = context.data_args

    # Multi-start (n_restarts > 1, no explicit init_from): route to whichever
    # vmapped restart implementation matches the optimizer family. An explicit
    # ``init_from`` is always honored as a single start, never multiplied.
    if n_restarts > 1 and init_from is None and isinstance(optimizer, str):
        if optimizer in _SCIPY_OPTIMIZERS:
            # ── multi-start scipy L-BFGS-B, one restart after another ──
            # The vmapped JAX-BFGS sibling (_run_map_multistart_qn, kept for
            # optimizer="bfgs_jax") was measured to stall from prior draws on
            # a tsnorm model (nlp 182 against 10 for one scipy start); see
            # _run_map_multistart_scipy.
            return _publish_map_init_cache(
                context,
                _run_map_multistart_scipy(
                    context,
                    key=key,
                    n_restarts=n_restarts,
                    n_steps=n_steps,
                    tol=tol,
                    optimizer=optimizer,
                    verbose=verbose,
                    loss_fn=loss_fn,
                    data_args=data_args,
                ),
            )
        if optimizer == "bfgs_jax":
            return _publish_map_init_cache(
                context,
                _run_map_multistart_qn(
                    context,
                    key=key,
                    n_restarts=n_restarts,
                    n_steps=n_steps,
                    tol=tol,
                    optimizer="lbfgs",
                    verbose=verbose,
                ),
            )
        # ── multi-start ADAM (vmap'd restarts, keep the lowest-loss) ──
        # Under the standardized prior an N(0,1) latent maps to a genuinely uniform
        # physical prior, so a single random init can land in a poor basin and a
        # single ADAM run stalls there. Running ``n_restarts`` independent inits
        # (seeded from splits of ``key``) in parallel via ``jax.vmap`` and keeping
        # the best-loss restart recovers robustness while staying fully JAX-native
        # (jittable, vmappable, scales to high-D). Only for the optax path and only
        # when no explicit ``init_from`` is given (an explicit start is honored).
        return _publish_map_init_cache(
            context,
            _run_map_multistart(
                context,
                key=key,
                n_restarts=n_restarts,
                n_steps=n_steps,
                learning_rate=learning_rate,
                optimizer=optimizer,
                verbose=verbose,
            ),
        )

    init_params = context.initial_params(key, init_from=init_from)

    # ── scipy quasi-Newton path (optimizer="lbfgs_scipy") ──
    if isinstance(optimizer, str) and optimizer in _SCIPY_OPTIMIZERS:
        return _publish_map_init_cache(
            context,
            _run_map_scipy(
                context,
                init_params=init_params,
                grad_fn=context.grad_fn,
                loss_fn=loss_fn,
                data_args=data_args,
                optimizer=optimizer,
                n_steps=n_steps,
                tol=tol,
                verbose=verbose,
                verbose_steps=verbose_steps,
                print_every=print_every,
            ),
        )

    # ── optax iterative path (adam / adamw / sgd / custom) ──
    scan_batch, single_step, opt, opt_name = _get_or_build_map_fns(
        context.model,
        loss_fn,
        optimizer,
        learning_rate,
    )
    opt_state = opt.init(init_params)

    # Warmup: trigger JIT compilation before timing.
    _, _, warmup_loss = single_step(init_params, opt_state, data_args)
    warmup_loss.block_until_ready()

    params = init_params
    best_loss = float("inf")
    best_params = params
    stale = 0
    losses = []
    t0 = time.time()

    n_done = 0
    n_full_batches = n_steps // _SCAN_BATCH
    n_remainder = n_steps % _SCAN_BATCH

    for _batch_idx in range(n_full_batches):
        params, opt_state, batch_losses = scan_batch(params, opt_state, data_args)
        batch_losses_list = batch_losses.tolist()
        losses.extend(batch_losses_list)
        n_done += _SCAN_BATCH
        last_loss = batch_losses_list[-1]

        if early_stopping:
            if last_loss < best_loss * (1 - rtol):
                best_loss = last_loss
                best_params = params
                stale = 0
            else:
                stale += _SCAN_BATCH
                if stale >= patience:
                    break

        if verbose and n_done % print_every == 0:
            print(f"    step {n_done}: loss={last_loss:.4f}")

    else:
        for _i in range(n_remainder):
            params, opt_state, loss = single_step(params, opt_state, data_args)
            loss_val = float(loss)
            losses.append(loss_val)
            n_done += 1

            if early_stopping:
                if loss_val < best_loss * (1 - rtol):
                    best_loss = loss_val
                    best_params = params
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            else:
                best_params = params
                best_loss = loss_val

    if not early_stopping:
        best_params = params
        best_loss = losses[-1] if losses else float("nan")

    wall_time = time.time() - t0
    n_actual = len(losses)
    final_loss = losses[-1] if losses else float("nan")
    _reject_nonfinite_map(best_params)
    best_params_physical = context.to_physical(best_params)

    if verbose:
        es_msg = ""
        if early_stopping and n_actual < n_steps:
            es_msg = f" (early stop at {n_actual})"
        print(
            f"  MAP ({opt_name}) complete in {wall_time:.1f}s, "
            f"{n_actual} steps{es_msg}, loss={final_loss:.4f}"
        )

    return _publish_map_init_cache(
        context,
        Posterior(
            samples=None,
            params=best_params_physical,
            method=f"MAP ({opt_name})",
            wall_time_s=wall_time,
            diagnostics={
                "n_steps": n_actual,
                "final_loss": final_loss,
                "optimizer": opt_name,
            },
            loss_history=jnp.asarray(losses),
            _model=context.model,
        ),
    )


def build_vectorized_map_solver(
    fitter,
    *,
    n_steps: int = 80,
    learning_rate: float = 0.05,
    optimizer: str = "adam",
):
    """Return a JIT-friendly per-galaxy MAP solver for population fitting.

    The returned ``map_solve_one(flux, noise, key) -> dict`` runs ``n_steps``
    optax updates via ``jax.lax.scan`` (zero Python overhead per step) and
    returns the final unbounded-parameter dict. Wrap in
    ``jax.lax.map(map_solve_one, (all_flux, all_noise, all_keys), batch_size=K)``
    for vectorized population MAP-init: compile cost stays O(K) regardless
    of catalog size, replacing the O(N) Python dispatch loop.

    Parameters
    ----------
    fitter : Fitter
        Template fitter, its model, observation, spec, and ``_data_args``
        layout are reused for every galaxy. The galaxy-varying fields
        (``data``, ``noise``, ``sqrt_noise_inv``) are replaced inside the
        scan; all other fields (filter curves, masks, spec covariance) are
        shared from the template.
    n_steps : int, optional
        Number of optax steps (default 200).  No early stopping, the
        scan length is static, so compile cost is independent of n_steps.
    learning_rate : float, optional
        Adam learning rate (default 0.03).
    optimizer : str, optional
        ``"adam"`` (default), ``"adamw"``, or ``"sgd"`` -- optax only, unlike
        :func:`run_map`. This solver builds its optimizer via
        :func:`_build_optax_optimizer`, which has no quasi-Newton entry, so
        ``"lbfgs"`` raises ``ValueError`` here rather than silently falling
        back to a different family. Left as ``"adam"`` deliberately: this is
        a short (``n_steps`` as low as 80), scan-based per-galaxy *warm start*
        for hierarchical/population fitting's own posterior, not a converged
        MAP fit in its own right, so it is scoped like
        :func:`~tengri.inference.backends.mcmc.catalog.build_catalog_map_init`
        (also an ADAM-only warm start) rather than like :func:`run_map`.

    Returns
    -------
    map_solve_one : callable
        ``map_solve_one(flux, noise, key) -> dict`` of unbounded params.

    Notes
    -----
    **JIT-compatible**: yes.  The body is one ``lax.scan`` over ``n_steps``
    optax updates.  All galaxy-varying state (flux, noise) is passed as
    arguments, not closures, so a single compiled artifact is reused
    across galaxies.
    """
    import optax

    loss_fn = fitter._get_or_build_loss_fn()
    template_data_args = fitter._data_args
    spec = fitter.spec
    free_names = list(fitter._free_names)

    opt, _opt_name = _build_optax_optimizer(optimizer, learning_rate)

    def _make_data_args(flux, noise):
        """Per-galaxy data_args with shared filter/obs fields preserved."""
        out = dict(template_data_args)
        out["data"] = flux
        out["noise"] = noise
        out["sqrt_noise_inv"] = inv_noise_std(noise)
        return out

    def _initial_unbounded(key):
        """Pure-JAX init mirroring Fitter._initialize_unbounded (avoids closing
        over self for cleaner tracing)."""
        from tengri.parameters.priors import Gaussian

        keys = jax.random.split(key, len(free_names) + 1)
        params: dict = {}
        for i, name in enumerate(free_names):
            dist = spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                params[name] = dist.standardize(jnp.array(dist.mu))
            else:
                params[name] = 0.1 * jax.random.normal(keys[i])
        if spec.stochastic:
            params["psd_xi"] = 0.1 * jax.random.normal(keys[-1], shape=(spec.n_grid,))
        return params

    def map_solve_one(flux, noise, key):
        data_args = _make_data_args(flux, noise)
        init_params = _initial_unbounded(key)
        opt_state = opt.init(init_params)

        def step(carry, _):
            params, ostate = carry
            _loss, grads = jax.value_and_grad(lambda p: loss_fn(p, data_args))(params)
            updates, new_ostate = opt.update(grads, ostate, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_ostate), None

        (final_params, _), _ = jax.lax.scan(
            step,
            (init_params, opt_state),
            None,
            length=n_steps,
        )
        return final_params

    return map_solve_one


def run_laplace(context, *, key, init_from=None, n_map_steps=1000, **kwargs):
    """Laplace approximation: Gaussian posterior from Hessian at MAP."""
    from tengri.inference.backends.laplace import run_laplace

    context = InferenceContext.from_target(context)

    if init_from is not None:
        map_params_posterior = init_from
    else:
        map_result = run_map(
            context,
            key=key,
            n_steps=n_map_steps,
            verbose=kwargs.get("verbose", True),
        )
        map_params_posterior = map_result

    map_params = context.unbounded_from_posterior(map_params_posterior)

    return run_laplace(
        key=key,
        loss_fn=context.neg_log_posterior_fn,
        data_args=context.data_args,
        map_params_unbounded=map_params,
        to_physical_fn=context.to_physical,
        model=context.model,
        grad_fn=context.grad_fn,
        **kwargs,
    )


def run_pathfinder(context, *, key, init_from=None, precondition=None, **kwargs):
    """Pathfinder: fast approximate posterior via L-BFGS path.

    ``precondition`` whitens the latent coordinates with the analytic
    ``J^T N^-1 J + I`` metric before the L-BFGS path is traced, and maps the draws
    back afterwards. It is the one geometric lever this backend has: Pathfinder's
    covariance is a **low-rank** inverse Hessian read off the L-BFGS history, and
    :mod:`tengri.inference.preconditioning` measures the raw posterior at
    condition 8.5e4 to 3.1e8 on every configuration tested. Off by default
    (#1397), like every other backend.
    """
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
    from tengri.inference.backends.pathfinder import run_pathfinder
    from tengri.inference.preconditioning import prepare_preconditioning

    context = InferenceContext.from_target(context)
    init_params = context.initial_params(key, init_from=init_from)

    # ``_get_flat_logdensity`` still takes a Fitter, reach through
    # the context until ``mcmc/_shared.py`` migrates (PR4).
    log_posterior_flat_2arg, unravel_fn, init_flat, data_args = _get_flat_logdensity(
        context.fitter,
        init_params,
    )

    problem = prepare_preconditioning(
        log_posterior_flat_2arg, init_flat, data_args, precondition=precondition
    )
    if problem.enabled and kwargs.get("verbose", True):
        logger.info(
            "Pathfinder preconditioning: strength=%.2f, cond %.2e -> %.2e at the initial point",
            problem.strength,
            problem.metric_condition,
            problem.whitened_condition,
        )
    whitened_logdensity = problem.logdensity

    def log_posterior_flat(pos):
        """Evaluate the flat log posterior with data_args bound from the enclosing scope."""
        return whitened_logdensity(pos, data_args)

    return run_pathfinder(
        key=key,
        log_posterior_flat=log_posterior_flat,
        init_flat=problem.init_flat,
        unravel_fn=unravel_fn,
        to_physical_fn=context.to_physical,
        model=context.model,
        restore_fn=problem.restore,
        precondition_diagnostics={
            "precondition_enabled": problem.enabled,
            "precondition_strength": problem.strength,
            "metric_condition": problem.metric_condition,
            "whitened_condition": problem.whitened_condition,
        },
        **kwargs,
    )
