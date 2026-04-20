"""MAP optimization, Laplace approximation, and Pathfinder.

Extracted from fitter.py.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_OPTAX_OPTIMIZERS = {"adam", "adamw", "sgd", "lbfgs"}
_SCIPY_OPTIMIZERS = {"lbfgs_scipy"}
_ALL_OPTIMIZERS = _OPTAX_OPTIMIZERS | _SCIPY_OPTIMIZERS

# Backward compatibility alias (used by fitter._fit_batch_vmap_map and tests)
_JAXOPT_SOLVERS = _SCIPY_OPTIMIZERS
_QUASI_NEWTON = _SCIPY_OPTIMIZERS


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
            "lbfgs": lambda: optax.chain(
                optax.scale_by_lbfgs(memory_size=10, scale_init_precond=True),
                optax.clip_by_global_norm(0.5),
                optax.scale(-1.0),
            ),
        }
        if optimizer not in opt_builders:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Use {sorted(_ALL_OPTIMIZERS)} or pass an optax optimizer."
            )
        display_names = {"lbfgs": "L-BFGS"}
        name = display_names.get(optimizer, optimizer.upper())
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
        raise ImportError(
            "jaxopt required for quasi-Newton MAP: pip install jaxopt"
        ) from None

    builders = {
        "lbfgs": lambda: jaxopt.LBFGS(
            fun=loss_fn, maxiter=maxiter, tol=tol, jit=True,
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

    if not hasattr(model, "_map_step_cache"):
        model._map_step_cache = {}

    if step_key in model._map_step_cache:
        return model._map_step_cache[step_key]

    opt, opt_name = _build_optax_optimizer(optimizer, learning_rate)

    @jax.jit
    def scan_batch(params, ostate, d_args):
        def _step_body(carry, _):
            params, ostate = carry
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, d_args))(params)
            updates, new_ostate = opt.update(grads, ostate, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_ostate), loss

        (params, ostate), losses = jax.lax.scan(
            _step_body, (params, ostate), None, length=_SCAN_BATCH,
        )
        return params, ostate, losses

    @jax.jit
    def single_step(params, ostate, d_args):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, d_args))(params)
        updates, new_ostate = opt.update(grads, ostate, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_ostate, loss

    model._map_step_cache[step_key] = (scan_batch, single_step, opt, opt_name)
    return scan_batch, single_step, opt, opt_name


# ---------------------------------------------------------------------------
# scipy quasi-Newton path (single-galaxy, no JAX compilation for optimizer)
# ---------------------------------------------------------------------------


def _run_map_scipy(
    fitter,
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

    Uses scipy's L-BFGS-B / BFGS — pure Fortran optimizer with zero
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

    # Warmup: ensure grad_fn is compiled before timing
    _warmup = grad_fn(init_params, data_args)
    jax.block_until_ready(_warmup)

    n_evals = [0]
    losses = []

    def objective(flat_params_np):
        params = unravel_fn(jnp.asarray(flat_params_np))
        val, grad = grad_fn(params, data_args)
        flat_grad, _ = ravel_pytree(grad)
        n_evals[0] += 1
        return float(val), np.asarray(flat_grad, dtype=np.float64)

    callback = None
    if verbose_steps:
        def callback(xk):
            params = unravel_fn(jnp.asarray(xk))
            loss_val = float(loss_fn(params, data_args))
            losses.append(loss_val)
            step = len(losses)
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
    best_params_physical = fitter._to_physical(best_params)
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

    loss_hist = jnp.asarray(losses) if losses else jnp.asarray([final_loss])

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
        _model=fitter.model,
    )


def run_map(
    fitter,
    *,
    key,
    init_from=None,
    n_steps=500,
    learning_rate=0.02,
    optimizer="adam",
    early_stopping=True,
    patience=100,
    rtol=1e-5,
    tol=1e-5,
    verbose=True,
    verbose_steps=False,
    print_every=200,
):
    """MAP optimization via gradient descent or quasi-Newton solvers.

    Parameters
    ----------
    n_steps : int
        Maximum number of optimization steps.
    learning_rate : float
        Learning rate (optax optimizers only; ignored for quasi-Newton).
    optimizer : str or optax optimizer
        Optax: ``"adam"``, ``"sgd"``, ``"adamw"``, ``"lbfgs"``, or a
        pre-built optax optimizer.  ``"lbfgs"`` uses ``optax.scale_by_lbfgs``
        through the scan-batch path (fast, no line search).
        ``"lbfgs_scipy"`` uses scipy L-BFGS-B with Wolfe line search
        (slower but guaranteed convergence).
    early_stopping : bool
        Stop if loss doesn't improve (optax only; quasi-Newton uses ``tol``).
    patience : int
        Steps to wait for improvement before stopping (optax only).
    rtol : float
        Relative tolerance for early stopping (optax only).
    tol : float
        Gradient norm tolerance for convergence (quasi-Newton solvers).
    verbose : bool
        Print progress summary.
    verbose_steps : bool
        Print per-step loss (quasi-Newton only).
    print_every : int
        Print interval.
    """
    from tengri.inference.posterior import Posterior

    loss_fn = fitter._get_or_build_loss_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    # ── scipy quasi-Newton path (optimizer="lbfgs_scipy") ──
    if isinstance(optimizer, str) and optimizer in _SCIPY_OPTIMIZERS:
        grad_fn = fitter._get_or_build_grad_fn()
        return _run_map_scipy(
            fitter,
            init_params=init_params,
            grad_fn=grad_fn,
            loss_fn=loss_fn,
            data_args=data_args,
            optimizer=optimizer,
            n_steps=n_steps,
            tol=tol,
            verbose=verbose,
            verbose_steps=verbose_steps,
            print_every=print_every,
        )

    # ── optax iterative path (adam / adamw / sgd / custom) ──
    scan_batch, single_step, opt, opt_name = _get_or_build_map_fns(
        fitter.model, loss_fn, optimizer, learning_rate,
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
    best_params_physical = fitter._to_physical(best_params)

    if verbose:
        es_msg = ""
        if early_stopping and n_actual < n_steps:
            es_msg = f" (early stop at {n_actual})"
        print(
            f"  MAP ({opt_name}) complete in {wall_time:.1f}s, "
            f"{n_actual} steps{es_msg}, loss={final_loss:.4f}"
        )

    return Posterior(
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
        _model=fitter.model,
    )


def run_laplace(fitter, *, key, init_from=None, n_map_steps=1000, **kwargs):
    """Laplace approximation: Gaussian posterior from Hessian at MAP."""
    from tengri.inference.backends.laplace import run_laplace

    loss_fn = fitter._get_or_build_loss_fn()
    grad_fn = fitter._get_or_build_grad_fn()
    data_args = fitter._data_args

    if init_from is not None:
        map_params = fitter._unbounded_from_posterior(init_from)
    else:
        map_result = fitter._run_map(
            key=key,
            n_steps=n_map_steps,
            verbose=kwargs.get("verbose", True),
        )
        map_params = fitter._unbounded_from_posterior(map_result)

    return run_laplace(
        key=key,
        loss_fn=loss_fn,
        data_args=data_args,
        map_params_unbounded=map_params,
        to_physical_fn=fitter._to_physical,
        model=fitter.model,
        grad_fn=grad_fn,
        **kwargs,
    )


def run_pathfinder(fitter, *, key, init_from=None, **kwargs):
    """Pathfinder: fast approximate posterior via L-BFGS path."""
    from tengri.inference.backends.pathfinder import run_pathfinder

    logdensity_2arg = fitter._get_or_build_logdensity_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    return run_pathfinder(
        key=key,
        logdensity_fn=logdensity_2arg,
        data_args=data_args,
        init_params=init_params,
        to_physical_fn=fitter._to_physical,
        model=fitter.model,
        **kwargs,
    )
