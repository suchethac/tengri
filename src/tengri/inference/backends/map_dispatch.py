"""MAP optimization, Laplace approximation, and Pathfinder.

Extracted from fitter.py.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

_OPTAX_OPTIMIZERS = {"adam", "adamw", "sgd"}
_JAXOPT_SOLVERS = {"lbfgs", "bfgs", "nonlinear_cg", "gradient_descent"}
_ALL_OPTIMIZERS = _OPTAX_OPTIMIZERS | _JAXOPT_SOLVERS


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
        return opt_builders[optimizer](), optimizer.upper()

    return optimizer, "custom"


def _build_jaxopt_solver(optimizer, loss_fn, *, maxiter, tol):
    """Build a jaxopt solver from a string name.

    Parameters
    ----------
    optimizer : str
        One of ``_JAXOPT_SOLVERS``.
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
        "bfgs": lambda: jaxopt.BFGS(
            fun=loss_fn, maxiter=maxiter, tol=tol, jit=True,
        ),
        "nonlinear_cg": lambda: jaxopt.NonlinearCG(
            fun=loss_fn, maxiter=maxiter, tol=tol, jit=True,
        ),
        "gradient_descent": lambda: jaxopt.GradientDescent(
            fun=loss_fn, maxiter=maxiter, tol=tol, jit=True,
        ),
    }

    display_names = {
        "lbfgs": "L-BFGS",
        "bfgs": "BFGS",
        "nonlinear_cg": "Nonlinear-CG",
        "gradient_descent": "GD-linesearch",
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


def _run_map_jaxopt(
    fitter,
    *,
    init_params,
    loss_fn,
    data_args,
    optimizer,
    n_steps,
    tol,
    verbose,
    print_every,
):
    """MAP optimization via jaxopt quasi-Newton / line-search solvers."""
    from tengri.inference.posterior import Posterior

    solver, opt_name = _build_jaxopt_solver(
        optimizer, loss_fn, maxiter=n_steps, tol=tol,
    )

    state = solver.init_state(init_params, data_args)

    # Warmup JIT
    _, warmup_state = solver.update(init_params, state, data_args)
    jax.block_until_ready(warmup_state)

    params = init_params
    state = solver.init_state(init_params, data_args)
    losses = []
    t0 = time.time()

    for i in range(n_steps):
        params, state = solver.update(params, state, data_args)
        loss_val = float(state.value) if hasattr(state, "value") else float(loss_fn(params, data_args))
        losses.append(loss_val)

        if verbose and (i % print_every == 0 or i == n_steps - 1):
            print(f"    step {i}: loss={loss_val:.6f}  err={float(state.error):.2e}")

        if state.error < tol:
            break

    wall_time = time.time() - t0
    n_actual = len(losses)
    final_loss = losses[-1] if losses else float("nan")
    best_params_physical = fitter._to_physical(params)

    if verbose:
        converged = state.error < tol
        cv_msg = " (converged)" if converged else ""
        print(
            f"  MAP ({opt_name}) complete in {wall_time:.1f}s, "
            f"{n_actual} steps{cv_msg}, loss={final_loss:.6f}"
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
            "converged": bool(state.error < tol),
            "grad_norm": float(state.error),
        },
        loss_history=jnp.asarray(losses),
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
    print_every=200,
):
    """MAP optimization via gradient descent or quasi-Newton solvers.

    Parameters
    ----------
    n_steps : int
        Maximum number of optimization steps.
    learning_rate : float
        Learning rate (optax optimizers only; ignored for jaxopt solvers).
    optimizer : str or optax optimizer
        Optax: ``"adam"``, ``"sgd"``, ``"adamw"``, or a pre-built optax optimizer.
        Jaxopt: ``"lbfgs"``, ``"bfgs"``, ``"nonlinear_cg"``, ``"gradient_descent"``.
    early_stopping : bool
        Stop if loss doesn't improve (optax only; jaxopt uses ``tol``).
    patience : int
        Steps to wait for improvement before stopping (optax only).
    rtol : float
        Relative tolerance for early stopping (optax only).
    tol : float
        Gradient norm tolerance for convergence (jaxopt solvers).
    verbose : bool
        Print progress.
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

    # ── jaxopt quasi-Newton / line-search path ──
    if isinstance(optimizer, str) and optimizer in _JAXOPT_SOLVERS:
        return _run_map_jaxopt(
            fitter,
            init_params=init_params,
            loss_fn=loss_fn,
            data_args=data_args,
            optimizer=optimizer,
            n_steps=n_steps,
            tol=tol,
            verbose=verbose,
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

    for batch_idx in range(n_full_batches):
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
        for i in range(n_remainder):
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
