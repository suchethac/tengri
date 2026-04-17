"""MAP optimization, Laplace approximation, and Pathfinder.

Extracted from fitter.py.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp


def run_map(
    fitter,
    *,
    key,
    init_from=None,
    n_steps=1000,
    learning_rate=0.02,
    optimizer="adam",
    early_stopping=True,
    patience=200,
    rtol=1e-5,
    verbose=True,
    print_every=200,
):
    """MAP optimization via gradient descent.

    Parameters
    ----------
    n_steps : int
        Maximum number of optimization steps.
    learning_rate : float
        Learning rate for the optimizer.
    optimizer : str or optax optimizer
        "adam", "sgd", "adamw", or a pre-built optax optimizer.
    early_stopping : bool
        Stop if loss doesn't improve by rtol over patience steps.
    patience : int
        Number of steps to wait for improvement before stopping.
    rtol : float
        Relative tolerance for early stopping.
    verbose : bool
        Print progress.
    print_every : int
        Print interval.
    """
    try:
        import optax
    except ImportError:
        raise ImportError("optax required for MAP: pip install optax") from None

    from tengri.inference.posterior import Posterior

    loss_fn = fitter._get_or_build_loss_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    # Build optimizer
    if isinstance(optimizer, str):
        opt_builders = {
            "adam": lambda: optax.adam(learning_rate),
            "adamw": lambda: optax.adamw(learning_rate),
            "sgd": lambda: optax.sgd(learning_rate, momentum=0.9),
        }
        if optimizer not in opt_builders:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Use {list(opt_builders.keys())} or pass an optax optimizer."
            )
        opt = opt_builders[optimizer]()
        opt_name = optimizer.upper()
    else:
        opt = optimizer
        opt_name = "custom"

    opt_state = opt.init(init_params)

    def _step_body(params, opt_state):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, data_args))(params)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Fused scan: runs all steps inside XLA with zero Python dispatch.
    # Early stopping is handled via a converged flag in the carry —
    # once converged, subsequent steps are no-ops (params unchanged).
    patience_i32 = jnp.int32(patience)
    rtol_f = jnp.float64(rtol)

    @jax.jit
    def _run_scan(init_params, opt_state):
        def scan_body(carry, _x):
            params, ostate, best_loss, stale, converged = carry
            # If already converged, skip (return same state)
            new_params, new_ostate, loss = _step_body(params, ostate)
            # Update early stopping counters
            improved = loss < best_loss * (1.0 - rtol_f)
            new_best = jnp.where(improved, loss, best_loss)
            new_stale = jnp.where(improved, jnp.int32(0), stale + 1)
            new_converged = converged | (new_stale >= patience_i32)
            # If converged, keep old params (don't take the step)
            out_params = jax.lax.cond(converged, lambda: params, lambda: new_params)
            out_ostate = jax.lax.cond(converged, lambda: ostate, lambda: new_ostate)
            out_loss = jax.lax.cond(converged, lambda: best_loss, lambda: loss)
            new_carry = (
                out_params,
                out_ostate,
                jnp.where(converged, best_loss, new_best),
                jnp.where(converged, stale, new_stale),
                new_converged,
            )
            return new_carry, out_loss

        init_carry = (
            init_params,
            opt_state,
            jnp.float64(jnp.inf),  # best_loss
            jnp.int32(0),  # steps without improvement
            jnp.bool_(not early_stopping),  # converged (skip ES if disabled)
        )
        final_carry, loss_history = jax.lax.scan(scan_body, init_carry, xs=None, length=n_steps)
        final_params = final_carry[0]
        final_converged = final_carry[4]
        final_stale = final_carry[3]
        return final_params, loss_history, final_converged, final_stale

    t0 = time.time()
    params, loss_arr, _converged, _stale_count = _run_scan(init_params, opt_state)
    # Block until done (single sync at the end, not per-step)
    loss_arr.block_until_ready()
    wall_time = time.time() - t0

    # Find actual number of steps taken (before convergence).
    # Single bulk transfer from device to host (not per-element).
    import numpy as np

    loss_np = np.asarray(loss_arr)

    if early_stopping:
        # Loss history has real values up to convergence, then repeats
        n_actual = n_steps
        best_l = np.inf
        stale = 0
        for i in range(n_steps):
            l_val = loss_np[i]
            if l_val < best_l * (1 - rtol):
                best_l = l_val
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    n_actual = i + 1
                    break
        loss_np = loss_np[:n_actual]
    else:
        n_actual = n_steps

    best_params = fitter._to_physical(params)

    final_loss = float(loss_np[-1]) if len(loss_np) > 0 else float("nan")

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
        params=best_params,
        method=f"MAP ({opt_name})",
        wall_time_s=wall_time,
        diagnostics={
            "n_steps": n_actual,
            "final_loss": final_loss,
            "optimizer": opt_name,
        },
        loss_history=jnp.asarray(loss_np),
        _model=fitter.model,
    )


def run_laplace(fitter, *, key, init_from=None, n_map_steps=1000, **kwargs):
    """Laplace approximation: Gaussian posterior from Hessian at MAP."""
    from tengri.inference.laplace import run_laplace

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
    from tengri.inference.pathfinder import run_pathfinder

    loss_fn = fitter._get_or_build_loss_fn()
    data_args = fitter._data_args

    if init_from is not None:
        init_params = fitter._unbounded_from_posterior(init_from)
    else:
        init_params = fitter._initialize_unbounded(key)

    return run_pathfinder(
        key=key,
        loss_fn=loss_fn,
        data_args=data_args,
        init_params=init_params,
        to_physical_fn=fitter._to_physical,
        model=fitter.model,
        **kwargs,
    )
