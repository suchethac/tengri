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

    @jax.jit
    def step(params, opt_state, data_args):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, data_args))(params)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    params = init_params
    loss_history = []
    best_loss = float("inf")
    steps_without_improvement = 0
    t0 = time.time()

    for i in range(n_steps):
        params, opt_state, loss_val = step(params, opt_state, data_args)
        current_loss = float(loss_val)
        loss_history.append(current_loss)

        if verbose and (i % print_every == 0 or i == n_steps - 1):
            print(f"  Step {i:5d}/{n_steps}: loss = {loss_val:.4f}")

        # Early stopping
        if early_stopping:
            if current_loss < best_loss * (1 - rtol):
                best_loss = current_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    if verbose:
                        print(
                            f"  Early stopping at step {i} (no improvement for {patience} steps)"
                        )
                    break

    wall_time = time.time() - t0
    best_params = fitter._to_physical(params)

    if verbose:
        print(f"  MAP ({opt_name}) complete in {wall_time:.1f}s, {len(loss_history)} steps")

    return Posterior(
        samples=None,
        params=best_params,
        method=f"MAP ({opt_name})",
        wall_time_s=wall_time,
        diagnostics={
            "n_steps": len(loss_history),
            "final_loss": loss_history[-1],
            "optimizer": opt_name,
        },
        loss_history=jnp.array(loss_history),
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
