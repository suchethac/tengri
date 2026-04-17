"""MAP (Maximum A Posteriori) optimization via optax (Adam, SGD, etc.).

The simplest inference method: find the single parameter set that
maximizes the posterior (equivalently, minimizes the loss).
No uncertainty quantification, but very fast — useful for:
- Quick initialization before MCMC
- Checking the forward model works
- Sanity checks on parameter recovery

Usage:
    from tengri.inference.map_optimizer import fit_map
    result = fit_map(model, data, noise, n_steps=1000)
"""

import time

import jax
import jax.numpy as jnp

from tengri.inference.common import (
    DEFAULT_PRIOR,
    InferenceResult,
    build_loss_fn,
    initialize_params,
    unbounded_to_physical,
)


def fit_map(
    forward_model,
    data,
    noise,
    prior_config=None,
    data_type="photometry",
    n_steps=1000,
    learning_rate=1e-2,
    key=None,
    init_params=None,
    verbose=True,
    print_every=200,
):
    """Fit a galaxy via MAP optimization using Adam.

    Parameters
    ----------
    forward_model : ForwardModel
        Configured forward model.
    data : array
        Observed data.
    noise : array
        1-sigma uncertainties.
    prior_config : PriorConfig, optional
        Prior bounds.
    data_type : str
        "photometry", "spectroscopy", or "joint".
    n_steps : int
        Number of Adam steps.
    learning_rate : float
        Adam learning rate.
    key : PRNGKey, optional
        For initialization. Defaults to PRNGKey(0).
    init_params : dict, optional
        Initial parameters (unbounded). If None, random init.
    verbose : bool
        Print progress.
    print_every : int
        Print interval.

    Returns
    -------
    InferenceResult
        Best-fit parameters, loss history, timing.
    """
    try:
        import optax
    except ImportError:
        raise ImportError("optax required for MAP optimization: pip install optax") from None

    if prior_config is None:
        prior_config = DEFAULT_PRIOR
    if key is None:
        key = jax.random.PRNGKey(0)

    # Build loss and initialize
    loss_fn = build_loss_fn(forward_model, data, noise, prior_config, data_type)

    if init_params is None:
        init_params = initialize_params(key, forward_model.config.n_grid, prior_config)

    # Set up Adam optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)

    # JIT-compile the update step
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Optimization loop
    params = init_params
    loss_history = []
    t0 = time.time()

    for i in range(n_steps):
        params, opt_state, loss_val = step(params, opt_state)
        loss_history.append(float(loss_val))

        if verbose and (i % print_every == 0 or i == n_steps - 1):
            print(f"  Step {i:5d}/{n_steps}: loss = {loss_val:.4f}")

    wall_time = time.time() - t0

    # Convert to physical parameters
    best_params = unbounded_to_physical(params, prior_config)

    if verbose:
        print(f"  MAP optimization complete in {wall_time:.1f}s")

    return InferenceResult(
        params=best_params,
        samples=None,
        loss_history=jnp.array(loss_history),
        wall_time_s=wall_time,
        method="MAP (Adam)",
        diagnostics={"n_steps": n_steps, "final_loss": loss_history[-1]},
    )
