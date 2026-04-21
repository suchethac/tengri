"""Protocol interfaces for inference backends in tengri.

This module defines the structural types (Protocols) that inference
backends (optimization, MCMC, VI, nested sampling) must implement.

Contributors adding new inference methods should ensure their
implementations match these signatures.
"""

from typing import Protocol

import jax.numpy as jnp

__all__ = ["InferenceBackend"]


class InferenceBackend(Protocol):
    """Protocol for inference backend runners.

    An inference backend encapsulates a statistical method (e.g., MAP
    optimization, MCMC, variational inference, nested sampling) and
    provides a consistent interface for the Fitter to call.

    Each backend receives a loss function (negative log posterior) and
    must return a Posterior object with samples/best-fit parameters,
    diagnostics, and timing information.

    Examples
    --------
    Implementing a simple optimization backend::

        def run_my_optimizer(
            fitter,
            *,
            key,
            init_from=None,
            n_steps=1000,
            learning_rate=1e-3,
            verbose=True,
        ):
            '''My custom optimization backend.'''
            import time
            import jax
            import optax

            t0 = time.time()

            # Get loss function from fitter
            loss_fn = fitter.loss_fn

            # Initialize parameters
            params = fitter.spec.initialize(key)
            if init_from is not None:
                params = init_from.params

            # Set up optimizer
            opt = optax.adam(learning_rate)
            opt_state = opt.init(params)

            # Optimization loop
            for step in range(n_steps):
                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            wall_time = time.time() - t0

            # Return Posterior
            from tengri.inference.posterior import Posterior

            return Posterior(
                samples=None,  # point estimate only
                params=params,
                method="my_optimizer",
                wall_time_s=wall_time,
                diagnostics={"n_steps": n_steps},
                _model=fitter.model,
                _fitter=fitter,
            )

    Registering it::

        from tengri.inference.fitter import _BACKEND_REGISTRY

        _BACKEND_REGISTRY["my_optimizer"] = run_my_optimizer

    Using it::

        result = fitter.run("my_optimizer", n_steps=1000, learning_rate=1e-3)
    """

    def __call__(
        self,
        fitter,
        *,
        key: jnp.ndarray,
        **backend_kwargs,
    ):
        """Run inference.

        Parameters
        ----------
        fitter : Fitter
            Fitter instance providing:
            - ``loss_fn(params) -> scalar``: negative log posterior
            - ``spec``: Parameters specification (for initialization)
            - ``model``: SEDModel instance
            - ``data_args``: (data, noise, etc.) for likelihood
        key : jax.Array
            PRNG key for random initialization and sampling.
        **backend_kwargs
            Method-specific hyperparameters (e.g., n_steps, n_samples,
            learning_rate, temperature, divergence_threshold).
            Common kwargs:
            - ``init_from``: Posterior or "map" to warm-start
            - ``verbose``: bool for progress logging
            - ``n_steps`` or ``n_samples``: iteration counts

        Returns
        -------
        Posterior
            Inference results with:
            - ``samples``: dict[str, Array] of posterior samples (or None for MAP)
            - ``params``: dict[str, float] of best-fit or posterior mean parameters
            - ``method``: str name of the inference method
            - ``wall_time_s``: float total runtime in seconds
            - ``diagnostics``: dict of method-specific metrics
            - ``_model``: reference to fitter.model (for derived quantities)
            - ``_fitter``: reference to fitter (for re-running, re-sampling)

        Notes
        -----
        The returned Posterior must pass:

        1. **Parameter consistency**: ``params`` keys match ``spec.free_params``
        2. **Sample shape**: ``samples[key].shape[0] == n_samples`` for all keys
        3. **Physical bounds**: Parameters respect prior bounds (checked at
           Posterior init via spec.validate)
        4. **Wall time**: ``wall_time_s >= 0`` and represents the backend
           computation only (not model setup or post-processing)
        """
        ...
