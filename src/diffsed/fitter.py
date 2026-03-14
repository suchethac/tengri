"""Inference engine: fit observed data using MAP, NUTS, or geoVI.

The Fitter separates inference strategy from the forward model. It builds
a loss function from the Model's predictions and the ParamSpec's priors,
then runs the chosen optimizer/sampler.

Usage:
    from diffsed import Model, Fitter

    fitter = Fitter(model, data, noise)
    result_map = fitter.run("map", n_steps=1500)
    result_nuts = fitter.run("nuts", init_from=result_map, n_warmup=500)
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from diffsed.distributions import Uniform, Gaussian, LogUniform, Fixed
from diffsed.utils.transforms import to_bounded, to_unbounded


class Fitter:
    """Inference engine for diffsed models.

    Parameters
    ----------
    model : Model
        Configured forward model.
    data : array
        Observed data (photometry or spectrum).
    noise : array
        1-sigma uncertainties.
    data_type : str
        "photometry", "spectroscopy", or "joint".
    """

    def __init__(self, model, data, noise, data_type="photometry"):
        self.model = model
        self.data = jnp.asarray(data)
        self.noise = jnp.asarray(noise)
        self.data_type = data_type
        self.spec = model.spec

        # Separate free and fixed parameters
        self._free_names = self.spec.free_params
        self._fixed_values = self.spec.get_fixed_values()

        # Build bounds for free params
        self._bounds = {}
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            self._bounds[name] = dist.bounds

    # -------------------------------------------------------------------
    # Loss function construction
    # -------------------------------------------------------------------

    def _build_loss_fn(self):
        """Build a differentiable loss function.

        The loss function takes an unbounded parameter dict and returns
        a scalar: chi² + prior penalties.
        """
        model = self.model
        data = self.data
        noise = self.noise
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        spec = self.spec
        stochastic = spec.stochastic

        def loss_fn(params_unbounded):
            # Convert unbounded → physical for free params
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(params_unbounded[name], lo, hi)

            # Merge fixed values
            for name, val in fixed_values.items():
                params[name] = val

            # Add psd_xi if stochastic
            if stochastic and "psd_xi" in params_unbounded:
                params["psd_xi"] = params_unbounded["psd_xi"]

            # Forward model prediction
            if data_type == "photometry":
                predicted = model.predict_photometry(params)
            elif data_type == "spectroscopy":
                predicted = model.predict_spectrum(params, model._wave_obs)
            elif data_type == "joint":
                pred_phot = model.predict_photometry(params)
                pred_spec = model.predict_spectrum(params, model._wave_obs)
                predicted = jnp.concatenate([pred_phot, pred_spec])
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            # Chi-squared
            chi2 = jnp.sum(((data - predicted) / noise) ** 2)

            # Prior contributions
            prior_penalty = 0.0

            # Standard normal prior on psd_xi
            if stochastic and "psd_xi" in params_unbounded:
                prior_penalty += jnp.sum(params_unbounded["psd_xi"] ** 2)

            # Additional prior contributions for non-Uniform distributions
            for name in free_names:
                dist = spec.get_distribution(name)
                if isinstance(dist, Gaussian):
                    val = params[name]
                    prior_penalty -= 2.0 * dist.log_prob(val)
                elif isinstance(dist, LogUniform):
                    val = params[name]
                    # LogUniform correction: log_prob difference from Uniform
                    uniform_lp = -jnp.log(dist.hi - dist.lo)
                    prior_penalty -= 2.0 * (dist.log_prob(val) - uniform_lp)

            return 0.5 * chi2 + 0.5 * prior_penalty

        return loss_fn

    def _initialize_unbounded(self, key):
        """Create initial unbounded parameter dict."""
        params = {}
        keys = jax.random.split(key, len(self._free_names) + 1)

        for i, name in enumerate(self._free_names):
            dist = self.spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                # Initialize at mu in unbounded space
                lo, hi = dist.bounds
                params[name] = to_unbounded(
                    jnp.array(dist.mu), lo, hi
                )
            else:
                # Initialize near midpoint (u=0) with small perturbation
                params[name] = 0.1 * jax.random.normal(keys[i])

        if self.spec.stochastic:
            params["psd_xi"] = 0.1 * jax.random.normal(
                keys[-1], shape=(self.spec.n_grid,)
            )

        return params

    def _unbounded_from_posterior(self, posterior):
        """Convert a Posterior's params to unbounded space for init."""
        params = {}
        for name in self._free_names:
            if name in posterior.params:
                lo, hi = self._bounds[name]
                val = jnp.clip(
                    jnp.array(posterior.params[name]),
                    lo + 1e-6, hi - 1e-6
                )
                params[name] = to_unbounded(val, lo, hi)
            else:
                params[name] = jnp.array(0.0)

        if self.spec.stochastic and "psd_xi" in posterior.params:
            params["psd_xi"] = posterior.params["psd_xi"]
        elif self.spec.stochastic:
            params["psd_xi"] = jnp.zeros(self.spec.n_grid)

        return params

    # -------------------------------------------------------------------
    # Convert unbounded samples to physical space
    # -------------------------------------------------------------------

    def _to_physical(self, params_unbounded):
        """Convert a single unbounded param dict to physical space."""
        params = {}
        for name in self._free_names:
            lo, hi = self._bounds[name]
            params[name] = to_bounded(params_unbounded[name], lo, hi)
        for name, val in self._fixed_values.items():
            params[name] = jnp.array(val)
        if self.spec.stochastic and "psd_xi" in params_unbounded:
            params["psd_xi"] = params_unbounded["psd_xi"]
        return params

    # -------------------------------------------------------------------
    # Inference methods
    # -------------------------------------------------------------------

    def run(self, method, *, init_from=None, key=None, **kwargs):
        """Run inference.

        Parameters
        ----------
        method : str
            "map", "nuts", or "geovi".
        init_from : Posterior, optional
            Use a previous result as initialization.
        key : PRNGKey, optional
            Random key.
        **kwargs
            Method-specific arguments.

        Returns
        -------
        Posterior
            Inference results.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        if method == "map":
            return self._run_map(key=key, init_from=init_from, **kwargs)
        elif method == "nuts":
            return self._run_nuts(key=key, init_from=init_from, **kwargs)
        elif method == "geovi":
            return self._run_geovi(key=key, init_from=init_from, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'map', 'nuts', or 'geovi'.")

    def _run_map(self, *, key, init_from=None,
                 n_steps=1000, learning_rate=0.02,
                 verbose=True, print_every=200):
        """MAP optimization via Adam."""
        try:
            import optax
        except ImportError:
            raise ImportError("optax required for MAP: pip install optax")

        from diffsed.posterior import Posterior

        loss_fn = self._build_loss_fn()

        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(init_params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        params = init_params
        loss_history = []
        t0 = time.time()

        for i in range(n_steps):
            params, opt_state, loss_val = step(params, opt_state)
            loss_history.append(float(loss_val))
            if verbose and (i % print_every == 0 or i == n_steps - 1):
                print(f"  Step {i:5d}/{n_steps}: loss = {loss_val:.4f}")

        wall_time = time.time() - t0
        best_params = self._to_physical(params)

        if verbose:
            print(f"  MAP complete in {wall_time:.1f}s")

        return Posterior(
            samples=None,
            params=best_params,
            method="MAP (Adam)",
            wall_time_s=wall_time,
            diagnostics={"n_steps": n_steps, "final_loss": loss_history[-1]},
            loss_history=jnp.array(loss_history),
            _model=self.model,
        )

    def _run_nuts(self, *, key, init_from=None,
                  n_warmup=500, n_samples=1000, verbose=True):
        """NUTS sampling via BlackJAX."""
        try:
            import blackjax
        except ImportError:
            raise ImportError("blackjax required for NUTS: pip install blackjax")

        from diffsed.posterior import Posterior

        # Warn about high dimensionality
        if self.spec.stochastic:
            n_total = self.spec.n_free + self.spec.n_grid
            warnings.warn(
                f"Stochastic SFH with NUTS: sampling {n_total} dimensions "
                f"({self.spec.n_grid} psd_xi + {self.spec.n_free} physical). "
                f"This is computationally expensive. "
                f"Recommended: method='geovi' (10-100x faster).",
                stacklevel=3,
            )

        loss_fn = self._build_loss_fn()

        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        # Flatten for BlackJAX
        init_flat, unravel_fn = ravel_pytree(init_params)

        def log_posterior_flat(position):
            params = unravel_fn(position)
            return -loss_fn(params)

        if verbose:
            n_dim = len(init_flat)
            print(f"NUTS: {n_dim} parameters, {n_warmup} warmup, "
                  f"{n_samples} samples")

        t0 = time.time()

        # Window adaptation
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, log_posterior_flat,
        )
        (state, parameters), _ = warmup.run(
            warmup_key, init_flat, num_steps=n_warmup
        )

        if verbose:
            print(f"  Warmup complete ({time.time() - t0:.1f}s). "
                  f"Step size: {float(parameters['step_size']):.4f}")

        # Sampling
        kernel = blackjax.nuts(log_posterior_flat, **parameters).step

        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state.position, info)

        key, sample_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, n_samples)

        all_positions = []
        n_divergent = 0

        for i, sk in enumerate(sample_keys):
            state, (position, info) = one_step(state, sk)
            all_positions.append(position)
            if hasattr(info, "is_divergent"):
                n_divergent += int(info.is_divergent)
            if verbose and ((i + 1) % 200 == 0 or i == n_samples - 1):
                print(f"  Sample {i+1}/{n_samples}")

        wall_time = time.time() - t0
        positions = jnp.stack(all_positions)

        # Unravel and convert to physical
        samples_phys = {}
        for i in range(n_samples):
            sample_u = unravel_fn(positions[i])
            sample_p = self._to_physical(sample_u)
            for k, v in sample_p.items():
                if k not in samples_phys:
                    samples_phys[k] = []
                samples_phys[k].append(v)

        samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
        best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

        if verbose:
            print(f"  NUTS complete in {wall_time:.1f}s. "
                  f"Divergences: {n_divergent}/{n_samples}")

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method="NUTS (BlackJAX)",
            wall_time_s=wall_time,
            diagnostics={
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "n_divergent": n_divergent,
                "step_size": float(parameters["step_size"]),
            },
            loss_history=None,
            _model=self.model,
        )

    def _run_geovi(self, *, key, init_from=None,
                   n_iterations=10, n_samples=6, verbose=True):
        """Geometric variational inference via NIFTy.re."""
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError(
                "nifty8.re required for geoVI: pip install nifty8[re]"
            )

        from diffsed.posterior import Posterior

        # geoVI implementation would go here — for now, raise a clear message
        raise NotImplementedError(
            "geoVI integration requires NIFTy.re signal_response setup. "
            "Use method='map' or method='nuts' for now."
        )
