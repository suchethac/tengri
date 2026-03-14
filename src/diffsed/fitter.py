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
                 optimizer="adam", early_stopping=True,
                 patience=200, rtol=1e-5,
                 verbose=True, print_every=200):
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
            raise ImportError("optax required for MAP: pip install optax")

        from diffsed.posterior import Posterior

        loss_fn = self._build_loss_fn()

        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

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
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        params = init_params
        loss_history = []
        best_loss = float("inf")
        steps_without_improvement = 0
        t0 = time.time()

        for i in range(n_steps):
            params, opt_state, loss_val = step(params, opt_state)
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
                            print(f"  Early stopping at step {i} "
                                  f"(no improvement for {patience} steps)")
                        break

        wall_time = time.time() - t0
        best_params = self._to_physical(params)

        if verbose:
            print(f"  MAP ({opt_name}) complete in {wall_time:.1f}s, "
                  f"{len(loss_history)} steps")

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
            _model=self.model,
        )

    def _run_nuts(self, *, key, init_from=None,
                  n_warmup=500, n_samples=1000,
                  target_accept_rate=0.8, max_num_doublings=10,
                  verbose=True):
        """NUTS sampling via BlackJAX.

        Parameters
        ----------
        n_warmup : int
            Warmup/adaptation steps (tunes step size and mass matrix).
        n_samples : int
            Post-warmup samples to collect.
        target_accept_rate : float
            Target acceptance rate for step size adaptation (0.6-0.9).
            Higher = smaller steps = fewer divergences but slower mixing.
        max_num_doublings : int
            Maximum tree depth for NUTS trajectory (2^max_num_doublings leapfrog steps).
        verbose : bool
            Print progress.
        """
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
                  f"{n_samples} samples, target_accept={target_accept_rate}")

        t0 = time.time()

        # Window adaptation with target acceptance rate
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, log_posterior_flat,
            target_acceptance_rate=target_accept_rate,
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
                   n_iterations=10, n_samples=6,
                   sample_mode="nonlinear_resample", verbose=True):
        """Geometric variational inference via NIFTy.re.

        geoVI finds a coordinate transformation where the posterior is
        approximately Gaussian, then draws samples in that space. Much
        faster than MCMC for high-dimensional problems.
        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError(
                "nifty8.re required for geoVI: pip install nifty8[re]"
            )

        from diffsed.posterior import Posterior

        model = self.model
        data = self.data
        noise = self.noise
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        spec = self.spec
        stochastic = spec.stochastic

        # Build signal_response: unbounded primals → predicted observables
        def signal_response(primals):
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(primals[name], lo, hi)
            for name, val in fixed_values.items():
                params[name] = val
            if stochastic and "psd_xi" in primals:
                params["psd_xi"] = primals["psd_xi"]

            if data_type == "photometry":
                return model.predict_photometry(params)
            elif data_type == "spectroscopy":
                return model.predict_spectrum(params, model._wave_obs)
            elif data_type == "joint":
                p = model.predict_photometry(params)
                s = model.predict_spectrum(params, model._wave_obs)
                return jnp.concatenate([p, s])
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

        # Build NIFTy.re domain
        domain = {}
        for name in free_names:
            domain[name] = jft.ShapeWithDtype(())
        if stochastic:
            domain["psd_xi"] = jft.ShapeWithDtype((spec.n_grid,))

        nifty_model = jft.Model(signal_response, domain=domain)

        # Gaussian likelihood: N^-1 = diag(1/noise^2)
        noise_cov_inv = 1.0 / noise ** 2
        likelihood = jft.Gaussian(data, noise_cov_inv).amend(nifty_model)

        # Initialize
        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        # Convert to jft.Vector
        init_pos = jft.Vector(init_params)

        if verbose:
            n_total = len(free_names) + (spec.n_grid if stochastic else 0)
            mode = "geoVI" if sample_mode == "nonlinear_resample" else "MGVI"
            print(f"{mode}: {n_total} params, {len(data)} data points, "
                  f"{n_iterations} iterations")

        t0 = time.time()

        # Sample schedule: increase samples over iterations
        delta = max(1, n_samples - 1)

        key, opt_key = jax.random.split(key)
        samples, state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=lambda i: max(1, 1 + int(i * delta / max(n_iterations - 1, 1))),
            key=opt_key,
            sample_mode=sample_mode,
            odir=None,
        )

        wall_time = time.time() - t0

        # Extract and convert samples to physical space
        sample_list = list(samples)
        n_posterior = len(sample_list)

        samples_phys = {}
        for s in sample_list:
            # jft.Vector → dict via .tree attribute
            sample_dict = s.tree if hasattr(s, 'tree') else dict(s)
            phys = self._to_physical(sample_dict)
            for k, v in phys.items():
                if k not in samples_phys:
                    samples_phys[k] = []
                samples_phys[k].append(v)

        samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
        best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

        if verbose:
            mode = "geoVI" if sample_mode == "nonlinear_resample" else "MGVI"
            print(f"  {mode} complete in {wall_time:.1f}s, "
                  f"{n_posterior} posterior samples")

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method=f"{'geoVI' if sample_mode == 'nonlinear_resample' else 'MGVI'} (NIFTy.re)",
            wall_time_s=wall_time,
            diagnostics={
                "n_iterations": n_iterations,
                "n_samples": n_posterior,
                "sample_mode": sample_mode,
            },
            loss_history=None,
            _model=self.model,
        )
