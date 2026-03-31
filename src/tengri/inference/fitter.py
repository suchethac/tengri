"""Inference engine: fit observed data using MAP, NUTS, Ray Tracing, or geoVI.

The Fitter separates inference strategy from the forward model. It builds
a loss function from the Model's predictions and the ParamSpec's priors,
then runs the chosen optimizer/sampler.

Usage:
    from tengri import Model, Fitter

    fitter = Fitter(model, data, noise)
    result_map = fitter.run("map", n_steps=1500)
    result_rts = fitter.run("raytrace", init_from=result_map)
    result_nuts = fitter.run("nuts", init_from=result_map, n_warmup=500)
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.distributions import Gaussian, LogUniform
from tengri.utils.transforms import to_bounded, to_unbounded


def _simple_cg(mat_fn, b, x0, maxiter=30, miniter=6):
    """Lightweight CG solve for catalog fitting. JIT-friendly."""
    _eps = 6.0 * jnp.finfo(jnp.float64).eps
    r = mat_fn(x0) - b
    d = r
    gamma = jnp.dot(r, r)
    energy = jnp.dot((r - b) / 2, x0)
    init = (x0, r, d, gamma, energy, jnp.int32(-2), jnp.int32(0))

    def cond(s):
        return s[5] < -1

    def body(s):
        x, r, d, pg, pe, info, i = s
        i = i + 1
        q = mat_fn(d)
        curv = jnp.dot(d, q)
        alpha = pg / curv
        info = jnp.where(curv <= 0.0, jnp.int32(0), info)
        alpha = jnp.where(curv <= 0.0, 0.0, alpha)
        x = x - alpha * d
        r = jnp.where((i % 20 == 0) & (info < -1), mat_fn(x) - b, r - alpha * q)
        gamma = jnp.dot(r, r)
        energy = jnp.dot((r - b) / 2, x)
        ed = pe - energy
        info = jnp.where(ed < -_eps * jnp.abs(energy), jnp.int32(-1), info)
        info = jnp.where((ed < 1e-4) & (i >= miniter) & (info < -1), jnp.int32(0), info)
        info = jnp.where((i >= maxiter) & (info < -1), i, info)
        d = d * jnp.maximum(0.0, gamma / (pg + 1e-30)) + r
        return (x, r, d, gamma, energy, info, i)

    return jax.lax.while_loop(cond, body, init)[0]


class Fitter:
    """Inference engine for tengri models.

    Parameters
    ----------
    model : Model
        Configured forward model.
    data : array
        Observed data (photometry or spectrum).
    noise : array
        1-sigma uncertainties.
    data_type : str or None
        ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
        If ``None`` (default), inferred from ``model.observation``.
        Explicit values override the inferred type.
    calibration_marginalize : bool
        If ``True``, analytically marginalize over spectroscopic
        calibration polynomial coefficients (Chebyshev) when computing
        the spectroscopic log-likelihood.  Only applies when
        ``data_type`` is ``"spectroscopy"`` or ``"joint"``.
        Follows the Prospector approach (Johnson et al. 2021).
        Default ``False``.
    cal_n_poly : int
        Number of Chebyshev polynomial coefficients for calibration
        marginalization (order 1 through ``cal_n_poly``).  Default 3.
    cal_prior_sigma : float
        Standard deviation of the Gaussian prior on each calibration
        coefficient.  Default 1.0.
    """

    def __init__(
        self,
        model,
        data,
        noise,
        data_type=None,
        calibration_marginalize=False,
        cal_n_poly=3,
        cal_prior_sigma=1.0,
    ):
        self.model = model
        self.data = jnp.asarray(data)
        self.noise = jnp.asarray(noise)

        # Infer data_type from Observation if not provided
        if data_type is None:
            obs = getattr(model, "observation", None)
            if obs is not None:
                data_type = obs.data_type
            else:
                data_type = "photometry"  # backward compat default

        self.data_type = data_type
        self.spec = model.spec

        # Calibration marginalization settings
        self._calibration_marginalize = calibration_marginalize
        self._cal_n_poly = cal_n_poly
        self._cal_prior_sigma = cal_prior_sigma
        self._has_spectroscopy = data_type in ("spectroscopy", "joint")

        # Separate free and fixed parameters
        self._free_names = self.spec.free_params
        self._fixed_values = self.spec.get_fixed_values()

        # Build bounds for free params
        self._bounds = {}
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            self._bounds[name] = dist.bounds

        # Pre-compute data-dependent arguments passed to JIT'd functions.
        # These are passed as explicit arguments (not closed over) so that
        # engines compiled for one galaxy can be reused for another with
        # the same model + parameter structure.
        noise_inv = 1.0 / self.noise**2
        self._data_args = {
            "data": self.data,
            "noise": self.noise,
            "noise_inv": noise_inv,
            "sqrt_noise_inv": jnp.sqrt(noise_inv),
            "n_data": jnp.int32(len(self.data)),
        }

        # JIT posterior sampler — call compile() to pre-compile, or it
        # compiles lazily on first VI run.
        self._jit_sampler = None

    def _engine_cache_key(self):
        """Return a hashable key identifying the JIT engine shape.

        Two Fitters sharing the same Model will reuse the same compiled
        engine if their cache keys match (same data_type, stochastic
        flag, latent dimension, data length, free parameter names, and
        noise model presence).
        """
        from tengri.core.noise import has_noise_model

        return (
            self.data_type,
            self.spec.stochastic,
            self.spec.n_grid if self.spec.stochastic else 0,
            len(self.data),
            tuple(sorted(self._free_names)),
            has_noise_model(self.spec),
        )

    def _get_or_build_engine(self, pos_dict):
        """Return the JIT engine, reusing a cached version when possible.

        Engines are cached on the Model object so that multiple Fitters
        created with the same Model (but different data) share the same
        compiled XLA programs.
        """
        if self._jit_sampler is not None:
            return self._jit_sampler

        cache_key = self._engine_cache_key()
        if not hasattr(self.model, "_jit_engine_cache"):
            self.model._jit_engine_cache = {}
        if cache_key in self.model._jit_engine_cache:
            self._jit_sampler = self.model._jit_engine_cache[cache_key]
            return self._jit_sampler

        engine = self._build_jit_engine(pos_dict)
        self.model._jit_engine_cache[cache_key] = engine
        self._jit_sampler = engine
        return engine

    def summary(self) -> str:
        """Return a human-readable summary of the fitting problem.

        Returns
        -------
        str
            Formatted summary showing data shape, free parameters,
            priors, bounds, and available inference methods.
        """
        sep = "─" * 66
        lines: list[str] = [f"Fitter  data_type: {self.data_type}", sep]

        # Data shape
        n_data = self.data.shape[0]
        snr_med = float(jnp.median(jnp.abs(self.data / self.noise)))
        lines.append(f"  Data points: {n_data}")
        lines.append(f"  Median S/N:  {snr_med:.1f}")

        # Dimensionality
        n_free = len(self._free_names)
        n_grid = self.model._n_grid if self.model._has_field else 0
        dim_str = f"{n_free} free"
        if n_grid:
            dim_str += f" + {n_grid} latent (ξ)"
        lines.append(f"  Parameters:  {dim_str}")
        lines.append("")

        # Free parameter table
        hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds'}"
        lines.append(hdr)
        lines.append("  " + "─" * 64)
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            lo, hi = dist.bounds
            lines.append(f"  {name:<32s} {dist!r:<26s} [{lo:.4g}, {hi:.4g}]")

        # Available methods
        lines.append("")
        lines.append(
            "  Methods:     map, raytrace, nuts, geovi, mgvi, geovi_nuts, "
            "laplace, pathfinder, elliptical_slice"
        )

        lines.append(sep)
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Loss function construction
    # -------------------------------------------------------------------

    def _build_loss_fn(self):
        """Build a differentiable loss function.

        The loss function takes an unbounded parameter dict and returns
        a scalar: chi² + prior penalties. When noise model is active,
        uses effective noise with calibration floor and log-determinant.
        """
        from tengri.core.noise import (
            get_noise_dof,
            has_noise_model,
            uses_student_t,
            variable_noise_hamiltonian,
        )

        model = self.model
        data = self.data
        noise = self.noise
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        spec = self.spec
        stochastic = spec.stochastic
        use_variable_noise = has_noise_model(spec)
        noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
        use_cal_marg = self._calibration_marginalize and self._has_spectroscopy
        cal_n_poly = self._cal_n_poly
        cal_prior_sigma = self._cal_prior_sigma

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

            # Likelihood energy (with variable noise / Student-t if active)
            if use_cal_marg and data_type == "spectroscopy":
                # Analytically marginalize over calibration polynomial
                from tengri.models.observation.calibration import (
                    marginalize_calibration,
                )

                log_like_spec, _c_hat, _c_err = marginalize_calibration(
                    predicted,
                    data,
                    noise,
                    model._wave_obs,
                    n_poly=cal_n_poly,
                    prior_sigma=cal_prior_sigma,
                )
                e_lh = -log_like_spec
            elif use_cal_marg and data_type == "joint":
                # Joint: marginalize spectroscopic part, standard chi2 for photometry
                from tengri.models.observation.calibration import (
                    marginalize_calibration,
                )

                n_phot = model.predict_photometry(params).shape[0]
                data_phot = data[:n_phot]
                data_spec = data[n_phot:]
                noise_phot = noise[:n_phot]
                noise_spec = noise[n_phot:]
                pred_phot = predicted[:n_phot]
                pred_spec = predicted[n_phot:]

                chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
                log_like_spec, _c_hat, _c_err = marginalize_calibration(
                    pred_spec,
                    data_spec,
                    noise_spec,
                    model._wave_obs,
                    n_poly=cal_n_poly,
                    prior_sigma=cal_prior_sigma,
                )
                e_lh = 0.5 * chi2_phot - log_like_spec
            elif use_variable_noise:
                f_cal = params.get("noise_frac_cal", 0.0)
                e_lh = variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
            else:
                chi2 = jnp.sum(((data - predicted) / noise) ** 2)
                e_lh = 0.5 * chi2

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

            return e_lh + 0.5 * prior_penalty

        return loss_fn

    def _build_logprior_fn(self):
        """Build a log-prior function in physical parameter space.

        Returns a function: dict of free params → scalar log-prior.
        """
        spec = self.spec
        free_names = self._free_names

        def logprior_fn(free_params):
            lp = 0.0
            for name in free_names:
                dist = spec.get_distribution(name)
                lp = lp + dist.log_prob(free_params[name])
            return lp

        return logprior_fn

    def _build_loglikelihood_fn(self):
        """Build a log-likelihood function in physical parameter space.

        Returns a function: dict of free params → scalar log-likelihood.
        Fixed parameters are automatically merged.
        """
        from tengri.core.noise import (
            get_noise_dof,
            has_noise_model,
            uses_student_t,
            variable_noise_hamiltonian,
        )

        model = self.model
        data = self.data
        noise = self.noise
        data_type = self.data_type
        fixed_values = self._fixed_values
        spec = self.spec
        use_variable_noise = has_noise_model(spec)
        noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
        use_cal_marg = self._calibration_marginalize and self._has_spectroscopy
        cal_n_poly = self._cal_n_poly
        cal_prior_sigma = self._cal_prior_sigma

        def loglikelihood_fn(free_params):
            # Merge free + fixed
            params = dict(free_params)
            for name, val in fixed_values.items():
                params[name] = val

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

            # Log-likelihood with optional calibration marginalization
            if use_cal_marg and data_type == "spectroscopy":
                from tengri.models.observation.calibration import (
                    marginalize_calibration,
                )

                log_like_spec, _c_hat, _c_err = marginalize_calibration(
                    predicted,
                    data,
                    noise,
                    model._wave_obs,
                    n_poly=cal_n_poly,
                    prior_sigma=cal_prior_sigma,
                )
                return log_like_spec
            elif use_cal_marg and data_type == "joint":
                from tengri.models.observation.calibration import (
                    marginalize_calibration,
                )

                n_phot = model.predict_photometry(params).shape[0]
                data_phot = data[:n_phot]
                data_spec = data[n_phot:]
                noise_phot = noise[:n_phot]
                noise_spec = noise[n_phot:]
                pred_phot = predicted[:n_phot]
                pred_spec = predicted[n_phot:]

                chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
                log_like_spec, _c_hat, _c_err = marginalize_calibration(
                    pred_spec,
                    data_spec,
                    noise_spec,
                    model._wave_obs,
                    n_poly=cal_n_poly,
                    prior_sigma=cal_prior_sigma,
                )
                return -0.5 * chi2_phot + log_like_spec
            elif use_variable_noise:
                f_cal = params.get("noise_frac_cal", 0.0)
                return -variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
            else:
                chi2 = jnp.sum(((data - predicted) / noise) ** 2)
                return -0.5 * chi2

        return loglikelihood_fn

    def _initialize_unbounded(self, key):
        """Create initial unbounded parameter dict."""
        params = {}
        keys = jax.random.split(key, len(self._free_names) + 1)

        for i, name in enumerate(self._free_names):
            dist = self.spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                # Initialize at mu in unbounded space
                lo, hi = dist.bounds
                params[name] = to_unbounded(jnp.array(dist.mu), lo, hi)
            else:
                # Initialize near midpoint (u=0) with small perturbation
                params[name] = 0.1 * jax.random.normal(keys[i])

        if self.spec.stochastic:
            params["psd_xi"] = 0.1 * jax.random.normal(keys[-1], shape=(self.spec.n_grid,))

        return params

    def _unbounded_from_posterior(self, posterior):
        """Convert a Posterior's params to unbounded space for init."""
        params = {}
        for name in self._free_names:
            if name in posterior.params:
                lo, hi = self._bounds[name]
                val = jnp.clip(jnp.array(posterior.params[name]), lo + 1e-6, hi - 1e-6)
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
    # Posterior sampling
    # -------------------------------------------------------------------

    def _draw_posterior_samples(
        self,
        likelihood,
        pos_dict,
        key,
        n_samples,
        existing_samples,
        *,
        method="jit",
        verbose=True,
    ):
        """Draw posterior samples from the converged geoVI approximation.

        Parameters
        ----------
        method : str
            "jit" (default) — JIT-compiled CG solve, ~0.2ms/sample.
            "blackjax" — BlackJAX NUTS (independent MCMC, not geoVI).
            "nifty" — NIFTy draw_linear_residual (slow, ~540ms/sample).
        """
        if method == "jit":
            return self._draw_jit_samples(
                pos_dict, key, n_samples, existing_samples, verbose=verbose
            )
        if method == "blackjax":
            try:
                return self._draw_blackjax_samples(
                    likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
                )
            except ImportError:
                if verbose:
                    print("  blackjax not installed, falling back to JIT sampling")
                return self._draw_jit_samples(
                    pos_dict, key, n_samples, existing_samples, verbose=verbose
                )
        return self._draw_nifty_samples(
            likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
        )

    def _build_jit_engine(self, pos_dict):
        """Build JIT-compiled inference engine: optimizer + posterior sampler.

        Returns a dict with compiled functions for the full EVI pipeline.
        All functions operate on flat arrays and use jax.lax.while_loop
        for zero Python overhead.

        The geoVI path uses NIFTy's actual implementations of CG,
        Newton-CG, sample drawing, and nonlinear curving — imported
        directly and called within the JIT boundary. This ensures
        mathematical equivalence with ``jft.optimize_kl``.
        """
        from tengri.core.noise import (
            compute_std_inv,
            get_noise_dof,
            has_noise_model,
            uses_student_t,
            variable_noise_hamiltonian,
            variable_noise_metric_vec,
        )

        # Import NIFTy for the exact geoVI path
        try:
            from nifty8.re.evi import Samples as NiftySamples
            from nifty8.re.optimize_kl import OptimizeVI

            _has_nifty = True
        except ImportError:
            _has_nifty = False

        model = self.model
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        stochastic = self.spec.stochastic
        # data/noise are NO LONGER captured here as local variables.
        # Instead they are passed at call-time via the ``data_args`` dict
        # so that the compiled engine can be reused across galaxies.
        use_variable_noise = has_noise_model(self.spec)
        noise_dof = get_noise_dof(self.spec) if uses_student_t(self.spec) else None

        # --- Signal response (physics only) ---
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
            raise ValueError(f"Unknown data_type: {data_type}")

        # --- Signal + noise response for variable noise ---
        if use_variable_noise:

            def signal_noise_response(primals, data_args):
                """Return (predicted, std_inv) tuple for variable noise metric."""
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(primals[name], lo, hi)
                for name, val in fixed_values.items():
                    params[name] = val
                if stochastic and "psd_xi" in primals:
                    params["psd_xi"] = primals["psd_xi"]
                if data_type == "photometry":
                    predicted = model.predict_photometry(params)
                elif data_type == "spectroscopy":
                    predicted = model.predict_spectrum(params, model._wave_obs)
                elif data_type == "joint":
                    p = model.predict_photometry(params)
                    s = model.predict_spectrum(params, model._wave_obs)
                    predicted = jnp.concatenate([p, s])
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
                f_cal = params.get("noise_frac_cal", 0.0)
                noise = data_args["noise"]
                std_inv = compute_std_inv(noise, predicted, f_cal)
                return predicted, std_inv

        # --- Flatten/unflatten (static shapes) ---
        param_keys = sorted(pos_dict.keys())
        slices = []
        idx = 0
        for k in param_keys:
            arr = jnp.atleast_1d(pos_dict[k]).ravel()
            shape = jnp.atleast_1d(pos_dict[k]).shape
            slices.append((idx, idx + arr.shape[0], shape))
            idx += arr.shape[0]
        d_total = idx
        n_data = len(self.data)  # static shape — same for all galaxies with same obs

        def flatten(d):
            return jnp.concatenate([jnp.atleast_1d(d[k]).ravel() for k in param_keys])

        def unflatten(x):
            d = {}
            for i_k, k in enumerate(param_keys):
                start, end, shape = slices[i_k]
                val = jax.lax.dynamic_slice(x, (start,), (end - start,)).reshape(shape)
                if shape == (1,):
                    val = val[0]
                d[k] = val
            return d

        # --- Core primitives ---
        _eps = 6.0 * jnp.finfo(jnp.float64).eps

        if use_variable_noise:

            def metric_vec(xi, v, data_args):
                """GGN metric for VariableCovarianceGaussian likelihood."""
                data = data_args["data"]

                def _snr(primals):
                    return signal_noise_response(primals, data_args)

                return variable_noise_metric_vec(xi, v, _snr, data, unflatten, flatten)

            def hamiltonian(xi, data_args):
                """E_lh + 0.5 ||xi||^2 with variable noise (includes logdet)."""
                data = data_args["data"]
                noise = data_args["noise"]
                pred = signal_response(unflatten(xi))
                primals = unflatten(xi)
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(primals[name], lo, hi)
                for name, val in fixed_values.items():
                    params[name] = val
                f_cal = params.get("noise_frac_cal", 0.0)
                return variable_noise_hamiltonian(
                    data, noise, pred, f_cal, dof=noise_dof
                ) + 0.5 * jnp.sum(xi**2)

        else:

            def metric_vec(xi, v, data_args):
                """M(xi) @ v = J^T N^{-1} J v + v."""
                noise_inv = data_args["noise_inv"]
                xi_d, v_d = unflatten(xi), unflatten(v)
                _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
                _, vjp_fn = jax.vjp(signal_response, xi_d)
                return flatten(vjp_fn(noise_inv * Jv)[0]) + v

            def hamiltonian(xi, data_args):
                """H(xi) = 0.5 chi2 + 0.5 ||xi||^2."""
                data = data_args["data"]
                noise = data_args["noise"]
                pred = signal_response(unflatten(xi))
                chi2 = jnp.sum(((data - pred) / noise) ** 2)
                return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

        def H_vg(xi, data_args):
            """Hamiltonian value and gradient w.r.t. xi only."""
            return jax.value_and_grad(lambda x: hamiltonian(x, data_args))(xi)

        _tiny = 6.0 * jnp.finfo(jnp.float64).tiny
        _n_reset = 20

        def cg_solve(mat_fn, b, x0, maxiter=30, miniter=6, absdelta=0.0, resnorm=0.0):
            """CG solve: mat_fn(x) = b.

            Exact port of NIFTy ``_static_cg`` (conjugate_gradient.py:217-388)
            for flat arrays.  Residual-norm (L2) is the primary convergence
            criterion; energy-based absdelta is secondary.  Negative curvature
            on the first CG iteration triggers a steepest-descent fallback.
            """
            r = mat_fn(x0) - b
            d = r
            gamma = jnp.dot(r, r)
            energy = jnp.dot((r - b) / 2, x0)
            init_info = jnp.where(gamma == 0.0, jnp.int32(0), jnp.int32(-2))
            init = (x0, r, d, gamma, energy, init_info, jnp.int32(0))

            def cond(s):
                return s[5] < -1

            def body(s):
                pos, r, d, prev_gamma, prev_energy, info, i = s
                i = i + 1

                q = mat_fn(d)
                curv = jnp.dot(d, q)
                alpha = prev_gamma / curv

                # Negative / zero curvature (NIFTy cg:278-286)
                info = jnp.where(curv <= 0.0, jnp.int32(0), info)
                alpha = jnp.where(curv <= 0.0, 0.0, alpha)
                pos = pos - alpha * d
                # First iter + negative curvature: steepest-descent fallback
                pos = jnp.where(
                    (curv < 0.0) & (i <= 1),
                    prev_energy / (-curv) * (-b),
                    pos,
                )

                # Periodic residual reset (NIFTy cg:287-291)
                r_reset = mat_fn(pos) - b
                r_step = r - q * alpha
                r = jnp.where((i % _n_reset == 0) & (info < -1), r_reset, r_step)

                gamma = jnp.dot(r, r)

                # Tiny gamma (NIFTy cg:295)
                info = jnp.where(
                    (gamma >= 0.0) & (gamma <= _tiny) & (info != -1),
                    jnp.int32(0),
                    info,
                )

                # Residual norm -- PRIMARY (NIFTy cg:296-298, norm_ord=2)
                r_norm = jnp.sqrt(gamma)
                info = jnp.where(
                    (resnorm > 0.0) & (r_norm < resnorm) & (i >= miniter) & (info != -1),
                    jnp.int32(0),
                    info,
                )

                # Energy -- SECONDARY (NIFTy cg:301-313)
                energy = jnp.dot((r - b) / 2, pos)
                energy_diff = prev_energy - energy
                neg_energy_eps = -_eps * jnp.abs(energy)
                info = jnp.where(
                    energy_diff < neg_energy_eps,
                    jnp.where(info < -1, i, info),
                    info,
                )
                info = jnp.where(
                    (absdelta > 0.0) & (energy_diff < absdelta) & (i >= miniter) & (info != -1),
                    jnp.int32(0),
                    info,
                )

                # Maxiter (NIFTy cg:314)
                info = jnp.where((i >= maxiter) & (info != -1), i, info)

                # Update search direction (NIFTy cg:316)
                d = d * jnp.maximum(0.0, gamma / prev_gamma) + r

                return (pos, r, d, gamma, energy, info, i)

            return jax.lax.while_loop(cond, body, init)[0]

        # --- Posterior sampler: draw linear residuals ---
        def draw_residuals(pos_f, subkeys, data_args):
            """Draw n linear residual samples (vmapped)."""
            sqrt_ni = data_args["sqrt_noise_inv"]
            n_d = n_data  # static, captured at engine-build time

            def draw_one(subkey):
                k1, k2 = jax.random.split(subkey)
                eta_pr = jax.random.normal(k1, shape=(d_total,))
                eta_lh = jax.random.normal(k2, shape=(n_d,))
                _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
                jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
                return cg_solve(
                    lambda v: metric_vec(pos_f, v, data_args),
                    jt + eta_pr,
                    eta_pr,
                    maxiter=30,
                    miniter=6,
                    absdelta=1e-4,
                )

            return jax.vmap(draw_one)(subkeys)

        def _draw_batch_fn(pos_f, k, data_args):
            return draw_residuals(pos_f, k, data_args)

        draw_batch = jax.jit(jax.vmap(_draw_batch_fn, in_axes=(None, 0, None)))

        # --- geoVI: nonlinear coordinate transform primitives ---

        def transformation_flat(pos_f, data_args):
            """t(x) = sqrt(N^{-1}) @ f(x). Maps to whitened data-space."""
            sqrt_ni = data_args["sqrt_noise_inv"]
            return sqrt_ni * signal_response(unflatten(pos_f))

        def left_sqrt_metric_flat(pos_f, v_data, data_args):
            """L^T(pos) @ v = J^T(pos) @ sqrt(N^{-1}) @ v.

            Maps whitened data-space vector to parameter-space.
            Matches NIFTy's ``likelihood.left_sqrt_metric(pos, v)``
            for the Gaussian case.
            """
            sqrt_ni = data_args["sqrt_noise_inv"]
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            return flatten(vjp_fn(sqrt_ni * v_data)[0])

        def right_sqrt_metric_flat(pos_f, v_param, data_args):
            """L(pos) @ v = sqrt(N^{-1}) @ J(pos) @ v.

            Maps parameter-space vector to whitened data-space.
            Matches NIFTy's ``likelihood.right_sqrt_metric(pos, v)``
            for the Gaussian case.
            """
            sqrt_ni = data_args["sqrt_noise_inv"]
            _, Jv = jax.jvp(signal_response, (unflatten(pos_f),), (unflatten(v_param),))
            return sqrt_ni * Jv

        def draw_metric_sample(pos_f, subkey, data_args):
            """Draw one sample with covariance M = J^T N^{-1} J + I.

            This is ``draw_linear_residual(..., from_inverse=False)``
            in NIFTy. The metric sample is NOT CG-inverted.
            """
            sqrt_ni = data_args["sqrt_noise_inv"]
            n_d = n_data  # static, captured at engine-build time
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=(d_total,))
            eta_lh = jax.random.normal(k2, shape=(n_d,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
            return jt + eta_pr

        def _newton_cg_flat(
            fun_and_grad,
            hessp,
            x0,
            custom_gradnorm=None,
            maxiter=10,
            miniter=0,
            xtol=1e-5,
            energy_reduction_factor=0.1,
        ):
            """Newton-CG with successive-halving line search.

            Exact port of NIFTy ``_static_newton_cg`` (optimize.py:285-449)
            for flat arrays.  Includes adaptive CG tolerance, steepest-descent
            reset after 5 line-search halvings, and custom gradient norm.
            """
            ncg_xtol = xtol * d_total  # NIFTy: xtol * size(x0)

            def gradnorm(v):
                if custom_gradnorm is not None:
                    return custom_gradnorm(v)
                return jnp.sum(jnp.abs(v))  # L1 norm (NIFTy default)

            energy, g = fun_and_grad(x0)
            init_state = (
                x0,
                energy,
                jnp.array(jnp.inf),
                g,
                jnp.where(maxiter == 0, jnp.int32(0), jnp.int32(-2)),
                jnp.int32(0),
            )

            def ncg_cond(state):
                return state[4] < -1

            def ncg_body(state):
                pos, energy, old_energy, g, status, i = state
                i = i + 1

                # Adaptive CG tolerance (NIFTy optimize.py:351-358)
                cg_abd_fallback = jnp.array(0.0, dtype=energy.dtype)
                cg_absdelta = jnp.where(
                    ~jnp.isinf(old_energy),
                    energy_reduction_factor * (old_energy - energy),
                    cg_abd_fallback,
                )
                cg_absdelta = jnp.array(cg_absdelta, dtype=energy.dtype)

                # CG resnorm (NIFTy optimize.py:359-360, norm_ord=1)
                mag_g = jnp.sum(jnp.abs(g))
                cg_resnorm = jnp.minimum(0.5, jnp.sqrt(mag_g)) * mag_g

                # CG solve (NIFTy: norm_ord=1, _raise_nonposdef=False)
                nat_g = cg_solve(
                    lambda v: hessp(pos, v),
                    g,
                    jnp.zeros_like(pos),
                    maxiter=min(200, 20 * d_total),
                    miniter=min(6, min(200, 20 * d_total)),
                    absdelta=cg_absdelta,
                    resnorm=cg_resnorm,
                )

                # Line search: successive halving (NIFTy optimize.py:452-523)
                # State: (status, iter, new_pos, new_energy, new_g,
                #         dd, grad_scaling, reset, nhev)
                ls_init = (
                    jnp.int32(-2),
                    jnp.int32(0),
                    pos,
                    jnp.array(jnp.inf),
                    g,
                    nat_g,
                    1.0,
                    jnp.bool_(False),
                    jnp.int32(0),
                )

                def ls_cond(ls):
                    return ls[0] < -1

                def ls_body(ls):
                    (
                        ls_st,
                        ls_i,
                        _np,
                        _ne,
                        _ng,
                        dd,
                        gs,
                        reset,
                        nhev,
                    ) = ls
                    new_pos = pos - gs * dd
                    new_e, new_g = fun_and_grad(new_pos)
                    ls_st = jnp.where(new_e <= energy, jnp.int32(0), ls_st)
                    gs = jnp.where(ls_st < -1, gs / 2.0, gs)
                    # Steepest descent reset at iteration 5
                    do_reset = (ls_i == 5) & (ls_st < -1)
                    reset = jnp.where(do_reset, jnp.bool_(True), reset)
                    gs = jnp.where(do_reset, 1.0, gs)
                    gam = jnp.dot(g, g)
                    curv = jnp.dot(g, hessp(pos, g))
                    sd_dd = gam / curv * g
                    dd = jnp.where(do_reset, sd_dd, dd)
                    nhev = nhev + do_reset.astype(jnp.int32)
                    # Abort after 8 iterations
                    do_abort = (ls_i == 8) & (ls_st < -1)
                    ls_st = jnp.where(do_abort, jnp.int32(-1), ls_st)
                    return (
                        ls_st,
                        ls_i + 1,
                        new_pos,
                        new_e,
                        new_g,
                        dd,
                        gs,
                        reset,
                        nhev,
                    )

                ls_result = jax.lax.while_loop(ls_cond, ls_body, ls_init)
                (
                    ls_status,
                    ls_iter,
                    new_pos,
                    new_energy,
                    new_g,
                    dd,
                    gs,
                    _reset,
                    _nhev,
                ) = ls_result

                status = jnp.where(ls_status != 0, jnp.int32(-1), status)

                # Update only if line search succeeded (NIFTy opt:381-385)
                success = status < -1
                old_energy = jnp.where(success, energy, old_energy)
                energy_out = jnp.where(success, new_energy, energy)
                energy_diff = jnp.where(success, old_energy - energy_out, 0.0)
                pos_out = jnp.where(success, new_pos, pos)
                g_out = jnp.where(success, new_g, g)
                gs_out = jnp.where(success, gs, 0.0)

                descent_norm = gs_out * gradnorm(dd)

                # absdelta convergence (NIFTy optimize.py:407-414)
                min_cond = (ls_iter < 2) & (i > miniter)
                status = jnp.where(
                    (energy_diff >= 0.0) & (energy_diff < 1e-3) & min_cond & (status != -1),
                    jnp.int32(0),
                    status,
                )
                # xtol convergence (NIFTy optimize.py:415-417)
                status = jnp.where(
                    (descent_norm <= ncg_xtol) & (i > miniter) & (status != -1),
                    jnp.int32(0),
                    status,
                )
                # maxiter (NIFTy optimize.py:418)
                status = jnp.where((i == maxiter) & (status < -1), i, status)

                return (pos_out, energy_out, old_energy, g_out, status, i)

            result = jax.lax.while_loop(ncg_cond, ncg_body, init_state)
            return result[0], result[1]

        def curve_residual(m, r_linear, metric_key, sign, data_args):
            """Nonlinearly update a linear residual to a geoVI curved residual.

            Exact port of NIFTy ``nonlinearly_update_residual``
            (evi.py:136-217) using ``_newton_cg_flat`` for the inner
            Newton-CG optimization.

            Parameters
            ----------
            m : flat array, expansion point
            r_linear : flat array, linear residual (covariance M^{-1})
            metric_key : PRNG key (same as used for draw_residuals)
            sign : +1.0 or -1.0 (for mirrored samples)
            data_args : dict, data-dependent arguments

            Returns
            -------
            flat array : curved residual (x_opt - m)
            """
            x0 = m + r_linear
            ms = sign * draw_metric_sample(m, metric_key, data_args)
            trafo_at_m = transformation_flat(m, data_args)

            def phi_vg(x):
                trafo_x = transformation_flat(x, data_args)
                delta_trafo = trafo_x - trafo_at_m
                g_x = (x - m) + left_sqrt_metric_flat(m, delta_trafo, data_args)
                r = ms - g_x
                val = 0.5 * jnp.dot(r, r)
                ngrad = r + left_sqrt_metric_flat(
                    x, right_sqrt_metric_flat(m, r, data_args), data_args
                )
                return val, -ngrad

            def phi_metric(x, v):
                tm = (
                    left_sqrt_metric_flat(m, right_sqrt_metric_flat(x, v, data_args), data_args)
                    + v
                )
                return (
                    left_sqrt_metric_flat(x, right_sqrt_metric_flat(m, tm, data_args), data_args)
                    + tm
                )

            # sampnorm (evi.py:178-181)
            def sampnorm(natgrad):
                fpp = right_sqrt_metric_flat(m, natgrad, data_args)
                return jnp.sqrt(jnp.dot(natgrad, natgrad) + jnp.dot(fpp, fpp))

            x_opt, _ = _newton_cg_flat(
                phi_vg,
                phi_metric,
                x0,
                custom_gradnorm=sampnorm,
                maxiter=3,
                miniter=0,
                xtol=1e-3,
                energy_reduction_factor=0.1,
            )
            return x_opt - m

        def draw_nonlinear_residuals(m, subkeys, data_args):
            """Draw geoVI nonlinear residuals: linear draw + curving + mirror.

            Returns (2*n_samples, D) array: curved residuals with mirrored pairs.
            Matches NIFTy's ``nonlinear_resample`` sample mode.
            """
            # First draw linear residuals
            linear_residuals = draw_residuals(m, subkeys, data_args)

            # Curve each residual and its mirror
            def curve_pair(r, subkey):
                r_pos = curve_residual(m, r, subkey, sign=1.0, data_args=data_args)
                r_neg = curve_residual(m, -r, subkey, sign=-1.0, data_args=data_args)
                return r_pos, r_neg

            pos_curved, neg_curved = jax.vmap(curve_pair)(linear_residuals, subkeys)
            return jnp.concatenate([pos_curved, neg_curved], axis=0)

        def update_nonlinear_residuals(m, prev_residuals, subkeys, data_args):
            """Re-curve existing residuals at updated expansion point.

            Takes 2*n_samples residuals (first half positive, second half
            negative mirrors) and re-applies geoVI curving at the new m.
            Matches NIFTy's ``nonlinear_update`` sample mode.
            """
            n_half = prev_residuals.shape[0] // 2
            r_pos = prev_residuals[:n_half]
            r_neg = prev_residuals[n_half:]

            def recurve_pair(r_p, r_n, subkey):
                new_p = curve_residual(m, r_p, subkey, sign=1.0, data_args=data_args)
                new_n = curve_residual(m, r_n, subkey, sign=-1.0, data_args=data_args)
                return new_p, new_n

            new_pos, new_neg = jax.vmap(recurve_pair)(r_pos, r_neg, subkeys)
            return jnp.concatenate([new_pos, new_neg], axis=0)

        # --- EVI optimizer: fully JIT'd optimize_kl ---
        def kl_vg(m, residuals, data_args):
            """KL value and gradient averaged over samples."""

            def single_vg(r):
                return H_vg(m + r, data_args)

            vals, grads = jax.vmap(single_vg)(residuals)
            return jnp.mean(vals), jnp.mean(grads, axis=0)

        def kl_metric(m, residuals, v, data_args):
            """KL metric-vector product averaged over samples."""

            def single_met(r):
                return metric_vec(m + r, v, data_args)

            return jnp.mean(jax.vmap(single_met)(residuals), axis=0)

        def evi_step(m, subkey, n_samples, data_args):
            """One EVI iteration: draw samples + Newton-CG KL minimize.

            Returns (m_new, kl_value).
            """
            # Draw linear residual samples + mirror
            sample_keys = jax.random.split(subkey, n_samples)
            residuals = draw_residuals(m, sample_keys, data_args)
            residuals = jnp.concatenate([residuals, -residuals], axis=0)

            # Newton-CG KL minimization (same path as evi_step_full)
            def _evi_kl_vg(m_cur):
                return kl_vg(m_cur, residuals, data_args)

            def _evi_kl_hessp(m_cur, v):
                return kl_metric(m_cur, residuals, v, data_args)

            m_opt, kl_val = _newton_cg_flat(
                _evi_kl_vg,
                _evi_kl_hessp,
                m,
                maxiter=10,
                miniter=0,
                xtol=1e-5,
                energy_reduction_factor=0.1,
            )
            return m_opt, kl_val

        def run_evi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol):
            """Run EVI with automatic convergence detection.

            Stops early when the relative KL change between iterations
            drops below ``kl_rtol``. Uses ``jax.lax.while_loop`` so
            the iteration count is dynamic.
            """
            keys = jax.random.split(key, n_iterations)

            # State: (m, prev_kl, iteration, converged)
            def cond_fn(state):
                _m, _prev_kl, i, converged = state
                return (~converged) & (i < n_iterations)

            def body_fn(state):
                m, prev_kl, i, converged = state
                subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
                m_new, kl_val = evi_step(m, subkey, n_samples, data_args)
                # Relative KL change
                rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
                # Converge if relative change < rtol and at least 5 iterations done
                converged = (rel_change < kl_rtol) & (i >= 5)
                return (m_new, kl_val, i + 1, converged)

            # First iteration (no convergence check)
            m0, kl0 = evi_step(init_pos, keys[0], n_samples, data_args)
            init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))

            m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
            return m_final, n_iters

        # --- geoVI optimizer: per-mode functions (no lax.switch) ---
        #
        # Each sample mode gets its own evi_step function so that JAX
        # compiles ONLY the code path actually used.  This avoids the
        # 56s compilation cost of tracing all three branches via
        # ``jax.lax.switch``.
        #
        # ``sample_mode`` is a **static** string argument: JAX caches
        # a separate compiled version for each mode.
        SAMPLE_LINEAR = jnp.int32(0)
        SAMPLE_NONLINEAR_RESAMPLE = jnp.int32(1)
        SAMPLE_NONLINEAR_UPDATE = jnp.int32(2)

        def _kl_minimize(m, residuals, constants_mask, data_args):
            """Newton-CG KL minimization with constants mask."""

            def _masked_kl_vg(m_cur, res):
                val, grad = kl_vg(m_cur, res, data_args)
                grad = jnp.where(constants_mask, 0.0, grad)
                return val, grad

            def _masked_kl_metric(m_cur, res, v):
                v_masked = jnp.where(constants_mask, 0.0, v)
                mv = kl_metric(m_cur, res, v_masked, data_args)
                return jnp.where(constants_mask, 0.0, mv)

            def _fun_and_grad(m_cur):
                return _masked_kl_vg(m_cur, residuals)

            def _hessp(m_cur, v):
                return _masked_kl_metric(m_cur, residuals, v)

            return _newton_cg_flat(
                _fun_and_grad,
                _hessp,
                m,
                maxiter=10,
                miniter=0,
                xtol=1e-5,
                energy_reduction_factor=0.1,
            )

        _RESAMPLE_EVERY = 5  # refresh stale samples every N iterations

        def evi_step_full(
            m,
            subkey,
            n_samples,
            sample_mode,
            prev_residuals,
            prev_keys,
            constants_mask,
            pe_mask,
            data_args,
            iteration=0,
        ):
            """One geoVI iteration — ``sample_mode`` must be a static string.

            When used inside ``run_evi_geovi`` (which marks ``sample_mode``
            as static), JAX compiles a separate version per mode.  The
            unused branches are never traced, so ``"linear"`` compiles in
            ~0.03s while ``"nonlinear_resample"`` compiles in ~56s.

            Parameters
            ----------
            sample_mode : str  (STATIC — triggers recompilation per value)
                ``"linear_resample"`` — fresh MGVI samples (standard MGVI)
                ``"linear_sample"`` — reuse keys from prev iter (deterministic MGVI)
                ``"nonlinear_resample"`` — fresh geoVI samples (standard geoVI)
                ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
                ``"nonlinear_update"`` — re-curve existing residuals at new m
            data_args : dict
                Data-dependent arguments (data, noise, noise_inv, etc.).

            Returns
            -------
            m_new, kl_value, new_residuals, used_keys
            """
            # Key handling: _resample = fresh keys, _sample = reuse prev keys
            if sample_mode.endswith("_resample") or sample_mode == "geovi":
                sample_keys = jax.random.split(subkey, n_samples)
            elif sample_mode == "nonlinear_update":
                sample_keys = prev_keys
            else:  # _sample modes: reuse
                sample_keys = prev_keys

            # Python if — only the used branch is traced by JAX
            if sample_mode == "geovi":
                # Optimal schedule: resample at iter 0 and every
                # _RESAMPLE_EVERY, nonlinear_update in between.
                # Uses jax.lax.cond (traces both branches, executes one).
                do_resample = (iteration == 0) | (iteration % _RESAMPLE_EVERY == 0)

                def _do_resample(_):
                    return draw_nonlinear_residuals(m, sample_keys, data_args)

                def _do_update(_):
                    return update_nonlinear_residuals(m, prev_residuals, prev_keys, data_args)

                residuals = jax.lax.cond(do_resample, _do_resample, _do_update, None)
            elif sample_mode in ("nonlinear_resample", "nonlinear_sample"):
                residuals = draw_nonlinear_residuals(m, sample_keys, data_args)
            elif sample_mode == "nonlinear_update":
                residuals = update_nonlinear_residuals(m, prev_residuals, sample_keys, data_args)
            else:  # linear_resample, linear_sample
                res = draw_residuals(m, sample_keys, data_args)
                residuals = jnp.concatenate([res, -res], axis=0)

            # Apply point estimates mask
            residuals = residuals * pe_mask[None, :]

            # KL minimization
            m_opt, kl_val = _kl_minimize(m, residuals, constants_mask, data_args)
            return m_opt, kl_val, residuals, sample_keys

        def run_evi_geovi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol, sample_mode):
            """Run geoVI with automatic convergence detection.

            ``sample_mode`` is a **static** string — JAX compiles a
            separate XLA program per mode.  All 5 NIFTy modes supported:

            - ``"linear_resample"`` — fresh MGVI samples each iteration
            - ``"linear_sample"`` — reuse PRNG keys (deterministic MGVI)
            - ``"nonlinear_resample"`` — fresh geoVI samples
            - ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
            - ``"nonlinear_update"`` — re-curve existing residuals at new m
            """
            keys = jax.random.split(key, n_iterations)
            dummy_residuals = jnp.zeros((2 * n_samples, d_total))
            dummy_keys = jax.random.split(keys[0], n_samples)
            no_constants = jnp.zeros(d_total, dtype=bool)
            all_sampled = jnp.ones(d_total)

            # State: (m, prev_kl, residuals, prev_keys, iter, converged)
            def cond_fn(state):
                _m, _prev_kl, _res, _pk, i, converged = state
                return (~converged) & (i < n_iterations)

            def body_fn(state):
                m, prev_kl, prev_res, prev_k, i, converged = state
                subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
                m_new, kl_val, new_res, new_k = evi_step_full(
                    m,
                    subkey,
                    n_samples,
                    sample_mode,
                    prev_res,
                    prev_k,
                    no_constants,
                    all_sampled,
                    data_args,
                    iteration=i,
                )
                rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
                converged = (rel_change < kl_rtol) & (i >= 5)
                return (m_new, kl_val, new_res, new_k, i + 1, converged)

            # First iteration (always resample to establish initial keys)
            m0, kl0, res0, keys0 = evi_step_full(
                init_pos,
                keys[0],
                n_samples,
                sample_mode,
                dummy_residuals,
                dummy_keys,
                no_constants,
                all_sampled,
                data_args,
            )
            init_state = (
                m0,
                kl0,
                res0,
                keys0,
                jnp.int32(1),
                jnp.bool_(False),
            )

            result = jax.lax.while_loop(cond_fn, body_fn, init_state)
            return result[0], result[4]  # m_final, n_iters

        # --- Parameter range mapping for mask construction ---
        param_ranges = {}
        for i_k, k in enumerate(param_keys):
            start, end, _shape = slices[i_k]
            param_ranges[k] = (start, end)

        def make_mask(param_names):
            """Create boolean mask: True for named params, False otherwise."""
            mask = jnp.zeros(d_total, dtype=bool)
            for name in param_names:
                if name in param_ranges:
                    start, end = param_ranges[name]
                    mask = mask.at[start:end].set(True)
            return mask

        def make_pe_mask(param_names):
            """Create point-estimate mask: 0.0 for PE params, 1.0 for sampled."""
            mask = jnp.ones(d_total)
            for name in param_names:
                if name in param_ranges:
                    start, end = param_ranges[name]
                    mask = mask.at[start:end].set(0.0)
            return mask

        # --- NIFTy-backed geoVI: exact NIFTy math, minimal Python overhead ---
        # Uses NIFTy's OptimizeVI.update directly (already JIT'd internally)
        # but skips logging, pickling, and callbacks for speed.
        nifty_likelihood = None
        nifty_opt_vi = None
        if _has_nifty:
            try:
                import nifty8.re as jft

                # Build the NIFTy likelihood (same as _run_nifty_vi)
                _nifty_domain = {}
                for name in self._free_names:
                    _nifty_domain[name] = jft.ShapeWithDtype(())
                if self.spec.stochastic:
                    _nifty_domain["psd_xi"] = jft.ShapeWithDtype((self.spec.n_grid,))
                _nifty_model = jft.Model(jax.jit(signal_response), domain=_nifty_domain)
                if not use_variable_noise:
                    nifty_likelihood = jft.Gaussian(self.data, self._data_args["noise_inv"]).amend(
                        _nifty_model
                    )
                # Build OptimizeVI with vmap and JIT
                # (this pre-compiles all the internal functions)
                nifty_opt_vi = OptimizeVI(
                    nifty_likelihood,
                    n_total_iterations=50,  # max, actual controlled by caller
                    kl_jit=True,
                    residual_jit=True,
                    kl_map=jax.vmap,
                    residual_map=jax.vmap,
                )
            except Exception:
                _has_nifty = False
                nifty_likelihood = None
                nifty_opt_vi = None

        def run_nifty_jit(
            init_pos_flat,
            key,
            n_iterations,
            n_samples,
            sample_mode_str,
            draw_linear_kwargs,
            nonlinearly_update_kwargs,
            kl_kwargs,
        ):
            """Run NIFTy's exact optimize_kl with minimal Python overhead.

            Uses NIFTy's OptimizeVI.update (already JIT'd) in a tight
            Python loop — no logging, no pickling, no callbacks.
            Exact same math as ``jft.optimize_kl``.

            Returns (converged_flat, n_iters).
            """
            import nifty8.re as jft

            pos_dict = unflatten(init_pos_flat)
            samples = NiftySamples(pos=jft.Vector(pos_dict), samples=None, keys=None)
            state = nifty_opt_vi.init_state(
                key,
                n_samples=n_samples,
                sample_mode=sample_mode_str,
                draw_linear_kwargs=draw_linear_kwargs,
                nonlinearly_update_kwargs=nonlinearly_update_kwargs,
                kl_kwargs=kl_kwargs,
            )
            for _i in range(n_iterations):
                samples, state = nifty_opt_vi.update(samples, state)
            converged = samples.pos
            pos_d = converged.tree if hasattr(converged, "tree") else dict(converged)
            return flatten(pos_d), samples

        # Compile the core functions with dummy data.
        # data_args is passed as argument (not closed over) to all JIT'd
        # functions so the same compiled XLA program can be reused with
        # different data of the same shape.
        dummy_pos = flatten(pos_dict)
        dummy_keys = jax.random.split(jax.random.PRNGKey(0), 2)
        dummy_data_args = self._data_args

        # Pre-compile posterior sampler
        draw_samples_jit = jax.jit(draw_residuals)
        _ = draw_samples_jit(dummy_pos, dummy_keys, dummy_data_args)

        # Pre-compile native optimizer (for n_iterations=10, n_samples=3)
        run_evi_jit = jax.jit(run_evi, static_argnames=("n_iterations", "n_samples"))
        _ = run_evi_jit(
            dummy_pos,
            jax.random.PRNGKey(0),
            dummy_data_args,
            n_iterations=2,
            n_samples=2,
            kl_rtol=1e-2,
        )

        # Pre-compile native geoVI optimizer.
        # sample_mode is STATIC: JAX compiles a separate XLA program per mode.
        # "linear" compiles in ~0.03s, "nonlinear_*" in ~56s (one-time cost).
        run_evi_geovi_jit = jax.jit(
            run_evi_geovi,
            static_argnames=("n_iterations", "n_samples", "sample_mode"),
        )

        return {
            "run_evi": run_evi_jit,
            "run_evi_geovi": run_evi_geovi_jit,
            "run_nifty_jit": run_nifty_jit if _has_nifty else None,
            "nifty_likelihood": nifty_likelihood,
            "draw_samples": draw_samples_jit,
            "draw_nonlinear_samples": jax.jit(draw_nonlinear_residuals),
            "draw_batch": draw_batch,
            "flatten": flatten,
            "unflatten": unflatten,
            "param_keys": param_keys,
            "param_ranges": param_ranges,
            "make_mask": make_mask,
            "make_pe_mask": make_pe_mask,
            "d_total": d_total,
            "SAMPLE_LINEAR": SAMPLE_LINEAR,
            "SAMPLE_NONLINEAR_RESAMPLE": SAMPLE_NONLINEAR_RESAMPLE,
            "SAMPLE_NONLINEAR_UPDATE": SAMPLE_NONLINEAR_UPDATE,
            "evi_step_full": evi_step_full,
            # geoVI-NUTS primitives (coordinate transform + metric)
            "transformation_flat": transformation_flat,
            "left_sqrt_metric_flat": left_sqrt_metric_flat,
            "right_sqrt_metric_flat": right_sqrt_metric_flat,
            "metric_vec": metric_vec,
            "cg_solve": cg_solve,
            "hamiltonian": hamiltonian,
        }

    def compile(
        self,
        *,
        n_iterations=15,
        n_samples=3,
        n_posterior_samples=200,
        modes=("linear_resample", "nonlinear_update"),
        verbose=True,
    ):
        """Pre-compile the JIT inference engine ahead of time.

        Triggers XLA compilation for all specified modes so that
        subsequent ``fitter.run()`` calls have zero compilation delay.
        Compiled programs are cached both in-memory (this session)
        and on disk (``/tmp/tengri_jax_cache``, survives restarts).

        Parameters
        ----------
        n_iterations : int
            Compile for this iteration count (recompilation if changed).
        n_samples : int
            Compile for this sample count (recompilation if changed).
        n_posterior_samples : int
            Compile posterior draw for this many samples.
        modes : tuple of str
            Which sample modes to pre-compile. Each mode compiles
            separately. Default covers MGVI + geoVI update (fastest).
            Add ``"nonlinear_resample"`` for full geoVI (~56s extra).
        verbose : bool
            Print compilation progress.

        Example
        -------
        >>> fitter = Fitter(model, data, noise)
        >>> fitter.compile()  # ~3s for default modes
        >>> fitter.compile(
        ...     modes=(  # ~60s for all modes
        ...         "linear_resample",
        ...         "nonlinear_update",
        ...         "nonlinear_resample",
        ...     )
        ... )
        >>> result = fitter.run("native_geovi")  # instant
        """
        dummy_pos = self._initialize_unbounded(jax.random.PRNGKey(0))
        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(dummy_pos)

        engine = self._jit_sampler
        flatten = engine["flatten"]
        pos_flat = flatten(dummy_pos)
        data_args = self._data_args

        if verbose:
            print(
                f"Compiling: n_iter={n_iterations}, n_samp={n_samples}, "
                f"n_post={n_posterior_samples}, modes={modes}"
            )

        # Pre-compile each optimization mode
        for mode in modes:
            if verbose:
                print(f"  Compiling {mode}...", end="", flush=True)
            t0 = time.time()
            engine["run_evi_geovi"](
                pos_flat,
                jax.random.PRNGKey(0),
                data_args,
                n_iterations=n_iterations,
                n_samples=n_samples,
                kl_rtol=0.0,
                sample_mode=mode,
            )
            if verbose:
                print(f" {time.time() - t0:.1f}s")

        # Pre-compile MGVI optimizer (old path, used by native_mgvi)
        if verbose:
            print("  Compiling MGVI (old path)...", end="", flush=True)
        t0 = time.time()
        engine["run_evi"](
            pos_flat,
            jax.random.PRNGKey(0),
            data_args,
            n_iterations=n_iterations,
            n_samples=n_samples,
            kl_rtol=1e-2,
        )
        if verbose:
            print(f" {time.time() - t0:.1f}s")

        # Pre-compile posterior draw
        if verbose:
            print(
                f"  Compiling posterior draw ({n_posterior_samples} samples)...",
                end="",
                flush=True,
            )
        t0 = time.time()
        draw_keys = jax.random.split(jax.random.PRNGKey(0), n_posterior_samples)
        engine["draw_samples"](pos_flat, draw_keys, data_args)
        if verbose:
            print(f" {time.time() - t0:.1f}s")

        if verbose:
            print("Compilation complete.")
        return self

    def _draw_jit_samples(self, pos_dict, key, n_samples, existing_samples, *, verbose=True):
        """Draw geoVI linear residual samples via JIT-compiled CG.

        Same math as NIFTy's draw_linear_residual but fully JIT-compiled:
        1. Draw z = J^T sqrt(N^{-1}) eta1 + eta2  (eta_i ~ N(0,I))
        2. Solve M @ residual = z via CG  (M = J^T N^{-1} J + I)
        3. Sample = pos + residual

        ~2000x faster than NIFTy's Python-loop CG.
        """
        if verbose:
            print(f"  Drawing {n_samples} posterior samples (JIT CG)...")

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        residuals_flat = engine["draw_samples"](pos_flat, draw_keys, self._data_args)

        for i in range(n_samples):
            res = unflatten(residuals_flat[i])
            combined = {k: pos_dict[k] + res[k] for k in pos_dict}
            existing_samples.append(combined)

        return existing_samples

    def _draw_nonlinear_jit_samples(
        self, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw geoVI nonlinear posterior samples via JIT engine.

        Unlike ``_draw_jit_samples`` (linear CG only), this applies
        the geoVI coordinate curving to each sample.  Produces
        samples from the nonlinear approximation, capturing
        banana-shaped degeneracies that the linear Gaussian misses.

        Uses ``draw_nonlinear_residuals`` from the JIT engine.
        """
        if verbose:
            print(f"  Drawing {n_samples} nonlinear posterior samples (JIT geoVI)...")

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        data_args = self._data_args

        # Draw in batches to avoid OOM for large n_samples
        batch_size = min(n_samples, 50)
        draw_keys = jax.random.split(key, n_samples)

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_keys = draw_keys[batch_start:batch_end]
            # draw_nonlinear_samples returns (2*n, D): first n positive, last n mirrors
            residuals_flat = engine["draw_nonlinear_samples"](pos_flat, batch_keys, data_args)
            n_batch = batch_end - batch_start
            # Use only the first n (positive) samples, not the mirrors
            for i in range(n_batch):
                res = unflatten(residuals_flat[i])
                combined = {k: pos_dict[k] + res[k] for k in pos_dict}
                existing_samples.append(combined)

        return existing_samples

    def _draw_blackjax_samples(
        self, likelihood, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw samples via BlackJAX NUTS (independent MCMC, not geoVI)."""
        import blackjax

        if verbose:
            print(f"  Drawing {n_samples} posterior samples via BlackJAX NUTS...")

        if likelihood is not None:

            @jax.jit
            def logdensity_fn(x):
                lh_val = likelihood(x)
                prior = 0.5 * sum(jnp.sum(v**2) for v in x.values())
                return -lh_val - prior

        else:
            # Build log-density from the loss function (used by _run_native_vi path)
            loss_fn = self._build_loss_fn()

            @jax.jit
            def logdensity_fn(x):
                return -loss_fn(x)

        warmup_key, sample_key = jax.random.split(key)
        n_warmup = min(200, n_samples)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (state, parameters), _ = warmup.run(warmup_key, pos_dict, num_steps=n_warmup)

        if verbose:
            print(f"  Warmup done ({n_warmup} steps). Sampling...")

        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(sample_key, n_samples)
        _, states = jax.lax.scan(one_step, state, keys)

        sample_positions = states.position
        for i in range(n_samples):
            sd = jax.tree.map(lambda x, _i=i: x[_i], sample_positions)
            existing_samples.append(sd)

        return existing_samples

    def _draw_nifty_samples(
        self, likelihood, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw samples via NIFTy's draw_linear_residual (slow, ~540ms/sample)."""
        import nifty8.re as jft

        if verbose:
            print(f"  Drawing {n_samples} posterior samples (NIFTy CG)...")

        converged_pos = jft.Vector(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        for sub_key in draw_keys:
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 30},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                existing_samples.append(combined)
            except Exception:
                break

        return existing_samples

    # -------------------------------------------------------------------
    # Fully JIT'd EVI optimizer
    # -------------------------------------------------------------------

    def _run_native_vi(
        self,
        *,
        key,
        init_from="auto",
        n_iterations=50,
        n_samples=3,
        n_posterior_samples=2000,
        kl_rtol=1e-2,
        n_seeds=5,
        sample_mode="linear",
        posterior_method="jit",
        parallel_seeds=None,
        verbose=True,
    ):
        """Native JIT-compiled geoVI/MGVI: ~500x faster than NIFTy's optimize_kl.

        Supports multiple sample modes:
        - ``"linear"`` (default): MGVI linear sampling (fastest).
        - ``"geovi"``: Full geoVI with nonlinear coordinate curving.
        - ``"nonlinear_update"``: geoVI with sample reuse (best convergence).

        The entire optimization loop (sample drawing + Newton-CG KL
        minimization) runs inside ``jax.lax.while_loop`` with zero
        Python overhead. Stops automatically when KL converges.

        Parameters
        ----------
        init_from : str, Posterior, or None
            ``"auto"`` (default): MAP for ``n_seeds=1``, random for
            ``n_seeds>1``. MAP gives better convergence for a single
            seed; random init is better for multi-seed because vmap
            needs diverse starting points to find the global mode.
            ``"map"``: quick MAP estimate as starting point for all seeds.
            ``"random"`` or ``None``: random init near prior midpoint.
            ``Posterior``: use a previous result as initialization.
        n_iterations : int
            Maximum KL iterations. Auto-stops when converged.
        n_samples : int
            Samples per iteration (doubled by mirror_samples).
        n_posterior_samples : int
            Posterior samples drawn after convergence.
        kl_rtol : float
            Relative KL tolerance for early stopping. Set to 0 to
            disable and run all ``n_iterations``.
        n_seeds : int
            Number of random seeds to run in parallel via ``jax.vmap``.
            The best result (lowest Hamiltonian) is returned. Multiple
            seeds catch bad initialization and multimodality.
        parallel_seeds : bool or None
            If ``None`` (default), auto-detect: ``True`` on GPU/TPU,
            ``False`` on CPU. On CPU, sequential is typically faster
            because early-converging seeds exit early, while vmap must
            run all seeds for the maximum iteration count.
            Set explicitly to override.
        verbose : bool
            Print progress.
        """
        import warnings

        from tengri.inference.posterior import Posterior

        # --- Parameter validation ---
        if n_samples > 12:
            warnings.warn(
                f"n_samples={n_samples} is unusually high. With mirror_samples "
                f"this gives {2 * n_samples} effective samples per iteration. "
                f"High sample counts reduce stochastic regularization and can "
                f"cause the Newton-CG optimizer to overshoot. "
                f"Recommended: n_samples=3 (Philipp Frank, private comm.).",
                UserWarning,
                stacklevel=2,
            )
        if n_iterations > 100 and kl_rtol <= 0:
            warnings.warn(
                f"n_iterations={n_iterations} with kl_rtol={kl_rtol} (no auto-stop). "
                f"Running many iterations without convergence detection can cause "
                f"divergence. Consider setting kl_rtol=1e-2 for automatic stopping.",
                UserWarning,
                stacklevel=2,
            )
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

        # Normalize init_from: None → "auto"
        if init_from is None:
            init_from = "auto"

        dummy_pos = self._initialize_unbounded(jax.random.PRNGKey(0))
        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(dummy_pos)

        engine = self._jit_sampler
        flatten = engine["flatten"]
        unflatten = engine["unflatten"]
        data_args = self._data_args

        n_total = len(self._free_names) + (self.spec.n_grid if self.spec.stochastic else 0)
        n_seeds = max(1, n_seeds)

        # --- Resolve init_from="auto" ---
        # "auto": MAP for 1 seed (best convergence), random for >1 seed
        # (diverse starts → better global mode search, required for vmap).
        if init_from == "auto":
            init_from = "map" if n_seeds == 1 else "random"
            if verbose and n_seeds > 1:
                print(
                    f"  init_from='auto' → 'random' (n_seeds={n_seeds}; "
                    f"random starts are better for multi-seed exploration)"
                )
            elif verbose:
                print("  init_from='auto' → 'map' (single seed; MAP warmstart)")

        # Warn about suboptimal combinations
        if init_from == "map" and n_seeds > 1:
            warnings.warn(
                f"init_from='map' with n_seeds={n_seeds}: MAP init gives all seeds "
                f"nearly identical starting points, defeating the purpose of multi-seed. "
                f"Consider init_from='random' for diverse exploration, or n_seeds=1 "
                f"for fast single-seed MAP-initialized convergence.",
                UserWarning,
                stacklevel=2,
            )

        # Auto-detect parallel_seeds based on backend
        if parallel_seeds is None:
            backend = jax.default_backend()
            parallel_seeds = backend in ("gpu", "tpu")
            if verbose and n_seeds > 1:
                if parallel_seeds:
                    print(f"  parallel_seeds=True (auto: {backend} backend)")
                else:
                    print(
                        f"  parallel_seeds=False (auto: {backend} backend; "
                        f"sequential is faster on CPU due to early stopping)"
                    )

        if verbose:
            seed_str = f", {n_seeds} seeds" if n_seeds > 1 else ""
            par_str = " (vmap)" if parallel_seeds and n_seeds > 1 else ""
            mode_labels = {
                "linear": "MGVI",
                "mgvi": "MGVI",
                "geovi": "geoVI",
                "nonlinear_resample": "geoVI",
                "nonlinear_update": "geoVI (update)",
            }
            mode_label = mode_labels.get(sample_mode, sample_mode)
            print(
                f"{mode_label} (JIT): {n_total} params, {len(self.data)} data points, "
                f"{n_iterations} iterations, {n_samples} samples/iter"
                f"{seed_str}{par_str}"
            )

        t0 = time.time()

        # --- Resolve sample_mode string ---
        _mode_str_map = {
            "linear": "linear_resample",
            "mgvi": "linear_resample",
            "geovi": "geovi",
            "linear_resample": "linear_resample",
            "linear_sample": "linear_sample",
            "nonlinear_resample": "nonlinear_resample",
            "nonlinear_sample": "nonlinear_sample",
            "nonlinear_update": "nonlinear_update",
        }
        mode_str = _mode_str_map.get(sample_mode, "linear_resample")
        _use_geovi = mode_str not in ("linear_resample", "linear_sample")

        # --- Build initial positions ---
        seed_keys = jax.random.split(key, n_seeds + 1)
        key = seed_keys[-1]

        map_result = None
        if init_from == "map":
            map_key, key = jax.random.split(key)
            map_result = self._run_map(key=map_key, n_steps=500, verbose=False)
            if verbose:
                print("  MAP warmstart done")

        init_flats = []
        for s in range(n_seeds):
            if map_result is not None:
                init_params = self._unbounded_from_posterior(map_result)
            elif isinstance(init_from, Posterior):
                init_params = self._unbounded_from_posterior(init_from)
            else:
                init_params = self._initialize_unbounded(seed_keys[s])
            init_flats.append(flatten(init_params))

        opt_keys = jnp.stack([jax.random.fold_in(seed_keys[s], 999) for s in range(n_seeds)])

        # --- Run optimization ---
        if parallel_seeds and n_seeds > 1:
            # === VMAP PATH: all seeds in parallel ===
            init_batch = jnp.stack(init_flats)  # (n_seeds, d_total)

            if _use_geovi:
                # vmap over (init_pos, key), static (n_iterations, n_samples, kl_rtol, mode)
                def _run_single_geovi(pos, k):
                    return engine["run_evi_geovi"](
                        pos,
                        k,
                        data_args,
                        n_iterations=n_iterations,
                        n_samples=n_samples,
                        kl_rtol=kl_rtol,
                        sample_mode=mode_str,
                    )

                vmapped_run = jax.vmap(_run_single_geovi)
            else:

                def _run_single_evi(pos, k):
                    return engine["run_evi"](
                        pos,
                        k,
                        data_args,
                        n_iterations=n_iterations,
                        n_samples=n_samples,
                        kl_rtol=kl_rtol,
                    )

                vmapped_run = jax.vmap(_run_single_evi)

            # Run all seeds in parallel
            all_converged, all_n_iters = vmapped_run(init_batch, opt_keys)
            # all_converged: (n_seeds, d_total), all_n_iters: (n_seeds,)

            # Batch Hamiltonian evaluation
            def _eval_hamiltonian(converged_flat):
                phys = self._to_physical(unflatten(converged_flat))
                if self.data_type == "photometry":
                    pred = self.model.predict_photometry(phys)
                elif self.data_type == "spectroscopy":
                    pred = self.model.predict_spectrum(phys)
                else:
                    pred = jnp.zeros_like(self.data)
                chi2 = jnp.sum(((self.data - pred) / self.noise) ** 2)
                prior = jnp.sum(converged_flat**2)
                return 0.5 * chi2 + 0.5 * prior

            seed_losses_arr = jax.vmap(_eval_hamiltonian)(all_converged)
            best_idx = jnp.argmin(seed_losses_arr)
            best_flat = all_converged[best_idx]
            best_iters = int(all_n_iters[best_idx])
            seed_losses = [float(seed_losses_arr[s]) for s in range(n_seeds)]

            if verbose and n_seeds > 1:
                for s in range(n_seeds):
                    marker = " ← best" if s == int(best_idx) else ""
                    print(
                        f"  Seed {s + 1}/{n_seeds}: H={seed_losses[s]:.1f}, "
                        f"{int(all_n_iters[s])} iters{marker}"
                    )

        else:
            # === SEQUENTIAL PATH: for loop (debugging / single seed) ===
            best_flat = None
            best_loss = jnp.inf
            best_iters = 0
            seed_losses = []

            for s in range(n_seeds):
                pos_flat = init_flats[s]
                opt_key = opt_keys[s]

                if _use_geovi:
                    converged_flat, n_iters = engine["run_evi_geovi"](
                        pos_flat,
                        opt_key,
                        data_args,
                        n_iterations=n_iterations,
                        n_samples=n_samples,
                        kl_rtol=kl_rtol,
                        sample_mode=mode_str,
                    )
                else:
                    converged_flat, n_iters = engine["run_evi"](
                        pos_flat,
                        opt_key,
                        data_args,
                        n_iterations=n_iterations,
                        n_samples=n_samples,
                        kl_rtol=kl_rtol,
                    )
                n_iters = int(n_iters)

                # Evaluate Hamiltonian to pick best seed
                phys = self._to_physical(unflatten(converged_flat))
                if self.data_type == "photometry":
                    pred = self.model.predict_photometry(phys)
                elif self.data_type == "spectroscopy":
                    pred = self.model.predict_spectrum(phys)
                else:
                    pred = jnp.zeros_like(self.data)
                chi2 = float(jnp.sum(((self.data - pred) / self.noise) ** 2))
                prior = float(jnp.sum(converged_flat**2))
                loss = 0.5 * chi2 + 0.5 * prior
                seed_losses.append(loss)

                if loss < best_loss:
                    best_flat = converged_flat
                    best_loss = loss
                    best_iters = n_iters

                if verbose and n_seeds > 1:
                    print(f"  Seed {s + 1}/{n_seeds}: H={loss:.1f}, {n_iters} iters")

        # --- Seed disagreement check ---
        if n_seeds > 1 and len(seed_losses) > 1:
            loss_std = float(jnp.std(jnp.array(seed_losses)))
            loss_mean = float(jnp.mean(jnp.array(seed_losses)))
            if loss_std > 0.1 * abs(loss_mean) and loss_mean != 0:
                warnings.warn(
                    f"Seeds disagree: H = {loss_mean:.1f} ± {loss_std:.1f} "
                    f"(CV={loss_std / abs(loss_mean):.0%}). "
                    f"This may indicate multimodality or poor convergence. "
                    f"Consider increasing n_iterations or inspecting the posterior.",
                    UserWarning,
                    stacklevel=2,
                )

        converged_flat = best_flat

        # --- Draw posterior samples ---
        key, draw_key = jax.random.split(key)
        all_sample_dicts = []
        converged_dict = unflatten(converged_flat)

        if n_posterior_samples > 0:
            if posterior_method == "blackjax":
                # NUTS posterior sampling from converged position
                all_sample_dicts = self._draw_blackjax_samples(
                    None,  # likelihood not needed — logdensity built internally
                    converged_dict,
                    draw_key,
                    n_posterior_samples,
                    all_sample_dicts,
                    verbose=verbose,
                )
            else:
                # Use nonlinear draws for geoVI modes, linear for MGVI
                use_nonlinear = sample_mode in (
                    "geovi",
                    "nonlinear_resample",
                    "nonlinear_update",
                    "nonlinear_sample",
                )
                if use_nonlinear:
                    all_sample_dicts = self._draw_nonlinear_jit_samples(
                        converged_dict,
                        draw_key,
                        n_posterior_samples,
                        all_sample_dicts,
                        verbose=verbose,
                    )
                else:
                    if verbose:
                        print(f"  Drawing {n_posterior_samples} posterior samples (JIT CG)...")
                    draw_keys = jax.random.split(draw_key, n_posterior_samples)
                    residuals_flat = engine["draw_samples"](converged_flat, draw_keys, data_args)
                    for i in range(n_posterior_samples):
                        res = unflatten(residuals_flat[i])
                        combined = {k: converged_dict[k] + res[k] for k in converged_dict}
                        all_sample_dicts.append(combined)

        wall_time = time.time() - t0
        n_posterior = len(all_sample_dicts)

        # Convert to physical space
        samples_phys = {}
        for sample_dict in all_sample_dicts:
            phys = self._to_physical(sample_dict)
            for k, v in phys.items():
                if k not in samples_phys:
                    samples_phys[k] = []
                samples_phys[k].append(v)

        samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
        best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

        # --- Post-fit diagnostics ---
        diag_warnings = []

        # Check chi2/dof
        if self.data_type == "photometry":
            pred = self.model.predict_photometry(best_params)
            chi2_dof = float(jnp.sum(((self.data - pred) / self.noise) ** 2)) / len(self.data)
            if chi2_dof > 5.0:
                diag_warnings.append(f"Poor fit: chi2/dof={chi2_dof:.1f} (expected ~1)")
            elif chi2_dof < 0.1:
                diag_warnings.append(f"Suspiciously good fit: chi2/dof={chi2_dof:.2f}")
        else:
            chi2_dof = None

        # Check parameters at bounds
        at_bounds = []
        for name in self._free_names:
            if name in samples_phys:
                lo, hi = self._bounds[name]
                med = float(jnp.median(samples_phys[name]))
                margin = 0.02 * (hi - lo)
                if med < lo + margin or med > hi - margin:
                    at_bounds.append(name)
        if at_bounds:
            diag_warnings.append(
                f"Parameters near bounds: {', '.join(at_bounds)}. Consider widening the prior."
            )

        # Check for NaN
        has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in samples_phys.values())
        if has_nan:
            diag_warnings.append("NaN detected in posterior samples!")

        if verbose:
            print(
                f"  EVI (JIT) complete in {wall_time:.1f}s, "
                f"{best_iters}/{n_iterations} iterations, "
                f"{n_posterior} posterior samples"
            )
            for w in diag_warnings:
                print(f"  WARNING: {w}")

        # Also emit as proper warnings for non-verbose mode
        for w in diag_warnings:
            warnings.warn(w, UserWarning, stacklevel=2)

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method="EVI (JIT)",
            wall_time_s=wall_time,
            diagnostics={
                "n_iterations": best_iters,
                "n_iterations_max": n_iterations,
                "n_samples": n_posterior,
                "n_seeds": n_seeds,
                "chi2_dof": chi2_dof,
                "sample_mode": "evi_jit",
            },
            loss_history=None,
            _model=self.model,
        )

    # -------------------------------------------------------------------
    # Inference methods
    # -------------------------------------------------------------------

    def run(self, method, *, init_from=None, key=None, **kwargs):
        """Run inference.

        Parameters
        ----------
        method : str
            **Default (native JIT, fully XLA-compiled):**
            ``"geovi"`` — geoVI with nonlinear coordinate curving.
            ``"mgvi"`` — MGVI (linearized geoVI).

            **Hybrid (native geoVI optimization + NUTS posterior):**
            ``"geovi_nuts"`` — geoVI optimization, then NUTS samples.

            **NIFTy (exact math, tight Python loop):**
            ``"fast_geovi"`` — geoVI via NIFTy OptimizeVI.update.
            ``"fast_mgvi"`` — MGVI via NIFTy OptimizeVI.update.

            **NIFTy (full jft.optimize_kl with logging/diagnostics):**
            ``"nifty_geovi"`` — Full NIFTy geoVI with minisanity.
            ``"nifty_mgvi"`` — Full NIFTy MGVI with logging.

            **Other:**
            ``"map"`` — MAP optimization (Adam/SGD).
            ``"raytrace"`` — Ray Tracing Sampler (exact MCMC).
            ``"nuts"`` — NUTS via BlackJAX (exact MCMC).
            ``"nss"`` — Nested Slice Sampling (evidence + posterior, smooth only).
            ``"laplace"`` — Laplace approximation (Gaussian from Hessian at MAP).
            ``"pathfinder"`` — Pathfinder (fast approximate via L-BFGS path).
            ``"elliptical_slice"`` — Elliptical Slice Sampling (exact, for GP latents).
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
        elif method == "raytrace":
            return self._run_raytrace(key=key, init_from=init_from, **kwargs)
        elif method == "nuts":
            return self._run_nuts(key=key, init_from=init_from, **kwargs)
        # --- Native JIT (DEFAULT): fully XLA-compiled geoVI/MGVI ---
        elif method in ("geovi", "native_geovi"):
            return self._run_native_vi(
                key=key,
                init_from=init_from,
                sample_mode="geovi",
                **kwargs,
            )
        elif method in ("mgvi", "native_mgvi", "evi", "native_evi"):
            return self._run_native_vi(
                key=key,
                init_from=init_from,
                sample_mode="linear",
                **kwargs,
            )
        # --- Hybrid: native geoVI optimization + NUTS posterior sampling ---
        elif method == "geovi_nuts":
            return self._run_native_vi(
                key=key,
                init_from=init_from,
                sample_mode="geovi",
                posterior_method="blackjax",
                **kwargs,
            )
        # --- NIFTy: exact math, tight Python loop ---
        elif method == "fast_geovi":
            return self._run_fast_vi(
                key=key,
                init_from=init_from,
                sample_mode="nonlinear_resample",
                posterior_method="nonlinear",
                **kwargs,
            )
        elif method == "fast_mgvi":
            return self._run_fast_vi(
                key=key,
                init_from=init_from,
                sample_mode="linear_resample",
                **kwargs,
            )
        # --- NIFTy: full jft.optimize_kl (with logging/minisanity) ---
        elif method == "nifty_geovi":
            return self._run_nifty_vi(key=key, init_from=init_from, **kwargs)
        elif method == "nifty_mgvi":
            return self._run_nifty_vi(
                key=key,
                init_from=init_from,
                sample_mode="linear_resample",
                **kwargs,
            )
        # --- Nested Slice Sampling (evidence computation) ---
        elif method == "nss":
            return self._run_nss(key=key, init_from=init_from, **kwargs)
        # --- Laplace approximation (Gaussian at MAP) ---
        elif method == "laplace":
            return self._run_laplace(key=key, init_from=init_from, **kwargs)
        # --- Pathfinder (fast approximate via L-BFGS path) ---
        elif method == "pathfinder":
            return self._run_pathfinder(key=key, init_from=init_from, **kwargs)
        # --- Elliptical Slice Sampling (exact MCMC for GP latents) ---
        elif method == "elliptical_slice":
            return self._run_elliptical_slice(key=key, init_from=init_from, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Default: 'geovi', 'mgvi'. "
                f"Hybrid: 'geovi_nuts'. "
                f"NIFTy loop: 'fast_geovi', 'fast_mgvi'. "
                f"NIFTy full: 'nifty_geovi', 'nifty_mgvi'. "
                f"Other: 'map', 'raytrace', 'nuts', 'nss', "
                f"'laplace', 'pathfinder', 'elliptical_slice'."
            )

    def _run_nss(
        self,
        *,
        key,
        init_from=None,
        n_live=500,
        num_delete=50,
        num_inner_steps=None,
        log_evidence_tol=-3.0,
        max_iterations=10000,
        n_posterior_samples=1000,
        max_steps=10,
        max_shrinkage=100,
        verbose=True,
    ):
        """Nested Slice Sampling for Bayesian evidence computation.

        Uses Hit-and-Run Slice Sampling (HRSS) as the inner kernel.
        Based on Yallup, Kroupa & Handley (2026, arXiv:2601.23252).

        Restricted to parametric (non-stochastic) SFH models where D ≲ 30.

        Parameters
        ----------
        n_live : int
            Number of live points.
        num_delete : int
            Points to replace per iteration.
        num_inner_steps : int or None
            HRSS walk length per replacement. Defaults to D.
        log_evidence_tol : float
            Terminate when log(Z_remaining) - log(Z_accumulated) < this.
        max_iterations : int
            Safety limit on iterations.
        n_posterior_samples : int
            Number of posterior samples to draw after convergence.
        max_steps : int
            Maximum stepping-out steps in slice sampling.
        max_shrinkage : int
            Maximum shrinking steps in slice sampling.
        verbose : bool
            Print progress.
        """
        from tengri.inference.ns.nss import as_top_level_api
        from tengri.inference.ns.utils import ess as ns_ess, finalise, sample as ns_sample
        from tengri.inference.posterior import Posterior

        # Guard stochastic models
        if self.spec.stochastic:
            raise ValueError(
                "NSS not supported for stochastic SFH models (D~137). "
                "Use 'geovi' or 'raytrace' instead."
            )

        logprior_fn = self._build_logprior_fn()
        loglikelihood_fn = self._build_loglikelihood_fn()

        D = len(self._free_names)
        if num_inner_steps is None:
            num_inner_steps = D

        if verbose:
            print(
                f"NSS: {D} parameters, {n_live} live points, "
                f"{num_delete} deletions/iter, {num_inner_steps} HRSS steps"
            )

        # Build NSS algorithm
        algo = as_top_level_api(
            logprior_fn,
            loglikelihood_fn,
            num_inner_steps,
            num_delete=num_delete,
            max_steps=max_steps,
            max_shrinkage=max_shrinkage,
        )

        # Initialize live points from prior
        key, init_key = jax.random.split(key)
        all_samples = self.spec.sample_batch(init_key, n_live)
        particles = {name: all_samples[name] for name in self._free_names}

        live = algo.init(particles)
        step = jax.jit(algo.step)

        dead_points = []
        n_iter = 0
        t0 = time.time()

        while True:
            key, subkey = jax.random.split(key)
            live, dead = step(subkey, live)
            dead_points.append(dead)
            n_iter += 1

            # Check termination
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

        # Finalise and extract posterior samples
        ns_run = finalise(live, dead_points)

        key, sample_key = jax.random.split(key)
        resampled = ns_sample(sample_key, ns_run, n_posterior_samples)

        key, ess_key = jax.random.split(key)
        ess_val = float(ns_ess(ess_key, ns_run))

        # Convert to physical param dict
        samples_phys = {}
        for name in self._free_names:
            samples_phys[name] = resampled.position[name]
        # Add fixed params (broadcast)
        for name, val in self._fixed_values.items():
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
                "log_evidence_err": float(jnp.sqrt(jnp.maximum(ess_val, 1.0)) / n_live),
                "ess": ess_val,
            },
            log_evidence=logZ,
            _model=self.model,
        )

    def _run_fast_vi(
        self,
        *,
        key,
        init_from=None,
        n_iterations=10,
        n_samples=3,
        n_posterior_samples=200,
        sample_mode="nonlinear_resample",
        vi_config=None,
        posterior_method="jit",
        verbose=True,
    ):
        """NIFTy geoVI via OptimizeVI.update in a tight loop.

        Uses NIFTy's exact CG, Newton-CG, line search, and sampnorm.
        Skips logging, pickling, and stdout capture for ~35% speedup
        over ``_run_nifty_vi`` while producing identical results.

        Parameters
        ----------
        n_iterations : int
            Number of KL minimization iterations.
        n_samples : int
            Samples per iteration (doubled by mirror_samples).
        n_posterior_samples : int
            Posterior samples drawn after convergence.
        sample_mode : str
            ``"nonlinear_resample"`` (geoVI), ``"linear_resample"`` (MGVI),
            or ``"evi"`` (MGVI first half, geoVI second half).
        vi_config : VIConfig, optional
            Advanced configuration. If None, uses defaults.
        posterior_method : str
            ``"jit"`` (default) — fast JIT CG sampling.
            ``"blackjax"`` — independent NUTS sampling.
        verbose : bool
            Print progress.
        """
        from tengri.inference.posterior import Posterior
        from tengri.inference.vi_config import VIConfig, evi_sample_mode

        cfg = vi_config or VIConfig()

        # Build JIT engine (includes NIFTy likelihood + OptimizeVI)
        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(init_params)

        engine = self._jit_sampler
        flatten = engine["flatten"]
        unflatten = engine["unflatten"]

        if engine["run_nifty_jit"] is None:
            # NIFTy not available, fall back to full _run_nifty_vi
            return self._run_nifty_vi(
                key=key,
                init_from=init_from,
                n_iterations=n_iterations,
                n_samples=n_samples,
                n_posterior_samples=n_posterior_samples,
                sample_mode=sample_mode,
                vi_config=vi_config,
                posterior_method=posterior_method,
                verbose=verbose,
            )

        # Resolve sample mode.
        # For geoVI: use periodic resample + update schedule (prevents
        # sample staleness while maintaining stable convergence).
        resample_every = 5  # refresh scouts every 5 iterations
        if sample_mode == "evi":
            resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
        elif sample_mode == "nonlinear_resample":
            # Optimal schedule: resample at 0, then every resample_every,
            # nonlinear_update in between.
            def resolved_mode(i: int) -> str:
                if i == 0 or i % resample_every == 0:
                    return "nonlinear_resample"
                return "nonlinear_update"
        else:
            resolved_mode = sample_mode

        n_total = len(self._free_names) + (self.spec.n_grid if self.spec.stochastic else 0)
        if verbose:
            _mode_labels = {
                "nonlinear_resample": "geovi",
                "linear_resample": "mgvi",
                "evi": "evi",
            }
            mode_label = _mode_labels.get(sample_mode, sample_mode)
            print(
                f"{mode_label}: {n_total} params, {len(self.data)} data points, "
                f"{n_iterations} iterations, {n_samples} samples/iter"
            )

        t0 = time.time()

        pos_flat = flatten(init_params)
        key, opt_key = jax.random.split(key)

        converged_flat, nifty_samples = engine["run_nifty_jit"](
            pos_flat,
            opt_key,
            n_iterations=n_iterations,
            n_samples=n_samples,
            sample_mode_str=resolved_mode,
            draw_linear_kwargs=cfg.draw_linear_kwargs,
            nonlinearly_update_kwargs=cfg.nonlinearly_update_kwargs,
            kl_kwargs=cfg.kl_kwargs,
        )

        converged_dict = unflatten(converged_flat)

        # Draw posterior samples (fast JIT path)
        key, draw_key = jax.random.split(key)
        all_sample_dicts = []

        # Include optimization samples from last iteration
        if nifty_samples is not None:
            for s in list(nifty_samples):
                sd = s.tree if hasattr(s, "tree") else dict(s)
                all_sample_dicts.append(sd)

        if n_posterior_samples > 0:
            if posterior_method == "nonlinear":
                # geoVI-curved posterior samples (captures non-Gaussian shapes)
                all_sample_dicts = self._draw_nonlinear_jit_samples(
                    converged_dict,
                    draw_key,
                    n_posterior_samples,
                    all_sample_dicts,
                    verbose=verbose,
                )
            elif posterior_method == "jit":
                # Linear CG posterior samples (MGVI approximation)
                all_sample_dicts = self._draw_jit_samples(
                    converged_dict,
                    draw_key,
                    n_posterior_samples,
                    all_sample_dicts,
                    verbose=verbose,
                )
            elif posterior_method == "blackjax":
                lh = engine["nifty_likelihood"]
                all_sample_dicts = self._draw_blackjax_samples(
                    lh,
                    converged_dict,
                    draw_key,
                    n_posterior_samples,
                    all_sample_dicts,
                    verbose=verbose,
                )
            else:
                all_sample_dicts = self._draw_jit_samples(
                    converged_dict,
                    draw_key,
                    n_posterior_samples,
                    all_sample_dicts,
                    verbose=verbose,
                )

        wall_time = time.time() - t0
        n_posterior = len(all_sample_dicts)

        # Convert to physical space
        samples_phys = {}
        for sample_dict in all_sample_dicts:
            phys = self._to_physical(sample_dict)
            for k, v in phys.items():
                if k not in samples_phys:
                    samples_phys[k] = []
                samples_phys[k].append(v)

        samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
        best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

        # Chi2/dof diagnostic
        chi2_dof = None
        if self.data_type == "photometry" and best_params:
            pred = self.model.predict_photometry(best_params)
            chi2_dof = float(jnp.sum(((self.data - pred) / self.noise) ** 2)) / len(self.data)

        if verbose:
            print(
                f"  {_mode_labels.get(sample_mode, sample_mode)} complete in "
                f"{wall_time:.1f}s, {n_iterations} iterations, "
                f"{n_posterior} posterior samples"
            )
            if chi2_dof is not None and chi2_dof > 5.0:
                print(f"  WARNING: Poor fit: chi2/dof={chi2_dof:.1f}")

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method=f"fast_{sample_mode}",
            wall_time_s=wall_time,
            diagnostics={
                "n_iterations": n_iterations,
                "n_samples": n_posterior,
                "chi2_dof": chi2_dof,
                "sample_mode": sample_mode,
            },
            loss_history=None,
            _model=self.model,
        )

    def fit_batch(
        self,
        batch,
        *,
        method="native_geovi",
        key=None,
        verbose=True,
        **kwargs,
    ):
        """Fit a batch of galaxies efficiently.

        Creates a Fitter per galaxy, sharing the XLA compilation cache.
        The first galaxy pays compile cost; subsequent galaxies load
        from the persistent XLA cache (milliseconds each).

        Works with any inference method — native_geovi (default) gives
        the best speed. Also usable for hierarchical individual fits.

        Parameters
        ----------
        batch : list of dict
            Each dict has "flux_obs" and "noise" arrays.
        method : str
            Default "native_geovi". Any method from run().
        key : PRNGKey, optional
        verbose : bool
        **kwargs
            Passed to run() (n_iterations, n_samples, n_seeds, etc).

        Returns
        -------
        list of Posterior

        Example
        -------
        >>> galaxies = [{"flux_obs": f, "noise": n} for f, n in zip(fluxes, noises)]
        >>> results = fitter.fit_batch(galaxies)
        >>> # First: ~15s compile. Rest: ~2ms each (native_geovi).
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        if "native" in method and "n_seeds" not in kwargs:
            kwargs["n_seeds"] = 5

        n_gal = len(batch)
        if verbose:
            print(f"fit_batch: {n_gal} galaxies, method={method}")

        results = []
        t0 = time.time()

        for i, gal in enumerate(batch):
            gal_key = jax.random.fold_in(key, i)
            t_gal = time.time()

            fitter_i = Fitter(
                self.model,
                gal["flux_obs"],
                gal["noise"],
                data_type=self.data_type,
            )
            result_i = fitter_i.run(method, key=gal_key, verbose=False, **kwargs)
            results.append(result_i)

            dt = time.time() - t_gal
            if verbose and (i < 3 or (i + 1) % max(1, n_gal // 10) == 0 or i == n_gal - 1):
                chi2 = result_i.diagnostics.get("chi2_dof", "?")
                chi2_str = f"{chi2:.2f}" if isinstance(chi2, float) else str(chi2)
                print(f"  Galaxy {i + 1}/{n_gal}: chi2/dof={chi2_str}, {dt:.1f}s")

        t_total = time.time() - t0
        if verbose:
            print(f"  Done: {n_gal} galaxies in {t_total:.1f}s ({t_total / n_gal:.1f}s/galaxy)")

        return results

    def _run_map(
        self,
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
                            print(
                                f"  Early stopping at step {i} "
                                f"(no improvement for {patience} steps)"
                            )
                        break

        wall_time = time.time() - t0
        best_params = self._to_physical(params)

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
            _model=self.model,
        )

    def _run_raytrace(
        self,
        *,
        key,
        init_from=None,
        n_burnin=100,
        n_steps=500,
        n_leapfrog_steps=10,
        step_size=None,
        refresh_rate=0.0,
        verbose=True,
    ):
        """Ray Tracing Sampler (Behroozi 2025).

        Propagates light rays through a medium where the refractive
        index n(x) = L(x)^{1/(D-1)}, using Snell's law to bend rays
        toward high-likelihood regions.

        The sampling proceeds in two phases:
        1. **Burn-in**: initial samples are discarded to let the chain
           forget its starting position and reach the typical set.
        2. **Sampling**: posterior samples are collected.

        Parameters
        ----------
        n_burnin : int
            Burn-in steps (discarded).
        n_steps : int
            Post-burn-in samples to collect.
        n_leapfrog_steps : int
            Leapfrog integration steps per trajectory.
        step_size : float, optional
            Integration step size. Default: 0.03 * sqrt(D).
        refresh_rate : float
            Partial momentum refresh rate. 0 = no refresh (pure ray tracing).
        verbose : bool
            Print progress.
        """
        from tengri.inference.posterior import Posterior
        from tengri.inference.raytrace import sample_raytrace

        loss_fn = self._build_loss_fn()

        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        # Flatten for the sampler (expects a flat 1D array)
        init_flat, unravel_fn = ravel_pytree(init_params)
        D = len(init_flat)

        if step_size is None:
            # Behroozi (2025) recommends 0.03 * sqrt(D), but for
            # stochastic SFH models the psd_xi variables create a
            # tighter curvature. Use a smaller default for D > 10.
            if D <= 10:
                step_size = 0.03 * jnp.sqrt(float(D))
            else:
                step_size = 0.01

        def log_prob_flat(position):
            params = unravel_fn(position)
            return -loss_fn(params)

        total_steps = n_burnin + n_steps

        if verbose:
            print(
                f"Ray Tracing: {D} params, {n_burnin} burn-in + "
                f"{n_steps} samples, {n_leapfrog_steps} leapfrog/step, "
                f"step_size={float(step_size):.4f}"
            )

        t0 = time.time()

        key, sample_key = jax.random.split(key)
        chain, log_likelihood, accept_prob = sample_raytrace(
            key=sample_key,
            params_init=init_flat,
            log_prob_fn=log_prob_flat,
            n_steps=total_steps,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=float(step_size),
            refresh_rate=float(refresh_rate),
            metro_check=1,
            sample_hmc=False,
        )

        wall_time = time.time() - t0

        # Discard burn-in
        chain = chain[n_burnin:]
        log_likelihood = log_likelihood[n_burnin:]
        accept_prob_post = accept_prob[n_burnin:]
        n_samples_out = chain.shape[0]

        mean_accept = float(jnp.mean(accept_prob))
        mean_accept_post = float(jnp.mean(accept_prob_post))

        # Convert to physical parameter space (vectorized)
        def _convert_one(flat_sample):
            return self._to_physical(unravel_fn(flat_sample))

        samples_phys = jax.vmap(_convert_one)(chain)
        best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

        if verbose:
            print(
                f"  Ray Tracing complete in {wall_time:.1f}s. "
                f"Acceptance: {mean_accept:.1%} (overall), "
                f"{mean_accept_post:.1%} (post burn-in). "
                f"Samples: {n_samples_out}"
            )

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method="Ray Tracing (Behroozi 2025)",
            wall_time_s=wall_time,
            diagnostics={
                "n_burnin": n_burnin,
                "n_steps": n_steps,
                "n_samples": n_samples_out,
                "n_leapfrog_steps": n_leapfrog_steps,
                "step_size": float(step_size),
                "refresh_rate": float(refresh_rate),
                "accept_rate": mean_accept,
                "accept_rate_post_burnin": mean_accept_post,
            },
            loss_history=None,
            _model=self.model,
        )

    def _run_nuts(
        self,
        *,
        key,
        init_from=None,
        n_warmup=500,
        n_burnin=0,
        n_samples=1000,
        target_accept_rate=0.8,
        max_num_doublings=10,
        verbose=True,
    ):
        """NUTS sampling via BlackJAX.

        The sampling proceeds in three phases:
        1. **Warmup**: BlackJAX window adaptation tunes step size
           and mass matrix.
        2. **Burn-in**: additional post-warmup steps that are
           discarded (lets the chain equilibrate at the tuned params).
        3. **Sampling**: posterior samples are collected.

        Parameters
        ----------
        n_warmup : int
            Warmup/adaptation steps (tunes step size and mass matrix).
        n_burnin : int
            Additional post-warmup burn-in steps (discarded).
        n_samples : int
            Post-burn-in samples to collect.
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
            raise ImportError("blackjax required for NUTS: pip install blackjax") from None

        from tengri.inference.posterior import Posterior

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
            burnin_msg = f", {n_burnin} burn-in" if n_burnin > 0 else ""
            print(
                f"NUTS: {n_dim} parameters, {n_warmup} warmup{burnin_msg}, "
                f"{n_samples} samples, target_accept={target_accept_rate}"
            )

        t0 = time.time()

        # Window adaptation with target acceptance rate
        key, warmup_key = jax.random.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_posterior_flat,
            target_acceptance_rate=target_accept_rate,
        )
        (state, parameters), _ = warmup.run(warmup_key, init_flat, num_steps=n_warmup)

        if verbose:
            print(
                f"  Warmup complete ({time.time() - t0:.1f}s). "
                f"Step size: {float(parameters['step_size']):.4f}"
            )

        # Sampling
        kernel = blackjax.nuts(log_posterior_flat, **parameters).step

        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state.position, info)

        # Burn-in: run steps but discard
        if n_burnin > 0:
            key, burnin_key = jax.random.split(key)
            burnin_keys = jax.random.split(burnin_key, n_burnin)
            for sk in burnin_keys:
                state, _ = one_step(state, sk)
            if verbose:
                print(f"  Burn-in complete ({n_burnin} steps discarded)")

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
                print(f"  Sample {i + 1}/{n_samples}")

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
            print(f"  NUTS complete in {wall_time:.1f}s. Divergences: {n_divergent}/{n_samples}")

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method="NUTS (BlackJAX)",
            wall_time_s=wall_time,
            diagnostics={
                "n_warmup": n_warmup,
                "n_burnin": n_burnin,
                "n_samples": n_samples,
                "n_divergent": n_divergent,
                "step_size": float(parameters["step_size"]),
            },
            loss_history=None,
            _model=self.model,
        )

    # -------------------------------------------------------------------
    # Laplace, Pathfinder, Elliptical Slice Sampling
    # -------------------------------------------------------------------

    def _build_loglikelihood_unbounded_fn(self):
        """Build a log-likelihood function in unbounded parameter space.

        For Elliptical Slice Sampling, which handles the N(0,I) prior
        internally. Returns only the data likelihood (no prior terms).
        """
        loglik_fn = self._build_loglikelihood_fn()
        bounds = self._bounds
        free_names = self._free_names
        fixed_values = self._fixed_values
        spec = self.spec

        def loglik_unbounded(params_unbounded):
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(params_unbounded[name], lo, hi)
            for name, val in fixed_values.items():
                params[name] = val
            if spec.stochastic and "psd_xi" in params_unbounded:
                params["psd_xi"] = params_unbounded["psd_xi"]
            return loglik_fn(params)

        return loglik_unbounded

    def _run_laplace(self, *, key, init_from=None, n_map_steps=1000, **kwargs):
        """Laplace approximation: Gaussian posterior from Hessian at MAP."""
        from tengri.inference.laplace import run_laplace

        loss_fn = self._build_loss_fn()

        if init_from is not None:
            map_params = self._unbounded_from_posterior(init_from)
        else:
            map_result = self._run_map(
                key=key,
                n_steps=n_map_steps,
                verbose=kwargs.get("verbose", True),
            )
            map_params = self._unbounded_from_posterior(map_result)

        return run_laplace(
            key=key,
            loss_fn=loss_fn,
            map_params_unbounded=map_params,
            to_physical_fn=self._to_physical,
            model=self.model,
            **kwargs,
        )

    def _run_pathfinder(self, *, key, init_from=None, **kwargs):
        """Pathfinder: fast approximate posterior via L-BFGS path."""
        from tengri.inference.pathfinder import run_pathfinder

        loss_fn = self._build_loss_fn()

        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        return run_pathfinder(
            key=key,
            loss_fn=loss_fn,
            init_params=init_params,
            to_physical_fn=self._to_physical,
            model=self.model,
            **kwargs,
        )

    def _run_elliptical_slice(self, *, key, init_from=None, **kwargs):
        """Elliptical Slice Sampling for Gaussian-prior latent models."""
        from tengri.inference.elliptical_slice import run_elliptical_slice

        loglik_fn = self._build_loglikelihood_unbounded_fn()

        if init_from is not None:
            init_params = self._unbounded_from_posterior(init_from)
        else:
            init_params = self._initialize_unbounded(key)

        return run_elliptical_slice(
            key=key,
            loglikelihood_unbounded_fn=loglik_fn,
            init_params=init_params,
            to_physical_fn=self._to_physical,
            model=self.model,
            **kwargs,
        )

    def _run_nifty_vi(
        self,
        *,
        key,
        init_from=None,
        n_iterations=10,
        n_samples=3,
        n_posterior_samples=200,
        sample_mode="nonlinear_resample",
        vi_config=None,
        posterior_method="jit",
        verbose=True,
    ):
        """Geometric variational inference via NIFTy.re.

        geoVI finds a coordinate transformation where the posterior is
        approximately Gaussian, then draws samples in that space. Much
        faster than MCMC for high-dimensional problems.

        Parameters
        ----------
        n_iterations : int
            Number of KL minimization iterations (optimization).
        n_samples : int
            Samples per iteration during optimization.
            With ``mirror_samples=True`` (default), this doubles internally.
        n_posterior_samples : int
            Number of posterior samples to draw after convergence.
            These are cheap to generate once the approximation is found.
        sample_mode : str
            "nonlinear_resample" (geoVI), "linear_resample" (MGVI),
            or "evi" (MGVI first, then geoVI — recommended).
        vi_config : VIConfig, optional
            Advanced configuration for NIFTy optimize_kl.
            If None, uses Philipp Frank's recommended defaults.
        posterior_method : str
            "linear" (default) — draw_linear_residual, consistent with geoVI.
            "blackjax" — BlackJAX NUTS, independent MCMC samples.
        verbose : bool
            Print progress.
        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError("nifty8.re required for geoVI: pip install nifty8[re]") from None

        from tengri.inference.posterior import Posterior
        from tengri.inference.vi_config import VIConfig, evi_sample_mode

        cfg = vi_config or VIConfig()

        from tengri.core.noise import (
            compute_effective_noise,
            compute_std_inv,
            has_noise_model,
            uses_student_t,
        )

        model = self.model
        data = self.data
        noise = self.noise
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        spec = self.spec
        stochastic = spec.stochastic
        use_variable_noise = has_noise_model(spec)
        use_student_t = uses_student_t(spec)

        def _predict(params):
            """Dispatch forward model by data type."""
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

        def _build_params(primals):
            """Transform unbounded primals → physical params dict."""
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(primals[name], lo, hi)
            for name, val in fixed_values.items():
                params[name] = val
            if stochastic and "psd_xi" in primals:
                params["psd_xi"] = primals["psd_xi"]
            return params

        # Build signal_response: unbounded primals → predicted observables
        # When noise model is active, returns tuple for variable-covariance
        # likelihoods:
        #   Gaussian: (predicted, std_inv) for VariableCovarianceGaussian
        #   Student-t: (predicted, sigma_eff) for VariableCovarianceStudentT
        if use_variable_noise:
            if use_student_t:

                def signal_response(primals):
                    params = _build_params(primals)
                    predicted = _predict(params)
                    f_cal = params.get("noise_frac_cal", 0.0)
                    sigma_eff = compute_effective_noise(noise, predicted, f_cal)
                    return predicted, sigma_eff

            else:

                def signal_response(primals):
                    params = _build_params(primals)
                    predicted = _predict(params)
                    f_cal = params.get("noise_frac_cal", 0.0)
                    std_inv = compute_std_inv(noise, predicted, f_cal)
                    return predicted, std_inv

        else:

            def signal_response(primals):
                return _predict(_build_params(primals))

        # Build NIFTy.re domain
        domain = {}
        for name in free_names:
            domain[name] = jft.ShapeWithDtype(())
        if stochastic:
            domain["psd_xi"] = jft.ShapeWithDtype((spec.n_grid,))

        signal_response_jit = jax.jit(signal_response)
        nifty_model = jft.Model(signal_response_jit, domain=domain)

        # Likelihood dispatch:
        #   Student-t + variable noise → VariableCovarianceStudentT
        #   Gaussian + variable noise → VariableCovarianceGaussian
        #   Fixed noise → Gaussian
        if use_student_t and use_variable_noise:
            dof = float(spec.get_distribution("noise_dof").value)
            likelihood = jft.VariableCovarianceStudentT(data, dof).amend(nifty_model)
        elif use_variable_noise:
            likelihood = jft.VariableCovarianceGaussian(data).amend(nifty_model)
        else:
            noise_cov_inv = 1.0 / noise**2
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
            _mode_labels = {
                "nonlinear_resample": "geoVI",
                "linear_resample": "MGVI",
                "evi": "EVI (MGVI→geoVI)",
            }
            mode_label = _mode_labels.get(sample_mode, sample_mode)
            print(
                f"{mode_label}: {n_total} params, {len(data)} data points, "
                f"{n_iterations} iterations, {n_samples} samples/iter"
            )

        t0 = time.time()

        # Resolve sample_mode — same optimal schedule as _run_fast_vi
        resample_every = 5
        if sample_mode == "evi":
            resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
        elif sample_mode == "nonlinear_resample":

            def resolved_mode(i: int) -> str:
                if i == 0 or i % resample_every == 0:
                    return "nonlinear_resample"
                return "nonlinear_update"

        else:
            resolved_mode = sample_mode

        key, opt_key = jax.random.split(key)
        samples, _state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            key=opt_key,
            sample_mode=resolved_mode,
            residual_map=jax.vmap if cfg.use_vmap else "lmap",
            draw_linear_kwargs=cfg.draw_linear_kwargs,
            nonlinearly_update_kwargs=cfg.nonlinearly_update_kwargs,
            kl_kwargs=cfg.kl_kwargs,
            odir=None,
        )

        # Draw additional posterior samples from the converged approximation
        converged_pos = samples.pos
        key, draw_key = jax.random.split(key)

        all_sample_dicts = []

        # Include the optimization samples (from the last iteration)
        for s in list(samples):
            sd = s.tree if hasattr(s, "tree") else dict(s)
            all_sample_dicts.append(sd)

        # Draw additional samples
        if n_posterior_samples > 0:
            pos_dict = (
                converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
            )
            all_sample_dicts = self._draw_posterior_samples(
                likelihood,
                pos_dict,
                draw_key,
                n_posterior_samples,
                all_sample_dicts,
                method=posterior_method,
                verbose=verbose,
            )

        wall_time = time.time() - t0
        n_posterior = len(all_sample_dicts)

        # Convert all samples to physical space
        samples_phys = {}
        for sample_dict in all_sample_dicts:
            phys = self._to_physical(sample_dict)
            for k, v in phys.items():
                if k not in samples_phys:
                    samples_phys[k] = []
                samples_phys[k].append(v)

        samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
        best_params = {k: jnp.mean(v, axis=0) for k, v in samples_phys.items()}

        _mode_labels = {
            "nonlinear_resample": "geoVI",
            "linear_resample": "MGVI",
            "evi": "EVI",
        }
        mode_label = _mode_labels.get(sample_mode, sample_mode)

        if verbose:
            print(f"  {mode_label} complete in {wall_time:.1f}s, {n_posterior} posterior samples")

        return Posterior(
            samples=samples_phys,
            params=best_params,
            method=f"{mode_label} (NIFTy.re)",
            wall_time_s=wall_time,
            diagnostics={
                "n_iterations": n_iterations,
                "n_samples": n_posterior,
                "sample_mode": sample_mode,
            },
            loss_history=None,
            _model=self.model,
        )
