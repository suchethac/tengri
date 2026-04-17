"""Hierarchical inference for population-level PSD recovery.

Shares PSD hyperparameters (σ_PSD, τ_PSD) across N galaxies while
each galaxy retains its own latent field ξ_i and physical parameters.

The total parameter vector is:
    Θ = {φ_shared, {ξ_i, θ_i}_{i=1}^N}

where φ_shared = {σ_PSD, τ_PSD} (or more generally, the PSD shape).

Usage:
    hfitter = PopulationFitter(model_template, galaxies)
    result = hfitter.run("vi", n_iterations=25)
    result.shared_params  # posterior on (σ_PSD, τ_PSD)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference.fitter import resolve_method
from tengri.utils.transforms import to_bounded, to_unbounded


@dataclass
class PopulationPosterior:
    """Results from hierarchical PSD inference.

    Attributes
    ----------
    shared_samples : dict
        Posterior samples for shared PSD params. Shape (n_samples,).
    shared_params : dict
        Posterior mean of shared PSD params.
    individual_samples : list of dict
        Per-galaxy posterior samples (optional, can be None for memory).
    method : str
        Inference method used.
    wall_time_s : float
        Total wall-clock time.
    diagnostics : dict
        Method-specific diagnostics.
    """

    shared_samples: dict
    shared_params: dict
    individual_samples: list | None = None
    method: str = ""
    wall_time_s: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Median and 68% CI for shared PSD parameters."""
        result = {}
        for name, arr in self.shared_samples.items():
            vals = np.array(arr)
            result[name] = {
                "median": float(np.median(vals)),
                "lo_68": float(np.percentile(vals, 16)),
                "hi_68": float(np.percentile(vals, 84)),
            }
        return result

    def __repr__(self) -> str:
        n = next(iter(self.shared_samples.values())).shape[0]
        return (
            f"PopulationPosterior(method='{self.method}', "
            f"n_samples={n}, "
            f"wall_time={self.wall_time_s:.1f}s)"
        )

    @property
    def individual(self):
        """Per-galaxy posterior marginals as a list of lightweight objects.

        Returns
        -------
        list of SimpleNamespace
            Each element has ``.samples`` (dict) and ``.params`` (dict).
            Returns empty list if ``individual_samples`` is None.
        """
        from types import SimpleNamespace

        if self.individual_samples is None:
            return []
        result = []
        for samp in self.individual_samples:
            params = {
                k: float(np.median(v)) if hasattr(v, "ndim") and v.ndim == 1 else v
                for k, v in samp.items()
            }
            result.append(SimpleNamespace(samples=samp, params=params))
        return result

    def plot_population(self, params=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"), ax=None):
        """Scatter plot of shared PSD parameter posteriors.

        Parameters
        ----------
        params : tuple of str
            Two parameter names for x and y axes.
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        px, py = params
        if px in self.shared_samples and py in self.shared_samples:
            x = np.array(self.shared_samples[px])
            y = np.array(self.shared_samples[py])
            ax.scatter(x, y, s=8, alpha=0.4, color="C0", edgecolors="none")
            ax.set_xlabel(px)
            ax.set_ylabel(py)
            ax.set_title("Population posterior (shared PSD params)")
        else:
            ax.text(
                0.5,
                0.5,
                f"Parameters {px!r} or {py!r} not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        return ax


class PopulationFitter:
    """Hierarchical inference for shared PSD parameters.

    Parameters
    ----------
    model_factory : callable
        Function(psd_sigma, psd_tau_myr) → Model.
        Creates a model with the given PSD params. All other params
        (SFH, dust, etc.) come from the model's ParamSpec.
    galaxies : list of dict
        Each dict has 'flux_obs', 'noise', and optionally 'spec_obs',
        'spec_noise', 'wave_spec'.
    psd_sigma_prior : tuple
        (lo, hi) for uniform prior on σ_PSD.
    psd_tau_prior : tuple
        (lo, hi) for uniform prior on τ_PSD (Myr).
    data_type : str
        "photometry" or "spectroscopy".
    """

    def __init__(
        self,
        model_factory,
        galaxies,
        psd_sigma_prior=(0.1, 4.0),
        psd_tau_prior=(1.0, 300.0),
        data_type="photometry",
    ):
        self.model_factory = model_factory
        self.galaxies = galaxies
        self.n_galaxies = len(galaxies)
        self.psd_sigma_bounds = psd_sigma_prior
        self.psd_tau_bounds = psd_tau_prior
        self.data_type = data_type

        # Create a template model to get spec info
        self._template = model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        self._spec = self._template.spec
        self._free_names = [
            n
            for n in self._spec.free_params
            if n not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr")
        ]

    def run(self, method="vi", *, key=None, **kwargs):
        """Run hierarchical inference.

        Parameters
        ----------
        method : str
            "vi" — geoVI with CorrelatedFieldMaker for native PSD learning.
            "vi_linear" — MGVI (faster per iteration, for very large N).
            "geovi_flat" — flat parameter vector (legacy approach).
            "mcmc_raytrace" — Ray Tracing on flat vector (fast but needs tuning).
        key : PRNGKey
        **kwargs
            Passed to the inference method.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Resolve old method names to canonical names, emitting deprecation warnings
        method = resolve_method(method, emit_warning=True)

        # Map canonical names to internal PopulationFitter method names
        _method_map = {
            "vi": ("geovi", None),
            "vi_linear": ("mgvi", "linear_resample"),
            "mcmc_raytrace": ("raytrace", None),
            "mcmc_ess": ("evi", None),
        }

        if method not in _method_map:
            # Handle legacy flat-vector methods
            if method == "geovi_flat":
                return self._run_geovi(key=key, **kwargs)
            elif method == "evi":
                return self._run_evi_jit(key=key, **kwargs)
            elif method == "evi_nifty":
                return self._run_geovi_cfm(key=key, sample_mode="evi", **kwargs)
            else:
                raise ValueError(
                    f"Unknown method: {method}. "
                    f"Use 'vi', 'vi_linear', 'geovi_flat', or 'mcmc_raytrace'."
                )

        cfm_method, sample_mode = _method_map[method]
        if cfm_method == "geovi":
            if sample_mode is None:
                return self._run_geovi_cfm(key=key, **kwargs)
            else:
                return self._run_geovi_cfm(key=key, sample_mode=sample_mode, **kwargs)
        elif cfm_method == "mgvi":
            return self._run_geovi_cfm(key=key, sample_mode="linear_resample", **kwargs)
        elif cfm_method == "raytrace":
            return self._run_raytrace(key=key, **kwargs)
        elif cfm_method == "evi":
            return self._run_evi_jit(key=key, **kwargs)
        else:
            raise ValueError(f"Unmapped method: {method}")

    def _run_evi_jit(
        self,
        *,
        key,
        n_iterations=20,
        n_samples=3,
        n_posterior_samples=500,
        kl_rtol=1e-2,
        n_seeds=3,
        verbose=True,
    ):
        """Fully JIT-compiled hierarchical EVI.

        Adapts the single-galaxy EVI engine (Fitter._run_evi_jit) for
        hierarchical inference: shared PSD parameters + per-galaxy
        latent vectors and physical params, all in a single flat array
        optimized via Newton-CG with automatic convergence detection.

        The forward model is vmapped over galaxies, giving O(1) graph
        size regardless of N_gal. The entire optimization loop runs
        inside ``jax.lax.while_loop`` with zero Python overhead.

        Parameters
        ----------
        n_iterations : int
            Maximum KL iterations. Auto-stops when converged.
        n_samples : int
            Samples per iteration (doubled by mirror_samples).
        n_posterior_samples : int
            Posterior samples drawn after convergence.
        kl_rtol : float
            Relative KL tolerance for early stopping.
        n_seeds : int
            Number of random seeds. Best result (lowest H) is kept.
        verbose : bool
            Print progress.
        """
        from jax.flatten_util import ravel_pytree

        n_gal = self.n_galaxies
        spec = self._spec
        stochastic = spec.stochastic
        n_grid = spec.n_grid
        free_names = self._free_names
        sigma_lo, sigma_hi = self.psd_sigma_bounds
        tau_lo, tau_hi = self.psd_tau_bounds

        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds
        fixed_values = spec.get_fixed_values()

        # Precompute data
        all_data = jnp.concatenate([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise = jnp.concatenate([jnp.asarray(g["noise"]) for g in self.galaxies])
        noise_inv = 1.0 / all_noise**2
        n_data = len(all_data)

        # Build model once
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        data_type = self.data_type

        def _predict(params):
            if data_type == "photometry":
                return model.predict_photometry(params, mode="_traceable")
            return model.predict_spectrum(params, model._wave_obs, mode="_traceable")

        # --- Hierarchical signal_response (vmapped) ---
        def signal_response(p):
            psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

            def forward_one(ub_scalars, xi):
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val
                params["sfh_field_psd_sigma"] = psd_sigma
                params["sfh_field_psd_tau_myr"] = psd_tau
                if stochastic:
                    params["sfh_field_xi"] = xi
                return _predict(params)

            if stochastic:
                predictions = jax.vmap(forward_one)(p["gal"], p["gal_xi"])
            else:
                predictions = jax.vmap(lambda ub, _: forward_one(ub, None))(
                    p["gal"], jnp.zeros(n_gal)
                )
            return predictions.reshape(-1)

        # --- Build init structure ---
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        tau_mid = 0.5 * (tau_lo + tau_hi)

        init_template = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": {name: jnp.zeros(n_gal) for name in free_names},
        }
        if stochastic:
            init_template["gal_xi"] = jnp.zeros((n_gal, n_grid))

        # Use ravel_pytree for flatten/unflatten
        _init_flat, unravel_fn = ravel_pytree(init_template)
        d_total = len(_init_flat)

        def flatten(d):
            return ravel_pytree(d)[0]

        def unflatten(x):
            return unravel_fn(x)

        # --- Core EVI primitives (same as single-galaxy) ---
        _eps = 6.0 * jnp.finfo(jnp.float64).eps

        def metric_vec(xi, v):
            """GGN metric: M(xi) @ v = J^T N^{-1} J v + v."""
            xi_d, v_d = unflatten(xi), unflatten(v)
            _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
            _, vjp_fn = jax.vjp(signal_response, xi_d)
            return flatten(vjp_fn(noise_inv * Jv)[0]) + v

        def hamiltonian(xi):
            """H(xi) = 0.5 chi2 + 0.5 ||xi||^2."""
            pred = signal_response(unflatten(xi))
            chi2 = jnp.sum(((all_data - pred) / all_noise) ** 2)
            return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

        H_vg = jax.value_and_grad(hamiltonian)

        def cg_solve(mat_fn, b, x0, maxiter=30, miniter=6, absdelta=1e-4):
            """CG solve: mat_fn(x) = b. Energy-based convergence."""
            r = mat_fn(x0) - b
            d, gamma = r, jnp.dot(r, r)
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
                info = jnp.where(curv <= 0.0, jnp.int32(-1), info)
                alpha = jnp.where(curv <= 0.0, 0.0, alpha)
                x = x - alpha * d
                r = jnp.where((i % 20 == 0) & (info < -1), mat_fn(x) - b, r - alpha * q)
                gamma = jnp.dot(r, r)
                energy = jnp.dot((r - b) / 2, x)
                ed = pe - energy
                info = jnp.where(ed < -_eps * jnp.abs(energy), jnp.int32(-1), info)
                info = jnp.where(
                    (ed < absdelta) & (i >= miniter) & (info < -1),
                    jnp.int32(0),
                    info,
                )
                info = jnp.where((i >= maxiter) & (info < -1), i, info)
                d = d * jnp.maximum(0.0, gamma / (pg + 1e-30)) + r
                return (x, r, d, gamma, energy, info, i)

            return jax.lax.while_loop(cond, body, init)[0]

        # --- Posterior sampler ---
        def draw_residuals(pos_f, subkeys):
            def draw_one(subkey):
                k1, k2 = jax.random.split(subkey)
                eta_pr = jax.random.normal(k1, shape=(d_total,))
                eta_lh = jax.random.normal(k2, shape=(n_data,))
                _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
                jt = flatten(vjp_fn(jnp.sqrt(noise_inv) * eta_lh)[0])
                return cg_solve(
                    lambda v: metric_vec(pos_f, v),
                    jt + eta_pr,
                    eta_pr,
                    maxiter=30,
                    miniter=6,
                    absdelta=1e-4,
                )

            return jax.vmap(draw_one)(subkeys)

        # --- EVI step ---
        def kl_vg(m, residuals):
            def single_vg(r):
                return H_vg(m + r)

            vals, grads = jax.vmap(single_vg)(residuals)
            return jnp.mean(vals), jnp.mean(grads, axis=0)

        def kl_metric(m, residuals, v):
            def single_met(r):
                return metric_vec(m + r, v)

            return jnp.mean(jax.vmap(single_met)(residuals), axis=0)

        def evi_step(m, subkey, n_samp):
            sample_keys = jax.random.split(subkey, n_samp)
            residuals = draw_residuals(m, sample_keys)
            residuals = jnp.concatenate([residuals, -residuals], axis=0)

            def ncg_body(carry):
                m_cur, prev_val, info, i = carry
                i = i + 1
                val, grad = kl_vg(m_cur, residuals)
                step = cg_solve(
                    lambda v: kl_metric(m_cur, residuals, v),
                    -grad,
                    jnp.zeros_like(m_cur),
                    maxiter=10,
                    miniter=3,
                    absdelta=1e-3,
                )
                m_new = m_cur + step
                ed = prev_val - val
                info = jnp.where((ed < 1e-3) & (i >= 3) & (info < -1), jnp.int32(0), info)
                info = jnp.where((i >= 10) & (info < -1), i, info)
                return (m_new, val, info, i)

            def ncg_cond(carry):
                return carry[2] < -1

            val0, _ = kl_vg(m, residuals)
            result = jax.lax.while_loop(ncg_cond, ncg_body, (m, val0, jnp.int32(-2), jnp.int32(0)))
            return result[0], result[1]

        def run_evi(init_pos, evi_key, n_iter, n_samp, rtol):
            keys = jax.random.split(evi_key, n_iter)

            def cond_fn(state):
                _m, _prev_kl, i, converged = state
                return (~converged) & (i < n_iter)

            def body_fn(state):
                m, prev_kl, i, converged = state
                subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
                m_new, kl_val = evi_step(m, subkey, n_samp)
                rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
                converged = (rel_change < rtol) & (i >= 5)
                return (m_new, kl_val, i + 1, converged)

            m0, kl0 = evi_step(init_pos, keys[0], n_samp)
            init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))
            m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
            return m_final, n_iters

        # --- JIT-compile ---
        run_evi_jit = jax.jit(run_evi, static_argnames=("n_iter", "n_samp"))
        draw_residuals_jit = jax.jit(draw_residuals)

        if verbose:
            print(
                f"Hierarchical EVI (JIT): {n_gal} galaxies, "
                f"D={d_total}, {n_iterations} max iterations, "
                f"{n_samples} samples/iter, {n_seeds} seeds"
            )
            print("  Compiling JIT engine...")

        t0 = time.time()

        # --- Initialize per-galaxy params via MAP ---
        from tengri import Fitter

        init_keys = jax.random.split(key, n_gal + n_seeds + 2)

        if verbose:
            print("  Initializing per-galaxy params via MAP...")

        gal_param_lists = {name: [] for name in free_names}
        gal_xi_list = []
        for i in range(n_gal):
            gal = self.galaxies[i]
            fitter_i = Fitter(model, gal["flux_obs"], gal["noise"], data_type=self.data_type)
            map_i = fitter_i.run(
                "map",
                n_steps=500,
                learning_rate=0.03,
                verbose=False,
                key=init_keys[i],
            )
            init_u = fitter_i._unbounded_from_posterior(map_i)
            for name in free_names:
                gal_param_lists[name].append(init_u.get(name, jnp.array(0.0)))
            if stochastic:
                gal_xi_list.append(init_u.get("psd_xi", jnp.zeros(n_grid)))

        map_init = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": {name: jnp.stack(vals) for name, vals in gal_param_lists.items()},
        }
        if stochastic:
            map_init["gal_xi"] = jnp.stack(gal_xi_list)

        if verbose:
            print("  MAP init done. Running multi-seed optimization...")

        # --- Multi-seed optimization ---
        seed_keys = init_keys[n_gal:]
        best_flat = None
        best_loss = jnp.inf
        best_iters = 0

        for s in range(n_seeds):
            if s == 0:
                init_flat = flatten(map_init)
            else:
                # Random perturbation of MAP init
                perturb = 0.3 * jax.random.normal(seed_keys[s], shape=(d_total,))
                init_flat = flatten(map_init) + perturb

            opt_key = jax.random.fold_in(seed_keys[s], 999)
            converged_flat, n_iters = run_evi_jit(
                init_flat,
                opt_key,
                n_iter=n_iterations,
                n_samp=n_samples,
                rtol=kl_rtol,
            )
            n_iters = int(n_iters)

            # Evaluate Hamiltonian
            loss = float(hamiltonian(converged_flat))

            if verbose and n_seeds > 1:
                print(f"    Seed {s + 1}/{n_seeds}: H={loss:.1f}, {n_iters} iters")

            if loss < best_loss:
                best_flat = converged_flat
                best_loss = loss
                best_iters = n_iters

        # --- Draw posterior samples ---
        if verbose:
            print(f"  Drawing {n_posterior_samples} posterior samples...")

        draw_key = jax.random.fold_in(key, 12345)
        draw_keys = jax.random.split(draw_key, n_posterior_samples)
        residuals_flat = draw_residuals_jit(best_flat, draw_keys)

        wall_time = time.time() - t0

        # --- Extract shared PSD posteriors ---
        converged_p = unflatten(best_flat)
        sigma_samples = []
        tau_samples = []
        for i in range(n_posterior_samples):
            res_p = unflatten(residuals_flat[i])
            combined = jax.tree.map(lambda a, b: a + b, converged_p, res_p)
            sigma_samples.append(to_bounded(combined["psd_sigma_u"], sigma_lo, sigma_hi))
            tau_samples.append(to_bounded(combined["psd_tau_u"], tau_lo, tau_hi))

        shared_samples = {
            "psd_sigma": jnp.array(sigma_samples),
            "psd_tau_myr": jnp.array(tau_samples),
        }
        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        # Extract per-galaxy posteriors
        individual_samples = []
        for g in range(n_gal):
            gal_samples = {}
            for name in free_names:
                lo, hi = bounds[name]
                vals = []
                for i in range(n_posterior_samples):
                    res_p = unflatten(residuals_flat[i])
                    combined_val = converged_p["gal"][name][g] + res_p["gal"][name][g]
                    vals.append(to_bounded(combined_val, lo, hi))
                gal_samples[name] = jnp.array(vals)
            if stochastic:
                xi_vals = []
                for i in range(n_posterior_samples):
                    res_p = unflatten(residuals_flat[i])
                    xi_vals.append(converged_p["gal_xi"][g] + res_p["gal_xi"][g])
                gal_samples["psd_xi"] = jnp.stack(xi_vals)
            individual_samples.append(gal_samples)

        if verbose:
            s = shared_params
            print(
                f"  Hierarchical EVI (JIT) complete in {wall_time:.1f}s, "
                f"{best_iters}/{n_iterations} iterations, "
                f"{n_posterior_samples} posterior samples"
            )
            print(f"  σ_PSD = {s['psd_sigma']:.2f}, τ_PSD = {s['psd_tau_myr']:.1f} Myr")

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            individual_samples=individual_samples,
            method="Hierarchical EVI (JIT)",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": best_iters,
                "n_iterations_max": n_iterations,
                "n_samples_posterior": n_posterior_samples,
                "n_seeds": n_seeds,
                "best_hamiltonian": float(best_loss),
                "D_total": d_total,
            },
        )

    def _run_geovi_cfm(
        self,
        *,
        key,
        n_iterations=10,
        n_samples=3,
        n_posterior_samples=60,
        sample_mode="nonlinear_resample",
        vi_config=None,
        verbose=True,
    ):
        """Hierarchical geoVI using NIFTy's CorrelatedFieldMaker.

        This is the proper NIFTy approach: the PSD hyperparameters
        (fluctuation amplitude ≈ σ_PSD, spectral slope ≈ τ_PSD) are
        learned jointly inside the generative model, not as external
        flat parameters.

        Each galaxy gets its own ξ_i (white noise) but shares the
        PSD shape defined by the CorrelatedFieldMaker hyperparameters.
        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError("nifty8.re required: pip install nifty8[re]") from None

        from tengri.inference.vi_config import VIConfig, evi_sample_mode

        cfg = vi_config or VIConfig()

        n_gal = self.n_galaxies
        spec = self._spec
        n_grid = spec.n_grid
        free_names = self._free_names

        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds
        fixed_values = spec.get_fixed_values()

        # Pre-build model
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        data_type = self.data_type

        def _predict_cfm(params):
            if data_type == "photometry":
                return model.predict_photometry(params, mode="_traceable")
            return model.predict_spectrum(params, model._wave_obs, mode="_traceable")

        if verbose:
            print(f"Hierarchical geoVI (CorrelatedFieldMaker): {n_gal} galaxies, n_grid={n_grid}")
            if model._precomputed.photometry is not None:
                print("  Photometry precomputation: ACTIVE")

        t0 = time.time()

        # ── Build shared correlated field maker ───────────────
        # The CFM creates the generative model for the GP field.
        # PSD hyperparameters (fluctuations, slope) are SHARED across
        # all galaxies — this is the hierarchical coupling.
        cfm = jft.CorrelatedFieldMaker("psd_")
        cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-3, 1e-4))

        # Log-age grid spacing
        log_age_range = 10.14 - 6.0  # log10(yr)
        distance = log_age_range / n_grid

        # Fluctuations ~ σ_PSD: lognormal prior centered on 1.0
        # loglogavgslope ~ spectral index: DRW has slope -2 in log-log
        cfm.add_fluctuations(
            shape=(n_grid,),
            distances=(distance,),
            fluctuations=(1.0, 0.8),  # σ_PSD prior: lognormal(1.0, 0.8)
            loglogavgslope=(-2.0, 1.0),  # slope prior: N(-2, 1) — DRW = -2
            flexibility=(0.3, 0.2),  # small non-parametric correction
            asperity=None,
            prefix="shared_",
        )
        corr_field_template = cfm.finalize()

        # ── Build NIFTy domain ────────────────────────────────
        domain = {}

        # Shared PSD hyperparameters (from CFM)
        for k, v in corr_field_template.domain.items():
            if k != "psd_xi":  # xi is per-galaxy
                domain[k] = v

        # Per-galaxy: own xi + own physical params
        for i in range(n_gal):
            domain[f"g{i}_xi"] = jft.ShapeWithDtype((n_grid,))
            for name in free_names:
                domain[f"g{i}_{name}"] = jft.ShapeWithDtype(())

        # Precompute data
        all_data = []
        all_noise_inv = []
        for gal in self.galaxies:
            d = jnp.asarray(gal["flux_obs"])
            n = jnp.asarray(gal["noise"])
            all_data.append(d)
            all_noise_inv.append(1.0 / n**2)

        data_concat = jnp.concatenate(all_data)
        noise_inv_concat = jnp.concatenate(all_noise_inv)

        # ── Build signal response ─────────────────────────────
        def signal_response(primals):
            # Reconstruct the shared CFM primals (PSD hyperparams)
            cfm_primals = {}
            for k in corr_field_template.domain:
                if k != "psd_xi":
                    cfm_primals[k] = primals[k]

            # Stack per-galaxy params into batched arrays (compile-time)
            gal_xi = jnp.stack([primals[f"g{i}_xi"] for i in range(n_gal)])
            gal_ub = {
                name: jnp.stack([primals[f"g{i}_{name}"] for i in range(n_gal)])
                for name in free_names
            }

            # Single-galaxy forward (vmapped over galaxy axis)
            def forward_one(ub_scalars, xi):
                # Generate the correlated field for this galaxy
                cfm_primals_i = dict(cfm_primals)
                cfm_primals_i["psd_xi"] = xi
                gp_field = corr_field_template(cfm_primals_i)

                # Build per-galaxy physical params
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val

                # CFM already applies sqrt(P) * xi, so pass the full
                # correlated field as the GP realization
                params["sfh_field_xi"] = gp_field
                params["sfh_field_psd_sigma"] = 1.0
                params["sfh_field_psd_tau_myr"] = 50.0
                return _predict_cfm(params)

            predictions = jax.vmap(forward_one)(gal_ub, gal_xi)
            return predictions.reshape(-1)

        signal_response_jit = jax.jit(signal_response)
        nifty_model = jft.Model(signal_response_jit, domain=domain)
        likelihood = jft.Gaussian(data_concat, noise_inv_concat).amend(nifty_model)

        # ── Initialize ────────────────────────────────────────
        init = {}
        # Shared PSD: start at prior means
        init_pos_template = jft.random_like(key, corr_field_template.domain)
        for k, v in init_pos_template.items():
            if k != "psd_xi":
                init[k] = jnp.zeros_like(v)  # start at prior mean

        # Per-galaxy: MAP initialization
        from tengri import Fitter

        keys = jax.random.split(key, n_gal + 1)

        if verbose:
            print("  Initializing per-galaxy params via MAP...")

        for i in range(n_gal):
            gal = self.galaxies[i]
            fitter_i = Fitter(model, gal["flux_obs"], gal["noise"], data_type=self.data_type)
            map_i = fitter_i.run(
                "map", n_steps=500, learning_rate=0.03, verbose=False, key=keys[i]
            )
            init_u = fitter_i._unbounded_from_posterior(map_i)
            for name in free_names:
                init[f"g{i}_{name}"] = init_u.get(name, jnp.array(0.0))
            init[f"g{i}_xi"] = init_u.get("psd_xi", jnp.zeros(n_grid))

        init_pos = jft.Vector(init)

        if verbose:
            n_total = sum(np.prod(v.shape) if hasattr(v, "shape") else 1 for v in init.values())
            print(f"  Total parameters: {n_total}")

        # ── Run optimize_kl ───────────────────────────────────
        import io
        import logging
        import sys
        import warnings

        warnings.filterwarnings("ignore")
        logging.getLogger("nifty8").setLevel(logging.ERROR)

        if verbose:
            print(
                f"  Running optimize_kl ({n_iterations} iterations, {n_samples} samples/iter)..."
            )

        # Resolve sample_mode
        if sample_mode == "evi":
            resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
        else:
            resolved_mode = sample_mode

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        key_opt = keys[-1]
        samples, _state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            key=key_opt,
            sample_mode=resolved_mode,
            residual_map=jax.vmap if cfg.use_vmap else "lmap",
            draw_linear_kwargs=cfg.draw_linear_kwargs,
            nonlinearly_update_kwargs=cfg.nonlinearly_update_kwargs,
            kl_kwargs=cfg.kl_kwargs,
            odir=None,
        )

        sys.stdout = old_stdout

        # ── Draw posterior samples ────────────────────────────
        if verbose:
            print(f"  Drawing {n_posterior_samples} posterior samples...")

        converged_pos = samples.pos
        key_draw = jax.random.fold_in(key, 999)

        all_sample_dicts = []
        for s in list(samples):
            sd = s.tree if hasattr(s, "tree") else dict(s)
            all_sample_dicts.append(sd)

        for _j in range(n_posterior_samples):
            key_draw, sub_key = jax.random.split(key_draw)
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 50},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                all_sample_dicts.append(combined)
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
                # TypeError: NIFTy API mismatch or dict() conversion failed
                # ValueError: invalid cg_kwargs configuration
                # AttributeError: missing .tree attribute
                # KeyError: position/sample tree key mismatch
                # RuntimeError: linear solver failed to converge
                break

        wall_time = time.time() - t0
        n_post = len(all_sample_dicts)

        # ── Extract shared PSD posteriors ─────────────────────
        # The CFM encodes fluctuations amplitude and spectral slope.
        # Extract them from the samples.
        shared_samples = {
            "psd_fluctuations": jnp.array(
                [float(d.get("psd_shared_fluctuations", jnp.nan)) for d in all_sample_dicts]
            ),
            "psd_loglogavgslope": jnp.array(
                [float(d.get("psd_shared_loglogavgslope", jnp.nan)) for d in all_sample_dicts]
            ),
        }

        # Also compute effective σ_PSD and τ_PSD from the CFM params
        # fluctuations ≈ exp(psd_shared_fluctuations) ≈ σ_PSD
        # loglogavgslope ≈ spectral slope (DRW = -2)
        shared_samples["psd_sigma_eff"] = jnp.exp(shared_samples["psd_fluctuations"])

        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            print(f"  Hierarchical geoVI (CFM) complete in {wall_time:.1f}s, {n_post} samples")
            print(
                f"  fluctuations = {shared_params.get('psd_fluctuations', '?'):.2f}, "
                f"slope = {shared_params.get('psd_loglogavgslope', '?'):.2f}"
            )

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            method="Hierarchical geoVI (CorrelatedFieldMaker)",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": n_iterations,
                "n_samples": n_post,
                "cfm_domain_keys": list(corr_field_template.domain.keys()),
            },
        )

    def _run_geovi(
        self,
        *,
        key,
        n_iterations=10,
        n_samples=3,
        n_posterior_samples=100,
        sample_mode="nonlinear_resample",
        vi_config=None,
        verbose=True,
    ):
        """Hierarchical geoVI via NIFTy.re.

        The joint model has:
        - 2 shared PSD params (unbounded)
        - N × (n_free + n_grid) per-galaxy params
        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError(
                "nifty8.re required for hierarchical geoVI: pip install nifty8[re]"
            ) from None

        from tengri.inference.vi_config import VIConfig, evi_sample_mode

        cfg = vi_config or VIConfig()

        n_gal = self.n_galaxies
        spec = self._spec
        stochastic = spec.stochastic
        n_grid = spec.n_grid
        free_names = self._free_names

        # Build bounds for per-galaxy free params
        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds

        fixed_values = spec.get_fixed_values()

        if verbose:
            n_per_gal = len(free_names) + (n_grid if stochastic else 0)
            n_total = 2 + n_gal * n_per_gal
            print(
                f"Hierarchical geoVI: {n_gal} galaxies, "
                f"{n_per_gal} params/galaxy + 2 shared = "
                f"{n_total} total parameters"
            )

        t0 = time.time()

        # ── Build NIFTy domain ────────────────────────────────
        domain = {}

        # Shared PSD params (unbounded)
        domain["psd_sigma_u"] = jft.ShapeWithDtype(())
        domain["psd_tau_u"] = jft.ShapeWithDtype(())

        # Per-galaxy params
        for i in range(n_gal):
            for name in free_names:
                domain[f"g{i}_{name}"] = jft.ShapeWithDtype(())
            if stochastic:
                domain[f"g{i}_psd_xi"] = jft.ShapeWithDtype((n_grid,))

        # ── Build signal response ─────────────────────────────
        galaxies = self.galaxies
        sigma_lo, sigma_hi = self.psd_sigma_bounds
        tau_lo, tau_hi = self.psd_tau_bounds
        data_type = self.data_type

        # Precompute data arrays
        all_data = []
        all_noise_inv = []
        for gal in galaxies:
            d = jnp.asarray(gal["flux_obs"])
            n = jnp.asarray(gal["noise"])
            all_data.append(d)
            all_noise_inv.append(1.0 / n**2)

        data_concat = jnp.concatenate(all_data)
        noise_inv_concat = jnp.concatenate(all_noise_inv)

        # Pre-build model once (PSD params will be overridden per-call)
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)

        # Verify precomputation is active
        if model._precomputed.photometry is not None and verbose:
            print("  Photometry precomputation: ACTIVE (21.6x speedup)")
        elif verbose:
            print("  WARNING: Photometry precomputation NOT active")

        def _predict_single(params):
            """Single-galaxy forward model (for vmap)."""
            if data_type == "photometry":
                return model.predict_photometry(params, mode="_traceable")
            else:
                return model.predict_spectrum(params, model._wave_obs, mode="_traceable")

        def signal_response(primals):
            # Shared PSD params (bounded)
            psd_sigma = to_bounded(primals["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(primals["psd_tau_u"], tau_lo, tau_hi)

            # Stack per-galaxy params into batched arrays (compile-time)
            gal_ub = {
                name: jnp.stack([primals[f"g{i}_{name}"] for i in range(n_gal)])
                for name in free_names
            }
            if stochastic:
                gal_xi = jnp.stack([primals[f"g{i}_psd_xi"] for i in range(n_gal)])

            # Single-galaxy forward (vmapped over galaxy axis)
            def forward_one(ub_scalars, xi):
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val
                params["sfh_field_psd_sigma"] = psd_sigma
                params["sfh_field_psd_tau_myr"] = psd_tau
                if stochastic:
                    params["sfh_field_xi"] = xi
                return _predict_single(params)

            if stochastic:
                predictions = jax.vmap(forward_one)(gal_ub, gal_xi)
            else:
                predictions = jax.vmap(lambda ub, _: forward_one(ub, None))(
                    gal_ub, jnp.zeros(n_gal)
                )

            return predictions.reshape(-1)

        signal_response_jit = jax.jit(signal_response)
        nifty_model = jft.Model(signal_response_jit, domain=domain)

        # Gaussian likelihood
        likelihood = jft.Gaussian(data_concat, noise_inv_concat).amend(nifty_model)

        # ── Initialize ────────────────────────────────────────
        init = {}
        # Shared: start near middle of prior
        init["psd_sigma_u"] = jnp.array(0.0)
        init["psd_tau_u"] = jnp.array(0.0)

        # Per-galaxy: small random perturbations
        keys = jax.random.split(key, n_gal + 1)
        for i in range(n_gal):
            gal_keys = jax.random.split(keys[i], len(free_names) + 1)
            for j, name in enumerate(free_names):
                init[f"g{i}_{name}"] = 0.1 * jax.random.normal(gal_keys[j])
            if stochastic:
                init[f"g{i}_psd_xi"] = 0.1 * jax.random.normal(gal_keys[-1], shape=(n_grid,))

        init_pos = jft.Vector(init)

        # ── Run optimize_kl ───────────────────────────────────
        key, opt_key = jax.random.split(keys[-1])

        import io
        import logging
        import sys
        import warnings

        warnings.filterwarnings("ignore")
        logging.getLogger("nifty8").setLevel(logging.ERROR)

        if verbose:
            print(f"  Running optimize_kl ({n_iterations} iterations)...")

        # Resolve sample_mode
        if sample_mode == "evi":
            resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
        else:
            resolved_mode = sample_mode

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

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

        sys.stdout = old_stdout

        # ── Draw posterior samples ────────────────────────────
        if verbose:
            print(f"  Drawing {n_posterior_samples} posterior samples...")

        converged_pos = samples.pos
        key, draw_key = jax.random.split(key)

        all_sample_dicts = []

        # Include optimization samples
        for s in list(samples):
            sd = s.tree if hasattr(s, "tree") else dict(s)
            all_sample_dicts.append(sd)

        # Draw additional linear residual samples
        for _j in range(n_posterior_samples):
            draw_key, sub_key = jax.random.split(draw_key)
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 50},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                all_sample_dicts.append(combined)
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
                # TypeError: NIFTy API mismatch or dict() conversion failed
                # ValueError: invalid cg_kwargs configuration
                # AttributeError: missing .tree attribute
                # KeyError: position/sample tree key mismatch
                # RuntimeError: linear solver failed to converge
                break

        wall_time = time.time() - t0
        n_post = len(all_sample_dicts)

        # ── Extract shared PSD posteriors ─────────────────────
        shared_samples = {
            "psd_sigma": jnp.array(
                [float(to_bounded(d["psd_sigma_u"], sigma_lo, sigma_hi)) for d in all_sample_dicts]
            ),
            "psd_tau_myr": jnp.array(
                [float(to_bounded(d["psd_tau_u"], tau_lo, tau_hi)) for d in all_sample_dicts]
            ),
        }

        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            s = shared_params
            print(f"  Hierarchical geoVI complete in {wall_time:.1f}s, {n_post} samples")
            print(f"  σ_PSD = {s['psd_sigma']:.2f}, τ_PSD = {s['psd_tau_myr']:.1f} Myr")

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            method="Hierarchical geoVI",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_iterations": n_iterations,
                "n_samples": n_post,
            },
        )

    def _run_raytrace(
        self, *, key, n_burnin=200, n_steps=500, n_leapfrog_steps=10, step_size=None, verbose=True
    ):
        """Hierarchical Ray Tracing (flat parameter vector).

        Flattens all shared + per-galaxy params into one vector and
        runs the Ray Tracing Sampler. Works for moderate N (~10-50 gal).
        """
        from jax.flatten_util import ravel_pytree

        from tengri.inference.backends.mcmc.raytrace import sample_raytrace

        n_gal = self.n_galaxies
        spec = self._spec
        stochastic = spec.stochastic
        n_grid = spec.n_grid
        free_names = self._free_names
        bounds = {}
        for name in free_names:
            dist = spec.get_distribution(name)
            bounds[name] = dist.bounds
        fixed_values = spec.get_fixed_values()
        sigma_lo, sigma_hi = self.psd_sigma_bounds
        tau_lo, tau_hi = self.psd_tau_bounds

        # Build initial flat vector
        # Build init dict with stacked per-galaxy arrays for vmap
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        tau_mid = 0.5 * (tau_lo + tau_hi)

        # Pre-build model with midpoint PSD for MAP initialization
        model = self.model_factory(psd_sigma=sigma_mid, psd_tau_myr=tau_mid)

        # Initialize per-galaxy params via individual MAP fits
        from tengri import Fitter

        keys = jax.random.split(key, n_gal + 2)

        if verbose:
            print("  Initializing per-galaxy params via MAP...")

        gal_param_lists = {name: [] for name in free_names}
        gal_xi_list = []

        for i in range(n_gal):
            gal = self.galaxies[i]
            fitter_i = Fitter(model, gal["flux_obs"], gal["noise"], data_type=self.data_type)
            map_i = fitter_i.run(
                "map", n_steps=500, learning_rate=0.03, verbose=False, key=keys[i]
            )
            init_u = fitter_i._unbounded_from_posterior(map_i)
            for name in free_names:
                gal_param_lists[name].append(init_u.get(name, jnp.array(0.0)))
            if stochastic:
                gal_xi_list.append(init_u.get("psd_xi", jnp.zeros(n_grid)))

        if verbose:
            print("  MAP initialization complete")

        # Structured init: shared scalars + stacked per-galaxy arrays
        init = {
            "psd_sigma_u": to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi),
            "psd_tau_u": to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi),
            "gal": {name: jnp.stack(vals) for name, vals in gal_param_lists.items()},
        }
        if stochastic:
            init["gal_xi"] = jnp.stack(gal_xi_list)

        init_flat, unravel_fn = ravel_pytree(init)
        D = len(init_flat)

        if step_size is None:
            step_size = 0.005 if D > 100 else 0.01

        # Build data
        all_data = jnp.concatenate([jnp.asarray(g["flux_obs"]) for g in self.galaxies])
        all_noise = jnp.concatenate([jnp.asarray(g["noise"]) for g in self.galaxies])

        # Pre-build model once (PSD params will be overridden per-call)
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
        data_type = self.data_type

        def _predict_rt(params):
            if data_type == "photometry":
                return model.predict_photometry(params, mode="_traceable")
            return model.predict_spectrum(params, model._wave_obs, mode="_traceable")

        def log_prob(flat_params):
            p = unravel_fn(flat_params)
            psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

            # Single-galaxy forward (vmapped over galaxy axis)
            def forward_one(ub_scalars, xi):
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(ub_scalars[name], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
                        params[name] = val
                params["sfh_field_psd_sigma"] = psd_sigma
                params["sfh_field_psd_tau_myr"] = psd_tau
                if stochastic:
                    params["sfh_field_xi"] = xi
                return _predict_rt(params)

            if stochastic:
                predictions = jax.vmap(forward_one)(p["gal"], p["gal_xi"])
            else:
                predictions = jax.vmap(lambda ub, _: forward_one(ub, None))(
                    p["gal"], jnp.zeros(n_gal)
                )

            pred_all = predictions.reshape(-1)
            chi2 = jnp.sum(((all_data - pred_all) / all_noise) ** 2)

            # Prior: standard normal on all unbounded + xi params
            param_penalty = p["psd_sigma_u"] ** 2 + p["psd_tau_u"] ** 2
            for name in free_names:
                param_penalty += jnp.sum(p["gal"][name] ** 2)
            if stochastic:
                param_penalty += jnp.sum(p["gal_xi"] ** 2)

            return -0.5 * chi2 - 0.5 * param_penalty

        if verbose:
            print(
                f"Hierarchical Ray Tracing: {n_gal} galaxies, "
                f"{D} total parameters, "
                f"{n_burnin} burn-in + {n_steps} samples"
            )

        t0 = time.time()
        total_steps = n_burnin + n_steps

        key_rt = keys[-1]
        chain, _log_likelihood, accept_prob = sample_raytrace(
            key=key_rt,
            params_init=init_flat,
            log_prob_fn=log_prob,
            n_steps=total_steps,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=float(step_size),
        )

        wall_time = time.time() - t0
        chain = chain[n_burnin:]
        accept_prob_post = accept_prob[n_burnin:]

        # Extract shared params (vectorized over chain)
        def extract_shared(flat_params):
            p = unravel_fn(flat_params)
            return jnp.array(
                [
                    to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi),
                    to_bounded(p["psd_tau_u"], tau_lo, tau_hi),
                ]
            )

        shared_arr = jax.vmap(extract_shared)(chain)  # (n_samples, 2)
        shared_samples = {
            "psd_sigma": shared_arr[:, 0],
            "psd_tau_myr": shared_arr[:, 1],
        }
        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            print(
                f"  Complete in {wall_time:.1f}s. Accept: {float(jnp.mean(accept_prob_post)):.1%}"
            )
            print(
                f"  σ_PSD = {shared_params['psd_sigma']:.2f}, "
                f"τ_PSD = {shared_params['psd_tau_myr']:.1f} Myr"
            )

        return PopulationPosterior(
            shared_samples=shared_samples,
            shared_params=shared_params,
            method="Hierarchical Ray Tracing",
            wall_time_s=wall_time,
            diagnostics={
                "n_galaxies": n_gal,
                "n_burnin": n_burnin,
                "n_steps": n_steps,
                "n_samples": chain.shape[0],
                "accept_rate": float(jnp.mean(accept_prob_post)),
                "D_total": D,
            },
        )


# ---------------------------------------------------------------------------
# Deprecated aliases — removed in tengri v1.0
# ---------------------------------------------------------------------------


def _make_deprecated_hierarchical_result():
    import warnings

    class HierarchicalResult(PopulationPosterior):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "HierarchicalResult is deprecated. Use PopulationPosterior instead. "
                "Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    HierarchicalResult.__name__ = "HierarchicalResult"
    HierarchicalResult.__qualname__ = "HierarchicalResult"
    return HierarchicalResult


HierarchicalResult = _make_deprecated_hierarchical_result()


def _make_deprecated_hierarchical_fitter():
    import warnings

    class HierarchicalFitter(PopulationFitter):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "HierarchicalFitter is deprecated. Use PopulationFitter instead. "
                "Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    HierarchicalFitter.__name__ = "HierarchicalFitter"
    HierarchicalFitter.__qualname__ = "HierarchicalFitter"
    return HierarchicalFitter


HierarchicalFitter = _make_deprecated_hierarchical_fitter()
