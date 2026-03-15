"""Hierarchical inference for population-level PSD recovery.

Shares PSD hyperparameters (σ_PSD, τ_PSD) across N galaxies while
each galaxy retains its own latent field ξ_i and physical parameters.

The total parameter vector is:
    Θ = {φ_shared, {ξ_i, θ_i}_{i=1}^N}

where φ_shared = {σ_PSD, τ_PSD} (or more generally, the PSD shape).

Usage:
    hfitter = HierarchicalFitter(model_template, galaxies)
    result = hfitter.run("geovi", n_iterations=25)
    result.shared_params  # posterior on (σ_PSD, τ_PSD)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from diffsed.distributions import Uniform
from diffsed.utils.transforms import to_bounded, to_unbounded


@dataclass
class HierarchicalResult:
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
    individual_samples: Optional[list] = None
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
            f"HierarchicalResult(method='{self.method}', "
            f"n_samples={n}, "
            f"wall_time={self.wall_time_s:.1f}s)"
        )


class HierarchicalFitter:
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

    def __init__(self, model_factory, galaxies,
                 psd_sigma_prior=(0.1, 4.0),
                 psd_tau_prior=(1.0, 300.0),
                 data_type="photometry"):
        self.model_factory = model_factory
        self.galaxies = galaxies
        self.n_galaxies = len(galaxies)
        self.psd_sigma_bounds = psd_sigma_prior
        self.psd_tau_bounds = psd_tau_prior
        self.data_type = data_type

        # Create a template model to get spec info
        self._template = model_factory(
            psd_sigma=1.0, psd_tau_myr=50.0
        )
        self._spec = self._template.spec
        self._free_names = [n for n in self._spec.free_params
                            if n not in ("psd_sigma", "psd_tau_myr")]

    def run(self, method="geovi", *, key=None, **kwargs):
        """Run hierarchical inference.

        Parameters
        ----------
        method : str
            "geovi" — geoVI with CorrelatedFieldMaker for native PSD learning.
            "mgvi" — MGVI (faster per iteration, for very large N).
            "geovi_flat" — flat parameter vector (legacy approach).
            "raytrace" — Ray Tracing on flat vector (fast but needs tuning).
        key : PRNGKey
        **kwargs
            Passed to the inference method.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if method == "geovi":
            return self._run_geovi_cfm(key=key, **kwargs)
        elif method == "mgvi":
            return self._run_geovi_cfm(
                key=key, sample_mode="linear_resample", **kwargs,
            )
        elif method == "geovi_flat":
            return self._run_geovi(key=key, **kwargs)
        elif method == "raytrace":
            return self._run_raytrace(key=key, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Use 'geovi', 'mgvi', 'geovi_flat', or 'raytrace'."
            )

    def _run_geovi_cfm(self, *, key, n_iterations=20, n_samples=4,
                       n_posterior_samples=60,
                       sample_mode="nonlinear_resample",
                       verbose=True):
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
            raise ImportError("nifty8.re required: pip install nifty8[re]")

        from diffsed.posterior import Posterior

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

        # Pre-build model
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)

        if verbose:
            print(f"Hierarchical geoVI (CorrelatedFieldMaker): "
                  f"{n_gal} galaxies, n_grid={n_grid}")
            if model._precomp is not None:
                print("  Photometry precomputation: ACTIVE")

        t0 = time.time()

        # ── Build shared correlated field maker ───────────────
        # The CFM creates the generative model for the GP field.
        # PSD hyperparameters (fluctuations, slope) are SHARED across
        # all galaxies — this is the hierarchical coupling.
        cfm = jft.CorrelatedFieldMaker("psd_")
        cfm.set_amplitude_total_offset(
            offset_mean=0.0, offset_std=(1e-3, 1e-4)
        )

        # Log-age grid spacing
        log_age_range = 10.14 - 6.0  # log10(yr)
        distance = log_age_range / n_grid

        # Fluctuations ~ σ_PSD: lognormal prior centered on 1.0
        # loglogavgslope ~ spectral index: DRW has slope -2 in log-log
        cfm.add_fluctuations(
            shape=(n_grid,),
            distances=(distance,),
            fluctuations=(1.0, 0.8),       # σ_PSD prior: lognormal(1.0, 0.8)
            loglogavgslope=(-2.0, 1.0),    # slope prior: N(-2, 1) — DRW = -2
            flexibility=(0.3, 0.2),        # small non-parametric correction
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

            predictions = []
            for i in range(n_gal):
                # Use shared PSD + per-galaxy xi
                cfm_primals_i = dict(cfm_primals)
                cfm_primals_i["psd_xi"] = primals[f"g{i}_xi"]

                # Generate the correlated field (GP realization)
                gp_field = corr_field_template(cfm_primals_i)

                # Build per-galaxy physical params
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(
                        primals[f"g{i}_{name}"], lo, hi
                    )
                for name, val in fixed_values.items():
                    if name not in ("psd_sigma", "psd_tau_myr"):
                        params[name] = val

                # Override: use CFM-generated field as psd_xi
                # The CFM already applies sqrt(P) * xi, so we pass
                # the full correlated field as the GP realization
                params["psd_xi"] = gp_field
                # Set dummy PSD params (not used when psd_xi is
                # already a correlated field, but Model expects them)
                params["psd_sigma"] = 1.0
                params["psd_tau_myr"] = 50.0

                pred = model.predict_photometry(params)
                predictions.append(pred)

            return jnp.concatenate(predictions)

        nifty_model = jft.Model(signal_response, domain=domain)
        likelihood = jft.Gaussian(
            data_concat, noise_inv_concat
        ).amend(nifty_model)

        # ── Initialize ────────────────────────────────────────
        init = {}
        # Shared PSD: start at prior means
        init_pos_template = jft.random_like(key, corr_field_template.domain)
        for k, v in init_pos_template.items():
            if k != "psd_xi":
                init[k] = jnp.zeros_like(v)  # start at prior mean

        # Per-galaxy: MAP initialization
        from diffsed import Fitter
        keys = jax.random.split(key, n_gal + 1)

        if verbose:
            print("  Initializing per-galaxy params via MAP...")

        for i in range(n_gal):
            gal = self.galaxies[i]
            fitter_i = Fitter(model, gal["flux_obs"], gal["noise"])
            map_i = fitter_i.run("map", n_steps=500, learning_rate=0.03,
                                 verbose=False, key=keys[i])
            init_u = fitter_i._unbounded_from_posterior(map_i)
            for name in free_names:
                init[f"g{i}_{name}"] = init_u.get(name, jnp.array(0.0))
            init[f"g{i}_xi"] = init_u.get("psd_xi", jnp.zeros(n_grid))

        init_pos = jft.Vector(init)

        if verbose:
            n_total = sum(
                np.prod(v.shape) if hasattr(v, 'shape') else 1
                for v in init.values()
            )
            print(f"  Total parameters: {n_total}")

        # ── Run optimize_kl ───────────────────────────────────
        import sys, io, warnings, logging
        warnings.filterwarnings('ignore')
        logging.getLogger('nifty8').setLevel(logging.ERROR)

        if verbose:
            print(f"  Running optimize_kl ({n_iterations} iterations, "
                  f"{n_samples} samples/iter)...")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        key_opt = keys[-1]
        delta = max(1, n_samples - 1)
        mode_label = "geoVI" if sample_mode == "nonlinear_resample" else "MGVI"
        samples, state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=lambda i: max(1, 1 + int(i * delta / max(n_iterations - 1, 1))),
            key=key_opt,
            sample_mode=sample_mode,
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
            sd = s.tree if hasattr(s, 'tree') else dict(s)
            all_sample_dicts.append(sd)

        for j in range(n_posterior_samples):
            key_draw, sub_key = jax.random.split(key_draw)
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood, converged_pos, sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 50},
                )
                sample_tree = residual.tree if hasattr(residual, 'tree') else dict(residual)
                pos_tree = converged_pos.tree if hasattr(converged_pos, 'tree') else dict(converged_pos)
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                all_sample_dicts.append(combined)
            except Exception:
                break

        wall_time = time.time() - t0
        n_post = len(all_sample_dicts)

        # ── Extract shared PSD posteriors ─────────────────────
        # The CFM encodes fluctuations amplitude and spectral slope.
        # Extract them from the samples.
        shared_samples = {
            "psd_fluctuations": jnp.array([
                float(d.get("psd_shared_fluctuations", jnp.nan))
                for d in all_sample_dicts
            ]),
            "psd_loglogavgslope": jnp.array([
                float(d.get("psd_shared_loglogavgslope", jnp.nan))
                for d in all_sample_dicts
            ]),
        }

        # Also compute effective σ_PSD and τ_PSD from the CFM params
        # fluctuations ≈ exp(psd_shared_fluctuations) ≈ σ_PSD
        # loglogavgslope ≈ spectral slope (DRW = -2)
        shared_samples["psd_sigma_eff"] = jnp.exp(
            shared_samples["psd_fluctuations"]
        )

        shared_params = {
            k: float(jnp.mean(v)) for k, v in shared_samples.items()
        }

        if verbose:
            print(f"  Hierarchical geoVI (CFM) complete in {wall_time:.1f}s, "
                  f"{n_post} samples")
            print(f"  fluctuations = {shared_params.get('psd_fluctuations', '?'):.2f}, "
                  f"slope = {shared_params.get('psd_loglogavgslope', '?'):.2f}")

        return HierarchicalResult(
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

    def _run_geovi(self, *, key, n_iterations=25, n_samples=6,
                   n_posterior_samples=100, verbose=True):
        """Hierarchical geoVI via NIFTy.re.

        The joint model has:
        - 2 shared PSD params (unbounded)
        - N × (n_free + n_grid) per-galaxy params
        """
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError(
                "nifty8.re required for hierarchical geoVI: "
                "pip install nifty8[re]"
            )

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
            print(f"Hierarchical geoVI: {n_gal} galaxies, "
                  f"{n_per_gal} params/galaxy + 2 shared = "
                  f"{n_total} total parameters")

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
        model_factory = self.model_factory
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
        n_data_per_gal = len(all_data[0])

        # Pre-build model once (PSD params will be overridden per-call)
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)

        # Verify precomputation is active
        if model._precomp is not None and verbose:
            print("  Photometry precomputation: ACTIVE (21.6x speedup)")
        elif verbose:
            print("  WARNING: Photometry precomputation NOT active")

        def _predict_single(params):
            """Single-galaxy forward model (for vmap)."""
            if data_type == "photometry":
                return model.predict_photometry(params)
            else:
                return model.predict_spectrum(params, model._wave_obs)

        def signal_response(primals):
            # Shared PSD params (bounded)
            psd_sigma = to_bounded(primals["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(primals["psd_tau_u"], tau_lo, tau_hi)

            # Build per-galaxy params and predict
            # Use jax.lax.scan for efficient sequential evaluation
            # (vmap not possible because each galaxy has different xi)
            def scan_body(carry, i):
                predictions = []
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(
                        primals[f"g{i}_{name}"], lo, hi
                    )
                for name, val in fixed_values.items():
                    if name not in ("psd_sigma", "psd_tau_myr"):
                        params[name] = val
                params["psd_sigma"] = psd_sigma
                params["psd_tau_myr"] = psd_tau
                if stochastic:
                    params["psd_xi"] = primals[f"g{i}_psd_xi"]
                return None, _predict_single(params)

            # Can't use scan with string-indexed primals, use loop
            # but at least each call uses precomputed photometry
            predictions = []
            for i in range(n_gal):
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(
                        primals[f"g{i}_{name}"], lo, hi
                    )
                for name, val in fixed_values.items():
                    if name not in ("psd_sigma", "psd_tau_myr"):
                        params[name] = val
                params["psd_sigma"] = psd_sigma
                params["psd_tau_myr"] = psd_tau
                if stochastic:
                    params["psd_xi"] = primals[f"g{i}_psd_xi"]
                predictions.append(_predict_single(params))

            return jnp.concatenate(predictions)

        nifty_model = jft.Model(signal_response, domain=domain)

        # Gaussian likelihood
        likelihood = jft.Gaussian(
            data_concat, noise_inv_concat
        ).amend(nifty_model)

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
                init[f"g{i}_psd_xi"] = 0.1 * jax.random.normal(
                    gal_keys[-1], shape=(n_grid,)
                )

        init_pos = jft.Vector(init)

        # ── Run optimize_kl ───────────────────────────────────
        key, opt_key = jax.random.split(keys[-1])
        delta = max(1, n_samples - 1)

        import sys, io, warnings, logging
        warnings.filterwarnings('ignore')
        logging.getLogger('nifty8').setLevel(logging.ERROR)

        if verbose:
            print(f"  Running optimize_kl ({n_iterations} iterations)...")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        samples, state = jft.optimize_kl(
            likelihood,
            init_pos,
            n_total_iterations=n_iterations,
            n_samples=lambda i: max(1, 1 + int(i * delta / max(n_iterations - 1, 1))),
            key=opt_key,
            sample_mode="nonlinear_resample",
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
            sd = s.tree if hasattr(s, 'tree') else dict(s)
            all_sample_dicts.append(sd)

        # Draw additional linear residual samples
        for j in range(n_posterior_samples):
            draw_key, sub_key = jax.random.split(draw_key)
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood, converged_pos, sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 50},
                )
                sample_tree = residual.tree if hasattr(residual, 'tree') else dict(residual)
                pos_tree = converged_pos.tree if hasattr(converged_pos, 'tree') else dict(converged_pos)
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                all_sample_dicts.append(combined)
            except Exception:
                break

        wall_time = time.time() - t0
        n_post = len(all_sample_dicts)

        # ── Extract shared PSD posteriors ─────────────────────
        shared_samples = {
            "psd_sigma": jnp.array([
                float(to_bounded(d["psd_sigma_u"], sigma_lo, sigma_hi))
                for d in all_sample_dicts
            ]),
            "psd_tau_myr": jnp.array([
                float(to_bounded(d["psd_tau_u"], tau_lo, tau_hi))
                for d in all_sample_dicts
            ]),
        }

        shared_params = {
            k: float(jnp.mean(v)) for k, v in shared_samples.items()
        }

        if verbose:
            s = shared_params
            print(f"  Hierarchical geoVI complete in {wall_time:.1f}s, "
                  f"{n_post} samples")
            print(f"  σ_PSD = {s['psd_sigma']:.2f}, "
                  f"τ_PSD = {s['psd_tau_myr']:.1f} Myr")

        return HierarchicalResult(
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

    def _run_raytrace(self, *, key, n_burnin=200, n_steps=500,
                      n_leapfrog_steps=10, step_size=None,
                      verbose=True):
        """Hierarchical Ray Tracing (flat parameter vector).

        Flattens all shared + per-galaxy params into one vector and
        runs the Ray Tracing Sampler. Works for moderate N (~10-50 gal).
        """
        from diffsed.raytrace_jax import sample_raytrace
        from jax.flatten_util import ravel_pytree

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
        init = {}
        # Initialize shared PSD near center of prior (in bounded space)
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        tau_mid = 0.5 * (tau_lo + tau_hi)
        init["psd_sigma_u"] = to_unbounded(jnp.array(sigma_mid), sigma_lo, sigma_hi)
        init["psd_tau_u"] = to_unbounded(jnp.array(tau_mid), tau_lo, tau_hi)

        # Pre-build model with midpoint PSD for MAP initialization
        model = self.model_factory(psd_sigma=sigma_mid, psd_tau_myr=tau_mid)

        # Initialize per-galaxy params via individual MAP fits
        from diffsed import Fitter
        keys = jax.random.split(key, n_gal + 2)

        if verbose:
            print("  Initializing per-galaxy params via MAP...")

        for i in range(n_gal):
            gal = self.galaxies[i]
            fitter_i = Fitter(model, gal["flux_obs"], gal["noise"])
            map_i = fitter_i.run("map", n_steps=500, learning_rate=0.03,
                                 verbose=False, key=keys[i])
            init_u = fitter_i._unbounded_from_posterior(map_i)
            for name in free_names:
                if name in init_u:
                    init[f"g{i}_{name}"] = init_u[name]
                else:
                    init[f"g{i}_{name}"] = jnp.array(0.0)
            if stochastic:
                if "psd_xi" in init_u:
                    init[f"g{i}_psd_xi"] = init_u["psd_xi"]
                else:
                    init[f"g{i}_psd_xi"] = jnp.zeros(n_grid)

        if verbose:
            print("  MAP initialization complete")

        init_flat, unravel_fn = ravel_pytree(init)
        D = len(init_flat)

        if step_size is None:
            step_size = 0.005 if D > 100 else 0.01

        # Build data
        all_data = jnp.concatenate([
            jnp.asarray(g["flux_obs"]) for g in self.galaxies
        ])
        all_noise = jnp.concatenate([
            jnp.asarray(g["noise"]) for g in self.galaxies
        ])

        # Pre-build model once (PSD params will be overridden per-call)
        model = self.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)

        def log_prob(flat_params):
            p = unravel_fn(flat_params)
            psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
            psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

            predictions = []
            xi_penalty = 0.0

            for i in range(n_gal):
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(p[f"g{i}_{name}"], lo, hi)
                for name, val in fixed_values.items():
                    if name not in ("psd_sigma", "psd_tau_myr"):
                        params[name] = val
                # Override PSD with shared values
                params["psd_sigma"] = psd_sigma
                params["psd_tau_myr"] = psd_tau
                if stochastic:
                    params["psd_xi"] = p[f"g{i}_psd_xi"]
                    xi_penalty += jnp.sum(p[f"g{i}_psd_xi"]**2)

                pred = model.predict_photometry(params)
                predictions.append(pred)

            pred_all = jnp.concatenate(predictions)
            chi2 = jnp.sum(((all_data - pred_all) / all_noise)**2)

            # Prior: standard normal on all unbounded params + xi
            param_penalty = (
                p["psd_sigma_u"]**2 + p["psd_tau_u"]**2 + xi_penalty
            )
            for i in range(n_gal):
                for name in free_names:
                    param_penalty += p[f"g{i}_{name}"]**2

            return -0.5 * chi2 - 0.5 * param_penalty

        if verbose:
            print(f"Hierarchical Ray Tracing: {n_gal} galaxies, "
                  f"{D} total parameters, "
                  f"{n_burnin} burn-in + {n_steps} samples")

        t0 = time.time()
        total_steps = n_burnin + n_steps

        key_rt = keys[-1]
        chain, log_likelihood, accept_prob = sample_raytrace(
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

        # Extract shared params
        shared_samples = {"psd_sigma": [], "psd_tau_myr": []}
        for i in range(chain.shape[0]):
            p = unravel_fn(chain[i])
            shared_samples["psd_sigma"].append(
                float(to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi))
            )
            shared_samples["psd_tau_myr"].append(
                float(to_bounded(p["psd_tau_u"], tau_lo, tau_hi))
            )

        shared_samples = {k: jnp.array(v) for k, v in shared_samples.items()}
        shared_params = {k: float(jnp.mean(v)) for k, v in shared_samples.items()}

        if verbose:
            print(f"  Complete in {wall_time:.1f}s. "
                  f"Accept: {float(jnp.mean(accept_prob_post)):.1%}")
            print(f"  σ_PSD = {shared_params['psd_sigma']:.2f}, "
                  f"τ_PSD = {shared_params['psd_tau_myr']:.1f} Myr")

        return HierarchicalResult(
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
