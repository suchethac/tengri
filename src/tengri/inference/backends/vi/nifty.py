# SPDX-License-Identifier: BSD-3-Clause
"""geoVI-preconditioned NUTS: exact MCMC in geoVI-flattened coordinates.

Uses geoVI's nonlinear coordinate transform g(ξ; m*) to precondition
NUTS sampling. The transform straightens banana-shaped degeneracies
(e.g., age-dust-metallicity) so NUTS can sample with an identity mass
matrix and short trajectories.

Algorithm:
    1. Run geoVI → expansion point m* and transform primitives
    2. Define η = g(ξ; m*) where posterior ≈ N(0, I) in η-space
    3. Run BlackJAX NUTS in η-space with identity mass matrix
    4. Back-transform: ξ = g⁻¹(η; m*)

Reference:
    Variant B of "Fisher-Informed NUTS" (Riemannian HMC with geoVI metric).
    Combines geoVI's geometric insight with NUTS's exactness guarantee.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.inference._model_cache import _default_owner as _model_cache_owner
from tengri.inference._sample_utils import _mean_params
from tengri.inference.likelihoods.gaussian import diag_noise_operators, standardized_residual


def run_nifty_fast_vi(
    context,
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
        ``"jit"`` (default), fast JIT CG sampling.
        ``"blackjax"``, independent NUTS sampling.
    verbose : bool
        Print progress.
    """
    import time

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior
    from tengri.inference.vi_config import VIConfig, evi_sample_mode

    context = InferenceContext.from_target(context)
    # VI backends still read most state directly off the Fitter
    # (``_jit_sampler``, ``_native_vi_nonlinear_engine``, ``_draw_*``,
    # ``data``, ``noise``, ``data_type``, ``_bounds``, ``_fixed_values``).
    # Those caches must outlive any single ``run()`` call, keep them
    # on the Fitter and reach through ``context.fitter`` until they
    # migrate in a follow-up.
    fitter = context.fitter

    try:
        import nifty8.re as jft
    except ImportError:
        return run_nifty_vi(
            context,
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

    cfg = vi_config or VIConfig()

    init_params = context.initial_params(key, init_from=init_from)

    # Resolve sample mode.
    # For geoVI: use periodic resample + update schedule (prevents
    # sample staleness while maintaining stable convergence).
    resample_every = 5
    if sample_mode == "evi":
        resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
    elif sample_mode == "nonlinear_resample":

        def resolved_mode(i: int) -> str:
            """Schedule nonlinear_resample periodically, nonlinear_update otherwise."""
            if i == 0 or i % resample_every == 0:
                return "nonlinear_resample"
            return "nonlinear_update"

    else:
        resolved_mode = sample_mode

    n_total = len(fitter._free_names) + (fitter.spec.n_grid if fitter.spec.stochastic else 0)
    _mode_labels = {
        "nonlinear_resample": "geovi",
        "linear_resample": "mgvi",
        "evi": "evi",
    }
    if verbose:
        mode_label = _mode_labels.get(sample_mode, sample_mode)
        print(
            f"{mode_label}: {n_total} params, {len(fitter.data)} data points, "
            f"{n_iterations} iterations, {n_samples} samples/iter"
        )

    t0 = time.time()
    key, opt_key = jax.random.split(key)

    likelihood = _get_or_build_nifty_likelihood(fitter)
    init_pos = jft.Vector(init_params)

    # Use jft.optimize_kl with odir=None (no pickling/logging overhead).
    # Shared likelihood means the physics kernel is already compiled, this
    # call only pays for the KL minimization itself, not the SPS/dust/AGN stack.
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

    converged_pos = samples.pos
    converged_dict = converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)

    # Draw posterior samples
    key, draw_key = jax.random.split(key)
    all_sample_dicts = []

    # Include optimization samples from last iteration
    for s in list(samples):
        sd = s.tree if hasattr(s, "tree") else dict(s)
        all_sample_dicts.append(sd)

    if n_posterior_samples > 0:
        if posterior_method == "nonlinear":
            all_sample_dicts = fitter._draw_nonlinear_jit_samples(
                converged_dict,
                draw_key,
                n_posterior_samples,
                all_sample_dicts,
                verbose=verbose,
            )
        elif posterior_method == "blackjax":
            lh = _get_or_build_nifty_likelihood(fitter)
            all_sample_dicts = fitter._draw_blackjax_samples(
                lh,
                converged_dict,
                draw_key,
                n_posterior_samples,
                all_sample_dicts,
                verbose=verbose,
            )
        else:
            all_sample_dicts = fitter._draw_jit_samples(
                converged_dict,
                draw_key,
                n_posterior_samples,
                all_sample_dicts,
                verbose=verbose,
            )

    wall_time = time.time() - t0
    n_posterior = len(all_sample_dicts)

    samples_phys = {}
    for sample_dict in all_sample_dicts:
        phys = fitter._to_physical(sample_dict)
        for k, v in phys.items():
            if k not in samples_phys:
                samples_phys[k] = []
            samples_phys[k].append(v)

    samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
    best_params = _mean_params(samples_phys)

    chi2_dof = None
    if fitter.data_type == "photometry" and best_params:
        pred = fitter.model.predict_photometry(best_params)
        chi2_dof = float(
            jnp.sum(standardized_residual(fitter.data, pred, fitter.noise) ** 2)
        ) / len(fitter.data)

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
        _model=fitter.model,
    )


def _get_or_build_nifty_likelihood(fitter):
    """Return cached NIFTy likelihood, building on first call.

    For the non-variable-noise case, uses the shared ``engine["nifty_model"]``
    (physics-only, data-free) so the physics stack compiles once per model
    structure regardless of galaxy count.  Variable-noise models build their
    own per-Fitter model because ``signal_response`` captures ``noise`` data.
    """
    cached = _model_cache_owner.get_or_compile_model(fitter.model).get("nifty_lh")
    if cached is not None:
        return cached

    import nifty8.re as jft

    from tengri.observation.noise import (
        compute_effective_noise,
        compute_std_inv,
        has_noise_model,
        uses_student_t,
    )

    data = fitter.data
    noise = fitter.noise
    spec = fitter.spec
    stochastic = spec.stochastic
    use_variable_noise = has_noise_model(spec)
    use_student_t = uses_student_t(spec)

    if use_variable_noise:
        # Variable-noise: signal_response returns (predicted, noise_scale)
        # and captures per-Fitter noise array, cannot be shared.
        model = fitter.model
        data_type = fitter.data_type
        free_names = fitter._free_names
        fixed_values = fitter._fixed_values

        def _predict(params):
            """Dispatch to the forward model for this data_type given params."""
            if data_type == "photometry":
                return model.predict_photometry(params)
            elif data_type == "spectroscopy":
                return model.predict_spectrum(params)
            elif data_type == "joint":
                p = model.predict_photometry(params)
                s = model.predict_spectrum(params)
                return jnp.concatenate([p, s])
            raise ValueError(f"Unknown data_type: {data_type}")

        def _build_params(primals):
            """Unstandardize primals and merge fixed values into a physical parameter dict."""
            params = {}
            for name in free_names:
                dist = spec.get_distribution(name)
                params[name] = dist.unstandardize(primals[name])
            for name, val in fixed_values.items():
                params[name] = val
            if stochastic and "psd_xi" in primals:
                params["psd_xi"] = primals["psd_xi"]
            params = spec.resolve_mirrors(params)
            return params

        if use_student_t:

            def signal_response(primals):
                """Map primals to (predicted, effective_noise) for the Student-t likelihood."""
                params = _build_params(primals)
                predicted = _predict(params)
                f_cal = params.get("noise_frac_cal", 0.0)
                return predicted, compute_effective_noise(noise, predicted, f_cal)

        else:

            def signal_response(primals):
                """Map primals to (predicted, std_inv) for the Gaussian likelihood."""
                params = _build_params(primals)
                predicted = _predict(params)
                f_cal = params.get("noise_frac_cal", 0.0)
                return predicted, compute_std_inv(noise, predicted, f_cal)

        domain = {}
        for name in fitter._free_names:
            domain[name] = jft.ShapeWithDtype(())
        if stochastic:
            domain["psd_xi"] = jft.ShapeWithDtype((spec.n_grid,))

        nifty_model = jft.Model(jax.jit(signal_response), domain=domain)

        if use_student_t:
            dof = float(spec.get_distribution("noise_dof").value)
            likelihood = jft.VariableCovarianceStudentT(data, dof).amend(nifty_model)
        else:
            likelihood = jft.VariableCovarianceGaussian(data).amend(nifty_model)

    else:
        # Non-variable-noise: compile only the physics kernel, not the full
        # native-VI engine (run_evi_geovi_jit etc.).  The full engine can be
        # ~2 GB of compiled XLA; signal_response_jit is ~100 MB.  Only build
        # the full engine if it is already in cache (built by a prior run() call).
        from tengri.inference.jit_engine import get_or_build_signal_response

        # Prefer the nifty_model already in the engine cache if available,
        # avoids creating a second jft.Model object for the same physics.
        nifty_model = None
        if fitter._jit_sampler is not None:
            nifty_model = fitter._jit_sampler.get("nifty_model")

        if nifty_model is None:
            _, sr_jit = get_or_build_signal_response(fitter)
            domain = {}
            for name in fitter._free_names:
                domain[name] = jft.ShapeWithDtype(())
            if stochastic:
                domain["psd_xi"] = jft.ShapeWithDtype((spec.n_grid,))
            nifty_model = jft.Model(sr_jit, domain=domain)

        # Operators, not arrays: NIFTy derives whichever of (cov_inv, std_inv)
        # it is not given from the other, so an array for either reintroduces
        # the 1/sigma**2 overflow in float32 (#1206).
        cov_inv, std_inv = diag_noise_operators(noise)
        likelihood = jft.Gaussian(data, noise_cov_inv=cov_inv, noise_std_inv=std_inv).amend(
            nifty_model
        )

    _model_cache_owner.get_or_compile_model(fitter.model)["nifty_lh"] = likelihood
    return likelihood


def run_nifty_vi(
    context,
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

    .. warning::
       The NIFTy ``vi`` path and ``native_vi_nonlinear`` target the same KL
       objective but are **not posterior-equivalent**: the SFH PSD
       timescale ``sfh_field_psd_tau_myr`` has been observed to differ
       by ~10× between the two paths (e.g. 82 Myr vs 6 Myr on a 137-D
       stochastic problem). NIFTy is ~19–25× slower but considered the
       reference; validate per-problem before swapping backends.
       See ``bench/reports/2026-04-17_native_vs_nifty.md``.

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
        or "evi" (MGVI first, then geoVI, recommended).
    vi_config : VIConfig, optional
        Advanced configuration for NIFTy optimize_kl.
        If None, uses Philipp Frank's recommended defaults.
    posterior_method : str
        "linear" (default), draw_linear_residual, consistent with geoVI.
        "blackjax", BlackJAX NUTS, independent MCMC samples.
    verbose : bool
        Print progress.
    """
    import time

    try:
        import nifty8.re as jft
    except ImportError:
        raise ImportError("nifty8.re required for geoVI: pip install nifty8[re]") from None

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior
    from tengri.inference.vi_config import VIConfig, evi_sample_mode

    context = InferenceContext.from_target(context)
    # See ``run_nifty_fast_vi``, the JIT sampler cache and friends
    # live on the Fitter; we reach through ``context.fitter``.
    fitter = context.fitter

    cfg = vi_config or VIConfig()

    likelihood = _get_or_build_nifty_likelihood(fitter)

    data = fitter.data
    free_names = fitter._free_names
    spec = context.spec

    init_params = context.initial_params(key, init_from=init_from)

    # Convert to jft.Vector
    init_pos = jft.Vector(init_params)

    if verbose:
        n_total = len(free_names) + (spec.n_grid if spec.stochastic else 0)
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

    # Resolve sample_mode, same optimal schedule as _run_nifty_fast_vi
    resample_every = 5
    if sample_mode == "evi":
        resolved_mode = evi_sample_mode(n_iterations, cfg.evi_linear_fraction)
    elif sample_mode == "nonlinear_resample":

        def resolved_mode(i: int) -> str:
            """Schedule nonlinear_resample periodically, nonlinear_update otherwise."""
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
        pos_dict = converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
        all_sample_dicts = fitter._draw_posterior_samples(
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
        phys = fitter._to_physical(sample_dict)
        for k, v in phys.items():
            if k not in samples_phys:
                samples_phys[k] = []
            samples_phys[k].append(v)

    samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
    best_params = _mean_params(samples_phys)

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
        _model=fitter.model,
    )
