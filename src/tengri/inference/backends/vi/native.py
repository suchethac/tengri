"""Variational inference runners for tengri.

Extracted from fitter.py. Called by Fitter.run() dispatch table.
Each function takes (fitter, *, key, **kwargs) and returns a Posterior.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from tengri.inference._sample_utils import _mean_params


def run_native_vi(
    fitter,
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
    """Native JIT-compiled VI variants: ~500x faster than NIFTy's optimize_kl.

    Supports multiple sample modes:
    - ``"linear"`` (default): VI linear sampling (fastest, equivalent to MGVI).
    - ``"vi"``: Full geoVI with nonlinear coordinate curving.
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

    dummy_pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    if fitter._jit_sampler is None:
        fitter._jit_sampler = fitter._get_or_build_engine(dummy_pos)

    engine = fitter._jit_sampler

    # n_samples is a static_argname in run_evi_geovi_jit, so any value other
    # than 3 (the background-compilation default) triggers a full XLA
    # recompilation (~30-60s).  Warn so the user isn't surprised.
    if n_samples != 3:
        warnings.warn(
            f"n_samples={n_samples} differs from the pre-compiled default (3). "
            "Because n_samples is a static JAX argument, this triggers a full "
            f"XLA recompilation (~30-60s). Call fitter.compile(n_samples={n_samples}) "
            "ahead of time to avoid the delay.",
            UserWarning,
            stacklevel=3,
        )

    flatten = engine["flatten"]
    unflatten = engine["unflatten"]
    data_args = fitter._data_args

    n_total = len(fitter._free_names) + (fitter.spec.n_grid if fitter.spec.stochastic else 0)
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
            f"{mode_label} (JIT): {n_total} params, {len(fitter.data)} data points, "
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
        map_result = fitter._run_map(key=map_key, n_steps=500, verbose=False)
        if verbose:
            print("  MAP warmstart done")

    init_flats = []
    for s in range(n_seeds):
        if map_result is not None:
            init_params = fitter._unbounded_from_posterior(map_result)
        elif isinstance(init_from, Posterior):
            init_params = fitter._unbounded_from_posterior(init_from)
        else:
            init_params = fitter._initialize_unbounded(seed_keys[s])
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
            phys = fitter._to_physical(unflatten(converged_flat))
            if fitter.data_type == "photometry":
                pred = fitter.model.predict_photometry(phys)
            elif fitter.data_type == "spectroscopy":
                pred = fitter.model.predict_spectrum(phys)
            else:
                pred = jnp.zeros_like(fitter.data)
            chi2 = jnp.sum(((fitter.data - pred) / fitter.noise) ** 2)
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
            phys = fitter._to_physical(unflatten(converged_flat))
            if fitter.data_type == "photometry":
                pred = fitter.model.predict_photometry(phys)
            elif fitter.data_type == "spectroscopy":
                pred = fitter.model.predict_spectrum(phys)
            else:
                pred = jnp.zeros_like(fitter.data)
            chi2 = float(jnp.sum(((fitter.data - pred) / fitter.noise) ** 2))
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
            all_sample_dicts = fitter._draw_blackjax_samples(
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
                all_sample_dicts = fitter._draw_nonlinear_jit_samples(
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
        phys = fitter._to_physical(sample_dict)
        for k, v in phys.items():
            if k not in samples_phys:
                samples_phys[k] = []
            samples_phys[k].append(v)

    samples_phys = {k: jnp.stack(v) for k, v in samples_phys.items()}
    best_params = _mean_params(samples_phys)

    # --- Post-fit diagnostics ---
    diag_warnings = []

    # Check chi2/dof
    if fitter.data_type == "photometry":
        pred = fitter.model.predict_photometry(best_params)
        chi2_dof = float(jnp.sum(((fitter.data - pred) / fitter.noise) ** 2)) / len(fitter.data)
        if chi2_dof > 5.0:
            diag_warnings.append(f"Poor fit: chi2/dof={chi2_dof:.1f} (expected ~1)")
        elif chi2_dof < 0.1:
            diag_warnings.append(f"Suspiciously good fit: chi2/dof={chi2_dof:.2f}")
    else:
        chi2_dof = None

    # Check parameters at bounds
    at_bounds = []
    for name in fitter._free_names:
        if name in samples_phys:
            lo, hi = fitter._bounds[name]
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
        _model=fitter.model,
    )
