# SPDX-License-Identifier: BSD-3-Clause
"""Variational inference runners for tengri.

Extracted from fitter.py. Called by Fitter.run() dispatch table.
Each function takes (fitter, *, key, **kwargs) and returns a Posterior.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from tengri.config.exceptions import warn_measured
from tengri.inference._sample_utils import _mean_params
from tengri.inference.likelihoods.gaussian import (
    inv_noise_std,
    standardized_residual,
    whiten,
)


def _cg_eps() -> float:
    """CG relative-tolerance floor, ``6 * eps`` of the **working** dtype.

    NIFTy's constant, but resolved against the dtype actually in use rather
    than pinned to float64 (#1568). float64's ``6 * eps`` is 1.33e-15; asking a
    float32 solve (``eps`` = 1.19e-7) to reach that is asking for a tolerance
    eight decades below its own resolution, so the criterion is unreachable and
    the solver runs to its iteration cap instead of converging.
    """
    return 6.0 * float(jnp.finfo(jnp.result_type(float)).eps)


def _cg_tiny() -> float:
    """CG absolute-residual floor, ``6 * tiny`` of the **working** dtype.

    float64's ``6 * tiny`` is 1.335e-307, which is **0.0** in float32, so
    ``(gamma >= 0.0) & (gamma <= _cg_tiny())`` degenerated to ``gamma == 0.0``
    and the "residual is numerically zero, stop" branch fired only on an exact
    zero (#1568).
    """
    return 6.0 * float(jnp.finfo(jnp.result_type(float)).tiny)


def _cg_solve(
    mat_fn,
    b,
    x0,
    maxiter=30,
    miniter=6,
    absdelta=None,
    resnorm=None,
    tol=1e-5,
    atol=0.0,
    norm_ord=None,
):
    """Conjugate-gradient solver (implements NIFTy ``_static_cg``).

    Implements NIFTy's ``_static_cg`` algorithm exactly
    (``nifty8.re.conjugate_gradient._static_cg``, Gordian Edenhofer,
    Philipp Frank, GPL-2.0+).  Deviations from upstream: (1) flat-array
    API (no pytree / Vector wrapper); (2) ``mat_fn`` is a plain callable
    rather than a ``jax.tree_util.Partial``; (3) no callback/logging;
    (4) returns the solution array directly rather than a ``CGResults``
    namedtuple.

    Convergence criteria (evaluated in order, matching NIFTy exactly):

    1. ``gamma <= tiny``, solution is numerically zero.
    2. ``||r||_{norm_ord} < resnorm`` after ``miniter`` iterations
       (primary; disabled when ``resnorm is None``).
    3. ``energy_diff < absdelta`` after ``miniter`` iterations
       (secondary; disabled when ``absdelta is None``).
    4. ``i >= maxiter``, hard cap.

    When both ``absdelta`` and ``resnorm`` are ``None``, falls back to
    ``resnorm = max(tol * ||b||_{norm_ord}, atol)``, matching NIFTy's
    default ``tol``-based convergence.

    Parameters
    ----------
    mat_fn: callable
        ``(x) -> Ax``. Symmetric positive-definite matrix-vector product.
    b: ndarray, shape (d,)
        Right-hand side.
    x0: ndarray, shape (d,)
        Initial guess.
    maxiter: int
        Hard iteration cap.  NIFTy default: ``max(min(200, 20*d), miniter)``.
    miniter: int
        Minimum iterations before convergence is checked.
        NIFTy default: ``min(6, maxiter)``.
    absdelta: float or None
        Energy-improvement convergence threshold (secondary).
        ``None`` (default) disables this criterion, matching NIFTy's
        default for all internal CG calls.
    resnorm: float or None
        Absolute residual-norm threshold (primary).
        ``None`` (default) disables unless ``absdelta`` is also ``None``,
        in which case ``tol``-based fallback activates.
    tol: float
        Relative tolerance for ``tol``-based fallback: ``tol * ||b||``.
        Used only when both ``absdelta`` and ``resnorm`` are ``None``.
        NIFTy default: ``1e-5``.
    atol: float
        Absolute tolerance for fallback.  NIFTy default: ``0.0``.
    norm_ord: int or None
        Norm order for residual and gradient norms.
        NIFTy default (CG): ``2`` (``None`` → 2).

    Returns
    -------
    ndarray, shape (d,)
        Solution x ≈ A^{-1} b.

    References
    ----------
    .. [1] Edenhofer G., Frank P. et al., "NIFTy.re: Towards a JAX-native
       library for variational inference at the petascale," arXiv:2402.16683
       (2024).  ``nifty8/re/conjugate_gradient.py``, ``_static_cg``.
    """
    norm_ord = 2 if norm_ord is None else norm_ord

    # Fallback when neither criterion is specified (NIFTy tol-based default).
    if absdelta is None and resnorm is None:
        b_norm = jnp.sum(jnp.abs(b)) if norm_ord == 1 else jnp.sqrt(jnp.dot(b, b))
        resnorm = jnp.maximum(tol * b_norm, atol)

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
        info = jnp.where(curv <= 0.0, jnp.int32(0), info)
        alpha = jnp.where(curv <= 0.0, 0.0, alpha)
        pos = pos - alpha * d
        pos = jnp.where((curv < 0.0) & (i <= 1), prev_energy / (-curv) * (-b), pos)
        r_reset = mat_fn(pos) - b
        r_step = r - q * alpha
        r = jnp.where((i % 20 == 0) & (info < -1), r_reset, r_step)
        gamma = jnp.dot(r, r)
        info = jnp.where((gamma >= 0.0) & (gamma <= _cg_tiny()) & (info != -1), jnp.int32(0), info)
        if norm_ord == 1:
            r_norm = jnp.sum(jnp.abs(r))
        else:
            r_norm = jnp.sqrt(gamma)
        if resnorm is not None:
            info = jnp.where(
                (r_norm < resnorm) & (i >= miniter) & (info != -1),
                jnp.int32(0),
                info,
            )
        energy = jnp.dot((r - b) / 2, pos)
        energy_diff = prev_energy - energy
        info = jnp.where(
            energy_diff < -_cg_eps() * jnp.abs(energy),
            jnp.where(info < -1, i, info),
            info,
        )
        if absdelta is not None:
            info = jnp.where(
                (energy_diff < absdelta) & (i >= miniter) & (info != -1),
                jnp.int32(0),
                info,
            )
        info = jnp.where((i >= maxiter) & (info != -1), i, info)
        d = d * jnp.maximum(0.0, gamma / prev_gamma) + r
        return (pos, r, d, gamma, energy, info, i)

    return jax.lax.while_loop(cond, body, init)[0]


def _newton_cg_flat(
    fun_and_grad,
    hessp,
    x0,
    *,
    miniter=0,
    maxiter=200,
    energy_reduction_factor=0.1,
    old_fval=None,
    absdelta=None,
    norm_ord=None,
    xtol=1e-5,
    custom_gradnorm=None,
):
    """Newton-CG with successive-halving line search (implements NIFTy ``_static_newton_cg``).

    Implements NIFTy's ``_static_newton_cg`` algorithm exactly
    (``nifty8.re.optimize._static_newton_cg``, Gordian Edenhofer,
    Philipp Frank, GPL-2.0+).  Deviations from upstream: (1) flat-array
    API (no pytree); (2) combined ``fun_and_grad`` rather than separate
    ``fun``/``jac``; (3) no callback/logging; (4) returns ``(pos, energy)``
    rather than ``OptimizeResults``.

    Parameters
    ----------
    fun_and_grad: callable
        ``(x) -> (value, grad)``.
    hessp: callable
        ``(x, v) -> Hessian-vector product``.
    x0: ndarray, shape (d,)
        Initial guess.  ``d = x0.shape[0]`` used for convergence scaling
        (``ncg_xtol = xtol * d``).
    miniter: int
        Minimum Newton iterations before convergence is checked.
        NIFTy default: ``0``.
    maxiter: int
        Newton iteration limit.  NIFTy default: ``200``.
    energy_reduction_factor: float
        Fraction of expected energy decrease used as CG ``absdelta``.
        NIFTy default: ``0.1``.
    old_fval: float or None
        Energy at a previous position for warm-starting the old-energy
        tracker.  ``None`` → ``inf`` (NIFTy default).
    absdelta: float or None
        Energy-improvement convergence threshold.  ``None`` (default)
        disables this criterion, matching NIFTy's default for
        ``nonlinearly_update_residual`` (no ``absdelta`` is passed).
    norm_ord: int or None
        Norm order for gradient norm.  NIFTy default: ``1`` (L1).
    xtol: float
        Descent-norm convergence threshold (scaled by ``d``).
        NIFTy default: ``1e-5``.
    custom_gradnorm: callable or None
        Custom gradient norm ``(v) -> scalar``.  Overrides ``norm_ord``
        when provided (used by ``curve_residual`` for Fisher-metric norm).

    Returns
    -------
    pos: ndarray, shape (d,)
        Converged position.
    energy: float
        Final objective value.

    References
    ----------
    .. [1] Edenhofer G., Frank P. et al., "NIFTy.re: Towards a JAX-native
       library for variational inference at the petascale," arXiv:2402.16683
       (2024).  ``nifty8/re/optimize.py``, ``_static_newton_cg``.
    """
    norm_ord = 1 if norm_ord is None else norm_ord
    d_total = x0.shape[0]
    ncg_xtol = xtol * d_total

    def gradnorm(v):
        if custom_gradnorm is not None:
            return custom_gradnorm(v)
        return jnp.sum(jnp.abs(v)) if norm_ord == 1 else jnp.sqrt(jnp.dot(v, v))

    energy, g = fun_and_grad(x0)
    init_state = (
        x0,
        energy,
        jnp.array(old_fval if old_fval is not None else jnp.inf),
        g,
        jnp.where(maxiter == 0, jnp.int32(0), jnp.int32(-2)),
        jnp.int32(0),
    )

    def ncg_cond(state):
        return state[4] < -1

    def ncg_body(state):
        pos, energy, old_energy, g, status, i = state
        i = i + 1

        cg_abd_fallback = jnp.array(0.0, dtype=energy.dtype)
        cg_absdelta = jnp.where(
            ~jnp.isinf(old_energy),
            energy_reduction_factor * (old_energy - energy),
            cg_abd_fallback,
        )
        cg_absdelta = jnp.array(cg_absdelta, dtype=energy.dtype)
        mag_g = jnp.sum(jnp.abs(g))
        cg_resnorm = jnp.minimum(0.5, jnp.sqrt(mag_g)) * mag_g

        nat_g = _cg_solve(
            lambda v: hessp(pos, v),
            g,
            jnp.zeros_like(pos),
            maxiter=min(200, 20 * d_total),
            miniter=min(6, min(200, 20 * d_total)),
            absdelta=cg_absdelta,
            resnorm=cg_resnorm,
        )

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
            ls_st, ls_i, _np, _ne, _ng, dd, gs, reset, nhev = ls
            new_pos = pos - gs * dd
            new_e, new_g = fun_and_grad(new_pos)
            ls_st = jnp.where(new_e <= energy, jnp.int32(0), ls_st)
            gs = jnp.where(ls_st < -1, gs / 2.0, gs)
            do_reset = (ls_i == 5) & (ls_st < -1)
            reset = jnp.where(do_reset, jnp.bool_(True), reset)
            gs = jnp.where(do_reset, 1.0, gs)
            # NIFTy uses lax.cond here to avoid computing hessp when not resetting.
            dd = jax.lax.cond(
                do_reset,
                lambda _: jnp.dot(g, g) / jnp.dot(g, hessp(pos, g)) * g,
                lambda _: dd,
                None,
            )
            nhev = nhev + do_reset.astype(jnp.int32)
            do_abort = (ls_i == 8) & (ls_st < -1)
            ls_st = jnp.where(do_abort, jnp.int32(-1), ls_st)
            return (ls_st, ls_i + 1, new_pos, new_e, new_g, dd, gs, reset, nhev)

        ls_result = jax.lax.while_loop(ls_cond, ls_body, ls_init)
        ls_status, ls_iter, new_pos, new_energy, new_g, dd, gs, _reset, _nhev = ls_result

        status = jnp.where(ls_status != 0, jnp.int32(-1), status)
        success = status < -1
        old_energy = jnp.where(success, energy, old_energy)
        energy_out = jnp.where(success, new_energy, energy)
        energy_diff = jnp.where(success, old_energy - energy_out, 0.0)
        pos_out = jnp.where(success, new_pos, pos)
        g_out = jnp.where(success, new_g, g)
        gs_out = jnp.where(success, gs, 0.0)
        descent_norm = gs_out * gradnorm(dd)

        # NaN guard (NIFTy: status = jnp.where(jnp.isnan(energy), -1, status)).
        status = jnp.where(jnp.isnan(energy_out), jnp.int32(-1), status)
        min_cond = (ls_iter < 2) & (i > miniter)
        # Energy-diff criterion: only active when absdelta is not None
        # (NIFTy default: disabled for nonlinearly_update_residual).
        if absdelta is not None:
            status = jnp.where(
                (energy_diff >= 0.0) & (energy_diff < absdelta) & min_cond & (status != -1),
                jnp.int32(0),
                status,
            )
        status = jnp.where(
            (descent_norm <= ncg_xtol) & (i > miniter) & (status != -1),
            jnp.int32(0),
            status,
        )
        status = jnp.where((i == maxiter) & (status < -1), i, status)
        return (pos_out, energy_out, old_energy, g_out, status, i)

    result = jax.lax.while_loop(ncg_cond, ncg_body, init_state)
    return result[0], result[1]


def run_native_vi(
    context,
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

    .. warning::
       ``native_vi_nonlinear`` and the NIFTy ``vi`` path target the same KL
       objective but are **not posterior-equivalent**: the SFH PSD
       timescale ``sfh_field_psd_tau_myr`` has been observed to differ
       by ~10× between the two paths (e.g. 6 Myr vs 82 Myr on a 137-D
       stochastic problem). The native path is ~19–25× faster but
       posterior shape must be validated per-problem before swapping.
       See ``bench/reports/2026-04-17_native_vs_nifty.md``.

    Parameters
    ----------
    init_from: str, Posterior, or None
        ``"auto"`` (default): MAP for ``n_seeds=1``, random for
        ``n_seeds>1``. MAP gives better convergence for a single
        seed; random init is better for multi-seed because vmap
        needs diverse starting points to find the global mode.
        ``"map"``: quick MAP estimate as starting point for all seeds.
        ``"random"`` or ``None``: random init near prior midpoint.
        ``Posterior``: use a previous result as initialization.
    n_iterations: int
        Maximum KL iterations. Auto-stops when converged.
    n_samples: int
        Samples per iteration (doubled by mirror_samples).
    n_posterior_samples: int
        Posterior samples drawn after convergence.
    kl_rtol: float
        Relative KL tolerance for early stopping. Set to 0 to
        disable and run all ``n_iterations``.
    n_seeds: int
        Number of random seeds to run in parallel via ``jax.vmap``.
        The best result (lowest Hamiltonian) is returned. Multiple
        seeds catch bad initialization and multimodality.
    parallel_seeds: bool or None
        If ``None`` (default), auto-detect: ``True`` on GPU/TPU,
        ``False`` on CPU. On CPU, sequential is typically faster
        because early-converging seeds exit early, while vmap must
        run all seeds for the maximum iteration count.
        Set explicitly to override.
    verbose: bool
        Print progress.
    """
    import warnings

    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    # Native VI carries the JIT sampler cache + native_vi engine on
    # the Fitter (long-lived across ``run()`` calls). Reach through
    # ``context.fitter``, these caches must not be re-created.
    fitter = context.fitter

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

    # Build/cache the native nonlinear engine for the geoVI path.
    # The engine captures data/noise in its closure; it is cached on the fitter
    # so the JIT compilation cost is paid once per Fitter instance.
    if not hasattr(fitter, "_native_vi_nonlinear_engine"):
        fitter._native_vi_nonlinear_engine = None
    if fitter._native_vi_nonlinear_engine is None:
        from tengri.inference.jit_engine import get_or_build_signal_response

        _sr, _ = get_or_build_signal_response(fitter)
        fitter._native_vi_nonlinear_engine = build_native_vi_nonlinear_engine(
            _sr, jnp.asarray(fitter.data), jnp.asarray(fitter.noise), flatten, unflatten
        )
    _nonlinear_run_fn, _nonlinear_draw_fn, _nonlinear_hamiltonian = (
        fitter._native_vi_nonlinear_engine
    )

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
            # vmap over (init_pos, key); factory closure captures data/noise
            def _run_single_geovi(pos, k):
                """Run native geoVI on a single initial position."""
                return _nonlinear_run_fn(
                    pos,
                    k,
                    n_iter=n_iterations,
                    n_samp=n_samples,
                    rtol=kl_rtol,
                )

            vmapped_run = jax.vmap(_run_single_geovi)
        else:

            def _run_single_evi(pos, k):
                """Run EVI (linear VI) on a single initial position."""
                return engine["native_vi_linear_run"](
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

        # Batch Hamiltonian evaluation via the nonlinear factory (closure captures data/noise)
        if _use_geovi:
            seed_losses_arr = jax.vmap(_nonlinear_hamiltonian)(all_converged)
        else:

            def _eval_hamiltonian_linear(converged_flat):
                """Compute Hamiltonian for linear (MGVI) seeds."""
                phys = fitter._to_physical(unflatten(converged_flat))
                if fitter.data_type == "photometry":
                    pred = fitter.model.predict_photometry(phys)
                elif fitter.data_type == "spectroscopy":
                    pred = fitter.model.predict_spectrum(phys)
                else:
                    pred = jnp.zeros_like(fitter.data)
                chi2 = jnp.sum(standardized_residual(fitter.data, pred, fitter.noise) ** 2)
                prior = jnp.sum(converged_flat**2)
                return 0.5 * chi2 + 0.5 * prior

            seed_losses_arr = jax.vmap(_eval_hamiltonian_linear)(all_converged)
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
                converged_flat, n_iters = _nonlinear_run_fn(
                    pos_flat,
                    opt_key,
                    n_iter=n_iterations,
                    n_samp=n_samples,
                    rtol=kl_rtol,
                )
            else:
                converged_flat, n_iters = engine["native_vi_linear_run"](
                    pos_flat,
                    opt_key,
                    data_args,
                    n_iterations=n_iterations,
                    n_samples=n_samples,
                    kl_rtol=kl_rtol,
                )
            n_iters = int(n_iters)

            # Evaluate Hamiltonian to pick best seed
            if _use_geovi:
                loss = float(_nonlinear_hamiltonian(converged_flat))
            else:
                phys = fitter._to_physical(unflatten(converged_flat))
                if fitter.data_type == "photometry":
                    pred = fitter.model.predict_photometry(phys)
                elif fitter.data_type == "spectroscopy":
                    pred = fitter.model.predict_spectrum(phys)
                else:
                    pred = jnp.zeros_like(fitter.data)
                chi2 = float(jnp.sum(standardized_residual(fitter.data, pred, fitter.noise) ** 2))
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
            warn_measured(
                f"Seeds disagree: H = {loss_mean:.1f} ± {loss_std:.1f} "
                f"(CV={loss_std / abs(loss_mean):.0%}). "
                f"This may indicate multimodality or poor convergence. "
                f"Consider increasing n_iterations or inspecting the posterior.",
                UserWarning,
                stacklevel=2,
                loss_mean=loss_mean,
                loss_std=loss_std,
                loss_cv=loss_std / abs(loss_mean),
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
                None,  # likelihood not needed, logdensity built internally
                converged_dict,
                draw_key,
                n_posterior_samples,
                all_sample_dicts,
                verbose=verbose,
            )
        else:
            # Use nonlinear draws for geoVI modes, linear for MGVI.
            # For native_vi_nonlinear, use the factory's draw_nonlinear_residuals_jit.
            # Each key produces one mirrored pair (2 residuals), so pass n // 2 keys.
            use_nonlinear = sample_mode in (
                "geovi",
                "nonlinear_resample",
                "nonlinear_update",
                "nonlinear_sample",
            )
            if use_nonlinear:
                if verbose:
                    print(f"  Drawing {n_posterior_samples} geoVI posterior samples (JIT)...")
                n_draw_keys = max(1, n_posterior_samples // 2)
                draw_keys = jax.random.split(draw_key, n_draw_keys)
                residuals_flat = _nonlinear_draw_fn(converged_flat, draw_keys)
                # residuals_flat: (2*n_draw_keys, d_total); trim to n_posterior_samples
                residuals_flat = residuals_flat[:n_posterior_samples]
                for i in range(residuals_flat.shape[0]):
                    res = unflatten(residuals_flat[i])
                    combined = {k: converged_dict[k] + res[k] for k in converged_dict}
                    all_sample_dicts.append(combined)
            else:
                if verbose:
                    print(f"  Drawing {n_posterior_samples} posterior samples (JIT CG)...")
                draw_keys = jax.random.split(draw_key, n_posterior_samples)
                residuals_flat = engine["native_vi_linear_draw"](
                    converged_flat, draw_keys, data_args
                )
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
        chi2_dof = float(
            jnp.sum(standardized_residual(fitter.data, pred, fitter.noise) ** 2)
        ) / len(fitter.data)
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


def build_native_vi_linear_engine(signal_response, data, noise, flatten, unflatten):
    """Build JIT-compiled native_vi_linear primitives for a fixed signal_response.

    Shared backend used by both Fitter (via jit_engine) and PopulationFitter
    (directly from hierarchical._run_native_vi_linear). Implements pure-JAX
    linear MGVI without NIFTy, the full optimization loop runs inside
    ``jax.lax.while_loop`` with zero Python overhead.

    Parameters
    ----------
    signal_response: callable
        ``(pytree) -> ndarray, shape (n_data,)``. Differentiable forward model.
    data: ndarray, shape (n_data,)
        Observed data vector.
    noise: ndarray, shape (n_data,)
        Per-datum noise standard deviations.
    flatten: callable
        ``pytree -> 1D ndarray``.
    unflatten: callable
        ``1D ndarray -> pytree``.

    Returns
    -------
    run_native_vi_linear_jit: callable
        ``(init_flat, vi_key, n_iter, n_samp, rtol) -> (best_flat, n_iters)``.
        JIT-compiled with ``static_argnames=("n_iter", "n_samp")``.
    draw_residuals_jit: callable
        ``(pos_flat, subkeys) -> residuals_flat``, shape ``(n_samples, d_total)``.
    hamiltonian_fn: callable
        ``(flat) -> scalar``. Useful for seed selection.

    Notes
    -----
    The CG solver implements NIFTy's ``_static_cg`` exactly, with residual-norm
    as the primary convergence criterion and energy-based ``absdelta`` as
    secondary. This matches ``jit_engine.py``'s ``cg_solve`` and is superior
    to the energy-only variant previously inlined in ``hierarchical.py``.

    **JIT-compatible**: the returned callables are pre-JIT'd.
    """
    sqrt_noise_inv = inv_noise_std(noise)

    def metric_vec(xi, v):
        # (J/sigma)^T (J/sigma) v + v -- never forms 1/sigma**2 (~1e59, inf in
        # float32). See likelihoods.gaussian.whiten (#1206).
        xi_d, v_d = unflatten(xi), unflatten(v)
        _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
        _, vjp_fn = jax.vjp(signal_response, xi_d)
        return flatten(vjp_fn(whiten(whiten(Jv, noise), noise))[0]) + v

    def hamiltonian(xi):
        pred = signal_response(unflatten(xi))
        chi2 = jnp.sum(standardized_residual(data, pred, noise) ** 2)
        return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

    H_vg = jax.value_and_grad(hamiltonian)

    def draw_residuals(pos_f, subkeys):
        def draw_one(subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=pos_f.shape)
            eta_lh = jax.random.normal(k2, shape=data.shape)
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
            return _cg_solve(
                lambda v: metric_vec(pos_f, v),
                jt + eta_pr,
                eta_pr,
                maxiter=30,
                miniter=6,
                absdelta=1e-4,
            )

        return jax.vmap(draw_one)(subkeys)

    def kl_vg(m, residuals):
        vals, grads = jax.vmap(lambda r: H_vg(m + r))(residuals)
        return jnp.mean(vals), jnp.mean(grads, axis=0)

    def kl_metric(m, residuals, v):
        return jnp.mean(jax.vmap(lambda r: metric_vec(m + r, v))(residuals), axis=0)

    def vi_linear_step(m, subkey, n_samp):
        sample_keys = jax.random.split(subkey, n_samp)
        residuals = draw_residuals(m, sample_keys)
        residuals = jnp.concatenate([residuals, -residuals], axis=0)

        def ncg_body(carry):
            m_cur, prev_val, info, i = carry
            i = i + 1
            val, grad = kl_vg(m_cur, residuals)
            step = _cg_solve(
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

        val0, _ = kl_vg(m, residuals)
        result = jax.lax.while_loop(
            lambda s: s[2] < -1,
            ncg_body,
            (m, val0, jnp.int32(-2), jnp.int32(0)),
        )
        return result[0], result[1]

    def run_vi_linear(init_pos, vi_key, n_iter, n_samp, rtol):
        keys = jax.random.split(vi_key, n_iter)

        def cond_fn(state):
            _m, _prev_kl, i, converged = state
            return (~converged) & (i < n_iter)

        def body_fn(state):
            m, prev_kl, i, converged = state
            subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
            m_new, kl_val = vi_linear_step(m, subkey, n_samp)
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < rtol) & (i >= 5)
            return (m_new, kl_val, i + 1, converged)

        m0, kl0 = vi_linear_step(init_pos, keys[0], n_samp)
        init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))
        m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return m_final, n_iters

    run_native_vi_linear_jit = jax.jit(run_vi_linear, static_argnames=("n_iter", "n_samp"))
    draw_residuals_jit = jax.jit(draw_residuals)

    return run_native_vi_linear_jit, draw_residuals_jit, hamiltonian


def build_native_vi_nonlinear_engine(signal_response, data, noise, flatten, unflatten):
    """Build JIT-compiled native_vi_nonlinear (geoVI) primitives.

    Pure-JAX implementation of geometric variational inference (geoVI /
    nonlinear MGVI). Shared backend used by both ``Fitter`` (via
    ``run_native_vi``) and ``PopulationFitter`` (via
    ``hierarchical._run_native_vi_nonlinear``).

    Each KL iteration draws fresh nonlinear residuals by:

    1. Drawing a CG-inverted linear residual (same as MGVI).
    2. Curving each residual via Newton-CG (``curve_residual``) to follow the
       nonlinear coordinate geometry, NIFTy's ``nonlinearly_update_residual``
       algorithm, step for step.
    3. Mirroring: both ``+r`` and ``-r`` are curved, giving ``2*n_samp``
       effective samples with zero first-order bias.

    Parameters
    ----------
    signal_response: callable
        ``(pytree) -> ndarray, shape (n_data,)``. Differentiable forward model.
    data: ndarray, shape (n_data,)
        Observed data vector.
    noise: ndarray, shape (n_data,)
        Per-datum noise standard deviations.
    flatten: callable
        ``pytree -> 1D ndarray``.
    unflatten: callable
        ``1D ndarray -> pytree``.

    Returns
    -------
    run_native_vi_nonlinear_jit: callable
        ``(init_flat, vi_key, n_iter, n_samp, rtol) -> (best_flat, n_iters)``.
        JIT-compiled with ``static_argnames=("n_iter", "n_samp")``.
    draw_nonlinear_residuals_jit: callable
        ``(pos_flat, subkeys) -> residuals_flat``, shape
        ``(2*n_samples, d_total)``. Each key produces one mirrored pair.
    hamiltonian_fn: callable
        ``(flat) -> scalar``. Useful for seed selection.

    Notes
    -----
    ``curve_residual`` implements NIFTy's ``nonlinearly_update_residual``
    algorithm exactly (evi.py:136-217). The inner Newton-CG runs
    with ``maxiter=3`` matching NIFTy's default for sample curving.

    **JIT-compatible**: all returned callables are pre-JIT'd.

    .. math::

        \\phi(x) = \\tfrac{1}{2} \\| m_s - g(x) \\|^2, \\quad
        g(x) = (x - m) + L(m)[t(x) - t(m)]

    where :math:`m_s` is the metric sample, :math:`t(x) = \\sqrt{N^{-1}} f(x)`
    is the whitened transform, and :math:`L(m)` is the left square-root metric.

    Notes
    -----
    The geoVI curving machinery (``curve_residual``/``sampnorm``) takes
    **data-space inner products** with ``jnp.dot``, which silently does a
    matrix product on a 2-D array. Hierarchical/population fits hand in
    ``(N_gal, n_pix)`` data and a ``signal_response`` returning the same shape,
    so the data space is flattened to 1-D here (and the prediction raveled to
    match). Single-galaxy 1-D data is unchanged. Without this the canonical path
    crashes under ``native_vi_nonlinear`` even though ``native_vi_linear``
    works, the topology-agnostic contract must hold across backends
    (suchethac/tengri#711).
    """
    _signal_response_nd = signal_response

    def signal_response(xi):
        return jnp.ravel(_signal_response_nd(xi))

    data = jnp.ravel(jnp.asarray(data))
    noise = jnp.ravel(jnp.asarray(noise))
    sqrt_noise_inv = inv_noise_std(noise)

    def metric_vec(xi, v):
        # (J/sigma)^T (J/sigma) v + v -- never forms 1/sigma**2 (~1e59, inf in
        # float32). See likelihoods.gaussian.whiten (#1206).
        xi_d, v_d = unflatten(xi), unflatten(v)
        _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
        _, vjp_fn = jax.vjp(signal_response, xi_d)
        return flatten(vjp_fn(whiten(whiten(Jv, noise), noise))[0]) + v

    def hamiltonian(xi):
        pred = signal_response(unflatten(xi))
        chi2 = jnp.sum(standardized_residual(data, pred, noise) ** 2)
        return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

    H_vg = jax.value_and_grad(hamiltonian)

    def transformation(xi):
        """t(x) = sqrt(N^-1) @ f(x): whitened data-space transform."""
        return sqrt_noise_inv * signal_response(unflatten(xi))

    def left_sqrt_metric(xi, v_data):
        """J^T(xi) @ sqrt(N^-1) @ v_data."""
        _, vjp_fn = jax.vjp(signal_response, unflatten(xi))
        return flatten(vjp_fn(sqrt_noise_inv * v_data)[0])

    def right_sqrt_metric(xi, v_param):
        """sqrt(N^-1) @ J(xi) @ v_param."""
        _, Jv = jax.jvp(signal_response, (unflatten(xi),), (unflatten(v_param),))
        return sqrt_noise_inv * Jv

    def draw_metric_sample(xi, subkey):
        """Metric sample (NOT CG-inverted): J^T sqrt_N^-1 eta_lh + eta_pr."""
        k1, k2 = jax.random.split(subkey)
        eta_pr = jax.random.normal(k1, shape=xi.shape)
        # Whitened-data draw follows the noise array's shape, 1-D for a single
        # galaxy, (N_gal, n_pix) for a hierarchical population fit. Was
        # ``shape=(data.shape[0],)``, which collapsed a batched (N_gal, n_pix)
        # data array to N_gal and broke population geoVI (suchethac/tengri#711).
        eta_lh = jax.random.normal(k2, shape=sqrt_noise_inv.shape)
        _, vjp_fn = jax.vjp(signal_response, unflatten(xi))
        jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
        return jt + eta_pr

    def draw_linear_residual(pos_f, subkey):
        """Draw one CG-inverted linear residual (same as MGVI draw)."""
        k1, k2 = jax.random.split(subkey)
        eta_pr = jax.random.normal(k1, shape=pos_f.shape)
        # See draw_metric_sample: shape follows the noise array, not data.shape[0].
        eta_lh = jax.random.normal(k2, shape=sqrt_noise_inv.shape)
        _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
        jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
        return _cg_solve(
            lambda v: metric_vec(pos_f, v),
            jt + eta_pr,
            eta_pr,
            maxiter=30,
            miniter=6,
            absdelta=1e-4,
        )

    def curve_residual(m, r_linear, metric_key, sign):
        """Nonlinearly curve a linear residual (geoVI).

        Implements NIFTy ``nonlinearly_update_residual`` (evi.py:136-217).
        Finds x* minimizing phi(x) = 0.5||m_s - g(x)||^2 via Newton-CG,
        then returns x* - m as the curved residual.
        """
        x0 = m + r_linear
        ms = sign * draw_metric_sample(m, metric_key)
        trafo_at_m = transformation(m)

        def phi_vg(x):
            trafo_x = transformation(x)
            delta_trafo = trafo_x - trafo_at_m
            g_x = (x - m) + left_sqrt_metric(m, delta_trafo)
            r = ms - g_x
            val = 0.5 * jnp.dot(r, r)
            ngrad = r + left_sqrt_metric(x, right_sqrt_metric(m, r))
            return val, -ngrad

        def phi_metric(x, v):
            tm = left_sqrt_metric(m, right_sqrt_metric(x, v)) + v
            return left_sqrt_metric(x, right_sqrt_metric(m, tm)) + tm

        def sampnorm(natgrad):
            fpp = right_sqrt_metric(m, natgrad)
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

    def draw_nonlinear_residuals(m, subkeys):
        """Draw geoVI residuals: ``(2*n_samp, d_total)`` with mirrored pairs.

        Uses ``lax.map`` (not ``vmap``) over both ``draw_linear_residual`` and
        ``curve_pair`` so that the compiled XLA graph is O(1) in n_samp.
        ``vmap`` over functions containing ``while_loop`` materializes O(n_samp)
        graph copies; ``lax.map`` compiles one body and sequences over the batch.
        """
        linear_residuals = jax.lax.map(lambda sk: draw_linear_residual(m, sk), subkeys)

        def curve_pair(args):
            r, subkey = args
            r_pos = curve_residual(m, r, subkey, sign=1.0)
            r_neg = curve_residual(m, -r, subkey, sign=-1.0)
            return r_pos, r_neg

        pos_curved, neg_curved = jax.lax.map(curve_pair, (linear_residuals, subkeys))
        return jnp.concatenate([pos_curved, neg_curved], axis=0)

    def kl_vg(m, residuals):
        vals, grads = jax.vmap(lambda r: H_vg(m + r))(residuals)
        return jnp.mean(vals), jnp.mean(grads, axis=0)

    def kl_metric(m, residuals, v):
        return jnp.mean(jax.vmap(lambda r: metric_vec(m + r, v))(residuals), axis=0)

    def vi_nonlinear_step(m, subkey, n_samp):
        sample_keys = jax.random.split(subkey, n_samp)
        residuals = draw_nonlinear_residuals(m, sample_keys)

        def ncg_body(carry):
            m_cur, prev_val, info, i = carry
            i = i + 1
            val, grad = kl_vg(m_cur, residuals)
            step = _cg_solve(
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

        val0, _ = kl_vg(m, residuals)
        result = jax.lax.while_loop(
            lambda s: s[2] < -1,
            ncg_body,
            (m, val0, jnp.int32(-2), jnp.int32(0)),
        )
        return result[0], result[1]

    def run_vi_nonlinear(init_pos, vi_key, n_iter, n_samp, rtol):
        keys = jax.random.split(vi_key, n_iter)

        def cond_fn(state):
            _m, _prev_kl, i, converged = state
            return (~converged) & (i < n_iter)

        def body_fn(state):
            m, prev_kl, i, converged = state
            subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
            m_new, kl_val = vi_nonlinear_step(m, subkey, n_samp)
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < rtol) & (i >= 5)
            return (m_new, kl_val, i + 1, converged)

        m0, kl0 = vi_nonlinear_step(init_pos, keys[0], n_samp)
        init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))
        m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return m_final, n_iters

    run_native_vi_nonlinear_jit = jax.jit(run_vi_nonlinear, static_argnames=("n_iter", "n_samp"))
    draw_nonlinear_residuals_jit = jax.jit(draw_nonlinear_residuals)

    return run_native_vi_nonlinear_jit, draw_nonlinear_residuals_jit, hamiltonian


def build_native_vi_catalog_linear_engine(signal_response, flatten, unflatten):
    """Like ``build_native_vi_linear_engine`` but ``data``/``noise`` are runtime arguments.

    Enables ``jax.vmap`` over independent per-galaxy fits in :class:`CatalogFitter`.
    The forward model (``signal_response``) is shared across all galaxies; each
    galaxy's observed data and noise are passed at call time.

    Parameters
    ----------
    signal_response: callable
        ``(pytree) -> ndarray, shape (n_data,)``. Must NOT capture any galaxy-specific
        data, depends only on the model and parameter structure.
    flatten: callable
        ``pytree -> 1D ndarray``.
    unflatten: callable
        ``1D ndarray -> pytree``.

    Returns
    -------
    run_fn: callable
        ``(init_flat, vi_key, data, noise, n_iter, n_samp, rtol) -> (best_flat, n_iters)``.
        JIT-compiled; vmappable over ``(init_flat, vi_key, data, noise)``.
    draw_fn: callable
        ``(pos_flat, subkeys, noise) -> residuals``, shape ``(n_samples, d_total)``.
        JIT-compiled; vmappable.
    hamiltonian_fn: callable
        ``(flat, data, noise) -> scalar``. Vmappable for seed selection.

    Notes
    -----
    Inner functions (metric_vec, draw_residuals, etc.) close over
    ``sqrt_noise_inv`` computed from the runtime ``noise`` argument.  JAX traces through
    these Python closures, so the traced values become XLA graph nodes, making the
    returned callables fully vmappable across different galaxies.

    The Fisher metric at position :math:`\\xi` in the linearized approximation is

    .. math::

        M(\\xi) = J^T \\Sigma^{-1} J + I,

    where :math:`J` is the Jacobian of ``signal_response``, :math:`\\Sigma =
    \\mathrm{diag}(\\sigma_i^2)` is the per-band noise covariance, and :math:`I` is the
    prior precision.  This is identical to the per-galaxy MGVI metric; the catalog
    variant makes :math:`\\Sigma` a runtime argument.

    **JIT-compatible** and **vmap-compatible**: the returned callables are pre-JIT'd and
    safe to wrap with :func:`jax.vmap` over ``(init_flat, vi_key, data, noise)``.

    References
    ----------
    .. [1] Knollmüller, J. & Enßlin, T. A. (2019). "Metric Gaussian Variational
       Inference." arXiv:1901.11033.
    """

    def hamiltonian_fn(xi, data, noise):
        pred = signal_response(unflatten(xi))
        chi2 = jnp.sum(standardized_residual(data, pred, noise) ** 2)
        return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

    def _metric_vec_with_noise(xi, v, noise):
        # (J/sigma)^T (J/sigma) v + v -- never forms 1/sigma**2 (#1206).
        xi_d, v_d = unflatten(xi), unflatten(v)
        _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
        _, vjp_fn = jax.vjp(signal_response, xi_d)
        return flatten(vjp_fn(whiten(whiten(Jv, noise), noise))[0]) + v

    def run_fn(init_pos, vi_key, data, noise, n_iter, n_samp, rtol):
        sqrt_noise_inv = inv_noise_std(noise)
        H_vg = jax.value_and_grad(lambda xi: hamiltonian_fn(xi, data, noise))

        def metric_vec(xi, v):
            return _metric_vec_with_noise(xi, v, noise)

        def draw_residuals(pos_f, subkeys):
            def draw_one(subkey):
                k1, k2 = jax.random.split(subkey)
                eta_pr = jax.random.normal(k1, shape=pos_f.shape)
                eta_lh = jax.random.normal(k2, shape=noise.shape)
                _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
                jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
                return _cg_solve(
                    lambda v: metric_vec(pos_f, v),
                    jt + eta_pr,
                    eta_pr,
                    maxiter=30,
                    miniter=6,
                    absdelta=1e-4,
                )

            return jax.vmap(draw_one)(subkeys)

        def kl_vg(m, residuals):
            vals, grads = jax.vmap(lambda r: H_vg(m + r))(residuals)
            return jnp.mean(vals), jnp.mean(grads, axis=0)

        def kl_metric(m, residuals, v):
            return jnp.mean(jax.vmap(lambda r: metric_vec(m + r, v))(residuals), axis=0)

        def vi_linear_step(m, subkey, n_samp):
            sample_keys = jax.random.split(subkey, n_samp)
            residuals = draw_residuals(m, sample_keys)
            residuals = jnp.concatenate([residuals, -residuals], axis=0)

            def ncg_body(carry):
                m_cur, prev_val, info, i = carry
                i = i + 1
                val, grad = kl_vg(m_cur, residuals)
                step = _cg_solve(
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

            val0, _ = kl_vg(m, residuals)
            result = jax.lax.while_loop(
                lambda s: s[2] < -1,
                ncg_body,
                (m, val0, jnp.int32(-2), jnp.int32(0)),
            )
            return result[0], result[1]

        keys = jax.random.split(vi_key, n_iter)

        def cond_fn(state):
            _m, _prev_kl, i, converged = state
            return (~converged) & (i < n_iter)

        def body_fn(state):
            m, prev_kl, i, converged = state
            subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
            m_new, kl_val = vi_linear_step(m, subkey, n_samp)
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < rtol) & (i >= 5)
            return (m_new, kl_val, i + 1, converged)

        m0, kl0 = vi_linear_step(init_pos, keys[0], n_samp)
        init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))
        m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return m_final, n_iters

    def draw_fn(pos_f, subkeys, noise):
        sqrt_noise_inv = inv_noise_std(noise)

        def draw_one(subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=pos_f.shape)
            eta_lh = jax.random.normal(k2, shape=noise.shape)
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
            return _cg_solve(
                lambda v: _metric_vec_with_noise(pos_f, v, noise),
                jt + eta_pr,
                eta_pr,
                maxiter=30,
                miniter=6,
                absdelta=1e-4,
            )

        return jax.vmap(draw_one)(subkeys)

    run_jit = jax.jit(run_fn, static_argnames=("n_iter", "n_samp"))
    draw_jit = jax.jit(draw_fn)
    return run_jit, draw_jit, hamiltonian_fn


def build_native_vi_catalog_nonlinear_engine(signal_response, flatten, unflatten):
    """Like ``build_native_vi_nonlinear_engine`` but ``data``/``noise`` are runtime arguments.

    Enables ``jax.vmap`` over independent per-galaxy geoVI fits in
    :class:`CatalogFitter`. The forward model is shared; per-galaxy data and noise
    are passed at call time.

    Parameters
    ----------
    signal_response: callable
        ``(pytree) -> ndarray, shape (n_data,)``. Must not capture galaxy data.
    flatten: callable
        ``pytree -> 1D ndarray``.
    unflatten: callable
        ``1D ndarray -> pytree``.

    Returns
    -------
    run_fn: callable
        ``(init_flat, vi_key, data, noise, n_iter, n_samp, rtol) -> (best_flat, n_iters)``.
        JIT-compiled; vmappable over ``(init_flat, vi_key, data, noise)``.
    draw_fn: callable
        ``(pos_flat, subkeys, noise) -> residuals``, shape ``(2*n_keys, d_total)``.
        Each key produces one mirrored pair (geoVI). JIT-compiled; vmappable.
    hamiltonian_fn: callable
        ``(flat, data, noise) -> scalar``. Vmappable.

    Notes
    -----
    Implements the geometric Variational Inference (geoVI) algorithm, which uses a
    non-linear coordinate transformation to account for posterior curvature.  The
    transformation :math:`T(\\xi)` maps the standardized parameter space to the
    likelihood-weighted data space:

    .. math::

        T(\\xi) = \\Sigma^{-1/2} f(\\xi),

    where :math:`f` is ``signal_response`` and :math:`\\Sigma^{-1/2} =
    \\mathrm{diag}(1/\\sigma_i)`.  Residual samples drawn from ``draw_fn`` come in
    mirrored pairs ``(+r, -r)`` to reduce variance in the KL estimate; ``draw_fn``
    therefore returns ``2 * n_keys`` residuals for ``n_keys`` input subkeys.

    Identical algorithm to ``build_native_vi_nonlinear_engine``. The only structural
    difference is that ``sqrt_noise_inv`` and ``n_data`` are derived
    from the runtime ``noise`` argument inside each callable rather than being captured
    in the outer closure.

    **JIT-compatible** and **vmap-compatible**: the returned callables are pre-JIT'd and
    safe to wrap with :func:`jax.vmap` over ``(init_flat, vi_key, data, noise)``.

    References
    ----------
    .. [1] Frank, P. et al. (2021). "Geometric Variational Inference." Entropy 23(7), 853.
       doi:10.3390/e23070853.
    """

    def hamiltonian_fn(xi, data, noise):
        pred = signal_response(unflatten(xi))
        chi2 = jnp.sum(standardized_residual(data, pred, noise) ** 2)
        return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

    def _metric_vec_with_noise(xi, v, noise):
        # (J/sigma)^T (J/sigma) v + v -- never forms 1/sigma**2 (#1206).
        xi_d, v_d = unflatten(xi), unflatten(v)
        _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
        _, vjp_fn = jax.vjp(signal_response, xi_d)
        return flatten(vjp_fn(whiten(whiten(Jv, noise), noise))[0]) + v

    def _make_geometry(sqrt_noise_inv):
        """Return geometry helpers closed over a single sqrt_noise_inv trace."""

        def transformation(xi):
            return sqrt_noise_inv * signal_response(unflatten(xi))

        def left_sqrt_metric(xi, v_data):
            _, vjp_fn = jax.vjp(signal_response, unflatten(xi))
            return flatten(vjp_fn(sqrt_noise_inv * v_data)[0])

        def right_sqrt_metric(xi, v_param):
            _, Jv = jax.jvp(signal_response, (unflatten(xi),), (unflatten(v_param),))
            return sqrt_noise_inv * Jv

        return transformation, left_sqrt_metric, right_sqrt_metric

    def run_fn(init_pos, vi_key, data, noise, n_iter, n_samp, rtol):
        sqrt_noise_inv = inv_noise_std(noise)
        n_data = noise.shape[0]
        H_vg = jax.value_and_grad(lambda xi: hamiltonian_fn(xi, data, noise))
        transformation, left_sqrt_metric, right_sqrt_metric = _make_geometry(sqrt_noise_inv)

        def metric_vec(xi, v):
            return _metric_vec_with_noise(xi, v, noise)

        def draw_metric_sample(xi, subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=xi.shape)
            eta_lh = jax.random.normal(k2, shape=(n_data,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(xi))
            jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
            return jt + eta_pr

        def draw_linear_residual(pos_f, subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=pos_f.shape)
            eta_lh = jax.random.normal(k2, shape=(n_data,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
            return _cg_solve(
                lambda v: metric_vec(pos_f, v),
                jt + eta_pr,
                eta_pr,
                maxiter=30,
                miniter=6,
                absdelta=1e-4,
            )

        def curve_residual(m, r_linear, metric_key, sign):
            x0 = m + r_linear
            ms = sign * draw_metric_sample(m, metric_key)
            trafo_at_m = transformation(m)

            def phi_vg(x):
                delta_trafo = transformation(x) - trafo_at_m
                g_x = (x - m) + left_sqrt_metric(m, delta_trafo)
                r = ms - g_x
                val = 0.5 * jnp.dot(r, r)
                ngrad = r + left_sqrt_metric(x, right_sqrt_metric(m, r))
                return val, -ngrad

            def phi_metric(x, v):
                tm = left_sqrt_metric(m, right_sqrt_metric(x, v)) + v
                return left_sqrt_metric(x, right_sqrt_metric(m, tm)) + tm

            def sampnorm(natgrad):
                fpp = right_sqrt_metric(m, natgrad)
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

        def draw_nonlinear_residuals(m, subkeys):
            linear_residuals = jax.lax.map(lambda sk: draw_linear_residual(m, sk), subkeys)

            def curve_pair(args):
                r, subkey = args
                return curve_residual(m, r, subkey, 1.0), curve_residual(m, -r, subkey, -1.0)

            pos_curved, neg_curved = jax.lax.map(curve_pair, (linear_residuals, subkeys))
            return jnp.concatenate([pos_curved, neg_curved], axis=0)

        def kl_vg(m, residuals):
            vals, grads = jax.lax.map(lambda r: H_vg(m + r), residuals)
            return jnp.mean(vals), jnp.mean(grads, axis=0)

        def kl_metric(m, residuals, v):
            return jnp.mean(jax.lax.map(lambda r: metric_vec(m + r, v), residuals), axis=0)

        def vi_nonlinear_step(m, subkey, n_samp):
            sample_keys = jax.random.split(subkey, n_samp)
            residuals = draw_nonlinear_residuals(m, sample_keys)

            def ncg_body(carry):
                m_cur, prev_val, info, i = carry
                i = i + 1
                val, grad = kl_vg(m_cur, residuals)
                step = _cg_solve(
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

            val0, _ = kl_vg(m, residuals)
            result = jax.lax.while_loop(
                lambda s: s[2] < -1,
                ncg_body,
                (m, val0, jnp.int32(-2), jnp.int32(0)),
            )
            return result[0], result[1]

        keys = jax.random.split(vi_key, n_iter)

        def cond_fn(state):
            _m, _prev_kl, i, converged = state
            return (~converged) & (i < n_iter)

        def body_fn(state):
            m, prev_kl, i, converged = state
            subkey = jax.lax.dynamic_index_in_dim(keys, i, keepdims=False)
            m_new, kl_val = vi_nonlinear_step(m, subkey, n_samp)
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < rtol) & (i >= 5)
            return (m_new, kl_val, i + 1, converged)

        m0, kl0 = vi_nonlinear_step(init_pos, keys[0], n_samp)
        init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))
        m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return m_final, n_iters

    def draw_fn(pos_f, subkeys, noise):
        sqrt_noise_inv = inv_noise_std(noise)
        n_data = noise.shape[0]
        transformation, left_sqrt_metric, right_sqrt_metric = _make_geometry(sqrt_noise_inv)

        def draw_metric_sample(xi, subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=xi.shape)
            eta_lh = jax.random.normal(k2, shape=(n_data,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(xi))
            jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
            return jt + eta_pr

        def draw_linear_residual(subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=pos_f.shape)
            eta_lh = jax.random.normal(k2, shape=(n_data,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_noise_inv * eta_lh)[0])
            return _cg_solve(
                lambda v: _metric_vec_with_noise(pos_f, v, noise),
                jt + eta_pr,
                eta_pr,
                maxiter=30,
                miniter=6,
                absdelta=1e-4,
            )

        def curve_residual(m, r_linear, metric_key, sign):
            x0 = m + r_linear
            ms = sign * draw_metric_sample(m, metric_key)
            trafo_at_m = transformation(m)

            def phi_vg(x):
                delta_trafo = transformation(x) - trafo_at_m
                g_x = (x - m) + left_sqrt_metric(m, delta_trafo)
                r = ms - g_x
                val = 0.5 * jnp.dot(r, r)
                ngrad = r + left_sqrt_metric(x, right_sqrt_metric(m, r))
                return val, -ngrad

            def phi_metric(x, v):
                tm = left_sqrt_metric(m, right_sqrt_metric(x, v)) + v
                return left_sqrt_metric(x, right_sqrt_metric(m, tm)) + tm

            def sampnorm(natgrad):
                fpp = right_sqrt_metric(m, natgrad)
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

        linear_residuals = jax.lax.map(draw_linear_residual, subkeys)

        def curve_pair(args):
            r, subkey = args
            return curve_residual(pos_f, r, subkey, 1.0), curve_residual(pos_f, -r, subkey, -1.0)

        pos_curved, neg_curved = jax.lax.map(curve_pair, (linear_residuals, subkeys))
        return jnp.concatenate([pos_curved, neg_curved], axis=0)

    run_jit = jax.jit(run_fn, static_argnames=("n_iter", "n_samp"))
    draw_jit = jax.jit(draw_fn)
    return run_jit, draw_jit, hamiltonian_fn
