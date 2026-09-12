#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Does a fixed dense Laplace metric get a D = 7-9 photometry posterior under 20 s?

This is built directly on BlackJAX 1.6.2 against the project's own log-posterior
(:class:`~tengri.inference.context.InferenceContext`), not through
``Fitter.run("mcmc_nuts")``: the shipped path always re-adapts a mass matrix from
scratch, and the hypothesis under test is that a single analytic Hessian at the
MAP -- :func:`~tengri.inference.preconditioning.negative_hessian_metric`, the
same metric NIFTy's MGVI/geoVI use -- makes that re-adaptation unnecessary. The
budget is 20 s of wall clock **including MAP**, to min-ESS >= 100 and split
R-hat < 1.01, on CPU and GPU.

THREE ARMS, ONE HARNESS
========================

``laplace``
    The hypothesis. ``H = negative_hessian_metric(MAP)``, floored SPD; NUTS runs
    with the DENSE ``inverse_mass_matrix = H^-1`` held fixed through warmup and
    sampling. Only the scalar step size is tuned (``find_reasonable_step_size``
    then dual averaging), because the metric is not being estimated, it is
    supplied.
``laplace-diagwin``
    The same Hessian, spent differently: whiten the coordinates with
    ``x = MAP + L z``, ``L = chol(H^-1)``, so the curvature the *sampler* sees in
    z-space is already close to isotropic, and hand BlackJAX's own
    ``window_adaptation`` a DIAGONAL metric to mop up whatever the single-point
    Hessian missed. Draws are mapped back to x for every diagnostic, so this row
    is directly comparable to the other two.
``diag-window``
    The control. Plain ``window_adaptation`` (diagonal mass matrix) from the MAP,
    no Hessian involved. Run through the exact same jitted adapt/sample harness
    as the other two arms so the seconds are comparable -- the shipped
    ``Fitter.run("mcmc_nuts")`` path is a different call stack entirely and would
    not isolate the metric as the only variable.

Every arm pays MAP and adaptation itself; nothing is precomputed once and reused
across arms, because the walls this script reports are exactly the walls a user
pays end to end.

COLD VS WARM, AND WHY BOTH ARE REPORTED
========================================

The 20 s budget is agreed to be warm-compile, but a benchmark that only prints
warm numbers hides the one-time XLA compilation cost from whoever reads it
later. MAP, then (metric + adaptation) then sampling are each called TWICE with
the identical key: the first call pays trace + compile + execute, the second
pays execute only. Both are recorded (``*_wall_cold`` / ``*_wall_warm``); the
metric itself (:func:`negative_hessian_metric`) is timed once as
``hessian_wall`` since it does not depend on the sampler's RNG key.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_laplace_nuts_20s.py \\
        --notebook ctl-dpl --seed 7 --arm laplace --n-chains 8 \\
        --json bench/results/2026-09-11_laplace_nuts_20s.jsonl

    # six seeds, one fit per subprocess so nothing (adaptation caches, XLA state)
    # leaks between them
    .venv/bin/python bench/scripts/benchmark_laplace_nuts_20s.py \\
        --notebook ctl-dpl --arm laplace --seeds 6
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")


def _prescan_arg(argv: list[str], flag: str, default, cast=str):
    """Read one ``--flag value`` / ``--flag=value`` out of argv, before argparse runs."""
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return cast(argv[i + 1])
        if tok.startswith(flag + "="):
            return cast(tok.split("=", 1)[1])
    return default


def _maybe_set_pmap_device_count(argv: list[str]) -> None:
    """Force XLA to expose one CPU device per chain, before ``jax`` is imported.

    ``jax.pmap`` needs as many devices as the axis it maps over; on CPU that
    means XLA's ``--xla_force_host_platform_device_count`` flag, which is only
    read at JAX's first import and cannot be changed afterwards. Rather than
    requiring the caller to export ``XLA_FLAGS`` themselves, this does a
    minimal pre-parse of ``sys.argv`` for ``--chain-parallel``/``--adapt-per-chain``
    and ``--n-chains`` (mirroring their argparse defaults below) and sets the
    flag before the real argument parser -- or ``jax`` -- ever runs. This is
    the simplest of the two options the spec allows; the other (requiring an
    exported ``XLA_FLAGS`` and just checking ``len(jax.devices())``) would
    push the bookkeeping onto every caller instead of the script.
    """
    chain_parallel = _prescan_arg(argv, "--chain-parallel", "vmap")
    adapt_per_chain = "--adapt-per-chain" in argv
    if chain_parallel != "pmap" and not adapt_per_chain:
        return
    n_chains = _prescan_arg(argv, "--n-chains", 8, int)
    flag = f"--xla_force_host_platform_device_count={n_chains}"
    existing = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


def _apply_precision_env(argv: list[str]) -> None:
    """Translate ``--float32`` into ``JAX_ENABLE_X64``, before ``jax`` is imported.

    Precision is a process-global JAX setting that must be chosen *before*
    ``import tengri``, not merely before the model is built: ``tengri/__init__.py``
    turns ``jax_enable_x64`` back **on** at import unless ``JAX_ENABLE_X64`` is
    already present in the environment (#1840), and by then several DSPS modules
    have already allocated float64 module-scope constants (#1880) that a later
    ``jax.config.update("jax_enable_x64", False)`` cannot un-allocate -- measured
    here as an exactly-zero gradient through one such stale-float64 constant, a
    silent-corruption failure mode rather than a raised error. Setting the
    variable here, from this script's own argv before ``jax`` exists, also arms
    tengri's own import-time x64 guard, which is what stops those DSPS modules
    allocating float64 in the first place. This mirrors
    ``benchmark_catalog_throughput.py``'s ``_apply_precision_env`` exactly -- the
    mechanism ``bench/reports/2026-08-31_float32_fitting_path.md`` actually
    validated, not the post-import ``jax.config.update`` this module used before.

    ``JAX_DEFAULT_MATMUL_PRECISION=highest`` rides along: on Ampere+ CUDA, XLA
    otherwise lowers float32 matmuls to TF32 (10-bit mantissa), worth 4.5% on
    parameter error bars per the same report.
    """
    if "--float32" in argv:
        os.environ.setdefault("JAX_ENABLE_X64", "0")
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")


_apply_precision_env(sys.argv[1:])
_maybe_set_pmap_device_count(sys.argv[1:])

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import jax.scipy.special
import jax.scipy.stats
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation, find_reasonable_step_size
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from jax import lax
from jax.flatten_util import ravel_pytree

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import benchmark_notebook_sampler as B

import tengri
from tengri import Data, ForwardModel, generate_mock
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.preconditioning import negative_hessian_metric

ARMS = ("laplace", "laplace-diagwin", "laplace-densewin", "diag-window", "dense-window")

#: Every key a JSONL row carries, in the order the spec lists them. Populated
#: with ``None`` up front so a raised exception still writes a complete row --
#: ``error`` set, everything downstream of the failure null -- rather than a
#: row with fewer keys than a converged one.
ROW_FIELDS = [
    "notebook",
    "seed",
    "arm",
    "dtype",
    "platform",
    "devices",
    "taskset",
    "git_sha",
    "timestamp",
    "D",
    "D_sampled",
    "free_names",
    "n_chains",
    "chain_parallel",
    "n_devices",
    "adapt_per_chain",
    "n_warmup",
    "n_samples",
    "n_burnin",
    "map_restarts",
    "map_steps",
    "map_optimizer",
    "warmup_max_doublings",
    "window_start",
    "jitter",
    "profile_mass",
    "map_nlp",
    "map_wall_cold",
    "map_wall_warm",
    "hessian_wall",
    "hess_eig_min",
    "hess_eig_max",
    "hess_cond",
    "step_size",
    "adapt_wall_cold",
    "adapt_wall_warm",
    "sample_wall_cold",
    "sample_wall_warm",
    "wall_total_warm",
    "wall_total_cold",
    "n_grad_adapt",
    "n_grad_sample",
    "n_grad_total",
    "grad_per_draw",
    "tree_depth_mean",
    "divergences",
    "div_frac",
    "accept_mean",
    "min_ess",
    "worst_param",
    "ess",
    "ess_phys_min",
    "rhat_max",
    "rhat",
    "unique_frac",
    "converged",
    "grads_to_100",
    "sec_to_100",
    "truth",
    "post_mean",
    "post_std",
    "z_truth",
    "nonfinite",
    "peak_rss_mb",
    "error",
]


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_HERE, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def _to_native(obj):
    """Recursively coerce numpy/JAX scalars so ``json.dumps`` does not choke."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "item") and np.ndim(obj) == 0:
        return obj.item()
    return obj


def _split_rhat(chains_by_samples: np.ndarray) -> float:
    """Potential scale reduction after splitting each chain in half.

    Doubles the chain count and halves the sample count, so a chain that has
    not mixed within itself shows up as disagreement between its own two
    halves, not just between chains (:cite:`stan_rhat`, informally "split
    R-hat").
    """
    n = chains_by_samples.shape[1]
    half = n // 2
    split = np.concatenate(
        [chains_by_samples[:, :half], chains_by_samples[:, half : 2 * half]], axis=0
    )
    return float(potential_scale_reduction(jnp.asarray(split)))


def _unique_fraction(positions: np.ndarray) -> float:
    """Mean, over chains, of the fraction of draws that moved from the prior one.

    Zero divergences is not evidence of health (#1999): a chain that rejects
    every proposal reports no divergence and an R-hat near 1 (both halves have
    zero variance). This is the column that catches a frozen chain instead.
    """
    moved = np.any(positions[:, 1:, :] != positions[:, :-1, :], axis=-1)
    return float(np.mean(moved.mean(axis=1)))


def _verify_precision(dtype: str) -> None:
    """Confirm the process is actually in ``dtype``; raise rather than fit silently wrong.

    The switch itself was thrown at module scope by ``_apply_precision_env``, before
    ``jax`` was ever imported -- see its docstring for why a *later*
    ``jax.config.update`` cannot be trusted to do this. This proves the precision on
    a real array's dtype, mirroring ``benchmark_catalog_throughput.py``'s
    ``set_precision``, because a measurement that cannot fail is not a measurement.
    """
    want = "float32" if dtype == "float32" else "float64"
    probe = jnp.zeros(1) + 1.0 if dtype == "float32" else jnp.zeros(1, dtype=jnp.float64) + 1.0
    if str(probe.dtype) != want:
        raise RuntimeError(
            f"dtype={dtype!r} asked for {want} but jnp allocates {probe.dtype}. "
            f"JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')!r}, "
            f"jax_enable_x64={jax.config.jax_enable_x64}. Run this script as its own "
            "process (it sets JAX_ENABLE_X64 from its own argv at module scope, "
            "before jax is imported) rather than calling into it from a session "
            "already latched to the other precision."
        )


def _build_problem(notebook: str, seed: int, dtype: str = "float64"):
    """Mock, forward model and inference context for one fit, mirroring ``run_one``.

    ``dtype`` was already applied at module scope by ``_apply_precision_env``
    (``JAX_ENABLE_X64`` set before ``jax``/``tengri`` were imported -- the
    validated route, see that function's docstring and
    ``bench/reports/2026-08-31_float32_fitting_path.md``); this only verifies it
    took, and additionally casts ``flux``/``noise`` explicitly as belt-and-braces.
    """
    _verify_precision(dtype)
    cfg = B.NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    truth = sed.spec.sample(key_truth)
    mock = generate_mock(sed, truth, key=key_mock, snr=cfg["snr"])
    np_dtype = np.float32 if dtype == "float32" else np.float64
    flux = np.asarray(mock["flux_obs"], dtype=np_dtype)
    noise = np.asarray(mock["noise"], dtype=np_dtype)
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    # The harness marginalizes the mass itself (``--profile-mass`` below) and
    # needs the FULL-D context to do it: since #2281 ``profile_mass="auto"`` is
    # the library default and would fix the mass in the spec before this
    # context is built, leaving ``_mass_free_name`` nothing to find.
    fitter = Fitter(forward, data=flux, noise=noise, profile_mass=False)
    ctx = InferenceContext.from_target(fitter)
    return cfg, forward, ctx, flux, noise, truth, key_fit


def _build_kernel_runner(
    logdensity_fn,
    n_warmup: int,
    n_samples: int,
    n_burnin: int,
    chain_parallel: str = "vmap",
    adapt_per_chain: bool = False,
):
    """One (v/p)mappable per-chain NUTS loop, shared by every arm's sampling phase.

    ``chain_parallel="vmap"`` (the default) batches the ``n_chains`` chains on
    one CPU device, as before -- the sampling wall then scales with
    ``n_chains``. ``chain_parallel="pmap"`` instead runs every chain on its
    own device (the caller must have arranged ``n_chains`` JAX devices, see
    ``_maybe_set_pmap_device_count``), so the chains sample concurrently.

    ``adapt_per_chain`` maps ``step_size``/``inverse_mass_matrix`` over the
    chain axis too (each chain samples with its own adaptation) instead of
    broadcasting one shared pair to every chain.
    """
    kernel = blackjax.nuts.build_kernel()

    def run_chain(key, start, step_size, inverse_mass_matrix):
        state0 = blackjax.nuts.init(start, logdensity_fn)
        keys = jax.random.split(key, n_burnin + n_samples)

        def step(state, k):
            new_state, info = kernel(k, state, logdensity_fn, step_size, inverse_mass_matrix)
            y = (
                new_state.position,
                info.is_divergent,
                info.num_integration_steps,
                info.acceptance_rate,
            )
            return new_state, y

        _, (positions, divergent, n_steps, accept) = lax.scan(step, state0, keys)
        return positions[n_burnin:], divergent[n_burnin:], n_steps[n_burnin:], accept[n_burnin:]

    in_axes = (0, 0, 0, 0) if adapt_per_chain else (0, 0, None, None)

    if chain_parallel == "pmap":
        run = jax.pmap(run_chain, in_axes=in_axes)

        def sample_impl(step_size, inverse_mass_matrix, starts, keys):
            return run(keys, starts, step_size, inverse_mass_matrix)

        return kernel, sample_impl, n_warmup

    run = jax.vmap(run_chain, in_axes=in_axes)

    def sample_impl(step_size, inverse_mass_matrix, starts, keys):
        return run(keys, starts, step_size, inverse_mass_matrix)

    return kernel, jax.jit(sample_impl), n_warmup


def _laplace_adapt_fn(logdensity_fn, inverse_mass_matrix, flat0, target_accept, n_warmup):
    """Dense metric held fixed; only the scalar step size is tuned."""
    kernel = blackjax.nuts.build_kernel()
    init_state = blackjax.nuts.init(flat0, logdensity_fn)

    def adapt_impl(key):
        key_reasonable, key_da = jax.random.split(key)

        def kernel_generator(step_size):
            def step(k, state):
                return kernel(k, state, logdensity_fn, step_size, inverse_mass_matrix)

            return step

        step0 = find_reasonable_step_size(
            key_reasonable, kernel_generator, init_state, 1.0, target_accept
        )
        da_init, da_update, da_final = dual_averaging_adaptation(target_accept)
        da_state0 = da_init(step0)
        keys = jax.random.split(key_da, n_warmup)

        def one_step(carry, k):
            state, da_state = carry
            step_size = jnp.exp(da_state.log_step_size)
            new_state, info = kernel(k, state, logdensity_fn, step_size, inverse_mass_matrix)
            new_da_state = da_update(da_state, info.acceptance_rate)
            return (new_state, new_da_state), info.num_integration_steps

        (_, final_da_state), n_int_steps = lax.scan(one_step, (init_state, da_state0), keys)
        return da_final(final_da_state), inverse_mass_matrix, jnp.sum(n_int_steps)

    return jax.jit(adapt_impl)


def _window_adapt_fn(
    logdensity_fn,
    position0,
    target_accept,
    n_warmup,
    diagonal=True,
    max_doublings=10,
    adapt_per_chain=False,
):
    """BlackJAX's own window adaptation (diagonal or dense mass matrix).

    ``max_doublings`` caps the warmup trees only: sampling stays at BlackJAX's
    default depth 10, the split ``bench/reports/2026-09-06_photometry_20s.md``
    Finding 6 could not express on the catalog path.

    By default (``adapt_per_chain=False``) this adapts ONE chain: ``position0``
    is a single ``(D,)`` start and the returned function takes one key. With
    ``adapt_per_chain=True``, ``position0`` is instead ``(n_chains, D)`` -- the
    same per-chain starts sampling will use -- and the returned function is
    ``jax.pmap``ped over the chain axis: every chain runs its own warmup, with
    its own key, to its own ``(step_size, inverse_mass_matrix)``, all in
    parallel. That is the standard multi-chain adaptation protocol -- chains
    adapt independently, then get diagnosed together afterwards with split
    R-hat across chains, exactly what ``_score`` already computes -- rather
    than the single shared adaptation the default reuses across every chain.
    """

    def adapt_impl(key, pos0):
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=diagonal,
            target_acceptance_rate=target_accept,
            max_num_doublings=max_doublings,
        )
        results, info = warmup.run(key, pos0, n_warmup)
        n_grad_adapt = jnp.sum(info.info.num_integration_steps)
        return (
            results.parameters["step_size"],
            results.parameters["inverse_mass_matrix"],
            (n_grad_adapt),
        )

    if adapt_per_chain:
        run = jax.pmap(adapt_impl, in_axes=(0, 0))

        def call(keys):
            return run(keys, position0)

        return call

    return jax.jit(lambda key: adapt_impl(key, position0))


# ── --profile-mass: analytic marginalization of the linear mass amplitude ────
#
# Photometry is linear in total stellar mass: flux_i(theta, M) = M * f_i(theta)
# with M = 10 ** log_total_mass. At fixed theta (the other D-1 parameters) the
# mass therefore has a closed-form Gaussian conditional posterior under the
# Gaussian photometric likelihood -- the standard result for marginalizing a
# linear amplitude out of a Gaussian likelihood (see e.g. D. S. Sivia and
# J. Skilling, *Data Analysis: A Bayesian Tutorial*, 2nd ed., Oxford University
# Press (2006), Sec. 3.2). Marginalizing it out removes the mass/age ridge
# that makes every metric tried on the full D-dimensional posterior need
# 34-550 gradients per draw, without a single extra forward-model evaluation
# inside the marginalization itself: the integral over the flat-in-log10(mass)
# prior is a fixed-size trapezoid quadrature over the *exact* quadratic
# chi^2(M), not additional model calls.
MASS_QUAD_NODES = 48
MASS_QUAD_HALF_WIDTH_SIGMAS = 8.0


def _mass_free_name(names: list[str]) -> str:
    """Name of the total-stellar-mass free parameter, e.g. ``sfh_dpl_log_total_mass``."""
    (name,) = [n for n in names if n.endswith("log_total_mass")]
    return name


def _mass_index(pu: dict, mass_name: str) -> tuple[int, np.ndarray]:
    """Index of ``mass_name`` within ``ravel_pytree(pu)``, and every other index.

    ``ravel_pytree`` flattens a dict pytree in whatever order
    ``jax.tree_util.tree_flatten`` visits it; asking the same traversal for its
    *paths* gives the exact slot the mass parameter lands at without assuming
    anything about dict insertion order versus JAX's internal pytree order.
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(pu)
    key_order = [path[0].key for path, _ in leaves_with_path]
    mass_idx = key_order.index(mass_name)
    non_mass_idx = np.array([i for i in range(len(key_order)) if i != mass_idx])
    return mass_idx, non_mass_idx


def _verify_mass_linearity(ctx, mass_name: str, phys_map: dict) -> float:
    """Photometry must be exactly linear in mass; prints max |flux ratio - 10|."""

    def phys_at(log_m):
        return {**phys_map, mass_name: jnp.asarray(log_m)}

    pred9 = ctx.model.predict_photometry(phys_at(9.0))
    pred10 = ctx.model.predict_photometry(phys_at(10.0))
    pred11 = ctx.model.predict_photometry(phys_at(11.0))
    ratios = jnp.concatenate([pred10 / pred9, pred11 / pred10])
    max_dev = float(jnp.max(jnp.abs(ratios - 10.0)))
    print(f"[profile-mass] linearity check: max|flux_ratio(+1 dex) - 10| = {max_dev:.3e}")
    return max_dev


def _verify_gaussian_likelihood(
    ctx, nlp, dargs, unravel, flat0, flux: np.ndarray, noise: np.ndarray, n_points: int = 3
) -> float:
    """Compare ``-nlp - log_prior`` to a hand-rolled 1/2 chi^2 at a few random points."""
    max_diff = 0.0
    for i in range(n_points):
        jitter = jax.random.normal(jax.random.PRNGKey(9000 + i), flat0.shape) * 0.3
        x = flat0 + jitter
        pu_i = unravel(x)
        analytic = -float(nlp(pu_i, dargs)) - float(ctx.log_prior_fn(pu_i))
        phys_i = ctx.to_physical(pu_i)
        pred_i = np.asarray(ctx.model.predict_photometry(phys_i))
        manual = -0.5 * float(np.sum(((flux - pred_i) / noise) ** 2))
        diff = analytic - manual
        max_diff = max(max_diff, abs(diff))
        print(
            f"[profile-mass] likelihood check point {i}: analytic={analytic:.6f} "
            f"manual={manual:.6f} diff={diff:.3e}"
        )
    print(f"[profile-mass] likelihood check: max diff over {n_points} points = {max_diff:.3e}")
    return max_diff


def _build_profiled_objects(ctx, unravel, mass_name, mass_idx, non_mass_idx, flux, noise) -> dict:
    """Build the 7-D (D-1) profiled log-density and its post-hoc mass sampler.

    Divides by ``noise`` before squaring/multiplying, never forms
    ``1/noise**2`` as a standalone array -- the same grouping
    :func:`~tengri.inference.likelihoods.gaussian.standardized_residual` makes
    binding on the compiler, and for the same reason: at real photometric
    scales ``noise ~ 1e-26`` so ``noise**2 ~ 1e-52`` and ``1/noise**2 ~ 1e52``
    both leave float32's representable range (max ~3.4e38, min normal
    ~1.2e-38) while ``flux/noise`` is O(1)-O(10) and perfectly representable.
    Forming ``inv_noise2`` directly (as an earlier version of this function
    did) is silently exact in float64 and silently ``inf`` -> ``0 * inf =
    nan`` in float32 (#1535's failure mode, reproduced here rather than only
    in ``src/``).
    """
    D = int(non_mass_idx.size) + 1
    flux_j = jnp.asarray(flux)
    noise_j = jnp.asarray(noise)
    flux_over_noise = flux_j / noise_j
    d2_sum = jnp.sum(flux_over_noise**2)
    non_mass_idx_j = jnp.asarray(non_mass_idx)

    def expand(x_rest, xi_mass):
        """(D-1,) non-mass coordinates + one mass coordinate -> the full (D,) vector."""
        return jnp.zeros(D).at[non_mass_idx_j].set(x_rest).at[mass_idx].set(xi_mass)

    def unit_mass_flux(x_rest):
        """Flux per unit mass at ``x_rest``. Any xi_mass works: the ratio cancels M.

        Only for introspection (returned in the dict below); ``profile_stats``
        does NOT route through this in-line, see its docstring for why.
        """
        phys = ctx.to_physical(unravel(expand(x_rest, 0.0)))
        pred = ctx.model.predict_photometry(phys)
        return pred / 10.0 ** phys[mass_name]

    def profile_stats(x_rest):
        """(A, M*, chi2_min) of the exact quadratic chi^2(M) = chi2_min + A(M-M*)^2.

        Computes ``A = sum((pred/noise)**2) / m0**2`` and ``B = sum((flux/noise)*
        (pred/noise)) / m0`` rather than forming ``f = pred / m0`` as an
        intermediate and squaring/summing *that* (mathematically identical, since
        ``m0`` does not depend on ``x_rest``). ``m0 ~ 1e10`` (a realistic total
        stellar mass) pushes ``pred/m0`` to ~1e-37 -- inside float32's
        representable range but at its ragged edge -- and reverse-mode AD *through*
        further nonlinear ops (squaring, summing) at that scale was measured to
        flush the resulting gradient to exactly 0 for every parameter (silent, not
        NaN). Dividing by ``m0``/``m0**2`` only once, on the aggregate scalar, after
        the sum/square, keeps every autodiff'd intermediate at ``pred/noise``'s
        native O(1)-O(10) scale; ``m0`` enters as an ordinary scalar division at
        the end, whose own derivative is trivial (``m0`` is constant in
        ``x_rest``). ``chi2_min`` does not carry ``m0`` at all: substituting
        ``B = B'/m0``, ``A = A'/m0**2`` into ``d2_sum - B**2/A`` cancels ``m0``
        exactly.
        """
        phys = ctx.to_physical(unravel(expand(x_rest, 0.0)))
        pred = ctx.model.predict_photometry(phys)
        m0 = 10.0 ** phys[mass_name]
        pred_over_noise = pred / noise_j
        a_prime = jnp.sum(pred_over_noise**2)
        b_prime = jnp.sum(flux_over_noise * pred_over_noise)
        A = a_prime / m0**2
        mstar = b_prime * m0 / a_prime
        chi2_min = d2_sum - b_prime**2 / a_prime
        return A, mstar, chi2_min

    def quad_nodes(A, mstar):
        """Fixed-size trapezoid grid in log10(M), +/- MASS_QUAD_HALF_WIDTH_SIGMAS sigma.

        The grid *placement* (``s``/``lo``/``hi``/``dm``/``weights``) is stop-gradiented:
        it is a numerical-quadrature implementation choice, not part of the quantity
        being differentiated, and ``s ~ 1/sqrt(A)`` carries a second derivative
        ~1/A**1.5 that is merely large in float64 (A is tiny -- the photometric
        likelihood pins total mass very precisely) but overflows float32's Hessian
        (measured: ``jax.hessian`` of ``log_mass_integral`` is all-NaN in float32
        with this term live, finite with it stopped). ``log_terms`` below still
        differentiates the *integrand* through the live ``A``/``mstar`` normally --
        only where the nodes sit is frozen, which is standard for a quadrature grid
        chosen by a local (Laplace) width estimate.
        """
        A_grid, mstar_grid = jax.lax.stop_gradient(A), jax.lax.stop_gradient(mstar)
        log10_mstar = jnp.log10(mstar_grid)
        s = 1.0 / (jnp.sqrt(A_grid) * mstar_grid * jnp.log(10.0))
        lo = jnp.maximum(7.0, log10_mstar - MASS_QUAD_HALF_WIDTH_SIGMAS * s)
        hi = jnp.minimum(12.5, log10_mstar + MASS_QUAD_HALF_WIDTH_SIGMAS * s)
        m_nodes = jnp.linspace(lo, hi, MASS_QUAD_NODES)
        dm = (hi - lo) / (MASS_QUAD_NODES - 1)
        weights = jnp.ones(MASS_QUAD_NODES).at[0].set(0.5).at[-1].set(0.5) * dm
        return m_nodes, weights

    def log_mass_integral(A, mstar):
        m_nodes, weights = quad_nodes(A, mstar)
        m_values = 10.0**m_nodes
        log_terms = -0.5 * A * (m_values - mstar) ** 2 + jnp.log(weights)
        return jax.scipy.special.logsumexp(log_terms)

    def profiled_logdensity(x_rest):
        A, mstar, chi2_min = profile_stats(x_rest)
        return -0.5 * chi2_min + log_mass_integral(A, mstar) - 0.5 * jnp.sum(x_rest**2)

    def sample_log_mass(key, A, mstar):
        """Inverse-CDF draw of log10(M) from the same quadrature grid as the density."""
        m_nodes, weights = quad_nodes(A, mstar)
        m_values = 10.0**m_nodes
        log_terms = -0.5 * A * (m_values - mstar) ** 2 + jnp.log(weights)
        probs = jax.nn.softmax(log_terms)
        cdf = jnp.cumsum(probs)
        u = jax.random.uniform(key)
        return jnp.interp(u, cdf / cdf[-1], m_nodes)

    return dict(
        D=D,
        expand=expand,
        unit_mass_flux=unit_mass_flux,
        profile_stats=profile_stats,
        profiled_logdensity=profiled_logdensity,
        sample_log_mass=sample_log_mass,
    )


def _draw_mass_positions(key, positions_rest: np.ndarray, profiled: dict, mass_idx, non_mass_idx):
    """(C, N, D-1) unbounded draws -> (C, N, D) with a post-hoc mass draw reinserted.

    Every (D-1)-dimensional draw gets an independent sample from its own exact
    closed-form conditional ``p(log10(M) | theta, d)``, mapped back through the
    standardized Uniform(7, 12.5) inverse-CDF to the same xi_mass axis the
    non-profiled arms sample in, so :func:`_score` runs unmodified and reports
    ESS/R-hat/z_truth for all D names including mass.
    """
    n_chains, n_samples, d_rest = positions_rest.shape
    flat_rest = jnp.asarray(positions_rest).reshape(-1, d_rest)
    A_all, mstar_all, _ = jax.vmap(profiled["profile_stats"])(flat_rest)
    keys = jax.random.split(key, flat_rest.shape[0])
    log_mass = jax.vmap(profiled["sample_log_mass"])(keys, A_all, mstar_all)
    xi_mass = jax.scipy.stats.norm.ppf((log_mass - 7.0) / 5.5)
    D = profiled["D"]
    positions_full = jnp.zeros((flat_rest.shape[0], D))
    positions_full = positions_full.at[:, non_mass_idx].set(flat_rest)
    positions_full = positions_full.at[:, mass_idx].set(xi_mass)
    return np.asarray(positions_full).reshape(n_chains, n_samples, D)


def _run_arm(args, logdensity_2arg, logdensity_flat, flat0, dargs, key_fit):
    """Metric + adaptation + sampling for one arm; returns everything a row needs.

    Draws come back already mapped to the unbounded x-space the MAP lives in --
    ``laplace-diagwin`` samples in whitened z-space internally but this function
    is the seam that maps every draw back before returning, so callers never see
    the difference.
    """
    key_hessian, key_adapt, key_disperse, key_sample = jax.random.split(key_fit, 4)
    del key_hessian  # the Hessian is deterministic given flat0; no RNG needed

    t0 = time.perf_counter()
    hessian = negative_hessian_metric(logdensity_2arg, flat0, dargs, floor=args.floor)
    jax.block_until_ready(hessian)
    hessian_wall = time.perf_counter() - t0

    eigenvalues = jnp.linalg.eigvalsh(hessian)
    hess_eig_min = float(jnp.min(eigenvalues))
    hess_eig_max = float(jnp.max(eigenvalues))
    hess_cond = hess_eig_max / max(hess_eig_min, 1e-300)
    inverse_hessian = jnp.linalg.inv(hessian)
    laplace_chol = jnp.linalg.cholesky(inverse_hessian)

    D = flat0.size
    zs = jax.random.normal(key_disperse, (args.n_chains, D))

    if args.arm == "laplace":
        if args.adapt_per_chain:
            raise NotImplementedError(
                "--adapt-per-chain is not implemented for the 'laplace' arm: its "
                "dense Hessian metric is supplied once, fixed, not adapted at all"
            )
        adapt_fn = _laplace_adapt_fn(
            logdensity_flat, inverse_hessian, flat0, args.target_accept, args.n_warmup
        )
        starts = flat0[None, :] + args.jitter * (zs @ laplace_chol.T)
        _, sample_fn, _ = _build_kernel_runner(
            logdensity_flat,
            args.n_warmup,
            args.n_samples,
            args.n_burnin,
            chain_parallel=args.chain_parallel,
        )

        def map_back(positions):
            return positions

    elif args.arm in ("laplace-diagwin", "laplace-densewin"):

        def logdensity_z(z):
            return logdensity_flat(flat0 + laplace_chol @ z)

        starts = args.jitter * zs
        adapt_position0 = starts if args.adapt_per_chain else jnp.zeros(D)
        adapt_fn = _window_adapt_fn(
            logdensity_z,
            adapt_position0,
            args.target_accept,
            args.n_warmup,
            diagonal=args.arm == "laplace-diagwin",
            max_doublings=args.warmup_max_doublings,
            adapt_per_chain=args.adapt_per_chain,
        )
        _, sample_fn, _ = _build_kernel_runner(
            logdensity_z,
            args.n_warmup,
            args.n_samples,
            args.n_burnin,
            chain_parallel=args.chain_parallel,
            adapt_per_chain=args.adapt_per_chain,
        )

        def map_back(positions):
            return flat0[None, None, :] + jnp.einsum("ij,cnj->cni", laplace_chol, positions)

    else:  # diag-window / dense-window: BlackJAX adaptation on the raw problem
        # The library's convention is MAP + 1e-3 jitter, which makes a
        # between-chain R-hat a weak test; ``--window-start laplace`` disperses
        # the chains as Laplace draws instead, like the laplace arms.
        if args.window_start == "laplace":
            starts = flat0[None, :] + args.jitter * (zs @ laplace_chol.T)
        else:
            starts = flat0[None, :] + 1e-3 * zs
        adapt_position0 = starts if args.adapt_per_chain else flat0
        adapt_fn = _window_adapt_fn(
            logdensity_flat,
            adapt_position0,
            args.target_accept,
            args.n_warmup,
            diagonal=args.arm == "diag-window",
            max_doublings=args.warmup_max_doublings,
            adapt_per_chain=args.adapt_per_chain,
        )
        _, sample_fn, _ = _build_kernel_runner(
            logdensity_flat,
            args.n_warmup,
            args.n_samples,
            args.n_burnin,
            chain_parallel=args.chain_parallel,
            adapt_per_chain=args.adapt_per_chain,
        )

        def map_back(positions):
            return positions

    # Per-chain adaptation needs one key per chain (``adapt_fn`` is pmapped over
    # that axis); shared adaptation keeps the single key it always used.
    key_adapt_arg = (
        jax.random.split(key_adapt, args.n_chains) if args.adapt_per_chain else key_adapt
    )

    t0 = time.perf_counter()
    adapt_result_cold = adapt_fn(key_adapt_arg)
    jax.block_until_ready(adapt_result_cold)
    adapt_wall_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    step_size, inverse_mass_matrix, n_grad_adapt = adapt_fn(key_adapt_arg)
    jax.block_until_ready((step_size, inverse_mass_matrix, n_grad_adapt))
    adapt_wall_warm = time.perf_counter() - t0

    sample_keys = jax.random.split(key_sample, args.n_chains)

    t0 = time.perf_counter()
    sample_result_cold = sample_fn(step_size, inverse_mass_matrix, starts, sample_keys)
    jax.block_until_ready(sample_result_cold)
    sample_wall_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    positions_native, divergent, n_steps, accept = sample_fn(
        step_size, inverse_mass_matrix, starts, sample_keys
    )
    jax.block_until_ready((positions_native, divergent, n_steps, accept))
    sample_wall_warm = time.perf_counter() - t0

    positions_x = map_back(positions_native)

    return dict(
        hessian_wall=hessian_wall,
        hess_eig_min=hess_eig_min,
        hess_eig_max=hess_eig_max,
        hess_cond=float(hess_cond),
        hess_eigenvalues=np.asarray(eigenvalues),
        # Per-chain adaptation gives one step size (and mass matrix) per
        # chain; report their mean (a no-op when adaptation is shared) and
        # the SUM of per-chain gradient counts, since every chain paid its
        # own warmup independently and in parallel.
        step_size=float(jnp.mean(jnp.asarray(step_size))),
        n_grad_adapt=int(jnp.sum(n_grad_adapt)),
        adapt_wall_cold=adapt_wall_cold,
        adapt_wall_warm=adapt_wall_warm,
        sample_wall_cold=sample_wall_cold,
        sample_wall_warm=sample_wall_warm,
        positions_x=np.asarray(positions_x),
        divergent=np.asarray(divergent),
        n_steps=np.asarray(n_steps),
        accept=np.asarray(accept),
    )


def _score(args, ctx, unravel, names, truth, sampled, map_nlp, map_wall_cold, map_wall_warm):
    """Diagnostics on the unbounded-space draws: ESS, split R-hat, and physical stats."""
    positions_x = sampled["positions_x"]
    n_chains, n_samples, D = positions_x.shape
    flat_all = jnp.asarray(positions_x.reshape(-1, D))

    unbounded_dict = jax.vmap(unravel)(flat_all)
    physical_dict = jax.vmap(ctx.to_physical)(unbounded_dict)

    unbounded_batched = {
        k: np.asarray(v).reshape(n_chains, n_samples) for k, v in unbounded_dict.items()
    }
    physical_batched = {
        k: np.asarray(v).reshape(n_chains, n_samples)
        for k, v in physical_dict.items()
        if k in names
    }

    ess = {k: float(effective_sample_size(jnp.asarray(v))) for k, v in unbounded_batched.items()}
    rhat = {k: _split_rhat(v) for k, v in unbounded_batched.items()}
    worst_param = min(ess, key=ess.get)
    min_ess = ess[worst_param]
    rhat_max = max(rhat.values())

    ess_phys = {
        k: float(effective_sample_size(jnp.asarray(v))) for k, v in physical_batched.items()
    }
    ess_phys_min = min(ess_phys.values())

    post_mean = {k: float(np.mean(v)) for k, v in physical_batched.items()}
    post_std = {k: float(np.std(v)) for k, v in physical_batched.items()}
    truth_phys = {name: float(truth[name]) for name in names}
    z_truth = {
        name: (post_mean[name] - truth_phys[name]) / post_std[name]
        if post_std[name] > 0
        else float("nan")
        for name in names
    }

    divergent, n_steps, accept = sampled["divergent"], sampled["n_steps"], sampled["accept"]
    n_draws_total = n_chains * n_samples
    n_grad_sample = int(np.sum(n_steps))
    n_grad_adapt = sampled["n_grad_adapt"]
    grad_per_draw = n_grad_sample / n_draws_total
    tree_depth_mean = float(np.mean(np.log2(np.maximum(n_steps, 1)) + 1.0))
    divergences = int(np.sum(divergent))
    converged = bool(rhat_max < 1.01 and min_ess >= 100)

    return dict(
        D=D,
        free_names=names,
        map_nlp=map_nlp,
        map_wall_cold=map_wall_cold,
        map_wall_warm=map_wall_warm,
        hessian_wall=sampled["hessian_wall"],
        hess_eig_min=sampled["hess_eig_min"],
        hess_eig_max=sampled["hess_eig_max"],
        hess_cond=sampled["hess_cond"],
        step_size=sampled["step_size"],
        adapt_wall_cold=sampled["adapt_wall_cold"],
        adapt_wall_warm=sampled["adapt_wall_warm"],
        sample_wall_cold=sampled["sample_wall_cold"],
        sample_wall_warm=sampled["sample_wall_warm"],
        wall_total_warm=(map_wall_warm + sampled["adapt_wall_warm"] + sampled["sample_wall_warm"]),
        wall_total_cold=(map_wall_cold + sampled["adapt_wall_cold"] + sampled["sample_wall_cold"]),
        n_grad_adapt=n_grad_adapt,
        n_grad_sample=n_grad_sample,
        n_grad_total=n_grad_adapt + n_grad_sample,
        grad_per_draw=grad_per_draw,
        tree_depth_mean=tree_depth_mean,
        divergences=divergences,
        div_frac=divergences / n_draws_total,
        accept_mean=float(np.mean(accept)),
        min_ess=min_ess,
        worst_param=worst_param,
        ess=ess,
        ess_phys_min=ess_phys_min,
        rhat_max=rhat_max,
        rhat=rhat,
        unique_frac=_unique_fraction(positions_x),
        converged=converged,
        grads_to_100=n_grad_adapt + (100.0 / min_ess) * n_grad_sample,
        sec_to_100=(
            sampled["adapt_wall_warm"]
            + map_wall_warm
            + (100.0 / min_ess) * sampled["sample_wall_warm"]
        ),
        truth=truth_phys,
        post_mean=post_mean,
        post_std=post_std,
        z_truth=z_truth,
        nonfinite=bool(not np.all(np.isfinite(positions_x))),
        error=None,
    )


def run_fit(args) -> dict:
    """Build the mock, MAP-seed it, run one arm, and score it. Never raises."""
    row = dict.fromkeys(ROW_FIELDS)
    row.update(
        notebook=args.notebook,
        seed=args.seed,
        arm=args.arm,
        dtype="float32" if args.float32 else "float64",
        platform=jax.devices()[0].platform,
        devices=str(jax.devices()),
        taskset=len(os.sched_getaffinity(0)),
        git_sha=_git_sha(),
        timestamp=datetime.now(UTC).isoformat(),
        n_chains=args.n_chains,
        chain_parallel=args.chain_parallel,
        n_devices=len(jax.devices()),
        adapt_per_chain=args.adapt_per_chain,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        map_restarts=args.map_restarts,
        map_steps=args.map_steps,
        map_optimizer=args.map_optimizer,
        warmup_max_doublings=args.warmup_max_doublings,
        window_start=args.window_start,
        jitter=args.jitter,
        profile_mass=args.profile_mass,
    )
    try:
        fit_dtype = jnp.float32 if args.float32 else jnp.float64
        _, forward, ctx, flux, noise, truth, key_fit = _build_problem(
            args.notebook, args.seed, dtype="float32" if args.float32 else "float64"
        )
        names = ctx.free_names
        nlp = ctx.neg_log_posterior_fn
        dargs = ctx.data_args
        key_map, key_arm = jax.random.split(key_fit)
        data = Data(photometry=(flux, noise))

        def do_map():
            t0 = time.perf_counter()
            map_post = forward.fit(
                data,
                method="map",
                key=key_map,
                profile_mass=False,
                n_restarts=args.map_restarts,
                n_steps=args.map_steps,
                optimizer=args.map_optimizer,
                verbose=False,
            )
            jax.block_until_ready(map_post.params)
            return map_post, time.perf_counter() - t0

        _, map_wall_cold = do_map()
        map_post, map_wall_warm = do_map()

        pu = ctx.unbounded_from_posterior(map_post)
        pu = {
            k: jnp.asarray(v, dtype=fit_dtype).reshape(())
            if jnp.ndim(v) <= 1 and jnp.size(v) == 1
            else v
            for k, v in pu.items()
        }
        flat0, unravel = ravel_pytree(pu)
        map_nlp = float(nlp(pu, dargs))

        profiled = None
        mass_idx = non_mass_idx = None
        if args.profile_mass:
            mass_name = _mass_free_name(names)
            mass_idx, non_mass_idx = _mass_index(pu, mass_name)
            phys_map = ctx.to_physical(unravel(flat0))
            _verify_mass_linearity(ctx, mass_name, phys_map)
            _verify_gaussian_likelihood(ctx, nlp, dargs, unravel, flat0, flux, noise)

            profiled = _build_profiled_objects(
                ctx, unravel, mass_name, mass_idx, non_mass_idx, flux, noise
            )
            profiled_logdensity = profiled["profiled_logdensity"]

            # Sanity check at the FULL-D MAP's theta_rest, before refining it:
            # the profiled gradient should be small and -logdensity_rest should
            # be in the same ballpark as the full-D nlp with the mass prior term
            # removed (they are not identical -- one is a point evaluation, the
            # other integrates the mass out, adding its own evidence term).
            flat_rest_from_full = flat0[non_mass_idx]
            grad_at_full_map = jax.grad(profiled_logdensity)(flat_rest_from_full)
            neg_logdensity_at_full_map = -float(profiled_logdensity(flat_rest_from_full))
            mass_prior_term = -0.5 * float(flat0[mass_idx]) ** 2
            print(
                "[profile-mass] grad(logdensity_rest) at full-D MAP theta_rest: "
                f"norm={float(jnp.linalg.norm(grad_at_full_map)):.3e} "
                f"max_abs={float(jnp.max(jnp.abs(grad_at_full_map))):.3e}"
            )
            print(
                f"[profile-mass] -logdensity_rest(theta_rest@full-D MAP)="
                f"{neg_logdensity_at_full_map:.6f} vs full-D nlp - mass_prior_term="
                f"{map_nlp - mass_prior_term:.6f}"
            )

            def obj_rest(x_rest):
                return -profiled_logdensity(x_rest)

            def do_map_rest():
                t0 = time.perf_counter()
                res = jax.scipy.optimize.minimize(obj_rest, flat_rest_from_full, method="BFGS")
                jax.block_until_ready(res.x)
                return res, time.perf_counter() - t0

            _, map_rest_wall_cold = do_map_rest()
            res_rest, map_rest_wall_warm = do_map_rest()
            map_wall_cold += map_rest_wall_cold
            map_wall_warm += map_rest_wall_warm
            flat0 = res_rest.x
            map_nlp = float(res_rest.fun)
            print(
                f"[profile-mass] {flat0.size}-D BFGS MAP: nlp={map_nlp:.6f} "
                f"success={bool(res_rest.success)} nit={int(res_rest.nit)} "
                f"move_from_full_D_theta_rest="
                f"{float(jnp.linalg.norm(flat0 - flat_rest_from_full)):.3e}"
            )

            def logdensity_2arg(x, _data_args):
                return profiled_logdensity(x)

            def logdensity_flat(x):
                return profiled_logdensity(x)
        else:

            def logdensity_2arg(x, data_args):
                return -nlp(unravel(x), data_args)

            def logdensity_flat(x):
                return -nlp(unravel(x), dargs)

        # Belt-and-braces: the Hessian/Laplace metric and the mass-profile
        # quadrature must run in the active dtype. ``flat0`` should already be
        # ``fit_dtype`` (jax_enable_x64 is off for the whole float32 session),
        # but an explicit cast here is cheap and is the documented fallback if
        # any upstream step (e.g. ``jax.scipy.optimize.minimize``) ever hands
        # back a wider dtype than the active config allows.
        flat0 = jnp.asarray(flat0, dtype=fit_dtype)
        d_sampled = int(flat0.size)
        sampled = _run_arm(args, logdensity_2arg, logdensity_flat, flat0, dargs, key_arm)

        if args.profile_mass:
            print(
                f"[profile-mass] {d_sampled}-D Hessian eigenvalues: "
                f"{np.array2string(sampled['hess_eigenvalues'], precision=4)}"
            )
            print(
                f"[profile-mass] {d_sampled}-D Hessian cond={sampled['hess_cond']:.6g} "
                f"(full-D cond ~= 3.7e4)"
            )
            key_mass = jax.random.fold_in(key_arm, 0xA55)
            sampled["positions_x"] = _draw_mass_positions(
                key_mass, sampled["positions_x"], profiled, mass_idx, non_mass_idx
            )

        scored = _score(
            args, ctx, unravel, names, truth, sampled, map_nlp, map_wall_cold, map_wall_warm
        )
        row.update(scored)
        row["D_sampled"] = d_sampled
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    # Peak resident set of this process (kB on Linux), because the D >= 8 dense
    # auto-policy in ``run_nuts`` exists for a memory spike and a dense row
    # without its memory column cannot argue with that policy.
    import resource

    row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return row


def _format_summary(row: dict) -> str:
    if row.get("error"):
        return f"{row['notebook']} seed={row['seed']} {row['arm']} FAILED: {row['error']}"
    return (
        f"{row['notebook']} seed={row['seed']} {row['arm']} "
        f"{row['n_chains']}x{row['n_samples']} "
        f"wall_warm={row['wall_total_warm']:.2f}s "
        f"(map {row['map_wall_warm']:.2f}+adapt {row['adapt_wall_warm']:.2f}"
        f"+sample {row['sample_wall_warm']:.2f}) "
        f"grads={row['n_grad_total']} g/draw={row['grad_per_draw']:.1f} "
        f"minESS={row['min_ess']:.1f} ({row['worst_param']}) "
        f"rhat={row['rhat_max']:.4f} div={row['divergences']} "
        f"uniq={row['unique_frac']:.3f} converged={row['converged']}"
    )


def _forward_argv_without(argv: list[str], flags: set[str]) -> list[str]:
    """Strip ``--flag value`` and ``--flag=value`` pairs for the given flag names."""
    out = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        bare = tok.split("=", 1)[0]
        if bare in flags:
            if "=" not in tok:
                skip_next = True
            continue
        out.append(tok)
    return out


def _run_seed_sweep(args) -> None:
    """One fit per subprocess per seed -- adaptation caches and XLA state share nothing."""
    cfg = B.NOTEBOOKS[args.notebook]
    base_seed = cfg["seed"] if args.seed is None else args.seed
    forwarded = _forward_argv_without(sys.argv[1:], {"--seeds", "--seed"})
    for i in range(args.seeds):
        seed = base_seed + i
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            *forwarded,
            "--seed",
            str(seed),
        ]
        proc = subprocess.run(cmd, env=os.environ.copy(), check=False)
        if proc.returncode != 0:
            print(f"[seeds] seed {seed} exited with code {proc.returncode}", file=sys.stderr)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--notebook", default="ctl-dpl")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="run seeds cfg['seed']..cfg['seed']+N-1, one fit per subprocess",
    )
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n-chains", type=int, default=8)
    ap.add_argument("--n-warmup", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--n-burnin", type=int, default=0)
    ap.add_argument("--map-restarts", type=int, default=1)
    ap.add_argument("--map-steps", type=int, default=800)
    # L-BFGS-B by default: measured on ctl-dpl seed 7, one start reaches nlp
    # 6.0008 in 0.42 s warm where Adam at 8x800 stops at 6.33 after 2.9 s, and
    # the Hessian is only a metric at a converged optimum.
    ap.add_argument("--map-optimizer", default="lbfgs", choices=("lbfgs", "adam"))
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--warmup-max-doublings", type=int, default=10)
    ap.add_argument("--window-start", default="map", choices=("map", "laplace"))
    ap.add_argument(
        "--profile-mass",
        action="store_true",
        help=(
            "Analytically marginalize the linear total-mass amplitude (photometry "
            "is linear in mass) and run every downstream step -- MAP, Hessian, "
            "adaptation, sampling -- on the remaining D-1 parameters. Mass draws "
            "are reattached post-hoc from their closed-form conditional posterior "
            "so _score still reports all D names."
        ),
    )
    ap.add_argument(
        "--float32",
        action="store_true",
        help=(
            "Run the whole fit (SSP load, mock, MAP, Hessian/Laplace metric, "
            "adaptation, sampling, mass quadrature) in float32. Sets "
            "JAX_ENABLE_X64=0 (and JAX_DEFAULT_MATMUL_PRECISION=highest) in the "
            "environment before jax/tengri are imported -- see "
            "_apply_precision_env, the mechanism "
            "bench/reports/2026-08-31_float32_fitting_path.md validated; a "
            "post-import jax.config.update is NOT equivalent (#1840, #1880) and "
            "was measured here to leave stale float64 DSPS constants live "
            "alongside float32 parameters. flux/noise are also cast to float32 "
            "explicitly. Bit-parity with a float64 run is not expected; the "
            "row's z_truth column shows any float32-induced bias. Default is "
            "float64, tengri's own import-time setting."
        ),
    )
    ap.add_argument(
        "--chain-parallel",
        default="vmap",
        choices=("vmap", "pmap"),
        help=(
            "'vmap' (default) batches all chains on one CPU device, as before. "
            "'pmap' instead runs each chain on its own device -- the caller "
            "needs >= n_chains JAX devices, which this script arranges by "
            "setting XLA_FLAGS from a pre-parse of sys.argv before jax is "
            "imported (see _maybe_set_pmap_device_count)."
        ),
    )
    ap.add_argument(
        "--adapt-per-chain",
        action="store_true",
        help=(
            "Run BlackJAX's window adaptation independently, in parallel (via "
            "jax.pmap, regardless of --chain-parallel), on every chain -- each "
            "chain gets its own step size and its own mass matrix from its own "
            "warmup, then samples with that adaptation. This is the standard "
            "multi-chain protocol (independent per-chain adaptation, diagnosed "
            "afterwards with split R-hat across chains); the default instead "
            "shares one chain's adaptation across every chain. Not implemented "
            "for the 'laplace' arm, whose metric is fixed, not adapted. Needs "
            ">= n_chains JAX devices, same as --chain-parallel pmap."
        ),
    )
    ap.add_argument("--json", default="bench/results/2026-09-11_laplace_nuts_20s.jsonl")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.seeds is not None:
        _run_seed_sweep(args)
        return

    if args.chain_parallel == "pmap" or args.adapt_per_chain:
        n_devices = jax.local_device_count()
        if n_devices != args.n_chains:
            raise SystemExit(
                f"--chain-parallel pmap / --adapt-per-chain need exactly "
                f"--n-chains ({args.n_chains}) JAX devices, one per chain; "
                f"found {n_devices} ({jax.devices()}). This script tries to "
                "arrange that automatically via XLA_FLAGS (see "
                "_maybe_set_pmap_device_count) -- this mismatch means that "
                "pre-parse did not see the --n-chains/--chain-parallel/"
                "--adapt-per-chain flags actually in effect."
            )

    cfg = B.NOTEBOOKS[args.notebook]
    if args.seed is None:
        args.seed = cfg["seed"]

    row = run_fit(args)

    json_path = pathlib.Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "a") as fh:
        fh.write(json.dumps(_to_native(row)) + "\n")

    if not args.quiet:
        print(_format_summary(row))


if __name__ == "__main__":
    main()
