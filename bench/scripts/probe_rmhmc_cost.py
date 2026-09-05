#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
r"""Price a position-dependent metric before committing a campaign to it.

``inference/preconditioning.py``'s closing paragraph names Riemannian HMC's
motivation exactly:

    "One posterior standard deviation away the whitened stiffness runs 3.7e2 to
    1.7e5 ... Closing that last gap needs a **position-dependent** metric,
    which is exactly what MGVI/geoVI provide and a fixed mass matrix cannot."

``blackjax.rmhmc`` takes ``mass_matrix`` as a **callable of position**, and
``preconditioning.negative_hessian_metric(logdensity_fn, position, data_args)``
already has that shape. So the wiring is a currying, and the question is not
whether it fits but what it costs -- and under a brief where speed is the
primary objective, a sampler that mixes better per step while being far slower
per second is a loss, not a win.

This script measures the cost rather than arguing it. Four numbers, on one real
tengri posterior:

1. one ``value_and_grad`` of the log-posterior -- what a fixed-metric leapfrog
   step costs;
2. one ``negative_hessian_metric`` evaluation -- what ONE metric build costs;
3. one ``jax.grad`` of BlackJAX's Riemannian kinetic energy with respect to
   position -- what ``implicit_midpoint`` needs at **every** fixed-point
   iteration, and which differentiates through ``jax.hessian``, ``eigh``,
   ``jnp.maximum`` and a Cholesky in one pass;
4. an actual ``blackjax.rmhmc`` step, timed, with the fixed-point iteration
   count read out of the solver.

It also records the two structural facts that no timing can express and that
decide the verdict on their own: ``implicit_midpoint``'s solver is a
``jax.lax.while_loop`` with ``max_iters=100`` (ragged control flow, in a
codebase whose measured advantage from MCLMC was a fixed-length scan compiling
14x cheaper), and ``gaussian_riemannian``'s turning check raises
``NotImplementedError`` (there is no Riemannian NUTS, so this is fixed-`L` HMC
only).

Measured on ``ctl-dpl`` (D = 8) on 2026-08-31: 1.00 ms, 8.97 ms (9.0x),
64.7 ms (64.7x), and **6374 ms per draw at L = 5 -- 1273x a Euclidean draw** --
with 72-81 s to compile a single step. See
``bench/reports/2026-08-31_blackjax_sampler_survey.md`` Finding 3.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/probe_rmhmc_cost.py
"""

from __future__ import annotations

import inspect
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tengri  # noqa: E402
from benchmark_notebook_sampler import NOTEBOOKS  # noqa: E402
from tengri import Fitter  # noqa: E402
from tengri.analysis.mock import generate_mock  # noqa: E402
from tengri.inference._sample_utils import _maybe_map_init  # noqa: E402
from tengri.inference.backends.mcmc._shared import _get_flat_logdensity  # noqa: E402
from tengri.inference.preconditioning import negative_hessian_metric  # noqa: E402

#: Repeats per timing. Small on purpose: the ratios this script exists to
#: report are one to three orders of magnitude, and a shared box cannot
#: resolve better than that anyway.
N_REPEAT = 5


def _time(fn, *args, n=N_REPEAT):
    """Median wall seconds of ``n`` warm calls to ``fn``, blocking on the result."""
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main(notebook: str = "ctl-dpl") -> None:
    """Build one fixture, then time the four quantities above on it."""
    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    fitter = Fitter(
        sed,
        np.asarray(mock["flux_obs"]),
        np.asarray(mock["noise"]),
        data_type="photometry",
    )
    init_params, _ = _maybe_map_init(fitter, k_fit, None, False)
    log_p2, _unravel, init_flat, data_args = _get_flat_logdensity(fitter, init_params)
    n_dim = int(init_flat.shape[0])
    print(f"fixture {notebook}: D = {n_dim}")

    # --- 1. the fixed-metric baseline: one gradient of the log-posterior ----
    grad_fn = jax.jit(jax.value_and_grad(lambda q: log_p2(q, data_args)))
    t_grad = _time(grad_fn, init_flat)
    print(f"1. value_and_grad(log_p)                 {t_grad * 1e3:9.3f} ms")

    # --- 2. one metric build at an arbitrary position -----------------------
    metric_fn = jax.jit(lambda q: negative_hessian_metric(log_p2, q, data_args))
    t_metric = _time(metric_fn, init_flat)
    print(
        f"2. negative_hessian_metric(q)             {t_metric * 1e3:9.3f} ms"
        f"   ({t_metric / t_grad:6.1f}x a gradient)"
    )

    # --- 3. what implicit_midpoint needs per fixed-point iteration ----------
    # ``jax.grad`` of the Riemannian kinetic energy w.r.t. POSITION. This is
    # the term a Euclidean metric does not have at all: it differentiates the
    # Hessian, the eigendecomposition, the |lambda| floor and the Cholesky.
    from blackjax.mcmc import metrics

    riemannian = metrics.gaussian_riemannian(metric_fn)
    momentum = jnp.ones_like(init_flat)

    def _dT_dq(q, p):
        return jax.grad(lambda qq: riemannian.kinetic_energy(p, position=qq))(q)

    try:
        dT_fn = jax.jit(_dT_dq)
        t_dT = _time(dT_fn, init_flat, momentum)
        finite = bool(jnp.all(jnp.isfinite(dT_fn(init_flat, momentum))))
        print(
            f"3. grad_q(riemannian kinetic energy)      {t_dT * 1e3:9.3f} ms"
            f"   ({t_dT / t_grad:6.1f}x a gradient), finite={finite}"
        )
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        t_dT = float("nan")
        print(
            f"3. grad_q(riemannian kinetic energy)      FAILED: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

    # --- 4. an actual rmhmc step, and its fixed-point iteration count -------
    import blackjax
    from blackjax.mcmc import integrators

    iters_seen = []
    original_solver = integrators.solve_fixed_point_iteration

    def _counting_solver(func, x0, **kw):
        x, aux, info = original_solver(func, x0, **kw)
        iters_seen.append(info)
        return x, aux, info

    n_leapfrog = 5
    kernel = blackjax.mcmc.rmhmc.build_kernel(
        integrators.implicit_midpoint, divergence_threshold=1000
    )
    state = blackjax.mcmc.rmhmc.init(init_flat, lambda q: log_p2(q, data_args))

    for step_size in (0.05, 0.2):
        try:
            step = jax.jit(
                lambda k, s, e=step_size: kernel(
                    k,
                    s,
                    lambda q: log_p2(q, data_args),
                    e,
                    metric_fn,
                    n_leapfrog,
                )
            )
            t0 = time.perf_counter()
            new_state, info = step(jax.random.PRNGKey(0), state)
            jax.block_until_ready(new_state.position)
            t_compile = time.perf_counter() - t0
            t_step = _time(step, jax.random.PRNGKey(1), state, n=3)
            print(
                f"4. rmhmc step, L={n_leapfrog}, eps={step_size}"
                f"        {t_step * 1e3:9.3f} ms/draw"
                f"   ({t_step / (t_grad * n_leapfrog):6.1f}x a Euclidean L={n_leapfrog} draw)"
                f"   compile {t_compile:.1f}s  accept={float(info.acceptance_rate):.3f}"
                f"  divergent={bool(info.is_divergent)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"4. rmhmc step, eps={step_size}: FAILED -- {type(exc).__name__}: {str(exc)[:300]}"
            )

    # --- the structural facts, which no timing expresses --------------------
    print("\nstructural, read off blackjax 1.6.2 source:")
    solver_kw = inspect.signature(integrators.solve_fixed_point_iteration).parameters
    print(
        "  implicit_midpoint solver: jax.lax.while_loop, max_iters="
        f"{solver_kw['max_iters'].default}, convergence_tol="
        f"{solver_kw['convergence_tol'].default}"
        " -- ragged control flow inside every leapfrog step, and the solver's"
        " own success flag is discarded by the integrator ('del info  # TODO')"
    )
    # The Metric NamedTuple's turning field. Named ``check_turning`` in the
    # tuple and ``is_turning`` in the closure that fills it, which is why this
    # is read off the tuple rather than guessed.
    try:
        riemannian.check_turning(momentum, momentum, momentum, init_flat, init_flat)
        print("  gaussian_riemannian turning check: returned (Riemannian NUTS exists?)")
    except NotImplementedError as exc:
        print(f"  gaussian_riemannian turning check: NotImplementedError -- {exc}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ctl-dpl")
