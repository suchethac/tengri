#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Where tempered SMC sits on the compile axis, against NUTS, HMC and ChEES.

Compile is not a footnote on this path. ``bench/reports/2026-08-30_gpu_catalog_throughput.md``
measured **75 %** of a NUTS fit as XLA compile (189.4 s cold against 46.8 s warm),
and ``bench/reports/2026-08-30_mclmc_tuning.md`` measured MCLMC's fixed-length
scan compiling **14x** cheaper than NUTS's ragged tree-doubling ``while_loop``
(10.4 s against 142.6 s). A sampler's control-flow shape is therefore a
first-order cost, not a detail of its implementation.

Adaptive tempered SMC has control flow of its own and it is worth being precise
about where: **a rung is lock-step and the ladder is not.** Every particle takes
the same fixed number of inner-HMC moves of the same fixed length, so the inner
program is a ``vmap`` over a ``scan`` with nothing ragged in it -- but the
*number of rungs* is chosen by the tempering solver from the particle weights, so
the outer loop is a ``lax.while_loop``. This script measures both arms, because
"SMC has no ragged control flow" is half true and the false half is the half
that shows up in a compile column:

* ``--fixed-ladder K`` replaces the ``while_loop`` with a ``lax.scan`` of exactly
  K rungs. Fixed-length, fully lock-step, and a different sampler.
* the default adaptive arm keeps the ``while_loop``.

Each cell runs ``jax.jit(...).lower(...).compile()`` so trace/lower, XLA compile
and the warm run are three separate numbers rather than one wall clock, and each
reports its StableHLO line count -- the measure that says whether two programs
are *the same program* rather than merely similar.

Usage::

    JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 .venv/bin/python \\
        bench/scripts/benchmark_smc_compile.py --notebook ctl-dpl \\
        --json bench/results/2026-08-31_smc_compile.json

``TENGRI_DISABLE_JAX_CACHE=1`` is not optional for a compile claim: with the
persistent cache on, the "compile" column is a cache load and can come out
negative, which is the defect
``bench/reports/2026-08-31_catalog_preconditioning.md`` Caveat 6 records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark_notebook_sampler import NOTEBOOKS

import tengri
from tengri import Data, ForwardModel, generate_mock


def _build(notebook: str, seed: int):
    """Build the harness fixture and MAP-seed it, exactly as a gate row would."""
    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    map_seed = forward.fit(data, method="map", key=k_fit, n_restarts=8, n_steps=800, verbose=False)
    return cfg, forward, data, map_seed, k_fit


def _smc_cell(
    notebook, seed, *, n_particles, n_chains, n_mcmc_steps, n_leapfrog, fixed_ladder, precondition
):
    """Lower and compile one SMC program, without running the whole fit twice."""
    from tengri.inference.backends.mcmc._shared import (
        _SMC_MAX_TEMPERATURES,
        _get_flat_logdensity,
        _get_flat_prior_and_likelihood,
        _smc_scan,
    )
    from tengri.inference.backends.mcmc.smc import SMC_TARGET_ACCEPT_RATE
    from tengri.inference.fitter import Fitter
    from tengri.inference.preconditioning import prepare_preconditioning

    _cfg, forward, data, map_seed, key = _build(notebook, seed)
    fitter = Fitter(forward, data=data.photometry[0], noise=data.photometry[1], approx="auto")
    init_params = fitter._unbounded_from_posterior(map_seed)
    logpost, _unravel, init_flat, data_args = _get_flat_logdensity(fitter, init_params)
    logprior, loglik = _get_flat_prior_and_likelihood(fitter, init_params)
    problem = prepare_preconditioning(logpost, init_flat, data_args, precondition=precondition)
    if problem.enabled:
        matrix = problem.preconditioner.matrix

        def lp(pos, da):
            """Prior in the whitened basis."""
            return logprior(matrix @ pos, da)

        def ll(pos, da):
            """Likelihood in the whitened basis."""
            return loglik(matrix @ pos, da)

        draw_matrix = problem.preconditioner.inverse
    else:
        lp, ll = logprior, loglik
        draw_matrix = jnp.eye(init_flat.shape[0], dtype=init_flat.dtype)

    run_keys = jax.random.split(key, n_chains)
    static = (
        lp,
        ll,
        n_particles,
        n_mcmc_steps,
        n_leapfrog,
        0.5,
        0.1,
        0.5,
        SMC_TARGET_ACCEPT_RATE,
        _SMC_MAX_TEMPERATURES,
        fixed_ladder,
    )
    fn = _smc_scan.__wrapped__
    jitted = jax.jit(fn, static_argnums=tuple(range(3, 14)))
    args = (draw_matrix, run_keys, data_args, *static)

    t0 = time.perf_counter()
    lowered = jitted.lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    out = compiled(draw_matrix, run_keys, data_args)
    jax.block_until_ready(out[0])
    t3 = time.perf_counter()
    out = compiled(draw_matrix, run_keys, data_args)
    jax.block_until_ready(out[0])
    t4 = time.perf_counter()

    hlo_lines = lowered.as_text().count("\n")
    return {
        "method": "mcmc_smc",
        "notebook": notebook,
        "seed": seed,
        "d_params": int(init_flat.shape[0]),
        "n_particles": n_particles,
        "n_chains": n_chains,
        "n_mcmc_steps": n_mcmc_steps,
        "n_leapfrog_steps": n_leapfrog,
        "fixed_ladder": fixed_ladder,
        "precondition": precondition,
        "schedule": "fixed" if fixed_ladder is not None else "adaptive",
        "trace_lower_s": round(t1 - t0, 3),
        "compile_s": round(t2 - t1, 3),
        "first_run_s": round(t3 - t2, 3),
        "warm_run_s": round(t4 - t3, 3),
        "stablehlo_lines": hlo_lines,
        "n_temperatures": [int(v) for v in out[2]],
    }


def _width_sweep(notebook, seed, widths, *, n_mcmc_steps, n_leapfrog, fixed_ladder, precondition):
    """Gradients per second against particle count -- SMC's one speed argument.

    The particle axis is a pure ``vmap`` with nothing ragged in it, so it is the
    natural batch axis an accelerator wants. ``bench/reports/2026-08-20_cuda_device_matrix.md``
    measured tengri's forward model crossing over from CPU to GPU only between
    128 and 512 *galaxies*; if particles behave the same way, a single-galaxy SMC
    fit reaches accelerator width without a catalog. That is a claim about
    throughput, so it is measured as gradients per second at fixed work per
    particle, never as a wall clock (which trivially grows with the width).

    A fixed ladder is used so every width does the SAME number of rungs -- under
    the adaptive schedule the rung count moves with the population size and the
    comparison would silently change the numerator.
    """
    rows = []
    for width in widths:
        row = _smc_cell(
            notebook,
            seed,
            n_particles=width,
            n_chains=1,
            n_mcmc_steps=n_mcmc_steps,
            n_leapfrog=n_leapfrog,
            fixed_ladder=fixed_ladder,
            precondition=precondition,
        )
        rungs = row["n_temperatures"][0]
        grads = width * rungs * n_mcmc_steps * n_leapfrog
        row["gradients"] = grads
        row["gradients_per_s"] = round(grads / row["warm_run_s"], 1)
        rows.append(row)
        print(
            f"width {width:>5}  warm {row['warm_run_s']:>7.2f}s  "
            f"{row['gradients_per_s']:>10.0f} grad/s  hlo {row['stablehlo_lines']:>6} lines",
            flush=True,
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default="ctl-dpl", choices=sorted(NOTEBOOKS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--mcmc-steps", type=int, default=2)
    ap.add_argument("--leapfrog", type=int, default=10)
    ap.add_argument(
        "--fixed-ladder",
        type=int,
        nargs="*",
        default=[16],
        help="fixed-ladder rung counts to measure beside the adaptive arm",
    )
    ap.add_argument(
        "--widths",
        type=int,
        nargs="*",
        default=None,
        help=(
            "particle counts for a throughput sweep instead of the schedule "
            "comparison; run with a FIXED ladder so every width does equal work"
        ),
    )
    ap.add_argument(
        "--precondition",
        type=float,
        nargs="*",
        default=None,
        help=(
            "whitening strengths to measure; omit for the default pair "
            "(off, then 0.5). A compile claim needs only one arm, and the "
            "unpreconditioned one is the slow one because it takes more rungs."
        ),
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    precond_arms = [None, True] if args.precondition is None else list(args.precondition)

    if not os.environ.get("TENGRI_DISABLE_JAX_CACHE"):
        print(
            "WARNING: TENGRI_DISABLE_JAX_CACHE is not set, so the compile column "
            "is a persistent-cache load and not a compile.",
            file=sys.stderr,
        )

    if args.widths:
        rows = _width_sweep(
            args.notebook,
            args.seed,
            args.widths,
            n_mcmc_steps=args.mcmc_steps,
            n_leapfrog=args.leapfrog,
            fixed_ladder=(args.fixed_ladder or [16])[0],
            precondition=precond_arms[-1],
        )
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(rows, indent=2))
            print(f"wrote {args.json}")
        return 0

    rows = []
    for precondition in precond_arms:
        for ladder in [None, *args.fixed_ladder]:
            row = _smc_cell(
                args.notebook,
                args.seed,
                n_particles=args.particles,
                n_chains=args.chains,
                n_mcmc_steps=args.mcmc_steps,
                n_leapfrog=args.leapfrog,
                fixed_ladder=ladder,
                precondition=precondition,
            )
            rows.append(row)
            print(
                f"{row['schedule']:<9} precond={precondition!s:<5} "
                f"lower {row['trace_lower_s']:>6.2f}s  compile {row['compile_s']:>7.2f}s  "
                f"warm {row['warm_run_s']:>7.2f}s  hlo {row['stablehlo_lines']:>6} lines  "
                f"rungs {row['n_temperatures']}",
                flush=True,
            )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
