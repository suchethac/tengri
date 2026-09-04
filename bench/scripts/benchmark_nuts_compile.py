#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Where the XLA seconds of a single-galaxy NUTS fit actually go.

``bench/reports/2026-08-30_mclmc_tuning.md`` measured one NUTS fit at **189.4 s
cold against 46.8 s warm** -- 75.3 %, i.e. **142.6 s of XLA** -- beside MCLMC's
10.4 s on the same model and seed, and attributed the 14x to NUTS compiling "a
ragged tree-doubling ``while`` loop with ``max_num_doublings=10``" where MCLMC
compiles "a fixed-length scan of one step".

That is a hypothesis about *which subgraph* the seconds are in, and a
whole-fit cold/warm difference cannot test it: a fit compiles the forward
model, its gradient, the window adaptation and the sampling scan, and the
difference lumps all four together. This script separates them.

Each stage is put through ``jax.jit(...).lower(...).compile()`` so three
numbers come out separately:

* **trace+lower** -- building the jaxpr and the StableHLO module, no XLA work;
* **compile** -- XLA's own optimization of that module;
* **StableHLO lines** -- the size XLA is being asked to optimize.

The stages, smallest first, so a difference between two adjacent rows names the
subgraph that carries it:

``logdensity``
    One evaluation of ``log_posterior_flat_2arg``. The forward model and the
    likelihood, and nothing of any sampler.
``grad``
    ``jax.value_and_grad`` of the same. Every sampler here pays this per
    leapfrog step; the brief's ~5-10 ms in-scan gradient is this program.
``nuts_warmup``
    ``_nuts_warmup_only`` -- ``blackjax.window_adaptation`` and nothing else.
``nuts_scan``
    ``_nuts_chain_scan`` -- the sampling half against fixed adaptation.
``hmc_warmup`` / ``hmc_scan``
    The same two halves for fixed-length HMC, as the control: HMC has the same
    forward model and the same gradient and *no* ragged tree, so
    ``nuts_* - hmc_*`` is what the tree machinery costs.

Sweeping ``--max-doublings`` against these rows is the direct test of the
"ragged ``while`` loop" story. A ``lax.while_loop`` is not unrolled, so if the
cap does not move the line count the graph size is not the tree's doing.

A second mode measures the other half of the compile question. tengri enables a
persistent on-disk XLA cache at import (``~/.cache/tengri_jax_cache``, capped at
8 GiB since #1507), and JAX itself counts every lookup:
``/jax/compilation_cache/cache_hits``, ``/jax/compilation_cache/cache_misses``
and ``/jax/compilation_cache/compile_time_saved_sec``. ``--cache-probe`` runs a
sequence of fits with those counters registered and reports the hit rate and
the seconds the cache actually saved, so "the cache is at its cap and evicting"
becomes a measurement rather than an inference from ``du``.

Examples
--------
The anatomy, on the nb05 fixture, at three tree caps::

    JAX_PLATFORMS=cpu python bench/scripts/benchmark_nuts_compile.py \\
        --notebook 05 --max-doublings 2 6 10

The cache, over a sequence of fits on distinct galaxies::

    python bench/scripts/benchmark_nuts_compile.py --notebook 05 --cache-probe 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

#: Stages measured by :func:`measure_stage`, in the order they are reported.
#: Smallest graph first, so the difference between two adjacent rows names the
#: subgraph that carries it.
STAGES = (
    "logdensity",
    "grad",
    "nuts_warmup",
    "nuts_scan",
    "hmc_warmup",
    "hmc_scan",
)


def _machine_load() -> dict:
    """Load average and free memory at the moment of measurement.

    Every wall clock in this file is contended on a shared box, and this
    project has already measured what that does: the same NUTS cell read
    **2450.7 s with five sibling fits on 24 cores and 257.5 s running clean**,
    a 9.5x spread from scheduling alone with R-hat, ESS and divergences
    unaffected. A wall clock recorded without its load average gets re-quoted
    later as if it were clean -- which is how a 142.6 s figure became a
    headline. Structural quantities (StableHLO line counts, R-hat, ESS,
    divergence counts) carry no such caveat and are reported as-is.
    """
    try:
        with open("/proc/loadavg") as fh:
            load1, load5, load15 = fh.read().split()[:3]
    except OSError:
        return {}
    return {
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
        "n_cpu": os.cpu_count(),
    }


def _build_problem(notebook: str, seed: int):
    """Build a notebook fixture's model, mock and MAP-seeded flat log posterior.

    Reuses :data:`benchmark_notebook_sampler.NOTEBOOKS` verbatim rather than
    defining a second fixture, for the reason ``check_harness_parity.py``
    exists: a compile number measured on a model no published fit uses explains
    nothing about the published fit.
    """
    import jax
    import numpy as np
    from benchmark_notebook_sampler import NOTEBOOKS

    import tengri
    from tengri import Data, ForwardModel, generate_mock

    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    flux = np.asarray(mock["flux_obs"])
    noise = np.asarray(mock["noise"])
    data = Data(photometry=(flux, noise))
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    map_seed = forward.fit(
        data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
    )
    return forward, (flux, noise), map_seed, key_fit, cfg


def _flat_pieces(forward, flux_noise, map_seed, key_fit, precondition=None):
    """``(logdensity, init_flat, data_args, n_dim)`` for the fitter's own problem."""
    import jax

    from tengri.inference._sample_utils import _maybe_map_init
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
    from tengri.inference.context import InferenceContext
    from tengri.inference.preconditioning import prepare_preconditioning

    from tengri.inference.fitter import Fitter

    flux, noise = flux_noise
    context = InferenceContext.from_target(Fitter(forward, data=flux, noise=noise, approx="auto"))
    fitter = context.fitter
    init_params, _key = _maybe_map_init(fitter, key_fit, map_seed, False)
    logdensity, _unravel, init_flat, data_args = _get_flat_logdensity(fitter, init_params)
    problem = prepare_preconditioning(logdensity, init_flat, data_args, precondition=precondition)
    return (
        problem.logdensity,
        problem.init_flat,
        data_args,
        int(len(problem.init_flat)),
        jax.random.PRNGKey(0),
    )


def measure_stage(
    stage: str,
    notebook: str,
    *,
    seed: int,
    n_warmup: int,
    n_samples: int,
    max_doublings: int,
    n_leapfrog: int,
    use_dense: bool,
    target_accept_rate: float,
    precondition: float | None,
    skip_run: bool = False,
) -> dict:
    """Trace/lower, compile and time one stage of a single-galaxy NUTS fit."""
    import blackjax
    import jax

    from tengri.inference.backends.mcmc._shared import (
        _hmc_chain_scan,
        _hmc_warmup_only,
        _nuts_chain_scan,
        _nuts_warmup_only,
    )

    forward, data, map_seed, key_fit, cfg = _build_problem(notebook, seed)
    ld, init_flat, data_args, n_dim, key = _flat_pieces(
        forward, data, map_seed, key_fit, precondition
    )

    def ld_1arg(pos):
        return ld(pos, data_args)

    n_iter = n_samples
    chain_keys = jax.random.split(key, n_iter)

    # ``args`` is what ``.lower()`` takes -- statics included, in signature
    # order. ``traced`` is the subset a compiled artifact is called with: the
    # static ones are baked into it and passing them again is a TypeError.
    if stage == "logdensity":
        fn = jax.jit(ld)
        args = (init_flat, data_args)
        traced = args
    elif stage == "grad":
        fn = jax.jit(jax.value_and_grad(ld))
        args = (init_flat, data_args)
        traced = args
    elif stage == "nuts_warmup":
        fn = _nuts_warmup_only
        args = (
            init_flat,
            key,
            ld,
            data_args,
            n_warmup,
            use_dense,
            target_accept_rate,
            False,
            int(max_doublings),
        )
        traced = (init_flat, key, data_args)
    elif stage == "nuts_scan":
        state = blackjax.mcmc.nuts.init(init_flat, ld_1arg)
        inv_mass = jax.numpy.ones(n_dim)
        fn = _nuts_chain_scan
        args = (state, chain_keys, ld, data_args, 0.1, inv_mass, int(max_doublings))
        traced = (state, chain_keys, data_args, 0.1, inv_mass)
    elif stage == "hmc_warmup":
        fn = _hmc_warmup_only
        args = (init_flat, key, ld, data_args, n_warmup, n_leapfrog, use_dense, target_accept_rate)
        traced = (init_flat, key, data_args)
    elif stage == "hmc_scan":
        state = blackjax.mcmc.hmc.init(init_flat, ld_1arg)
        inv_mass = jax.numpy.ones(n_dim)
        fn = _hmc_chain_scan
        args = (state, chain_keys, ld, data_args, 0.1, inv_mass, int(n_leapfrog))
        traced = (state, chain_keys, data_args, 0.1, inv_mass)
    else:  # pragma: no cover - argparse constrains the choices
        raise ValueError(f"unknown stage {stage!r}")

    t0 = time.perf_counter()
    lowered = fn.lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    run_s = None
    if not skip_run:
        out = compiled(*traced)
        jax.block_until_ready(out)
        run_s = round(time.perf_counter() - t2, 3)

    try:
        hlo_lines = lowered.as_text().count("\n")
    except Exception:
        hlo_lines = None

    return {
        "stage": stage,
        "notebook": notebook,
        "seed": seed,
        "n_dim": n_dim,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "max_doublings": int(max_doublings),
        "n_leapfrog": int(n_leapfrog),
        "precondition": precondition,
        "device": str(jax.devices()[0].platform),
        **_machine_load(),
        "trace_lower_s": round(t1 - t0, 3),
        "compile_s": round(t2 - t1, 3),
        "run_s": run_s,
        "stablehlo_lines": hlo_lines,
    }


def cache_probe(notebook: str, *, seed: int, n_fits: int, kwargs: dict) -> dict:
    """Hit rate and seconds saved by the persistent cache over a sequence of fits.

    Registers JAX's own ``/jax/compilation_cache/*`` counters, then runs
    ``n_fits`` fits on *different* mock galaxies drawn from the same model. Data
    is a traced argument (``_get_flat_logdensity``'s contract), so galaxies 2..N
    must be pure cache hits or pure in-process ``jax.jit`` hits -- either way
    zero XLA. What the counters separate is which.
    """
    import jax
    import numpy as np
    from benchmark_notebook_sampler import NOTEBOOKS
    from jax._src import monitoring

    import tengri
    from tengri import Data, ForwardModel, generate_mock
    from tengri.utils.jax_cache import cache_size_bytes, is_cache_enabled

    counts: dict[str, int] = {}
    saved: dict[str, float] = {}
    monitoring.register_event_listener(
        lambda ev, **_: counts.__setitem__(ev, counts.get(ev, 0) + 1)
    )
    monitoring.register_event_duration_secs_listener(
        lambda ev, d, **_: saved.__setitem__(ev, saved.get(ev, 0.0) + d)
    )

    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    size_before = cache_size_bytes() if is_cache_enabled() else None

    rows = []
    for i in range(n_fits):
        sed = cfg["build"](ssp)
        keys = jax.random.split(jax.random.PRNGKey(seed + i), 3)
        mock = generate_mock(sed, sed.spec.sample(keys[0]), key=keys[1], snr=cfg["snr"])
        data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))
        # A fresh ForwardModel per galaxy is the catalog-of-galaxies shape: no
        # in-process jit cache carries over from the previous galaxy's build.
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        map_seed = forward.fit(
            data, method="map", key=keys[2], n_restarts=8, n_steps=800, verbose=False
        )
        before = dict(counts)
        t0 = time.perf_counter()
        forward.fit(
            data,
            key=keys[2],
            init_from=map_seed,
            n_chains=cfg["n_chains"],
            verbose=False,
            **kwargs,
        )
        wall = time.perf_counter() - t0
        rows.append(
            {
                "fit": i,
                "wall_s": round(wall, 2),
                "hits": counts.get("/jax/compilation_cache/cache_hits", 0)
                - before.get("/jax/compilation_cache/cache_hits", 0),
                "misses": counts.get("/jax/compilation_cache/cache_misses", 0)
                - before.get("/jax/compilation_cache/cache_misses", 0),
            }
        )

    return {
        "notebook": notebook,
        "seed": seed,
        "n_fits": n_fits,
        "kwargs": {k: v for k, v in kwargs.items()},
        "device": str(jax.devices()[0].platform),
        "cache_enabled": bool(is_cache_enabled()),
        "cache_bytes_before": size_before,
        "cache_bytes_after": cache_size_bytes() if is_cache_enabled() else None,
        "totals": {k: v for k, v in counts.items() if "compilation_cache" in k},
        "compile_time_saved_s": round(
            saved.get("/jax/compilation_cache/compile_time_saved_sec", 0.0), 2
        ),
        "cache_retrieval_time_s": round(
            saved.get("/jax/compilation_cache/cache_retrieval_time_sec", 0.0), 2
        ),
        "per_fit": rows,
    }


def fit_decompose(notebook: str, *, seed: int, kwargs: dict) -> dict:
    """Split a cold NUTS fit into compile, warmup and sampling — by three fits.

    ``bench/reports/2026-08-30_mclmc_tuning.md`` measured one NUTS fit at
    189.4 s cold and 46.8 s warm in the same process and read the 142.6 s
    difference as **compile**. It is not, or not only. ``run_nuts`` caches its
    adaptation on the Model keyed by ``_adaptation_cache_key``, which includes
    a fingerprint of the target data, so a second ``fit`` on the *same* galaxy
    hits that cache and skips ``_nuts_warmup_only`` **entirely** — its execution
    as well as its compile. A cold-minus-warm difference taken that way is
    ``compile + the whole warmup run``, and the two want opposite fixes.

    Three fits separate them. All three are in one process, so only the first
    pays XLA:

    ``t1`` galaxy A, cold
        compile + warmup + sampling.
    ``t2`` galaxy A again
        sampling only: same data fingerprint, so the adaptation is reused.
    ``t3`` galaxy B, same shape and model
        warmup + sampling: a different fingerprint forces a real adaptation,
        while the compiled programs are already in hand.

    Then ``sampling ~= t2``, ``warmup ~= t3 - t2`` and ``compile ~= t1 - t3``.
    Galaxy B is a different draw from the same generative model, so its
    sampling cost is not identical to A's; the residual shows up as noise in
    the warmup column and the ``t2_b`` row below prices it directly.
    """
    import jax
    import numpy as np
    from benchmark_notebook_sampler import NOTEBOOKS

    import tengri
    from tengri import Data, ForwardModel, generate_mock

    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)

    def mock_at(offset: int):
        sed = cfg["build"](ssp)
        k = jax.random.split(jax.random.PRNGKey(seed + offset), 3)
        m = generate_mock(sed, sed.spec.sample(k[0]), key=k[1], snr=cfg["snr"])
        return Data(photometry=(np.asarray(m["flux_obs"]), np.asarray(m["noise"]))), k[2]

    data_a, key_a = mock_at(0)
    data_b, _key_b = mock_at(1000)

    # ONE ForwardModel for all three fits: the caches this measurement is about
    # (compiled kernels, adaptation) live on the Model, so a fresh build per fit
    # would make every fit cold and measure nothing.
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    map_a = forward.fit(data_a, method="map", key=key_a, n_restarts=8, n_steps=800, verbose=False)
    map_b = forward.fit(data_b, method="map", key=key_a, n_restarts=8, n_steps=800, verbose=False)

    def timed(data, init_from):
        t0 = time.perf_counter()
        post = forward.fit(
            data,
            key=key_a,
            init_from=init_from,
            n_chains=cfg["n_chains"],
            verbose=False,
            **kwargs,
        )
        return time.perf_counter() - t0, post

    t1, _p1 = timed(data_a, map_a)
    t2, _p2 = timed(data_a, map_a)
    t3, _p3 = timed(data_b, map_b)
    t2b, _p4 = timed(data_b, map_b)

    return {
        "probe": "fit_decompose",
        **_machine_load(),
        "notebook": notebook,
        "seed": seed,
        "kwargs": dict(kwargs),
        "device": str(jax.devices()[0].platform),
        "t1_cold_a_s": round(t1, 2),
        "t2_warm_a_s": round(t2, 2),
        "t3_warm_b_s": round(t3, 2),
        "t2_warm_b_s": round(t2b, 2),
        # WITHIN-GALAXY, and therefore sound. Both terms are galaxy B's own, in
        # one process under one machine load.
        "warmup_s_b": round(t3 - t2b, 2),
        "sampling_s_b": round(t2b, 2),
        "warmup_over_sampling_b": round((t3 - t2b) / t2b, 2),
        "warmup_share_of_compiled_fit_b": round((t3 - t2b) / t3, 3),
        # WITHIN-GALAXY upper bound on compile: galaxy A's cold fit minus its own
        # warm one is compile AND warmup together, never compile alone.
        "compile_plus_warmup_s_a": round(t1 - t2, 2),
        # The control that decides whether a cross-galaxy subtraction is allowed
        # at all. Two galaxies of one model, same computation: if this is not
        # ~1.0, between-galaxy variance swamps the compile term and no
        # cross-galaxy estimator of it is meaningful. Measured 2.09 on ctl-jwst,
        # which is why ``compile_s = t1 - t3`` is NOT reported -- it came out
        # negative. Finding 7's 9.45x spread in adapted step size is the same
        # fact seen upstream.
        "sampling_ratio_a_over_b": round(t2 / t2b, 2),
        "zero_compile_over_full_compile": round(t3 / t1, 3),
    }


def _run_cell(argv: list[str], timeout: int) -> dict | None:
    """One cell in its own subprocess, so a hang is reported rather than fatal."""
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, os.path.abspath(__file__), *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    sys.stderr.write(proc.stderr[-2000:])
    return None


def main(argv=None) -> int:
    from benchmark_notebook_sampler import NOTEBOOKS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default="05", choices=sorted(NOTEBOOKS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--max-doublings", type=int, nargs="+", default=[10])
    ap.add_argument("--n-leapfrog", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--target-accept", type=float, default=0.85)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--precondition", type=float, default=None)
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="lower and compile only. The compile question does not need the "
        "warm call, and a 600-step NUTS warmup at cap 10 is minutes of it.",
    )
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--cache-probe",
        type=int,
        default=None,
        metavar="N",
        help="instead of the anatomy, run N fits on distinct galaxies with JAX's "
        "own compilation-cache counters registered, and report the hit rate.",
    )
    ap.add_argument(
        "--fit-decompose",
        action="store_true",
        help="instead of the anatomy, split one cold fit into compile, warmup "
        "and sampling by fitting the same galaxy twice and a second galaxy twice.",
    )
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.fit_decompose:
        cfg = NOTEBOOKS[args.notebook]
        row = fit_decompose(
            args.notebook,
            seed=args.seed,
            kwargs=dict(cfg["shipped"], max_num_doublings=args.max_doublings[0]),
        )
        print(json.dumps(row, indent=2))
        if args.json:
            _append(args.json, row)
        return 0

    if args.cache_probe is not None and args.cell is None:
        cfg = NOTEBOOKS[args.notebook]
        row = cache_probe(
            args.notebook,
            seed=args.seed,
            n_fits=args.cache_probe,
            kwargs=dict(cfg["shipped"], max_num_doublings=args.max_doublings[0]),
        )
        print(json.dumps(row))
        if args.json:
            _append(args.json, row)
        return 0

    if args.cell is not None:
        row = measure_stage(
            args.cell,
            args.notebook,
            seed=args.seed,
            n_warmup=args.warmup,
            n_samples=args.samples,
            max_doublings=args.max_doublings[0],
            n_leapfrog=args.n_leapfrog,
            use_dense=args.dense,
            target_accept_rate=args.target_accept,
            precondition=args.precondition,
            skip_run=args.skip_run,
        )
        print(json.dumps(row))
        return 0

    rows = []
    for cap in args.max_doublings:
        for stage in args.stage:
            if not stage.startswith("nuts") and cap != args.max_doublings[0]:
                continue  # only the NUTS stages read the cap; the rest would repeat
            cell = [
                "--cell",
                stage,
                "--notebook",
                args.notebook,
                "--seed",
                str(args.seed),
                "--max-doublings",
                str(cap),
                "--n-leapfrog",
                str(args.n_leapfrog),
                "--warmup",
                str(args.warmup),
                "--samples",
                str(args.samples),
                "--target-accept",
                str(args.target_accept),
            ]
            if args.dense:
                cell.append("--dense")
            if args.precondition is not None:
                cell += ["--precondition", str(args.precondition)]
            if args.skip_run:
                cell.append("--skip-run")
            row = _run_cell(cell, args.timeout)
            if row is None:
                row = {"stage": stage, "max_doublings": cap, "status": "failed"}
            rows.append(row)
            print(json.dumps(row), flush=True)

    print()
    print(f"{'stage':<14}{'cap':>5}{'trace+lower':>13}{'compile':>10}{'run':>9}{'HLO lines':>12}")
    for r in rows:
        if r.get("status") == "failed":
            print(f"{r['stage']:<14}{r['max_doublings']:>5}{'FAILED':>13}")
            continue
        print(
            f"{r['stage']:<14}{r['max_doublings']:>5}{r['trace_lower_s']:>13.2f}"
            f"{r['compile_s']:>10.2f}"
            f"{'n/a' if r['run_s'] is None else format(r['run_s'], '.2f'):>9}"
            f"{r['stablehlo_lines']:>12}"
        )
    if args.json:
        for r in rows:
            _append(args.json, r)
    return 0


def _append(path: str, row: dict) -> None:
    existing = []
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)
    existing.append(row)
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
