# SPDX-License-Identifier: BSD-3-Clause
"""Separate compile from sampling for the catalog MCMC engines.

``bench/reports/2026-08-30_gpu_catalog_throughput.md`` Finding 3 records that
**no** catalog ``mcmc_nuts`` cell ever produced a row: six cells, GPU and CPU,
all killed by their timeout, including one capped to at most three leapfrogs per
step against an ``mcmc_hmc`` cell of the same shape that finished in 30 s. Its
own closing paragraph names the measurement it did not take:

    "Timing ``build_catalog_mcmc_engine`` for ``nuts`` against ``hmc`` with
    ``TENGRI_DISABLE_JAX_CACHE=1``, at one galaxy, is the obvious next
    measurement. It is left undone rather than guessed at."

This script is that measurement. Because the throughput harness only ever times
a whole ``CatalogFitter.run``, a cell that dies in XLA and a cell that dies in
the sampler are indistinguishable in it. Here the two halves are timed
separately, through ``jax.jit(...).lower(...).compile()``:

* **trace+lower** — building the jaxpr and the HLO module, no XLA backend work;
* **compile** — XLA's own optimization of that module;
* **run** — one warm call, after ``block_until_ready`` on a first call.

and reported per ``(method, K)`` so the *scaling in K* is visible. That is the
axis that decides the question: a fixed cost per NUTS build is a compile
constant, while a cost that grows with K is the batched-``while_loop`` story.

Every cell is run in its own subprocess with its own timeout, so a hang in one
does not take the sweep with it, and a killed cell is *reported as killed*
rather than dropped.

Examples
--------
Both methods, K = 1 and 8, on whatever device JAX picks::

    JAX_DEFAULT_MATMUL_PRECISION=highest \\
    python bench/scripts/benchmark_catalog_compile.py --chunk 1 8

One cell, in-process (what the subprocess driver runs)::

    python bench/scripts/benchmark_catalog_compile.py --cell mcmc_nuts 1
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


def _fixture(n_data_only: bool = False):
    """The throughput harness's own dpl fixture, reused verbatim.

    Reusing it rather than writing a second one is deliberate: this script
    exists to explain a row that harness failed to produce, and a different
    model would leave "different fixture" as a live alternative explanation.
    """
    import jax
    from benchmark_catalog_throughput import _load_or_synth_ssp, build_model, make_catalog

    ssp, tag = _load_or_synth_ssp()
    model = build_model(ssp, tag)
    galaxies = make_catalog(model, 1, jax.random.PRNGKey(0))
    return model, galaxies, tag


def measure_cell(
    method: str,
    chunk: int,
    *,
    n_warmup: int,
    n_samples: int,
    max_doublings: int = 10,
    use_dense: bool = False,
    n_gal: int | None = None,
    n_ensemble: int = 8,
    max_leapfrog_steps: int = 64,
    precondition: float | None = None,
) -> dict:
    """Trace/lower, compile and run one catalog engine at batch width ``chunk``.

    Parameters
    ----------
    method : str
        ``"mcmc_nuts"`` or ``"mcmc_hmc"``.
    chunk : int
        Batch width K. ``1`` traces ``run_one`` unbatched; ``>1`` wraps it in
        ``jax.vmap``, which is exactly what ``lax.map(batch_size=K)`` does per
        step in :meth:`CatalogFitter._run_native_mcmc`.
    n_warmup, n_samples : int
        Adaptation and kept-draw budget. Kept tiny; the point is the *fixed*
        cost, and a budget that scales the sampling half would hide it.
    max_doublings : int
        NUTS tree-depth cap. Since Phase 3 this reaches the *warmup* kernel too;
        before it, only the sampling scan honored it, which is why capping looked
        ineffective in ``2026-08-30_gpu_catalog_throughput.md`` Finding 3.
    use_dense : bool
        Dense vs diagonal mass matrix.
    n_gal : int, optional
        Catalog size N. ``None`` measures one ``lax.map`` step (a bare
        ``vmap`` of width ``chunk``); an int wraps ``run_one`` in
        ``jax.lax.map(..., batch_size=chunk)`` over N rows, which is exactly what
        :meth:`CatalogFitter._run_native_mcmc` builds. **This is the axis the
        O(1)-in-N contract is about**: sweep it at fixed ``chunk`` and both the
        compile column and the StableHLO line count must stay flat.
    n_ensemble, max_leapfrog_steps : int
        ChEES only. The ensemble is chains-*within*-galaxy, so a cell carries
        ``chunk * n_ensemble`` live chains through every adaptation step.
    precondition : float or None
        Whitening strength for the per-galaxy analytic metric, or ``None`` for
        off. **The point of the sweep when it is set**: the metric is built,
        factorized and applied inside the ``lax.map``, so it enlarges the graph
        once. If it enlarged it per galaxy the StableHLO line count would grow
        with ``n_gal`` and the O(1)-in-N contract would be broken.

    Returns
    -------
    dict
        ``trace_lower_s``, ``compile_s``, ``run_s``, plus the HLO instruction
        count, which is the size XLA is actually being asked to optimize.
    """
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
    from tengri.inference.catalog_fitter import CatalogFitter

    model, galaxies, ssp_tag = _fixture()
    cat = CatalogFitter(model, galaxies)
    fitter = cat._get_dummy_fitter()

    from tengri.inference.backends.mcmc.chees import CHEES_TARGET_ACCEPT_RATE

    sampler = {"mcmc_nuts": "nuts", "mcmc_hmc": "hmc", "mcmc_chees": "chees"}[method]
    run_one, _unravel = build_catalog_mcmc_engine(
        fitter,
        sampler,
        n_warmup=n_warmup,
        n_burnin=0,
        n_samples=n_samples,
        max_num_doublings=max_doublings,
        n_leapfrog=10,
        # ChEES's own dual-averaging target is 0.651, not NUTS's 0.85 -- carrying
        # the NUTS value across would still run, just at a step size chosen for a
        # different proposal.
        target_accept_rate=CHEES_TARGET_ACCEPT_RATE if sampler == "chees" else 0.85,
        use_dense=use_dense,
        n_ensemble=n_ensemble,
        max_leapfrog_steps=max_leapfrog_steps,
        precondition=precondition,
    )

    d_params = ravel_pytree(fitter._initialize_unbounded(jax.random.PRNGKey(0)))[0].shape[0]
    n_data = len(galaxies[0]["flux_obs"])

    def _stack(x, k):
        return jnp.broadcast_to(jnp.asarray(x), (k, *jnp.asarray(x).shape))

    width = chunk if n_gal is None else n_gal
    init = jnp.zeros((width, d_params))
    keys = jax.random.split(jax.random.PRNGKey(0), width)
    data = _stack(galaxies[0]["flux_obs"], width)
    noise = _stack(galaxies[0]["noise"], width)
    presence = jnp.ones((width, n_data))
    redshift = jnp.zeros((width,))
    lf = jnp.zeros((width, 0))
    le = jnp.zeros((width, 0))
    args = (init, keys, data, noise, presence, redshift, lf, le)

    if n_gal is not None:
        # The fitter's own shape: scan N/chunk steps of a chunk-wide vmap. This
        # is the arrangement the O(1)-in-N contract is a claim about.
        fn = jax.jit(lambda *a: jax.lax.map(lambda row: run_one(*row), a, batch_size=chunk))
    elif chunk == 1:
        fn = jax.jit(run_one)
        args = tuple(a[0] for a in args)
    else:
        fn = jax.jit(jax.vmap(run_one))

    t0 = time.perf_counter()
    lowered = fn.lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    out = compiled(*args)
    jax.block_until_ready(out)
    t3 = time.perf_counter()
    out = compiled(*args)
    jax.block_until_ready(out)
    t4 = time.perf_counter()

    try:
        hlo = lowered.as_text()
        hlo_lines = hlo.count("\n")
    except Exception:
        hlo_lines = None

    return {
        "method": method,
        "chunk": chunk,
        "n_gal": n_gal,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "max_doublings": int(max_doublings),
        "precondition": precondition,
        "use_dense": bool(use_dense),
        "d_params": int(d_params),
        "n_data": int(n_data),
        "ssp": ssp_tag,
        "device": str(jax.devices()[0].platform),
        "trace_lower_s": round(t1 - t0, 3),
        "compile_s": round(t2 - t1, 3),
        "first_run_s": round(t3 - t2, 3),
        "warm_run_s": round(t4 - t3, 3),
        "stablehlo_lines": hlo_lines,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", nargs="+", default=["mcmc_hmc", "mcmc_nuts"])
    ap.add_argument("--chunk", type=int, nargs="+", default=[1, 8])
    ap.add_argument(
        "--n-gal",
        type=int,
        nargs="+",
        default=[None],
        help="catalog sizes N to sweep at each chunk. Omitted measures one "
        "lax.map step; given, wraps run_one in lax.map(batch_size=chunk) over N "
        "rows -- the axis the O(1)-in-N compile contract is about.",
    )
    ap.add_argument("--n-ensemble", type=int, default=8, help="mcmc_chees ensemble width")
    ap.add_argument(
        "--precondition",
        type=float,
        default=None,
        metavar="ALPHA",
        help="per-galaxy analytic J^T N^-1 J + I metric at whitening strength "
        "ALPHA. Omit for off. Sweep --n-gal with it set to check the metric did "
        "not break compile O(1) in N.",
    )
    ap.add_argument(
        "--max-leapfrog-steps", type=int, default=64, help="mcmc_chees trajectory-length cap"
    )
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--max-doublings", type=int, default=10, help="NUTS tree-depth cap")
    ap.add_argument("--dense", action="store_true", help="dense mass matrix")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell subprocess timeout [s]")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--cell",
        nargs=2,
        metavar=("METHOD", "CHUNK"),
        default=None,
        help="run one cell in-process and print its JSON (the subprocess entry point)",
    )
    args = ap.parse_args(argv)

    if args.cell is not None:
        row = measure_cell(
            args.cell[0],
            int(args.cell[1]),
            n_warmup=args.warmup,
            n_samples=args.samples,
            max_doublings=args.max_doublings,
            use_dense=args.dense,
            n_gal=None if args.n_gal[0] in (None, 0) else int(args.n_gal[0]),
            n_ensemble=args.n_ensemble,
            max_leapfrog_steps=args.max_leapfrog_steps,
            precondition=args.precondition,
        )
        print("__ROW__" + json.dumps(row))
        return 0

    rows = []
    for method in args.method:
        for chunk in args.chunk:
            for n_gal in args.n_gal:
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--cell",
                    method,
                    str(chunk),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--max-doublings",
                    str(args.max_doublings),
                    "--n-ensemble",
                    str(args.n_ensemble),
                    "--max-leapfrog-steps",
                    str(args.max_leapfrog_steps),
                ]
                if n_gal is not None:
                    cmd += ["--n-gal", str(n_gal)]
                if args.precondition is not None:
                    cmd += ["--precondition", str(args.precondition)]
                if args.dense:
                    cmd.append("--dense")
                label = f"K={chunk}" + ("" if n_gal is None else f" N={n_gal}")
                print(f"-> {method} {label} (timeout {args.timeout}s)", flush=True)
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                        env=os.environ.copy(),
                    )
                except subprocess.TimeoutExpired:
                    rows.append(
                        {
                            "method": method,
                            "chunk": chunk,
                            "n_gal": n_gal,
                            "outcome": "timeout",
                            "timeout_s": args.timeout,
                        }
                    )
                    print(f"   TIMED OUT after {args.timeout}s", flush=True)
                    continue
                wall = time.perf_counter() - t0
                row = None
                for line in proc.stdout.splitlines():
                    if line.startswith("__ROW__"):
                        row = json.loads(line[len("__ROW__") :])
                if row is None:
                    rows.append(
                        {
                            "method": method,
                            "chunk": chunk,
                            "n_gal": n_gal,
                            "outcome": "error",
                            "stderr": proc.stderr[-2000:],
                        }
                    )
                    print(f"   FAILED\n{proc.stderr[-2000:]}", flush=True)
                    continue
                row["outcome"] = "ok"
                row["subprocess_wall_s"] = round(wall, 3)
                rows.append(row)
                print(
                    f"   lower {row['trace_lower_s']}s  compile {row['compile_s']}s  "
                    f"warm {row['warm_run_s']}s  hlo {row['stablehlo_lines']} lines",
                    flush=True,
                )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
