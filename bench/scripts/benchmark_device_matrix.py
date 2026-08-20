#!/usr/bin/env python
"""Device matrix: CPU against GPU, float64 against float32, prediction and inference.

Answers two questions that the CPU-only benchmark tree cannot: does a shape run
at all on CUDA, and is it faster there. The axes are device, precision, work
shape, and ``approx``.

Device- and precision-agnostic by construction. It does **not** call
``os.environ.setdefault("JAX_PLATFORMS", "cpu")`` the way the rest of
``bench/scripts`` does, and it does **not** force ``jax_enable_x64``: both are
selected in the environment before the process starts, which is the only route
that works. Selecting float32 after ``import tengri`` is too late — constants
allocated during the import are already placed — and
``SEDModel.build(forward_dtype=...)`` has been inert since #1433.

One shape per process. Running a forward pass, a gradient and a fit in one
process makes the later ones look worse; that contamination alone has flipped a
conclusion before (``notebooks/apple_mps.py``). ``--all`` therefore spawns one
child per cell and merges their JSON.

Shapes
------
``A``  forward ``predict_photometry``, one galaxy
``B``  gradient of ``sum(predict_photometry)``, one galaxy
``C``  batch sweep over ``predict_photometry_batch`` (a ``vmap``)
``D``  MAP fit, adam
``E``  catalog NUTS, sweeping ``forward_chunk_size``
``G``  dump raw outputs for the cross-precision accuracy comparison

Usage::

    # one cell
    JAX_PLATFORMS=cuda python bench/scripts/benchmark_device_matrix.py --shape C

    # float32 on the GPU
    JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda \\
        python bench/scripts/benchmark_device_matrix.py --shape C --precision f32

    # the whole matrix, one child process per cell. Pin the DRIVER to the CPU:
    # this module imports jax at import scope, so a driver left on the GPU
    # preallocates most of the card and every GPU child then competes with its
    # own parent for VRAM (measured: 9-10 GiB held by the driver alone).
    JAX_PLATFORMS=cpu python bench/scripts/benchmark_device_matrix.py --all

    # accuracy table from the shape-G dumps
    python bench/scripts/benchmark_device_matrix.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np

# Deliberately no jax.config.update("jax_enable_x64", ...) here: tengri's import
# honors JAX_ENABLE_X64 and holds the choice for the whole import (#1880).
import tengri  # noqa: F401  (import for side effect: the x64 guard)

DEFAULT_OUT = os.path.join("bench", "results", "device_matrix.json")
DEFAULT_DUMP_DIR = os.path.join("bench", "results", "device_matrix_arrays")
SSP_BARE = "fsps_prsc_miles_chabrier"  # bare-stellar; load_ssp resolves the path
BATCHES = (1, 8, 32, 128, 512, 2048)
SHAPES = ("A", "B", "C", "D", "E", "G")


# ── precision and device bookkeeping ──────────────────────────────────────


def precision_tag() -> str:
    """Return ``"f64"`` or ``"f32"`` for the session's default float width."""
    return "f64" if str(jnp.zeros(1).dtype) == "float64" else "f32"


def require_precision(want: str) -> None:
    """Abort unless the session really runs at the requested width.

    ``src/tengri/__init__.py`` records a float32 probe, benchmark and bug report
    that all silently ran in float64 because ``JAX_ENABLE_X64=0`` was being
    discarded. A cell that cannot prove its precision must not produce a row.
    """
    got = precision_tag()
    if got != want:
        raise SystemExit(
            f"precision mismatch: asked for {want}, session is {got} "
            f"(x64={jax.config.jax_enable_x64}, default dtype={jnp.zeros(1).dtype}). "
            "Set JAX_ENABLE_X64=0 in the environment before starting python."
        )


def confirm_output_precision(arr, want: str) -> str:
    """Re-check the width on a real output array, after the first forward call.

    The import-time guard cannot close one hole: DSPS modules imported lazily
    inside function bodies run a bare ``jax.config.update("jax_enable_x64",
    True)`` and can flip the flag mid-run (``__init__.py``). So the flag is
    checked again here, and against an array rather than the config.
    """
    dtype = str(np.asarray(jax.tree_util.tree_leaves(arr)[0]).dtype)
    expect = "float64" if want == "f64" else "float32"
    if dtype != expect:
        raise SystemExit(f"output dtype drifted to {dtype}, expected {expect}")
    if bool(jax.config.jax_enable_x64) != (want == "f64"):
        raise SystemExit(f"x64 flag flipped mid-run to {jax.config.jax_enable_x64}")
    return dtype


def gpu_snapshot() -> dict:
    """VRAM in use and utilization, or an empty dict when nvidia-smi is absent."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        used, total, util = (x.strip() for x in out.stdout.strip().split(",")[:3])
        return {
            "vram_used_mib": int(used),
            "vram_total_mib": int(total),
            "gpu_util_pct": int(util),
        }
    except Exception:
        return {}


def environment() -> dict:
    """Everything a reader needs to judge whether a row is comparable to theirs."""
    dev = jax.devices()
    return {
        "platform": dev[0].platform,
        "device": str(dev[0]),
        "n_devices": len(dev),
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "default_dtype": str(jnp.zeros(1).dtype),
        "precision": precision_tag(),
        **gpu_snapshot(),
    }


# ── timing ────────────────────────────────────────────────────────────────


def bench_call(fn, n_warmup: int = 3, n_runs: int = 20) -> tuple[float, float]:
    """Return ``(compile_ms, steady_us)``; the first call's wall is the compile.

    Every call is blocked on: JAX dispatch is async, and without the block the
    GPU rows measure enqueue time rather than execution.
    """
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(max(0, n_warmup - 1)):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(fn())
    steady_us = (time.perf_counter() - t0) / n_runs * 1e6
    return compile_ms, steady_us


def first_call_ms(fn) -> float:
    """Wall of the very first call — the compile. Only meaningful once."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return round((time.perf_counter() - t0) * 1e3, 2)


def bench_rotated(fn, reps: int, n_runs: int) -> dict:
    """Time one callable ``reps`` times and report the spread as an A/A floor.

    The same arm measured twice bounds what this machine can resolve. A ratio
    between two different arms is only worth quoting if it clears this.

    Assumes the callable is already compiled: compile time is a separate
    measurement (:func:`first_call_ms`) because it can only be taken once, and
    reporting a warm call under that name is how a compile-time claim goes
    wrong.
    """
    steady = [bench_call(fn, n_warmup=1, n_runs=n_runs)[1] for _ in range(reps)]
    lo, hi = min(steady), max(steady)
    return {
        "steady_us": round(lo, 3),
        "steady_us_all": [round(s, 3) for s in steady],
        "aa_ratio": round(hi / lo, 3) if lo > 0 else None,
    }


def flops(fn, *args) -> int | None:
    """Compiled-HLO FLOP count — machine-independent, unlike a wall clock.

    The house instrument whenever the question is "does this path do less
    work"; device choice is not that question, so this is reported beside the
    wall clock as the invariant, never instead of it.
    """
    try:
        return int(jax.jit(fn).lower(*args).compile().cost_analysis()["flops"])
    except Exception:
        return None


# ── fixture ───────────────────────────────────────────────────────────────


def build(approx: str):
    """Build the benchmark model: ``recipes.mock_recovery_minimal`` at z = 0.05.

    Nebular emission and AGN are off. That is not only for speed: the pure-f32
    inventory in ``tests/regression/precision`` carries strict xfails for three
    AGN discs and records SKIRTOR interpolation failures, so an AGN model would
    measure a known-broken path rather than the common one.

    The SSP is loaded here, inside whatever precision context the process is
    in — a grid loaded while x64 was still on never reaches the float32 gates,
    and 13 downstream gates key on ``wave.dtype``.
    """
    from tengri import SEDModel, WavePrecomp, load_ssp, recipes
    from tengri.observation import Observation, Photometry

    ssp = load_ssp(SSP_BARE)
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["approx"] = WavePrecomp() if approx == "wave_precomp" else None
    t0 = time.perf_counter()
    model = SEDModel.build(ssp_data=ssp, observation=obs, **recipe)
    build_s = time.perf_counter() - t0
    return model, build_s, str(np.asarray(ssp.ssp_flux).dtype)


def reference_params(model) -> dict:
    """One galaxy, specified identically at either precision.

    Not a prior draw: ``jax.random`` returns different numbers for the same key
    at float32 and float64, so a sampled fixture compares two *different*
    galaxies across the precision arms and reports the difference as precision
    error. Measured, that confound was a factor of 152 on summed photometry.

    Each free parameter is taken at the median of its declared prior — the
    ``unstandardize(0)`` pushforward every distribution provides — and rounded,
    so both arms receive the same Python floats and cast them once.
    """
    spec = model.spec
    out = {}
    for name in spec.free_params:
        value = np.asarray(spec.get_distribution(name).unstandardize(jnp.zeros(())))
        out[name] = round(float(value), 6) if value.ndim == 0 else value.astype(np.float64)
    return out


def batch_params(model, n: int):
    """Broadcast the reference galaxy to a batch of ``n``."""
    p = reference_params(model)
    return {
        k: jnp.broadcast_to(jnp.asarray(v), (n, *jnp.shape(jnp.asarray(v)))) for k, v in p.items()
    }


def mock_catalog(model, n_gal: int, key):
    """A photometric mock catalog, 5% noise, as the catalog cells' data."""
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        flux = model.predict_photometry(dict(model.spec.sample(k)))
        noise = jnp.abs(flux) * 0.05
        draw = jax.random.normal(jax.random.fold_in(k, 1), flux.shape)
        galaxies.append({"flux_obs": flux + noise * draw, "noise": noise})
    return galaxies


# ── the cells ─────────────────────────────────────────────────────────────


def cell_forward(model, args) -> dict:
    """Shape A — one galaxy through ``predict_photometry``."""
    p = reference_params(model)
    fn = jax.jit(model.predict_photometry)
    compile_ms = first_call_ms(lambda: fn(p))
    dtype = confirm_output_precision(fn(p), args.precision)
    row = bench_rotated(lambda: fn(p), args.reps, args.runs)
    return {
        **row,
        "compile_ms": compile_ms,
        "out_dtype": dtype,
        "flops": flops(model.predict_photometry, p),
    }


def cell_gradient(model, args) -> dict:
    """Shape B — gradient of the summed photometry, one galaxy."""
    p = reference_params(model)

    def loss(q):
        return jnp.sum(model.predict_photometry(q))

    fn = jax.jit(jax.grad(loss))
    compile_ms = first_call_ms(lambda: fn(p))
    dtype = confirm_output_precision(fn(p), args.precision)
    row = bench_rotated(lambda: fn(p), args.reps, args.runs)
    return {
        **row,
        "compile_ms": compile_ms,
        "out_dtype": dtype,
        "flops": flops(jax.grad(loss), p),
    }


def cell_batch(model, args) -> dict:
    """Shape C — the batch sweep. The widest shape, and the GPU's best case."""
    fwd = jax.jit(model.predict_photometry_batch)

    def grad_one(p):
        return jax.grad(lambda q: jnp.sum(model.predict_photometry(q)))(p)

    grad_batch = jax.jit(jax.vmap(grad_one))

    rows = []
    for n in args.batches:
        pb = batch_params(model, n)
        try:
            compile_ms = first_call_ms(lambda pb=pb: fwd(pb))
            dtype = confirm_output_precision(fwd(pb), args.precision)
            fwd_row = bench_rotated(lambda pb=pb: fwd(pb), args.reps, max(3, args.runs // 2))
            grad_row = bench_rotated(
                lambda pb=pb: grad_batch(pb), args.reps, max(3, args.runs // 2)
            )
        except Exception as exc:  # OOM is a result on a 12 GB card, not a crash
            rows.append({"n": n, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            break
        rows.append(
            {
                "n": n,
                "out_dtype": dtype,
                "forward_compile_ms": compile_ms,
                "forward_us": fwd_row["steady_us"],
                "forward_us_per_gal": round(fwd_row["steady_us"] / n, 4),
                "forward_aa": fwd_row["aa_ratio"],
                "grad_us": grad_row["steady_us"],
                "grad_us_per_gal": round(grad_row["steady_us"] / n, 4),
                "grad_aa": grad_row["aa_ratio"],
                "vram": gpu_snapshot().get("vram_used_mib"),
            }
        )
    return {"sweep": rows}


def cell_map(model, args) -> dict:
    """Shape D — a MAP fit with adam, the optimizer that stays on the device.

    ``optimizer="lbfgs_scipy"`` is deliberately not swept: scipy drives a host
    loop and converts every gradient to ``np.float64``, so it measures the host.
    """
    from tengri import Fitter

    key = jax.random.PRNGKey(0)
    truth = reference_params(model)
    flux = model.predict_photometry(truth)
    sigma = jnp.abs(flux) * 0.05
    data = flux + sigma * jax.random.normal(jax.random.fold_in(key, 1), flux.shape)

    fitter = Fitter(model, data, sigma, data_type="photometry")
    resolved = str(fitter.model.approx)

    walls = []
    for _ in range(args.reps + 1):  # the first includes compilation
        t0 = time.perf_counter()
        post = fitter.run("map", n_steps=args.map_steps, verbose=False)
        # MAP returns a point estimate in `params`; `samples` is None.
        jax.block_until_ready(jax.tree_util.tree_leaves(post.params)[0])
        walls.append(round(time.perf_counter() - t0, 4))

    best = {k: float(np.asarray(v).ravel()[0]) for k, v in post.params.items()}
    finite = all(np.isfinite(list(best.values())))
    moved = {k: round(best[k] - truth[k], 6) for k in truth if k in best}
    return {
        "resolved_approx": resolved,
        "cold_s": walls[0],
        "warm_s": min(walls[1:]),
        "warm_s_all": walls[1:],
        "n_steps": args.map_steps,
        "params_finite": finite,
        "params_moved": moved,
        "any_movement": any(abs(v) > 1e-9 for v in moved.values()),
        "map_params": best,
        "truth": {k: float(np.asarray(v).ravel()[0]) for k, v in truth.items()},
    }


def cell_catalog(model, args) -> dict:
    """Shape E — vectorized per-galaxy sampling, sweeping ``forward_chunk_size``.

    ``mcmc_nuts`` and ``mcmc_hmc`` are the only backends the catalog path maps
    over galaxies; every other name falls back to a sequential per-galaxy
    ``Fitter``, where a device has nothing wide to do.

    Which of the two matters on a device. NUTS chooses its trajectory length per
    draw, so under ``vmap`` every galaxy in the batch runs until the *longest*
    tree in that batch finishes; fixed-length HMC gives every galaxy identical
    work. Compare them with ``--method``.
    """
    from tengri.inference.catalog_fitter import CatalogFitter

    key = jax.random.PRNGKey(0)
    galaxies = mock_catalog(model, max(args.n_gal), key)
    run_kw = {"n_warmup": args.warmup, "n_burnin": 10, "n_samples": args.samples}

    rows = []
    for n_gal in args.n_gal:
        cat = CatalogFitter(model, galaxies[:n_gal], data_type="photometry")
        for chunk in args.chunk:
            if n_gal < chunk:
                continue
            try:
                walls = []
                for _ in range(2):  # cold then warm
                    t0 = time.perf_counter()
                    cp = cat.run(
                        args.method,
                        key=key,
                        forward_chunk_size=chunk,
                        verbose=False,
                        dense_mass_matrix=False,
                        **run_kw,
                    )
                    jax.block_until_ready(cp[0].samples)
                    walls.append(time.perf_counter() - t0)
                draws = np.asarray(jax.tree_util.tree_leaves(cp[0].samples)[0])
                rows.append(
                    {
                        "n_gal": n_gal,
                        "chunk": chunk,
                        "cold_s": round(walls[0], 3),
                        "warm_s": round(walls[1], 3),
                        "gal_per_s": round(n_gal / walls[1], 2),
                        "draws_finite": bool(np.all(np.isfinite(draws))),
                        "vram": gpu_snapshot().get("vram_used_mib"),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "n_gal": n_gal,
                        "chunk": chunk,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                )
    return {"method": args.method, "sweep": rows}


def cell_dump(model, args) -> dict:
    """Shape G — write raw outputs so accuracy is compared across processes.

    Comparing precisions inside one process would share the ``tengri_precomp``
    npz cache, which is keyed on the SSP bytes and filters but not on the x64
    flag or the backend. Dumping and comparing later keeps the arms apart.
    """
    os.makedirs(args.dump_dir, exist_ok=True)
    p = reference_params(model)
    phot = np.asarray(model.predict_photometry(p), dtype=np.float64)
    grad = jax.grad(lambda q: jnp.sum(model.predict_photometry(q)))(p)
    grad_flat = {k: float(np.asarray(v, dtype=np.float64)) for k, v in grad.items()}

    dev = jax.devices()[0].platform
    path = os.path.join(args.dump_dir, f"G_{dev}_{args.precision}_{args.approx}.npz")
    np.savez(
        path,
        photometry=phot,
        grad_keys=np.array(list(grad_flat), dtype=object),
        grad_vals=np.array(list(grad_flat.values()), dtype=np.float64),
    )
    return {
        "dump": path,
        "photometry_finite": bool(np.all(np.isfinite(phot))),
        "grad_finite": bool(np.all(np.isfinite(list(grad_flat.values())))),
        "photometry_sum": float(phot.sum()),
    }


CELLS = {
    "A": cell_forward,
    "B": cell_gradient,
    "C": cell_batch,
    "D": cell_map,
    "E": cell_catalog,
    "G": cell_dump,
}


# ── comparison ────────────────────────────────────────────────────────────


def compare(dump_dir: str) -> dict:
    """Relative error of every shape-G dump against the float64 CPU reference.

    Errors are masked at ``|reference| > 1e-45``: below that a relative error is
    a ratio of two numbers that are both noise.
    """
    ref_name = None
    dumps = {}
    for fn in sorted(os.listdir(dump_dir)):
        if not fn.startswith("G_") or not fn.endswith(".npz"):
            continue
        tag = fn[2:-4]
        dumps[tag] = np.load(os.path.join(dump_dir, fn), allow_pickle=True)
        if tag.startswith("cpu_f64"):
            ref_name = tag
    if ref_name is None:
        raise SystemExit(f"no float64 CPU reference in {dump_dir}; run --device cpu --shape G")

    ref = dumps[ref_name]
    out = {"reference": ref_name, "rows": []}
    for tag, d in dumps.items():
        rp, dp = ref["photometry"], d["photometry"]
        mask = np.abs(rp) > 1e-45
        rel = np.abs(dp[mask] - rp[mask]) / np.abs(rp[mask]) if mask.any() else np.array([np.nan])
        rg, dg = ref["grad_vals"], d["grad_vals"]
        gmask = np.abs(rg) > 1e-45
        grel = (
            np.abs(dg[gmask] - rg[gmask]) / np.abs(rg[gmask])
            if gmask.any()
            else np.array([np.nan])
        )
        out["rows"].append(
            {
                "arm": tag,
                "phot_max_rel": float(np.max(rel)),
                "phot_med_rel": float(np.median(rel)),
                "grad_max_rel": float(np.max(grel)),
                "n_bands_compared": int(mask.sum()),
                "finite": bool(np.all(np.isfinite(dp))),
            }
        )
    return out


# ── driver ────────────────────────────────────────────────────────────────


def spawn(shape: str, device: str, precision: str, approx: str, extra: list[str]) -> dict:
    """Run one cell in its own process, with its own platform and precision.

    Each child gets its own precompute cache directory: that cache is not keyed
    on dtype or backend, so sharing one between a float32 and a float64 cell
    would let the first contaminate the second.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["TENGRI_NO_BACKGROUND_COMPILE"] = "1"
    env["TENGRI_PRECOMP_CACHE_DIR"] = os.path.expanduser(
        f"~/.cache/tengri_precomp_bench_{device}_{precision}"
    )
    if precision == "f32":
        env["JAX_ENABLE_X64"] = "0"
    else:
        env.pop("JAX_ENABLE_X64", None)

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--shape",
        shape,
        "--precision",
        precision,
        "--approx",
        approx,
        "--emit-json",
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    for line in proc.stdout.splitlines():
        if line.startswith("__ROW__"):
            return json.loads(line[len("__ROW__") :])
    return {
        "shape": shape,
        "device": device,
        "precision": precision,
        "approx": approx,
        "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:300],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", choices=SHAPES)
    ap.add_argument("--precision", choices=("f64", "f32"), default=None)
    ap.add_argument("--approx", choices=("wave_precomp", "exact"), default="wave_precomp")
    ap.add_argument("--device", choices=("cpu", "gpu"), default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    ap.add_argument("--n-gal", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--chunk", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument(
        "--method",
        default="mcmc_nuts",
        choices=("mcmc_nuts", "mcmc_hmc"),
        help="catalog sampler (shape E); the only two the catalog path vmaps",
    )
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--map-steps", type=int, default=300)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR)
    ap.add_argument("--all", action="store_true", help="spawn the full matrix, one child per cell")
    ap.add_argument("--compare", action="store_true", help="accuracy table from shape-G dumps")
    ap.add_argument("--emit-json", action="store_true", help="print the row as __ROW__<json>")
    args = ap.parse_args(argv)

    if args.compare:
        print(json.dumps(compare(args.dump_dir), indent=2))
        return 0

    if args.all:
        rows = []
        for device in ("cpu", "gpu"):
            for precision in ("f64", "f32"):
                for shape in [args.shape] if args.shape else list(SHAPES):
                    extra: list[str] = []
                    if shape == "C":
                        extra = ["--batches", *[str(b) for b in args.batches]]
                    elif shape == "E":
                        extra = [
                            "--n-gal",
                            *[str(n) for n in args.n_gal],
                            "--chunk",
                            *[str(c) for c in args.chunk],
                        ]
                    print(f"  → {device:3s} {precision} shape {shape}", flush=True)
                    row = spawn(shape, device, precision, args.approx, extra)
                    rows.append(row)
                    print(f"    {json.dumps(row)[:160]}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {len(rows)} rows to {args.out}")
        return 0

    if args.shape is None:
        ap.error("pass --shape, or --all, or --compare")

    want = args.precision or precision_tag()
    require_precision(want)
    args.precision = want

    model, build_s, ssp_dtype = build(args.approx)
    row = {
        "shape": args.shape,
        "device": jax.devices()[0].platform,
        "precision": want,
        "approx": args.approx,
        "build_s": round(build_s, 3),
        "ssp_dtype": ssp_dtype,
        "n_free": len(list(model.spec.free_params)),
        "env": environment(),
        **CELLS[args.shape](model, args),
    }
    row["env_after"] = {"x64": bool(jax.config.jax_enable_x64), **gpu_snapshot()}

    if args.emit_json:
        print("__ROW__" + json.dumps(row))
    else:
        print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
