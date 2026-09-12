"""Library path vs harness: is ``forward.fit()`` as fast as the recipe it ships?

``bench/reports/2026-09-11_profile_mass_20s.md`` Finding 9 measured the
default ``forward.fit(data)`` (``mcmc_nuts_fast``) at 28 s on ``ctl-dpl`` seed 7
against the harness's 15.6 s for the same recipe, on a *fresh model per call*
-- a number that includes tracing and compilation the harness excludes. This
script separates the two questions the gap raises:

1. **Per-gradient parity.** Do the two paths pay the same per gradient?
   ``grad`` times one jitted gradient of each objective (library profiled loss
   vs the harness's own profiled log-density) and reads the compiled FLOPs;
   ``warmup`` and ``scan`` run the two adaptation and sampling implementations
   head-to-head from the same start, key, step size and mass matrix and report
   wall per gradient.
2. **Fixed per-fit overhead.** ``phases`` wraps every phase of one warm
   ``forward.fit`` call (MAP init, warmup, dense probe, sampling, finalization)
   and itemizes the wall; ``row`` runs one library fit per process on one seed
   and appends a JSONL row with cold/warm walls and gradient counts, the
   library-side twin of ``benchmark_laplace_nuts_20s.py``'s harness rows.

Every wall here is a *warm* wall (second call in the process) unless labelled
cold. ``row`` also records the cold wall, which is what a notebook sees, and
which the persistent JAX compile cache halves between the first and second
process on a box.

Usage::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        python bench/scripts/benchmark_library_path_parity.py grad|warmup|scan|phases [--seed 7]
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        python bench/scripts/benchmark_library_path_parity.py row --seed 7 --json out.jsonl

``XLA_FLAGS`` is set explicitly rather than via ``TENGRI_HOST_DEVICES`` because
this script imports ``benchmark_laplace_nuts_20s`` for its fixture and
profiled objective, and that module imports ``jax`` (and reads its own device
flag from *its* argv) before ``tengri``'s hook can run.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import benchmark_laplace_nuts_20s as H
import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri import Data
from tengri.inference._model_cache import _default_owner as _model_cache_owner
from tengri.inference.backends.mcmc import _shared as S
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter

N_CHAINS = 4
N_WARMUP = 150
N_SAMPLES = 300
TARGET_ACCEPT = 0.8
MAX_DOUBLINGS = 10


def _bench(fn, *args, n=3, warm=True):
    """Warm wall of ``fn(*args)`` in seconds, best of ``n``; the first call compiles."""
    if warm:
        jax.block_until_ready(fn(*args))
    walls = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        walls.append(time.perf_counter() - t0)
    return min(walls), out


def _problem(seed):
    """Harness context (unprofiled) and the library's profiled fitter, one model."""
    _, forward, ctx, flux, noise, _truth, key_fit = H._build_problem("ctl-dpl", seed)
    lib = Fitter(forward, data=flux, noise=noise)  # profile_mass="auto" -> engaged
    return forward, ctx, lib, flux, noise, key_fit


def cmd_grad(args):
    """One gradient of each objective: wall (ms) and compiled FLOPs."""
    _, ctx, lib, flux, noise, key_fit = _problem(args.seed)
    nlp, dargs = ctx.neg_log_posterior_fn, ctx.data_args
    pu = {
        k: jnp.asarray(v, dtype=jnp.float64).reshape(())
        for k, v in ctx.initial_params(key_fit, init_from=None).items()
    }
    flat0, unravel = ravel_pytree(pu)
    mass_name = H._mass_free_name(ctx.free_names)
    mass_idx, non_mass_idx = H._mass_index(pu, mass_name)
    prof = H._build_profiled_objects(ctx, unravel, mass_name, mass_idx, non_mass_idx, flux, noise)
    h_grad = jax.jit(jax.value_and_grad(prof["profiled_logdensity"]))
    x_h = flat0[non_mass_idx]

    lctx = InferenceContext.from_target(lib)
    l_nlp, l_dargs = lctx.neg_log_posterior_fn, lctx.data_args
    lp = {
        k: jnp.asarray(v, dtype=jnp.float64).reshape(())
        for k, v in lctx.initial_params(key_fit, init_from=None).items()
    }
    l_flat, l_unravel = ravel_pytree(lp)
    l_grad = jax.jit(lambda x, d: jax.value_and_grad(lambda xx: -l_nlp(l_unravel(xx), d))(x))
    u_grad = jax.jit(lambda x, d: jax.value_and_grad(lambda xx: -nlp(unravel(xx), d))(x))

    def per_call_ms(f, *a, n=2000):
        jax.block_until_ready(f(*a))
        t0 = time.perf_counter()
        for _ in range(n):
            out = f(*a)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / n * 1e3

    def flops(f, *a):
        c = f.lower(*a).compile().cost_analysis()
        c = c[0] if isinstance(c, list) else c
        return int(c.get("flops", 0)), int(c.get("bytes accessed", 0))

    print(f"D: harness profiled {x_h.size}, library profiled {l_flat.size}, full {flat0.size}")
    for rep in range(3):
        print(
            f"rep{rep}: harness profiled {per_call_ms(h_grad, x_h):.3f} ms | "
            f"library profiled {per_call_ms(l_grad, l_flat, l_dargs):.3f} ms | "
            f"library unprofiled full-D {per_call_ms(u_grad, flat0, dargs):.3f} ms"
        )
    print("flops/bytes: harness", flops(h_grad, x_h), "library", flops(l_grad, l_flat, l_dargs))


def _library_flat(lib, key_fit):
    ctx = InferenceContext.from_target(lib)
    init = ctx.initial_params(key_fit, init_from=None)
    return S._get_flat_logdensity(lib, init)


def cmd_warmup(args):
    """Window adaptation head-to-head from the same start and key (+ the dense probe)."""
    _, _, lib, _, _, key_fit = _problem(args.seed)
    ld2, _, init_flat, dargs = _library_flat(lib, key_fit)

    def ld1(x):
        return ld2(x, dargs)

    k_w, _ = jax.random.split(key_fit)

    def lib_warmup():
        return S._nuts_warmup_only(
            init_flat, k_w, ld2, dargs, N_WARMUP, True, TARGET_ACCEPT, False, MAX_DOUBLINGS
        )

    def lib_probe(step, imm):
        state = blackjax.nuts.init(init_flat, ld1)
        return S._stabilize_dense_mass_step(
            S._get_nuts_kernel(), state, ld2, dargs, float(step), imm, MAX_DOUBLINGS
        )

    har_adapt = H._window_adapt_fn(
        ld1, init_flat, TARGET_ACCEPT, N_WARMUP, diagonal=False, max_doublings=MAX_DOUBLINGS
    )
    out = lib_warmup()
    lib_probe(out[0], out[1])
    har_adapt(k_w)
    for rep in range(3):
        t0 = time.perf_counter()
        o = lib_warmup()
        jax.block_until_ready(o[0])
        w_lib = time.perf_counter() - t0
        t0 = time.perf_counter()
        lib_probe(o[0], o[1])
        w_probe = time.perf_counter() - t0
        t0 = time.perf_counter()
        h = har_adapt(k_w)
        jax.block_until_ready(h)
        w_har = time.perf_counter() - t0
        print(
            f"rep{rep}: library warmup {w_lib:.2f}s ({int(o[3])} grads, "
            f"{1e3 * w_lib / int(o[3]):.3f} ms/grad) + dense probe {w_probe:.2f}s | "
            f"harness warmup {w_har:.2f}s ({int(h[2])} grads, "
            f"{1e3 * w_har / int(h[2]):.3f} ms/grad) | step size library {float(o[0]):.4f} "
            f"harness {float(h[0]):.4f}"
        )


def cmd_scan(args):
    """Sampling scan head-to-head: library ``_vmap_chains`` (pmap) vs the harness's pmap."""
    _, _, lib, _, _, key_fit = _problem(args.seed)
    ld2, _, init_flat, dargs = _library_flat(lib, key_fit)

    def ld1(x):
        return ld2(x, dargs)

    k_w, k_c = jax.random.split(key_fit)
    step, imm, _, _ = S._nuts_warmup_only(
        init_flat, k_w, ld2, dargs, N_WARMUP, True, TARGET_ACCEPT, False, MAX_DOUBLINGS
    )
    keys = jax.random.split(k_c, N_CHAINS + 2)
    starts = init_flat[None, :] + 1e-3 * jax.random.normal(keys[1], (N_CHAINS, init_flat.size))
    chain_keys = jax.random.split(keys[0], N_CHAINS * N_SAMPLES).reshape(N_CHAINS, N_SAMPLES, 2)

    def _init(p):
        return blackjax.mcmc.nuts.init(p, ld1)

    def _scan(s, ks):
        return S._nuts_chain_scan(s, ks, ld2, dargs, step, imm, MAX_DOUBLINGS)

    def lib_run():
        return S._vmap_chains(
            _init,
            _scan,
            init_flat=init_flat,
            chain_key=keys[0],
            n_chains=N_CHAINS,
            n_iter=N_SAMPLES,
            n_burnin=0,
            chain_parallel="pmap",
        )

    kernel = blackjax.nuts.build_kernel()

    def run_chain(key_block, start):
        s0 = blackjax.nuts.init(start, ld1)

        def stp(s, k):
            s2, info = kernel(k, s, ld1, step, imm)
            return s2, (s2.position, info.is_divergent, info.num_integration_steps)

        return jax.lax.scan(stp, s0, key_block)[1]

    har_run = jax.pmap(run_chain, devices=jax.devices()[:N_CHAINS])
    lib_run()
    har_run(chain_keys, starts)
    for rep in range(3):
        w_lib, o = _bench(lib_run, n=1, warm=False)
        w_har, h = _bench(har_run, chain_keys, starts, n=1, warm=False)
        g_lib, g_har = int(jnp.sum(o[3])), int(jnp.sum(h[2]))
        print(
            f"rep{rep}: library scan {w_lib:.2f}s ({g_lib} grads, "
            f"{1e3 * w_lib / (g_lib / N_CHAINS):.3f} ms per chain-grad) | "
            f"harness scan {w_har:.2f}s ({g_har} grads, {1e3 * w_har / (g_har / N_CHAINS):.3f} ms)"
        )


def _evict(model):
    mc = _model_cache_owner.get_or_compile_model(model)
    for k in ("map_params_physical", "map_data_fingerprint", "adaptation"):
        mc.pop(k, None)


def cmd_phases(args):
    """Itemize one warm ``forward.fit`` (caches evicted) by wrapping its phases."""
    from tengri.inference import fitter as F, mass_profile as MP
    from tengri.inference.backends import map_dispatch as MD
    from tengri.inference.backends.mcmc import nuts as N

    acc: dict[str, float] = {}

    def wrap(mod, name, label):
        orig = getattr(mod, name)

        @functools.wraps(orig)
        def wrapped(*a, **k):
            t0 = time.perf_counter()
            out = orig(*a, **k)
            with contextlib.suppress(Exception):  # not every phase returns arrays
                jax.block_until_ready(out)
            acc[label] = acc.get(label, 0.0) + time.perf_counter() - t0
            return out

        setattr(mod, name, wrapped)

    wrap(F.Fitter, "__init__", "Fitter.__init__")
    wrap(F.Fitter, "_auto_prewarm", "Fitter._auto_prewarm")
    wrap(N, "_maybe_map_init", "MAP init (8 scipy L-BFGS-B restarts)")
    wrap(MD, "_run_map_multistart_scipy", "  of which the multistart itself")
    wrap(N, "_nuts_warmup_only", "warmup (window adaptation)")
    wrap(N, "_stabilize_dense_mass_step", "dense-mass step probe (#1999)")
    wrap(N, "_vmap_chains", "sampling (pmap, 4 chains)")
    wrap(N, "_vmap_samples_to_physical", "samples -> physical")
    wrap(N, "sampling_diagnostics", "sampling diagnostics")
    wrap(MP, "finalize_profile_mass", "finalize (mass reinsertion)")

    forward, _, lib, flux, noise, key_fit = _problem(args.seed)
    data = Data(photometry=(flux, noise))
    forward.fit(data, key=key_fit, verbose=False)  # cold: compile everything
    for rep in range(2):
        _evict(lib.model)
        acc.clear()
        t0 = time.perf_counter()
        post = forward.fit(data, key=key_fit, verbose=False)
        jax.block_until_ready(post.samples[next(iter(post.samples))])
        wall = time.perf_counter() - t0
        print(
            f"\n[warm rep{rep}] fit wall {wall:.2f}s (run_nuts wall_time_s {post.wall_time_s:.2f})"
        )
        itemized = 0.0
        for label, seconds in acc.items():
            print(f"    {seconds:6.2f}s  {label}")
            if not label.startswith("  "):
                itemized += seconds
        print(f"    {itemized:6.2f}s  itemized  ->  unitemized {wall - itemized:.2f}s")


def cmd_row(args):
    """One library fit per process: cold + warm walls and gradient counts as a JSONL row."""
    forward, _, lib, flux, noise, key_fit = _problem(args.seed)
    data = Data(photometry=(flux, noise))

    def fit():
        t0 = time.perf_counter()
        post = forward.fit(data, key=key_fit, verbose=False)
        jax.block_until_ready(post.samples[next(iter(post.samples))])
        return post, time.perf_counter() - t0

    _, cold = fit()
    _evict(lib.model)
    post, warm = fit()
    d = post.diagnostics
    rhat = post.rhat()
    worst = max(rhat, key=lambda k: float(rhat[k]))
    row = dict(
        path="library",
        notebook="ctl-dpl",
        seed=args.seed,
        wall_cold=cold,
        wall_warm=warm,
        run_nuts_wall=post.wall_time_s,
        n_grad_adapt=d["n_grad_adapt"],
        n_grad_sample=d["n_grad_sample"],
        n_grad_total=d["n_grad_total"],
        step_size=d["step_size"],
        tree_depth_mean=d["tree_depth_mean"],
        n_divergent=d["n_divergent"],
        rhat_max=float(rhat[worst]),
        worst_param=worst,
        profile_mass=d["profile_mass_resolved"],
        chain_parallel=d["chain_parallel"],
        n_devices=jax.device_count(),
        taskset=len(os.sched_getaffinity(0)),
        git_sha=H._git_sha(),
    )
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(H._to_native(row)) + "\n")
    print(json.dumps(H._to_native(row)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["grad", "warmup", "scan", "phases", "row"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None, help="row: JSONL file to append to")
    args = ap.parse_args()
    {
        "grad": cmd_grad,
        "warmup": cmd_warmup,
        "scan": cmd_scan,
        "phases": cmd_phases,
        "row": cmd_row,
    }[args.command](args)


if __name__ == "__main__":
    main()
