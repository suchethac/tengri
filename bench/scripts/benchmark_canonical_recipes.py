#!/usr/bin/env python
"""Canonical recipe perf sweep: forward + grad + MAP + (optional) HMC/NUTS.

For each recipe in ``tengri.recipes``, this benchmark builds the model two
ways — ``approx=None`` (exact wave-grid) and ``approx=WavePrecomp()`` (LUT) —
and reports:

* compile wall (first call vs cached)
* steady-state forward eval (microseconds, n_runs averaged)
* gradient eval (microseconds)
* MAP convergence wall + final logpost
* (with --mcmc) one HMC and one NUTS run, warmup + sampling broken out

Also samples RSS at every stage to surface tracer-OOM / recompile patterns.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_canonical_recipes.py
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_canonical_recipes.py --mcmc
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_canonical_recipes.py --recipe star_forming_photometry
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
import traceback
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import tengri
from tengri import Observation, Photometry, SEDModel, WavePrecomp, recipes
from tengri.sps.dsps_wrapper import load_ssp_data

N_WARMUP = 3
N_RUNS = 100


def rss_mb() -> float:
    """Process RSS in MiB (macOS reports bytes, Linux reports KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def bench_call(fn, n_warmup=N_WARMUP, n_runs=N_RUNS) -> tuple[float, float]:
    """Return (compile_ms, steady_us). compile_ms is the FIRST call's wall."""
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(n_warmup - 1):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n_runs):
        jax.block_until_ready(fn())
    steady_us = (time.perf_counter() - t0) / n_runs * 1e6
    return compile_ms, steady_us


SSP_BARE = "data/ssp_prsc_miles_chabrier.h5"
SSP_WNE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
SSP_MIST_WNE = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


def pick_ssp_for(recipe_name: str):
    """Each recipe declares SSP requirement in its docstring."""
    if recipe_name in {"mock_recovery_minimal", "dust_demo"}:
        # any SSP works; use wNE since bare-stellar may be missing
        for p in (SSP_WNE, SSP_MIST_WNE):
            if os.path.exists(p):
                return p, "wNE"
    # rest need bare-stellar — fall back to wNE if missing so we can still bench forward eval
    if os.path.exists(SSP_BARE):
        return SSP_BARE, "bare"
    for p in (SSP_WNE, SSP_MIST_WNE):
        if os.path.exists(p):
            return p, "wNE"
    raise FileNotFoundError("No SSP data file found in data/")


DEFAULT_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j", "2mass_h", "2mass_ks"]
SDSS5 = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


def standard_obs(bands=None):
    return Observation(photometry=Photometry.from_names(bands or DEFAULT_BANDS))


def report(label, **kv):
    line = f"  {label:<48} "
    for k, v in kv.items():
        if isinstance(v, float):
            if abs(v) >= 1000:
                line += f"{k}={v:>9.0f}  "
            else:
                line += f"{k}={v:>9.3f}  "
        else:
            line += f"{k}={v}  "
    print(line)


def bench_recipe(name: str, with_precomp: bool, ssp_data, mcmc: bool, ssp_kind: str = "bare", bands=None):
    rss0 = rss_mb()
    recipe_dict = getattr(recipes, name)()
    obs = standard_obs(bands=bands)
    kw = dict(recipe_dict)
    # If we only have a wNE SSP but the recipe asks for Cue, swap to none so we can
    # still benchmark the rest of the physics stack. Mark the label so the swap
    # is visible in the output.
    swapped = False
    if ssp_kind == "wNE":
        import tengri.builders as builders
        neb = kw.get("neb")
        if isinstance(neb, dict) and neb.get("type") == "cue":
            kw["neb"] = builders.neb.none()
            swapped = True
    if with_precomp:
        kw["approx"] = WavePrecomp()
    suffix = "PRECOMP" if with_precomp else "exact"
    if swapped:
        suffix += "/neb=none"
    label = f"{name} [{suffix}]"

    t_build0 = time.perf_counter()
    try:
        model = SEDModel.build(ssp_data=ssp_data, observation=obs, **kw)
    except Exception as exc:
        print(f"  {label:<48} BUILD FAIL: {type(exc).__name__}: {exc!s:.100s}")
        return
    build_ms = (time.perf_counter() - t_build0) * 1e3

    params = model.spec.sample(jax.random.PRNGKey(0))
    n_free = len(model.spec.free_params)

    # Forward eval timing
    try:
        compile_ms, steady_us = bench_call(lambda: model.predict_photometry(params))
    except Exception as exc:
        print(f"  {label:<48} FWD FAIL: {type(exc).__name__}: {exc!s:.100s}")
        traceback.print_exc()
        return

    # Gradient timing  (loss = sum of predicted fluxes; depends on all free params)
    def loss(p):
        return jnp.sum(model.predict_photometry(p))

    grad_fn = jax.jit(jax.grad(loss))
    try:
        gcompile_ms, gsteady_us = bench_call(lambda: grad_fn(params))
    except Exception as exc:
        gcompile_ms, gsteady_us = -1.0, -1.0
        print(f"    [grad failed: {type(exc).__name__}: {exc!s:.100s}]")

    rss1 = rss_mb()

    report(
        label,
        D=n_free,
        build_ms=build_ms,
        fwd_compile_ms=compile_ms,
        fwd_us=steady_us,
        grad_compile_ms=gcompile_ms,
        grad_us=gsteady_us,
        rss_mb=rss1,
        d_rss=rss1 - rss0,
    )

    if not mcmc:
        del model, params, grad_fn
        gc.collect()
        return

    # MAP + HMC + NUTS via the standard fitter path
    try:
        from tengri.inference import Fitter

        # Synthetic data from the model
        truth_fluxes = model.predict_photometry(params)
        noise = 0.05 * jnp.abs(truth_fluxes) + 1e-12
        key = jax.random.PRNGKey(7)
        data = truth_fluxes + noise * jax.random.normal(key, truth_fluxes.shape)
        jax.block_until_ready(data)

        fitter = Fitter(model, data=data, noise=noise)

        t0 = time.perf_counter()
        post_map = fitter.run("map", n_steps=300, key=jax.random.PRNGKey(11))
        # MAP returns a Posterior; access ANY leaf and block on it
        try:
            leaves = jax.tree.leaves(post_map)
            if leaves:
                jax.block_until_ready(leaves[0])
        except Exception:
            pass
        map_ms = (time.perf_counter() - t0) * 1e3
        print(f"    MAP                                            time_ms={map_ms:>9.1f}")
    except Exception as exc:
        print(f"    [MAP failed: {type(exc).__name__}: {exc!s:.180s}]")
        return

    try:
        t0 = time.perf_counter()
        post_hmc = fitter.run(
            "mcmc_hmc",
            n_warmup=200,
            n_samples=200,
            key=jax.random.PRNGKey(13),
        )
        try:
            leaves = jax.tree.leaves(post_hmc)
            if leaves:
                jax.block_until_ready(leaves[0])
        except Exception:
            pass
        hmc_ms = (time.perf_counter() - t0) * 1e3
        print(f"    HMC (200+200, 1 chain)                         time_ms={hmc_ms:>9.1f}")
    except Exception as exc:
        print(f"    [HMC failed: {type(exc).__name__}: {exc!s:.180s}]")

    try:
        t0 = time.perf_counter()
        post_nuts = fitter.run(
            "mcmc_nuts",
            n_warmup=200,
            n_samples=200,
            key=jax.random.PRNGKey(17),
        )
        try:
            leaves = jax.tree.leaves(post_nuts)
            if leaves:
                jax.block_until_ready(leaves[0])
        except Exception:
            pass
        nuts_ms = (time.perf_counter() - t0) * 1e3
        print(f"    NUTS (200+200, 1 chain)                        time_ms={nuts_ms:>9.1f}")
    except Exception as exc:
        print(f"    [NUTS failed: {type(exc).__name__}: {exc!s:.180s}]")

    del model, params, grad_fn
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", action="store_true", help="also run MAP/HMC/NUTS per recipe")
    ap.add_argument("--recipe", default=None, help="run only one named recipe")
    ap.add_argument("--no-precomp", action="store_true", help="skip WavePrecomp arm")
    ap.add_argument("--no-exact", action="store_true", help="skip exact arm")
    args = ap.parse_args()

    print()
    print("tengri canonical recipe benchmark")
    print(f"  tengri version: {tengri.__version__}")
    print(f"  JAX backend: {jax.default_backend().upper()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  RSS at start: {rss_mb():.0f} MiB")
    print()

    all_recipes = [
        "mock_recovery_minimal",
        "star_forming_photometry",
        "quiescent_z0",
        "stochastic_sfh_jwst",
        # AGN deferred — high-D, may need raytrace; we'll handle separately
    ]
    if args.recipe:
        all_recipes = [args.recipe]

    # Two SSPs cover all recipes. Load once.
    print("Loading SSPs...")
    ssp_cache = {}
    for r in all_recipes:
        path, kind = pick_ssp_for(r)
        if path not in ssp_cache:
            t0 = time.perf_counter()
            ssp_cache[path] = (load_ssp_data(path), kind)
            print(f"  {path} ({kind}) loaded in {time.perf_counter() - t0:.1f}s")
    print()

    print("=" * 140)
    print(f"  {'Recipe [path]':<48} {'D':>3}  "
          f"{'build_ms':>10} {'fwd_cc_ms':>11} {'fwd_us':>10} "
          f"{'grad_cc_ms':>12} {'grad_us':>10} {'rss_MiB':>9} {'d_rss':>7}")
    print("=" * 140)

    for r in all_recipes:
        ssp_path, kind = pick_ssp_for(r)
        ssp, _ = ssp_cache[ssp_path]
        if not args.no_exact:
            bench_recipe(r, with_precomp=False, ssp_data=ssp, mcmc=args.mcmc, ssp_kind=kind)
        if not args.no_precomp:
            bench_recipe(r, with_precomp=True, ssp_data=ssp, mcmc=args.mcmc, ssp_kind=kind)
        gc.collect()

    # Probe 1: does filter count matter? Same recipe, 5 vs 8 bands, both precomp
    print()
    print("# Probe: filter count (mock_recovery_minimal, PRECOMP)")
    print("-" * 140)
    for nbands in (5, 8):
        bench_recipe(
            "mock_recovery_minimal",
            with_precomp=True,
            ssp_data=list(ssp_cache.values())[0][0],
            mcmc=False,
            ssp_kind=list(ssp_cache.values())[0][1],
            bands=SDSS5 if nbands == 5 else DEFAULT_BANDS,
        )

    print("=" * 140)
    print(f"  Final RSS: {rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
