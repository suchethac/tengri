#!/usr/bin/env python
"""Fit one slice of a photometric catalog with ``CatalogFitter`` — a SLURM worker.

Reads a catalog of observed fluxes + noise, selects the ``--slice`` of
``--n-slices`` slices, builds the shared model from a user-supplied builder, runs
vectorized per-galaxy MCMC (NUTS by default — ``forward_chunk_size`` galaxies
sample in parallel per step), and writes a per-galaxy posterior-summary shard.
Launched by ``fit_catalog.sbatch`` (one GPU, whole catalog) or
``fit_catalog_array.sbatch`` (one slice per GPU, scaled across nodes).

Catalog file (``--catalog``): an ``.npz`` with
``flux_obs`` shape ``(N, n_band)`` and ``noise`` shape ``(N, n_band)``
(per-band 1-sigma errors), both [erg/s/cm2/Hz] or any consistent flux unit.

Model builder (``--model-builder``): ``"package.module:function"`` — an importable
zero-argument callable returning a built :class:`~tengri.SEDModel` whose
observation matches the catalog's bands. Keep it in your project, e.g.::

    # myfit.py
    def build_model():
        from tengri import SEDModel, Observation, Photometry, Parameters, Uniform, Fixed
        from tengri.sps.dsps_wrapper import load_ssp_data
        ...
        return SEDModel(spec, ssp, observation=obs)

Then submit with ``MODEL_BUILDER=myfit:build_model``.

Usage (standalone, one slice on CPU)::

    python run_catalog_slice.py --catalog cat.npz --model-builder myfit:build_model \
        --out shards/ --slice 0 --n-slices 1 --chunk 32 --n-samples 200
"""

from __future__ import annotations

import argparse
import importlib
import math
import os

import numpy as np


def slice_bounds(n_total: int, index: int, n_slices: int) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` bounds for slice ``index`` of ``n_slices``.

    Even split with the remainder spread over the leading slices, so the union
    of all slices covers ``range(n_total)`` exactly with no overlap.
    """
    if not (0 <= index < n_slices):
        raise ValueError(f"slice index {index} out of range for {n_slices} slices")
    per = math.ceil(n_total / n_slices)
    lo = min(index * per, n_total)
    hi = min(lo + per, n_total)
    return lo, hi


def _load_builder(spec: str):
    if ":" not in spec:
        raise ValueError(f"--model-builder must be 'module:function', got {spec!r}")
    module_name, fn_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), fn_name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit one catalog slice with CatalogFitter.")
    ap.add_argument("--catalog", required=True, help="npz with flux_obs (N,B) and noise (N,B)")
    ap.add_argument(
        "--model-builder", required=True, help="'module:function' returning an SEDModel"
    )
    ap.add_argument("--out", required=True, help="output directory for the shard npz")
    ap.add_argument("--slice", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument(
        "--n-slices", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )
    ap.add_argument("--method", default="mcmc_nuts")
    ap.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="forward_chunk_size (galaxies fit in parallel per step)",
    )
    ap.add_argument("--n-warmup", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap galaxies in the slice (cache warmup / testing)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    from tengri.inference.catalog_fitter import CatalogFitter

    print(
        f"[slice {args.slice}/{args.n_slices}] backend={jax.devices()[0].platform} "
        f"devices={jax.device_count()} x64={jax.config.x64_enabled}",
        flush=True,
    )

    data = np.load(args.catalog)
    flux = np.asarray(data["flux_obs"])
    noise = np.asarray(data["noise"])
    n_total = flux.shape[0]
    lo, hi = slice_bounds(n_total, args.slice, args.n_slices)
    if args.limit is not None:
        hi = min(hi, lo + args.limit)
    galaxies = [{"flux_obs": flux[i], "noise": noise[i]} for i in range(lo, hi)]
    if not galaxies:
        print(f"[slice {args.slice}] empty (n_total={n_total}); nothing to do", flush=True)
        return 0
    print(
        f"[slice {args.slice}] fitting galaxies {lo}:{hi} ({len(galaxies)}) "
        f"method={args.method} chunk={args.chunk}",
        flush=True,
    )

    model = _load_builder(args.model_builder)()
    cat = CatalogFitter(model, galaxies, data_type="photometry")
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), args.slice)
    cp = cat.run(
        args.method,
        key=key,
        forward_chunk_size=args.chunk,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        verbose=True,
    )

    # Per-galaxy posterior summaries (mean + 16/50/84 percentiles), keyed by
    # the galaxy's global catalog index so shards merge back in order.
    os.makedirs(args.out, exist_ok=True)
    out = {"global_index": np.arange(lo, hi)}
    props = cp.properties
    for name in props:
        arr = np.asarray(props[name])
        if arr.ndim == 2:  # (n_gal, n_samples)
            out[f"{name}_mean"] = arr.mean(axis=1)
            out[f"{name}_p16"] = np.percentile(arr, 16, axis=1)
            out[f"{name}_p50"] = np.percentile(arr, 50, axis=1)
            out[f"{name}_p84"] = np.percentile(arr, 84, axis=1)
        elif arr.ndim == 1:  # point estimate per galaxy (e.g. MAP)
            out[f"{name}_mean"] = arr
    path = os.path.join(args.out, f"shard_{args.slice:05d}.npz")
    np.savez(path, **out)
    print(f"[slice {args.slice}] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
