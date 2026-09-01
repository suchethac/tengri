# SPDX-License-Identifier: BSD-3-Clause
"""Does a float32 catalog fit land on the same posterior as a float64 one?

A gradient that tracks float64 to 1e-4 is a statement about one point. A
posterior is a statement about a distribution, and the honest test is to put the
two marginals side by side on the *same* data and the *same* seed, with the
difference expressed as a z-score on the combined Monte Carlo standard error --
the methodology of ``compare_mclmc_nuts_posteriors.py``, applied across a
precision axis instead of a sampler axis.

Precision is process-global (``JAX_ENABLE_X64`` is latched by ``import jax``, and
``import tengri`` re-enables x64 unless it is already set -- #1840), so the two
arms **must** be separate processes. This script therefore has two modes:

``--dtype f32|f64 --out FILE``
    run one arm and write per-galaxy summaries;
``--compare A.json B.json``
    read two arms and report the z-scores, sd ratios, R-hat and min ESS.

Peak device memory is recorded per arm, because the second question this answers
is what float32 is worth on the fitting path: not a faster clock (PR #2097
measured tengri's forward model at ~0.12 FLOP/byte, memory- and dispatch-bound)
but halved memory traffic, i.e. **galaxies per GB**.

Usage
-----
::

    python bench/scripts/compare_float32_catalog_posteriors.py --dtype f64 \
        --n-gal 64 --warmup 400 --samples 500 --out bench/results/f64.json
    python bench/scripts/compare_float32_catalog_posteriors.py --dtype f32 \
        --n-gal 64 --warmup 400 --samples 500 --out bench/results/f32.json
    python bench/scripts/compare_float32_catalog_posteriors.py \
        --compare bench/results/f64.json bench/results/f32.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported for its module-scope precision latch: it reads --dtype off *this*
# process's argv and puts JAX_ENABLE_X64 (and JAX_DEFAULT_MATMUL_PRECISION on
# float32) into the environment BEFORE jax is imported. Also the source of the
# shared fixture, so this comparison is on the same catalog as the throughput
# report.
import benchmark_catalog_throughput as B
import jax
import jax.numpy as jnp
import numpy as np


def _summarize(cp, max_gal):
    """Per-galaxy, per-parameter mean / sd / ESS / split-R-hat.

    ESS rides beside R-hat everywhere in this repository for a reason: split-R-hat
    reads 1.00 over two equally badly-mixed halves, and this project has measured
    ESS ~2 of 200-500 draws behind a passing R-hat repeatedly. A z-score built on
    ``sd / sqrt(ESS)`` is the only one that stays honest when that happens -- it
    widens the error bar instead of hiding behind it.
    """
    from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

    n = min(len(cp.posteriors), max_gal)
    names = sorted(cp.posteriors[0].samples)
    mean = np.full((n, len(names)), np.nan)
    sd = np.full((n, len(names)), np.nan)
    ess = np.full((n, len(names)), np.nan)
    rhat = np.full((n, len(names)), np.nan)
    frozen = np.zeros(n, dtype=bool)
    ndiv = np.zeros(n, dtype=int)
    for i in range(n):
        post = cp.posteriors[i]
        ndiv[i] = int((post.diagnostics or {}).get("n_divergent", 0) or 0)
        s = {k: np.asarray(v, dtype=np.float64) for k, v in post.samples.items()}
        e = effective_sample_size(s)
        if i == 0 and set(e) != set(names):
            print(f"  note: ESS keys {sorted(e)} != sample keys {names}")
        try:
            rh = post.rhat()
        except ValueError:
            frozen[i] = True
            rh = {}
        for j, k in enumerate(names):
            mean[i, j] = float(np.mean(s[k]))
            sd[i, j] = float(np.std(s[k], ddof=1))
            rec = e.get(k)
            val = rec.get("ess") if isinstance(rec, dict) else rec
            # NaN rather than a crash: a frozen chain has no autocorrelation time, so
            # effective_sample_size legitimately returns None for it, and a galaxy whose
            # ESS is undefined must drop out of the z-score rather than take the run down.
            ess[i, j] = float(val) if val is not None else np.nan
            if k in rh:
                rhat[i, j] = float(rh[k])
    return {
        "names": names,
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "ess": ess.tolist(),
        "rhat": rhat.tolist(),
        "frozen": frozen.tolist(),
        "n_divergent": ndiv.tolist(),
    }


def _shared_catalog(B, model, args, key):
    """The **same** mock for both arms, generated once in float64 and cached on disk.

    Not optional. ``make_catalog`` draws its noise with ``jax.random.normal`` in the
    process's precision, and float32 and float64 PRNG draws are *different numbers*, not
    rounded versions of each other — the first attempt at this comparison produced
    catalogs with median SNR 19.8 and 20.0, i.e. two different datasets, and any z-score
    between them would have measured the noise realization rather than the precision.
    The float64 arm writes the file; the float32 arm loads it and casts.
    """
    path = args.catalog
    if path and os.path.exists(path):
        z = np.load(path)
        n = int(z["n_gal"])
        if n < args.n_gal:
            raise SystemExit(f"{path} holds {n} galaxies, fewer than the {args.n_gal} asked for")
        print(f"  catalog: loaded {args.n_gal} of {n} galaxies from {path}")
        return [
            {"flux_obs": jnp.asarray(z["flux_obs"][i]), "noise": jnp.asarray(z["noise"][i])}
            for i in range(args.n_gal)
        ]
    catalog = B.make_catalog(model, args.n_gal, key, noise_frac=args.noise_frac)
    if path:
        np.savez(
            path,
            n_gal=args.n_gal,
            flux_obs=np.stack([np.asarray(g["flux_obs"], dtype=np.float64) for g in catalog]),
            noise=np.stack([np.asarray(g["noise"], dtype=np.float64) for g in catalog]),
        )
        print(f"  catalog: generated in {args.dtype} and wrote {path}")
    return catalog


def run_arm(args):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Explicitly requested dtype float64.*")
        warnings.filterwarnings("ignore", message=".*JAX_ENABLE_X64=.*")
        from tengri.inference.catalog_fitter import CatalogFitter

        flags = B.set_precision(args.dtype)
        ssp, ssp_tag = B._load_or_synth_ssp()
        model = B.build_model(ssp, ssp_tag)

    key = jax.random.PRNGKey(args.seed)
    catalog = _shared_catalog(B, model, args, key)
    snr = B.catalog_snr(catalog)
    cat = CatalogFitter(model, catalog, data_type="photometry")

    dev = jax.devices()
    print(
        f"  backend {dev[0].platform} | dtype {args.dtype} | probe {flags['probe_dtype']} "
        f"| x64 {jax.config.x64_enabled} | SNR median {snr['snr_median']:.1f}"
    )

    K = args.chunk or args.n_gal
    run_kw = dict(n_warmup=args.warmup, n_burnin=args.burnin, n_samples=args.samples)
    if args.n_leapfrog is not None:
        run_kw["n_leapfrog_steps"] = args.n_leapfrog

    before = B.device_peak_bytes()
    t0 = time.perf_counter()
    cp = cat.run(args.method, key=key, forward_chunk_size=K, verbose=False, **run_kw)
    jax.block_until_ready(cp[0].samples)
    wall = time.perf_counter() - t0
    peak = B.device_peak_bytes()

    out = {
        "dtype": args.dtype,
        "dtype_flags": flags,
        "platform": dev[0].platform,
        "device": str(dev[0]),
        "method": args.method,
        "n_gal": args.n_gal,
        "chunk": K,
        "seed": args.seed,
        "warm_s": round(wall, 3),
        "peak_bytes": peak,
        "peak_bytes_before": before,
        "ssp": ssp_tag,
        "n_warmup": args.warmup,
        "n_burnin": args.burnin,
        "n_samples": args.samples,
        "n_leapfrog": args.n_leapfrog,
        **snr,
        **B.approx_tag(cat),
        **B._diagnostics(cp, args.n_gal),
        "summary": _summarize(cp, args.n_gal),
    }
    peak_gib = None if peak is None else peak / 2**30
    print(
        f"  wall {wall:.1f}s | peak {'n/a' if peak_gib is None else f'{peak_gib:.3f} GiB'} "
        f"| maxRhat {out['max_rhat']} | minESS {out['min_ess']} | frozen "
        f"{out['n_frozen_chains']} | converged {out['n_gal_converged']}/{args.n_gal}"
    )
    if peak_gib:
        print(f"  galaxies per GiB of device peak: {args.n_gal / peak_gib:.1f}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {args.out}")
    return 0


def compare(path_a, path_b):
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    if a["summary"]["names"] != b["summary"]["names"]:
        raise SystemExit("the two arms have different parameters; not comparable")
    names = a["summary"]["names"]
    get = lambda d, k: np.asarray(d["summary"][k], dtype=np.float64)  # noqa: E731

    ma, mb = get(a, "mean"), get(b, "mean")
    sa, sb = get(a, "sd"), get(b, "sd")
    ea, eb = get(a, "ess"), get(b, "ess")
    # MCSE from each arm's OWN ESS, so an arm that mixed badly gets a wider bar
    # rather than a spurious detection.
    mcse = np.sqrt(sa**2 / np.maximum(ea, 1e-12) + sb**2 / np.maximum(eb, 1e-12))
    z = (mb - ma) / np.maximum(mcse, 1e-300)

    for tag, d in ((os.path.basename(path_a), a), (os.path.basename(path_b), b)):
        peak = d.get("peak_bytes")
        gib = None if peak is None else peak / 2**30
        peak_s = "n/a" if gib is None else f"{gib:.3f} GiB"
        per_gib = "n/a" if not gib else f"{d['n_gal'] / gib:.1f}"
        print(
            f"{tag:38s} dtype={d['dtype']} probe={d['dtype_flags']['probe_dtype']} "
            f"platform={d['platform']} wall={d['warm_s']}s peak={peak_s} "
            f"gal/GiB={per_gib} maxRhat={d['max_rhat']} minESS={d['min_ess']} "
            f"frozen={d['n_frozen_chains']}"
        )
    # ``Posterior.samples`` carries every declared parameter, Fixed ones included, while
    # ``effective_sample_size`` returns only the sampled ones — so a Fixed parameter has
    # sd 0 and no ESS, and its z-score is 0/0. Those are dropped rather than printed as
    # NaN: a comparison over a parameter that never moved is not a comparison.
    free = [j for j in range(len(names)) if np.any(np.isfinite(z[:, j]))]
    fixed = [names[j] for j in range(len(names)) if j not in free]
    print(f"\n  n_gal={a['n_gal']}  sampled parameters={[names[j] for j in free]}")
    if fixed:
        print(f"  not sampled (no ESS, sd 0), excluded: {fixed}")
    print(f"  {'parameter':<32} {'max|z|':>8} {'med|z|':>8} {'frac|z|>2':>10} {'sd ratio':>10}")
    for j in free:
        zj = np.abs(z[:, j])
        zj = zj[np.isfinite(zj)]
        ratio = np.nanmedian(sb[:, j] / np.maximum(sa[:, j], 1e-300))
        print(
            f"  {names[j]:<32} {zj.max():>8.2f} {np.median(zj):>8.2f} "
            f"{float(np.mean(zj > 2)):>10.3f} {ratio:>10.4f}"
        )
    zf = np.abs(z[:, free])[np.isfinite(z[:, free])]
    print(
        f"\n  overall max|z| {zf.max():.2f} over {zf.size} galaxy-parameter pairs; "
        f"{float(np.mean(zf > 2)) * 100:.1f} % beyond 2 MC sigma "
        f"(expected ~4.6 % if the two agree)"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="f64", choices=list(B.DTYPES))
    ap.add_argument("--method", default="mcmc_hmc", choices=list(B.METHODS))
    ap.add_argument("--n-gal", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=None, help="forward_chunk_size, default n_gal")
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--burnin", type=int, default=0)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--n-leapfrog", type=int, default=None)
    ap.add_argument("--noise-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--catalog",
        default=None,
        help=(
            "npz path for the shared mock. Written by the first arm to run, loaded by "
            "the second. REQUIRED for a valid comparison — see _shared_catalog."
        ),
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        ap.error("--out is required unless --compare is given")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
