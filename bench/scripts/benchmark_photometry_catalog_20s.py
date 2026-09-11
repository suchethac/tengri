#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Per-galaxy seconds on the batched catalog path, at the D = 8 photometry fixture.

``bench/scripts/benchmark_catalog_throughput.py`` already measures the batched
path, but on its own **D = 3** five-band fixture. That is the wrong model for the
question "can a photometry posterior be fitted in 20 s per galaxy": the whole
cost structure of a NUTS fit is the tree depth its geometry forces, and a
three-parameter posterior does not have the geometry of an eight-parameter one.
This script runs the same measurement on ``benchmark_notebook_sampler``'s
``ctl-dpl`` fixture -- nb05's 14 bands, mock, seed, SNR and dust over a DPL SFH,
D = 8 -- so the catalog number and the single-galaxy number are about the same
posterior.

THE TWO ADAPTATION CONVENTIONS ARE NOT THE SAME MEASUREMENT
===========================================================

They are separate arms here, never merged, because they answer different
questions and one of them is only valid conditionally:

``catalog``
    :meth:`CatalogFitter._run_native_mcmc` -- every galaxy runs its **own**
    window adaptation inside the ``lax.map``. Statistically the correct choice:
    a galaxy's posterior does not depend on which other galaxies were in the
    batch.
``fit_batch``
    :meth:`Fitter._fit_batch_vmap_mcmc` -- **one** adaptation, on the first
    galaxy, whose step size and mass matrix are then reused for every other
    galaxy (``fitter.py``). This is the convention that makes a 20 s/galaxy
    budget plausible, because warmup is 71.6 % of a zero-compile NUTS fit
    (``bench/reports/2026-08-31_fast_nuts.md`` Finding 2) and this arm pays it
    once for the whole batch instead of N times.

    It is **only valid if one galaxy's adapted metric actually serves the
    others**, which is a measurement, not an assumption. The ``solo`` arm below
    is the reference that decides it.
``solo``
    ``ForwardModel.fit`` on one galaxy at a time, same budget, same MAP seed
    policy. Two jobs: it is the ground truth each batched galaxy's R-hat and ESS
    are compared against, and it is the only place a **per-lane gradient count**
    can be read, because neither batched engine returns
    ``num_trajectory_expansions`` per galaxy (``catalog.py`` discards
    ``_expansions``; ``_fit_batch_vmap_mcmc`` never collects it).

WHY PER-LANE COST VARIANCE IS A COLUMN AND NOT A FOOTNOTE
=========================================================

A vmapped batch runs to its **slowest lane**. NUTS's per-step cost is a
data-dependent ``lax.while_loop`` tree doubling, so a batch of N galaxies costs
roughly ``N * max_i(grads_i)``, not ``sum_i(grads_i)``. The ratio of those two
is the lock-step tax, and it is invisible in a throughput number: a batch whose
mean per-galaxy gradient count is 60 and whose worst lane is 600 reports the
same galaxies/minute as a uniform batch ten times more expensive on average.
The ``solo`` arm's spread of ``grad_per_draw`` is what makes that ratio
measurable. ``bench/reports/2026-09-06_low_rank_metric_d74.md`` could not
measure it at all because ``mcmc_hmc_lowrank`` is not wired into the batched
engine; NUTS is.

Usage::

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \\
      .venv/bin/python bench/scripts/benchmark_photometry_catalog_20s.py \\
      --notebook ctl-dpl --n-gal 32 128 --arms catalog fit_batch \\
      --warmup 600 --samples 600 --json bench/results/2026-09-06_photometry_catalog.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import time
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import jax
import numpy as np

import tengri
from tengri.inference.catalog_convergence import catalog_convergence

_HERE = pathlib.Path(__file__).resolve().parent


def _load_registry():
    """Import ``benchmark_notebook_sampler``'s fixture registry, not a copy of it.

    ``diagnose_ghmc_meads.py`` already does this. A second spelling of the same
    model is the drift defect #2096 exists to prevent, and the ``ctl-dpl``
    fixture in particular is a *control*: it is only a control while it is
    byte-for-byte the same model the single-galaxy rows measured.
    """
    spec = importlib.util.spec_from_file_location(
        "bns", str(_HERE / "benchmark_notebook_sampler.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loadavg() -> float:
    """One-minute load average, stamped on every row.

    This box has shown a 9.5x wall-clock spread from scheduling alone -- one
    byte-identical fit measured 2834.9 s contended against 1541.6 s idle
    (``bench/reports/2026-08-31_fast_nuts.md``). A seconds column without the
    load it was measured under is not reproducible.
    """
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def build_catalog(cfg, model, n_gal, key):
    """``n_gal`` independent mock galaxies from the fixture's own prior and SNR.

    Each galaxy is its own prior draw, so the catalog spans the geometry the
    posterior actually has rather than N noise realizations of one truth. That
    is what makes the per-lane spread meaningful.
    """
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        truth = model.spec.sample(k)
        mock = tengri.generate_mock(model, truth, key=jax.random.fold_in(k, 1), snr=cfg["snr"])
        galaxies.append(
            {
                "flux_obs": np.asarray(mock["flux_obs"]),
                "noise": np.asarray(mock["noise"]),
            }
        )
    return galaxies


def make_per_galaxy(bns):
    """Bind a per-galaxy scorer -- R-hat, min ESS, divergences, unique fraction.

    All four, for every galaxy, because none of them is sufficient alone. This
    project has measured cells at R-hat 2.97 with **zero** divergences, and its
    two worst cells had zero divergences at unique-draw fractions 1.000 and
    0.982 (#1999: a completely frozen chain reports zero divergences and an
    R-hat near 1.0, because both halves of the split have zero variance).
    """

    def per_galaxy(posteriors):
        report = catalog_convergence(posteriors)
        rows = []
        for i, (post, rep) in enumerate(zip(posteriors, report.per_galaxy)):
            diag = post.diagnostics or {}
            rows.append(
                {
                    "galaxy": i,
                    "max_rhat": None if rep.max_rhat is None else float(rep.max_rhat),
                    "max_rhat_param": rep.max_rhat_param,
                    "min_ess": None if rep.min_ess is None else float(rep.min_ess),
                    "min_ess_param": rep.min_ess_param,
                    "divergences": int(diag.get("n_divergent") or 0),
                    "unique_frac": bns._unique_draw_fraction(post),
                    "grad_per_draw": bns._gradients_per_draw(diag),
                    "tree_depth_mean": diag.get("tree_depth_mean"),
                }
            )
        return rows, report

    return per_galaxy


def _summarize(rows, report, wall, n_gal, warmup, samples):
    """Throughput WITH the convergence columns attached, never throughput alone."""
    rhats = [r["max_rhat"] for r in rows if r["max_rhat"] is not None]
    esses = [r["min_ess"] for r in rows if r["min_ess"] is not None]
    uniq = [r["unique_frac"] for r in rows if r["unique_frac"] == r["unique_frac"]]
    gpd = [r["grad_per_draw"] for r in rows if r["grad_per_draw"] is not None]
    out = {
        "n_gal": n_gal,
        "wall_s": round(wall, 2),
        "s_per_galaxy": round(wall / max(n_gal, 1), 3),
        "gal_per_min": round(60.0 * n_gal / max(wall, 1e-9), 2),
        "n_converged": report.n_converged,
        "n_frozen": report.n_frozen,
        "n_unconverged": report.n_unconverged,
        "frac_converged": report.frac_converged,
        "divergence_rate": report.divergence_rate,
        "max_rhat": max(rhats) if rhats else None,
        "median_rhat": float(np.median(rhats)) if rhats else None,
        "min_ess": min(esses) if esses else None,
        "median_ess": float(np.median(esses)) if esses else None,
        "min_unique_frac": min(uniq) if uniq else None,
        "n_warmup": warmup,
        "n_samples": samples,
    }
    if gpd:
        # The lock-step tax. A vmapped batch runs to its slowest lane, so the
        # cost is ~N * max rather than the sum; this ratio is how much of the
        # batch's work is spent waiting for one galaxy.
        out.update(
            {
                "gpd_min": float(np.min(gpd)),
                "gpd_median": float(np.median(gpd)),
                "gpd_mean": float(np.mean(gpd)),
                "gpd_max": float(np.max(gpd)),
                "lockstep_tax": float(np.max(gpd) / np.mean(gpd)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default="ctl-dpl")
    ap.add_argument("--n-gal", type=int, nargs="+", default=[32])
    ap.add_argument("--chunk", type=int, default=32, help="forward_chunk_size K (memory knob)")
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--burnin", type=int, default=0)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--max-doublings", type=int, default=None)
    ap.add_argument(
        "--dense-mass",
        default="false",
        choices=["true", "false"],
        help=(
            "Passed EXPLICITLY to every arm, and defaulting to false, because the "
            "two engines disagree on it by default and the disagreement is not "
            "small. ``CatalogFitter`` resolves ``dense_mass_matrix=None`` through "
            "tengri's auto-policy -- dense below D = 8, diagonal at or above it "
            "(#319) -- so this D = 8 fixture gets a DIAGONAL metric there. "
            "``Fitter._fit_batch_vmap_mcmc`` reads ``kwargs.get('dense_mass_matrix', "
            "True)`` and so gets a DENSE one, which is the configuration CLAUDE.md "
            "records peaking at 20+ GB in NUTS warmup at this dimension. Left "
            "unset, the shared-adaptation arm would be slower for a reason that "
            "has nothing to do with sharing adaptation."
        ),
    )
    ap.add_argument("--precondition", type=float, default=None)
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["catalog", "fit_batch"],
        choices=["catalog", "fit_batch", "solo"],
    )
    ap.add_argument("--n-solo", type=int, default=8, help="galaxies fitted one at a time")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    bns = _load_registry()
    per_galaxy = make_per_galaxy(bns)
    cfg = bns.NOTEBOOKS[args.notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    model = cfg["build"](ssp)
    d = len(model.spec.free_params)
    key = jax.random.PRNGKey(args.seed)
    n_max = max(args.n_gal)
    galaxies = build_catalog(cfg, model, n_max, key)

    meta = {
        "notebook": args.notebook,
        "n_dim": d,
        "n_bands": len(model.observation.photometry.filters),
        "snr": cfg["snr"],
        "device": str(jax.devices()[0]),
        "platform": jax.devices()[0].platform,
        "jax": jax.__version__,
        "precondition": args.precondition,
        "max_doublings": args.max_doublings,
        "tag": args.tag,
        "loadavg": _loadavg(),
    }
    print(f"{args.notebook}: D = {d}, {meta['n_bands']} bands, device {meta['device']}")
    print(f"budget: {args.warmup} warmup + {args.samples} draws, K = {args.chunk}\n")

    run_kw = dict(n_warmup=args.warmup, n_burnin=args.burnin, n_samples=args.samples)
    if args.max_doublings is not None:
        run_kw["max_num_doublings"] = args.max_doublings
    if args.precondition is not None:
        run_kw["precondition"] = args.precondition
    run_kw["dense_mass_matrix"] = args.dense_mass == "true"

    rows = []
    header = (
        f"{'arm':<10}{'N':>5}{'wall_s':>9}{'s/gal':>8}{'gal/min':>9}{'conv':>7}"
        f"{'frozen':>7}{'maxRhat':>10}{'minESS':>9}{'medESS':>9}{'minUniq':>8}"
    )
    print(header)
    print("-" * len(header), flush=True)

    for n_gal in args.n_gal:
        subset = galaxies[:n_gal]

        if "catalog" in args.arms:
            from tengri.inference.catalog_fitter import CatalogFitter

            cat = CatalogFitter(model, subset, data_type="photometry")
            # Cold call first so the timed one is not paying XLA compile: the
            # claim under test is per-galaxy STEADY-STATE cost across a catalog,
            # and a compile amortized over N galaxies is a different number for
            # every N. The cold wall is recorded separately.
            t0 = time.perf_counter()
            cp = cat.run(
                "mcmc_nuts", key=key, forward_chunk_size=args.chunk, verbose=False, **run_kw
            )
            jax.block_until_ready(cp[0].samples)
            cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            cp = cat.run(
                "mcmc_nuts", key=key, forward_chunk_size=args.chunk, verbose=False, **run_kw
            )
            jax.block_until_ready(cp[0].samples)
            warm = time.perf_counter() - t0
            gal_rows, report = per_galaxy(list(cp.posteriors))
            row = _summarize(gal_rows, report, warm, n_gal, args.warmup, args.samples)
            row.update(arm="catalog", adaptation="per-galaxy", cold_s=round(cold, 2), **meta)
            row["per_galaxy"] = gal_rows
            rows.append(row)
            print(_fmt(row), flush=True)

        if "fit_batch" in args.arms:
            from tengri.inference.fitter import Fitter

            fitter = Fitter(model, subset[0]["flux_obs"], subset[0]["noise"])
            t0 = time.perf_counter()
            res = fitter.fit_batch(subset, method="mcmc_nuts", key=key, verbose=False, **run_kw)
            jax.block_until_ready(res[0].samples)
            cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = fitter.fit_batch(subset, method="mcmc_nuts", key=key, verbose=False, **run_kw)
            jax.block_until_ready(res[0].samples)
            warm = time.perf_counter() - t0
            gal_rows, report = per_galaxy(list(res))
            row = _summarize(gal_rows, report, warm, n_gal, args.warmup, args.samples)
            row.update(
                arm="fit_batch", adaptation="shared-first-galaxy", cold_s=round(cold, 2), **meta
            )
            row["per_galaxy"] = gal_rows
            rows.append(row)
            print(_fmt(row), flush=True)

    if "solo" in args.arms:
        from tengri import ForwardModel
        from tengri.config.exceptions import DeadFitError

        n_solo = min(args.n_solo, len(galaxies))
        gal_rows = []
        walls = []
        refusals = []
        # ONE ForwardModel for the whole arm. ``benchmark_notebook_sampler.run_one``
        # deliberately rebuilds per row because its rows differ in tuning settings
        # and the adaptation cache is keyed on those (#1853); here every galaxy runs
        # the same settings and only the data changes, so a rebuild per galaxy would
        # re-pay compile N times and report it as sampling.
        forward = ForwardModel.build(sed=model)
        for i in range(n_solo):
            gal = galaxies[i]
            data = tengri.Data(photometry=(gal["flux_obs"], gal["noise"]))
            k = jax.random.fold_in(key, 10_000 + i)
            map_seed = forward.fit(
                data, method="map", key=k, n_restarts=8, n_steps=800, verbose=False
            )
            t0 = time.perf_counter()
            try:
                post = forward.fit(
                    data,
                    key=k,
                    init_from=map_seed,
                    n_chains=1,
                    method="mcmc_nuts",
                    verbose=False,
                    **run_kw,
                )
            except DeadFitError as exc:
                # A refusal is an OUTCOME, not a harness failure (#2088, #2093),
                # and on THIS arm it is the load-bearing one: the batched engines
                # cannot raise per galaxy -- ``run_one`` lives inside ``lax.map``,
                # where a Python raise is not expressible -- so the same galaxy
                # comes back from them as a frozen lane with a plausible-looking
                # R-hat instead. Dropping it here would delete the only place the
                # library says out loud that this fit is dead.
                walls.append(time.perf_counter() - t0)
                refusals.append({"galaxy": i, "reason": str(exc)[:300]})
                continue
            walls.append(time.perf_counter() - t0)
            gal_rows.append(post)
        scored, report = per_galaxy(gal_rows)
        for r, w in zip(scored, walls):
            r["wall_s"] = round(w, 2)
        row = _summarize(scored, report, float(np.sum(walls)), n_solo, args.warmup, args.samples)
        row.update(arm="solo", adaptation="per-galaxy-solo", cold_s=None, **meta)
        row["per_galaxy"] = scored
        row["refusals"] = refusals
        row["n_refused"] = len(refusals)
        rows.append(row)
        print(_fmt(row), flush=True)

    if args.json:
        with open(args.json, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"\nappended {len(rows)} rows to {args.json}")


def _fmt(row) -> str:
    def g(k, spec, default="n/a"):
        v = row.get(k)
        return default.rjust(int(spec.split(".")[0].lstrip(">"))) if v is None else format(v, spec)

    return (
        f"{row['arm']:<10}{row['n_gal']:>5}{row['wall_s']:>9.1f}{row['s_per_galaxy']:>8.2f}"
        f"{row['gal_per_min']:>9.1f}{row['n_converged']:>7}{row['n_frozen']:>7}"
        f"{g('max_rhat', '>10.4f')}{g('min_ess', '>9.1f')}{g('median_ess', '>9.1f')}"
        f"{g('min_unique_frac', '>8.3f')}"
    )


if __name__ == "__main__":
    main()
