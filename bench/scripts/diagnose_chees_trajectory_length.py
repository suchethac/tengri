#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Why does ChEES adapt a short trajectory on some posteriors and a long one on others?

Across the first campaign rows the split is total: **every configuration that
converged learned a long trajectory, and every one that failed learned a short
one.**

    fixture  config          adapted L   % of the 200 cap   min ESS   max R-hat
    nb00     chees+precond       183.5              92%        13.2      1.003
    nb01     chees+precond       166.7              83%       336.6      1.003
    nb05     chees+precond        77.2              39%         1.7      1.239
    ctl      chees+precond        18.9               9%         2.2      1.142
    nb05     chees (no metric)    25.7              13%         1.1      2.461

The obvious reading -- ``max_leapfrog_steps=200`` is clipping the adaptation --
is **refuted by the same table**: the two rows that fail sit at 39% and 9% of
the ceiling, nowhere near it. The ceiling does bind, but on the rows that
*work* (92%, 83%), which is a separate finding and the reason this script also
sweeps it.

The mechanism BlackJAX's own ``chees_adaptation`` docstring names is the other
way round:

    per-chain ... inits on the unconstrained space ... catastrophically break
    identity-metric ChEES on scale-separated targets (**dispersion inflates the
    cross-chain jump-distance criterion, driving the adapted trajectory length
    down**)

ChEES's criterion is the change in the *cross-chain* expected square. A
dispersed ensemble already has a large expected square before the sampler does
anything, so the criterion is dominated by the initial spread rather than by
what the trajectory achieved, and the optimizer settles on a shorter length.
``run_chees`` seeds its ensemble by dispersing ``ensemble_jitter`` around the
MAP, so that dial is exactly the one the docstring is describing.

This script sweeps it, against the ceiling, on one notebook's own mock, and
reports the adapted length beside the diagnostics -- so "ChEES underperforms
here" becomes a mechanism or is ruled out.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python \\
        bench/scripts/diagnose_chees_trajectory_length.py --notebook ctl
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np
from benchmark_notebook_sampler import (
    NOTEBOOKS,
    _unique_draw_fraction,
)

import tengri
from tengri import Data, ForwardModel, generate_mock
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

#: (label, ensemble_jitter, max_leapfrog_steps, chain_jitter).
#:
#: Three dials, deliberately separated, because collapsing any two of them is
#: what made the first reading of these rows ambiguous.
#:
#: * ``ensemble_jitter`` -- the hypothesis under test. Dispersion inflates
#:   ChEES's cross-chain criterion and drives the adapted length down, so this
#:   should want to be TIGHT.
#: * ``max_leapfrog_steps`` -- the control for it. The first campaign put the
#:   failing rows at 9-39% of the 200 ceiling, so clipping cannot be their
#:   explanation; but it put the *converging* rows at 83-92%, so the ceiling is
#:   binding exactly where it must not. Both are swept against BlackJAX's own
#:   default of 1000.
#: * ``chain_jitter`` -- ``None`` seeds the sampling chains from the ensemble's
#:   warmed final states (correlated with the ensemble that tuned them, so
#:   split R-hat over them is closer to a consistency check); a float seeds them
#:   independently and OVERDISPERSED, which is what makes R-hat a real test.
#:   This is a *diagnostic* dial and it wants to be WIDE.
#:
#: The last arm is the one that matters if the hypothesis holds: a tight ensemble
#: for the criterion AND wide sampling chains for R-hat, which is only expressible
#: because the two chain sets were separated. If the tight-ensemble arms win on L
#: and ESS but their R-hat is flattered by a narrow init, that arm is what
#: distinguishes a real improvement from a better-looking one.
ARMS = (
    ("jitter 0.5  cap 200", 0.5, 200, None),
    ("jitter 0.1  cap 200", 0.1, 200, None),
    ("jitter 0.01 cap 200", 0.01, 200, None),
    ("jitter 0.1  cap 1000", 0.1, 1000, None),
    ("jitter 0.01 cap 1000", 0.01, 1000, None),
    ("tight ens + wide chains", 0.01, 1000, 0.5),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), default="ctl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--n-burnin", type=int, default=500)
    parser.add_argument("--n-ensemble", type=int, default=32)
    parser.add_argument(
        "--precondition",
        default="0.5",
        help="'none' or a whitening strength; the metric ChEES does NOT estimate itself",
    )
    parser.add_argument("--json", default=None, help="append one JSON row per arm")
    parser.add_argument(
        "--only-arm",
        default=None,
        help="run only arms whose label contains this substring (resume a partial sweep)",
    )
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.notebook]
    seed = cfg["seed"] if args.seed is None else args.seed
    precondition = None if args.precondition == "none" else float(args.precondition)

    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    print(
        f"notebook {args.notebook}: D = {len(sed.spec.free_params)} free parameters, "
        f"{cfg['n_chains']} chains, seed {seed}, precondition={precondition}"
    )
    header = (
        f"{'arm':<24}{'wall s':>9}{'adapted L':>11}{'% cap':>7}{'step':>10}"
        f"{'maxRhat':>10}{'div':>5}{'minESS':>9}{'uniq':>7}  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    arms = ARMS if args.only_arm is None else [a for a in ARMS if args.only_arm in a[0]]
    if not arms:
        parser.error(f"--only-arm {args.only_arm!r} matched no arm of {[a[0] for a in ARMS]}")
    for label, jitter, cap, chain_jitter in arms:
        # A fresh model per arm: adaptation caches key on the tuning tuple, and a
        # fresh build keeps the MAP seed identical across arms.
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        map_seed = forward.fit(
            data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
        )
        started = time.perf_counter()
        post = forward.fit(
            data,
            key=key_fit,
            init_from=map_seed,
            method="mcmc_chees",
            n_chains=cfg["n_chains"],
            n_warmup=args.n_warmup,
            n_burnin=args.n_burnin,
            n_samples=args.n_samples,
            n_ensemble=args.n_ensemble,
            ensemble_jitter=jitter,
            chain_jitter=chain_jitter,
            max_leapfrog_steps=cap,
            precondition=precondition,
            verbose=False,
        )
        wall = time.perf_counter() - started

        rhats = post.rhat()
        ess = effective_sample_size({k: np.asarray(v) for k, v in post.samples.items()})
        finite = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
        worst_name, worst_ess = min(finite, key=lambda p: p[1]) if finite else ("?", float("nan"))
        adapted = float(post.diagnostics["n_leapfrog_steps"])
        row = {
            "notebook": args.notebook,
            "seed": seed,
            "arm": label,
            "ensemble_jitter": jitter,
            "chain_jitter": chain_jitter,
            "max_leapfrog_steps": cap,
            "precondition": precondition,
            "wall": wall,
            "adapted_L": adapted,
            "frac_of_cap": adapted / cap,
            "step_size": float(post.diagnostics["step_size"]),
            "rhat": max(float(v) for v in rhats.values()) if rhats else float("nan"),
            "divergences": int(post.diagnostics.get("n_divergent", 0) or 0),
            "min_ess": worst_ess,
            "worst": worst_name,
            "unique_frac": _unique_draw_fraction(post),
        }
        print(
            f"{label:<24}{wall:>9.1f}{adapted:>11.1f}{100 * adapted / cap:>6.0f}%"
            f"{row['step_size']:>10.4g}{row['rhat']:>10.4f}{row['divergences']:>5}"
            f"{worst_ess:>9.1f}{row['unique_frac']:>7.3f}  {worst_name}",
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
