#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Re-measure the phase's load-bearing rows with an R-hat that can fail.

Every ChEES row in ``bench/reports/2026-08-30_chees_hmc.md`` was measured with
``chain_jitter=None``: the sampling chains inherit the adaptation ensemble's
warmed final states. Those chains are correlated with each other and with the
ensemble that tuned the sampler, so split R-hat over them is closer to a
consistency check than to an independent test -- the failure BlackJAX's own
``chees_adaptation`` docstring names ("initializing all chains at a single point
... can produce clean R-hat that is structurally blind to same-basin
non-equilibrium").

The sweep measured the size of that flattery **once**, on the control's seed 0:
the same configuration reads split-R-hat 1.0097 with chains from the ensemble
and **1.0374** with chains seeded independently and overdispersed. A shift of
**+0.0277**.

That is roughly **three times the margin** by which the phase's two clearing
rows clear::

    nb05 seed 7   R-hat 1.0000   (margin 0.0100)
    nb05 seed 2   R-hat 1.0023   (margin 0.0077)

So "unlikely to be overturned" was not a supportable reading of the arithmetic.
Applying the measured shift puts both near 1.03, i.e. **missing** the 1.01 bar.
Those two rows are the entire answer to this phase's question -- whether
cross-chain adaptive trajectory length converges where a fixed ``L`` cannot --
and every other row in the report can carry the caveat while these two cannot.

This script re-runs exactly those rows with ``chain_jitter`` set, changing
nothing else. Three outcomes, all publishable:

* they hold near 1.00 -- the headline is real and now rests on an R-hat that
  could have failed;
* they shift to ~1.03 -- ChEES clears nothing anywhere and Phase 2's answer is a
  clean, well-evidenced NO;
* they land between -- report the number and let the bar decide.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/verify_chees_rhat_honesty.py \\
        --notebook 05 --seeds 2 7 --json bench/results/2026-08-30_chees_rhat_honesty.jsonl
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
from benchmark_notebook_sampler import NOTEBOOKS, _unique_draw_fraction

import tengri
from tengri import Data, ForwardModel, generate_mock
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size
from tengri.inference.backends.mcmc._shared import total_draws

#: The gate configuration, reproduced exactly. Every value here matches the
#: ``chees+precond`` rows in ``configurations()``; the ONLY thing this script
#: changes is ``chain_jitter``, so the comparison isolates the diagnostic dial.
GATE_CONFIG = dict(
    method="mcmc_chees",
    n_warmup=1000,
    n_burnin=500,
    n_samples=600,
    n_ensemble=32,
    precondition=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="05", choices=sorted(NOTEBOOKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 7])
    parser.add_argument(
        "--chain-jitter",
        type=float,
        default=0.5,
        help="sampling-chain overdispersion; the dial under test",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.notebook]
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)

    print(
        f"notebook {args.notebook}, chees+precond, chain_jitter={args.chain_jitter} "
        f"(the gate rows were measured with chain_jitter=None)"
    )
    header = (
        f"{'seed':>5}{'wall s':>9}{'maxRhat':>10}{'div':>5}{'minESS':>9}"
        f"{'uniq':>7}{'adapted L':>11}  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    for seed in args.seeds:
        sed = cfg["build"](ssp)
        key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
        mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
        data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

        # A fresh model per row, matching the harness: adaptation caches key on
        # the tuning tuple, and a fresh build keeps the MAP seed identical.
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        map_seed = forward.fit(
            data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
        )
        started = time.perf_counter()
        post = forward.fit(
            data,
            key=key_fit,
            init_from=map_seed,
            n_chains=cfg["n_chains"],
            chain_jitter=args.chain_jitter,
            verbose=False,
            **GATE_CONFIG,
        )
        wall = time.perf_counter() - started

        rhats = post.rhat()
        ess = effective_sample_size({k: np.asarray(v) for k, v in post.samples.items()})
        finite = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
        worst_name, worst_ess = min(finite, key=lambda p: p[1]) if finite else ("?", float("nan"))
        row = {
            "notebook": args.notebook,
            "seed": seed,
            "config": "chees+precond (chains overdispersed)",
            "chain_jitter": args.chain_jitter,
            "wall": wall,
            "rhat": max(float(v) for v in rhats.values()) if rhats else float("nan"),
            "divergences": int(post.diagnostics.get("n_divergent", 0) or 0),
            "n_draws_total": total_draws(post.diagnostics),
            "min_ess": worst_ess,
            "worst": worst_name,
            "sec_per_ess": wall / max(worst_ess, 1e-9),
            "unique_frac": _unique_draw_fraction(post),
            "n_leapfrog": post.diagnostics.get("n_leapfrog_steps"),
            "step_size": post.diagnostics.get("step_size"),
        }
        print(
            f"{seed:>5}{wall:>9.1f}{row['rhat']:>10.4f}{row['divergences']:>5}"
            f"{worst_ess:>9.1f}{row['unique_frac']:>7.3f}{row['n_leapfrog']:>11.1f}"
            f"  {worst_name}",
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
