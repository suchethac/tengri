#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Convert ``benchmark_notebook_sampler`` rows into gradients-to-a-fixed-ESS.

A wall clock at a fixed draw count measures the budget someone chose, not the
sampler. Two rows that both ran 600 draws are only comparable if they returned
the same number of *effective* draws, and on these fixtures they do not: min ESS
across the rows scored here spans 1.1 to 264. So every row is converted to the
work needed to reach one fixed effective-sample-size target, **warmup included
in the numerator**, because warmup is where a NUTS fit's seconds are
(``bench/reports/2026-08-31_fast_nuts.md`` Finding 2: 2.52x the sampling half,
71.6 % of a fit that pays no compile at all).

    grads_to_target = n_chains * n_warmup * g
                    + (target / min_ess) * n_draws_total * g

with ``g`` the measured gradients per sampling draw (``2**tree_depth_mean`` for
NUTS) and ``n_draws_total = n_chains * n_samples``.

TWO THINGS THIS FORMULA GETS WRONG, BOTH NAMED RATHER THAN HIDDEN
=================================================================

1. **It prices warmup at the SAMPLING tree depth**, and warmup trees are deeper:
   during warmup the step size has not converged, so the tree doubles further
   before its U-turn. That is the entire mechanism the ``wcap`` rows exploit. So
   ``grads_to_target`` is a **lower bound** on an uncapped row's true cost, and a
   tighter one on a capped row's, which biases the comparison *against* the cap.
   The bracket is printed beside it: warmup can cost at most
   ``n_chains * n_warmup * (2**cap - 1)``, where ``cap`` is
   ``warmup_max_num_doublings`` when set and BlackJAX's default 10 when not.
2. **It extrapolates ESS linearly in draws**, which is only true of a chain that
   is mixing. A row missing max split-R-hat < 1.01 therefore has its projection
   reported as a LOWER BOUND, marked ``>=``, and is never averaged into a
   summary. The one thing a stuck chain reliably does is fail to improve
   linearly with more draws.

Usage::

    .venv/bin/python bench/scripts/score_photometry_20s.py \\
        bench/results/2026-09-06_photometry_20s.jsonl --target 100
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

#: The adoption bar every row is measured against. ``benchmark_notebook_sampler``
#: uses the same 1.01, which is the notebooks' own published criterion.
MAX_RHAT = 1.01

#: BlackJAX's ``max_num_doublings`` default. A row that did not cap warmup ran
#: its adaptation at this depth, i.e. up to 1023 leapfrog steps per warmup step.
DEFAULT_MAX_DOUBLINGS = 10

#: Chains per fit, per fixture. Read off ``benchmark_notebook_sampler.NOTEBOOKS``
#: rather than assumed: ``n_draws_total`` in a row already carries the factor, so
#: pricing warmup without it would compare a two-chain warmup against a
#: four-chain sampling budget.
N_CHAINS = {"ctl-dpl": 2, "ctl-jwst": 2, "05": 2, "05pre": 2, "01": 4, "00": 4, "00now": 4}

#: Warmup steps per fit, per (fixture, config). Every config in this campaign
#: runs its fixture's own shipped warmup, which is what makes the arms an A/B.
N_WARMUP = {"ctl-dpl": 600, "05": 600, "ctl-jwst": 1000, "01": 100}

#: Warmup tree-depth cap per config label, or ``None`` for BlackJAX's default.
WARMUP_CAP = {
    "nuts (shipped)": None,
    "nuts (shipped)+precond": None,
    "nuts wcap=5": 5,
    "nuts wcap=5+precond": 5,
    "nuts wcap=3": 3,
}


def load(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def project(row, target):
    """Gradients and seconds to ``target`` effective samples, with the bracket.

    Returns ``None`` for a row the sampler refused: a ``DeadFitError`` has no
    posterior, so there is no ESS to extrapolate and a projection would be a
    number about nothing.
    """
    if row.get("dead_fit"):
        return None
    gpd = row.get("grad_per_draw")
    ess = row.get("min_ess")
    n_draws = row.get("n_draws_total")
    if not gpd or not ess or not n_draws:
        return None
    nb, cfg = row["notebook"], row["config"]
    n_chains = N_CHAINS[nb]
    n_warmup = N_WARMUP[nb]
    warm_grads = n_chains * n_warmup * gpd
    sample_grads = (target / ess) * n_draws * gpd
    grads = warm_grads + sample_grads
    spent = warm_grads + n_draws * gpd
    cap = WARMUP_CAP.get(cfg) or DEFAULT_MAX_DOUBLINGS
    # The other end of the bracket: every warmup step paid the cap.
    warm_grads_max = n_chains * n_warmup * (2.0**cap - 1.0)
    return {
        "grads_to_target": grads,
        "grads_to_target_max": warm_grads_max + sample_grads,
        "grads_spent": spent,
        "sec_to_target": row["wall"] * grads / spent,
        "sec_per_grad": row["wall"] / spent,
        "converged": np.isfinite(row["rhat"]) and row["rhat"] < MAX_RHAT,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--target", type=float, default=100.0, help="min ESS the projection targets")
    ap.add_argument("--budget", type=float, default=20.0, help="the seconds/galaxy claim")
    ap.add_argument("--json", default=None, help="write the scored rows here as one JSON array")
    args = ap.parse_args()

    rows = load(args.jsonl)
    # Append-only files supersede: last line for a (notebook, config, seed) wins.
    latest = {}
    for row in rows:
        latest[(row["notebook"], row["config"], row["seed"])] = row
    by_cell = defaultdict(list)
    for (nb, cfg, _), row in latest.items():
        by_cell[(nb, cfg)].append(row)

    print(f"target min ESS = {args.target:g}; adoption bar max split R-hat < {MAX_RHAT}")
    print(f"budget under test = {args.budget:g} s per galaxy\n")
    header = (
        f"{'fixture':<9}{'config':<24}{'n':>3}{'wR-hat':>9}{'div':>5}{'mESS':>7}"
        f"{'uniq':>6}{'g/draw':>8}{'Mgrad->tgt':>11}{'sec->tgt':>10}{'ms/grad':>9}  conv"
    )
    print(header)
    print("-" * len(header))
    out = []
    for (nb, cfg), cell in sorted(by_cell.items()):
        live = [r for r in cell if not r.get("dead_fit")]
        n_dead = len(cell) - len(live)
        if not live:
            print(f"{nb:<9}{cfg:<24}{len(cell):>3}  every seed REFUSED (DeadFitError)")
            continue
        # The WORST seed decides the row. A campaign that averaged seeds would
        # report a row as converged while a sixth of its fits did not.
        worst = max(live, key=lambda r: r["rhat"])
        proj = [project(r, args.target) for r in live]
        proj = [p for p in proj if p is not None]
        if not proj:
            continue
        # Worst seed on the projection too, and it need not be the worst-R-hat
        # seed: a row can converge and still be expensive.
        grads = max(p["grads_to_target"] for p in proj)
        secs = max(p["sec_to_target"] for p in proj)
        all_conv = all(p["converged"] for p in proj)
        divs = [r["divergences"] for r in live if r["divergences"] is not None]
        rec = {
            "notebook": nb,
            "config": cfg,
            "n_seeds": len(cell),
            "n_dead": n_dead,
            "worst_rhat": worst["rhat"],
            "max_div": max(divs) if divs else None,
            "min_ess": min(r["min_ess"] for r in live),
            "min_unique_frac": min(r["unique_frac"] for r in live),
            "median_grad_per_draw": float(np.median([r["grad_per_draw"] for r in live])),
            "grads_to_target_worst": grads,
            "grads_to_target_max_worst": max(p["grads_to_target_max"] for p in proj),
            "sec_to_target_worst": secs,
            "median_ms_per_grad": 1e3 * float(np.median([p["sec_per_grad"] for p in proj])),
            "median_wall": float(np.median([r["wall"] for r in live])),
            "all_seeds_converged": all_conv,
            "worst_param": worst["worst"],
            "target_ess": args.target,
            "meets_budget": bool(all_conv and secs <= args.budget),
        }
        out.append(rec)
        bound = "" if all_conv else ">="
        print(
            f"{nb:<9}{cfg:<24}{len(cell):>3}{rec['worst_rhat']:>9.4f}"
            f"{(rec['max_div'] if rec['max_div'] is not None else -1):>5}"
            f"{rec['min_ess']:>7.1f}{rec['min_unique_frac']:>6.3f}"
            f"{rec['median_grad_per_draw']:>8.1f}"
            f"{bound + format(grads / 1e6, '.2f'):>11}"
            f"{bound + format(secs, '.0f'):>10}"
            f"{rec['median_ms_per_grad']:>9.3f}"
            f"  {'yes' if all_conv else 'NO'}"
        )

    print(
        f"\n'conv' = every seed cleared R-hat < {MAX_RHAT}. A 'NO' row's projection is a "
        "LOWER BOUND (>=):\nlinear-ESS-in-draws fails exactly when a chain is not mixing."
    )
    cheapest = [r for r in out if r["all_seeds_converged"]]
    if cheapest:
        best = min(cheapest, key=lambda r: r["sec_to_target_worst"])
        print(
            f"\ncheapest CONVERGED row: {best['notebook']} / {best['config']} at "
            f"{best['sec_to_target_worst']:.0f} s and "
            f"{best['grads_to_target_worst'] / 1e6:.2f} M gradients on its worst seed "
            f"({best['sec_to_target_worst'] / args.budget:.0f}x the {args.budget:g} s budget)"
            if best["sec_to_target_worst"] > args.budget
            else f"\ncheapest CONVERGED row clears the {args.budget:g} s budget"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
