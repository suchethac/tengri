#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Cost to reach a fixed effective sample size, from a sampler sweep's JSONL.

**Wall clock at a fixed draw count measures the budget someone chose, not the
sampler.** A configuration that finishes sooner because it drew fewer samples has
achieved nothing, and two rows at the same draw count are only comparable if
their effective sample sizes happen to match -- which is exactly what a better
metric is supposed to change. So this scorer converts each row to the quantity
that is actually comparable: **gradient evaluations to reach a target minimum
effective sample size**, warmup included.

    grads_to_target = n_warmup * grads_per_draw
                    + (target / min_ess) * n_draws_total * grads_per_draw

The first term is the one usually left out and it is not small: warmup has
measured at 2.52x the sampling half on this project's fits, so a metric that
reaches a usable mass matrix in fewer warmup steps can matter more than one that
samples more efficiently afterwards.

The extrapolation in the second term assumes effective sample size grows linearly
with draws, which holds once a chain is mixing and **fails for a chain that is
not**. A row that misses the convergence bar therefore has its projection
reported as a lower bound and flagged, never quietly averaged in.

**No row is scored on cost alone.** Divergence count, unique-draw fraction and
max R-hat are carried on every line, because none of the three is sufficient by
itself here: this project has measured cells at R-hat 2.97 with zero divergences,
and cells with zero divergences whose unique-draw fractions were 1.000.

Seeds are summarized on the **worst** seed, not the mean. Configurations that
cleared a bar on one seed and failed on its partner are common enough here that a
mean is a way of not noticing.

Usage::

    python bench/scripts/score_low_rank_campaign.py /path/to/sweep.jsonl
    python bench/scripts/score_low_rank_campaign.py sweep.jsonl --target-ess 100 \
        --n-warmup 1000 --seconds-target 20
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

#: Convergence bar every row is held to before its cost is believed. Matches the
#: benchmark harness's own primary bar so rows join that table rather than
#: starting a new one.
MAX_RHAT = 1.01

#: A chain whose draws are mostly repeats is not mixing whatever its R-hat says.
#: 0.9 is deliberately loose: it is here to catch a frozen or near-frozen chain,
#: not to grade a healthy one.
MIN_UNIQUE_FRAC = 0.9


def load_rows(paths: list[str]) -> list[dict]:
    """Read every JSONL row from every path, newest line per key winning.

    The harness appends, so a re-run of one cell leaves both lines in the file
    and the later one is the measurement that stands.
    """
    latest: dict[tuple, dict] = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                latest[(row["notebook"], row["config"], row["seed"])] = row
    return list(latest.values())


def verdict(row: dict) -> str:
    """Three-way health check: R-hat, divergences and unique draws together."""
    if row.get("dead_fit"):
        return "REFUSED"
    rhat = row.get("rhat")
    if rhat is None or not math.isfinite(rhat) or rhat >= MAX_RHAT:
        return "not-converged"
    if (row.get("unique_frac") or 0.0) < MIN_UNIQUE_FRAC:
        return "repeating"
    if (row.get("divergences") or 0) > 0:
        return "divergent"
    return "ok"


def cost_to_target(row: dict, target_ess: float, n_warmup: int) -> tuple[float, float]:
    """Gradients and seconds to reach ``target_ess``, warmup included.

    Parameters
    ----------
    row : dict
        One sweep row.
    target_ess : float
        Minimum effective sample size the projection aims at, per parameter,
        taken on the worst-mixing parameter.
    n_warmup : int
        Warmup steps the row was run with. Not carried in the sweep JSONL, so it
        is supplied by the caller and reported alongside the numbers.

    Returns
    -------
    gradients : float
        Total gradient evaluations, ``nan`` when the row cannot be priced.
    seconds : float
        The same budget in seconds, at this row's own measured seconds per
        gradient. Contention-sensitive; the gradient column is not.
    """
    grads_per_draw = row.get("grad_per_draw")
    draws = row.get("n_draws_total")
    ess = row.get("min_ess")
    if not grads_per_draw or not draws or not ess or ess <= 0:
        return float("nan"), float("nan")
    warmup_grads = n_warmup * grads_per_draw
    sample_grads = draws * grads_per_draw
    needed = warmup_grads + (target_ess / ess) * sample_grads
    wall = row.get("wall")
    if not wall:
        return needed, float("nan")
    seconds_per_grad = wall / (warmup_grads + sample_grads)
    return needed, needed * seconds_per_grad


def main() -> None:
    """Print a per-seed table and a worst-seed summary for each configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", help="sweep JSONL files")
    parser.add_argument("--target-ess", type=float, default=100.0)
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help="warmup steps the sweep ran with (the harness's non-quick default)",
    )
    parser.add_argument(
        "--seconds-target",
        type=float,
        default=20.0,
        help="performance target to measure the gap against [s]",
    )
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["notebook"], row["config"])].append(row)

    for (notebook, config), seed_rows in sorted(grouped.items()):
        seed_rows.sort(key=lambda r: r["seed"])
        print(f"\n=== {notebook} / {config} ===")
        print(
            f"  {'seed':>5} {'maxRhat':>10} {'div':>5} {'uniq':>6} {'minESS':>8}"
            f" {'g/draw':>7} {'grads->ESS':>12} {'secs->ESS':>10}  verdict"
        )
        converged = []
        for row in seed_rows:
            grads, seconds = cost_to_target(row, args.target_ess, args.n_warmup)
            state = verdict(row)
            mark = "" if state == "ok" else " (lower bound)"
            print(
                f"  {row['seed']:>5} {row.get('rhat', float('nan')):>10.4f}"
                f" {row.get('divergences')!s:>5} {row.get('unique_frac', 0.0):>6.3f}"
                f" {row.get('min_ess', 0.0):>8.1f} {row.get('grad_per_draw') or 0:>7.1f}"
                f" {grads:>12.0f} {seconds:>10.1f}  {state}{mark}"
            )
            if state == "ok":
                converged.append((grads, seconds, row))

        n_ok = len(converged)
        print(f"  seeds clearing the three-way bar: {n_ok} of {len(seed_rows)}")
        if n_ok == len(seed_rows) and n_ok:
            worst_grads = max(c[0] for c in converged)
            worst_seconds = max(c[1] for c in converged)
            gap = worst_seconds / args.seconds_target
            print(
                f"  WORST SEED to min-ESS {args.target_ess:.0f}:"
                f" {worst_grads:.0f} gradients, {worst_seconds:.1f} s"
                f"  ({gap:.1f}x the {args.seconds_target:.0f} s target)"
            )
        else:
            print(
                "  no worst-seed cost is reported: a projection from a chain that "
                "is not mixing is not a measurement of how long it would take to "
                "mix. The per-seed numbers above are lower bounds."
            )


if __name__ == "__main__":
    main()
