#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Score every ChEES campaign row against BOTH adoption criteria.

Two criteria, reported side by side, because one of them cannot do the job it
was borrowed for.

**Primary** -- the notebooks' own claim: ``max split-R-hat < 1.01``, **zero**
divergences, and ``min ESS >= mcmc_nuts`` on the same mock and seed. This is an
absolute self-assessment of one shipped fit and it is reported unchanged.

**Secondary** -- comparative, and applied identically to the NUTS baseline row:
``max split-R-hat < 1.01``, divergence **rate** below
:data:`MAX_DIVERGENCE_RATE` of *total* draws, and the same ESS clause. The
primary bar does not discriminate when used to rank samplers -- NUTS on the
healthy DPL control is R-hat 1.0002 at min ESS 223 with **17 divergences**, a
plainly good fit that "zero divergences" calls a miss -- and a criterion that
fails the incumbent and the challenger alike cannot separate them.

Neither criterion is met by a ``REFUSED`` row (``DeadFitError``, #2088): there
is no posterior to score.

The divergence denominator is ``n_samples * n_chains``, never ``n_samples``:
every backend records ``n_samples`` per chain while ``n_divergent`` is summed
over every chain, so a rate taken against ``n_samples`` is ``n_chains`` times
too large (#2087). Rows measured before the harness recorded that total have it
reconstructed from the notebook's own chain count, which is why
:data:`_CHAINS` exists.

Usage::

    .venv/bin/python bench/scripts/score_chees_campaign.py
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

MAX_RHAT = 1.01
MAX_DIVERGENCES = 0
MAX_DIVERGENCE_RATE = 0.005

#: Sampling chains per fixture, for reconstructing the divergence denominator on
#: rows measured before ``n_draws_total`` was recorded. Mirrors ``NOTEBOOKS``.
_CHAINS = {"00": 4, "01": 4, "05": 2, "ctl": 2}

#: Per-chain draws per configuration, same purpose. The ChEES and non-shipped
#: rows all take the harness's ``draws`` budget; the shipped NUTS row takes the
#: notebook's own committed ``n_samples``.
_SHIPPED_SAMPLES = {"00": 600, "01": 600, "05": 600, "ctl": 600}


def _total_draws(row: dict) -> int:
    """Draws summed over chains -- recorded if present, reconstructed if not."""
    if row.get("n_draws_total"):
        return int(row["n_draws_total"])
    nb = row["notebook"]
    return _SHIPPED_SAMPLES.get(nb, 600) * _CHAINS.get(nb, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="bench/results/2026-08-30_chees_campaign_*.jsonl")
    args = parser.parse_args()

    latest: dict[tuple, dict] = {}
    for path in sorted(glob.glob(args.glob)):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                # Append-only files: a re-measured cell has two records and the
                # later one is the current code version.
                latest[(r["notebook"], r["config"], r["seed"])] = r

    # The ESS clause is per (notebook, seed) against that cell's own NUTS row.
    baseline_ess: dict[tuple, float] = {}
    for (nb, config, seed), r in latest.items():
        if config.startswith("nuts") and not r.get("dead_fit") and "returncode" not in r:
            baseline_ess[(nb, seed)] = r["min_ess"]

    per_config = defaultdict(list)
    print(
        f"{'nb':>4}{'seed':>5}  {'config':<15}{'R-hat':>12}{'div':>7}{'/total':>8}"
        f"{'rate%':>8}{'minESS':>9}{'vs nuts':>9}  primary  secondary"
    )
    for key in sorted(latest, key=lambda k: (k[0], k[2], k[1])):
        nb, config, seed = key
        r = latest[key]
        if "returncode" in r:
            print(f"{nb:>4}{seed:>5}  {config:<15}  SUBPROCESS FAILED rc={r['returncode']}")
            continue
        if r.get("dead_fit"):
            print(
                f"{nb:>4}{seed:>5}  {config:<15}{'REFUSED':>12}{'-':>7}{'-':>8}"
                f"{'-':>8}{'-':>9}{'-':>9}   MISSES   MISSES"
            )
            per_config[(nb, config)].append((False, False))
            continue
        total = _total_draws(r)
        rate = r["divergences"] / total
        ref = baseline_ess.get((nb, seed))
        ess_ok = ref is None or r["min_ess"] >= ref
        primary = r["rhat"] < MAX_RHAT and r["divergences"] <= MAX_DIVERGENCES and ess_ok
        secondary = r["rhat"] < MAX_RHAT and rate < MAX_DIVERGENCE_RATE and ess_ok
        per_config[(nb, config)].append((primary, secondary))
        ess_ratio = f"{r['min_ess'] / ref:.2f}x" if ref else "-"
        print(
            f"{nb:>4}{seed:>5}  {config:<15}{r['rhat']:>12.4g}{r['divergences']:>7}"
            f"{total:>8}{100 * rate:>8.2f}{r['min_ess']:>9.1f}{ess_ratio:>9}"
            f"   {'clears' if primary else 'MISSES':<8} {'clears' if secondary else 'MISSES'}"
        )

    print("\n--- worst across seeds (the adoption rule) ---")
    print(f"{'nb':>4}  {'config':<15}{'seeds':>6}  primary   secondary")
    for (nb, config), verdicts in sorted(per_config.items()):
        p = all(v[0] for v in verdicts)
        s = all(v[1] for v in verdicts)
        print(
            f"{nb:>4}  {config:<15}{len(verdicts):>6}  "
            f"{'CLEARS' if p else 'MISSES':<9} {'CLEARS' if s else 'MISSES'}"
        )


if __name__ == "__main__":
    main()
