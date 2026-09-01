#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Six seeds per row, ONE FIT PER SUBPROCESS, across the three notebook models.

This is the 2026-08-21 campaign protocol and it exists for two reasons, both of
which have already cost a table:

* **One fit per subprocess.** ``_get_cached_adaptation`` keys on the model, the
  tuning tuple and the data fingerprint, and ``_maybe_map_init`` caches the MAP
  point the same way. Two fits in one process therefore share an adaptation and
  a compiled program, so the second seed is not an independent measurement of
  anything. A subprocess boundary is the only thing that makes it one.
* **Six seeds.** A single seed on a D=8 posterior with a weakly-identified SFH
  direction reports whichever tail that seed's mock happened to land in. The
  reported number is the *worst* R-hat across seeds, because the adoption bar is
  a claim about the sampler, not about a lucky mock.

Each subprocess also re-reports its own blackjax version: a shared venv once
held blackjax 1.3 while the table claimed 1.6, and that invalidated the whole
campaign.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/run_ghmc_meads_campaign.py \\
        --notebooks 00 01 05 --seeds 0 1 2 3 4 5 --out bench/results/ghmc_meads.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "benchmark_notebook_sampler.py"


def summarize(path: Path) -> None:
    """Fold the per-seed rows into the worst-across-seeds row per configuration.

    The reported R-hat, divergence count and minimum ESS are the *worst* over
    the seeds, not the mean. The bar is a claim about the sampler; a mean would
    let one good mock cover for five bad ones.

    The file is append-only, so a re-run of one ``(notebook, config, seed)`` cell
    leaves two records for it. The later one wins: appending is how the campaign
    survives an interrupted run, and counting a repeated seed twice would let a
    re-run silently reweight the worst-case.
    """
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[(record["notebook"], record["config"], record["seed"])] = record

    rows = defaultdict(list)
    for (notebook, config, _seed), record in latest.items():
        rows[(notebook, config)].append(record)

    print(
        f"{'nb':>4}  {'config':<22}{'n':>3}{'mean wall':>11}{'worst Rhat':>12}"
        f"{'max div':>9}{'min ESS':>9}{'s/ESS':>9}{'min uniq':>10}  worst-mixing parameter"
    )
    for (nb, config), records in sorted(rows.items()):
        if "returncode" in records[0]:
            print(f"{nb:>4}  {config:<22}{len(records):>3}  SUBPROCESS FAILED")
            continue
        worst = max(records, key=lambda r: r["rhat"])
        rhat_cell = (
            f"{worst['rhat']:>12.4f}" if abs(worst["rhat"]) < 1e4 else f"{worst['rhat']:>12.3e}"
        )
        print(
            f"{nb:>4}  {config:<22}{len(records):>3}"
            f"{sum(r['wall'] for r in records) / len(records):>11.1f}"
            f"{rhat_cell}"
            f"{max(r['divergences'] for r in records):>9}"
            f"{min(r['min_ess'] for r in records):>9.1f}"
            f"{max(r['sec_per_ess'] for r in records):>9.3f}"
            # Zero divergences is not evidence of health (#1999): mcmc_nuts
            # froze completely on 3.1% of galaxies with none reported, and split
            # R-hat scores ~1.0 on a chain that never moved. The worst
            # distinct-draw fraction across seeds is the column that sees it.
            f"{min(r.get('unique_frac', float('nan')) for r in records):>10.3f}"
            f"  {worst['worst']}"
        )
        per_seed = ", ".join(
            f"{r['seed']}: R{r['rhat']:.3f} E{r['min_ess']:.1f} D{r['divergences']}"
            if abs(r["rhat"]) < 1e4
            else f"{r['seed']}: R{r['rhat']:.2e} E{r['min_ess']:.1f} D{r['divergences']}"
            for r in sorted(records, key=lambda r: r["seed"])
        )
        print(f"      seeds -> {per_seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", nargs="+", default=["00", "01", "05"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--methods", default="nuts,ghmc")
    parser.add_argument("--out", default="bench/results/ghmc_meads_campaign.jsonl")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="re-print the table from an existing JSONL without running anything",
    )
    args = parser.parse_args()

    out = Path(args.out)
    if args.summarize_only:
        summarize(out)
        return

    out.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")

    for nb in args.notebooks:
        for seed in args.seeds:
            cmd = [
                sys.executable,
                "-u",
                str(RUNNER),
                "--notebook",
                nb,
                "--seed",
                str(seed),
                "--methods",
                args.methods,
                "--json",
                str(out),
            ]
            if args.quick:
                cmd.append("--quick")
            print(f"\n=== notebook {nb}, seed {seed} ===", flush=True)
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                # A crashed fit is data too, and silently skipping it is how a
                # table ends up describing only the seeds that happened to work.
                with out.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "notebook": nb,
                                "seed": seed,
                                "config": "SUBPROCESS FAILED",
                                "returncode": proc.returncode,
                            }
                        )
                        + "\n"
                    )
                print(f"!!! notebook {nb} seed {seed} exited {proc.returncode}", flush=True)

    print(f"\nwrote {out}\n")
    summarize(out)


if __name__ == "__main__":
    main()
