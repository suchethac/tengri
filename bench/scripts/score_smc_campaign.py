#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Fold the SMC campaign JSONL into the tables ``2026-08-31_smc_evaluation.md`` prints.

Two things this does that a hand-written table cannot be trusted to do.

**It computes the divergence rate against the right denominator.** A chain
sampler makes one Metropolis transition per kept draw, so ``n_divergent /
total_draws`` is a rate. Tempered SMC makes ``n_temperatures * n_mcmc_steps``
transitions per particle and keeps one draw from each, so the same ratio
overshoots by that factor -- it read **205 %** on the first row measured, which
at least announces itself, where a configuration with fewer rungs would have
produced a plausible-looking number instead. That is #2087's arithmetic one
sampler further out. Rows written before ``n_inner_transitions`` was published
are reconstructed from ``n_temperatures`` and the arm's own inner-move count.

**It prints the particle diagnostic beside the autocorrelation one.** The
``minESS`` column of every other report in ``bench/`` is an autocorrelation ESS,
and for a resampled particle population that estimator is measuring the *order*
duplicates happen to sit in rather than any mixing. ``ancESS`` -- the effective
number of distinct particles surviving the worst resample -- is the number that
means what the column header suggests. A row with only one of the two is
unreadable in exactly one direction.

Usage::

    .venv/bin/python bench/scripts/score_smc_campaign.py \\
        bench/results/2026-08-31_smc_campaign.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Inner Metropolis moves per particle per rung, by config label.
#:
#: Only needed for rows written before the backend published
#: ``n_inner_transitions``. Keyed on the label because the label is what the
#: campaign recorded; a row whose label is not here reports its rate as unknown
#: rather than guessing a denominator, which is the whole failure being avoided.
_INNER_MOVES = {
    "smc": (5, 512),
    "smc+precond": (5, 512),
    "smc+precond cheap": (2, 512),
    "smc+precond n1": (1, 512),
    "smc+precond nogain": (2, 512),
    "smc+precond fixed16": (5, 512),
}


def _transitions(row: dict) -> int | None:
    """Inner Metropolis transitions this row made, or None when unknowable."""
    if row.get("n_transitions_total"):
        return int(row["n_transitions_total"])
    rungs = row.get("n_temperatures")
    arm = _INNER_MOVES.get(row.get("config", ""))
    if rungs is None or arm is None:
        return None
    moves, particles = arm
    return int(sum(rungs)) * particles * moves


def _rate(row: dict) -> float | None:
    """Divergent fraction, against transitions for SMC and kept draws otherwise."""
    if row.get("divergences") is None:
        return None
    denominator = _transitions(row) or row.get("n_draws_total")
    if not denominator:
        return None
    return row["divergences"] / denominator


def _grad_per_effective(row: dict) -> float | None:
    """Gradients per effective sample, using each sampler's OWN honest ESS.

    For a chain sampler that is the autocorrelation ESS, which is what it
    measures. For tempered SMC it is the ancestor ESS summed over populations:
    the autocorrelation estimator applied to a particle population is reading
    the order duplicates happen to sit in -- systematic resampling leaves copies
    adjacent, so it reports a small number for a perfectly healthy population --
    and it is not a mixing measure at all.

    **Neither number is the true effective sample size**, and the SMC one is an
    upper bound: particles sharing an ancestor are correlated, so the ancestor
    count over-credits. Reported because a column that is wrong in a known
    direction is worth more than one that is wrong in an unknown direction, and
    because the alternative -- the autocorrelation ESS on both -- is wrong in
    the unknown direction for one of the two samplers.
    """
    anc = row.get("min_ancestor_ess")
    grad = row.get("grad_per_draw")
    if grad is None:
        return None
    if anc is None:
        return row.get("grad_per_ess")
    rungs = row.get("n_temperatures") or [1]
    return grad * (row.get("n_draws_total") or 0) / max(anc * len(rungs), 1e-9)


def _fmt(value, spec="", dash="-"):
    """Format a number, or a dash when the column does not apply to this sampler."""
    return dash if value is None else format(value, spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", type=Path)
    args = ap.parse_args(argv)

    latest: dict[tuple, dict] = {}
    for path in args.jsonl:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                latest[(record["notebook"], record["config"], record["seed"])] = record

    header = (
        f"{'nb':<8}{'seed':>5}  {'config':<22}{'wall s':>8}{'maxRhat':>10}"
        f"{'div':>7}{'div rate':>10}{'minESS':>8}{'ancESS':>8}{'rungs':>7}"
        f"{'g/draw':>8}{'g/ESS':>10}{'g/effS':>10}{'uniq':>7}"
    )
    print(header)
    print("-" * len(header))
    for (nb, config, seed), row in sorted(latest.items()):
        if row.get("dead_fit"):
            print(f"{nb:<8}{seed:>5}  {config:<22}  REFUSED (DeadFitError)")
            continue
        rate = _rate(row)
        rungs = row.get("n_temperatures")
        rhat = row["rhat"]
        rhat_cell = f"{rhat:>10.4f}" if abs(rhat) < 1e4 else f"{rhat:>10.3e}"
        print(
            f"{nb:<8}{seed:>5}  {config:<22}{row['wall']:>8.0f}{rhat_cell}"
            f"{_fmt(row['divergences'], '>7'):>7}"
            f"{_fmt(rate and 100 * rate, '>9.2f'):>9}%"
            f"{row['min_ess']:>8.1f}"
            f"{_fmt(row.get('min_ancestor_ess'), '>8.0f'):>8}"
            f"{_fmt(rungs and max(rungs), '>7'):>7}"
            f"{_fmt(row.get('grad_per_draw'), '>8.0f'):>8}"
            f"{_fmt(row.get('grad_per_ess'), '>10.0f'):>10}"
            f"{_fmt(_grad_per_effective(row), '>10.0f'):>10}"
            f"{row.get('unique_frac', float('nan')):>7.3f}"
        )
        if row.get("reached_target") is False:
            print(f"{'':<15}  !! lambda < 1: these draws are TEMPERED, not the posterior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
