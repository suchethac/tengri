#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Score ``benchmark_laplace_nuts_20s`` rows into a per-cell pass/fail table.

A cell is one ``(notebook, arm, platform, n_chains, n_samples, n_warmup,
map_restarts, map_steps, map_optimizer, warmup_max_doublings, window_start)``
configuration, scored over its seeds. Following
``score_photometry_20s.py``'s house conventions:

* **The worst seed decides.** A cell that ran six seeds and choked on one is
  not a cell that clears the budget; every worst-seed statistic below (wall,
  gradients-to-target, min ESS, max R-hat, ...) is a max or min taken across
  the cell's seeds, never an average.
* **A non-converging row's projection is a lower bound.** ``grads_to_100`` and
  ``sec_to_100`` extrapolate ESS linearly in draws, which only holds for a
  chain that is mixing. A cell where any seed fails ``rhat_max < 1.01 and
  min_ess >= 100`` has its worst-seed projection printed with a ``>=`` prefix
  and is barred from ``under_budget`` regardless of wall clock.
* **An errored seed is a failed seed, not a blank.** ``benchmark_laplace_nuts_20s``
  writes a complete row on any exception (``error`` set, every numeric field
  ``None``); such a row counts in ``n_seeds`` and ``n_error``, sinks
  ``all_converge`` for its cell, and is excluded from every numeric
  aggregate (there is nothing to aggregate).

Unlike the photometry campaign, this benchmark's own rows already carry a
gradients/seconds-to-ESS=100 projection (:func:`_score` in
``benchmark_laplace_nuts_20s.py``), computed from ``n_grad_adapt``,
``n_grad_sample``, ``min_ess`` and the warm walls. ``--target`` recomputes
that same formula at an arbitrary target instead of trusting the ``_100``
suffix, so it is not just a label:

    grads_to_target = n_grad_adapt + (target / min_ess) * n_grad_sample
    sec_to_target   = adapt_wall_warm + map_wall_warm + (target / min_ess) * sample_wall_warm

which is exactly ``grads_to_100`` / ``sec_to_100`` at ``target=100``. The
convergence bar itself (``rhat_max < 1.01 and min_ess >= 100``) is NOT
adjustable by ``--target`` -- it is the notebooks' own published criterion,
reproduced here rather than trusted from the row's ``converged`` field so
this script has one self-contained definition of "converged".

Usage::

    .venv/bin/python bench/scripts/score_laplace_nuts_20s.py \\
        bench/results/2026-09-11_laplace_nuts_20s.jsonl --budget 20 --target 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

#: The notebooks' own published criterion, reproduced here (not trusted from
#: a row's ``converged`` field) so this script has one definition of "passes".
MAX_RHAT = 1.01
MIN_ESS_BAR = 100.0

#: Every field a cell is keyed on. Two rows agreeing on all of them are the same
#: sampler configuration; they differ only in seed. Rows written before a key
#: field existed carry ``None`` for it and group together, which is right: they
#: were all run at what became that field's default.
CELL_KEY = (
    "notebook",
    "arm",
    "platform",
    "n_chains",
    "n_samples",
    "n_warmup",
    "map_restarts",
    "map_steps",
    "map_optimizer",
    "warmup_max_doublings",
    "window_start",
    "profile_mass",
    "chain_parallel",
    "dtype",
)

#: Worst-seed reducers, keyed by the row field they read. ``max``/``min`` match
#: whichever direction is bad for that quantity; ``median`` fields are
#: reported at the middle seed because there is no single "worst" side to a
#: wall-clock split, a tree depth, or a Hessian condition number.
WORST_MAX_FIELDS = ("wall_total_warm", "rhat_max", "divergences")
WORST_MIN_FIELDS = ("min_ess", "unique_frac")
MEDIAN_FIELDS = (
    "grad_per_draw",
    "tree_depth_mean",
    "map_wall_warm",
    "adapt_wall_warm",
    "sample_wall_warm",
    "hess_cond",
)


def load(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _dedupe(rows):
    """Append-only files supersede: last line for a (cell key, seed) wins."""
    latest = {}
    for row in rows:
        key = (*(row.get(k) for k in CELL_KEY), row["seed"])
        latest[key] = row
    return list(latest.values())


def _converged(row) -> bool:
    """The notebooks' bar, re-derived from ``rhat_max``/``min_ess`` directly."""
    rhat_max, min_ess = row.get("rhat_max"), row.get("min_ess")
    if rhat_max is None or min_ess is None:
        return False
    return rhat_max < MAX_RHAT and min_ess >= MIN_ESS_BAR


def project(row, target):
    """Gradients and seconds to ``target`` effective samples, reusing the row's
    own warm walls and gradient counts rather than the row's fixed-at-100
    ``grads_to_100``/``sec_to_100`` fields (equal to this at ``target=100``).
    """
    min_ess = row["min_ess"]
    scale = target / min_ess
    grads = row["n_grad_adapt"] + scale * row["n_grad_sample"]
    secs = row["adapt_wall_warm"] + row["map_wall_warm"] + scale * row["sample_wall_warm"]
    return grads, secs


def score_cell(seeds, target, budget):
    """Reduce one cell's seed rows to the worst-seed summary dict."""
    ok = [r for r in seeds if not r.get("error")]
    errored = [r for r in seeds if r.get("error")]
    n_seeds = len(seeds)
    n_error = len(errored)
    converged_flags = [_converged(r) for r in ok]
    n_converged = sum(converged_flags)
    all_converge = n_seeds > 0 and n_error == 0 and n_converged == n_seeds

    cell = dict(
        n_seeds=n_seeds,
        n_converged=n_converged,
        n_error=n_error,
        all_converge=all_converge,
        under_budget=False,
        worst_param=None,
        wall_total_warm=None,
        rhat_max=None,
        divergences=None,
        min_ess=None,
        unique_frac=None,
        grads_to_target=None,
        sec_to_target=None,
        seeds=[
            dict(
                seed=r["seed"],
                error=r.get("error"),
                wall_total_warm=r.get("wall_total_warm"),
                min_ess=r.get("min_ess"),
                worst_param=r.get("worst_param"),
                rhat_max=r.get("rhat_max"),
                divergences=r.get("divergences"),
                unique_frac=r.get("unique_frac"),
                converged=_converged(r) if not r.get("error") else False,
                z_truth_max_abs=(
                    max((abs(v) for v in r["z_truth"].values()), default=None)
                    if r.get("z_truth")
                    else None
                ),
            )
            for r in sorted(seeds, key=lambda r: r["seed"])
        ],
    )
    for field in MEDIAN_FIELDS:
        cell[field] = statistics.median(r[field] for r in ok) if ok else None
    if not ok:
        return cell

    for field in WORST_MAX_FIELDS:
        cell[field] = max(r[field] for r in ok)
    for field in WORST_MIN_FIELDS:
        cell[field] = min(r[field] for r in ok)

    # Worst seed on the projection, not necessarily the worst-R-hat seed: a
    # row can converge and still be the expensive one.
    projected = [project(r, target) for r in ok]
    cell["grads_to_target"] = max(g for g, _ in projected)
    cell["sec_to_target"] = max(s for _, s in projected)

    # Worst seed for reporting a single failing parameter: highest R-hat,
    # mirroring score_photometry_20s.py's `max(live, key=...)`.
    worst = max(ok, key=lambda r: r["rhat_max"])
    cell["worst_param"] = worst["worst_param"]

    cell["under_budget"] = all_converge and all(r["wall_total_warm"] < budget for r in ok)
    return cell


def build_cells(rows, target, budget):
    by_cell = defaultdict(list)
    for row in _dedupe(rows):
        by_cell[tuple(row.get(k) for k in CELL_KEY)].append(row)

    cells = []
    for key, seeds in by_cell.items():
        cell = dict(zip(CELL_KEY, key, strict=True))
        cell.update(score_cell(seeds, target, budget))
        cells.append(cell)
    return cells


def _fmt_wall(x):
    return f"{x:.2f}" if x is not None else "n/a"


def render_report(cells, target, budget) -> str:
    lines = []
    lines.append(f"target min ESS = {target:g}; adoption bar max split R-hat < {MAX_RHAT}")
    lines.append(f"budget under test = {budget:g} s wall (MAP + adapt + sample), warm compile\n")

    best_by_notebook_platform = {}
    for cell in cells:
        key = (cell["notebook"], cell["platform"])
        if cell["under_budget"]:
            current = best_by_notebook_platform.get(key)
            if current is None or cell["wall_total_warm"] < current["wall_total_warm"]:
                best_by_notebook_platform[key] = cell
    for nb, plat in sorted({(c["notebook"], c["platform"]) for c in cells}):
        best = best_by_notebook_platform.get((nb, plat))
        if best is None:
            lines.append(
                f"verdict [{nb}/{plat}]: none (no cell is both all-converged and under budget)"
            )
        else:
            lines.append(
                f"verdict [{nb}/{plat}]: {best['arm']} {best['n_chains']}x{best['n_samples']} "
                f"at {best['wall_total_warm']:.2f} s worst wall (under {budget:g} s budget)"
            )
    lines.append("")

    header = (
        f"| notebook | platform | arm | C×N | seeds | worst wall (s) | "
        f"med map/adapt/sample (s) | med g/draw | worst min ESS (param) | "
        f"worst R-hat | max div | min uniq | Mgrad→{target:g} (worst) | "
        f"all converge | under budget |"
    )
    sep = "|" + "---|" * 13
    lines.append(header)
    lines.append(sep)

    ordered = sorted(cells, key=lambda c: (c["notebook"], c["platform"], c["arm"], c["n_chains"]))
    for c in ordered:
        cn = f"{c['n_chains']}x{c['n_samples']}"
        # Tuning knobs that are not the default are part of the arm's identity
        # in the table, so two dense-window rows that differ only in them do not
        # read as one configuration.
        knobs = []
        if c.get("warmup_max_doublings") not in (None, 10):
            knobs.append(f"wcap={c['warmup_max_doublings']}")
        if c.get("window_start") not in (None, "map"):
            knobs.append(f"start={c['window_start']}")
        if c.get("profile_mass"):
            knobs.append("profile-mass")
        if c.get("chain_parallel") not in (None, "vmap"):
            knobs.append(str(c["chain_parallel"]))
        if c.get("dtype") not in (None, "float64"):
            knobs.append(str(c["dtype"]))
        arm = c["arm"] + (" [" + ",".join(knobs) + "]" if knobs else "")
        seeds_col = f"{c['n_seeds']}" + (f" ({c['n_error']} err)" if c["n_error"] else "")
        wall = _fmt_wall(c["wall_total_warm"])
        med = (
            f"{_fmt_wall(c['map_wall_warm'])}/{_fmt_wall(c['adapt_wall_warm'])}"
            f"/{_fmt_wall(c['sample_wall_warm'])}"
        )
        gdraw = f"{c['grad_per_draw']:.2f}" if c["grad_per_draw"] is not None else "n/a"
        min_ess = f"{c['min_ess']:.1f} ({c['worst_param']})" if c["min_ess"] is not None else "n/a"
        rhat = f"{c['rhat_max']:.4f}" if c["rhat_max"] is not None else "n/a"
        div = f"{c['divergences']}" if c["divergences"] is not None else "n/a"
        uniq = f"{c['unique_frac']:.3f}" if c["unique_frac"] is not None else "n/a"
        bound = "" if c["all_converge"] else ">="
        grads = (
            f"{bound}{c['grads_to_target']:,.0f}" if c["grads_to_target"] is not None else "n/a"
        )
        lines.append(
            f"| {c['notebook']} | {c['platform']} | {arm} | {cn} | {seeds_col} | "
            f"{wall} | {med} | {gdraw} | {min_ess} | {rhat} | {div} | {uniq} | {grads} | "
            f"{'yes' if c['all_converge'] else 'NO'} | {'yes' if c['under_budget'] else 'NO'} |"
        )

    lines.append("")
    lines.append(
        f"'all converge' = every seed cleared R-hat < {MAX_RHAT} and min ESS >= {MIN_ESS_BAR:g}. "
        "A 'NO' row's Mgrad projection is a LOWER BOUND (>=): linear-ESS-in-draws fails exactly "
        "when a chain is not mixing. An errored seed counts against 'seeds' and 'all converge' "
        "but is excluded from every worst-seed number (nothing to aggregate)."
    )

    failing = [c for c in ordered if not c["all_converge"]]
    if failing:
        lines.append("\n### Per-seed detail for cells that are not all_converge\n")
        for c in failing:
            lines.append(
                f"**{c['notebook']} / {c['platform']} / {c['arm']} "
                f"{c['n_chains']}x{c['n_samples']}**\n"
            )
            sub_header = "| seed | wall (s) | min ESS | R-hat | div | uniq | z_truth max \\|z\\| |"
            lines.append(sub_header)
            lines.append("|---|---|---|---|---|---|---|")
            for s in c["seeds"]:
                if s["error"]:
                    lines.append(f"| {s['seed']} | FAILED: {s['error']} | | | | | |")
                    continue
                z = f"{s['z_truth_max_abs']:.2f}" if s["z_truth_max_abs"] is not None else "n/a"
                lines.append(
                    f"| {s['seed']} | {_fmt_wall(s['wall_total_warm'])} | "
                    f"{s['min_ess']:.1f} | {s['rhat_max']:.4f} | {s['divergences']} | "
                    f"{s['unique_frac']:.3f} | {z} |"
                )
            lines.append("")

    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", help="JSONL files from benchmark_laplace_nuts_20s.py")
    ap.add_argument("--budget", type=float, default=20.0, help="the seconds-under-20s claim")
    ap.add_argument("--target", type=float, default=100.0, help="min ESS the projection targets")
    ap.add_argument("--md", default=None, help="also write the rendered report to this path")
    ap.add_argument("--json", default=None, help="write the scored cell dicts as a JSON array")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = load(args.paths)
    cells = build_cells(rows, args.target, args.budget)

    report = render_report(cells, args.target, args.budget)
    print(report)

    if args.md:
        with open(args.md, "w") as fh:
            fh.write(report + "\n")
        print(f"\nwrote {args.md}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(cells, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
