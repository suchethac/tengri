# SPDX-License-Identifier: BSD-3-Clause
"""Section 7's grid-wide numbers, computed from the cells rather than by hand.

Section 7 carries nine ``\\confirm{TBD}`` entries and every one of them is a
function of the cell diagnostics on disk: how many cells were adopted, the
retune rung each cleared on, the wall-time range, the s/ESS range, the minimum
effective sample size, and which configurations mix worst. Writing them out by
reading a directory is how a number gets transcribed wrong once and then
quoted forever.

Two things this refuses to do.

**It will not report a grid-wide figure from part of a grid.** "Which
configurations mix worst" needs more than one configuration by construction,
and an adoption count over one row of six is not the grid's adoption count. A
partial directory gets per-configuration rows and an explicit refusal on the
aggregates, because a plausible number computed from a sixth of the data is
worse than no number: it is quotable.

**It will not paper over two meanings of ``adoption_pass``.** On the paper's
pinned branch ``fit_one`` computes it as ``n_divergent == 0 and rhat_max <
1.01``; the grid session's driver adds ``ess_min >= 100`` (03ffff576, on
``paper1/nss-profile-mass``). A cell written by the older driver and one
written by the newer carry the same key meaning different predicates, so the
census reports the split and names the cells rather than summing them as
though they agreed.

CLI::

    python -m paper1.grid_summary [--results DIR] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_DIR = RESULTS / "fits"
DEFAULT_OUT = RESULTS / "grid_summary.json"

#: The grid the demonstration claims: twenty galaxies, six configurations.
EXPECTED_CONFIGS = 6
EXPECTED_GALAXIES = 20

#: Present only in cells written after the ESS leg joined the adoption bar.
THREE_LEG_MARKER = "priors"
#: Present only after the driver began recording what produced the cell.
REVISION_KEY = "code_revision"


class Missing(SystemExit):
    """A required diagnostic is absent. Absent is never treated as a value."""


def _require(cell: dict, name: str, key: str):
    if key not in cell or cell[key] is None:
        raise Missing(
            f"{name} records no {key!r}. This summary will not substitute a "
            f"value for it; the cell has {sorted(cell)}"
        )
    return cell[key]


def load_cells(results_dir: Path) -> dict[str, dict]:
    """Every cell JSON in the directory, keyed ``<gal>_<config>``."""
    if not results_dir.is_dir():
        raise SystemExit(f"no results directory at {results_dir}")
    cells = {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))}
    if not cells:
        raise SystemExit(
            f"{results_dir} holds no cell JSON. An empty grid is not a grid with "
            "nothing to say about it; it is a grid that has not run."
        )
    return cells


def per_cell(name: str, cell: dict) -> dict:
    """The quantities Section 7 quotes, for one cell."""
    ess = float(_require(cell, name, "ess_min"))
    wall = float(_require(cell, name, "wall_time_s"))
    if ess <= 0:
        raise Missing(f"{name} reports ess_min {ess}; seconds per effective sample is undefined")
    return {
        "config": _require(cell, name, "config"),
        "gal_id": _require(cell, name, "gal_id"),
        "adopted": bool(_require(cell, name, "adoption_pass")),
        "ess_min": ess,
        "rhat_max": float(_require(cell, name, "rhat_max")),
        "divergences": int(_require(cell, name, "divergences")),
        "wall_s": wall,
        "s_per_ess": wall / ess,
        # Which rung the kept attempt came from. 1 means it cleared first try.
        "rung": int(cell.get("retune_attempt", 1)),
        "three_leg": THREE_LEG_MARKER in cell,
        "revision": cell.get(REVISION_KEY),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate, refusing the grid-wide figures when the grid is partial."""
    configs = sorted({r["config"] for r in rows})
    galaxies = sorted({r["gal_id"] for r in rows})
    complete = len(configs) >= EXPECTED_CONFIGS and len(galaxies) >= EXPECTED_GALAXIES

    adopted = [r for r in rows if r["adopted"]]
    by_config = {}
    for config in configs:
        sub = [r for r in rows if r["config"] == config]
        sub_adopted = [r for r in sub if r["adopted"]]
        by_config[config] = {
            "cells": len(sub),
            "adopted": len(sub_adopted),
            "ess_min_worst": min((r["ess_min"] for r in sub_adopted), default=None),
            "ess_min_median": (
                statistics.median([r["ess_min"] for r in sub_adopted]) if sub_adopted else None
            ),
            "rhat_max_worst": max((r["rhat_max"] for r in sub), default=None),
            "wall_median_s": statistics.median([r["wall_s"] for r in sub]),
        }

    two_leg = [r["gal_id"] for r in rows if not r["three_leg"]]
    three_leg = [r["gal_id"] for r in rows if r["three_leg"]]

    summary: dict = {
        "n_cells": len(rows),
        "configs_present": configs,
        "galaxies_present": len(galaxies),
        "grid_complete": complete,
        "adoption_bar_split": {
            "two_leg_cells": len(two_leg),
            "three_leg_cells": len(three_leg),
            "three_leg_galaxies": sorted(three_leg),
            "note": (
                "adoption_pass is n_divergent == 0 and rhat_max < 1.01 on the "
                "paper's pin, plus ess_min >= 100 on the grid session's driver "
                "(03ffff576). Cells of the two kinds are not summable without "
                "saying which is which."
            ),
        },
        "per_configuration": by_config,
    }

    # The aggregates Section 7 quotes. Withheld unless the grid is whole.
    aggregate_keys = (
        "n_adopted",
        "wall_min_s",
        "wall_max_s",
        "wall_median_s",
        "s_per_ess_min",
        "s_per_ess_max",
        "ess_min",
        "mixes_worst",
        "rung_distribution",
    )
    if not complete:
        summary["aggregates"] = None
        summary["aggregates_withheld_because"] = (
            f"{len(configs)} of {EXPECTED_CONFIGS} configurations and "
            f"{len(galaxies)} of {EXPECTED_GALAXIES} galaxies are present. "
            f"Section 7 quotes {', '.join(aggregate_keys)} of the whole grid; "
            "computed here they would be a sixth of it wearing the grid's name."
        )
        return summary

    rungs: dict[int, int] = {}
    for r in adopted:
        rungs[r["rung"]] = rungs.get(r["rung"], 0) + 1
    worst = min(by_config, key=lambda c: by_config[c]["ess_min_worst"] or float("inf"))
    summary["aggregates"] = {
        "n_adopted": len(adopted),
        "n_cells": len(rows),
        "wall_min_s": min(r["wall_s"] for r in rows),
        "wall_max_s": max(r["wall_s"] for r in rows),
        "wall_median_s": statistics.median([r["wall_s"] for r in rows]),
        "s_per_ess_min": min(r["s_per_ess"] for r in adopted),
        "s_per_ess_max": max(r["s_per_ess"] for r in adopted),
        "ess_min": min(r["ess_min"] for r in adopted),
        "mixes_worst": worst,
        "rung_distribution": {str(k): v for k, v in sorted(rungs.items())},
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cells = load_cells(args.results)
    rows = [per_cell(name, cell) for name, cell in cells.items()]
    summary = summarize(rows)

    print(
        f"{'config':<8}{'cells':>7}{'adopted':>9}{'worst ESS':>11}{'worst rhat':>12}{'med wall s':>12}"
    )
    print("-" * 59)
    for config, row in summary["per_configuration"].items():
        we = "-" if row["ess_min_worst"] is None else f"{row['ess_min_worst']:.1f}"
        wr = "-" if row["rhat_max_worst"] is None else f"{row['rhat_max_worst']:.4f}"
        print(
            f"{config:<8}{row['cells']:>7}{row['adopted']:>9}{we:>11}{wr:>12}"
            f"{row['wall_median_s']:>12.0f}"
        )

    split = summary["adoption_bar_split"]
    print(
        f"\nadoption bar: {split['two_leg_cells']} two-leg, {split['three_leg_cells']} three-leg"
    )

    if summary["aggregates"] is None:
        print(f"\ngrid-wide figures withheld: {summary['aggregates_withheld_because']}")
    else:
        for key, value in summary["aggregates"].items():
            print(f"  {key:<20} {value}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
