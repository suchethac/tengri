#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Every number Section 7 asks of the grid, computed from the cells.

Section 7 carries seventeen ``\\confirm{TBD}`` entries, each a statistic over
the finished grid, each with a source comment naming the field it must be read
from. Filling them by hand is seventeen chances to read one cell's value as
the sample's, and the paper has already been wrong that way: it named a
machine nobody had measured and described a two-legged adoption bar the grid
applies three legs of.

So the numbers come from here instead, and this script holds the rules the
hand-fill kept losing:

**Absent is not zero.** A field a cell does not carry is reported as not
recorded. ``meta.get("divergences") or 0`` once turned a missing diagnostic
into a clean bill of health, and ``rhat_max: None`` made every mock fit look
converged. Every quantity below states its own coverage.

**A partial grid cannot answer a question about the whole one.** Nineteen
adopted cells of one configuration say nothing about how six configurations
compare, so an incomplete grid exits non-zero and every figure is labeled with
what it was computed over. ``--allow-partial`` is for watching a run, not for
filling the paper.

**The grid is not uniform in what it records.** Row III predates 837ac7663 and
carries no seed; later rows carry ``seed`` and ``prng_key_seed``. Both cases
are held at once rather than averaged into a sentence true of neither.

CLI:
    python analysis/paper1/grid_census.py [--results-dir DIR] [--allow-partial]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _adoption import is_adopted
from _figure_style import CONFIG_ORDER
from _grid_completeness import load_expected_galaxy_ids

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = ANALYSIS_DIR / "results" / "fits"
SELECTION_20 = ANALYSIS_DIR / "results" / "selected_galaxies_20.json"

#: Field a cell may or may not carry, depending on when it was run.
NOT_RECORDED = "not recorded"


def load_cells(results_dir: Path) -> dict[str, dict]:
    """Every cell on disk, keyed ``<galaxy>_<config>``."""
    cells = {}
    for path in sorted(results_dir.glob("*_*.json")):
        try:
            cells[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARNING: {path.name} unreadable ({exc}); excluded", file=sys.stderr)
    return cells


def coverage(cells: dict[str, dict], field: str) -> tuple[list, int]:
    """Values present for ``field``, and how many cells lack it.

    Returned rather than defaulted, so a caller cannot average over a field
    half the grid does not carry without seeing that it did.
    """
    present, missing = [], 0
    for cell in cells.values():
        value = cell.get(field)
        if value is None:
            missing += 1
        else:
            present.append(value)
    return present, missing


def _fmt(values, spec=".4g"):
    if not values:
        return NOT_RECORDED
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"{lo:{spec}}"
    return f"{lo:{spec}} to {hi:{spec}}"


def report(cells: dict[str, dict], expected_ids, config_keys) -> bool:
    """Print the census. Returns True when the grid is complete."""
    total = len(expected_ids) * len(config_keys)
    have = len(cells)
    by_config = defaultdict(dict)
    for name, cell in cells.items():
        by_config[cell.get("config") or name.rsplit("_", 1)[-1]][name] = cell

    print(f"cells present        : {have} of {total}")
    print(f"configurations seen  : {sorted(by_config)}")
    complete = have == total
    if not complete:
        print(f"\n*** PARTIAL GRID -- {have} of {total} cells. Every figure below is")
        print("*** computed over what is present and is NOT a statement about the")
        print("*** grid. Do not fill Section 7 from a partial run.\n")

    # is_adopted judges per configuration -- the relaxed bar applies to some
    # rows and not others -- so the config is read from the cell, with the
    # filename as the fallback for a cell that does not record one.
    def _config_of(name: str, cell: dict) -> str:
        return cell.get("config") or name.rsplit("_", 1)[-1]

    adopted = {k: c for k, c in cells.items() if is_adopted(c, _config_of(k, c)).adopted}
    refused = {k: c for k, c in cells.items() if k not in adopted}
    print(f"\nadopted              : {len(adopted)} of {have}")
    if refused:
        print(f"not adopted          : {sorted(refused)}")

    # --- the adoption bar's three legs, over the adopted set ----------------
    for field, spec in (("divergences", ".0f"), ("rhat_max", ".4f"), ("ess_min", ".1f")):
        values, missing = coverage(adopted, field)
        note = f"  ({missing} adopted cell(s) do not record it)" if missing else ""
        print(f"adopted {field:<13}: {_fmt(values, spec)}{note}")

    # --- retune ladder ------------------------------------------------------
    rungs = defaultdict(int)
    unknown_rung = 0
    for cell in adopted.values():
        rung = cell.get("retune_attempt")
        if rung is None:
            unknown_rung += 1
        else:
            rungs[int(rung)] += 1
    print(
        f"adopted by rung      : {dict(sorted(rungs.items()))}"
        + (f"  ({unknown_rung} without a rung recorded)" if unknown_rung else "")
    )

    # --- why the refusals were refused -------------------------------------
    # "Of the non-adopted cells, how many fail on divergences alone while
    # meeting both other legs" is a TBD in its own right, and the answer is
    # different from "how many had divergences".
    div_only = [
        k
        for k, c in refused.items()
        if c.get("divergences") not in (None, 0)
        and c.get("rhat_max") is not None
        and c["rhat_max"] <= 1.01
        and c.get("ess_min") is not None
        and c["ess_min"] >= 100
    ]
    print(
        f"refused on divergences alone : {len(div_only)}{' -> ' + str(sorted(div_only)) if div_only else ''}"
    )

    # --- cost, and the contention it was measured under --------------------
    walls, wall_missing = coverage(adopted, "wall_time_s")
    print(
        f"\nadopted wall_time_s  : {_fmt(walls, '.0f')}"
        + (f"  (median {statistics.median(walls):.0f})" if walls else "")
        + (f"  ({wall_missing} not recorded)" if wall_missing else "")
    )
    s_per_ess = [
        c["wall_time_s"] / c["ess_min"]
        for c in adopted.values()
        if c.get("wall_time_s") and c.get("ess_min")
    ]
    print(
        f"adopted s per ESS    : {_fmt(s_per_ess, '.2f')}  "
        f"(from {len(s_per_ess)} of {len(adopted)} adopted cells)"
    )

    cpus, loads, concurrent = set(), [], set()
    for cell in cells.values():
        for where in ("load_at_start", "load_at_end"):
            load = cell.get(where) or {}
            if load.get("n_cpus") is not None:
                cpus.add(int(load["n_cpus"]))
            if load.get("load_avg_1m") is not None:
                loads.append(float(load["load_avg_1m"]))
            if load.get("n_concurrent_fits") is not None:
                concurrent.add(int(load["n_concurrent_fits"]))
    print(f"machine n_cpus       : {sorted(cpus) or NOT_RECORDED}")
    print(f"concurrent fits      : {sorted(concurrent) or NOT_RECORDED}")
    if loads and cpus:
        n = max(cpus)
        print(
            f"load average         : {min(loads):.1f} to {max(loads):.1f} on {n} CPUs "
            f"({min(loads) / n:.2f}x to {max(loads) / n:.2f}x oversubscribed)"
        )
        print("  ^ wall and s/ESS above are contended; quote them with this or not at all.")
    print('  ^ no hostname or CPU model is recorded; say "N-core", do not name a machine.')

    # --- per-configuration facts -------------------------------------------
    print("\nper configuration:")
    for config in sorted(
        by_config, key=lambda c: config_keys.index(c) if c in config_keys else 99
    ):
        group = by_config[config]
        adopted_here = {k: c for k, c in group.items() if k in adopted}
        ess, _ = coverage(adopted_here, "ess_min")
        profiled = {c.get("profile_mass") for c in group.values()}
        print(
            f"  {config:<4} cells {len(group):>3}  adopted {len(adopted_here):>3}  "
            f"ess_min {_fmt(ess, '.1f'):<20} profile_mass {sorted(str(p) for p in profiled)}"
        )
    if not complete:
        print("  ^ 'which configurations mix worst' needs every row; this is not that.")

    # --- seed provenance, which the grid is not uniform about --------------
    seeds, seed_missing = coverage(cells, "seed")
    keys, key_missing = coverage(cells, "prng_key_seed")
    print(
        f"\nseed recorded        : {len(seeds)} of {have} cells"
        f"{'  values ' + str(sorted(set(seeds))) if seeds else ''}"
    )
    print(
        f"prng_key_seed        : {len(keys)} of {have} cells"
        f"{'  values ' + str(sorted(set(keys))) if keys else ''}"
    )
    if seed_missing or key_missing:
        print("  ^ cells without these predate 837ac7663. For them the key is")
        print("    PRNGKey(seed + attempt), attempt 1-based: rung 1 -> 43, 2 -> 44, 3 -> 45.")
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--selection", type=Path, default=SELECTION_20)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 on an incomplete grid; for watching a run, not for filling the paper",
    )
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"no results directory at {args.results_dir}", file=sys.stderr)
        return 2
    cells = load_cells(args.results_dir)
    if not cells:
        print(f"no cells in {args.results_dir}", file=sys.stderr)
        return 2
    try:
        expected_ids = load_expected_galaxy_ids(args.selection)
    except (OSError, ValueError, KeyError) as exc:
        print(f"cannot read the locked sample at {args.selection}: {exc}", file=sys.stderr)
        return 2

    complete = report(cells, expected_ids, list(CONFIG_ORDER))
    if not complete and not args.allow_partial:
        print(
            "\nexiting non-zero: the grid is incomplete, so these are not its numbers.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
