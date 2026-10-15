#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Surviving stellar mass for every adopted cell, recomputed from the draws.

The saved NPZ records carry ``stellar_mass``, which the registry defines as
total *formed* stellar mass (``forward/properties.py``). Four of the five
published codes this grid is compared against report *surviving* mass, and on
real cells of this grid the two differ by 0.17-0.19 dex -- comparable to the
published inter-code spread itself, so the choice is not cosmetic.

Nothing stores the surviving quantity: it has to be pushed back through the
model per draw. This script does that once for every adopted cell and records
both masses, so the comparison in Section 7 can be made like for like without
every consumer paying the recompute.

The per-draw spread on the offset is small -- fig06 measures 0.1878 dex median
with a 0.1745-0.1968 range over 200 draws on one cell -- so a few tens of
draws already pin the median. ``--max-samples`` is recorded in the output
rather than assumed by the reader.

Run from the repository root or this directory::

    python analysis/paper1/surviving_mass_census.py --max-samples 60
    python analysis/paper1/surviving_mass_census.py --suite xlike --max-samples 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

# Absolute imports on the repository root, so the file runs both as a script
# (the documented command line) and as ``analysis.paper1.surviving_mass_census``.
REPO_ROOT = ANALYSIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.paper1._figure_style import CONFIG_ORDER
from analysis.paper1._paths import repo_relative
from analysis.paper1.config_metadata import XLIKE_KEYS
from analysis.paper1.fig06_code_overlay import load_fit_results
from analysis.paper1.run_candels_fits import GALAXIES, SUITE_CONFIGS

DEFAULT_OUT = ANALYSIS_DIR / "results" / "surviving_mass_census.json"
#: Where each suite's census lands by default; fig10 reads the X-like one.
DEFAULT_OUT_BY_SUITE = {
    "grid": DEFAULT_OUT,
    "xlike": ANALYSIS_DIR / "results" / "surviving_mass_xlike.json",
}


def suite_config_keys(suite: str) -> list[str]:
    """Configuration keys the census walks for ``suite`` (``"grid"`` or ``"xlike"``).

    The grid keeps ``CONFIG_ORDER``, the figure order; X-like is the suite's own
    ``XLIKE_KEYS``.
    """
    if suite == "grid":
        return list(CONFIG_ORDER)
    if suite == "xlike":
        return list(XLIKE_KEYS)
    raise ValueError(f"unknown suite {suite!r}; expected one of {sorted(SUITE_CONFIGS)}")


def build_census(
    results_dir: Path,
    max_samples: int,
    out_path: Path,
    galaxies: list[int],
    suite: str = "grid",
) -> dict:
    """Walk every (galaxy, configuration) cell and record both stellar masses.

    A cell that ``load_fit_results`` declines -- missing on disk, or failing
    the adoption bar -- is recorded by name rather than skipped silently, so
    the output states its own coverage instead of leaving a short census
    looking complete.
    """
    cells: dict[str, dict] = {}
    declined: list[str] = []
    started = time.time()
    config_keys = suite_config_keys(suite)

    for gal_id in galaxies:
        for config in config_keys:
            key = f"{gal_id}_{config}"
            data = load_fit_results(gal_id, config, results_dir, max_samples)
            if data is None:
                declined.append(key)
                continue
            formed = np.asarray(data.mass_formed, dtype=float)
            survived = np.asarray(data.mass_survived, dtype=float)
            offset = formed - survived
            cells[key] = {
                "galaxy": int(gal_id),
                "config": config,
                "n_draws": int(formed.shape[0]),
                "log_mass_formed_p50": float(np.median(formed)),
                "log_mass_survived_p16": float(np.percentile(survived, 16)),
                "log_mass_survived_p50": float(np.median(survived)),
                "log_mass_survived_p84": float(np.percentile(survived, 84)),
                "offset_dex_p50": float(np.median(offset)),
                "offset_dex_min": float(offset.min()),
                "offset_dex_max": float(offset.max()),
            }
            print(
                f"{key}: formed {cells[key]['log_mass_formed_p50']:.4f}  "
                f"survived {cells[key]['log_mass_survived_p50']:.4f}  "
                f"offset {cells[key]['offset_dex_p50']:.4f} dex",
                flush=True,
            )

    offsets = [c["offset_dex_p50"] for c in cells.values()]
    payload = {
        "max_samples_requested": max_samples,
        "results_dir": repo_relative(results_dir),
        "suite": suite,
        "configurations": config_keys,
        "galaxies": [int(g) for g in galaxies],
        "n_galaxies": len(galaxies),
        "n_cells_recorded": len(cells),
        "n_cells_declined": len(declined),
        "declined_cells": declined,
        "offset_dex_median": float(np.median(offsets)) if offsets else None,
        "offset_dex_min": float(np.min(offsets)) if offsets else None,
        "offset_dex_max": float(np.max(offsets)) if offsets else None,
        "elapsed_s": round(time.time() - started, 1),
        "cells": cells,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def merge_shards(paths: list[Path], out_path: Path) -> dict:
    """Combine per-shard censuses into the single file consumers read.

    A shard that names a cell another shard already recorded is a bug in how
    the run was split, not something to silently resolve, so the overlap is
    refused by name rather than last-writer-wins.
    """
    cells: dict[str, dict] = {}
    galaxies: list[int] = []
    declined: list[str] = []
    requested: set[int] = set()
    suites: set[str] = set()
    configurations: list[str] = []
    for path in paths:
        shard = json.loads(path.read_text())
        clash = sorted(set(shard["cells"]) & set(cells))
        if clash:
            raise SystemExit(f"{path} re-records cell(s) {clash}; shards must be disjoint")
        cells.update(shard["cells"])
        galaxies.extend(shard.get("galaxies", []))
        declined.extend(shard.get("declined_cells", []))
        requested.add(int(shard["max_samples_requested"]))
        suites.add(shard.get("suite", "grid"))
        configurations.extend(
            c for c in shard.get("configurations", []) if c not in configurations
        )

    offsets = [c["offset_dex_p50"] for c in cells.values()]
    payload = {
        "max_samples_requested": sorted(requested)[0]
        if len(requested) == 1
        else sorted(requested),
        "merged_from": [repo_relative(x) for x in paths],
        "suite": sorted(suites)[0] if len(suites) == 1 else sorted(suites),
        "configurations": configurations,
        "galaxies": sorted(set(galaxies)),
        "n_galaxies": len(set(galaxies)),
        "n_cells_recorded": len(cells),
        "n_cells_declined": len(declined),
        "declined_cells": sorted(declined),
        "offset_dex_median": float(np.median(offsets)) if offsets else None,
        "offset_dex_min": float(np.min(offsets)) if offsets else None,
        "offset_dex_max": float(np.max(offsets)) if offsets else None,
        "cells": cells,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="fit cells to read; default is the chosen suite's results directory",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_CONFIGS),
        default="grid",
        help="configuration suite to walk: the six grid configurations or the five X-like ones",
    )
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="census JSON to write; default is the chosen suite's file (DEFAULT_OUT_BY_SUITE)",
    )
    parser.add_argument(
        "--galaxies",
        type=str,
        default="",
        help=(
            "comma-separated galaxy IDs; default is every galaxy in the grid. "
            "Shard a long run across processes by giving each a disjoint set, "
            "then merge the per-shard JSONs."
        ),
    )
    parser.add_argument(
        "--merge",
        type=str,
        default="",
        help="comma-separated shard JSONs to combine into --out, instead of computing",
    )
    args = parser.parse_args(argv)
    if args.results_dir is None:
        args.results_dir = SUITE_CONFIGS[args.suite].get_results_dir(ANALYSIS_DIR)
    if args.out is None:
        args.out = DEFAULT_OUT_BY_SUITE[args.suite]

    if args.merge:
        paths = [Path(x) for x in args.merge.split(",") if x.strip()]
        payload = merge_shards(paths, args.out)
        print(
            f"merged {len(paths)} shard(s): {payload['n_cells_recorded']} cell(s) "
            f"over {payload['n_galaxies']} galaxies -> {args.out}"
        )
        print(
            f"formed - survived: median {payload['offset_dex_median']:.4f} dex "
            f"(min {payload['offset_dex_min']:.4f}, max {payload['offset_dex_max']:.4f})"
        )
        return 0

    if args.galaxies:
        wanted = [int(g) for g in args.galaxies.split(",") if g.strip()]
        unknown = sorted(set(wanted) - set(int(g) for g in GALAXIES))
        if unknown:
            raise SystemExit(f"not in the grid: {unknown}")
        galaxies = wanted
    else:
        galaxies = [int(g) for g in GALAXIES]

    payload = build_census(args.results_dir, args.max_samples, args.out, galaxies, args.suite)
    print(
        f"\nrecorded {payload['n_cells_recorded']} cell(s), declined {payload['n_cells_declined']}"
    )
    if payload["offset_dex_median"] is not None:
        print(
            f"formed - survived: median {payload['offset_dex_median']:.4f} dex "
            f"(min {payload['offset_dex_min']:.4f}, max {payload['offset_dex_max']:.4f})"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
