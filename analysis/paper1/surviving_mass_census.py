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

from ._figure_style import CONFIG_ORDER
from .fig06_code_overlay import load_fit_results
from .run_candels_fits import GALAXIES

DEFAULT_RESULTS_DIR = ANALYSIS_DIR / "results" / "fits"
DEFAULT_OUT = ANALYSIS_DIR / "results" / "surviving_mass_census.json"


def build_census(results_dir: Path, max_samples: int, out_path: Path) -> dict:
    """Walk every (galaxy, configuration) cell and record both stellar masses.

    A cell that ``load_fit_results`` declines -- missing on disk, or failing
    the adoption bar -- is recorded by name rather than skipped silently, so
    the output states its own coverage instead of leaving a short census
    looking complete.
    """
    cells: dict[str, dict] = {}
    declined: list[str] = []
    started = time.time()

    for gal_id in GALAXIES:
        for config in CONFIG_ORDER:
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
        "results_dir": str(results_dir),
        "configurations": list(CONFIG_ORDER),
        "n_galaxies": len(GALAXIES),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    payload = build_census(args.results_dir, args.max_samples, args.out)
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
