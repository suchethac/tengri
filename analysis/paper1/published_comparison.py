#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""How many adopted cells land inside the published codes' range.

Section 7 quotes this count. It had no committed producer -- it was computed
once, by hand, which is how it came to be computed on the wrong quantity.

The quantity matters. ``fit_one`` saves ``stellar_mass``, which the registry
defines as total *formed* stellar mass; four of the five published codes report
*surviving* mass (``published_code_spread.SURVIVED_MASS_CODES``). Measured on
real cells of this grid the two differ by 0.17-0.19 dex, which is larger than
the +0.026 dex offset between Prospector and the survived-mass codes that the
text cites as the definitional correction. So ``--mass surviving`` is the
like-for-like comparison against that subset, and ``--mass formed`` is the
like-for-like comparison against Prospector alone.

``--mass surviving`` reads ``results/surviving_mass_census.json``, which
``surviving_mass_census.py`` writes, because the saved fit records do not carry
the surviving quantity.

Run::

    python -m paper1.published_comparison --mass formed
    python -m paper1.published_comparison --mass surviving
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from ._adoption import is_adopted
from ._figure_style import CONFIG_ORDER
from .published_code_spread import CATALOG, SURVIVED_MASS_CODES, load
from .run_candels_fits import GALAXIES

RESULTS_DIR = ANALYSIS_DIR / "results" / "fits"
CENSUS_PATH = ANALYSIS_DIR / "results" / "surviving_mass_census.json"


def _adopted_cells(results_dir: Path) -> dict[str, dict]:
    """Every (galaxy, configuration) cell that clears its adoption bar."""
    cells = {}
    for gal_id in GALAXIES:
        for config in CONFIG_ORDER:
            meta_path = results_dir / f"{gal_id}_{config}.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            if not is_adopted(meta, config).adopted:
                continue
            cells[f"{gal_id}_{config}"] = {"galaxy": int(gal_id), "config": config}
    return cells


def _formed_mass(results_dir: Path, key: str) -> float:
    npz = np.load(results_dir / f"{key}.npz")
    return float(np.median(np.log10(np.asarray(npz["stellar_mass"], dtype=float))))


def _surviving_mass(census: dict, results_dir: Path, key: str) -> float | None:
    """Surviving mass at full-chain precision, via the offset rather than directly.

    The census subsamples a few tens of draws per cell, because rebuilding the
    model is what costs. Taking its surviving median directly would set a
    40-draw median against the formed arm's full-chain median, and the
    difference in sampling noise would enter the range test as though it were
    physics.

    The *offset* does not have that problem: it is a difference of two
    quantities computed from the same draws, and it barely moves across them --
    fig06 measures 0.1878 dex median over a 0.1745-0.1968 range on 200 draws.
    So the offset is taken from the subsample and applied to the full-chain
    formed median, which keeps both arms at the same precision.
    """
    cell = census["cells"].get(key)
    if cell is None:
        return None
    return _formed_mass(results_dir, key) - float(cell["offset_dex_p50"])


def compare(mass: str, results_dir: Path, catalog: Path) -> dict:
    per_galaxy, _, _ = load(catalog)
    cells = _adopted_cells(results_dir)

    census = None
    if mass == "surviving":
        if not CENSUS_PATH.is_file():
            raise SystemExit(
                f"{CENSUS_PATH} is missing -- run surviving_mass_census.py first. "
                "The saved fit records carry only the formed mass."
            )
        census = json.loads(CENSUS_PATH.read_text())

    counts = {"all_codes": 0, "survived_codes": 0}
    considered, missing = 0, []

    for key, cell in sorted(cells.items()):
        value = (
            _formed_mass(results_dir, key)
            if mass == "formed"
            else _surviving_mass(census, results_dir, key)
        )
        if value is None:
            missing.append(key)
            continue
        published = per_galaxy.get(cell["galaxy"]) or per_galaxy.get(str(cell["galaxy"]))
        if not published:
            missing.append(key)
            continue
        considered += 1

        all_vals = [v[0] for v in published.values() if np.isfinite(v[0])]
        surv_vals = [
            v[0]
            for code, v in published.items()
            if code in SURVIVED_MASS_CODES and np.isfinite(v[0])
        ]
        if len(all_vals) >= 2 and min(all_vals) <= value <= max(all_vals):
            counts["all_codes"] += 1
        if len(surv_vals) >= 2 and min(surv_vals) <= value <= max(surv_vals):
            counts["survived_codes"] += 1

    return {
        "mass_quantity": mass,
        "n_adopted": len(cells),
        "n_considered": considered,
        "n_missing": len(missing),
        "missing_cells": missing,
        "inside_all_codes": counts["all_codes"],
        "inside_survived_codes": counts["survived_codes"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", choices=("formed", "surviving"), default="formed")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args(argv)

    r = compare(args.mass, args.results_dir, args.catalog)
    print(f"mass quantity        : tengri {r['mass_quantity']} stellar mass")
    print(f"adopted cells        : {r['n_adopted']}")
    print(f"cells compared       : {r['n_considered']}  (missing {r['n_missing']})")
    if r["missing_cells"]:
        print(f"  missing            : {', '.join(r['missing_cells'][:8])}")
    n = r["n_considered"] or 1
    print(
        f"inside all 5 codes   : {r['inside_all_codes']} of {r['n_considered']}"
        f"  ({100 * r['inside_all_codes'] / n:.0f} per cent)"
    )
    print(
        f"inside survived-only : {r['inside_survived_codes']} of {r['n_considered']}"
        f"  ({100 * r['inside_survived_codes'] / n:.0f} per cent)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
