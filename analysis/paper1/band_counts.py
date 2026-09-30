#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""How many bands each locked galaxy is actually fitted with.

Section 7 quotes a band-count range. The obvious source is the catalog's own
detection count, and it is the wrong one: it disagrees with the fits for four
of the twenty galaxies, always by exactly one.

The catalog carries **two** Ks measurements, ISAAC and HAWK-I. Both can be
detected for the same galaxy, and they are two measurements of one band, not
two bands. ``candels_io.photometry_for_row`` takes the first and skips the
second (``KS_COLUMNS`` / ``ks_taken``), so a galaxy with both is fitted with
one fewer band than the catalog reports as detected. Quoting the catalog
number would overstate the information every such fit actually had.

This reads the band list through the same function the fitter uses, so the
printed range is what was fitted rather than what was available, and it covers
all twenty locked galaxies rather than only the cells that have finished.

CLI: python -m analysis.paper1.band_counts [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .candels_io import KS_COLUMNS, is_detected, load_candels_z1, photometry_for_row

SELECTION = Path(__file__).with_name("results") / "selected_galaxies_20.json"


def locked_galaxy_ids(path: Path = SELECTION) -> list[int]:
    payload = json.loads(Path(path).read_text())
    return [int(e["id"]) for e in payload["selected_galaxies"]]


def band_counts(galaxy_ids: list[int]) -> dict[int, int]:
    """Bands each galaxy is fitted with, via the fitter's own selection."""
    data = load_candels_z1()
    header, rows = data["header"], data["data"]
    ids = np.asarray([int(r[header.index("ID")]) for r in rows])
    counts: dict[int, int] = {}
    for gid in galaxy_ids:
        where = np.where(ids == gid)[0]
        if not where.size:
            raise KeyError(f"galaxy {gid} is not in the catalog")
        names, _, _ = photometry_for_row(header, rows[int(where[0])])
        counts[gid] = len(names)
    return counts


def duplicate_ks(galaxy_ids: list[int]) -> dict[int, int]:
    """How many of the two Ks columns are detected, per galaxy."""
    data = load_candels_z1()
    header, rows = data["header"], data["data"]
    ids = np.asarray([int(r[header.index("ID")]) for r in rows])
    out: dict[int, int] = {}
    for gid in galaxy_ids:
        row = rows[int(np.where(ids == gid)[0][0])]
        out[gid] = sum(
            1
            for c in KS_COLUMNS
            if c in header
            and is_detected(float(row[header.index(c)]), float(row[header.index("e" + c)]))
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).with_name("results") / "fits"
    )
    args = parser.parse_args(argv)

    ids = locked_galaxy_ids()
    counts = band_counts(ids)
    ks = duplicate_ks(ids)
    values = sorted(counts.values())

    print(f"{'galaxy':>8} {'bands':>6} {'ks cols detected':>18}")
    for gid in ids:
        print(f"{gid:>8} {counts[gid]:>6} {ks[gid]:>18}")
    print(f"\nrange over {len(ids)} locked galaxies: {min(values)} to {max(values)}")
    print(f"median: {int(np.median(values))}")
    print(f"distribution: {dict(sorted(Counter(values).items()))}")
    print(f"galaxies with both Ks columns detected: {sum(1 for v in ks.values() if v > 1)}")

    # Any finished cell must agree exactly; a mismatch means the fitter and
    # this script disagree about what was fitted, which would invalidate both.
    mismatches = []
    for path in sorted(args.results_dir.glob("*_*.json")):
        try:
            cell = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        gid = int(path.stem.split("_")[0])
        if gid in counts and cell.get("n_bands") is not None and cell["n_bands"] != counts[gid]:
            mismatches.append((path.stem, cell["n_bands"], counts[gid]))
    if mismatches:
        print("\nMISMATCH against finished cells (cell, fitted, computed):")
        for row in mismatches:
            print(f"  {row}")
        return 1
    print("\nevery finished cell agrees with this computation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
