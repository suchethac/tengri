# SPDX-License-Identifier: BSD-3-Clause
"""The published inter-code spread, measured from the catalog rather than quoted.

Section 7 compares tengri's configurations against the spread across the codes
in Pacifici et al. 2023. That comparison has two ways to go wrong quietly, and
both are checked here rather than assumed.

**The statistic.** The paper quotes about 0.1 dex in stellar mass and about 0.3
dex in star formation rate. Those are the STANDARD DEVIATION across codes, not
the range: measured on this catalog the median across-code stdev is 0.111 and
0.295 dex, while the median range is 0.268 and 0.725. Comparing a range against
a scatter overstates by a factor of about 2.5 and would make "inside the
community spread" either trivially true or impossible depending on which way
round the mistake went. Both are reported below so the text can name which it
means.

**The definitions.** The codes do not all report the same quantity. Prospector
reports formed stellar mass while the other four report survived; BAGPIPES
reports instantaneous SFR while the other four report a 100 Myr average. Those
are different physical quantities, so part of any "inter-code spread" is
definitional. Measured, the mass effect is small (+0.026 dex) but the SFR one is
not: dropping the instantaneous code takes the 90th-percentile range from 4.57
to 1.91 dex, so the disagreement lives in the tail rather than the median.

tengri's own ``stellar_mass`` is FORMED mass (``prediction.py``: "Total formed
stellar mass"), with ``stellar_mass_surviving`` as a separate property, and
``fit_one`` saves the formed one. So a like-for-like comparison against the four
survived-mass codes needs either the surviving property or an explicit statement
of which quantity the figure plots.

Run::

    python -m paper1.published_code_spread
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CATALOG = Path(__file__).parent / "results" / "art_sedfitting_z1.csv"

#: Codes reporting survived stellar mass; Prospector reports formed.
SURVIVED_MASS_CODES = frozenset({"BAGPIPES", "BEAGLE", "CIGALE", "Dense_Basis"})

#: Codes reporting a 100 Myr averaged SFR; BAGPIPES reports instantaneous.
AVERAGED_SFR_CODES = frozenset({"BEAGLE", "CIGALE", "Dense_Basis", "Prospector"})


def load(path: Path):
    """Per-galaxy {code: (log_mstar, log_sfr)}, plus the declared definitions."""
    per_galaxy: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    mass_notes: dict[str, set[str]] = defaultdict(set)
    sfr_notes: dict[str, set[str]] = defaultdict(set)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            mass_notes[row["mass_definition_note"]].add(row["code"])
            sfr_notes[row["sfr_timescale_note"]].add(row["code"])
            try:
                per_galaxy[row["id"]][row["code"]] = (
                    float(row["logmstar"]),
                    float(row["logsfr"]),
                )
            except (ValueError, KeyError):
                continue
    return per_galaxy, mass_notes, sfr_notes


def spread_stats(per_galaxy, codes, index, min_codes):
    """Median range and median across-code stdev, over galaxies with enough codes."""
    ranges, stdevs = [], []
    for values in per_galaxy.values():
        present = [values[c][index] for c in codes if c in values]
        if len(present) >= min_codes:
            ranges.append(max(present) - min(present))
            stdevs.append(statistics.stdev(present))
    if not ranges:
        return None
    return {
        "n_galaxies": len(ranges),
        "median_range": statistics.median(ranges),
        "median_stdev": statistics.median(stdevs),
        "p90_range": sorted(ranges)[int(0.9 * len(ranges))],
    }


def offset_against(per_galaxy, code, reference_codes, index, min_ref):
    """Median (code - median of reference codes), the size of a definitional shift."""
    deltas = []
    for values in per_galaxy.values():
        if code not in values:
            continue
        reference = [values[c][index] for c in reference_codes if c in values]
        if len(reference) >= min_ref:
            deltas.append(values[code][index] - statistics.median(reference))
    return (statistics.median(deltas), len(deltas)) if deltas else None


def main() -> int:
    if not CATALOG.exists():
        print(f"no catalog at {CATALOG}")
        return 1
    per_galaxy, mass_notes, sfr_notes = load(CATALOG)
    all_codes = frozenset(c for v in per_galaxy.values() for c in v)
    print(f"catalog: {CATALOG.name}, {len(per_galaxy)} galaxies, codes {sorted(all_codes)}\n")

    print("declared definitions (part of any spread is this, not physics):")
    for note, codes in sorted(mass_notes.items()):
        print(f"   mass  {note!r:<26} {sorted(codes)}")
    for note, codes in sorted(sfr_notes.items()):
        print(f"   sfr   {note!r:<26} {sorted(codes)}")

    print(f"\n{'quantity':<34}{'n':>5}{'median range':>15}{'median stdev':>15}{'p90 range':>12}")
    print("-" * 81)
    for label, codes, index, need in (
        ("log M*, all codes", all_codes, 0, len(all_codes)),
        ("log M*, survived-mass codes only", SURVIVED_MASS_CODES, 0, len(SURVIVED_MASS_CODES)),
        ("log SFR, all codes", all_codes, 1, len(all_codes)),
        ("log SFR, 100 Myr codes only", AVERAGED_SFR_CODES, 1, len(AVERAGED_SFR_CODES)),
    ):
        stats = spread_stats(per_galaxy, codes, index, need)
        if stats:
            print(
                f"{label:<34}{stats['n_galaxies']:>5}"
                f"{stats['median_range']:>14.3f} {stats['median_stdev']:>14.3f} "
                f"{stats['p90_range']:>11.3f}"
            )

    print("\ndefinitional offsets:")
    mass = offset_against(per_galaxy, "Prospector", SURVIVED_MASS_CODES, 0, 3)
    if mass:
        print(f"   Prospector (formed) - median(survived codes) : {mass[0]:+.3f} dex, n={mass[1]}")
    sfr = offset_against(per_galaxy, "BAGPIPES", AVERAGED_SFR_CODES, 1, 3)
    if sfr:
        print(f"   BAGPIPES (instantaneous) - median(100 Myr)   : {sfr[0]:+.3f} dex, n={sfr[1]}")

    print(
        "\nThe ~0.1 and ~0.3 dex the section quotes are the STDEV column, not the range.\n"
        "tengri's stellar_mass is FORMED mass; four of these five codes report survived."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
