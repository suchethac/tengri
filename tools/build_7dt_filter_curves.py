# SPDX-License-Identifier: BSD-3-Clause
"""Convert the delivered 7DT transmission curves into bundled tengri filter files.

The delivery is one CSV per band, ``lam,trans``, wavelength in **nanometers** on
a uniform 0.1 nm grid spanning 300-1000 nm, zero-padded outside the passband.
Tengri wavelengths are Angstrom throughout, so this multiplies by 10 and writes
two-column ``.dat`` files next to a ``PROVENANCE.md``, matching the layout of
``src/tengri/data/agn_bbb``.

Zero-padding is stripped. Every band is >80 percent exact zeros as delivered,
and shipping 23 x 7001 rows of it would add ~2.9 MB to the wheel for no signal.
One zero row is kept on each side so interpolation still lands on zero at the
edges rather than extrapolating the first real value outward.

Run from the repository root::

    python tools/build_7dt_filter_curves.py --source /path/to/7DT_transmission_23bands

The AB-magnitude equivalence of the trim is checked by
``tests/contract/test_bundled_7dt_filters.py``, not here.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

BANDS: tuple[str, ...] = ("g", "r", "i", *(f"m{lam}" for lam in range(400, 900, 25)))

NM_TO_AA = 10.0

HEADER = """\
# 7DT {band} band -- total system transmission.
# Columns: wavelength [Angstrom]   transmission [dimensionless].
# Source: 7DT_transmission_23bands/{band}.csv, delivered 2026-08-27 by
#   Eunjae Herr (Im Lab, Seoul National University) for the NGC 1380 program.
#   Source file sha256 {src_sha}.
# Conversion: wavelength multiplied by 10 (delivered in nanometers on a uniform
#   0.1 nm grid, 300-1000 nm); transmission untouched. Leading and trailing runs
#   of exact zeros trimmed, one zero row kept on each side as an edge anchor.
# Peak transmission is {peak:.4f}, well below 1: detector QE and optics are
#   already folded into these curves. Do not renormalize. For AB synthetic
#   photometry the overall scale cancels in the ratio of the two bandpass
#   integrals, so only the shape matters.
"""


def trim_zero_padding(wave: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop leading and trailing exact-zero runs, keeping one anchor row each side.

    Returns the curve unchanged when it has no zero padding, or when it is
    entirely zero (which would be a corrupt input and is left for the caller
    to notice rather than silently reshaped).
    """
    nonzero = np.flatnonzero(trans != 0.0)
    if nonzero.size == 0:
        return wave, trans

    lo = max(nonzero[0] - 1, 0)
    hi = min(nonzero[-1] + 1, trans.size - 1)
    return wave[lo : hi + 1], trans[lo : hi + 1]


def sha256_of(path: Path) -> str:
    """Hex sha256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_band(band: str, source_dir: Path, out_dir: Path) -> dict[str, object]:
    """Convert one band CSV to a bundled ``.dat``, returning a provenance record."""
    src = source_dir / f"{band}.csv"
    raw = np.loadtxt(src, delimiter=",", skiprows=1)
    wave_nm, trans = raw[:, 0], raw[:, 1]

    wave_aa = wave_nm * NM_TO_AA
    kept_wave, kept_trans = trim_zero_padding(wave_aa, trans)

    out = out_dir / f"7dt_{band}.dat"
    header = HEADER.format(band=band, src_sha=sha256_of(src), peak=trans.max())
    with open(out, "w") as handle:
        handle.write(header)
        for lam, tr in zip(kept_wave, kept_trans, strict=True):
            # The 0.1 nm delivery grid becomes an exact 1 Angstrom integer grid,
            # and transmission carries at most 5 significant figures, so this
            # encoding is lossless and about a quarter shorter than %.5f pairs.
            handle.write(f"{lam:.1f} {tr:.5g}\n")

    return {
        "band": band,
        "name": f"7dt_{band}",
        "rows_in": int(trans.size),
        "rows_out": int(kept_trans.size),
        "wave_min_aa": float(kept_wave.min()),
        "wave_max_aa": float(kept_wave.max()),
        "peak": float(trans.max()),
        "src_sha": sha256_of(src),
        "out_sha": sha256_of(out),
    }


def main() -> None:
    """Convert every band and print a provenance table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory holding the delivered <band>.csv files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/tengri/data/filters_7dt"),
        help="Output directory for the bundled .dat files.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = [convert_band(band, args.source, args.out) for band in BANDS]

    total_in = sum(int(r["rows_in"]) for r in records)
    total_out = sum(int(r["rows_out"]) for r in records)
    print(f"{'name':12} {'rows':>13}  {'range [AA]':>20}  {'peak':>7}  sha256")
    for r in records:
        rows = f"{r['rows_out']}/{r['rows_in']}"
        rng = f"{r['wave_min_aa']:.0f}-{r['wave_max_aa']:.0f}"
        print(f"{r['name']:12} {rows:>13}  {rng:>20}  {r['peak']:7.4f}  {str(r['out_sha'])[:16]}")
    print(f"\ntotal rows {total_out}/{total_in} ({100 * total_out / total_in:.1f}% kept)")


if __name__ == "__main__":
    main()
