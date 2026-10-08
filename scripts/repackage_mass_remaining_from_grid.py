#!/usr/bin/env python3
"""Repackage the surviving-mass table a published SSP grid carries (#2751).

``data/fsps_prsc_miles_chabrier.h5`` ships its own ``ssp_mass_remaining``
(living stars + remnants per unit formed mass, python-fsps, PARSEC isochrones,
Chabrier IMF). The table does not depend on the spectral library, so it is
repackaged as the companion ``mass_remaining_prsc_chabrier.h5`` that every
PARSEC + Chabrier grid (MILES, C3K, BaSeL, with or without nebular emission)
resolves to, with the grid's own table kept as a cross-check.

Usage::

    python scripts/repackage_mass_remaining_from_grid.py \
        --grid data/fsps_prsc_miles_chabrier.h5 --isoc prsc --imf chabrier
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

_CITATION = (
    "Conroy, Gunn & White 2009, ApJ 699, 486 (doi:10.1088/0004-637X/699/1/486); "
    "Renzini & Ciotti 1993, ApJ 416, L49 (doi:10.1086/187068)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--isoc", required=True)
    parser.add_argument("--imf", required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(args.grid.read_bytes()).hexdigest()
    with h5py.File(args.grid, "r") as f:
        age_yr = np.asarray(f["ssp_lg_age_gyr"][:], dtype=np.float64) + 9.0
        log_z = np.asarray(f["ssp_lgmet"][:], dtype=np.float64)
        table = np.asarray(f["ssp_mass_remaining"][:], dtype=np.float64)
        description = f["ssp_mass_remaining"].attrs["description"]
    if table.shape != (log_z.shape[0], age_yr.shape[0]):
        raise ValueError(f"unexpected table shape {table.shape}")

    repo_root = Path(__file__).resolve().parent.parent
    out = (
        repo_root
        / "src/tengri/data/ssp_mass_remaining"
        / f"mass_remaining_{args.isoc}_{args.imf}.h5"
    )
    with h5py.File(out, "w") as f:
        f.create_dataset("log10_age_yr", data=age_yr)
        f.create_dataset("log10_z_abs", data=log_z)
        f.create_dataset("mass_remaining", data=table)
        f.attrs["quantity"] = "living stars + remnants per unit formed mass"
        f.attrs["isochrones"] = args.isoc
        f.attrs["imf"] = args.imf
        f.attrs["source"] = (
            f"ssp_mass_remaining of {args.grid.name} (python-fsps product; "
            f"grid attribute: {description})"
        )
        f.attrs["remnant_prescription"] = "Renzini & Ciotti 1993 (FSPS add_remnants.f90)"
        f.attrs["citation"] = _CITATION
        f.attrs["generator"] = str(Path(__file__).resolve().relative_to(repo_root))
        f.attrs["generator_args"] = f"--grid {args.grid.name} --isoc {args.isoc} --imf {args.imf}"
        f.attrs["input_sha256"] = digest
    print(f"wrote {out} shape={table.shape} input_sha256={digest}")


if __name__ == "__main__":
    main()
