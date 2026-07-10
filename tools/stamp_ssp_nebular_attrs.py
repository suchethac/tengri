#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Stamp machine-readable nebular-provenance attrs onto SSP HDF5 files (#1014).

The wNE (with-Nebular-Emission) grids historically carried no metadata — the
filename was the only signal, and a retained-LyC wNE grid is indistinguishable
from its bare parent by any Q_H heuristic. This tool writes the attributes
``load_ssp_data`` and the nebular backends now act on:

- ``nebular_included`` (bool) — nebular continuum + lines baked in?
- ``log_gas_u`` (float, optional) — ionization parameter of the baked layer.
- ``log_gas_z`` (float, optional) — gas-phase log10(Z/Zsun) of the baked layer.

By default every value is parsed from the filename convention
``*_wNE_logGasU<u>_logGasZ<z>.h5`` (no wNE token → bare). Override with the
explicit flags.

Usage
-----
    # Parse from filename convention (wNE token, logGasU/logGasZ values)
    python tools/stamp_ssp_nebular_attrs.py data/ssp_*_wNE_*.h5

    # Explicitly mark a bare grid (writes nebular_included=False)
    python tools/stamp_ssp_nebular_attrs.py --bare data/fsps_prsc_miles_chabrier.h5

    # Explicit values (filename parsing skipped)
    python tools/stamp_ssp_nebular_attrs.py --included --log-gas-u -3.0 \\
        --log-gas-z 0.0 data/my_custom_grid.h5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_GAS_TOKEN = re.compile(r"logGas([UZ])(-?[0-9.]+)")


def parse_filename(name: str) -> dict:
    """Parse nebular provenance from the ``*_wNE_logGasU<u>_logGasZ<z>`` convention.

    Parameters
    ----------
    name : str
        SSP file basename.

    Returns
    -------
    dict
        ``nebular_included`` (bool) plus ``log_gas_u`` / ``log_gas_z``
        (float) when the corresponding tokens are present.
    """
    out: dict = {"nebular_included": "wne" in name.lower()}
    for axis, value in _GAS_TOKEN.findall(name):
        key = "log_gas_u" if axis == "U" else "log_gas_z"
        out[key] = float(value.rstrip("."))
    return out


def stamp(path: Path, attrs: dict, *, dry_run: bool = False) -> None:
    """Write the provenance attrs onto ``path`` in place.

    Parameters
    ----------
    path : Path
        SSP HDF5 file to stamp.
    attrs : dict
        Attribute name → value pairs to write.
    dry_run : bool, optional
        Print what would be written without touching the file.
    """
    import h5py

    if dry_run:
        print(f"[dry-run] {path.name}: {attrs}")
        return
    with h5py.File(path, "r+") as f:
        for key, value in attrs.items():
            f.attrs[key] = value
    print(f"stamped {path.name}: {attrs}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on bad input).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("files", nargs="+", type=Path, help="SSP HDF5 files to stamp")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--included", action="store_true", help="force nebular_included=True for all files"
    )
    mode.add_argument(
        "--bare", action="store_true", help="force nebular_included=False for all files"
    )
    ap.add_argument("--log-gas-u", type=float, default=None, help="ionization parameter logU")
    ap.add_argument("--log-gas-z", type=float, default=None, help="gas-phase log10(Z/Zsun)")
    ap.add_argument("--dry-run", action="store_true", help="print without writing")
    args = ap.parse_args(argv)

    status = 0
    for path in args.files:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            status = 2
            continue
        attrs = parse_filename(path.name)
        if args.included:
            attrs["nebular_included"] = True
        elif args.bare:
            attrs["nebular_included"] = False
        if args.log_gas_u is not None:
            attrs["log_gas_u"] = args.log_gas_u
        if args.log_gas_z is not None:
            attrs["log_gas_z"] = args.log_gas_z
        stamp(path, attrs, dry_run=args.dry_run)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
