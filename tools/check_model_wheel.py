#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fail when the model catalog wheel and the live registries disagree.

Why this exists
---------------

``docs/model_catalog.md`` draws tengri's whole modeling menu as a wheel, from
``docs/model_wheel.yaml`` resolved against the live ``tengri.list_*`` registries
at build time. The YAML sets where each model sits, so a newly registered model
has no wedge until someone places it. Without this guard the catalog would go
quietly incomplete, which looks exactly like the model never existing.

Checks, using the same resolver the docs build uses (``tools/model_wheel_data.py``):

1. every ``lister:`` source resolves to a registry entry;
2. every registry entry is drawn by some wheel model or matches ``excluded``;
3. every ``ssp:<regex>`` matches at least one known SSP grid;
4. each stellar axis group (SPS codes, isochrones, spectral libraries, IMF,
   multiplicity) covers every known SSP grid.

Check 4 is reported but fails only with ``--strict`` until the owner places the
grids whose code, isochrones or library are still open.

Usage
-----

::

    python tools/check_model_wheel.py
    python tools/check_model_wheel.py --strict
    python tools/check_model_wheel.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_wheel_data import coverage_problems, load_wheel, resolve  # noqa: E402


def _list(wheel: dict) -> None:
    for m in resolve(wheel)["modules"]:
        print(m["label"])
        for g in m["groups"]:
            print(f"  {g['label']}")
            for md in g["models"]:
                status = f" [{md['status']}]" if md["status"] else ""
                print(f"    {md['label']}{status}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="also fail on SSP-axis coverage gaps")
    ap.add_argument("--list", action="store_true", help="print the resolved wheel and exit")
    ap.add_argument("--wheel", type=Path, default=None, help="path to model_wheel.yaml")
    args = ap.parse_args(argv)

    wheel = load_wheel(args.wheel) if args.wheel else load_wheel()
    if args.list:
        _list(wheel)
        return 0
    reg_errors, axis_errors, drawn = coverage_problems(wheel)
    for e in reg_errors:
        print("FAIL", e)
    for e in axis_errors:
        print("FAIL" if args.strict else "PENDING", e)
    failing = len(reg_errors) + (len(axis_errors) if args.strict else 0)
    print(f"{failing} problem(s); {drawn} registry entries drawn; {len(axis_errors)} SSP axis gap(s)")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
