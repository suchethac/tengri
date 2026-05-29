#!/usr/bin/env python3
"""Regenerate ``data/dale2014_templates.h5`` from CIGALE's Dale2014 database.

Closes issue #528: tengri's previous Dale2014 templates were generated
from a different source than CIGALE's, with an inconsistent ``alpha``
labelling — tengri ``alpha=2.0`` peaked at 116 µm while CIGALE
``alpha=2.0`` peaks at 90 µm. The ratios at ``§9`` MIR/FIR were 12-15%
off because of this template shape mismatch.

This script reads CIGALE's ``dale2014`` ``SimpleDatabase`` directly,
pulls the 64-alpha SF-only templates (``fracAGN=0``), and writes them
into tengri's ``data/dale2014_templates.h5`` format. After running,
tengri's Dale2014 produces SEDs that match CIGALE bit-for-bit at the
same alpha.

Requirements
------------
- pcigale installed (``pip install pcigale`` or via tengri's venv —
  tengri's ``.venv`` already has it).

Output: ``data/dale2014_templates.h5``

Usage
-----
    PYTHONPATH=. python scripts/regenerate_dale2014_from_cigale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# CIGALE's published alpha grid (64 values, 0.0625 → 4.0 in steps of 0.0625).
# Matches ``pcigale.sed_modules.dale2014`` exactly.
ALPHA_GRID = np.array(
    [
        0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375, 0.5000,
        0.5625, 0.6250, 0.6875, 0.7500, 0.8125, 0.8750, 0.9375, 1.0000,
        1.0625, 1.1250, 1.1875, 1.2500, 1.3125, 1.3750, 1.4375, 1.5000,
        1.5625, 1.6250, 1.6875, 1.7500, 1.8125, 1.8750, 1.9375, 2.0000,
        2.0625, 2.1250, 2.1875, 2.2500, 2.3125, 2.3750, 2.4375, 2.5000,
        2.5625, 2.6250, 2.6875, 2.7500, 2.8125, 2.8750, 2.9375, 3.0000,
        3.0625, 3.1250, 3.1875, 3.2500, 3.3125, 3.3750, 3.4375, 3.5000,
        3.5625, 3.6250, 3.6875, 3.7500, 3.8125, 3.8750, 3.9375, 4.0000,
    ]
)


def main() -> int:
    try:
        from pcigale.data import SimpleDatabase as Database
    except ImportError:
        print(
            "Error: pcigale is not importable in this environment.\n"
            "Install it (``pip install pcigale``) or run with tengri's "
            "main venv (``/Users/suchethacooray/Projects/tengri/.venv/bin/python``).",
            file=sys.stderr,
        )
        return 1

    print(f"Pulling {len(ALPHA_GRID)} SF-only (fracAGN=0) Dale2014 spectra from CIGALE...")
    spectra_lam_per_W = []
    wl_nm = None
    with Database("dale2014") as db:
        for a in ALPHA_GRID:
            model = db.get(fracAGN=0.0, alpha=float(a))
            if wl_nm is None:
                wl_nm = np.array(model.wl, dtype=np.float64)
            spectra_lam_per_W.append(np.array(model.spec, dtype=np.float64))
    spectra_lam_per_W = np.array(spectra_lam_per_W)

    # CIGALE spec is L_λ-like per W of input dust luminosity, normalised
    # so ``∫spec dλ_nm = 1.0`` (energy conservation: dust IR = L_absorbed).
    # tengri stores the wavelength axis in Å — convert per-nm density to
    # per-Å (÷10) so the integral over the Å grid stays 1.0.
    wl_aa = wl_nm * 10.0
    spectra_lam_per_Aa = spectra_lam_per_W / 10.0

    # Sanity check — integrated value at alpha=2.0 should be 1.000.
    i_2 = np.argmin(np.abs(ALPHA_GRID - 2.0))
    integral_check = np.trapezoid(spectra_lam_per_Aa[i_2], wl_aa)
    print(f"Integration check at alpha=2.0: ∫spec dλ_Å = {integral_check:.6f}  (expect 1.0)")
    assert abs(integral_check - 1.0) < 1e-3, "Template normalisation drift"

    out_path = Path("data/dale2014_templates.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("alpha_grid", data=ALPHA_GRID, dtype=np.float64)
        f.create_dataset("wavelength_aa", data=wl_aa, dtype=np.float64)
        f.create_dataset("templates_sf", data=spectra_lam_per_Aa, dtype=np.float64)
        f.attrs["source"] = "CIGALE pcigale.data Dale2014 (SimpleDatabase)"
        f.attrs["spectra_unit"] = "L_lambda per W input (integral over lambda_Aa = 1.0)"
        f.attrs["generated_by"] = "scripts/regenerate_dale2014_from_cigale.py"
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  shape: alpha_grid={ALPHA_GRID.shape}, wave={wl_aa.shape}, templates={spectra_lam_per_Aa.shape}")
    print("Done. Run reproduction/cigale/_drivers/consistency_audit.py to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
