#!/usr/bin/env python3
r"""Build ``data/schreiber2016_templates.h5`` from CIGALE's database.

tengri previously approximated the Schreiber et al. (2016) dust IR SED with
an analytic modified-blackbody + Drude-profile PAH construct. CIGALE ships
the *tabulated* Schreiber+2016 templates (a dust continuum and a PAH template
per dust temperature). This script repacks them so tengri can reproduce
CIGALE's Schreiber2016 SED bit-for-bit.

Faithful to ``pcigale.sed_modules.schreiber2016._init_code``:

- ``continuum[t, :]`` = ``db.get(type=0, tdust=t)``  (dust continuum)
- ``pah[t, :]``       = ``db.get(type=1, tdust=t)``  (PAH template)
- Mixing at predict time: ``(1 - fpah) * continuum + fpah * pah``,
  energy-balance normalised so ``\int spec dlambda = 1`` (= L_absorbed).

Requirements: pcigale installed (tengri's main ``.venv`` has it).
Output: ``data/schreiber2016_templates.h5``.

Usage
-----
    PYTHONPATH=. .venv/bin/python \
        scripts/regenerate_schreiber2016_from_cigale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    try:
        from pcigale.data import SimpleDatabase as Database
    except ImportError:
        print(
            "Error: pcigale is not importable. Run with tengri's main venv "
            "(``.venv/bin/python``).",
            file=sys.stderr,
        )
        return 1

    with Database("schreiber2016") as db:
        tdust_grid = np.array(sorted(set(db.parameters["tdust"])), dtype=np.float64)
        wl_nm = None
        continuum = None
        pah = None
        for i, t in enumerate(tdust_grid):
            m_cont = db.get(type=0, tdust=float(t))
            m_pah = db.get(type=1, tdust=float(t))
            if wl_nm is None:
                wl_nm = np.array(m_cont.wl, dtype=np.float64)
                n_wave = wl_nm.shape[0]
                continuum = np.zeros((tdust_grid.shape[0], n_wave))
                pah = np.zeros((tdust_grid.shape[0], n_wave))
            continuum[i] = np.array(m_cont.spec, dtype=np.float64)
            pah[i] = np.array(m_pah.spec, dtype=np.float64)

    # CIGALE spectra are W/nm/kg. Convert wl to Angstrom and per-nm -> per-Aa.
    wl_aa = wl_nm * 10.0
    continuum_aa = continuum / 10.0
    pah_aa = pah / 10.0

    print(
        f"Pulled Schreiber2016: tdust={tdust_grid.shape} "
        f"({tdust_grid[0]:.0f}-{tdust_grid[-1]:.0f} K), wave={wl_aa.shape}"
    )

    out_path = Path("data/schreiber2016_templates.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("wavelength_aa", data=wl_aa, dtype=np.float64)
        f.create_dataset("tdust_grid", data=tdust_grid, dtype=np.float64)
        f.create_dataset("continuum", data=continuum_aa, dtype=np.float64, compression="gzip")
        f.create_dataset("pah", data=pah_aa, dtype=np.float64, compression="gzip")
        f.attrs["model"] = "Schreiber et al. 2016 via CIGALE SimpleDatabase"
        f.attrs["paper"] = "Schreiber, Elbaz, Pannella et al. 2016 A&A 609 A30"
        f.attrs["arxiv"] = "1606.00841"
        f.attrs["upstream"] = "pcigale.data.SimpleDatabase('schreiber2016')"
        f.attrs["axes"] = "(tdust, wavelength)"
        f.attrs["spectra_unit"] = "L_lambda per W input (raw W/nm/kg; loader normalises)"
        f.attrs["mixing"] = "(1-fpah)*continuum + fpah*pah, energy-balance to L_absorbed"
        f.attrs["generated_by"] = "scripts/regenerate_schreiber2016_from_cigale.py"
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
