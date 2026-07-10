#!/usr/bin/env python3
"""Vendor the AGNfitter-rX S17 cold-dust FITS tables into the committed h5.

The reproduction driver (``reproduction/agnfitter/_drivers/agnfitter_driver.py``)
promises that runtime never reads the ``/tmp/AGNfitter-rX`` clone — every
reference template lives in the committed ``data/agnfitter_*.h5``. The S17
(Schreiber et al. 2018) cold-dust library violated that: ``_s17_tables()``
read ``models/STARBURST/s17_lowvsg_{dust,pah}.fits`` straight from the clone,
so the notebook only ran on machines where the clone happened to exist.

This script downloads the two FITS tables from the pinned upstream tag
(``AGNfitter-rX_v0.1``, byte-identical rebuilds — see
``scripts/_agnfitter_download.py``) and stores the raw arrays as an ``s17``
group inside ``data/agnfitter_cold_dust_reference.h5``, next to the existing
``dh02_ce01`` group. The driver mirrors MODEL_AGNfitter's math from these
arrays; no astropy or clone needed at runtime.

Usage::

    .venv/bin/python scripts/build_agnfitter_s17_reference.py

References
----------
- C. Schreiber et al., "Dust temperature and mid-to-total infrared color
  distributions for star-forming galaxies at 0 < z < 4," A&A 609, A30 (2018).
  arXiv:1710.10276. DOI: 10.1051/0004-6361/201731506.
- L. N. Martinez-Ramirez et al., "AGNfitter-rx: Modeling the radio-to-X-ray
  spectral energy distributions of AGNs," A&A 688, A46 (2024).
  arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from _agnfitter_download import fetch

_OUT = Path(__file__).resolve().parents[1] / "data" / "agnfitter_cold_dust_reference.h5"


def main() -> None:
    from astropy.table import Table

    dust = Table.read(fetch("models/STARBURST/s17_lowvsg_dust.fits"))
    pah = Table.read(fetch("models/STARBURST/s17_lowvsg_pah.fits"))

    # Single-row tables whose LAM/SED columns are 2-D (n_tdust, n_wave)
    # arrays — mirror MODEL_AGNfitter's ``column[0]`` access.
    payload = {
        "dust_lam_um": np.asarray(dust["LAM"][0], dtype=np.float64),
        "dust_sed_nulnu": np.asarray(dust["SED"][0], dtype=np.float64),
        "pah_lam_um": np.asarray(pah["LAM"][0], dtype=np.float64),
        "pah_sed_nulnu": np.asarray(pah["SED"][0], dtype=np.float64),
        "tdust": np.asarray(dust["TDUST"][0], dtype=np.float64),
    }

    if not _OUT.is_file():
        raise SystemExit(
            f"{_OUT} is missing — run scripts/build_agnfitter_bbb_reference.py first "
            "(it creates the dh02_ce01 group this script appends to)."
        )
    with h5py.File(_OUT, "a") as f:
        if "s17" in f:
            del f["s17"]
        g = f.create_group("s17")
        g.attrs["source"] = "AGNfitter-rX models/STARBURST s17_lowvsg_{dust,pah}.fits"
        g.attrs["lam_unit"] = "micron"
        g.attrs["sed_unit"] = "nu*L_nu (relative)"
        for k, v in payload.items():
            g.create_dataset(k, data=v.astype(np.float32), compression="gzip")
    print(f"Wrote s17 group to {_OUT} ({_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
