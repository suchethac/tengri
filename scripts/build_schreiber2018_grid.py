#!/usr/bin/env python3
"""Build ``data/schreiber2018_templates.h5`` from AGNfitter-rX's S17 FITS.

The Schreiber et al. (2018) dust-SED library (the ``S17`` cold-dust model in
AGNFITTER-RX, the very-small-grain-corrected variant) ships as two FITS tables
under ``models/STARBURST/``:

- ``s17_lowvsg_dust.fits`` — dust-continuum SEDs, one per dust temperature.
- ``s17_lowvsg_pah.fits``  — PAH-feature SEDs, one per dust temperature.

Each table has a single row whose ``LAM`` / ``SED`` columns are 2-D arrays of
shape ``(n_Tdust, n_wave)`` (micron, ``nu L_nu`` in L_sun), with ``TDUST`` the
per-temperature axis. AGNFITTER-RX forms the cold-dust SED as the *native*
mixture ``(1 - f_PAH)·dust + f_PAH·PAH`` and renormalises (see
``MODEL_AGNfitter.STARBURST`` S17 branch). This script repackages both tables onto a
common ascending-wavelength [Å] grid as ``L_nu`` so tengri's ``schreiber2018``
emission model can reproduce that mixture at runtime.

HDF5 schema
-----------
``/schreiber2018``

==============  =====================  =====================================
Dataset         Shape                  Description
==============  =====================  =====================================
``tdust``       ``(n_T,)``             dust temperature [K], ascending
``wavelength``  ``(n_wave,)``          common wavelength grid [Å], ascending
``dust``        ``(n_T, n_wave)``      dust-continuum L_nu (native, unnormalised)
``pah``         ``(n_T, n_wave)``      PAH L_nu (native, same scale as ``dust``)
==============  =====================  =====================================

References
----------
- Schreiber, C., et al., "Dust temperature and mid-to-total infrared color
  distributions of star-forming galaxies at 0 < z < 4," A&A 609, A30 (2018).
- Martinez-Ramirez et al. 2024, A&A 688, A46 (AGNfitter-rX, the S17 packaging).

Usage
-----
::

    python scripts/build_schreiber2018_grid.py \\
        --input-dir /tmp/AGNfitter-rX/models/STARBURST \\
        --output data/schreiber2018_templates.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

_C_AA_PER_S = 2.99792458e18  # speed of light [Å·Hz]


def _table_to_lnu(fits_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an S17 FITS table → (tdust [K], wave [Å] per-T, L_nu per-T).

    Returns the per-temperature 2-D ``(n_T, n_wave)`` wavelength and L_nu
    arrays (wavelengths are identical per row but returned in full for the
    caller to regrid).
    """
    from astropy import units as u
    from astropy.table import Table

    t = Table.read(fits_path)
    lam_um = np.asarray(t["LAM"][0], dtype=np.float64)  # (n_T, n_wave) micron
    nu_lnu = np.asarray(t["SED"][0], dtype=np.float64)  # (n_T, n_wave) nuLnu [Lsun]
    tdust = np.asarray(t["TDUST"][0], dtype=np.float64)  # (n_T,) K
    nu_hz = (lam_um * u.micron).to(u.Hz, equivalencies=u.spectral()).value  # (n_T, n_wave)
    l_nu = nu_lnu / nu_hz  # (n_T, n_wave) L_nu (relative)
    wave_aa = lam_um * 1.0e4  # micron → Å
    return tdust, wave_aa, l_nu


def build(input_dir: Path, output_h5: Path, n_wave: int = 1024) -> None:
    """Read the S17 dust + PAH FITS and emit ``schreiber2018_templates.h5``."""
    tdust_d, wave_d, dust_lnu = _table_to_lnu(input_dir / "s17_lowvsg_dust.fits")
    tdust_p, wave_p, pah_lnu = _table_to_lnu(input_dir / "s17_lowvsg_pah.fits")

    if tdust_d.shape != tdust_p.shape or not np.allclose(tdust_d, tdust_p):
        raise RuntimeError("S17 dust and PAH tables have mismatched T_dust axes.")
    n_t = tdust_d.size

    # Common ascending wavelength grid over the overlap of both tables.
    w_min = max(float(wave_d.min()), float(wave_p.min()))
    w_max = min(float(wave_d.max()), float(wave_p.max()))
    common_wave = np.geomspace(w_min, w_max, n_wave)

    dust_grid = np.zeros((n_t, n_wave), dtype=np.float64)
    pah_grid = np.zeros((n_t, n_wave), dtype=np.float64)
    for i in range(n_t):
        od = np.argsort(wave_d[i])
        op = np.argsort(wave_p[i])
        dust_grid[i] = np.interp(common_wave, wave_d[i][od], dust_lnu[i][od])
        pah_grid[i] = np.interp(common_wave, wave_p[i][op], pah_lnu[i][op])

    # Sort the temperature axis ascending (carry the templates with it).
    order = np.argsort(tdust_d)
    tdust = tdust_d[order]
    dust_grid = dust_grid[order]
    pah_grid = pah_grid[order]

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("schreiber2018")
        g.create_dataset("tdust", data=tdust, compression="gzip")
        g.create_dataset("wavelength", data=common_wave, compression="gzip")
        g.create_dataset("dust", data=dust_grid, compression="gzip")
        g.create_dataset("pah", data=pah_grid, compression="gzip")
        g.attrs["source_dir"] = str(input_dir)
        g.attrs["n_tdust"] = n_t
        g.attrs["n_wave"] = n_wave
        g.attrs["tdust_unit"] = "K"
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "L_nu (native relative; mix renormalised at runtime)"

    print(
        f"wrote {output_h5} — {n_t} T_dust [{tdust.min():.1f}, {tdust.max():.1f}] K × "
        f"{n_wave} wavelengths [{w_min:.1f}, {w_max:.2e}] Å"
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/STARBURST"),
        help="Directory holding s17_lowvsg_{dust,pah}.fits.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/schreiber2018_templates.h5"),
        help="Destination HDF5 path.",
    )
    p.add_argument("--n-wave", type=int, default=1024, help="Common wavelength grid size.")
    args = p.parse_args()
    build(args.input_dir, args.output, n_wave=args.n_wave)


if __name__ == "__main__":
    _cli()
