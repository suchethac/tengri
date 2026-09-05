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
``MODEL_AGNfitter.STARBURST`` S17 branch). This script repackages both tables
onto a common ascending-wavelength [Å] grid as ``L_nu`` so tengri's
``schreiber2018`` emission model can reproduce that mixture at runtime.

Native-sampling preservation (#task3-M-B / D1)
-----------------------------------------------
Both FITS tables share **exactly one** native wavelength grid (721 points,
identical across every ``T_dust`` row and identical between the dust and PAH
tables). This script stores that grid **verbatim** — no resampling — so the
narrow 3.3/6.2/7.7/8.6/11.3/12.7 µm PAH complex keeps its native ~0.005 µm
sampling. An earlier version of this script threw that resolution away by
regridding both tables onto a shared 1024-point ``np.geomspace`` axis, which
smeared the 3.3 µm PAH feature by up to 32% (see the ``colddust_radio.md``
D1 finding). If a future upstream release ever ships dust/PAH tables on
*different* native grids, :func:`_native_wavelength_grid` falls back to the
sorted union of every distinct native grid, so every native sample is still
preserved (other rows are then linearly interpolated only onto that union,
never resampled away from their own tabulated points).

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

Group attributes record the upstream FITS file SHA-256 hashes
(``source_sha256_dust`` / ``source_sha256_pah``) and whether the wavelength
grid is the shared native grid verbatim or a union (``native_sampling``).

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
import hashlib
from pathlib import Path

import h5py
import numpy as np

_C_AA_PER_S = 2.99792458e18  # speed of light [Å·Hz]


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes, for provenance tracking."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _native_wavelength_grid(wave_rows: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """Return the axis every row should be expressed on, preserving native samples.

    Parameters
    ----------
    wave_rows : list of ndarray
        One per-row wavelength array [Å] (any order; sorted internally).

    Returns
    -------
    grid : ndarray
        Ascending wavelength axis [Å].
    mode : str
        ``"verbatim"`` when every row shares exactly one native grid (the
        returned axis IS that grid, unmodified); ``"union"`` when rows differ
        and the axis is the sorted union of every distinct native grid (every
        native sample from every row is a point on this axis).
    """
    unique_grids: list[np.ndarray] = []
    for w in wave_rows:
        w_sorted = np.sort(np.asarray(w, dtype=np.float64))
        is_new = not any(
            w_sorted.shape == u.shape and np.allclose(w_sorted, u, rtol=1e-10, atol=0.0)
            for u in unique_grids
        )
        if is_new:
            unique_grids.append(w_sorted)
    if len(unique_grids) == 1:
        return unique_grids[0], "verbatim"
    return np.unique(np.concatenate(unique_grids)), "union"


def _place_on_grid(wave_row: np.ndarray, sed_row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Express one row's SED on ``grid``, without resampling away native points.

    If ``wave_row`` (sorted) already equals ``grid``, the row's own tabulated
    values are returned unchanged (no interpolation arithmetic at all).
    Otherwise linear interpolation fills the grid points this row lacks —
    every point that IS one of this row's own native samples is still
    reproduced exactly, since it is present verbatim in ``grid`` (the union of
    every native grid) and ``np.interp`` returns the exact node value at an
    exact-match query point.
    """
    order = np.argsort(wave_row)
    w = wave_row[order]
    s = sed_row[order]
    if w.shape == grid.shape and np.allclose(w, grid, rtol=1e-10, atol=0.0):
        return s
    return np.interp(grid, w, s)


def build(input_dir: Path, output_h5: Path) -> None:
    """Read the S17 dust + PAH FITS and emit ``schreiber2018_templates.h5``."""
    dust_path = input_dir / "s17_lowvsg_dust.fits"
    pah_path = input_dir / "s17_lowvsg_pah.fits"
    tdust_d, wave_d, dust_lnu = _table_to_lnu(dust_path)
    tdust_p, wave_p, pah_lnu = _table_to_lnu(pah_path)

    if tdust_d.shape != tdust_p.shape or not np.allclose(tdust_d, tdust_p):
        raise RuntimeError("S17 dust and PAH tables have mismatched T_dust axes.")
    n_t = tdust_d.size

    common_wave, mode = _native_wavelength_grid(
        [wave_d[i] for i in range(n_t)] + [wave_p[i] for i in range(n_t)]
    )
    n_wave = common_wave.size

    dust_grid = np.zeros((n_t, n_wave), dtype=np.float64)
    pah_grid = np.zeros((n_t, n_wave), dtype=np.float64)
    for i in range(n_t):
        dust_grid[i] = _place_on_grid(wave_d[i], dust_lnu[i], common_wave)
        pah_grid[i] = _place_on_grid(wave_p[i], pah_lnu[i], common_wave)

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
        g.attrs["native_sampling"] = mode
        g.attrs["source_sha256_dust"] = _sha256(dust_path)
        g.attrs["source_sha256_pah"] = _sha256(pah_path)

    print(
        f"wrote {output_h5} — {n_t} T_dust [{tdust.min():.1f}, {tdust.max():.1f}] K × "
        f"{n_wave} native wavelengths [{common_wave[0]:.4f}, {common_wave[-1]:.2e}] Å "
        f"(native_sampling={mode})"
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
    args = p.parse_args()
    build(args.input_dir, args.output)


if __name__ == "__main__":
    _cli()
