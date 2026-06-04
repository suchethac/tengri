#!/usr/bin/env python3
r"""Build a Boquien & Salim 2021 BOSA dust emission HDF5 grid.

Downloads the canonical BOSA template FITS file from the project page
and repacks it into the simple HDF5 schema used by tengri's other
template-driven dust IR loaders.

Source
------
    https://salims.pages.iu.edu/bosa/bosa_LTIR-sSFR.fits

Reference
---------
    Boquien M. & Salim S. 2021, A&A 653 A149.
    "BOSA: a new mid- to far-infrared luminosity-dependent dust
    emission templates library."  arXiv:2106.04595.

Grid
----
    log10(L_TIR / L_sun): 8.5 -> 12.5 in 0.1 dex steps (41 values)
    log10(sSFR / yr^-1):  -11.0 -> -8.4 in 0.2 dex steps (14 values)
    574 templates total, 1001 wavelengths, all per-template
    ``nu*L_nu`` in solar luminosities.

The original synthetic ``data/bosa_templates.h5`` (11 x 11 grid on
[-13, -8] sSFR) was scientifically incorrect and is replaced by this
build's output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

BOSA_FITS_URL = "https://salims.pages.iu.edu/bosa/bosa_LTIR-sSFR.fits"
BOSA_FITS_NAME = "bosa_LTIR-sSFR.fits"

# Published grid axes (Boquien & Salim 2021).
LOG_LTIR_GRID = np.round(np.arange(8.5, 12.5 + 0.05, 0.1), 1)   # 41 values
LOG_SSFR_GRID = np.round(np.arange(-11.0, -8.4 + 0.05, 0.2), 1)  # 14 values


def download_fits(raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / BOSA_FITS_NAME
    if target.exists() and not force:
        return target
    print(f"[download] {BOSA_FITS_URL}  ->  {target}")
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry", "5",
            "--retry-connrefused",
            "--retry-delay", "5",
            "--connect-timeout", "30",
            "-C", "-",
            "-o", str(target),
            BOSA_FITS_URL,
        ],
        check=True,
    )
    return target


def parse_fits(fits_path: Path) -> dict:
    r"""Parse the BOSA two-parameter FITS table to a 3-D (LTIR, sSFR, wave) array.

    The published file is a binary table where:

    * column ``"wavelength"`` is in nm,
    * each remaining column is named ``nuLnu[LTIR=X.X, param2=Y.Y]`` and
      gives :math:`\nu L_\nu` in solar luminosities at that grid point.

    Returns
    -------
    dict
        ``wavelength_aa`` (Å), ``log_ltir_grid``, ``log_ssfr_grid``,
        ``L_nu`` shape ``(n_ltir, n_ssfr, n_wave)`` in :math:`L_\odot/{\rm Hz}`,
        ``L_nu_normalized`` with :math:`\int L_\nu \, d\nu = 1`.
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        tbl = hdul[1].data
        names = tbl.names

    if names[0] != "wavelength":
        raise ValueError(f"unexpected first column {names[0]!r}; expected 'wavelength'")

    wave_nm = np.asarray(tbl["wavelength"], dtype=np.float64)
    wave_aa = wave_nm * 10.0   # nm -> Å
    wave_um = wave_nm / 1.0e3  # nm -> μm

    n_wave = wave_nm.size
    n_ltir = LOG_LTIR_GRID.size
    n_ssfr = LOG_SSFR_GRID.size

    nu_Lnu = np.full((n_ltir, n_ssfr, n_wave), np.nan, dtype=np.float64)

    # Parse each non-wavelength column header to (LTIR, sSFR).
    expected_cols = 1 + n_ltir * n_ssfr
    if len(names) != expected_cols:
        raise ValueError(
            f"unexpected column count {len(names)}; "
            f"expected {expected_cols} (1 wavelength + {n_ltir}*{n_ssfr} templates)"
        )

    import re

    pat = re.compile(r"nuLnu\[LTIR=([-\d.]+),\s*param2=([-\d.]+)\]")
    for col in names[1:]:
        m = pat.match(col)
        if m is None:
            raise ValueError(f"unexpected column name: {col!r}")
        log_ltir = float(m.group(1))
        log_ssfr = float(m.group(2))
        i = int(np.argmin(np.abs(LOG_LTIR_GRID - log_ltir)))
        j = int(np.argmin(np.abs(LOG_SSFR_GRID - log_ssfr)))
        if not (
            np.isclose(LOG_LTIR_GRID[i], log_ltir, atol=0.05)
            and np.isclose(LOG_SSFR_GRID[j], log_ssfr, atol=0.05)
        ):
            raise ValueError(
                f"column {col!r} maps to grid ({LOG_LTIR_GRID[i]}, {LOG_SSFR_GRID[j]})"
            )
        nu_Lnu[i, j] = np.asarray(tbl[col], dtype=np.float64)

    if np.isnan(nu_Lnu).any():
        n_missing = int(np.isnan(nu_Lnu[..., 0]).sum())
        raise ValueError(f"{n_missing} (LTIR, sSFR) cells unfilled after parsing")

    # Convert nu*L_nu (Lsun) to L_nu (Lsun/Hz).
    c_cgs = 2.99792458e10
    lam_cm = wave_um * 1.0e-4
    nu_hz = c_cgs / lam_cm
    L_nu = nu_Lnu / nu_hz[None, None, :]  # broadcast

    # Normalised templates: int L_nu dnu = 1.  Same convention as
    # tengri's other dust IR loaders so the runtime path can rescale
    # by the energy-balance ``L_ir`` directly.
    L_sun = 3.828e33  # erg/s
    # Convert Lsun/Hz -> erg/s/Hz, then normalise.
    L_nu_cgs = L_nu * L_sun
    # int L_nu d nu — wavelength in increasing order, so integrate
    # against -d ln(λ) trick used elsewhere.
    sort = np.argsort(nu_hz)
    norms = np.trapezoid(L_nu_cgs[..., sort], nu_hz[sort], axis=-1)
    L_nu_normalized = L_nu_cgs / norms[..., None]

    return {
        "wavelength_aa": wave_aa.astype(np.float64),
        "log_ltir_grid": LOG_LTIR_GRID.astype(np.float32),
        "log_ssfr_grid": LOG_SSFR_GRID.astype(np.float32),
        "spectra": L_nu_normalized.astype(np.float64),  # (n_ltir, n_ssfr, n_wave) normalised
        "L_nu_solLum_per_Hz": L_nu.astype(np.float64),  # absolute (Lsun/Hz)
    }


def write_hdf5(out_path: Path, grid: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write]   {out_path}")
    with h5py.File(out_path, "w") as f:
        f.attrs["model"] = "BOSA (Boquien & Salim 2021)"
        f.attrs["paper"] = "Boquien & Salim 2021, A&A 653 A149"
        f.attrs["arxiv"] = "2106.04595"
        f.attrs["url"] = "https://salims.pages.iu.edu/bosa/"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["axes"] = "spectra[log_ltir, log_ssfr, wavelength]"
        for k, v in grid.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path,
                   default=Path("~/.cache/tengri/bosa_raw").expanduser())
    p.add_argument("--output", type=Path, default=Path("data/bosa_templates.h5"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args(argv)

    raw_dir = args.raw_dir.expanduser().resolve()
    fits_path = raw_dir / BOSA_FITS_NAME
    if not fits_path.exists() or args.force_download:
        if not (args.download or args.force_download):
            print(
                f"[error] {fits_path} not present. Re-run with --download.",
                file=sys.stderr,
            )
            return 1
        download_fits(raw_dir, force=args.force_download)

    grid = parse_fits(fits_path)
    write_hdf5(args.output, grid)
    print(
        f"[ok] grid: spectra={grid['spectra'].shape}, "
        f"log_ltir=[{grid['log_ltir_grid'].min()}..{grid['log_ltir_grid'].max()}], "
        f"log_ssfr=[{grid['log_ssfr_grid'].min()}..{grid['log_ssfr_grid'].max()}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
