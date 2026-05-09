#!/usr/bin/env python3
r"""Build a Hensley & Draine 2023 Astrodust+PAH emission HDF5 grid.

Downloads the canonical FITS file from Harvard Dataverse and repacks
HDU 7 (the per-component emission grid) into the same simple HDF5
schema used by tengri's other dust IR templates.

Source
------
    doi:10.7910/DVN/3B6E6S
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3B6E6S
    File: astrodust+PAH_MW_RV3.1.fits

Reference
---------
    Hensley, B.S. & Draine, B.T. 2023, ApJ 948, 55.

The real published file is a *single configuration*: one fiducial PAH
size distribution + ionization fraction (Hensley & Draine 2022) and
one MW R_V=3.1 sightline.  The only continuous knob is
:math:`\log_{10} U` over -3..6 in 0.1 steps (91 points).

This is **fundamentally different** from the (qpah x umin) grid that
``data/astrodust_templates.h5`` previously assumed; that legacy file
was synthetic.  After this script runs the legacy file is replaced
with real data and the schema documented in :func:`tengri.components.
dust.emission_templates.load_astrodust_hd23_templates`.

Usage
-----
    python scripts/build_astrodust_hdf5.py --download

The download is ~3 MB; the resulting HDF5 is ~3 MB.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

DATAVERSE_DOI = "10.7910/DVN/3B6E6S"
DATAVERSE_FILE_ID = 6433719  # astrodust+PAH_MW_RV3.1.fits
DATAVERSE_API = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
DEFAULT_FITS_NAME = "astrodust+PAH_MW_RV3.1.fits"


def download_fits(raw_dir: Path, force: bool = False) -> Path:
    """Download the canonical FITS file via the Dataverse API.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Cache directory; created if missing.
    force : bool
        Re-download even if a cached copy exists.

    Returns
    -------
    pathlib.Path
        Local path to the FITS file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / DEFAULT_FITS_NAME
    if target.exists() and not force:
        return target
    url = DATAVERSE_API.format(file_id=DATAVERSE_FILE_ID)
    print(f"[download] {url}  ->  {target}")
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-connrefused",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "-C",
        "-",
        "-o",
        str(target),
        url,
    ]
    subprocess.run(cmd, check=True)
    return target


def _convert_lambdaI_to_Lnu(
    lambda_I_lambda: np.ndarray,
    wave_um: np.ndarray,
) -> np.ndarray:
    r"""Convert Hensley & Draine units to L_nu per H atom.

    The published FITS column carries
    :math:`\lambda I_\lambda / N_H` in erg s^-1 sr^-1 H^-1 — sky surface
    brightness per H atom assuming isotropic emission.  The downstream
    tengri pipeline wants :math:`L_\nu` in erg s^-1 Hz^-1 H^-1.

    For an optically-thin, isotropically-emitting volume,

    .. math::
        \nu P_\nu \;=\; 4\pi\, \nu I_\nu \;=\; 4\pi\, \lambda I_\lambda

    so

    .. math::
        L_\nu \;=\; \frac{\nu P_\nu}{\nu}
              \;=\; \frac{4\pi\, \lambda I_\lambda \, \lambda}{c} \, .

    Parameters
    ----------
    lambda_I_lambda : ndarray, shape ``(..., n_wave)``
        :math:`\lambda I_\lambda / N_H` in erg/s/sr/H.
    wave_um : ndarray, shape ``(n_wave,)``
        Wavelength in microns.

    Returns
    -------
    ndarray, shape ``(..., n_wave)``
        :math:`L_\nu / N_H` in erg/s/Hz/H.
    """
    c_cgs = 2.99792458e10  # cm/s
    lam_cm = wave_um * 1.0e-4
    four_pi = 4.0 * np.pi
    return four_pi * lambda_I_lambda * lam_cm[..., :] / c_cgs


def parse_fits(fits_path: Path) -> dict:
    """Extract emission, spinning dust, and size distribution from FITS.

    Reads three HDUs:

    * HDU 7 EMISSION — thermal IR/sub-mm emission, ``(91 lgU, 1000 wave,
      3 components)`` in :math:`\\lambda I_\\lambda/N_H` [erg/s/sr/H].
      Components: ``astrodust [0]``, ``pah [1]``, ``sum [2]``.
    * HDU 9 SPINNING DUST EMISSION — microwave AME, ``(1000 wave,
      6 columns: wave_um, Ad-CNM, Ad-WNM, PAH-CNM, PAH-WNM, total)``.
      Assumes ``f_CNM = 0.28``.  Mostly constant in U; provided
      independent of the lgU axis.
    * HDU 1 SIZE DISTRIBUTION — ``(167 sizes, 5 columns: a_um, dn_Ad/nH,
      dn_PAH/nH, f_ion, f_align)``.  Stored for traceability.

    Returns
    -------
    dict
        Float32 arrays ready for HDF5 write.
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        wave_um = np.asarray(hdul["WAVELENGTHS"].data, dtype=np.float64)
        lgU = np.asarray(hdul["LOG10 U"].data, dtype=np.float64)
        emission = np.asarray(hdul["EMISSION"].data, dtype=np.float64)
        spdust = np.asarray(hdul["SPINNING DUST EMISSION"].data, dtype=np.float64)
        size_dist = np.asarray(hdul["SIZE DISTRIBUTION"].data, dtype=np.float64)
        # HDU 2 EXTINCTION:    (1000, 4) — wave_um, tau_Ad/NH, tau_PAH/NH, tau_total/NH
        # HDU 3 SCATTERING:    (1000, 4) — same column layout
        # HDU 4 POL EXTINCTION:(1000, 2) — wave_um, p_Ad^max/NH
        ext = np.asarray(hdul["EXTINCTION"].data, dtype=np.float64)
        scatt = np.asarray(hdul["SCATTERING"].data, dtype=np.float64)
        extpol = np.asarray(hdul["POLARIZED EXTINCTION"].data, dtype=np.float64)
        # HDU 8 POLARIZED EMISSION: (91 lgU, 1000 wave), λP_λ/NH from astrodust.
        emisspol = np.asarray(hdul["POLARIZED EMISSION"].data, dtype=np.float64)

    if emission.shape != (lgU.size, wave_um.size, 3):
        raise ValueError(
            f"unexpected EMISSION shape {emission.shape}; expected ({lgU.size}, {wave_um.size}, 3)"
        )
    if spdust.shape != (wave_um.size, 6):
        raise ValueError(
            f"unexpected SPINNING DUST EMISSION shape {spdust.shape}; expected ({wave_um.size}, 6)"
        )

    # Thermal emission: convert to L_nu per H atom.
    lambda_I_astrodust = emission[..., 0]
    lambda_I_pah = emission[..., 1]
    lambda_I_total = emission[..., 2]

    L_nu_astrodust = _convert_lambdaI_to_Lnu(lambda_I_astrodust, wave_um)
    L_nu_pah = _convert_lambdaI_to_Lnu(lambda_I_pah, wave_um)
    L_nu_total = _convert_lambdaI_to_Lnu(lambda_I_total, wave_um)

    summed = L_nu_astrodust + L_nu_pah
    rel_err = np.abs(L_nu_total - summed).max() / np.abs(L_nu_total).max()
    if rel_err > 1e-4:
        print(f"[warn] total != astrodust + PAH at relative precision {rel_err:.2e}")

    # Spinning dust: same lambda*I_lambda -> L_nu conversion.
    # Column layout: (wave, Ad_CNM, Ad_WNM, PAH_CNM, PAH_WNM, total).
    spd_wave = spdust[:, 0]
    if not np.allclose(spd_wave, wave_um, rtol=1e-6):
        raise ValueError("spinning-dust wavelength axis disagrees with HDU 6")
    L_nu_spdust_Ad_CNM = _convert_lambdaI_to_Lnu(spdust[:, 1], wave_um)
    L_nu_spdust_Ad_WNM = _convert_lambdaI_to_Lnu(spdust[:, 2], wave_um)
    L_nu_spdust_PAH_CNM = _convert_lambdaI_to_Lnu(spdust[:, 3], wave_um)
    L_nu_spdust_PAH_WNM = _convert_lambdaI_to_Lnu(spdust[:, 4], wave_um)
    L_nu_spdust_total = _convert_lambdaI_to_Lnu(spdust[:, 5], wave_um)

    # Spinning dust per-H values are O(1e-40) — float32 denormals or
    # underflow.  Store as float64.  Thermal emission is O(1e-23) per H,
    # safely in float32 range.
    return {
        "wavelength_um": wave_um.astype(np.float32),
        "lgU": lgU.astype(np.float32),
        "L_nu_total": L_nu_total.astype(np.float32),
        "L_nu_astrodust": L_nu_astrodust.astype(np.float32),
        "L_nu_pah": L_nu_pah.astype(np.float32),
        "L_nu_spdust_total": L_nu_spdust_total.astype(np.float64),
        "L_nu_spdust_Ad_CNM": L_nu_spdust_Ad_CNM.astype(np.float64),
        "L_nu_spdust_Ad_WNM": L_nu_spdust_Ad_WNM.astype(np.float64),
        "L_nu_spdust_PAH_CNM": L_nu_spdust_PAH_CNM.astype(np.float64),
        "L_nu_spdust_PAH_WNM": L_nu_spdust_PAH_WNM.astype(np.float64),
        "size_distribution": size_dist.astype(np.float32),
        # Extinction + scattering + polarization (cm^2 / H).
        "extinction": ext.astype(np.float32),
        "scattering": scatt.astype(np.float32),
        "polarized_extinction": extpol.astype(np.float32),
        # Polarized emission (lgU, wave) — astrodust only.
        "lambda_P_lambda_polarized": emisspol.astype(np.float64),
    }


def write_hdf5(out_path: Path, grid: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write]   {out_path}")
    with h5py.File(out_path, "w") as f:
        f.attrs["paper"] = "Hensley & Draine 2023, ApJ 948, 55"
        f.attrs["arxiv"] = "2208.12365"
        f.attrs["doi"] = DATAVERSE_DOI
        f.attrs["model"] = "Astrodust+PAH"
        f.attrs["sightline"] = "MW R_V=3.1"
        f.attrs["size_distribution_ref"] = "Hensley & Draine 2022"
        f.attrs["units_emission"] = "erg/s/Hz per H atom"
        f.attrs["axes"] = "L_nu_*[lgU, wavelength]"
        # Dust-mass-to-H conversions per Hensley & Draine 2023 README:
        # M_Ad / M_H = 0.00642 and M_PAH / M_H = 0.000659.  Stored so
        # downstream code can convert per-H luminosities to per-mass.
        f.attrs["M_Ad_over_M_H"] = 0.00642
        f.attrs["M_PAH_over_M_H"] = 0.000659
        for k, v in grid.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--raw-dir", type=Path, default=Path("~/.cache/tengri/astrodust_raw").expanduser()
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/astrodust_templates.h5"),
        help="output HDF5 file (created/overwritten)",
    )
    p.add_argument("--download", action="store_true", help="download the FITS file if missing")
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args(argv)

    raw_dir = args.raw_dir.expanduser().resolve()
    fits_path = raw_dir / DEFAULT_FITS_NAME

    if not fits_path.exists() or args.force_download:
        if not (args.download or args.force_download):
            print(
                f"[error] {fits_path} not present.  Re-run with --download "
                f"to fetch from doi:{DATAVERSE_DOI}.",
                file=sys.stderr,
            )
            return 1
        download_fits(raw_dir, force=args.force_download)

    grid = parse_fits(fits_path)
    write_hdf5(args.output, grid)
    print(
        f"[ok] grid shape: L_nu_total={grid['L_nu_total'].shape}, "
        f"lgU=[{grid['lgU'].min()}..{grid['lgU'].max()}], "
        f"wave_um=[{grid['wavelength_um'].min():.3f}..{grid['wavelength_um'].max():.0f}]"
    )
    # Suppress unused-import warning in some checkers.
    _ = shutil
    return 0


if __name__ == "__main__":
    sys.exit(main())
