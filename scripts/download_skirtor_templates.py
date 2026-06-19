#!/usr/bin/env python3
"""Download and convert SKIRTOR SED template library.

Converts raw SKIRTOR ``.dat`` files (Stalevski et al. 2012, 2016) into an
HDF5 grid with **separate** disk and dust-thermal spectra, following the
CIGALE ``database_builder/skirtor2016`` convention.

The raw files must be downloaded manually from Google Drive:
    https://drive.google.com/drive/folders/0B7M0xXOPD2djZWlqMG9YbFlNR1U

Grid dimensions (CIGALE standard subset):
    tau     (5 values):  3, 5, 7, 9, 11
    p       (4 values):  0.0, 0.5, 1.0, 1.5
    q       (4 values):  0.0, 0.5, 1.0, 1.5
    oa      (8 values):  10, 20, 30, 40, 50, 60, 70, 80  [degrees]
    i       (10 values): 0, 10, 20, ..., 90  [degrees] -> cos(i)
    R       (fixed):     20
    Mcl     (fixed):     0.97

Raw file format (7 columns):
    col 0: wavelength [micron]
    col 1: total flux  lambda*F_lambda [W/m^2]
    col 2: direct accretion disk
    col 3: scattered accretion disk
    col 4: total dust thermal emission
    col 5: scattered dust emission
    col 6: transparent (unattenuated) disk

Output: data/skirtor_templates_v3.h5

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
   torus," MNRAS, 420, 2756 (2012). https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN," MNRAS, 458,
   2288 (2016). https://doi.org/10.1093/mnras/stw444

Usage:
    python scripts/download_skirtor_templates.py --input-dir <path-to-dat-files>
    python scripts/download_skirtor_templates.py \
        --input-dir <path> --output data/skirtor_templates_v3.h5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

FILENAME_RE = re.compile(r"t(\d+)_p([\d.]+)_q([\d.]+)_oa(\d+)_R(\d+)_Mcl([\d.]+)_i(\d+)_sed\.dat")

TAU_VALUES = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
P_VALUES = np.array([0.0, 0.5, 1.0, 1.5])
Q_VALUES = np.array([0.0, 0.5, 1.0, 1.5])
OA_VALUES = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
R_VALUES = np.array([10.0, 20.0, 30.0])  # outer/inner radius ratio (Stalevski grid)
INC_VALUES = np.arange(0.0, 91.0, 10.0)
COS_INC_VALUES = np.cos(np.deg2rad(INC_VALUES))


def parse_filename(name: str) -> dict | None:
    """Extract grid parameters from a SKIRTOR filename.

    Parameters
    ----------
    name : str
        Filename like ``t7_p1.0_q0.5_oa40_R20_Mcl0.97_i30_sed.dat``.

    Returns
    -------
    dict or None
        Parameter dict with keys ``tau, p, q, oa, R, Mcl, i``, or None
        if the filename does not match the expected pattern.
    """
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    return {
        "tau": int(m.group(1)),
        "p": float(m.group(2)),
        "q": float(m.group(3)),
        "oa": int(m.group(4)),
        "R": int(m.group(5)),
        "Mcl": float(m.group(6)),
        "i": int(m.group(7)),
    }


def _find_index(arr: np.ndarray, value: float) -> int | None:
    """Return index of ``value`` in ``arr``, or None if not found."""
    idx = np.argmin(np.abs(arr - value))
    if np.abs(arr[idx] - value) < 1e-6:
        return int(idx)
    return None


def _extrapolate_to_10mm(wl: np.ndarray, disk: np.ndarray, dust: np.ndarray):
    """Extrapolate spectra to 10 mm following CIGALE convention.

    Parameters
    ----------
    wl : ndarray, shape (n,)
        Wavelength in nm (after micron -> nm conversion).
    disk : ndarray, shape (n,)
        Disk spectrum (direct + scattered) divided by wavelength.
    dust : ndarray, shape (n,)
        Dust thermal spectrum divided by wavelength.

    Returns
    -------
    wl, disk, dust : ndarray
        Extended arrays.
    """
    wl_ext = np.array([2e6, 4e6, 8e6, 1e7])  # nm
    disk_ext = np.full(len(wl_ext), 1e-99)
    if dust[-1] == 0 or dust[-1] <= 0:
        dust_ext = np.full(len(wl_ext), 1e-99)
    else:
        dust_ext = 10.0 ** (
            np.log10(dust[-1])
            + np.log10(wl_ext / wl[-1]) * np.log10(dust[-2] / dust[-1]) / np.log10(wl[-2] / wl[-1])
        )
    disk[-1] = 1e-99
    wl = np.append(wl, wl_ext)
    disk = np.append(disk, disk_ext)
    dust = np.append(dust, dust_ext)
    return wl, disk, dust


def convert_skirtor_grid(input_dir: Path, output_path: Path) -> None:
    """Convert raw SKIRTOR SED files to HDF5 with separate disk/dust.

    Follows the CIGALE ``database_builder/skirtor2016`` processing:
    1. Read columns 0 (wavelength), 2 (direct disk), 3 (scattered), 4 (dust)
    2. Combine disk = direct + scattered
    3. Divide both by wavelength (λF_λ → F_ν-like)
    4. Extrapolate to 10 mm
    5. Normalize so dust thermal integrates to 1
    6. Store disk and dust separately

    Parameters
    ----------
    input_dir : Path
        Directory containing raw ``t*_sed.dat`` files.
    output_path : Path
        Output HDF5 file path.
    """
    import h5py

    sed_files = sorted(input_dir.glob("t*_sed.dat"))
    if not sed_files:
        print(f"Error: No t*_sed.dat files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(sed_files)} SKIRTOR SED files")

    reference_wl = None
    n_tau = len(TAU_VALUES)
    n_p = len(P_VALUES)
    n_q = len(Q_VALUES)
    n_oa = len(OA_VALUES)
    n_R = len(R_VALUES)
    n_inc = len(COS_INC_VALUES)

    disk_spectra = {}
    dust_spectra = {}
    norm_values = {}
    n_loaded = 0
    n_skipped = 0

    for sed_file in sed_files:
        params = parse_filename(sed_file.name)
        if params is None:
            print(f"  Skipping (unrecognized name): {sed_file.name}")
            n_skipped += 1
            continue

        i_tau = _find_index(TAU_VALUES, params["tau"])
        i_p = _find_index(P_VALUES, params["p"])
        i_q = _find_index(Q_VALUES, params["q"])
        i_oa = _find_index(OA_VALUES, params["oa"])
        i_R = _find_index(R_VALUES, params["R"])
        i_inc = _find_index(INC_VALUES, params["i"])

        if any(idx is None for idx in [i_tau, i_p, i_q, i_oa, i_R, i_inc]):
            print(f"  Skipping (out of grid): {sed_file.name}")
            n_skipped += 1
            continue

        wl, disk_col, scatt_col, dust_col = np.genfromtxt(
            sed_file, unpack=True, usecols=(0, 2, 3, 4)
        )
        wl *= 1e3  # micron -> nm

        disk = disk_col + scatt_col
        disk /= wl
        dust_col /= wl

        wl, disk, dust_col = _extrapolate_to_10mm(wl, disk, dust_col)

        norm = np.trapezoid(dust_col, x=wl)
        if norm > 0:
            disk /= norm
            dust_col /= norm
        else:
            norm = 1.0

        key = (i_tau, i_p, i_q, i_oa, i_R, i_inc)
        disk_spectra[key] = disk
        dust_spectra[key] = dust_col
        norm_values[key] = norm

        if reference_wl is None:
            reference_wl = wl

        n_loaded += 1

    print(f"Loaded {n_loaded} files, skipped {n_skipped}")

    if n_loaded == 0:
        print("Error: No valid files loaded")
        sys.exit(1)

    n_wave = len(reference_wl)
    wl_angstrom = reference_wl * 10.0  # nm -> Angstrom

    disk_grid = np.full((n_tau, n_p, n_q, n_oa, n_R, n_inc, n_wave), 1e-99)
    dust_grid = np.full((n_tau, n_p, n_q, n_oa, n_R, n_inc, n_wave), 1e-99)
    total_grid = np.full((n_tau, n_p, n_q, n_oa, n_R, n_inc, n_wave), 1e-99)
    norms = np.zeros((n_tau, n_p, n_q, n_oa, n_R, n_inc))

    for key, disk_spec in disk_spectra.items():
        i_tau, i_p, i_q, i_oa, i_R, i_inc = key
        disk_grid[i_tau, i_p, i_q, i_oa, i_R, i_inc, :] = disk_spec
        dust_grid[i_tau, i_p, i_q, i_oa, i_R, i_inc, :] = dust_spectra[key]
        total_grid[i_tau, i_p, i_q, i_oa, i_R, i_inc, :] = disk_spec + dust_spectra[key]
        norms[i_tau, i_p, i_q, i_oa, i_R, i_inc] = norm_values[key]

    _ncell = n_tau * n_p * n_q * n_oa * n_R * n_inc
    coverage = n_loaded / _ncell
    print(f"Grid coverage: {n_loaded}/{_ncell} ({coverage:.1%})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("wavelength", data=wl_angstrom)

        grp = f.create_group("grid")
        grp.create_dataset("tau_97", data=TAU_VALUES)
        grp.create_dataset("p", data=P_VALUES)
        grp.create_dataset("q", data=Q_VALUES)
        grp.create_dataset("opening_angle", data=OA_VALUES)
        grp.create_dataset("radius_ratio", data=R_VALUES)
        grp.create_dataset("cos_inclination", data=COS_INC_VALUES)

        spec = f.create_group("spectra")
        spec.create_dataset(
            "disk_emission", data=disk_grid, compression="gzip", compression_opts=9
        )
        spec.create_dataset(
            "dust_emission", data=dust_grid, compression="gzip", compression_opts=9
        )
        spec.create_dataset(
            "torus_emission", data=total_grid, compression="gzip", compression_opts=9
        )
        spec.create_dataset("norm", data=norms)

        meta = f.create_group("metadata")
        meta.attrs["version"] = 3
        meta.attrs["created_by"] = "tengri download_skirtor_templates.py"
        meta.attrs["description"] = (
            "Clumpy two-phase AGN torus model with separate disk and dust "
            "components. Follows CIGALE skirtor2016 processing convention."
        )
        meta.attrs["model_name"] = "SKIRTOR (Stalevski et al. 2012, 2016)"
        meta.attrs["reference"] = (
            "Stalevski, M. et al. 2012, MNRAS, 420, 2756; 2016, MNRAS, 458, 2288"
        )
        meta.attrs["wavelength_unit"] = "Angstrom"
        meta.attrs["flux_unit"] = "normalized (dust thermal integrates to 1)"
        meta.attrs["processing"] = (
            "disk = direct_disk + scattered; both divided by wavelength; "
            "extrapolated to 10mm; normalized so dust integrates to 1W"
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved v3 HDF5 to {output_path} ({size_mb:.1f} MB)")
    print("  Datasets: disk_emission, dust_emission, torus_emission, norm")
    print(f"  Grid shape: {total_grid.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw SKIRTOR SED files to HDF5 with separate components.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing raw SKIRTOR t*_sed.dat files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/skirtor_templates_v3.h5",
        help="Output HDF5 file path (default: data/skirtor_templates_v3.h5)",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        print("SKIRTOR Template Grid Converter")
        print("=" * 50)
        print()
        print("The SKIRTOR SED library must be downloaded manually from:")
        print()
        print("  https://drive.google.com/drive/folders/0B7M0xXOPD2djZWlqMG9YbFlNR1U")
        print()
        print("After downloading, run:")
        print()
        print(f"  python {__file__} --input-dir <path-to-dat-files>")
        print()
        print("This will produce an HDF5 file with separate disk and dust")
        print("emission components for use with tengri's SKIRTOR model.")
        print()
        print("Expected raw file format:")
        print("  t{tau}_p{p}_q{q}_oa{oa}_R{R}_Mcl{Mcl}_i{i}_sed.dat")
        print()
        print("Grid parameters (CIGALE standard):")
        print(f"  tau:  {TAU_VALUES.tolist()}")
        print(f"  p:    {P_VALUES.tolist()}")
        print(f"  q:    {Q_VALUES.tolist()}")
        print(f"  oa:   {OA_VALUES.tolist()} degrees")
        print(f"  i:    {INC_VALUES.tolist()} degrees")
        return

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' not found")
        sys.exit(1)

    convert_skirtor_grid(input_dir, Path(args.output))


if __name__ == "__main__":
    main()
