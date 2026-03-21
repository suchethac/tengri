#!/usr/bin/env python3
"""Convert FSPS ASCII CLOUDY grids to tengri HDF5 format.

Reads ZAU_ND_{isoc}.lines and .cont files from $SPS_HOME/nebular/
or from data/cloudy_raw/ and writes a single HDF5 file.

Usage:
    python scripts/convert_fsps_cloudy_grid.py [--input-dir data/cloudy_raw] [--isoc mist]
    python scripts/convert_fsps_cloudy_grid.py --sps-home  # reads from $SPS_HOME/nebular/

Output: data/cloudy_grid_{isoc}.h5
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def parse_header(line: str) -> dict:
    """Parse FSPS CLOUDY grid header line.

    Format: '#<ncols> cols <nrows> rows <nZ> logZ <nage> Age <nU> logU'
    """
    parts = line.strip().lstrip("#").split()
    return {
        "n_cols": int(parts[0]),
        "n_rows": int(parts[2]),
        "n_Z": int(parts[4]),
        "n_age": int(parts[6]),
        "n_logU": int(parts[8]),
    }


def read_fsps_grid(filepath: str) -> dict:
    """Read an FSPS CLOUDY grid file (lines or continuum).

    Returns dict with keys:
        wavelength: (n_cols,) array of wavelengths in Angstrom
        data: (n_Z, n_age, n_logU, n_cols) array of luminosities
        logZ: (n_Z,) array of log10(Z) values
        age_yr: (n_age,) array of ages in years
        logU: (n_logU,) array of log10(U) values
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header = parse_header(lines[0])
    n_cols = header["n_cols"]
    n_Z = header["n_Z"]
    n_age = header["n_age"]
    n_logU = header["n_logU"]
    n_rows = header["n_rows"]

    assert n_rows == n_Z * n_age * n_logU, (
        f"Row count mismatch: {n_rows} != {n_Z}*{n_age}*{n_logU}"
    )

    # Line 2: wavelength array
    wavelength = np.array(lines[1].split(), dtype=np.float64)
    # Header n_cols can be stale in newer FSPS versions; use actual count
    if len(wavelength) != n_cols:
        print(f"  Note: header says {n_cols} cols but file has "
              f"{len(wavelength)} wavelengths; using actual count")
        n_cols = len(wavelength)

    # Lines 3+: alternating parameter line + data line
    data = np.zeros((n_Z, n_age, n_logU, n_cols), dtype=np.float64)
    logZ_vals = []
    age_vals = []
    logU_vals = []

    line_idx = 2
    for iz in range(n_Z):
        for ia in range(n_age):
            for iu in range(n_logU):
                # Parameter line: logZ age logU
                params = lines[line_idx].split()
                logZ_val = float(params[0])
                age_val = float(params[1])
                logU_val = float(params[2])

                if ia == 0 and iu == 0:
                    logZ_vals.append(logZ_val)
                if iz == 0 and iu == 0:
                    age_vals.append(age_val)
                if iz == 0 and ia == 0:
                    logU_vals.append(logU_val)

                line_idx += 1

                # Data line
                values = np.array(lines[line_idx].split(), dtype=np.float64)
                assert len(values) == n_cols, (
                    f"Data count mismatch at ({iz},{ia},{iu}): "
                    f"{len(values)} != {n_cols}"
                )
                data[iz, ia, iu, :] = values
                line_idx += 1

    return {
        "wavelength": wavelength,
        "data": data,
        "logZ": np.array(logZ_vals),
        "age_yr": np.array(age_vals),
        "logU": np.array(logU_vals),
    }


def read_emlines_info(filepath: str) -> list[tuple[float, str]]:
    """Read emlines_info.dat for line names.

    Format: wavelength,name (CSV)
    """
    result = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                result.append((float(parts[0]), parts[1].strip()))
    return result


def convert(
    input_dir: str,
    output_path: str,
    isoc: str = "mist",
    dust_variant: str = "ND",
) -> None:
    """Convert FSPS CLOUDY grids to tengri HDF5.

    Parameters
    ----------
    input_dir : str
        Directory containing ZAU_*.lines and .cont files
    output_path : str
        Output HDF5 file path
    isoc : str
        Isochrone type (mist, prsc, pdva, bpss)
    dust_variant : str
        ND (no dust) or WD (with dust depletion)
    """
    lines_file = os.path.join(input_dir, f"ZAU_{dust_variant}_{isoc}.lines")
    cont_file = os.path.join(input_dir, f"ZAU_{dust_variant}_{isoc}.cont")
    emlines_file = os.path.join(input_dir, "emlines_info.dat")

    if not os.path.exists(lines_file):
        print(f"Error: {lines_file} not found")
        sys.exit(1)
    if not os.path.exists(cont_file):
        print(f"Error: {cont_file} not found")
        sys.exit(1)

    print(f"Reading emission lines: {lines_file}")
    lines_grid = read_fsps_grid(lines_file)

    print(f"Reading continuum: {cont_file}")
    cont_grid = read_fsps_grid(cont_file)

    # Note: lines and continuum may have DIFFERENT axis grids
    # (different CLOUDY runs with different metallicity/age sampling)
    axes_match = (
        np.allclose(lines_grid["logZ"], cont_grid["logZ"])
        and np.allclose(lines_grid["age_yr"], cont_grid["age_yr"])
        and np.allclose(lines_grid["logU"], cont_grid["logU"])
    )
    if not axes_match:
        print("  Note: lines and continuum have different grid axes "
              "(stored independently)")

    # Read line names if available
    line_names = []
    if os.path.exists(emlines_file):
        emlines_info = read_emlines_info(emlines_file)
        # Match by wavelength (closest match within 1 Angstrom)
        for wl in lines_grid["wavelength"]:
            matched = False
            for ref_wl, name in emlines_info:
                if abs(wl - ref_wl) < 1.0:
                    line_names.append(name)
                    matched = True
                    break
            if not matched:
                line_names.append(f"unknown_{wl:.1f}A")
    else:
        line_names = [f"line_{wl:.1f}A" for wl in lines_grid["wavelength"]]

    # Write HDF5
    print(f"Writing: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # --- Lines group (with own axes) ---
        lines_grp = f.create_group("lines")

        # Lines axes
        lines_axes = lines_grp.create_group("axes")
        log_age_lines = np.log10(lines_grid["age_yr"])
        lines_axes.create_dataset("log_age_yr", data=log_age_lines)
        lines_axes["log_age_yr"].attrs["description"] = "log10(age / yr)"
        lines_axes.create_dataset("log_met", data=lines_grid["logZ"])
        lines_axes["log_met"].attrs["description"] = (
            "log10(Z), absolute metallicity"
        )
        lines_axes.create_dataset("log_U", data=lines_grid["logU"])
        lines_axes["log_U"].attrs["description"] = (
            "log10(ionization parameter U)"
        )

        lines_grp.create_dataset(
            "wavelength", data=lines_grid["wavelength"]
        )
        lines_grp["wavelength"].attrs["units"] = "Angstrom"
        lines_grp["wavelength"].attrs["description"] = (
            "Rest-frame vacuum wavelength of emission lines"
        )

        dt = h5py.special_dtype(vlen=str)
        names_ds = lines_grp.create_dataset(
            "names", (len(line_names),), dtype=dt
        )
        for i, name in enumerate(line_names):
            names_ds[i] = name

        lines_grp.create_dataset("luminosity", data=lines_grid["data"])
        lines_grp["luminosity"].attrs["units"] = "Lsun / Q_H"
        lines_grp["luminosity"].attrs["shape"] = (
            "(n_met, n_age, n_logU, n_lines)"
        )
        lines_grp["luminosity"].attrs["description"] = (
            "Line luminosity per ionizing photon rate. "
            "Multiply by Q_H(age, Z) to get L_line in Lsun."
        )

        # --- Continuum group (with own axes) ---
        cont_grp = f.create_group("continuum")

        cont_axes = cont_grp.create_group("axes")
        log_age_cont = np.log10(cont_grid["age_yr"])
        cont_axes.create_dataset("log_age_yr", data=log_age_cont)
        cont_axes["log_age_yr"].attrs["description"] = "log10(age / yr)"
        cont_axes.create_dataset("log_met", data=cont_grid["logZ"])
        cont_axes["log_met"].attrs["description"] = (
            "log10(Z), absolute metallicity"
        )
        cont_axes.create_dataset("log_U", data=cont_grid["logU"])
        cont_axes["log_U"].attrs["description"] = (
            "log10(ionization parameter U)"
        )

        cont_grp.create_dataset(
            "wavelength", data=cont_grid["wavelength"]
        )
        cont_grp["wavelength"].attrs["units"] = "Angstrom"
        cont_grp["wavelength"].attrs["description"] = (
            "Nebular continuum wavelength grid"
        )

        cont_grp.create_dataset("luminosity", data=cont_grid["data"])
        cont_grp["luminosity"].attrs["units"] = "Lsun_Hz / Q_H"
        cont_grp["luminosity"].attrs["shape"] = (
            "(n_met, n_age, n_logU, n_wave)"
        )
        cont_grp["luminosity"].attrs["description"] = (
            "Nebular continuum luminosity density per ionizing photon rate. "
            "Multiply by Q_H(age, Z) to get L_nu in Lsun/Hz."
        )

        # --- Root metadata ---
        f.attrs["source"] = "FSPS (Conroy & Gunn 2010)"
        f.attrs["cloudy_reference"] = "Byler et al. 2017"
        f.attrs["isoc_type"] = isoc
        f.attrs["dust_variant"] = dust_variant
        f.attrs["cloudy_dust"] = dust_variant == "WD"
        f.attrs["axes_match"] = axes_match
        f.attrs["n_lines"] = len(lines_grid["wavelength"])
        f.attrs["n_wave_cont"] = len(cont_grid["wavelength"])
        f.attrs["description"] = (
            "CLOUDY photoionization grids for nebular emission, "
            "converted from FSPS ASCII format to tengri HDF5. "
            "Lines and continuum may have different grid axes."
        )

    # Print summary
    print(f"\nGrid summary:")
    print(f"  Metallicities: {len(lines_grid['logZ'])} "
          f"[{lines_grid['logZ'][0]:.2f} to {lines_grid['logZ'][-1]:.2f}]")
    print(f"  Ages: {len(lines_grid['age_yr'])} "
          f"[{lines_grid['age_yr'][0]:.0e} to {lines_grid['age_yr'][-1]:.0e} yr]")
    print(f"  log U: {len(lines_grid['logU'])} "
          f"[{lines_grid['logU'][0]:.1f} to {lines_grid['logU'][-1]:.1f}]")
    print(f"  Emission lines: {len(lines_grid['wavelength'])}")
    print(f"  Continuum wavelengths: {len(cont_grid['wavelength'])}")
    print(f"\nKey lines:")
    key_lines = {
        "Ly-alpha": 1215.67,
        "H-alpha": 6562.80,
        "H-beta": 4861.33,
        "[OIII] 5007": 5006.84,
        "[OII] 3727": 3726.03,
        "[NII] 6583": 6583.45,
    }
    for name, target_wl in key_lines.items():
        idx = np.argmin(np.abs(lines_grid["wavelength"] - target_wl))
        actual_wl = lines_grid["wavelength"][idx]
        if abs(actual_wl - target_wl) < 5.0:
            print(f"  {name}: {actual_wl:.2f} A (index {idx})")

    print(f"\nWrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSPS CLOUDY grids to tengri HDF5"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with ZAU_*.lines/cont files "
        "(default: data/cloudy_raw/ or $SPS_HOME/nebular/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--isoc",
        default="mist",
        choices=["mist", "prsc", "pdva", "bpss"],
        help="Isochrone type (default: mist)",
    )
    parser.add_argument(
        "--dust",
        default="ND",
        choices=["ND", "WD"],
        help="Dust variant: ND=no dust, WD=with dust depletion (default: ND)",
    )
    parser.add_argument(
        "--sps-home",
        action="store_true",
        help="Read from $SPS_HOME/nebular/ instead of data/cloudy_raw/",
    )
    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
    elif args.sps_home:
        sps_home = os.environ.get("SPS_HOME")
        if not sps_home:
            print("Error: $SPS_HOME not set")
            sys.exit(1)
        input_dir = os.path.join(sps_home, "nebular")
    else:
        # Default: look in data/cloudy_raw/ relative to repo root
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        input_dir = str(repo_root / "data" / "cloudy_raw")

    dust_suffix = f"_{args.dust.lower()}" if args.dust != "ND" else ""
    output_path = os.path.join(
        args.output_dir, f"cloudy_grid_{args.isoc}{dust_suffix}.h5"
    )

    convert(input_dir, output_path, args.isoc, args.dust)


if __name__ == "__main__":
    main()
