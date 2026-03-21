#!/usr/bin/env python3
"""Convert Draine & Li 2007 dust emission templates to tengri HDF5.

Reads the ASCII template files from the DL07spec tarball and builds a
single HDF5 grid for fast interpolation in tengri.

Download source:
    https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz

Usage:
    python scripts/convert_dl07_templates.py [--input-dir data/dl07_raw]

Output: data/dl07_templates.h5
"""

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np


# DL07 dust models and their q_PAH values (percent)
# From Draine & Li 2007 Table 1
DUST_MODELS = {
    "MW3.1_00": 0.47,
    "MW3.1_10": 1.12,
    "MW3.1_20": 1.77,
    "MW3.1_30": 2.50,
    "MW3.1_40": 3.19,
    "MW3.1_50": 3.90,
    "MW3.1_60": 4.58,
    "LMC2_00": 0.75,
    "LMC2_05": 1.49,
    "LMC2_10": 2.37,
    "smc": 0.10,
}

# U_min values present in the grid
# U_min values that actually exist as directories in the DL07 tarball
UMIN_VALUES = [
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80,
    1.00, 1.20, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00,
    8.00, 12.0, 15.0, 20.0, 25.0,
]

# U_max values (power-law upper bound)
UMAX_VALUES = ["1e2", "1e3", "1e4", "1e5", "1e6"]


def _umin_to_str(umin: float) -> str:
    """Convert U_min float to the DL07 naming convention.

    0.10 → 'U0.10', 1.00 → 'U1.00', 12.0 → 'U12.0', 25.0 → 'U25.0'
    """
    if umin >= 10:
        return f"U{umin:.1f}"
    return f"U{umin:.2f}"


def read_dl07_template(filepath: str) -> dict:
    """Read a single DL07 template file.

    Returns
    -------
    dict with keys:
        wavelength_um: (n_wave,) in microns
        j_nu: (n_wave,) in Jy cm^2 sr^-1 H^-1
        nu_pnu: (n_wave,) in erg s^-1 H^-1
        umin: float
        umax: float or str
        dust_model: str
        mean_u: float
        power_per_h: float
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse header
    umin = umax = beta = mean_u = power_h = None
    data_start = None
    for i, line in enumerate(lines):
        if "Umin , Umax, beta" in line:
            parts = line.split("=")[0].split()
            umin = float(parts[0])
            umax = float(parts[1])
            beta = float(parts[2])
        elif "<U>" in line:
            mean_u = float(line.split("=")[0].strip())
        elif "power/H" in line:
            power_h = float(line.split("=")[0].strip())
        elif line.strip().startswith("(um)"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data start in {filepath}")

    # Read data: wavelength(um), nu*P_nu, j_nu [, optional band name]
    # The file has broadband photometry first (with band names in col 4),
    # then the continuous spectrum (3 columns only, wavelength descending).
    # We skip the broadband lines and keep only the spectrum.
    wavelengths = []
    j_nu_values = []
    nu_pnu_values = []

    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        # Skip broadband photometry lines (have 4+ columns with band name)
        if len(parts) >= 4 and not parts[3].replace(".", "").replace("-", "").replace("+", "").replace("E", "").replace("e", "").isdigit():
            continue
        try:
            wl = float(parts[0])
            nupnu = float(parts[1])
            jnu = float(parts[2])
            wavelengths.append(wl)
            nu_pnu_values.append(nupnu)
            j_nu_values.append(jnu)
        except ValueError:
            continue

    # Sort by wavelength ascending
    wavelengths = np.array(wavelengths)
    j_nu_values = np.array(j_nu_values)
    nu_pnu_values = np.array(nu_pnu_values)
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    j_nu_values = j_nu_values[sort_idx]
    nu_pnu_values = nu_pnu_values[sort_idx]

    return {
        "wavelength_um": wavelengths,
        "j_nu": j_nu_values,
        "nu_pnu": nu_pnu_values,
        "umin": umin,
        "umax": umax,
        "beta": beta,
        "mean_u": mean_u,
        "power_per_h": power_h,
    }


def convert(input_dir: str, output_path: str) -> None:
    """Convert DL07 templates to tengri HDF5 grid.

    The output grid has shape (n_qpah, n_umin, n_umax, n_wave) for j_nu.
    For the standard DL07 usage with gamma, you need:
      - Single-U template: file U{umin}_{umin}_model.txt (umax = umin)
      - Power-law template: file U{umin}_{umax}_model.txt
    Then: j_nu_total = (1-gamma) * j_nu_single + gamma * j_nu_powerlaw
    """
    # Focus on MW models (most commonly used in SED fitting)
    mw_models = {k: v for k, v in DUST_MODELS.items() if k.startswith("MW")}
    qpah_values = sorted(set(mw_models.values()))
    model_by_qpah = {v: k for k, v in mw_models.items()}

    # Read one file to get wavelength grid
    test_file = None
    for d in sorted(os.listdir(input_dir)):
        dpath = os.path.join(input_dir, d)
        if os.path.isdir(dpath):
            for f in os.listdir(dpath):
                if f.endswith(".txt"):
                    test_file = os.path.join(dpath, f)
                    break
        if test_file:
            break

    if test_file is None:
        print(f"Error: no template files found in {input_dir}")
        sys.exit(1)

    test_data = read_dl07_template(test_file)
    wave_um = test_data["wavelength_um"]
    n_wave = len(wave_um)
    print(f"Wavelength grid: {n_wave} points, {wave_um[0]:.3f} - {wave_um[-1]:.3f} um")

    # Build grids for single-U and power-law templates
    n_qpah = len(qpah_values)
    n_umin = len(UMIN_VALUES)

    # Single-U templates: j_nu(qpah, umin, wave) — for the (1-gamma) component
    single_u = np.zeros((n_qpah, n_umin, n_wave))
    # Power-law templates: j_nu(qpah, umin, wave) — for the gamma component
    # Using Umax=1e6 (standard DL07)
    powerlaw = np.zeros((n_qpah, n_umin, n_wave))

    found = 0
    missing = 0

    for iq, qpah in enumerate(qpah_values):
        model_name = model_by_qpah[qpah]
        for iu, umin in enumerate(UMIN_VALUES):
            u_str = _umin_to_str(umin)  # e.g., "U1.00" or "U12.0"

            # Single-U file: U{umin}/U{umin}_{umin_val}_{model}.txt
            # Note: second field has no "U" prefix
            u_val = u_str[1:]  # strip leading "U"
            single_fname = f"{u_str}_{u_val}_{model_name}.txt"
            single_path = os.path.join(input_dir, u_str, single_fname)

            if os.path.exists(single_path):
                data = read_dl07_template(single_path)
                single_u[iq, iu, :] = np.interp(wave_um, data["wavelength_um"], data["j_nu"])
                found += 1
            else:
                missing += 1
                if iu == 0:
                    print(f"  Missing single: {single_path}")

            # Power-law file: U{umin}/U{umin}_1e6_{model}.txt
            pl_fname = f"{u_str}_1e6_{model_name}.txt"
            pl_path = os.path.join(input_dir, u_str, pl_fname)

            if os.path.exists(pl_path):
                data = read_dl07_template(pl_path)
                powerlaw[iq, iu, :] = np.interp(wave_um, data["wavelength_um"], data["j_nu"])
                found += 1
            else:
                missing += 1
                if iu == 0:
                    print(f"  Missing powerlaw: {pl_path}")

    print(f"Templates read: {found} found, {missing} missing")

    # Convert j_nu from Jy cm^2 sr^-1 H^-1 to Lsun/Hz per Msun_dust
    # Following CIGALE/FSPS convention:
    # j_nu is emissivity per H atom. To get per unit dust mass:
    # j_nu_per_Mdust = j_nu / (m_H * (M_dust/M_gas)) * 4*pi
    # But for energy-balance normalization, we just need the SED *shape*
    # normalized so integral = 1. The absolute scaling comes from L_absorbed.

    # Normalize each template to unit integral (shape only)
    c_um = 2.99792458e14  # c in um/s
    nu_from_um = c_um / wave_um  # Hz

    for iq in range(n_qpah):
        for iu in range(n_umin):
            for grid in [single_u, powerlaw]:
                total = -np.trapz(grid[iq, iu, :], nu_from_um)
                if total > 0:
                    grid[iq, iu, :] /= total

    # Write HDF5
    print(f"Writing: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    wave_aa = wave_um * 1e4  # convert to Angstrom for tengri convention

    with h5py.File(output_path, "w") as f:
        f.create_dataset("wavelength", data=wave_aa)
        f["wavelength"].attrs["units"] = "Angstrom"

        f.create_dataset("qpah_grid", data=np.array(qpah_values))
        f["qpah_grid"].attrs["units"] = "percent"
        f["qpah_grid"].attrs["description"] = "PAH mass fraction (%)"

        f.create_dataset("umin_grid", data=np.array(UMIN_VALUES))
        f["umin_grid"].attrs["description"] = (
            "Minimum radiation field intensity (Mathis ISRF units)"
        )

        f.create_dataset("single_u", data=single_u)
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f["single_u"].attrs["description"] = (
            "Single-U template: dust heated by U=U_min only. "
            "Normalized to unit frequency integral (shape only)."
        )

        f.create_dataset("powerlaw", data=powerlaw)
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f["powerlaw"].attrs["description"] = (
            "Power-law template: dust heated by U^{-2} from U_min to 1e6. "
            "Normalized to unit frequency integral (shape only)."
        )

        f.attrs["source"] = "Draine & Li 2007, ApJ 657, 810"
        f.attrs["url"] = "https://www.astro.princeton.edu/~draine/dust/irem4/"
        f.attrs["dust_models"] = "MW3.1 (Milky Way R_V=3.1)"
        f.attrs["n_qpah"] = n_qpah
        f.attrs["n_umin"] = n_umin
        f.attrs["n_wave"] = n_wave
        f.attrs["umax_powerlaw"] = 1e6
        f.attrs["description"] = (
            "DL07 IR emission templates for tengri. "
            "Usage: j_nu = (1-gamma)*single_u[iq,iu] + gamma*powerlaw[iq,iu], "
            "then multiply by L_absorbed for energy balance normalization."
        )

    # Summary
    print(f"\nDL07 template grid:")
    print(f"  q_PAH: {qpah_values} ({n_qpah} values)")
    print(f"  U_min: {UMIN_VALUES} ({n_umin} values)")
    print(f"  Wavelength: {n_wave} points ({wave_um[0]:.3f} - {wave_um[-1]:.3f} um)")
    print(f"  Grid size: {single_u.nbytes / 1e6:.1f} MB (single) + "
          f"{powerlaw.nbytes / 1e6:.1f} MB (power-law)")
    print(f"\nWrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DL07 dust templates to tengri HDF5"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with DL07 template folders (default: data/dl07_raw/)",
    )
    parser.add_argument(
        "--output",
        default="data/dl07_templates.h5",
        help="Output HDF5 file (default: data/dl07_templates.h5)",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        args.input_dir = str(repo_root / "data" / "dl07_raw")

    convert(args.input_dir, args.output)


if __name__ == "__main__":
    main()
