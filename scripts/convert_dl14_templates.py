#!/usr/bin/env python3
"""Convert DL14 dust emission templates to diffsed HDF5 grid.

Reads the ASCII template files downloaded from CIGALE's GitLab repository
(originally from Draine's models with 2014 updates) and builds a single
HDF5 grid for fast interpolation in diffsed.

DL14 extends DL07 with:
- Variable alpha parameter (power-law slope 1.0-3.0, was fixed at 2 in DL07)
- Extended q_PAH range (0.47-7.32% vs 0.47-4.58%)
- Extended U_min range (0.1-50 vs 0.1-25)
- U_max raised to 10^7 (was 10^6)

Download source (via scripts/download_dl14_templates.py):
    CIGALE GitLab: gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder/dl2014

Usage:
    python scripts/convert_dl14_templates.py [--input-dir data/dl14_raw]

Output: data/dl14_templates.h5
"""

import argparse
import io
import os
import sys
from pathlib import Path

import h5py
import numpy as np


# DL14 dust models and their q_PAH values (percent)
QPAH_MAP = {
    "000": 0.47,
    "010": 1.12,
    "020": 1.77,
    "030": 2.50,
    "040": 3.19,
    "050": 3.90,
    "060": 4.58,
    "070": 5.26,
    "080": 5.95,
    "090": 6.63,
    "100": 7.32,
}

# Dust mass per H atom (Mdust/MH) for each model
MDMH = {
    "000": 0.0100,
    "010": 0.0100,
    "020": 0.0101,
    "030": 0.0102,
    "040": 0.0102,
    "050": 0.0103,
    "060": 0.0104,
    "070": 0.0105,
    "080": 0.0106,
    "090": 0.0107,
    "100": 0.0108,
}

# U_min values in the grid (as strings matching directory names)
UMIN_STRS = [
    "0.100", "0.120", "0.150", "0.170", "0.200", "0.250",
    "0.300", "0.350", "0.400", "0.500", "0.600", "0.700",
    "0.800", "1.000", "1.200", "1.500", "1.700", "2.000",
    "2.500", "3.000", "3.500", "4.000", "5.000", "6.000",
    "7.000", "8.000", "10.00", "12.00", "15.00", "17.00",
    "20.00", "25.00", "30.00", "35.00", "40.00", "50.00",
]
UMIN_VALUES = [float(u) for u in UMIN_STRS]

# Alpha values (power-law slope of radiation field distribution)
ALPHA_STRS = [
    "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7",
    "1.8", "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5",
    "2.6", "2.7", "2.8", "2.9", "3.0",
]
ALPHA_VALUES = [float(a) for a in ALPHA_STRS]

# Number of spectral points in the template files (last N lines are spectrum)
N_SPEC_LINES = 1001


def read_dl14_spectrum(filepath: str) -> tuple:
    """Read a single DL14 template file.

    The file format (from Draine) has a header with broadband photometry,
    followed by the continuous spectrum in the last 1001 lines.

    Returns
    -------
    wavelength_um : array (n_wave,)
        Wavelengths in microns (ascending order).
    j_nu : array (n_wave,)
        Specific intensity in Jy cm^2 sr^-1 H^-1 (ascending wavelength).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # The last N_SPEC_LINES lines are the continuous spectrum
    spec_lines = lines[-N_SPEC_LINES:]
    data_str = "".join(spec_lines)
    data = np.genfromtxt(io.BytesIO(data_str.encode()), usecols=(0, 2))

    wavelength_um = data[:, 0]
    j_nu = data[:, 1]  # total j_nu (column index 2 = j_nu(tot))

    # Files have wavelengths in decreasing order; reverse to ascending
    wavelength_um = wavelength_um[::-1]
    j_nu = j_nu[::-1]

    return wavelength_um, j_nu


def convert(input_dir: str, output_path: str) -> None:
    """Convert DL14 templates to diffsed HDF5 grid.

    Output datasets:
        /wavelength     (n_wave,) in Angstrom
        /qpah_grid      (n_qpah,)
        /umin_grid      (n_umin,)
        /alpha_grid     (n_alpha,)
        /single_u       (n_qpah, n_umin, n_wave)
        /powerlaw       (n_qpah, n_umin, n_alpha, n_wave)
    """
    model_keys = sorted(QPAH_MAP.keys())
    qpah_values = [QPAH_MAP[k] for k in model_keys]

    n_qpah = len(qpah_values)
    n_umin = len(UMIN_VALUES)
    n_alpha = len(ALPHA_VALUES)

    # Read one file to get wavelength grid
    test_model = model_keys[0]
    test_umin = UMIN_STRS[0]
    test_dir = f"U{test_umin}_{test_umin}_MW3.1_{test_model}"
    test_path = os.path.join(input_dir, test_dir, "spec_1.0.dat")

    if not os.path.exists(test_path):
        print(f"Error: template file not found: {test_path}")
        print("Run scripts/download_dl14_templates.py first.")
        sys.exit(1)

    wave_um, _ = read_dl14_spectrum(test_path)
    n_wave = len(wave_um)
    print(f"Wavelength grid: {n_wave} points, {wave_um[0]:.4f} - {wave_um[-1]:.4f} um")

    # Allocate arrays
    single_u = np.zeros((n_qpah, n_umin, n_wave))
    powerlaw = np.zeros((n_qpah, n_umin, n_alpha, n_wave))

    found = 0
    missing = 0

    for iq, model_key in enumerate(model_keys):
        for iu, umin_str in enumerate(UMIN_STRS):
            # --- Single-U template ---
            single_dir = f"U{umin_str}_{umin_str}_MW3.1_{model_key}"
            single_path = os.path.join(input_dir, single_dir, "spec_1.0.dat")

            if os.path.exists(single_path):
                _, j_nu = read_dl14_spectrum(single_path)
                single_u[iq, iu, :] = j_nu
                found += 1
            else:
                missing += 1
                if iu == 0 and iq == 0:
                    print(f"  Missing single: {single_path}")

            # --- Power-law templates (one per alpha) ---
            pl_dir = f"U{umin_str}_1e7_MW3.1_{model_key}"
            for ia, alpha_str in enumerate(ALPHA_STRS):
                pl_path = os.path.join(input_dir, pl_dir, f"spec_{alpha_str}.dat")

                if os.path.exists(pl_path):
                    _, j_nu = read_dl14_spectrum(pl_path)
                    powerlaw[iq, iu, ia, :] = j_nu
                    found += 1
                else:
                    missing += 1
                    if iu == 0 and iq == 0 and ia == 0:
                        print(f"  Missing powerlaw: {pl_path}")

        print(f"\r  Read model {model_key} (q_PAH={qpah_values[iq]:.2f}%)", end="", flush=True)

    print(f"\n\nTemplates read: {found} found, {missing} missing")

    if found == 0:
        print("Error: no templates found. Check input directory.")
        sys.exit(1)

    # Normalize each template to unit integral over frequency (shape only).
    # This follows the same convention as the DL07 conversion script.
    # j_nu is in Jy cm^2 sr^-1 H^-1; we only care about the shape.
    c_um = 2.99792458e14  # c in um/s
    nu_from_um = c_um / wave_um  # Hz (descending since wave is ascending)

    print("Normalizing templates...")
    for iq in range(n_qpah):
        for iu in range(n_umin):
            # Single-U
            total = -np.trapz(single_u[iq, iu, :], nu_from_um)
            if total > 0:
                single_u[iq, iu, :] /= total

            # Power-law (each alpha separately)
            for ia in range(n_alpha):
                total = -np.trapz(powerlaw[iq, iu, ia, :], nu_from_um)
                if total > 0:
                    powerlaw[iq, iu, ia, :] /= total

    # Write HDF5
    wave_aa = wave_um * 1e4  # convert to Angstrom for diffsed convention

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Writing: {output_path}")

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

        f.create_dataset("alpha_grid", data=np.array(ALPHA_VALUES))
        f["alpha_grid"].attrs["description"] = (
            "Power-law slope of radiation field distribution "
            "(dM/dU ~ U^{-alpha}, range 1.0-3.0)"
        )

        f.create_dataset("single_u", data=single_u)
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f["single_u"].attrs["description"] = (
            "Single-U template: dust heated by U=U_min only. "
            "Normalized to unit frequency integral (shape only)."
        )

        f.create_dataset("powerlaw", data=powerlaw)
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_alpha, n_wave)"
        f["powerlaw"].attrs["description"] = (
            "Power-law template: dust heated by U^{-alpha} from U_min to U_max=1e7. "
            "Normalized to unit frequency integral (shape only)."
        )

        f.attrs["source"] = (
            "Draine & Li 2007 (ApJ 657, 810) with 2014 updates: "
            "variable alpha, extended q_PAH & U_min ranges, U_max=1e7"
        )
        f.attrs["cigale_source"] = (
            "https://gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder/dl2014"
        )
        f.attrs["n_qpah"] = n_qpah
        f.attrs["n_umin"] = n_umin
        f.attrs["n_alpha"] = n_alpha
        f.attrs["n_wave"] = n_wave
        f.attrs["umax_powerlaw"] = 1e7
        f.attrs["description"] = (
            "DL14 IR emission templates for diffsed. "
            "Usage: j_nu = (1-gamma)*single_u[iq,iu] + gamma*powerlaw[iq,iu,ia], "
            "then multiply by L_absorbed for energy balance normalization."
        )

    # Summary
    single_mb = single_u.nbytes / 1e6
    pl_mb = powerlaw.nbytes / 1e6
    print(f"\nDL14 template grid:")
    print(f"  q_PAH: {qpah_values} ({n_qpah} values)")
    print(f"  U_min: {n_umin} values ({UMIN_VALUES[0]} - {UMIN_VALUES[-1]})")
    print(f"  alpha: {n_alpha} values ({ALPHA_VALUES[0]} - {ALPHA_VALUES[-1]})")
    print(f"  Wavelength: {n_wave} points ({wave_um[0]:.4f} - {wave_um[-1]:.4f} um)")
    print(f"  Grid size: {single_mb:.1f} MB (single) + {pl_mb:.1f} MB (power-law)")
    print(f"\nWrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DL14 dust templates to diffsed HDF5"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with DL14 template folders (default: data/dl14_raw/)",
    )
    parser.add_argument(
        "--output",
        default="data/dl14_templates.h5",
        help="Output HDF5 file (default: data/dl14_templates.h5)",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        args.input_dir = str(repo_root / "data" / "dl14_raw")

    convert(args.input_dir, args.output)


if __name__ == "__main__":
    main()
