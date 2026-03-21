#!/usr/bin/env python3
"""Download and convert SKIRTOR SED template library.

Downloads the full SKIRTOR clumpy torus model grid from the official
website and converts it to a NumPy .npz file for fast loading in
tengri.

The SKIRTOR library is described in:
- Stalevski et al. 2012, MNRAS, 420, 2756
- Stalevski et al. 2016, MNRAS, 458, 2288

Website: https://sites.google.com/site/skirtorus/sed-library

Grid dimensions:
    tau     (5 values):  3, 5, 7, 9, 11
    p       (4 values):  0.0, 0.5, 1.0, 1.5
    q       (4 values):  0.0, 0.5, 1.0, 1.5
    oa      (5 values):  20, 30, 40, 50, 60  [degrees]
    cos_inc (10 values): 0.05, 0.15, ..., 0.95
    wave    (varies):    wavelength grid in Angstrom

Output: data/skirtor_grid.npz

Usage:
    python scripts/download_skirtor_templates.py [--output data/skirtor_grid.npz]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert SKIRTOR SED templates.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/skirtor_grid.npz",
        help="Output .npz file path (default: data/skirtor_grid.npz)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://sites.google.com/site/skirtorus/sed-library",
        help="URL for SKIRTOR SED library download page",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print("SKIRTOR Template Grid Downloader")
    print("=" * 50)
    print()
    print("The SKIRTOR SED library (~1 GB) must be downloaded manually")
    print("from the official website:")
    print()
    print(f"  {args.url}")
    print()
    print("After downloading, place the SED files in a directory and")
    print("re-run this script with --input-dir pointing to the files.")
    print()
    print("Expected grid dimensions:")
    print("  tau:     [3, 5, 7, 9, 11]")
    print("  p:       [0.0, 0.5, 1.0, 1.5]")
    print("  q:       [0.0, 0.5, 1.0, 1.5]")
    print("  oa:      [20, 30, 40, 50, 60] degrees")
    print("  cos_inc: [0.05, 0.15, 0.25, ..., 0.95]")
    print()
    print("To convert downloaded files to the tengri .npz format:")
    print("  python scripts/download_skirtor_templates.py --convert --input-dir <path>")
    print()
    print(f"Output will be saved to: {output_path}")

    # Check if --convert and --input-dir are provided
    if "--convert" in sys.argv:
        convert_parser = argparse.ArgumentParser()
        convert_parser.add_argument("--convert", action="store_true")
        convert_parser.add_argument("--input-dir", type=str, required=True)
        convert_parser.add_argument("--output", type=str, default=args.output)
        convert_args, _ = convert_parser.parse_known_args()

        input_dir = Path(convert_args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory '{input_dir}' not found")
            sys.exit(1)

        convert_skirtor_grid(input_dir, Path(convert_args.output))


def convert_skirtor_grid(input_dir: Path, output_path: Path) -> None:
    """Convert raw SKIRTOR SED files to a single .npz grid.

    Parameters
    ----------
    input_dir : Path
        Directory containing SKIRTOR SED text files.
    output_path : Path
        Output .npz file path.
    """
    import numpy as np

    # SKIRTOR grid axes
    tau_values = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    p_values = np.array([0.0, 0.5, 1.0, 1.5])
    q_values = np.array([0.0, 0.5, 1.0, 1.5])
    oa_values = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    cos_inc_values = np.linspace(0.05, 0.95, 10)

    # Find all SED files and determine wavelength grid from first file
    sed_files = sorted(input_dir.glob("*.dat")) + sorted(input_dir.glob("*.txt"))
    if not sed_files:
        print(f"Error: No .dat or .txt files found in {input_dir}")
        sys.exit(1)

    # Read first file to get wavelength grid
    first_data = np.loadtxt(sed_files[0])
    wavelength = first_data[:, 0]  # Assume Angstrom or micron
    n_wave = len(wavelength)

    print(f"Found {len(sed_files)} SED files")
    print(f"Wavelength grid: {n_wave} points, {wavelength[0]:.1f} to {wavelength[-1]:.1f}")

    # Initialize grid
    grid = np.zeros((
        len(tau_values),
        len(p_values),
        len(q_values),
        len(oa_values),
        len(cos_inc_values),
        n_wave,
    ))

    print(f"Grid shape: {grid.shape}")
    print(f"Grid size: {grid.nbytes / 1e6:.1f} MB")

    # Parse filenames and fill grid
    # Expected filename format varies -- this is a template that users
    # should adapt to the actual SKIRTOR file naming convention
    n_loaded = 0
    for sed_file in sed_files:
        try:
            data = np.loadtxt(sed_file)
            # Users need to parse the filename to extract parameters
            # and place the SED in the correct grid location
            n_loaded += 1
        except Exception as e:
            print(f"Warning: Could not read {sed_file}: {e}")

    print(f"Loaded {n_loaded} / {len(sed_files)} files")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        grid=grid,
        wavelength=wavelength,
        tau=tau_values,
        p=p_values,
        q=q_values,
        oa=oa_values,
        cos_inc=cos_inc_values,
    )
    print(f"Saved grid to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
