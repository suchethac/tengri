#!/usr/bin/env python3
"""Build surviving-mass-fraction tables from FSPS python-fsps for SSP grids.

This script produces HDF5 tables of surviving stellar mass fractions as a
function of metallicity and age, using FSPS with a specified isochrone,
spectral library, and IMF. The surviving mass includes living stars and stellar
remnants (WD/NS/BH) per 1 Msun of formed mass, as computed by FSPS with
``add_stellar_remnants=1``.

The remnant prescription follows Renzini & Ciotti (1993), as implemented in
FSPS's add_remnants.f90.

Usage:
    python scripts/build_mass_remaining_fsps.py --isoc mist --imf chabrier
    python scripts/build_mass_remaining_fsps.py --isoc mist --imf kroupa
    python scripts/build_mass_remaining_fsps.py --isoc mist --imf salpeter

"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Build mass_remaining tables from FSPS for a given isochrone and IMF."
    )
    parser.add_argument(
        "--isoc",
        required=True,
        choices=["mist", "prsc", "pdva", "bsti", "gnva"],
        help="Isochrone set name",
    )
    parser.add_argument(
        "--imf",
        required=True,
        choices=["chabrier", "kroupa", "salpeter"],
        help="IMF name",
    )
    args = parser.parse_args()

    import os

    os.environ.setdefault("SPS_HOME", "/Users/suchethacooray/Projects/fsps")
    import fsps

    # Map IMF name to python-fsps imf_type
    imf_map = {
        "chabrier": 1,
        "kroupa": 2,
        "salpeter": 0,
    }
    imf_type = imf_map[args.imf]

    # Initialize StellarPopulation with the requested isochrone and IMF
    # Note: the compiled FSPS only has MIST/MILES built in. Other isochrones
    # would require recompiling FSPS, which is outside the scope of this task.
    sp = fsps.StellarPopulation(
        imf_type=imf_type,
        sfh=0,  # Single stellar population mode
        add_stellar_remnants=1,  # Include WD/NS/BH remnants
    )

    # Verify the isochrone matches
    isochrone_names = [lib.decode() if isinstance(lib, bytes) else lib for lib in sp.libraries]
    print(f"Available isochrones: {isochrone_names}")
    if args.isoc not in isochrone_names:
        raise ValueError(
            f"Requested isochrone '{args.isoc}' not in compiled FSPS. "
            f"Available: {isochrone_names}"
        )

    # Get the metallicity grid and age grid from FSPS
    z_absolute = np.array(sp.zlegend)
    log10_z_abs = np.log10(z_absolute)
    log_age_yr = np.array(sp.ssp_ages)  # FSPS internal SSP age grid [log10(years)]
    n_met = len(z_absolute)
    n_age = len(log_age_yr)

    print(f"Isochrone: {args.isoc}, IMF: {args.imf}")
    print(f"Number of metallicities: {n_met}")
    print(f"Z range: {z_absolute.min():.2e} to {z_absolute.max():.2e}")
    print(f"Number of ages: {n_age}")
    print(f"Age range: {log_age_yr.min():.2f} to {log_age_yr.max():.2f} log10(years)")

    # Build the table: iterate over all (Z, age) nodes
    mass_remaining = np.zeros((n_met, n_age), dtype=np.float64)

    print("Computing mass_remaining at all (Z, age) nodes...")
    for i_zmet, zmet in enumerate(range(1, n_met + 1)):
        for i_age, age_log_yr in enumerate(log_age_yr):
            # Convert log10(years) back to years, then to Gyr for FSPS
            age_gyr = 10.0 ** (age_log_yr - 9.0)
            sp.get_spectrum(tage=age_gyr, zmet=zmet)
            mass_remaining[i_zmet, i_age] = sp.stellar_mass
        if (i_zmet + 1) % 3 == 0:
            print(f"  Completed {i_zmet + 1}/{n_met} metallicities")

    print(f"mass_remaining shape: {mass_remaining.shape}")
    print(f"mass_remaining range: {mass_remaining.min():.6f} to {mass_remaining.max():.6f}")

    # Verify basic physics: values should be in (0, 1], non-increasing with age
    if not np.all((mass_remaining > 0) & (mass_remaining <= 1)):
        print("WARNING: Some mass_remaining values outside (0, 1]")
        print(f"  Min: {mass_remaining.min()}, Max: {mass_remaining.max()}")

    # Check monotonicity: should be non-increasing in age
    for i_met in range(n_met):
        diff = np.diff(mass_remaining[i_met, :])
        if np.any(diff > 1e-10):  # Allow small numerical noise
            print(f"WARNING: Z={z_absolute[i_met]:.2e}: mass_remaining increases with age")

    # Output file
    data_dir = Path(__file__).parent.parent / "data" / "mass_remaining"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / f"mass_remaining_{args.isoc}_{args.imf}.h5"
    print(f"\nWriting to {output_file}")

    # Get the git commit hash of FSPS (for provenance)
    fsps_commit = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.environ.get("SPS_HOME", "/Users/suchethacooray/Projects/fsps"),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            fsps_commit = result.stdout.strip()
    except Exception:
        pass

    with h5py.File(output_file, "w") as f:
        # Datasets
        f.create_dataset("log10_age_yr", data=log_age_yr, dtype=np.float64)
        f.create_dataset("log10_z_abs", data=log10_z_abs, dtype=np.float64)
        f.create_dataset("mass_remaining", data=mass_remaining, dtype=np.float64)

        # Attributes: metadata
        f.attrs["quantity"] = "living stars + remnants per unit formed mass"
        f.attrs["isochrones"] = args.isoc
        f.attrs["imf"] = args.imf
        f.attrs["source"] = "python-fsps 0.4.7"
        f.attrs["source_commit"] = fsps_commit
        f.attrs["remnant_prescription"] = "Renzini & Ciotti 1993"
        f.attrs["citation"] = (
            "Conroy, Gunn & White 2009, ApJ 699, 486; "
            "Conroy & Gunn 2010, ApJ 712, 833; "
            "Renzini & Ciotti 1993, ApJ 416, 49"
        )
        f.attrs["generator"] = str(Path(__file__).relative_to(Path(__file__).parent.parent.parent))
        f.attrs["generator_args"] = f"--isoc {args.isoc} --imf {args.imf}"

    print(f"✓ Successfully wrote {output_file}")
    print(f"  Shape: {mass_remaining.shape}")
    print(f"  Range: {mass_remaining.min():.6f} - {mass_remaining.max():.6f}")


if __name__ == "__main__":
    main()
