#!/usr/bin/env python
"""
Build mass_remaining_bc03pdva94_chabrier.h5 from BC03 *.4color files.

BC03 (Bruzual & Charlot 2003, MNRAS 344, 1000) provides the stellar mass
remaining (living stars + remnants per unit formed mass) in column (11)
of the *.4color files. This script reads the six metallicity files
(m22..m72, corresponding to Z = 0.0001..0.05) and constructs the HDF5 table.

Remnant prescription: BC03's internal (different from FSPS Renzini & Ciotti).
Isochrones: Padova 1994.
Spectral library: STELIB.
IMF: Chabrier (lognormal + power law).
"""

import gzip
import numpy as np
import h5py
from pathlib import Path


BC03_FILES = {
    'm22': (0.0001, -4.0),
    'm32': (0.0004, -3.39794),
    'm42': (0.004, -2.39794),
    'm52': (0.008, -2.09691),
    'm62': (0.02, -1.6989701),
    'm72': (0.05, -1.30103),
}


def read_bc03_4color(filepath):
    """Read BC03 4color file and extract log-age-yr and M*_tot columns."""
    with gzip.open(filepath, 'rt') as f:
        lines = f.readlines()

    # Skip header lines (those starting with '#')
    data_lines = [line for line in lines if not line.startswith('#')]

    ages_yr = []
    masses = []

    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 11:  # Need at least 11 columns
            try:
                age_yr = float(parts[0])  # Column (1)
                mass_tot = float(parts[10])  # Column (11) is index 10
                ages_yr.append(age_yr)
                masses.append(mass_tot)
            except ValueError:
                continue

    return np.array(ages_yr), np.array(masses)


def main():
    tengri_root = Path('/Users/suchethacooray/Projects/tengri')
    bc03_dir = Path('/Users/suchethacooray/.claude/jobs/da5e6ce4/tmp/bc03/Stelib_Atlas/Chabrier_IMF')
    output_file = tengri_root / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read Tengri's BC03 grid to get the reference Z and age nodes
    print("Reading Tengri BC03 grid metadata...")
    with h5py.File(tengri_root / 'data' / 'bc03_pdva_stelib_chabrier.h5', 'r') as f:
        tengri_ages_gyr = f['ssp_lg_age_gyr'][:]
        tengri_z_lg = f['ssp_lgmet'][:]

    print(f"  Tengri ages (log Gyr): {len(tengri_ages_gyr)} points")
    print(f"  Tengri Z (log10): {tengri_z_lg}")

    # Read all BC03 files
    all_ages_yr = None
    mass_remaining_all = []
    z_log10_read = []

    for suffix in ['m22', 'm32', 'm42', 'm52', 'm62', 'm72']:
        z_abs, z_lg = BC03_FILES[suffix]
        filepath = bc03_dir / f'bc2003_hr_stelib_{suffix}_chab_ssp.4color.gz'

        print(f"\nReading {filepath.name}...")
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ages_yr, masses = read_bc03_4color(filepath)

        # Store first set of ages for reference
        if all_ages_yr is None:
            all_ages_yr = ages_yr
            print(f"  Found {len(ages_yr)} age points")
            print(f"  Age range: {ages_yr[0]:.6f} to {ages_yr[-1]:.6f} (log yr)")
        else:
            if len(ages_yr) != len(all_ages_yr):
                raise ValueError(f"Age mismatch: expected {len(all_ages_yr)}, got {len(ages_yr)}")

        z_log10_read.append(z_lg)
        mass_remaining_all.append(masses)
        print(f"  Z = {z_abs:.6f} (log10 Z = {z_lg:.6f})")
        print(f"  Mass range: {masses.min():.6f} to {masses.max():.6f}")

    # Validate Z
    z_lg_array = np.array(z_log10_read)
    print(f"\nValidating Z values...")
    print(f"  Expected: {tengri_z_lg}")
    print(f"  Got:      {z_lg_array}")
    if not np.allclose(z_lg_array, tengri_z_lg):
        raise ValueError("Z values do not match Tengri grid")

    # BC03 ages are in log(yr), convert to log(Gyr)
    age_lg_yr = np.array(all_ages_yr, dtype=np.float64)
    age_lg_gyr = age_lg_yr - 9

    print(f"\nAge axes:")
    print(f"  BC03 (log yr) first 5: {age_lg_yr[:5]}")
    print(f"  BC03 (log yr) last 5:  {age_lg_yr[-5:]}")
    print(f"  Tengri (log Gyr) first 5: {tengri_ages_gyr[:5]}")
    print(f"  Tengri (log Gyr) last 5: {tengri_ages_gyr[-5:]}")

    # Prepare output age grid: include -inf point from Tengri for compatibility
    output_ages = tengri_ages_gyr
    print(f"\nUsing Tengri's age axis ({len(output_ages)} points, including -inf)")

    # Construct mass array: (n_z, n_age)
    mass_array = np.empty((len(z_log10_read), len(output_ages)), dtype=np.float64)

    for iz, masses in enumerate(mass_remaining_all):
        # Interpolate BC03 masses to Tengri's age grid
        # BC03 ages in log(Gyr)
        bc03_lg_gyr = age_lg_yr - 9

        # Handle -inf: set to 1.0 (youngest SSP)
        if not np.isfinite(output_ages[0]):
            mass_array[iz, 0] = 1.0
            # Interpolate for finite ages
            finite_gyr = bc03_lg_gyr
            finite_masses = masses
            output_finite = output_ages[1:]  # Skip the -inf point
            interp_result = np.interp(output_finite, finite_gyr, finite_masses)
            mass_array[iz, 1:] = interp_result
        else:
            # All finite
            interp_result = np.interp(output_ages, bc03_lg_gyr, masses)
            mass_array[iz, :] = interp_result

    # Validate bounds
    print(f"\nValidating bounds...")
    assert np.all(mass_array > 0), f"Some masses <= 0: min = {mass_array.min()}"
    assert np.all(mass_array <= 1.0001), f"Some masses > 1: max = {mass_array.max()}"
    mass_array = np.clip(mass_array, 0, 1)

    # Check monotonicity
    mono_diff = np.diff(mass_array, axis=1)
    non_mono = np.sum(mono_diff > 1e-5)
    print(f"  All masses in [0, 1]")
    print(f"  Non-monotonic points: {non_mono}")
    print(f"  Max mass at youngest age: {mass_array[:, 0]}")

    # Write HDF5
    # Convert Tengri ages back to log(yr) for storage
    output_age_yr_lg = output_ages + 9

    print(f"\nWriting {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('log10_age_yr', data=output_age_yr_lg, dtype=np.float64)
        f.create_dataset('log10_z_abs', data=z_lg_array, dtype=np.float64)
        f.create_dataset('mass_remaining', data=mass_array, dtype=np.float64)

        f.attrs['quantity'] = 'living stars + remnants per unit formed mass'
        f.attrs['isochrones'] = 'Padova 1994'
        f.attrs['imf'] = 'Chabrier'
        f.attrs['source'] = 'BC03 Updated version (2016)'
        f.attrs['remnant_prescription'] = 'BC03 internal (Bruzual & Charlot)'
        f.attrs['citation'] = 'Bruzual, G., & Charlot, S. (2003). MNRAS, 344, 1000-1028.'
        f.attrs['generator'] = 'scripts/build_mass_remaining_bc03.py'
        f.attrs['generator_args'] = 'bc03_stelib_chabrier'
        f.attrs['input_sha256'] = '44887abce0755c97d4273397b3e54606fbd7b28d931b61f8f148956bcd656b66'
        f.attrs['input_url'] = 'http://www.bruzual.org/bc03/Updated_version_2016/BC03_stelib_chabrier.tgz'

        print(f"  mass_remaining shape: {f['mass_remaining'].shape}")
        print(f"  log10_age_yr shape: {f['log10_age_yr'].shape}")
        print(f"  log10_z_abs shape: {f['log10_z_abs'].shape}")

    print(f"\nDone! File: {output_file}")


if __name__ == '__main__':
    main()
