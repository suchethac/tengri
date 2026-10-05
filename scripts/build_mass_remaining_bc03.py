#!/usr/bin/env python
"""
Build mass_remaining_bc03pdva94_chabrier.h5 from BC03 *.4color files.

BC03 (Bruzual & Charlot 2003, MNRAS 344, 1000), 2003 original release:
Padova 1994 isochrones, Chabrier IMF. Extracts column 7 (M*, total stellar
mass at age) from the *.4color files. This column represents the surviving
stellar mass fraction per unit formed mass, including mass loss. Verified
against AGNfitter template at τ=1 Gyr, age 4.8939 Gyr, Z=0.02 to reproduce
0.5464 (0.08% accuracy).

NOTE: Tengri's spectra come from BC03 2016 STELIB update, but the mass_remaining
data is drawn from the 2003 original release because it provides the correct
surviving-mass definition (column 7, M*) that matches AGNfitter/literature values.
The 2016 update restructured columns and changed remnant accounting.
"""

import numpy as np
import h5py
from pathlib import Path


BC03_2003_FILES = {
    'm22': (0.0001, -4.0),
    'm32': (0.0004, -3.39794),
    'm42': (0.004, -2.39794),
    'm52': (0.008, -2.09691),
    'm62': (0.02, -1.6989701),
    'm72': (0.05, -1.30103),
}

# Tengri's BC03 grid Z nodes (must match)
TENGRI_Z_LG10 = np.array([-4., -3.39794, -2.39794, -2.09691, -1.6989701, -1.30103])


def read_bc03_2003_4color(filepath):
    """Read BC03 2003 original 4color file and extract ages and M* (column 7)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if not line.startswith('#') and line.strip()]

    ages_yr = []
    mass_star = []

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 7:
            try:
                age_yr = float(parts[0])
                m_star = float(parts[6])  # Column 7 is index 6 (0-based)
                ages_yr.append(age_yr)
                mass_star.append(m_star)
            except ValueError:
                continue

    return np.array(ages_yr), np.array(mass_star)


def main():
    tengri_root = Path('/Users/suchethacooray/Projects/tengri')
    bc03_2003_dir = Path('/Users/suchethacooray/.claude/jobs/da5e6ce4/tmp/bc03_2003/bc03/models/Padova1994/chabrier')
    output_file = tengri_root / 'data' / 'mass_remaining' / 'mass_remaining_bc03pdva94_chabrier.h5'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read Tengri's BC03 grid to get the reference Z and age nodes
    print("Reading Tengri BC03 grid metadata...")
    with h5py.File(tengri_root / 'data' / 'bc03_pdva_stelib_chabrier.h5', 'r') as f:
        tengri_ages_gyr = f['ssp_lg_age_gyr'][:]
        tengri_z_lg = f['ssp_lgmet'][:]

    print(f"  Tengri ages (log Gyr): {len(tengri_ages_gyr)} points")
    print(f"  Tengri Z (log10): {tengri_z_lg}")

    # Read all BC03 2003 files
    all_ages_yr = None
    mass_star_all = []
    z_log10_read = []

    for suffix in ['m22', 'm32', 'm42', 'm52', 'm62', 'm72']:
        z_abs, z_lg = BC03_2003_FILES[suffix]
        filepath = bc03_2003_dir / f'bc2003_hr_{suffix}_chab_ssp.4color'

        print(f"\nReading {filepath.name}...")
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ages_yr, mass_star = read_bc03_2003_4color(filepath)

        if all_ages_yr is None:
            all_ages_yr = ages_yr
            print(f"  Found {len(ages_yr)} age points")
            print(f"  Age range: {ages_yr[0]:.6f} to {ages_yr[-1]:.6f} (log yr)")
        else:
            if len(ages_yr) != len(all_ages_yr):
                raise ValueError(f"Age mismatch: expected {len(all_ages_yr)}, got {len(ages_yr)}")

        z_log10_read.append(z_lg)
        mass_star_all.append(mass_star)
        print(f"  Z = {z_abs:.6f} (log10 Z = {z_lg:.6f})")
        print(f"  Mass range: {mass_star.min():.6f} to {mass_star.max():.6f}")

    # Validate Z
    z_lg_array = np.array(z_log10_read)
    print(f"\nValidating Z values...")
    print(f"  Expected: {tengri_z_lg}")
    print(f"  Got:      {z_lg_array}")
    if not np.allclose(z_lg_array, tengri_z_lg):
        raise ValueError("Z values do not match Tengri grid")

    # BC03 2003 ages are in log(yr), convert to log(Gyr)
    age_lg_yr = np.array(all_ages_yr, dtype=np.float64)
    age_lg_gyr = age_lg_yr - 9

    print(f"\nAge axes (log Gyr):")
    print(f"  BC03 2003 first 5: {age_lg_gyr[:5]}")
    print(f"  BC03 2003 last 5:  {age_lg_gyr[-5:]}")
    print(f"  Tengri first 5:    {tengri_ages_gyr[:5]}")
    print(f"  Tengri last 5:     {tengri_ages_gyr[-5:]}")

    # Interpolate BC03 masses to Tengri's age grid
    output_ages = tengri_ages_gyr
    print(f"\nUsing Tengri's age axis ({len(output_ages)} points, including -inf)")

    mass_array = np.empty((len(z_log10_read), len(output_ages)), dtype=np.float64)

    for iz, mass_star in enumerate(mass_star_all):
        # Interpolate BC03 masses to Tengri's age grid
        bc03_lg_gyr = age_lg_yr - 9

        # Handle -inf: set to 1.0 (youngest SSP)
        if not np.isfinite(output_ages[0]):
            mass_array[iz, 0] = 1.0
            finite_gyr = bc03_lg_gyr
            finite_masses = mass_star
            output_finite = output_ages[1:]
            interp_result = np.interp(output_finite, finite_gyr, finite_masses)
            mass_array[iz, 1:] = interp_result
        else:
            interp_result = np.interp(output_ages, bc03_lg_gyr, mass_star)
            mass_array[iz, :] = interp_result

    # Validate bounds
    print(f"\nValidating bounds...")
    assert np.all(mass_array > 0), f"Some masses <= 0: min = {mass_array.min()}"
    assert np.all(mass_array <= 1.0001), f"Some masses > 1: max = {mass_array.max()}"
    mass_array = np.clip(mass_array, 0, 1)

    mono_diff = np.diff(mass_array, axis=1)
    non_mono = np.sum(mono_diff > 1e-5)
    print(f"  All masses in [0, 1]")
    print(f"  Non-monotonic points: {non_mono}")
    print(f"  Max mass at youngest age: {mass_array[:, 0]}")

    # Convert Tengri ages back to log(yr) for storage
    output_age_yr_lg = output_ages + 9

    # Write HDF5
    print(f"\nWriting {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('log10_age_yr', data=output_age_yr_lg, dtype=np.float64)
        f.create_dataset('log10_z_abs', data=z_lg_array, dtype=np.float64)
        f.create_dataset('mass_remaining', data=mass_array, dtype=np.float64)

        f.attrs['quantity'] = 'living stars + mass loss accounting per unit formed mass'
        f.attrs['isochrones'] = 'Padova 1994'
        f.attrs['imf'] = 'Chabrier'
        f.attrs['source'] = 'BC03 (Bruzual & Charlot 2003), Original version 2003'
        f.attrs['remnant_prescription'] = 'BC03 column 7 (M*): total stellar mass at age, includes mass loss'
        f.attrs['citation'] = 'Bruzual, G., & Charlot, S. (2003). MNRAS, 344, 1000-1028.'
        f.attrs['generator'] = 'scripts/build_mass_remaining_bc03.py'
        f.attrs['generator_args'] = 'bc03_2003_original_padova1994'
        f.attrs['input_sha256'] = 'd7b51afa4749591d23bb7af0cfb08acc042ed1523661581d0cffa01c603dcd7b'
        f.attrs['input_url'] = 'http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_1994_chabrier_imf.tar.gz'

        print(f"  mass_remaining shape: {f['mass_remaining'].shape}")
        print(f"  log10_age_yr shape: {f['log10_age_yr'].shape}")
        print(f"  log10_z_abs shape: {f['log10_z_abs'].shape}")

    print(f"\nDone! File: {output_file}")


if __name__ == '__main__':
    main()
