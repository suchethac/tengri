#!/usr/bin/env python3
"""List available emission lines in a tengri CLOUDY grid HDF5 file.

Usage:
    python scripts/list_emission_lines.py [data/cloudy_grid_mist.h5]
"""

import sys

import h5py
import numpy as np


def list_lines(filepath: str) -> None:
    """Print emission line info from a tengri CLOUDY grid."""
    with h5py.File(filepath, "r") as f:
        wavelengths = f["lines/wavelength"][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in f["lines/names"][:]]
        luminosity = f["lines/luminosity"][:]  # (n_met, n_age, n_logU, n_lines)

        # Grid info
        log_met = f["axes/log_met"][:]
        log_age = f["axes/log_age_yr"][:]
        log_U = f["axes/log_U"][:]

        print(f"CLOUDY Nebular Emission Grid: {filepath}")
        print(f"  Source: {f.attrs.get('source', 'unknown')}")
        print(f"  Isochrone: {f.attrs.get('isoc_type', 'unknown')}")
        print(f"  Metallicities: {len(log_met)} [{log_met[0]:.2f} to {log_met[-1]:.2f}]")
        print(f"  Ages: {len(log_age)} [10^{log_age[0]:.1f} to 10^{log_age[-1]:.1f} yr]")
        print(f"  log U: {len(log_U)} [{log_U[0]:.1f} to {log_U[-1]:.1f}]")
        print()

    # Compute typical luminosity at solar Z, 3 Myr, logU=-2
    iz_solar = np.argmin(np.abs(log_met - (-1.85)))  # ~solar
    ia_3myr = np.argmin(np.abs(log_age - 6.5))  # ~3 Myr
    iu_m2 = np.argmin(np.abs(log_U - (-2.0)))

    typical_lum = luminosity[iz_solar, ia_3myr, iu_m2, :]

    print(f"{'Idx':>4}  {'Wavelength':>12}  {'Name':<30}  {'L (Lsun/Q_H)':>14}")
    print("-" * 70)
    for i in range(len(wavelengths)):
        wl = wavelengths[i]
        name = names[i] if i < len(names) else f"line_{wl:.1f}A"
        lum = typical_lum[i]
        # Highlight strong lines
        marker = "*" if lum > 1e-13 else " "
        print(f"{i:4d}  {wl:12.2f} A  {name:<30}  {lum:14.4e} {marker}")

    print()
    print("* = strong line (L > 1e-13 Lsun/Q_H at solar Z, 3 Myr, logU=-2)")


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/cloudy_grid_mist.h5"

    list_lines(filepath)


if __name__ == "__main__":
    main()
