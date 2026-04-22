"""Convert BPASS v2.2.1 SSP spectra to tengri HDF5 format.

BPASS (Binary Population and Spectral Synthesis) provides SEDs for
simple stellar populations across a wide range of metallicities and ages.

Data source: https://bpass.auckland.ac.nz/

Download instructions:
1. Visit https://bpass.auckland.ac.nz/
2. Select "Binary + Single Stars" and "v2.2.1"
3. Choose "Chabrier IMF" (or your preferred IMF)
4. Download the "spectra-bin-imf135_300-* files (metallicity files)
5. Extract to a directory, e.g., ~/data/bpass_v2.2.1/

File format:
- ASCII files named spectra-bin-imf135_300.zXXX.dat
- Metallicities: z001, z002, z003, z004, z006, z008, z010, z014, z020, z030, z040
  (Z = 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040)
- Wavelength: 1-100000 Angstrom
- Flux units: L_sun/Angstrom per 1e6 Msun (must convert to erg/s/Hz/Msun)

Conversion formula:
  F_erg = F_Lsun_A * L_sun_erg * (lambda_A)^2 / c / 1e6
  where:
    - lambda_A is wavelength in Angstrom
    - c = 2.99792458e18 Angstrom/s
    - L_sun_erg = 3.828e33 erg/s (IAU 2015)
    - 1e6 factor because flux is per 1e6 Msun
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# Physical constants
L_SUN_ERG = 3.828e33  # erg/s (IAU 2015)
C_ANGSTROM_PER_S = 2.99792458e18  # Angstrom/s

# BPASS metallicity mapping
BPASS_METALLICITIES = {
    "z001": 0.001,
    "z002": 0.002,
    "z003": 0.003,
    "z004": 0.004,
    "z006": 0.006,
    "z008": 0.008,
    "z010": 0.010,
    "z014": 0.014,
    "z020": 0.020,
    "z030": 0.030,
    "z040": 0.040,
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def read_bpass_spectrum(filepath: Path) -> tuple:
    """Read BPASS ASCII spectrum file.

    Parameters
    ----------
    filepath : Path
        Path to BPASS .dat file.

    Returns
    -------
    wavelength : ndarray, shape (n_wave,)
        Wavelength in Angstrom.
    flux : ndarray, shape (n_age, n_wave)
        Flux in L_sun/Angstrom per 1e6 Msun.
    ages_myr : ndarray, shape (n_age,)
        Ages in Myr.
    """
    data = np.loadtxt(filepath, skiprows=1)
    # First column is age in Myr, rest are flux at each wavelength
    ages_myr = data[:, 0]
    wavelengths = np.loadtxt(filepath, skiprows=0, max_rows=1)[1:]
    flux = data[:, 1:]

    return wavelengths, flux, ages_myr


def convert_flux_to_erg_per_hz(
    flux_lsun_per_a: np.ndarray, wavelength_a: np.ndarray
) -> np.ndarray:
    """Convert flux from L_sun/A to erg/s/Hz/Msun.

    Parameters
    ----------
    flux_lsun_per_a : ndarray
        Flux in L_sun/Angstrom per 1e6 Msun.
    wavelength_a : ndarray, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    ndarray
        Flux in erg/s/Hz/Msun.
    """
    # Convert: L_sun/A to erg/s
    flux_erg_per_a = flux_lsun_per_a * L_SUN_ERG

    # Convert per-unit-mass from 1e6 Msun to 1 Msun
    flux_erg_per_a /= 1e6

    # Convert A to Hz: F_nu = F_lambda * (lambda^2 / c)
    flux_erg_per_hz = flux_erg_per_a * (wavelength_a**2) / C_ANGSTROM_PER_S

    return flux_erg_per_hz


def convert_bpass(input_dir: Path, output_path: Path, dry_run: bool = False) -> bool:
    """Convert BPASS data to HDF5.

    Parameters
    ----------
    input_dir : Path
        Directory containing BPASS .dat files.
    output_path : Path
        Output HDF5 file path.
    dry_run : bool
        If True, print summary without writing.

    Returns
    -------
    bool
        True if successful.
    """
    input_dir = Path(input_dir)

    # Find all metallicity files
    files = sorted(input_dir.glob("spectra-bin-imf135_300.z???.dat"))
    if not files:
        logger.error(f"No BPASS files found in {input_dir}")
        logger.error("Expected files matching: spectra-bin-imf135_300.zXXX.dat")
        return False

    logger.info(f"Found {len(files)} metallicity files")

    # Read first file to get wavelength grid and age structure
    logger.info(f"Reading {files[0].name}...")
    wavelength, flux_data, ages_myr = read_bpass_spectrum(files[0])
    n_wave = len(wavelength)
    n_age = flux_data.shape[0]

    logger.info(f"  n_wave={n_wave}, n_age={n_age}")
    logger.info(f"  wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} A")
    logger.info(f"  age range: {ages_myr[0]:.2f} - {ages_myr[-1]:.2e} Myr")

    # Allocate arrays
    ssp_flux = np.zeros((len(files), n_age, n_wave), dtype=np.float32)
    ssp_lgmet = np.zeros(len(files), dtype=np.float32)
    ssp_wave = np.array(wavelength, dtype=np.float32)
    ssp_lg_age_gyr = np.log10(ages_myr / 1000.0)

    # Read all files and convert flux
    for i, filepath in enumerate(files):
        filename = filepath.name
        # Extract metallicity code from filename
        zcode = filename.split(".")[-2]  # e.g., "z001"

        if zcode not in BPASS_METALLICITIES:
            logger.warning(f"Unknown metallicity code: {zcode} in {filename}")
            continue

        z_absolute = BPASS_METALLICITIES[zcode]
        ssp_lgmet[i] = np.log10(z_absolute)

        logger.info(f"Reading {filename}... Z={z_absolute:.4f}, log10(Z)={ssp_lgmet[i]:.4f}")

        wavelength, flux_data, ages_myr_check = read_bpass_spectrum(filepath)

        # Verify consistency
        if len(ages_myr_check) != n_age:
            logger.error(f"Age mismatch in {filename}: {len(ages_myr_check)} vs {n_age}")
            return False
        if len(wavelength) != n_wave:
            logger.error(f"Wavelength mismatch in {filename}: {len(wavelength)} vs {n_wave}")
            return False

        # Convert flux
        flux_erg = convert_flux_to_erg_per_hz(flux_data, wavelength)
        ssp_flux[i, :, :] = flux_erg.astype(np.float32)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BPASS Conversion Summary")
    logger.info("=" * 60)
    logger.info(f"n_met={len(files)}, n_age={n_age}, n_wave={n_wave}")
    logger.info(f"wavelength: {wavelength[0]:.1f} - {wavelength[-1]:.1f} A")
    logger.info(f"age: {ages_myr[0]:.2e} - {ages_myr[-1]:.2e} Myr")
    logger.info(f"metallicity (Z): {10 ** ssp_lgmet[0]:.4f} - {10 ** ssp_lgmet[-1]:.4f}")
    logger.info(f"log10(Z): {ssp_lgmet[0]:.4f} - {ssp_lgmet[-1]:.4f}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60 + "\n")

    if dry_run:
        logger.info("(dry-run mode: not writing file)")
        return True

    # Write HDF5
    logger.info("Writing HDF5...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("ssp_wave", data=ssp_wave)
        f.create_dataset("ssp_flux", data=ssp_flux, compression="gzip", compression_opts=4)
        f.create_dataset("ssp_lg_age_gyr", data=ssp_lg_age_gyr)
        f.create_dataset("ssp_lgmet", data=ssp_lgmet)

    logger.info(f"✓ Wrote {output_path}")

    # Validate by loading
    logger.info("Validating...")
    try:
        from tengri.components.sps.dsps_wrapper import load_ssp_data

        ssp_data = load_ssp_data(str(output_path))
        logger.info(f"✓ Loaded successfully: {ssp_data.ssp_flux.shape}")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert BPASS v2.2.1 SSP spectra to tengri HDF5 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing BPASS .dat files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ssp_bpass_v2.2.h5",
        help="Output HDF5 file path (default: data/ssp_bpass_v2.2.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing",
    )

    args = parser.parse_args()

    success = convert_bpass(args.input_dir, Path(args.output), dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
