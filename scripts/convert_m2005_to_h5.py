"""Convert Maraston 2005 (M2005) SSP spectra to tengri HDF5 format.

Maraston 2005 provides stellar population synthesis models with emphasis
on the near and mid-infrared properties of stellar populations.

Data source: http://www.icg.port.ac.uk/~maraMdston/M05

Download instructions:
1. Visit http://www.icg.port.ac.uk/~maraMdston/M05
2. Download the SED files for your preferred IMF (default: Kroupa)
3. Select metallicity range
4. Extract to a directory, e.g., ~/data/m2005/

File format:
- ASCII files with wavelength and flux columns
- One file per metallicity
- Metallicities: Z=0.001, 0.01, 0.02, 0.04
- Wavelength: varying range (typically 100 A to 1 mm)
- Flux units: L_sun/A per 1 Msun (must convert to erg/s/Hz/Msun)

Conversion formula:
  F_erg = F_Lsun_A * L_sun_erg * (lambda_A)^2 / c
  where:
    - lambda_A is wavelength in Angstrom
    - c = 2.99792458e18 Angstrom/s
    - L_sun_erg = 3.828e33 erg/s (IAU 2015)
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

# M2005 metallicity mapping (Z absolute values to log10(Z))
M2005_METALLICITIES = {
    0.001: np.log10(0.001),  # -3.0
    0.01: np.log10(0.01),  # -2.0
    0.02: np.log10(0.02),  # -1.699
    0.04: np.log10(0.04),  # -1.398
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def read_m2005_sed(filepath: Path) -> tuple:
    """Read M2005 SED ASCII file.

    Format: ASCII with wavelength and flux columns.
    May have a header line(s) indicating age and other metadata.

    Parameters
    ----------
    filepath : Path
        Path to M2005 SED file.

    Returns
    -------
    wavelength : ndarray, shape (n_wave,)
        Wavelength in Angstrom.
    flux : ndarray, shape (n_age, n_wave)
        Flux in L_sun/Angstrom per Msun.
    ages_gyr : ndarray, shape (n_age,)
        Ages in Gyr.
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Parse for age information in header comments
    ages = []
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            # Try to extract age from comment
            if "age" in line.lower() or "t=" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() in ["age:", "t="]:
                        try:
                            age = float(parts[i + 1].rstrip("Gyr"))
                            ages.append(age)
                        except (ValueError, IndexError):
                            pass
            continue
        data_lines.append(line)

    # Parse numeric data
    data = []
    for line in data_lines:
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 2:
                data.append(parts)
        except ValueError:
            pass

    if not data:
        raise ValueError(f"No numeric data found in {filepath}")

    data = np.array(data)

    # M2005 format: wavelength (first column) + flux columns (one per age or all SEDs together)
    wavelength = data[:, 0]
    flux_data = data[:, 1:].T  # (n_age, n_wave) if multiple columns are ages

    # If ages weren't parsed from header, assume standard grid or single age
    if not ages:
        n_age = flux_data.shape[0]
        if n_age == 1:
            # Single age file; try to infer from filename
            ages = [1.0]  # Default to 1 Gyr
        else:
            # Assume ages from 0.1 to 13.8 Gyr (typical for M2005 grid)
            ages = np.linspace(0.1, 13.8, n_age).tolist()

    ages_gyr = np.array(ages[: flux_data.shape[0]])

    return wavelength, flux_data, ages_gyr


def convert_flux_to_erg_per_hz(
    flux_lsun_per_a: np.ndarray, wavelength_a: np.ndarray
) -> np.ndarray:
    """Convert flux from L_sun/A to erg/s/Hz/Msun.

    Parameters
    ----------
    flux_lsun_per_a : ndarray
        Flux in L_sun/Angstrom per Msun.
    wavelength_a : ndarray, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    ndarray
        Flux in erg/s/Hz/Msun.
    """
    # Convert: L_sun/A to erg/s
    flux_erg_per_a = flux_lsun_per_a * L_SUN_ERG

    # Convert A to Hz: F_nu = F_lambda * (lambda^2 / c)
    flux_erg_per_hz = flux_erg_per_a * (wavelength_a**2) / C_ANGSTROM_PER_S

    return flux_erg_per_hz


def convert_m2005(
    input_dir: Path, output_path: Path, imf: str = "kroupa", dry_run: bool = False
) -> bool:
    """Convert M2005 data to HDF5.

    Parameters
    ----------
    input_dir : Path
        Directory containing M2005 SED files.
    output_path : Path
        Output HDF5 file path.
    imf : str
        IMF name to filter files (e.g., "kroupa", "salpeter").
    dry_run : bool
        If True, print summary without writing.

    Returns
    -------
    bool
        True if successful.
    """
    input_dir = Path(input_dir)

    # Find all SED files matching the IMF
    all_files = sorted(input_dir.glob("*.sed"))
    files = [f for f in all_files if imf.lower() in f.name.lower()]

    if not files:
        logger.error(f"No .sed files found matching IMF='{imf}' in {input_dir}")
        logger.error(f"Available files: {[f.name for f in all_files[:5]]}")
        return False

    logger.info(f"Found {len(files)} files for IMF='{imf}'")

    # Read first file to get wavelength grid and age structure
    logger.info(f"Reading {files[0].name}...")
    wavelength, flux_data, ages_gyr = read_m2005_sed(files[0])
    n_wave = len(wavelength)
    n_age = flux_data.shape[0]

    logger.info(f"  n_wave={n_wave}, n_age={n_age}")
    logger.info(f"  wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} A")
    logger.info(f"  age range: {ages_gyr[0]:.4f} - {ages_gyr[-1]:.4f} Gyr")

    # Allocate arrays
    ssp_flux = np.zeros((len(files), n_age, n_wave), dtype=np.float32)
    ssp_lgmet = np.zeros(len(files), dtype=np.float32)
    ssp_wave = np.array(wavelength, dtype=np.float32)
    ssp_lg_age_gyr = np.log10(np.maximum(ages_gyr, 1e-5))

    # Read all files
    for i, filepath in enumerate(files):
        filename = filepath.name
        logger.info(f"Reading {filename}...")

        wavelength, flux_data, ages_gyr_check = read_m2005_sed(filepath)

        # Verify consistency
        if len(ages_gyr_check) != n_age:
            logger.warning(
                f"Age mismatch in {filename}: {len(ages_gyr_check)} vs {n_age}, "
                f"will use first file's age grid"
            )
        if len(wavelength) != n_wave:
            logger.error(f"Wavelength mismatch in {filename}: {len(wavelength)} vs {n_wave}")
            return False

        # Infer metallicity from filename
        # Common patterns: z001, z010, z020, z040 or 001, 010, 020, 040
        z_absolute = None
        for z_val in M2005_METALLICITIES:
            z_str = f"{z_val:.0%}".replace("%", "").lstrip("0") or "0"  # e.g., "001" from 0.001
            if z_str in filename or str(z_val) in filename:
                z_absolute = z_val
                ssp_lgmet[i] = np.log10(z_val)
                logger.info(f"  Z={z_absolute:.4f}, log10(Z)={ssp_lgmet[i]:.4f}")
                break

        if z_absolute is None:
            logger.warning(f"Could not infer metallicity from filename: {filename}")
            ssp_lgmet[i] = np.log10(0.02)  # Default to solar

        # Convert flux
        flux_erg = convert_flux_to_erg_per_hz(flux_data, wavelength)
        ssp_flux[i, :, :] = flux_erg.astype(np.float32)

    # Sort by metallicity
    sort_idx = np.argsort(ssp_lgmet)
    ssp_flux = ssp_flux[sort_idx, :, :]
    ssp_lgmet = ssp_lgmet[sort_idx]

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("M2005 Conversion Summary")
    logger.info("=" * 60)
    logger.info(f"IMF: {imf}")
    logger.info(f"n_met={len(files)}, n_age={n_age}, n_wave={n_wave}")
    logger.info(f"wavelength: {wavelength[0]:.1f} - {wavelength[-1]:.1f} A")
    logger.info(f"age: {ages_gyr[0]:.4e} - {ages_gyr[-1]:.4f} Gyr")
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
        description="Convert Maraston 2005 SSP spectra to tengri HDF5 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing M2005 .sed files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ssp_m2005.h5",
        help="Output HDF5 file path (default: data/ssp_m2005.h5)",
    )
    parser.add_argument(
        "--imf",
        type=str,
        default="kroupa",
        help="IMF name to filter files (default: kroupa)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing",
    )

    args = parser.parse_args()

    success = convert_m2005(args.input_dir, Path(args.output), imf=args.imf, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
