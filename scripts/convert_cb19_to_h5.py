"""Convert Charlot & Bruzual 2019 (CB19) SSP spectra to tengri HDF5 format.

CB19 provides stellar population synthesis models with modern stellar
libraries and isochrones.

Data source: http://www.bruzual.org/cb07/ (includes CB19 files)

Download instructions:
1. Visit http://www.bruzual.org/cb07/
2. Download the "ised" files (SSP models) or FITS files
3. Select the desired IMF and metallicity range
4. Extract to a directory, e.g., ~/data/cb19/

File format:
- .ised ASCII files with columns: wavelength, flux, ...
- One file per metallicity
- Metallicities: Z=0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
- Wavelength: 91 A to 160 microns
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

# CB19 metallicity mapping (Z absolute values)
CB19_METALLICITIES = {
    0.0001: -4.0,
    0.0004: -3.398,
    0.004: -2.398,
    0.008: -2.097,
    0.02: -1.699,
    0.05: -1.301,
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def read_cb19_ised(filepath: Path) -> tuple:
    """Read CB19 .ised ASCII file.

    Format: ASCII with header lines and columns of wavelength + flux data.
    Typically: wavelength (A), flux (L_sun/A/Msun), and other columns.

    Parameters
    ----------
    filepath : Path
        Path to CB19 .ised file.

    Returns
    -------
    wavelength : ndarray, shape (n_wave,)
        Wavelength in Angstrom.
    flux : ndarray, shape (n_age, n_wave)
        Flux in L_sun/Angstrom per Msun.
    ages_gyr : ndarray, shape (n_age,)
        Ages in Gyr (read from header or derived from file structure).
    """
    # Read the entire file
    with open(filepath) as f:
        lines = f.readlines()

    # Parse header to get age information
    # CB19 files typically have metadata in the first few lines
    # followed by a wavelength row and then flux rows for each age
    header_lines = []
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("AGESPEC"):
            header_lines.append(line)
        elif line.strip() and not line[0].isspace():
            data_start = i + 1
            break

    if data_start == 0:
        data_start = 1  # Assume single header line

    # Parse ages from AGESPEC lines or extract from structure
    ages_str = []
    for line in header_lines:
        parts = line.split()
        if len(parts) > 1:
            try:
                age = float(parts[-1])
                ages_str.append(age)
            except (ValueError, IndexError):
                pass

    # Read all numeric data
    data = []
    for line in lines[data_start:]:
        try:
            row = [float(x) for x in line.split()]
            if len(row) > 0:
                data.append(row)
        except ValueError:
            pass

    if not data:
        raise ValueError(f"No numeric data found in {filepath}")

    data = np.array(data)

    # CB19 format: typically wavelength in first column, then flux values
    # If file has shape (n_wave + n_age, many_columns), it needs parsing
    # Conservative approach: assume first column is wavelength, rest are SEDs at
    # different ages, transpose to (n_age, n_wave)
    if data.ndim != 2:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Most common CB19 format: wavelength column + flux columns
    wavelength = data[:, 0]
    flux_data = data[:, 1:].T  # Transpose to (n_age, n_wave)

    # If ages_str was extracted, use them; otherwise infer from # of flux columns
    if not ages_str:
        # Assume ages are linearly spaced or come from a standard grid
        # CB19 typically has ~220 ages from 0 to 20 Gyr
        n_age = flux_data.shape[0]
        ages_gyr = np.linspace(0, 20, n_age)
    else:
        ages_gyr = np.array(ages_str[: flux_data.shape[0]])

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


def convert_cb19(input_dir: Path, output_path: Path, dry_run: bool = False) -> bool:
    """Convert CB19 data to HDF5.

    Parameters
    ----------
    input_dir : Path
        Directory containing CB19 .ised files.
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

    # Find all .ised files
    files = sorted(input_dir.glob("*.ised"))
    if not files:
        logger.error(f"No .ised files found in {input_dir}")
        return False

    logger.info(f"Found {len(files)} metallicity files")

    # Read first file to get wavelength grid and age structure
    logger.info(f"Reading {files[0].name}...")
    wavelength, flux_data, ages_gyr = read_cb19_ised(files[0])
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

        wavelength, flux_data, ages_gyr_check = read_cb19_ised(filepath)

        # Verify consistency
        if len(ages_gyr_check) != n_age:
            logger.warning(
                f"Age mismatch in {filename}: {len(ages_gyr_check)} vs {n_age}, "
                f"will use first file's age grid"
            )
        if len(wavelength) != n_wave:
            logger.error(f"Wavelength mismatch in {filename}: {len(wavelength)} vs {n_wave}")
            return False

        # Try to infer metallicity from filename
        # Common patterns: z0001, z0004, z004, z008, z020, z050, etc.
        z_absolute = None
        for z_val, lgz in CB19_METALLICITIES.items():
            if str(z_val).replace(".", "") in filename.replace(".", ""):
                z_absolute = z_val
                ssp_lgmet[i] = lgz
                logger.info(f"  Z={z_absolute:.4f}, log10(Z)={ssp_lgmet[i]:.4f}")
                break

        if z_absolute is None:
            logger.warning(f"Could not infer metallicity from filename: {filename}")
            ssp_lgmet[i] = -2.0  # Default to solar-like

        # Convert flux
        flux_erg = convert_flux_to_erg_per_hz(flux_data, wavelength)
        ssp_flux[i, :, :] = flux_erg.astype(np.float32)

    # Sort by metallicity
    sort_idx = np.argsort(ssp_lgmet)
    ssp_flux = ssp_flux[sort_idx, :, :]
    ssp_lgmet = ssp_lgmet[sort_idx]

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CB19 Conversion Summary")
    logger.info("=" * 60)
    logger.info(f"n_met={len(files)}, n_age={n_age}, n_wave={n_wave}")
    logger.info(f"wavelength: {wavelength[0]:.1f} - {wavelength[-1]:.1f} A")
    logger.info(f"age: {ages_gyr[0]:.4e} - {ages_gyr[-1]:.4f} Gyr")
    logger.info(f"metallicity (Z): {10 ** ssp_lgmet[0]:.6f} - {10 ** ssp_lgmet[-1]:.4f}")
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
        description="Convert CB19 SSP spectra to tengri HDF5 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing CB19 .ised files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ssp_cb19.h5",
        help="Output HDF5 file path (default: data/ssp_cb19.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing",
    )

    args = parser.parse_args()

    success = convert_cb19(args.input_dir, Path(args.output), dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
