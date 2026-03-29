#!/usr/bin/env python3
"""Download Astrodust+PAH templates (Hensley & Draine 2023).

The Astrodust model replaces the classical DL07 grain model with an
improved dust composition (astrodust + PAH) that better reproduces the
observed polarization and emission properties of interstellar dust.

The template grid is parameterized by (qPAH, Umin) with both single-U
and power-law U components, identical to the DL07 mixing formula.

Source: Harvard Dataverse
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BMPNML

The raw FITS files are converted to a single NPZ for fast loading.

Usage:
    python scripts/download_astrodust_templates.py
    python scripts/download_astrodust_templates.py --output data/astrodust_templates.npz
    python scripts/download_astrodust_templates.py --dry-run

References:
    Hensley, B. S. & Draine, B. T. 2023, ApJ, 948, 55.
"""

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


# -----------------------------------------------------------------------
# Dataverse dataset URL (Hensley & Draine 2023)
# -----------------------------------------------------------------------
# Users must check the Dataverse page for the exact download URLs.
# The dataset DOI is: 10.7910/DVN/BMPNML
DATAVERSE_BASE = (
    "https://dataverse.harvard.edu/api/access/datafile/"
)

# Placeholder file IDs — update these after checking the Dataverse page.
# The actual file IDs depend on the specific version published.
DATAVERSE_DOI = "doi:10.7910/DVN/BMPNML"


def download_and_convert(output_path: str, dry_run: bool = False) -> None:
    """Download Astrodust templates and convert to NPZ format.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    dry_run : bool
        If True, only print what would be done.
    """
    print("Astrodust+PAH template downloader (Hensley & Draine 2023)")
    print(f"  Dataset DOI: {DATAVERSE_DOI}")
    print(f"  Output: {output_path}")
    print()

    if dry_run:
        print("[dry-run] Would download Astrodust templates from Harvard Dataverse.")
        print("[dry-run] Convert raw FITS to NPZ with keys:")
        print("  - wavelength_um: (n_wave,)")
        print("  - qpah_grid: (n_qpah,)")
        print("  - umin_grid: (n_umin,)")
        print("  - spectra_single: (n_qpah, n_umin, n_wave)")
        print("  - spectra_pdr: (n_qpah, n_umin, n_wave)")
        return

    print(
        "ERROR: Automatic download not yet implemented.\n"
        "\n"
        "To create the template file manually:\n"
        "  1. Visit https://dataverse.harvard.edu/dataset.xhtml?"
        f"persistentId={DATAVERSE_DOI}\n"
        "  2. Download the Astrodust+PAH emission spectra\n"
        "  3. Convert to NPZ with the following keys:\n"
        "     - wavelength_um: (n_wave,) wavelength grid in microns\n"
        "     - qpah_grid: (n_qpah,) PAH mass fractions\n"
        "     - umin_grid: (n_umin,) radiation field intensities\n"
        "     - spectra_single: (n_qpah, n_umin, n_wave) single-U SEDs\n"
        "     - spectra_pdr: (n_qpah, n_umin, n_wave) power-law U SEDs\n"
        f"  4. Save to: {output_path}\n"
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Astrodust+PAH templates (Hensley & Draine 2023)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path (default: data/astrodust_templates.npz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    args = parser.parse_args()

    if args.output is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.output = str(repo_root / "data" / "astrodust_templates.npz")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    download_and_convert(args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
