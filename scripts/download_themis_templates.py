#!/usr/bin/env python3
"""Download THEMIS dust emission templates (Jones et al. 2017).

The THEMIS model uses a different grain composition than DL07: a-C(:H)
aromatic carbon instead of PAH, with the aromatic fraction qhac replacing
qpah as the key parameter.

The templates are generated using DustEM and distributed with CIGALE.

Source: CIGALE project
    https://gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder/themis

Usage:
    python scripts/download_themis_templates.py
    python scripts/download_themis_templates.py --output data/themis_templates.npz
    python scripts/download_themis_templates.py --dry-run

References:
    Jones, A. P. et al. 2017, A&A, 602, A46.
    Compiègne, M. et al. 2011, A&A, 525, A103 (DustEM).
"""

import argparse
import os
import sys
from pathlib import Path


# -----------------------------------------------------------------------
# THEMIS template source (from CIGALE's DustEM pre-computed grid)
# -----------------------------------------------------------------------
CIGALE_GITLAB_BASE = (
    "https://gitlab.lam.fr/api/v4/projects/cigale%2Fcigale"
    "/repository/files/{path}/raw?ref=master"
)


def download_and_convert(output_path: str, dry_run: bool = False) -> None:
    """Download THEMIS templates and convert to NPZ format.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    dry_run : bool
        If True, only print what would be done.
    """
    print("THEMIS template downloader (Jones et al. 2017)")
    print(f"  Output: {output_path}")
    print()

    if dry_run:
        print("[dry-run] Would download THEMIS templates from CIGALE GitLab.")
        print("[dry-run] Convert to NPZ with keys:")
        print("  - wavelength_um: (n_wave,)")
        print("  - qhac_grid: (n_qhac,) a-C(:H) aromatic fraction")
        print("  - umin_grid: (n_umin,)")
        print("  - spectra_single: (n_qhac, n_umin, n_wave)")
        print("  - spectra_pdr: (n_qhac, n_umin, n_wave)")
        return

    print(
        "ERROR: Automatic download not yet implemented.\n"
        "\n"
        "To create the template file manually:\n"
        "  1. Install CIGALE: pip install pcigale\n"
        "  2. Extract THEMIS template grid from the CIGALE database\n"
        "     (module: dustemission / themis)\n"
        "  3. Convert to NPZ with the following keys:\n"
        "     - wavelength_um: (n_wave,) wavelength grid in microns\n"
        "     - qhac_grid: (n_qhac,) a-C(:H) aromatic fraction\n"
        "     - umin_grid: (n_umin,) radiation field intensities\n"
        "     - spectra_single: (n_qhac, n_umin, n_wave) single-U SEDs\n"
        "     - spectra_pdr: (n_qhac, n_umin, n_wave) power-law U SEDs\n"
        f"  4. Save to: {output_path}\n"
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download THEMIS dust emission templates (Jones+2017)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path (default: data/themis_templates.npz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    args = parser.parse_args()

    if args.output is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.output = str(repo_root / "data" / "themis_templates.npz")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    download_and_convert(args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
