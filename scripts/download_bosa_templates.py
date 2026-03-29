#!/usr/bin/env python3
"""Download BOSA dust emission templates (Boquien & Salim 2021).

The BOSA templates parameterize dust emission by (L_TIR, sSFR) instead
of radiation field parameters.  This provides a direct link between
star formation activity and dust SED shape.

Source: CIGALE project / CDS
    Boquien, M. & Salim, S. 2021, A&A, 653, A149.

Usage:
    python scripts/download_bosa_templates.py
    python scripts/download_bosa_templates.py --output data/bosa_templates.npz
    python scripts/download_bosa_templates.py --dry-run

References:
    Boquien, M. & Salim, S. 2021, A&A, 653, A149.
"""

import argparse
import os
import sys
from pathlib import Path


# -----------------------------------------------------------------------
# BOSA template source
# -----------------------------------------------------------------------
# The templates are distributed with CIGALE (Boquien et al. 2019).
# Check the CIGALE GitLab for the exact database path:
# https://gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder

CIGALE_GITLAB_BASE = (
    "https://gitlab.lam.fr/api/v4/projects/cigale%2Fcigale"
    "/repository/files/{path}/raw?ref=master"
)


def download_and_convert(output_path: str, dry_run: bool = False) -> None:
    """Download BOSA templates and convert to NPZ format.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    dry_run : bool
        If True, only print what would be done.
    """
    print("BOSA template downloader (Boquien & Salim 2021)")
    print(f"  Output: {output_path}")
    print()

    if dry_run:
        print("[dry-run] Would download BOSA templates from CIGALE GitLab.")
        print("[dry-run] Convert to NPZ with keys:")
        print("  - wavelength_um: (n_wave,)")
        print("  - log_ltir_grid: (n_ltir,) log10(L_TIR/Lsun)")
        print("  - log_ssfr_grid: (n_ssfr,) log10(sSFR/yr^-1)")
        print("  - spectra: (n_ltir, n_ssfr, n_wave)")
        return

    print(
        "ERROR: Automatic download not yet implemented.\n"
        "\n"
        "To create the template file manually:\n"
        "  1. Install CIGALE: pip install pcigale\n"
        "  2. Extract the BOSA template grid from the CIGALE database\n"
        "     (module: dustemission / bosa)\n"
        "  3. Convert to NPZ with the following keys:\n"
        "     - wavelength_um: (n_wave,) wavelength grid in microns\n"
        "     - log_ltir_grid: (n_ltir,) log10(L_TIR/Lsun)\n"
        "     - log_ssfr_grid: (n_ssfr,) log10(sSFR/yr^-1)\n"
        "     - spectra: (n_ltir, n_ssfr, n_wave) normalized SEDs\n"
        f"  4. Save to: {output_path}\n"
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BOSA dust emission templates (Boquien & Salim 2021)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path (default: data/bosa_templates.npz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    args = parser.parse_args()

    if args.output is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.output = str(repo_root / "data" / "bosa_templates.npz")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    download_and_convert(args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
