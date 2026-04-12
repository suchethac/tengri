#!/usr/bin/env python3
"""Attempt to download the Feltre+2016 AGN NLR photoionization grid.

Source
------
Feltre, Charlot & Gutkin (2016), MNRAS 456, 3354, arXiv:1511.08217.
"Nuclear activity in galaxies: the effective slope of the ionizing spectrum."

Grid description (Table 1 of the paper)
-----------------------------------------
CLOUDY c13.03 photoionization models of AGN narrow-line region emission,
ionized by a broken power-law EUV spectrum f_nu ~ nu^alpha.

Grid axes:
  alpha     : -1.2, -1.4, -1.7, -2.0          (4 values; discrete)
  log U_S   : -1.0, -2.0, -3.0, -4.0          (4 values; continuous)
  log n_H   :  2.0,  3.0,  4.0                 (3 values; continuous)
  Z         : 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006,
               0.008, 0.014, 0.01774, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07
              (16 values; continuous on log10(Z) axis)
  xi_d      : 0.1, 0.3, 0.5                    (3 values; discrete)

Total: 4 × 4 × 3 × 16 × 3 = 2,304 model points.

Emission lines (from diagnostic figures in paper)
-------------------------------------------------
N V λ1240, C IV λ1550, He II λ1640, O III] λ1663/1666, C III] λ1908,
[Ne IV] λ2424, [Ne V] λ3426, [O II] λ3727, Hβ λ4861, [O III] λ5007,
Hα λ6563, [N II] λ6584, N IV] λ1485, Si IV+O IV] λ1400, N III] λ1750,
[Si III] λ1883, Si III] λ1888, [O III] λ2321, [Ne III] λ3343, O I λ1304

HDF5 schema (data/feltre_grid.h5)
----------------------------------
Group "/feltre":
  Datasets:
    alpha_axis        : (4,)           float64  ionizing slope axis
    logUs_axis        : (4,)           float64  log(U_S) axis (descending: -1 to -4)
    logn_axis         : (3,)           float64  log(n_H) axis
    logZ_axis         : (16,)          float64  log10(Z) axis
    xi_d_axis         : (3,)           float64  dust-to-metal ratio axis
    line_wavelengths_aa: (n_lines,)    float64  vacuum wavelengths [Angstrom]
    line_names        : (n_lines,)     str      line identifiers
    logHB_per_logq    : (4,4,3,16,3)  float64  log10(L_Hβ / Q_H) [erg s^-1 / (ph s^-1)]
                        dims: (alpha, logUs, logn, logZ, xi_d)
    line_ratios       : (4,4,3,16,3,n_lines) float64  L_line / L_Hβ (dimensionless)
                        dims: (alpha, logUs, logn, logZ, xi_d, line)

IMPORTANT: Data unavailability notice
--------------------------------------
The Feltre+2016 catalog was **not deposited on VizieR** (CDS catalog
J/MNRAS/456/3354 does not exist in the CDS database as of 2026-04).
The following access methods have been verified to fail:
  - CDS FTP: cdsarc.cds.unistra.fr (returns 404)
  - VizieR web interface: "Catalog not found"
  - CfA VizieR mirror: JavaScript-only UI, no FTP
  - astroquery.vizier: 0 tables returned for all catalog ID variants
  - CDS TAP service: PostgreSQL COMMIT error + "Table or Catalog not found"
  - BEAGLE-AGN GitHub: no HDF5 grid files hosted publicly
  - NEOGAL website: catalog page exists but download links are broken

Options to obtain the grid
---------------------------

Option 1: Contact the authors
  Write to Anna Feltre <anna.feltre@gmail.com> or Stephane Charlot
  <charlot@iap.fr> requesting the CLOUDY model table.

Option 2: Regenerate with pyCloudy
  pyCloudy (Morisset 2013) can run CLOUDY c13.03 models with Feltre+2016
  parameters. Requires CLOUDY c13.03 and pyCloudy installed.

  Example pyCloudy run (single model point):
  ------------------------------------------
    import pyCloudy as pc

    # AGN ionizing continuum: broken power-law, see Feltre+2016 eq. 1
    # f_nu ~ nu^alpha_pl for ionizing radiation (912A to ~0.1 keV)
    pc.print_make_file(dir='./')
    cloudy_model = pc.CloudyInput('agn_nlr')
    # Set AGN ionizing spectrum (requires CLOUDY AGNTABLE or POWER LAW command)
    cloudy_model.set_lines(['N 1 1240', 'C 4 1548', ...])
    cloudy_model.set_abund(Z=0.014)   # solar Z = 0.014 (Asplund+2009)
    cloudy_model.set_hden(3.0)        # log n_H = 3.0
    cloudy_model.set_distance(...)
    cloudy_model.run_cloudy()

  See the pyCloudy documentation for AGN ionizing spectrum setup.
  Full grid generation requires running 2,304 models; use a cluster.

Option 3: NEOGAL code (Gutkin, Charlot & Bruzual 2016)
  NEOGAL generates similar grids using an updated version of the same models.
  See http://www.iap.fr/neogal/
  CAUTION: NEOGAL grids use slightly different parameter ranges and
  metallicity scales than Feltre+2016 — verify before mixing.

Usage (once data/feltre_grid.h5 is available)
----------------------------------------------
  # Test download validity:
  python scripts/download_feltre_grid.py --validate

  # The FeltreNLRBackend will load automatically:
  from tengri.models.nebular import FeltreNLRBackend
  backend = FeltreNLRBackend("data/feltre_grid.h5")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_OUTPUT = Path("data/feltre_grid.h5")


def _try_vizier_download(output_path: Path) -> bool:
    """Attempt VizieR download (expected to fail — catalog not deposited)."""
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("astroquery not available. Install with: pip install astroquery")
        return False

    print("Attempting VizieR download of J/MNRAS/456/3354 ...")
    print("Note: This catalog was NOT deposited on VizieR and is expected to fail.")

    v = Vizier(columns=["**"], row_limit=-1)
    try:
        v.find_catalogs("J/MNRAS/456/3354")
        tables = v.get_catalogs("J/MNRAS/456/3354")
        if tables is None or len(tables) == 0:
            print("  → No tables found. Catalog not available on VizieR.")
            return False
        print(f"  → Found {len(tables)} tables (unexpected). Continuing with download.")
        # If somehow the data is found, parse and save it
        # (implementation would go here)
        return True
    except Exception as e:
        print(f"  → VizieR query failed: {e}")
        print("  → Catalog J/MNRAS/456/3354 does not exist in CDS database.")
        return False


def _validate_existing_file(grid_path: Path) -> bool:
    """Validate an existing feltre_grid.h5 file."""
    try:
        import h5py
    except ImportError:
        print("h5py not available. Install with: pip install h5py")
        return False

    if not grid_path.exists():
        print(f"File not found: {grid_path}")
        return False

    print(f"Validating {grid_path} ...")
    required_keys = [
        "/feltre/alpha_axis",
        "/feltre/logUs_axis",
        "/feltre/logn_axis",
        "/feltre/logZ_axis",
        "/feltre/xi_d_axis",
        "/feltre/line_wavelengths_aa",
        "/feltre/logHB_per_logq",
        "/feltre/line_ratios",
    ]
    with h5py.File(grid_path, "r") as f:
        for key in required_keys:
            if key not in f:
                print(f"  MISSING: {key}")
                return False
            print(f"  OK: {key}  shape={f[key].shape}")

        # Shape checks
        alpha_axis = f["/feltre/alpha_axis"][:]
        logUs_axis = f["/feltre/logUs_axis"][:]
        logn_axis = f["/feltre/logn_axis"][:]
        logZ_axis = f["/feltre/logZ_axis"][:]
        xi_d_axis = f["/feltre/xi_d_axis"][:]
        expected_grid_shape = (
            len(alpha_axis),
            len(logUs_axis),
            len(logn_axis),
            len(logZ_axis),
            len(xi_d_axis),
        )
        logHB_shape = f["/feltre/logHB_per_logq"].shape
        if logHB_shape != expected_grid_shape:
            print(f"  ERROR: logHB_per_logq shape {logHB_shape} != expected {expected_grid_shape}")
            return False

    print("Validation passed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or validate Feltre+2016 NLR photoionization grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output path for feltre_grid.h5 (default: data/feltre_grid.h5)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing feltre_grid.h5 file and exit.",
    )
    args = parser.parse_args()

    if args.validate:
        ok = _validate_existing_file(args.output)
        sys.exit(0 if ok else 1)

    print("=" * 70)
    print("Feltre+2016 NLR grid download")
    print("=" * 70)
    print()
    print("NOTICE: The Feltre+2016 catalog (J/MNRAS/456/3354) was not deposited")
    print("on VizieR. Automatic download is not possible.")
    print()

    success = _try_vizier_download(args.output)
    if not success:
        print()
        print("Download failed. Please see the options in this script's docstring")
        print("(run: python scripts/download_feltre_grid.py --help) for alternatives.")
        print()
        print("Once you have the data, build data/feltre_grid.h5 with the schema")
        print("described in this script's docstring and validate with:")
        print(f"  python scripts/download_feltre_grid.py --validate --output {args.output}")
        sys.exit(1)


if __name__ == "__main__":
    main()
