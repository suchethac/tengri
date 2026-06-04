#!/usr/bin/env python3
"""Build ``data/fritz2006_torus_grid.h5`` from CIGALE's Fritz2006 database.

The Fritz et al. (2006) smooth-dust AGN torus model provides a grid of
SEDs parameterized by six dimensions (r_ratio, tau, beta, gamma, opening_angle,
psy). This script reads the full grid from CIGALE's ``SimpleDatabase("fritz2006")``
and writes it into a tengri-native HDF5 file suitable for ND triweight
interpolation.

Parameters
----------
The grid axes are:
- ``r_ratio`` [10, 30, 60, 100, 150]: maximum-to-minimum dust torus radius ratio
- ``tau`` [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0]: optical depth at 9.7 µm
- ``beta`` [-1.0, -0.75, -0.5, -0.25, 0.0]: radial dust density power-law index
- ``gamma`` [0, 2, 4, 6]: polar dust density gradient
- ``opening_angle`` [20, 40, 60]: half-opening angle stored in CIGALE's database
  (the user-facing "full opening angle" 60/100/140 maps to this via
  ``(180 - oa) / 2``) [degrees]
- ``psy`` [0.001, 10.1, ..., 89.99]: viewing angle [degrees from torus axis]

The grid size is 5 × 8 × 5 × 4 × 3 × 10 = 12,000 SEDs, each with 178 wavelength points.

CIGALE interface
----------------
The CIGALE ``fritz2006`` module exposes a ``SimpleDatabase`` API::

    from pcigale.data import SimpleDatabase as Database

    with Database("fritz2006") as db:
        model = db.get(r_ratio=60, tau=1.0, beta=-0.5, gamma=4.0, opening_angle=60.0, psy=0.001)
        # model has: .wl (wavelengths in nm), .disk, .dust, .norm

Each model has:
- ``.wl`` — wavelength array [nm]
- ``.disk`` — accretion disk SED (direct + scattered) [erg/s/Hz]
- ``.dust`` — torus thermal dust emission [erg/s/Hz]
- ``.norm`` — overall normalization factor

HDF5 schema
-----------
The output file is organized as::

    /fritz2006/
      r_ratio_axis       (5,)        — r_ratio values [10, 30, 60, 100, 150]
      tau_axis           (8,)        — tau values [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0]
      beta_axis          (5,)        — beta values [-1.0, -0.75, -0.5, -0.25, 0.0]
      gamma_axis         (4,)        — gamma values [0, 2, 4, 6]
      opening_angle_axis (3,)        — opening_angle values [20, 40, 60] (degrees)
      psy_axis           (10,)       — psy values [0.001, 10.1, ..., 89.99]
      wavelength_aa      (178,)      — common wavelength grid [Angstrom]
      dust               (5,8,5,4,3,10,178)  — torus dust emission [erg/s/Hz]
      disk               (5,8,5,4,3,10,178)  — accretion disk SED [erg/s/Hz]

dtype: float64 (matching CIGALE precision)
compression: gzip level 4 (balance speed vs. file size)

Reference
---------
.. [1] O. Fritz et al., "Dust tori around Type II active nuclei. I. Observational
   constraints and allowed dust models," A&A, 470, 221 (2006).
   arXiv:0606147. https://doi.org/10.1051/0004-6361:20066130
.. [2] M. Boquien et al., "CIGALE: Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
   https://doi.org/10.1051/0004-6361/201834156

Usage
-----
::

    python scripts/build_fritz2006_grid.py

Requires pcigale to be installed (included in tengri's venv).
Output written to ``data/fritz2006_torus_grid.h5``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# Grid axes — exact ordering and values as in CIGALE fritz2006.py
R_RATIO_GRID = np.array([10.0, 30.0, 60.0, 100.0, 150.0])
TAU_GRID = np.array([0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0])
BETA_GRID = np.array([-1.0, -0.75, -0.5, -0.25, 0.0])
GAMMA_GRID = np.array([0.0, 2.0, 4.0, 6.0])
# SimpleDatabase('fritz2006') keys the half-opening angle as [20, 40, 60], NOT
# the user-facing "full opening angle". CIGALE's fritz2006.py maps user input to
# the stored key via (180 - oa) / 2, so we must query the database with these.
OPENING_ANGLE_GRID = np.array([20.0, 40.0, 60.0])
PSY_GRID = np.array([0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 89.99])


def build_fritz_grid(dest: Path | str | None = None, *, force: bool = False) -> Path:
    """Build the Fritz2006 torus grid HDF5 from CIGALE's ``SimpleDatabase``.

    Parameters
    ----------
    dest : path-like, optional
        Target directory for ``fritz2006_torus_grid.h5``. Defaults to
        ``$TENGRI_DATA_DIR`` if set, else ``data/`` relative to the cwd.
    force : bool, optional
        Rebuild even if the grid already exists. Default ``False``.

    Returns
    -------
    pathlib.Path
        Path to the written (or already-present) grid file.

    Raises
    ------
    RuntimeError
        If ``pcigale`` is not importable — the grid can only be built from a
        local CIGALE install (it ships the Fritz templates as pcigale pickles).
    """
    import os

    if dest is None:
        dest = os.environ.get("TENGRI_DATA_DIR", "data")
    out_path = Path(dest) / "fritz2006_torus_grid.h5"
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"Fritz2006 grid already present at {out_path}; skipping build.")
        return out_path

    try:
        from pcigale.data import SimpleDatabase as Database
    except ImportError as e:
        raise RuntimeError(
            "pcigale is not importable, so the Fritz2006 grid cannot be built "
            "from CIGALE. Install it (``pip install pcigale``) or run from "
            "tengri's main venv, then retry."
        ) from e

    n_total = int(
        np.prod(
            [
                len(R_RATIO_GRID),
                len(TAU_GRID),
                len(BETA_GRID),
                len(GAMMA_GRID),
                len(OPENING_ANGLE_GRID),
                len(PSY_GRID),
            ]
        )
    )
    print(
        f"Building Fritz2006 grid: {len(R_RATIO_GRID)} × {len(TAU_GRID)} × "
        f"{len(BETA_GRID)} × {len(GAMMA_GRID)} × {len(OPENING_ANGLE_GRID)} × "
        f"{len(PSY_GRID)} = {n_total} SEDs"
    )

    # Allocate output arrays
    n_r = len(R_RATIO_GRID)
    n_tau = len(TAU_GRID)
    n_beta = len(BETA_GRID)
    n_gamma = len(GAMMA_GRID)
    n_oa = len(OPENING_ANGLE_GRID)
    n_psy = len(PSY_GRID)
    n_wave = None
    wavelength_aa = None

    # Placeholder for grid data
    dust_grid = None
    disk_grid = None

    # CIGALE's SimpleDatabase keys the torus by the HALF-OPENING angle in
    # degrees [20, 40, 60]. fritz2006.py maps the user-facing "full opening
    # angle" to this half-angle via (180 - oa) / 2; we query the database with
    # the stored half-angle values directly.

    with Database("fritz2006") as db:
        for i_r, r in enumerate(R_RATIO_GRID):
            for i_tau, tau in enumerate(TAU_GRID):
                for i_beta, beta in enumerate(BETA_GRID):
                    for i_gamma, gamma in enumerate(GAMMA_GRID):
                        for i_oa, oa in enumerate(OPENING_ANGLE_GRID):
                            for i_psy, psy in enumerate(PSY_GRID):
                                # Query CIGALE for this point
                                model = db.get(
                                    r_ratio=float(r),
                                    tau=float(tau),
                                    beta=float(beta),
                                    gamma=float(gamma),
                                    opening_angle=float(oa),
                                    psy=float(psy),
                                )

                                # Initialize arrays on first model
                                if n_wave is None:
                                    n_wave = len(model.wl)
                                    wavelength_aa = (
                                        np.array(model.wl, dtype=np.float64) * 10.0
                                    )  # nm → Å
                                    dust_grid = np.zeros(
                                        (n_r, n_tau, n_beta, n_gamma, n_oa, n_psy, n_wave),
                                        dtype=np.float64,
                                    )
                                    disk_grid = np.zeros_like(dust_grid)

                                # Store dust (torus thermal emission) and disk
                                dust_grid[i_r, i_tau, i_beta, i_gamma, i_oa, i_psy, :] = np.array(
                                    model.dust, dtype=np.float64
                                )
                                disk_grid[i_r, i_tau, i_beta, i_gamma, i_oa, i_psy, :] = np.array(
                                    model.disk, dtype=np.float64
                                )

                                # Progress
                                if (i_psy + 1) % 5 == 0:
                                    total_idx = (
                                        i_r * n_tau * n_beta * n_gamma * n_oa * n_psy
                                        + i_tau * n_beta * n_gamma * n_oa * n_psy
                                        + i_beta * n_gamma * n_oa * n_psy
                                        + i_gamma * n_oa * n_psy
                                        + i_oa * n_psy
                                        + i_psy
                                    )
                                    total_seds = np.prod(
                                        [n_r, n_tau, n_beta, n_gamma, n_oa, n_psy]
                                    )
                                    pct = 100.0 * (total_idx + 1) / total_seds
                                    print(
                                        f"  [{pct:5.1f}%] r={r:5.0f}, tau={tau:4.1f}, "
                                        f"beta={beta:5.2f}, gamma={gamma:5.1f}, "
                                        f"oa={oa:5.0f}, psy={psy:6.2f}"
                                    )

    # Write HDF5 file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        g = f.create_group("fritz2006")

        # Write axes
        g.create_dataset("r_ratio_axis", data=R_RATIO_GRID, dtype=np.float64, compression="gzip")
        g.create_dataset("tau_axis", data=TAU_GRID, dtype=np.float64, compression="gzip")
        g.create_dataset("beta_axis", data=BETA_GRID, dtype=np.float64, compression="gzip")
        g.create_dataset("gamma_axis", data=GAMMA_GRID, dtype=np.float64, compression="gzip")
        g.create_dataset(
            "opening_angle_axis", data=OPENING_ANGLE_GRID, dtype=np.float64, compression="gzip"
        )
        g.create_dataset("psy_axis", data=PSY_GRID, dtype=np.float64, compression="gzip")

        # Write wavelength (ascending order for interpolation)
        g.create_dataset("wavelength_aa", data=wavelength_aa, dtype=np.float64, compression="gzip")

        # Write grid data (the primary output — dust is what we use for the torus block)
        g.create_dataset("dust", data=dust_grid, dtype=np.float32, compression="gzip")
        g.create_dataset("disk", data=disk_grid, dtype=np.float32, compression="gzip")

        # Metadata
        g.attrs["title"] = "Fritz et al. (2006) AGN torus SED grid"
        g.attrs["source"] = "CIGALE pcigale.data.SimpleDatabase('fritz2006')"
        g.attrs["grid_shape"] = "[r_ratio, tau, beta, gamma, opening_angle, psy, wavelength]"
        g.attrs["dust_unit"] = "erg/s/Hz (accretion-disk normalized, pre-interpolation)"
        g.attrs["disk_unit"] = "erg/s/Hz (accretion-disk component, pre-interpolation)"
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["opening_angle_unit"] = "degrees (half-opening angle, direct grid parameter)"
        g.attrs["psy_unit"] = "degrees (viewing angle from torus axis; 0=type2, 90=type1)"
        g.attrs["generated_by"] = "scripts/build_fritz2006_grid.py"
        g.attrs["citation"] = (
            "Fritz et al. (2006) MNRAS 366, 767; Boquien et al. (2019) CIGALE A&A 622, A103"
        )

    file_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(
        f"  Grid shape (r_ratio, tau, beta, gamma, oa, psy, wave): "
        f"({n_r}, {n_tau}, {n_beta}, {n_gamma}, {n_oa}, {n_psy}, {n_wave})"
    )
    print(f"  Wavelength range: {wavelength_aa[0]:.2f} – {wavelength_aa[-1]:.2f} Å")
    print("Done. Use in SEDModel.build(agn={'torus': {'type': 'fritz', ...}}).")
    return out_path


def main() -> int:
    """CLI: build the Fritz2006 grid into ``--dest`` (or ``data/``)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build data/fritz2006_torus_grid.h5 from CIGALE's fritz2006 database."
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Target directory (default: $TENGRI_DATA_DIR or ./data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the grid already exists.",
    )
    args = parser.parse_args()
    try:
        build_fritz_grid(args.dest, force=args.force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
