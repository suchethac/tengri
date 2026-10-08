#!/usr/bin/env python3
"""Build the ProGeny surviving-mass table from ProGeny's own ``SMstar`` output (#2751).

ProGeny (Robotham & Bellstedt 2024; Bellstedt & Robotham 2024) reports, for every
(metallicity, age) node of an SSP, ``SMstar``: the fraction of the formed mass that
is in living stars plus stellar remnants (``SMgas = 1 - SMstar`` is the gas returned
to the ISM). It is stored in the ``Zevo<i>`` tables of the ProSpect speclib file
``PG_Ch_Mi_C3K.fits`` (MIST isochrones, C3K atmospheres, Chabrier IMF 0.1-100 Msun),
which is the package's own output for the isochrones and IMF of the grid
``pgny_mist_c3k_chabrier.h5``. This script repackages that quantity; it does not
rerun ProGeny.

ProGeny's remnant treatment (``progenyMakeSSP``, ``rem_frac = "get"``): ``SMstar``
sums the isochrone ``Mass`` times the IMF weight over the tracked points, and adds
the remnants of stars above the isochrone's most massive point from the
remnant-to-initial mass ratio of that point (``missing_func``). That is not the
Renzini & Ciotti (1993) prescription FSPS uses; the table records it in its
``remnant_prescription`` attribute.

Output: a table in ``src/tengri/data/ssp_mass_remaining/`` by default, or ``--out``.
The registry row is not flipped by this script.

Usage::

    python scripts/build_mass_remaining_pgny.py \\
        --input /path/to/PG_Ch_Mi_C3K.fits \\
        --grid /path/to/pgny_mist_c3k_chabrier.h5

Requires ``astropy`` (to read the FITS speclib) and ``h5py``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

#: Node agreement between the table and the grid, in log10 [dex].
NODE_ATOL = 1e-6

_CITATION = (
    "Robotham & Bellstedt 2024, ProGeny I, RASTI 4, 19 (arXiv:2410.17697); "
    "Bellstedt & Robotham 2024, ProGeny II, MNRAS 540, 2703 (arXiv:2410.17698)"
)
_REMNANT = (
    "ProGeny progenyMakeSSP rem_frac='get': tracked remnants are in the isochrone "
    "Mass; untracked remnants above the isochrone's top mass use its remnant-to-"
    "initial mass ratio (not Renzini & Ciotti 1993)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_progeny_smstar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ProGeny's metallicities, ages and ``SMstar`` from a ProSpect speclib FITS.

    Parameters
    ----------
    path : Path
        ``PG_Ch_Mi_C3K.fits`` (or another ProGeny speclib FITS with ``Z``, ``Age``
        and ``Zevo<i>`` extensions).

    Returns
    -------
    z_abs : ndarray, shape (n_z,)
        Absolute metal mass fractions [dimensionless].
    age_yr : ndarray, shape (n_age,)
        SSP ages [yr].
    smstar : ndarray, shape (n_z, n_age)
        Living stars + remnants per unit formed mass [dimensionless].

    Raises
    ------
    ValueError
        The extensions are missing or the values are non-finite or outside (0, 1].
    """
    with fits.open(path, memmap=True) as hdul:
        names = [h.name.strip() for h in hdul]
        for required in ("Z", "Age"):
            if required not in names:
                raise ValueError(f"{path}: no {required!r} extension")
        z_abs = np.array(hdul[names.index("Z")].data, dtype=np.float64)
        age_yr = np.array(hdul[names.index("Age")].data, dtype=np.float64)
        rows = []
        for i_z in range(1, z_abs.size + 1):
            key = f"Zevo{i_z}"
            if key not in names:
                raise ValueError(f"{path}: no {key!r} extension")
            rows.append(np.array(hdul[names.index(key)].data["SMstar"], dtype=np.float64))
    smstar = np.stack(rows)
    if smstar.shape != (z_abs.size, age_yr.size):
        raise ValueError(f"SMstar shape {smstar.shape} != (n_z, n_age)")
    if not np.all(np.isfinite(smstar)) or np.any(smstar <= 0.0) or np.any(smstar > 1.0):
        raise ValueError("SMstar is non-finite or outside (0, 1]")
    if np.any(np.diff(age_yr) <= 0.0) or np.any(np.diff(z_abs) <= 0.0):
        raise ValueError("ProGeny Z or age axes are not strictly increasing")
    return z_abs, age_yr, smstar


def check_grid_nodes(z_abs: np.ndarray, age_yr: np.ndarray, grid: Path) -> None:
    """Refuse a table whose nodes differ from the grid's ``ssp_lgmet`` / ``ssp_lg_age_gyr``.

    Parameters
    ----------
    z_abs : ndarray, shape (n_z,)
        Table metallicities [absolute Z].
    age_yr : ndarray, shape (n_age,)
        Table ages [yr].
    grid : Path
        SSP grid HDF5 file.

    Raises
    ------
    ValueError
        A node differs by more than ``NODE_ATOL`` [dex] in log10.
    """
    with h5py.File(grid, "r") as f:
        lgmet = np.array(f["ssp_lgmet"][:], dtype=np.float64)
        lg_age_yr = np.array(f["ssp_lg_age_gyr"][:], dtype=np.float64) + 9.0
    if lgmet.shape != z_abs.shape or not np.allclose(
        lgmet, np.log10(z_abs), rtol=0.0, atol=NODE_ATOL
    ):
        raise ValueError("metallicity nodes differ from the grid's ssp_lgmet")
    if lg_age_yr.shape != age_yr.shape or not np.allclose(
        lg_age_yr, np.log10(age_yr), rtol=0.0, atol=NODE_ATOL
    ):
        raise ValueError("age nodes differ from the grid's ssp_lg_age_gyr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="ProGeny speclib FITS")
    parser.add_argument("--grid", required=True, type=Path, help="pgny_mist_c3k_chabrier.h5")
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "src/tengri/data/ssp_mass_remaining/mass_remaining_pgny_mist_chabrier.h5"
        ),
        help="output table",
    )
    args = parser.parse_args()

    z_abs, age_yr, smstar = read_progeny_smstar(args.input)
    check_grid_nodes(z_abs, age_yr, args.grid)
    input_sha = _sha256(args.input)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("log10_age_yr", data=np.log10(age_yr))
        f.create_dataset("log10_z_abs", data=np.log10(z_abs))
        f.create_dataset("mass_remaining", data=smstar)
        f.attrs["quantity"] = "living stars + remnants per unit formed mass"
        f.attrs["isochrones"] = "pgny_mist (MIST isochrones via ProGeny)"
        f.attrs["imf"] = "chabrier (0.1-100 Msun)"
        f.attrs["source"] = (
            "ProGeny SMstar, ProSpect speclib PG_Ch_Mi_C3K.fits (ProGeny code 0.8.3 "
            "consulted, asgr/ProGeny)"
        )
        f.attrs["remnant_prescription"] = _REMNANT
        f.attrs["citation"] = _CITATION
        repo_root = Path(__file__).resolve().parent.parent
        f.attrs["generator"] = str(Path(__file__).resolve().relative_to(repo_root))
        f.attrs["generator_args"] = f"--input {args.input.name} --grid {args.grid.name}"
        f.attrs["input_sha256"] = input_sha

    print(
        f"wrote {args.out}  shape={smstar.shape}  range=[{smstar.min():.6f}, "
        f"{smstar.max():.6f}]  input_sha256={input_sha}"
    )


if __name__ == "__main__":
    main()
