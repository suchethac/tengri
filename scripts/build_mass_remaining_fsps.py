#!/usr/bin/env python3
"""Build surviving-mass-fraction tables from python-fsps.

Each table holds the fraction of the formed mass of a single-age population
that is in living stars plus stellar remnants, per (metallicity, age) node, for
the isochrone set and IMF compiled into the local FSPS. FSPS computes it with
``add_stellar_remnants=1`` (Renzini & Ciotti 1993, as implemented in
``add_remnants.f90``; Conroy, Gunn & White 2009).

The table depends on the isochrones and the IMF, not on the spectral library,
so one table serves every spectral library built on those isochrones.

The FSPS build is made by ``scripts/build_fsps_isochrone_env.sh``, which writes
the compile flags into the virtual environment as ``FSPS_BUILD_FLAGS``. This
script reads that file and records it, with the python-fsps version and a
checksum of the bundled Fortran sources, in the table's attributes.

Output: ``src/tengri/data/ssp_mass_remaining/mass_remaining_<isoc>_<imf>.h5``
(package data, loaded through ``importlib.resources``), or the path given by
``--out``.

Usage::

    <venv>/bin/python scripts/build_mass_remaining_fsps.py --isoc mist --imf chabrier

Only the isochrone set compiled into the local FSPS can be built; any other
request is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

#: python-fsps ``imf_type`` for each IMF name.
_IMF_TYPE = {"salpeter": 0, "chabrier": 1, "kroupa": 2}

#: Table ``isochrones`` code -> the token python-fsps reports in ``libraries``
#: (FSPS names the PARSEC, Padova and BaSTI sets by these codes).
_ISOC_LIBRARY_TOKEN = {"mist": "mist", "prsc": "prsc", "pdva": "pdva", "bsti": "bsti"}

#: Name of the build-flag record that ``build_fsps_isochrone_env.sh`` writes into a venv.
BUILD_FLAGS_FILE = "FSPS_BUILD_FLAGS"

_CITATION = (
    "Conroy, Gunn & White 2009, ApJ 699, 486 (doi:10.1088/0004-637X/699/1/486); "
    "Renzini & Ciotti 1993, ApJ 416, L49 (doi:10.1086/187068)"
)


def _decode(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


def read_build_flags(prefix: Path) -> str:
    """Return the FSPS compile flags recorded in a build environment.

    Parameters
    ----------
    prefix : Path
        Virtual environment prefix (``sys.prefix``) that holds the record.

    Returns
    -------
    str
        The ``FFLAGS`` string the FSPS build was compiled with.

    Raises
    ------
    FileNotFoundError
        If the environment has no ``FSPS_BUILD_FLAGS`` record.
    """
    record = prefix / BUILD_FLAGS_FILE
    if not record.is_file():
        raise FileNotFoundError(
            f"{record} is missing: build the FSPS environment with "
            "scripts/build_fsps_isochrone_env.sh so the compile flags are recorded"
        )
    return record.read_text(encoding="utf-8").strip()


def fsps_source_sha256(fsps_pkg: Path) -> str:
    """Checksum of the Fortran sources bundled with python-fsps.

    Parameters
    ----------
    fsps_pkg : Path
        Directory of the installed ``fsps`` package.

    Returns
    -------
    str
        sha256 over the sorted (relative path, file sha256) pairs of
        ``libfsps/src/*.f90``, the FSPS source the local build compiled.
    """
    digest = hashlib.sha256()
    src = fsps_pkg / "libfsps" / "src"
    for path in sorted(src.glob("*.f90")):
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{path.name} {file_digest}\n".encode())
    return digest.hexdigest()


def build_table(sp, n_age_ages: np.ndarray) -> np.ndarray:
    """Evaluate ``sp.stellar_mass`` on every (zmet, age) node.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        Population with ``sfh=0`` and ``add_stellar_remnants=1``.
    n_age_ages : ndarray, shape (n_age,)
        FSPS SSP ages [log10(yr)].

    Returns
    -------
    ndarray, shape (n_met, n_age)
        Living stars + remnants per unit formed mass [dimensionless].
    """
    n_met = len(sp.zlegend)
    table = np.zeros((n_met, len(n_age_ages)), dtype=np.float64)
    for i_z in range(n_met):
        for i_age, lg_age in enumerate(n_age_ages):
            sp.get_spectrum(tage=10.0 ** (lg_age - 9.0), zmet=i_z + 1)
            table[i_z, i_age] = float(np.atleast_1d(sp.stellar_mass)[0])
    return table


def _default_out(isoc: str, imf: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return (
        repo_root
        / "src"
        / "tengri"
        / "data"
        / "ssp_mass_remaining"
        / (f"mass_remaining_{isoc}_{imf}.h5")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isoc", required=True, choices=sorted(_ISOC_LIBRARY_TOKEN))
    parser.add_argument("--imf", required=True, choices=sorted(_IMF_TYPE))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file (default: the package data table for --isoc and --imf).",
    )
    args = parser.parse_args()

    import fsps

    build_flags = read_build_flags(Path(sys.prefix))
    sps_home = Path(_required_env("SPS_HOME"))
    sp = fsps.StellarPopulation(
        zcontinuous=0, imf_type=_IMF_TYPE[args.imf], sfh=0, add_stellar_remnants=1
    )
    libraries = [_decode(x).lower() for x in sp.libraries]
    if _ISOC_LIBRARY_TOKEN[args.isoc] not in libraries:
        raise ValueError(
            f"requested isochrones {args.isoc!r} but this FSPS build reports "
            f"{libraries}; recompile FSPS with the matching isochrone set"
        )

    zlegend = np.asarray(sp.zlegend, dtype=np.float64)
    log_age_yr = np.asarray(sp.log_age, dtype=np.float64)
    table = build_table(sp, log_age_yr)

    if not np.all(np.isfinite(table)) or not np.all(table > 0.0):
        raise ValueError("FSPS returned non-finite or non-positive surviving mass")
    if np.any(np.diff(log_age_yr) <= 0.0):
        raise ValueError("FSPS SSP ages are not strictly increasing")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sps_home, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = args.out if args.out is not None else _default_out(args.isoc, args.imf)
    fsps_pkg = Path(fsps.__file__).resolve().parent

    with h5py.File(out, "w") as f:
        f.create_dataset("log10_age_yr", data=log_age_yr)
        f.create_dataset("log10_z_abs", data=np.log10(zlegend))
        f.create_dataset("mass_remaining", data=table)
        f.attrs["quantity"] = "living stars + remnants per unit formed mass"
        f.attrs["isochrones"] = args.isoc
        f.attrs["imf"] = args.imf
        f.attrs["source"] = f"python-fsps {fsps.__version__}"
        f.attrs["source_commit"] = commit
        f.attrs["fsps_build_flags"] = build_flags
        f.attrs["fsps_source_sha256"] = fsps_source_sha256(fsps_pkg)
        f.attrs["remnant_prescription"] = "Renzini & Ciotti 1993 (FSPS add_remnants.f90)"
        f.attrs["citation"] = _CITATION
        f.attrs["generator"] = "scripts/build_mass_remaining_fsps.py"
        f.attrs["generator_args"] = f"--isoc {args.isoc} --imf {args.imf}"

    print(f"wrote {out}  shape={table.shape}  range=[{table.min():.6f}, {table.max():.6f}]")


def _required_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise OSError(f"set {name} to the python-fsps source checkout")
    return value


if __name__ == "__main__":
    main()
