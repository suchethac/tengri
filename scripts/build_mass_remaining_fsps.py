#!/usr/bin/env python3
"""Build surviving-mass-fraction tables from python-fsps (#2751).

Each table holds the fraction of the formed mass of a single-age population
that is in living stars plus stellar remnants, per (metallicity, age) node, for
the isochrone set and IMF compiled into the local FSPS. FSPS computes it with
``add_stellar_remnants=1`` (Renzini & Ciotti 1993, as implemented in
``add_remnants.f90``; Conroy, Gunn & White 2009).

The table depends on the isochrones and the IMF, not on the spectral library,
so one table serves every spectral library built on those isochrones.

Output: ``src/tengri/data/ssp_mass_remaining/mass_remaining_<isoc>_<imf>.h5``
(package data, loaded through ``importlib.resources``).

Usage::

    SPS_HOME=~/Projects/fsps python scripts/build_mass_remaining_fsps.py \
        --isoc mist --imf chabrier

Only the isochrone set compiled into the local FSPS can be built; any other
request is refused.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np

#: python-fsps ``imf_type`` for each IMF name.
_IMF_TYPE = {"salpeter": 0, "chabrier": 1, "kroupa": 2}

#: Table ``isochrones`` code -> the token python-fsps reports in ``libraries``.
_ISOC_LIBRARY_TOKEN = {"mist": "mist", "prsc": "parsec", "pdva": "padova", "bsti": "basti"}

_CITATION = (
    "Conroy, Gunn & White 2009, ApJ 699, 486 (doi:10.1088/0004-637X/699/1/486); "
    "Renzini & Ciotti 1993, ApJ 416, L49 (doi:10.1086/187068)"
)


def _decode(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isoc", required=True, choices=sorted(_ISOC_LIBRARY_TOKEN))
    parser.add_argument("--imf", required=True, choices=sorted(_IMF_TYPE))
    args = parser.parse_args()

    sps_home = os.environ["SPS_HOME"]
    import fsps

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

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "src" / "tengri" / "data" / "ssp_mass_remaining"
    out = out_dir / f"mass_remaining_{args.isoc}_{args.imf}.h5"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=sps_home, capture_output=True, text=True, check=True
    ).stdout.strip()

    with h5py.File(out, "w") as f:
        f.create_dataset("log10_age_yr", data=log_age_yr)
        f.create_dataset("log10_z_abs", data=np.log10(zlegend))
        f.create_dataset("mass_remaining", data=table)
        f.attrs["quantity"] = "living stars + remnants per unit formed mass"
        f.attrs["isochrones"] = args.isoc
        f.attrs["imf"] = args.imf
        f.attrs["source"] = f"python-fsps {fsps.__version__}"
        f.attrs["source_commit"] = commit
        f.attrs["remnant_prescription"] = "Renzini & Ciotti 1993 (FSPS add_remnants.f90)"
        f.attrs["citation"] = _CITATION
        f.attrs["generator"] = str(Path(__file__).resolve().relative_to(repo_root))
        f.attrs["generator_args"] = f"--isoc {args.isoc} --imf {args.imf}"

    print(f"wrote {out}  shape={table.shape}  range=[{table.min():.6f}, {table.max():.6f}]")


if __name__ == "__main__":
    main()
