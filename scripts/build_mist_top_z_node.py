#!/usr/bin/env python3
"""Build the MIST top-metallicity node that ``load_ssp(..., top_z="mist")`` inserts.

The Z = 0.06 node of every FSPS-PARSEC grid lacks stars above 12 Msun for log age
<= 7.15 (the PARSEC massive-star tracks stop at Z = 0.04). The opt-in replaces the
clip target with a MIST [Z/H] = +0.5 simple stellar population: the highest MIST
metallicity node of FSPS, Z = 0.0142 x 10^0.5 = 0.0449 (MIST's Z_sun = 0.0142 is the
reference of [Z/H]; Choi et al. 2016, ApJ 823, 102, arXiv:1604.08592).

The node uses the same spectral library (MILES), IMF and age grid as the PARSEC grid
it is inserted into. Its surviving mass comes from the same FSPS build as its spectra
(``add_stellar_remnants=1``, Renzini & Ciotti 1993).

The isochrone set is a compile-time choice of FSPS, so this needs a python-fsps that
was compiled with MIST (the default build; ``fsps.StellarPopulation().libraries``
reports ``mist``). A PARSEC, Padova or BaSTI build is refused. To make one without
touching a shared install, run this script from a scratch virtual environment.

Usage::

    SPS_HOME=<fsps checkout> <venv>/bin/python scripts/build_mist_top_z_node.py \\
        --grid data/fsps_prsc_miles_chabrier.h5
    SPS_HOME=<fsps checkout> <venv>/bin/python scripts/build_mist_top_z_node.py \\
        --grid data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5 --wne

``--wne`` adds FSPS nebular emission at ``--gas-logu`` / ``--gas-logz`` (defaults
-3.0 and 0.0, the setting of the wNE grid; with a PARSEC build the same call
reproduces that grid's Z = 0.04 node to 1e-7).

Output: ``src/tengri/data/ssp_top_z/mist_top_node_<library>_<imf>[_wNE...].h5``,
float64, gzip-compressed, in FSPS's native Lsun/Hz per Msun formed, exactly as the
grid files store it (the loader applies the same unit factor to both).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

_IMF_TYPE = {"salpeter": 0, "chabrier": 1, "kroupa": 2}

#: [Z/H] = +0.5 is the highest MIST node; it is the last entry of the FSPS zlegend.
_MIST_FEH = 0.5
#: MIST protosolar Z (Choi et al. 2016, Table 1).
_MIST_ZSUN = 0.0142

_CITATION = (
    "Choi et al. 2016, ApJ 823, 102 (arXiv:1604.08592); Dotter 2016, ApJS 222, 8 "
    "(arXiv:1601.05144); Conroy, Gunn & White 2009, ApJ 699, 486; "
    "Renzini & Ciotti 1993, ApJ 416, L49"
)


def _stem_tokens(grid: Path) -> tuple[str, str, str]:
    """Return (library, imf, wNE suffix) read off a grid file name."""
    base, sep, neb = grid.stem.partition("_wNE")
    tokens = base.split("_")
    return tokens[2], tokens[-1], sep + neb


def _fsps_sources_sha256(fsps_pkg: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((fsps_pkg / "libfsps" / "src").glob("*.f90")):
        digest.update(f"{path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, required=True, help="The PARSEC grid to match.")
    parser.add_argument("--wne", action="store_true", help="Add FSPS nebular emission.")
    parser.add_argument("--gas-logu", type=float, default=-3.0)
    parser.add_argument("--gas-logz", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import fsps

    library, imf, neb_suffix = _stem_tokens(args.grid)
    if bool(neb_suffix) != args.wne:
        raise ValueError(f"{args.grid.name}: --wne must be given exactly for a wNE grid")
    kwargs = {"zcontinuous": 0, "imf_type": _IMF_TYPE[imf], "sfh": 0, "add_stellar_remnants": 1}
    if args.wne:
        kwargs.update(add_neb_emission=True, gas_logu=args.gas_logu, gas_logz=args.gas_logz)
    sp = fsps.StellarPopulation(**kwargs)
    libs = [x.decode().lower() if isinstance(x, bytes) else str(x).lower() for x in sp.libraries]
    if "mist" not in libs or library not in libs:
        raise ValueError(f"this FSPS build reports {libs}; need a MIST + {library} compile")

    zlegend = np.asarray(sp.zlegend, dtype=np.float64)
    z_expected = _MIST_ZSUN * 10.0**_MIST_FEH
    if abs(zlegend[-1] / z_expected - 1.0) > 1e-6:
        raise ValueError(f"top FSPS-MIST node is Z={zlegend[-1]}, expected {z_expected}")
    wave, spec = sp.get_spectrum(tage=0.0, zmet=len(zlegend))
    mass = np.asarray(sp.stellar_mass, dtype=np.float64)
    log_age = np.asarray(sp.log_age, dtype=np.float64)

    with h5py.File(args.grid, "r") as g:
        grid_age = g["ssp_lg_age_gyr"][:] + 9.0
        grid_wave = g["ssp_wave"][:]
    if not np.array_equal(np.asarray(wave), grid_wave):
        raise ValueError("FSPS wavelength grid differs from the PARSEC grid's")
    idx = np.array([int(np.argmin(np.abs(log_age - a))) for a in grid_age])
    if np.max(np.abs(log_age[idx] - grid_age)) > 1e-6:
        raise ValueError("the PARSEC grid's ages are not all FSPS-MIST ages")
    flux = np.asarray(spec, dtype=np.float64)[idx]
    mass = mass[idx]
    if not (np.all(np.isfinite(flux)) and np.all(flux >= 0.0) and np.all(mass > 0.0)):
        raise ValueError("non-finite or non-positive FSPS output")

    name = f"mist_top_node_{library}_{imf}{neb_suffix}.h5"
    out = args.out or Path(__file__).resolve().parent.parent / "src/tengri/data/ssp_top_z" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = ""
    sps_home = Path(__import__("os").environ["SPS_HOME"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=sps_home, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    with h5py.File(out, "w") as f:
        f.create_dataset("log10_z_abs", data=float(np.log10(zlegend[-1])))
        f.create_dataset("log10_age_yr", data=grid_age)
        f.create_dataset("ssp_wave", data=grid_wave, compression="gzip", compression_opts=9)
        f.create_dataset(
            "ssp_flux", data=flux, compression="gzip", compression_opts=9, shuffle=True
        )
        f.create_dataset("mass_remaining", data=mass)
        f.attrs["isochrones"] = "mist"
        f.attrs["spectral_library"] = library
        f.attrs["imf"] = imf
        f.attrs["z_mass_fraction"] = float(zlegend[-1])
        f.attrs["feh"] = _MIST_FEH
        f.attrs["z_sun_reference"] = _MIST_ZSUN
        f.attrs["units_flux"] = "Lsun/Hz/Msun (FSPS native Lsun = 3.839e33 erg/s)"
        f.attrs["nebular_emission"] = (
            f"FSPS gas_logu={args.gas_logu} gas_logz={args.gas_logz}" if args.wne else "none"
        )
        f.attrs["source"] = f"python-fsps {fsps.__version__}"
        f.attrs["source_commit"] = commit
        f.attrs["fsps_source_sha256"] = _fsps_sources_sha256(Path(fsps.__file__).resolve().parent)
        f.attrs["remnant_prescription"] = "Renzini & Ciotti 1993 (FSPS add_remnants.f90)"
        f.attrs["citation"] = _CITATION
        f.attrs["generator"] = "scripts/build_mist_top_z_node.py"
        f.attrs["generator_args"] = re.sub(
            r"\s+", " ", f"--grid {args.grid.name} {'--wne' if args.wne else ''}"
        )
    print(
        f"wrote {out}  Z={zlegend[-1]:.8f}  flux{flux.shape}  {out.stat().st_size / 2**20:.2f} MiB"
    )


if __name__ == "__main__":
    main()
