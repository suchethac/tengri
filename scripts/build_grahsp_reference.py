#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Build committed upstream GRAHSP reference spectra and their builder.

The reproduction notebook and a new crossval test compare tengri's GRAHSP AGN
model against upstream GRAHSP code. This script runs the upstream modules to
produce reference SEDs, committed to data/grahsp_upstream_reference.h5, so
tests do not import upstream code.

The builder has two modes:
  1. Worker (--worker): evaluate ONE parameter set, save wavelengths and
     contributions to an .npz file. Run under upstream venv via subprocess.
  2. Driver (default): loop parameter sets, spawn a worker subprocess per set,
     collect results, write HDF5 file with provenance attributes.

Upstream sources (file:line for parameter ranges):
  - activate.py:46-61 (fracAGN=-1 -> lum5100A=1, free normalization)
  - activatepl.py:77-107 (plslope, plbendloc, plbendwidth, uvslope, cutoff)
  - activatelines.py:65-97 (FeIItemplate, AFeII, AGNtype, linewidth, Alines, ABC)
  - activategtorus.py:23-84 (fcov, Si, COOLlam, COOLwidth, HOTlam, HOTwidth, HOTfcov)
  - biattenuation.py:34-72 (E(B-V), E(B-V)-AGN)

Wavelength grid: 5000 log-spaced points 10 nm – 1e6 nm (float64, gzip).

Gotchas (from probes/p5_recon/recon.md §Q3):
  - Process-global memoisation (sed/utils.py): ONE parameter set per process.
    Anything that changes linewidth/ABC must subprocess.
  - Upstream default plslope=6.0 violates assert uvslope>plslope: always pass.
  - Pass name= to module constructors (py3.12 inspect.getfile bug).
  - Output is on union of base grid and template nodes: compare at base grid.
  - Units: lum5100A=1 => L_lam(510 nm) = 1/510 [1/nm].

Parameter sets (one-at-a-time sweeps around fiducial; fiducial defaults except
plslope=-1.7):
  0. fiducial
  1-3. plslope ∈ {-2.0, -1.5, -1.0}
  4-6. uvslope ∈ {-0.5, 0.0, 0.5}
  7-9. plbendloc ∈ {80, 100, 120} nm
  10-12. plbendwidth ∈ {0.5, 1.0, 2.0}
  13-15. AFeII ∈ {0, 2, 5}
  16-17. FeIItemplate ∈ {BruhweilerVerner08, Veron-Cetty04}
  18-20. Alines ∈ {0.5, 1, 2}
  21-23. linewidth ∈ {1000, 5000, 10000} km/s
  24-26. ABC ∈ {0, 0.3, 1.0}
  27-29. fcov ∈ {0.2, 0.4, 0.8}
  30-32. Si ∈ {-1.0, 0.0, 1.0}
  33-35. COOLlam ∈ {15.0, 17.0, 19.0} um
  36-38. COOLwidth ∈ {0.35, 0.45, 0.55}
  39-41. HOTlam ∈ {1.5, 2.0, 2.5} um
  42-44. HOTwidth ∈ {0.4, 0.5, 0.6}
  45-47. HOTfcov ∈ {0.5, 1.0, 2.0}
  48-50. E(B-V) (galaxy) ∈ {0.0, 0.05, 0.1}
  51-53. E(B-V)-AGN ∈ {0.0, 0.1, 0.2}
  54-55. AGNtype ∈ {1, 2}

Total: 56 parameter sets (~500-1000 MB raw, ~2-5 MB gzip).

Usage (driver mode — default):
  python scripts/build_grahsp_reference.py \\
    --upstream /path/to/grahsp_upstream/GRAHSP \\
    --python /path/to/venv/bin/python \\
    --out data/grahsp_upstream_reference.h5

Usage (worker mode — internal):
  python scripts/build_grahsp_reference.py --worker '{...params...}' output.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Map parameter set index to (key, value, fiducial_value)
def _parameter_sets() -> list[dict[str, Any]]:
    """Generate all parameter sets: fiducial + one-at-a-time sweeps."""
    fiducial = dict(
        plslope=-1.7,
        uvslope=0.0,
        plbendloc=100.0,
        plbendwidth=1.0,
        cutoff=10000.0,
        afeii=5.0,
        feii="BruhweilerVerner08",
        alines=1.0,
        linewidth=5000.0,
        abc=0.0,
        type=1,
        fcov=0.4,
        si=1.0,
        coollam=17.0,
        coolwidth=0.45,
        hotlam=2.0,
        hotwidth=0.5,
        hotfcov=1.0,
        ebv=0.0,
        ebv_agn=0.0,
    )

    sets = [fiducial.copy()]
    sets[0]["_name"] = "fiducial"

    # Sweeps: each varies ONE parameter from fiducial
    sweeps = [
        ("plslope", [-2.0, -1.5, -1.0]),
        ("uvslope", [-0.5, 0.0, 0.5]),
        ("plbendloc", [80.0, 100.0, 120.0]),
        ("plbendwidth", [0.5, 1.0, 2.0]),
        ("afeii", [0, 2, 5]),
        ("feii", ["BruhweilerVerner08", "Veron-Cetty04"]),
        ("alines", [0.5, 1, 2]),
        ("linewidth", [1000.0, 5000.0, 10000.0]),
        ("abc", [0.0, 0.3, 1.0]),
        ("fcov", [0.2, 0.4, 0.8]),
        ("si", [-1.0, 0.0, 1.0]),
        ("coollam", [15.0, 17.0, 19.0]),
        ("coolwidth", [0.35, 0.45, 0.55]),
        ("hotlam", [1.5, 2.0, 2.5]),
        ("hotwidth", [0.4, 0.5, 0.6]),
        ("hotfcov", [0.5, 1.0, 2.0]),
        ("ebv", [0.0, 0.05, 0.1]),
        ("ebv_agn", [0.0, 0.1, 0.2]),
        ("type", [1, 2]),
    ]

    for key, values in sweeps:
        for val in values:
            if val == fiducial.get(key):
                continue  # skip fiducial
            s = fiducial.copy()
            s[key] = val
            s["_name"] = f"{key}_{val}".replace(".", "_p")
            sets.append(s)

    return sets


def _worker(params: dict[str, Any], out_npz: Path, upstream_path: str) -> None:
    """Worker: evaluate ONE parameter set on upstream GRAHSP."""
    sys.path.insert(0, str(upstream_path))

    # Import the harness (minimal stub, lives in scripts/)
    import types
    import importlib

    U = upstream_path
    AG = U + "/database_builder/activate/agn"

    # Stub pcigale without heavy imports
    pk = types.ModuleType("pcigale")
    pk.__path__ = [U + "/pcigale"]
    sys.modules["pcigale"] = pk

    class _NS:
        def __init__(s, **k):
            s.__dict__.update(k)

    def _feii(name):
        c = 3.0e18
        fn, z = {
            "BruhweilerVerner08": ("Fe_d11-m20-20.5.txt", 4593.4 / 4575 - 1),
            "Veron-Cetty04": ("Veron-Cetty_template.txt", 0),
        }[name]
        d = np.genfromtxt(AG + "/FeII_template/" + fn)
        wavez = d[:, 0]
        wave = wavez / (1 + z)
        Llam = d[:, 1] * c / wavez**2
        norm = np.max(Llam[np.argmin(np.abs(wave - 4575))])
        return _NS(wave=wave * 0.1, lumin=Llam / norm)

    def _lines():
        d = np.loadtxt(
            AG + "/mor_netzer_2012/emission_line_table.formatted",
            dtype=[("name", "S10"), ("wave", "f"), ("broad", "f"), ("S2", "f"), ("LINER", "f")],
        )
        return _NS(wave=d["wave"] * 0.1, lumin_BLAGN=d["broad"], lumin_Sy2=d["S2"], lumin_LINER=d["LINER"])

    class Database:
        def __enter__(s):
            return s

        def __exit__(s, *a):
            return False

        def get_ActivateFeII(s, n):
            return _feii(n)

        def get_ActivateMorNetzerEmLines(s):
            return _lines()

    dm = types.ModuleType("pcigale.data")
    dm.Database = Database
    sys.modules["pcigale.data"] = dm

    from pcigale.sed import SED

    def mod(n):
        K = importlib.import_module("pcigale.creation_modules." + n).Module
        return lambda **kw: K(name=n, **kw)

    # Wavelength grid: 5000 log-spaced points 10 nm – 1e6 nm
    grid_nm = np.logspace(np.log10(10.0), np.log10(1e6), 5000)

    # Evaluate
    sed = SED()
    sed.add_contribution("stellar.dummy", grid_nm.copy(), np.zeros_like(grid_nm))
    mod("activate")(fracAGN=-1).process(sed)
    mod("activatelines")(
        AGNtype=params["type"],
        AFeII=params["afeii"],
        Alines=params["alines"],
        linewidth=params["linewidth"],
        ABC=params["abc"],
        FeIItemplate=params.get("feii", "BruhweilerVerner08"),
    ).process(sed)
    mod("activategtorus")(
        fcov=params["fcov"],
        Si=params["si"],
        COOLlam=params["coollam"],
        COOLwidth=params["coolwidth"],
        HOTlam=params["hotlam"],
        HOTwidth=params["hotwidth"],
        HOTfcov=params["hotfcov"],
    ).process(sed)
    mod("activatepl")(
        plslope=params["plslope"],
        plbendloc=params["plbendloc"],
        plbendwidth=params["plbendwidth"],
        uvslope=params["uvslope"],
        cutoff=params["cutoff"],
    ).process(sed)
    mod("activatebol")().process(sed)
    mod("biattenuation")(**{"E(B-V)": params["ebv"], "E(B-V)-AGN": params["ebv_agn"]}).process(sed)

    # Extract contributions on the base grid
    names = list(sed.contribution_names)
    w = sed.wavelength_grid
    L = sed.luminosities

    # Searchsorted to align with base grid
    idx = np.searchsorted(w, grid_nm)
    assert np.allclose(w[idx], grid_nm), "Grid mismatch"

    # Save: wavelength and each contribution
    save_dict = {"wavelength": grid_nm, "params": json.dumps(params)}
    for i, name in enumerate(names):
        if name.startswith("agn.") or name.startswith("attenuation."):
            save_dict[name] = L[i][idx]

    np.savez_compressed(out_npz, **save_dict)


def _driver(upstream_path: str, python_path: str, out_h5: Path) -> None:
    """Driver: spawn workers for each parameter set, build HDF5."""
    upstream_path = str(Path(upstream_path).resolve())
    python_path = str(Path(python_path).resolve())

    # Verify paths
    if not Path(upstream_path).is_dir():
        raise FileNotFoundError(f"Upstream not found: {upstream_path}")
    if not Path(python_path).is_file():
        raise FileNotFoundError(f"Python not found: {python_path}")

    # Get parameter sets
    param_sets = _parameter_sets()
    print(f"Building {len(param_sets)} parameter sets...")

    # Run workers, collect results
    results = {}
    for i, params in enumerate(param_sets):
        pname = params.pop("_name")
        print(f"  [{i + 1}/{len(param_sets)}] {pname}...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / f"{pname}.npz"
            cmd = [
                python_path,
                str(Path(__file__).resolve()),
                "--worker",
                json.dumps(params),
                str(npz_path),
                "--upstream",
                upstream_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"FAILED")
                print(result.stderr)
                raise RuntimeError(f"Worker failed for {pname}")

            # Load npz
            data = np.load(npz_path, allow_pickle=True)
            results[pname] = dict(data)
        print("OK")

    # Write HDF5
    print(f"\nWriting {out_h5}...")
    with h5py.File(out_h5, "w") as f:
        # Root attributes
        f.attrs["upstream_repo"] = "https://github.com/kuntzer/GRAHSP"
        f.attrs["upstream_commit"] = "45054ddf44eef7bb1abb0ab3b54eee5574a28f77"
        f.attrs["upstream_modules"] = (
            "pcigale/creation_modules/activate.py:46-61, activatepl.py:77-107, "
            "activatelines.py:65-97, activategtorus.py:23-84, activatebol.py, "
            "biattenuation.py:34-72"
        )
        f.attrs["wavelength_unit"] = "nm"
        f.attrs["luminosity_unit"] = "L_lambda [1/nm]"
        f.attrs["build_date"] = datetime.utcnow().isoformat()
        f.attrs["build_command"] = " ".join(sys.argv)

        # Get numpy/scipy versions from upstream venv
        versions = subprocess.run(
            [
                python_path,
                "-c",
                "import numpy; import scipy; print(f'{numpy.__version__} {scipy.__version__}')",
            ],
            capture_output=True,
            text=True,
        )
        if versions.returncode == 0:
            vers = versions.stdout.strip().split()
            f.attrs["numpy_version"] = vers[0]
            f.attrs["scipy_version"] = vers[1]

        # Shared wavelength grid
        f.create_dataset("wavelength_nm", data=results[list(results.keys())[0]]["wavelength"])

        # One group per parameter set
        for pname, data in results.items():
            grp = f.create_group(pname)
            params_dict = json.loads(data["params"].item() if hasattr(data["params"], "item") else data["params"])
            for k, v in params_dict.items():
                grp.attrs[f"param_{k}"] = v

            # Store all contributions
            for key in data.keys():
                if key not in ("wavelength", "params"):
                    grp.create_dataset(key, data=data[key], compression="gzip", dtype=np.float64)

    size_mb = out_h5.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_h5} ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--worker", help="JSON parameters (worker mode)")
    ap.add_argument("out_npz", nargs="?", help="Output NPZ file (worker mode only)")
    ap.add_argument("--upstream", required=True, help="Upstream GRAHSP path")
    ap.add_argument("--python", help="Python interpreter in upstream venv (driver mode only)")
    ap.add_argument(
        "--out",
        default=Path(__file__).resolve().parents[1] / "data" / "grahsp_upstream_reference.h5",
        type=Path,
        help="Output HDF5 file (driver mode only)",
    )

    args = ap.parse_args()

    # Determine mode
    if args.worker:
        # Worker mode
        if not args.out_npz:
            raise ValueError("--worker mode requires out_npz positional argument")
        params = json.loads(args.worker)
        _worker(params, Path(args.out_npz), args.upstream)
    else:
        # Driver mode (default)
        if not args.python:
            raise ValueError("Driver mode requires --python argument")
        _driver(args.upstream, args.python, args.out)


if __name__ == "__main__":
    main()
