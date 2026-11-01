#!/usr/bin/env python3
r"""Build data/beagle_workshop_catalog.h5 from the Pacifici+2023 CANDELS workshop catalogs.

Source
------
The CANDELS GOODS-S SED-fitting workshop photometry and the filter curves it uses, from the
MIT-licensed repository https://github.com/camipacifici/art_sedfitting (commit 32d23a2). Pass
a local checkout with ``--art-sedfitting``. The inputs read here are::

    catalogs/CANDELS_GDSS_workshop_z1.dat
    catalogs/CANDELS_GDSS_workshop_z3.dat
    filters/<curve>.dat          (one per photometric band; see COLUMN_TO_CURVE)

No output of any SED-fitting code is read. In particular the BEAGLE summary catalogs in
``code_outputs/`` are not repackaged; the driver reads them at run time from
``$ART_SEDFITTING_DIR`` and they are not committed.

Photometry columns are AB magnitudes with magnitude errors. The sentinels ``98.9x`` (no
measurement) and ``-99`` (no error) are kept as published; the driver masks them.

Filter curves: the two K-band photometry columns (``ISAAC_KS`` and ``HAWKI_KS``) resolve to one
curve, ``HAWKI_KS.dat``, which the upstream notebooks use for the K band. Its response is stored
as published; the upstream fig1 notebook divides it by 100 (its values are below 100, not below
1), and that convention is recorded in the file's attributes. ``ISAAC_KS.dat`` is not stored.

Usage
-----
    python scripts/build_beagle_workshop_catalog.py --art-sedfitting /path/to/art_sedfitting

HDF5 schema (data/beagle_workshop_catalog.h5)
---------------------------------------------
``/catalog_z1`` and ``/catalog_z3``: one dataset per source column, named as in the source
header (``ID``, ``zz``, ``CTIO_U``, ``eCTIO_U``, ..., ``flg1``, ``flg2``, and ``RA``, ``DEC`` in
the z3 catalog). Attributes: ``columns``, ``bands``, ``curve_for_band`` (JSON strings).

``/filters/<curve>``: datasets ``wave_aa`` and ``response``.

Root attributes carry the provenance: source repository, commit, licence, paper DOI, build
date, and the SHA-256 of this builder and of each input file.

References
----------
Pacifici, C., Iyer, K. G., Mobasher, B., et al. 2023, ApJ, 944, 141,
doi:10.3847/1538-4357/acacff ("The Art of Measuring Physical Parameters in Galaxies: A Critical
Assessment of Spectral Energy Distribution Fitting Techniques").
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "beagle_workshop_catalog.h5"
UPSTREAM_REPO = "https://github.com/camipacifici/art_sedfitting"
UPSTREAM_COMMIT = "32d23a2"
UPSTREAM_LICENSE = "MIT"
PAPER_DOI = "10.3847/1538-4357/acacff"
CATALOG_FILES = {
    1: "catalogs/CANDELS_GDSS_workshop_z1.dat",
    3: "catalogs/CANDELS_GDSS_workshop_z3.dat",
}
# Photometric band (catalog column stem) -> filter curve file stem under filters/.
COLUMN_TO_CURVE = {"ISAAC_KS": "HAWKI_KS"}
TRAILING_FLAGS = ("flg1", "flg2")
EXTRA_TRAILING = ("RA", "DEC")
FILTER_COMPRESSION = "gzip"


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkout_commit(root: Path) -> str:
    """Return the short commit of the upstream checkout, or ``"unknown"`` without git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def parse_header(header_line: str) -> tuple[list[str], list[str], list[str]]:
    """Split a workshop catalog header into (all columns, band stems, trailing columns).

    Parameters
    ----------
    header_line : str
        The ``#``-prefixed header line. Band columns come in ``NAME eNAME`` pairs.

    Returns
    -------
    columns : list of str
        Every column name in file order.
    bands : list of str
        Photometric band stems, e.g. ``CTIO_U``, ``ISAAC_KS``.
    trailing : list of str
        Columns after the band pairs (flags, and RA/DEC in the z3 catalog).

    Raises
    ------
    ValueError
        If a band column is not followed by its error column.
    """
    columns = header_line.lstrip("#").split()
    if columns[:2] != ["ID", "zz"]:
        raise ValueError(f"unexpected leading columns: {columns[:2]}")
    bands: list[str] = []
    i = 2
    while i < len(columns) and not columns[i].startswith("flg"):
        name = columns[i]
        if i + 1 >= len(columns) or columns[i + 1] != "e" + name:
            raise ValueError(f"band column {name} is not followed by e{name}")
        bands.append(name)
        i += 2
    return columns, bands, columns[i:]


def read_catalog(path: Path) -> dict[str, np.ndarray]:
    """Read one workshop catalog into a dict of column name -> array.

    Parameters
    ----------
    path : Path
        A ``CANDELS_GDSS_workshop_z*.dat`` file.

    Returns
    -------
    dict
        Column name -> 1-D array. ``ID`` is int64; every other column is float64.
    """
    with open(path) as handle:
        header = handle.readline()
    columns, _bands, _trailing = parse_header(header)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(columns):
        raise ValueError(f"{path.name}: {data.shape[1]} values per row, {len(columns)} columns")
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(columns):
        out[name] = data[:, j].astype(np.int64) if name == "ID" else data[:, j].astype(np.float64)
    return out


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column ``wavelength [Angstrom], response`` filter file."""
    table = np.loadtxt(path, comments="#", ndmin=2)
    return table[:, 0].astype(np.float64), table[:, 1].astype(np.float64)


def _write_catalog(root: h5py.File, z: int, source: Path, art_dir: Path) -> None:
    """Write one ``/catalog_z<z>`` group with one dataset per source column."""
    with open(source) as handle:
        columns, bands, trailing = parse_header(handle.readline())
    cols = read_catalog(source)
    group = root.create_group(f"catalog_z{z}")
    for name in columns:
        group.create_dataset(name, data=cols[name], compression=FILTER_COMPRESSION)
    curve_for_band = {b: COLUMN_TO_CURVE.get(b, b) for b in bands}
    group.attrs["columns"] = json.dumps(columns)
    group.attrs["bands"] = json.dumps(bands)
    group.attrs["trailing_columns"] = json.dumps(trailing)
    group.attrs["curve_for_band"] = json.dumps(curve_for_band)
    group.attrs["n_rows"] = len(cols["ID"])
    group.attrs["source_file"] = source.relative_to(art_dir).as_posix()
    group.attrs["source_sha256"] = _sha256(source)


def _write_filters(root: h5py.File, bands_by_z: dict[int, list[str]], art_dir: Path) -> None:
    """Write one ``/filters/<curve>`` group for every curve any catalog band uses."""
    curves = sorted({COLUMN_TO_CURVE.get(b, b) for bands in bands_by_z.values() for b in bands})
    group = root.create_group("filters")
    for curve in curves:
        path = art_dir / "filters" / f"{curve}.dat"
        wave, response = read_curve(path)
        sub = group.create_group(curve)
        sub.create_dataset("wave_aa", data=wave, compression=FILTER_COMPRESSION)
        sub.create_dataset("response", data=response, compression=FILTER_COMPRESSION)
        sub.attrs["source_file"] = path.relative_to(art_dir).as_posix()
        sub.attrs["source_sha256"] = _sha256(path)
        if curve == "HAWKI_KS":
            sub.attrs["response_note"] = (
                "as published; values below 100, so the upstream fig1 notebook divides by 100"
            )


def build(art_dir: Path, output: Path = OUTPUT_PATH) -> None:
    """Write the repackaged catalog and filter curves to ``output``.

    Parameters
    ----------
    art_dir : Path
        Root of the local art_sedfitting checkout (commit 32d23a2).
    output : Path
        Destination HDF5 file.

    Raises
    ------
    FileNotFoundError
        If a required catalog or filter file is missing from ``art_dir``.
    """
    for rel in CATALOG_FILES.values():
        if not (art_dir / rel).is_file():
            raise FileNotFoundError(f"{art_dir / rel} is missing; check --art-sedfitting")
    commit = _checkout_commit(art_dir)
    bands_by_z: dict[int, list[str]] = {}
    with h5py.File(output, "w") as root:
        for z, rel in CATALOG_FILES.items():
            source = art_dir / rel
            with open(source) as handle:
                _cols, bands, _trail = parse_header(handle.readline())
            bands_by_z[z] = bands
            _write_catalog(root, z, source, art_dir)
        _write_filters(root, bands_by_z, art_dir)
        root.attrs["source_repository"] = UPSTREAM_REPO
        root.attrs["source_commit"] = commit
        root.attrs["source_commit_expected"] = UPSTREAM_COMMIT
        root.attrs["license"] = UPSTREAM_LICENSE
        root.attrs["paper_doi"] = PAPER_DOI
        root.attrs["paper"] = (
            "Pacifici et al. 2023, ApJ 944, 141: The Art of Measuring Physical Parameters in "
            "Galaxies: A Critical Assessment of Spectral Energy Distribution Fitting Techniques"
        )
        root.attrs["build_date"] = datetime.date.today().isoformat()
        root.attrs["builder_sha256"] = _sha256(Path(__file__).resolve())


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--art-sedfitting",
        type=Path,
        required=True,
        help="local checkout of camipacifici/art_sedfitting at commit 32d23a2",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    build(args.art_sedfitting.resolve(), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
