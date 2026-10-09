#!/usr/bin/env python3
"""Convert the public Lyu et al. AGN and Haro 11 tables to HDF5.

Run from the repository root after placing the public repository files under
``/tmp/lyu2018_public``::

    python scripts/build_lyu2018_templates.py

The script checks every reddened AGN table before writing the two package data
files and a compact source-data audit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # NumPy < 2.0, still supported by pyproject.toml
    from numpy import trapz as _trapezoid  # type: ignore[no-redef]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tengri.utils.physics_constants import C_AA

_SOURCE_REPOSITORY = "https://github.com/karlan/AGN_templates"
_FAMILIES = ("norm", "hdd", "wdd")
_INTRINSIC_COLUMN = {"norm": 1, "hdd": 2, "wdd": 3}
_INDEX_COLUMNS = (
    ("source_id", 0, "Template identifier in the public library."),
    ("tau_v", 1, "Polar-dust optical depth from the public index."),
    ("f_polar_emission_10um", 2, "Polar-dust emission fraction at 10 microns."),
    ("f_polar_scattered_10um", 3, "Scattered fraction at 10 microns."),
    ("f_polar_total_10um", 4, "Total polar contribution at 10 microns."),
)


def _header(path: Path) -> str:
    """Return all leading comment lines from a source table."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines)


def _read_table(path: Path, columns: int) -> np.ndarray:
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.shape[1] != columns or not np.all(np.isfinite(table)):
        raise ValueError(f"Expected finite {columns}-column data in {path}.")
    return table


def _read_agn_families(source_dir: Path) -> tuple[dict[str, dict], dict[str, tuple[float, float]]]:
    """Read and check all 123 public reddened AGN spectra."""
    families: dict[str, dict] = {}
    c0_values: dict[str, tuple[float, float]] = {}

    for family in _FAMILIES:
        index_path = source_dir / "AGN_reddend_lib" / f"obsagn_temp.{family}.index.lis"
        index = _read_table(index_path, 5)
        ids = index[:, 0].astype(int)
        tau_v = index[:, 1]
        if len(ids) != 41 or not np.array_equal(ids, np.arange(41)):
            raise ValueError(f"{index_path} must list template ids 0 through 40.")
        if not np.allclose(tau_v, np.arange(41) * 0.25, rtol=0, atol=1e-6):
            raise ValueError(f"{index_path} must cover tau_V=0..10 in steps of 0.25.")

        wavelength_um = None
        spectra = []
        for template_id, tau in zip(ids, tau_v, strict=True):
            path = source_dir / "AGN_reddend_lib" / family / f"{family}-{template_id}.dat"
            table = _read_table(path, 2)
            if len(table) != 119 or np.any(np.diff(table[:, 0]) <= 0):
                raise ValueError(f"{path} must have 119 increasing wavelength rows.")
            if wavelength_um is None:
                wavelength_um = table[:, 0]
            elif not np.array_equal(table[:, 0], wavelength_um):
                raise ValueError(f"Wavelength rows differ within {family}: {path}.")

            header_tau = re.search(r"tau_V\s*=\s*([0-9.]+)", _header(path))
            if header_tau is None or not np.isclose(float(header_tau.group(1)), tau):
                raise ValueError(f"The tau_V header in {path} does not match its index row.")
            spectra.append(table[:, 1])

        wavelength_aa = wavelength_um * 1e4
        flux = np.asarray(spectra)
        c0 = float(abs(_trapezoid(flux[0], C_AA / wavelength_aa)))

        families[family] = {
            "wavelength_aa": wavelength_aa,
            "tau_v": tau_v,
            "flux_nu_relative": flux,
            "c0_reference": c0,
            "source_index": index,
            "index_header": _header(index_path),
        }
        c0_values[family] = (float(wavelength_um[0]), float(wavelength_um[-1]))

    return families, c0_values


def _write_agn_hdf5(path: Path, families: dict[str, dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with h5py.File(temporary, "w") as h5:
        h5.attrs["source_repository"] = _SOURCE_REPOSITORY
        h5.attrs["retrieved_date"] = "2026-10-07"
        h5.attrs["flux_convention"] = (
            "Published reddened templates stored as relative F_nu; "
            "the .dat files do not state an absolute unit."
        )
        h5.attrs["normalization"] = (
            "One native-frequency integral of each family's tau_V=0 template, "
            "shared by all tau_V rows."
        )

        root = h5.create_group("families")
        for family, values in families.items():
            group = root.create_group(family)
            wave = group.create_dataset("wavelength_aa", data=values["wavelength_aa"])
            wave.attrs["units"] = "Angstrom"
            tau = group.create_dataset("tau_v", data=values["tau_v"])
            tau.attrs["units"] = "dimensionless"
            tau.attrs["meaning"] = "Polar-dust optical depth, not line-of-sight optical depth."
            flux = group.create_dataset("flux_nu_relative", data=values["flux_nu_relative"])
            flux.attrs["density"] = "F_nu"
            flux.attrs["units"] = "relative; no absolute unit is specified in the source table"
            c0 = group.create_dataset("c0_reference", data=values["c0_reference"])
            c0.attrs["units"] = "relative F_nu times Hz"
            c0.attrs["meaning"] = "Absolute native-frequency integral of the tau_V=0 row."

            index_group = group.create_group("source_index")
            index_group.attrs["header"] = values["index_header"]
            for name, column, description in _INDEX_COLUMNS:
                data = values["source_index"][:, column]
                if name == "source_id":
                    data = data.astype(np.int32)
                dataset = index_group.create_dataset(name, data=data)
                dataset.attrs["description"] = description
    temporary.replace(path)


def _write_haro_hdf5(path: Path, source_dir: Path) -> tuple[int, float, float]:
    source_path = source_dir / "host_galaxy" / "lometal_starburst" / "Haro11.temp.full.txt"
    raw = _read_table(source_path, 2)
    wavelength_um, flux = raw.T
    if len(raw) != 5000 or np.any(np.diff(wavelength_um) <= 0):
        raise ValueError(f"{source_path} must contain 5000 increasing wavelength rows.")
    valid = wavelength_um > 5.0
    wave_aa = wavelength_um[valid] * 1e4

    temporary = path.with_name(f".{path.name}.tmp")
    with h5py.File(temporary, "w") as h5:
        h5.attrs["source_repository"] = _SOURCE_REPOSITORY
        h5.attrs["retrieved_date"] = "2026-10-07"
        h5.attrs["valid_wavelength_min_um"] = 5.0
        h5.attrs["valid_wavelength_min_inclusive"] = False
        h5.attrs["valid_wavelength_max_um"] = float(wavelength_um[-1])
        h5.attrs["flux_convention"] = "Jansky-like relative F_nu, as labeled in the public table."

        wave = h5.create_dataset("wavelength_aa", data=wave_aa)
        wave.attrs["units"] = "Angstrom"
        h5.create_dataset("flux_nu_relative", data=flux[valid])
        source = h5.create_group("source")
        source.attrs["header"] = _header(source_path)
        source.attrs["columns"] = "wavelength_um, flux_jansky_like"
        source.create_dataset("wavelength_um", data=wavelength_um)
        source.create_dataset("flux_jansky_like", data=flux)
    temporary.replace(path)

    return int(valid.sum()), float(wavelength_um[valid][0]), float(wavelength_um[-1])


def _intrinsic_ratios(
    source_dir: Path, families: dict[str, dict]
) -> dict[str, tuple[float, float, float]]:
    path = source_dir / "AGN_intrinsic" / "AGN_torus_faceon.temp.full.txt"
    intrinsic = _read_table(path, 4)
    ratios = {}
    for family, values in families.items():
        wave_um = values["wavelength_aa"] / 1e4
        column = _INTRINSIC_COLUMN[family]
        expected = np.interp(wave_um, intrinsic[:, 0], intrinsic[:, column])
        relative = values["flux_nu_relative"][0] / expected
        relative = relative[np.isfinite(relative) & (relative > 0)]
        ratios[family] = (
            float(np.median(relative)),
            float(np.percentile(relative, 5)),
            float(np.percentile(relative, 95)),
        )
    return ratios


def _write_audit(
    path: Path,
    families: dict[str, dict],
    c0_values: dict[str, tuple[float, float]],
    ratios: dict[str, tuple[float, float, float]],
    haro: tuple[int, float, float],
) -> None:
    rows = []
    for family in _FAMILIES:
        values = families[family]
        median, p5, p95 = ratios[family]
        lo, hi = c0_values[family]
        rows.append(
            f"| {family} | {len(values['tau_v'])} | {len(values['wavelength_aa'])} "
            f"| {lo:g}–{hi:g} | {values['c0_reference']:.8g} "
            f"| {median:.6g} ({p5:.6g}–{p95:.6g}) |"
        )
    haro_n, haro_lo, haro_hi = haro
    content = "\n".join(
        (
            "# Public Lyu2018 template data audit",
            "",
            f"Source: {_SOURCE_REPOSITORY}; retrieved 2026-10-07.",
            "All three reddened families contain 41 indexed templates and 119 native wavelength "
            "points per template. The 123 spectra were checked individually for two columns, "
            "finite values, increasing wavelengths, a matching tau_V header, and a shared "
            "wavelength axis within each family.",
            "",
            "| Family | tau rows | wavelength points | wavelength range (μm) | C0 reference | "
            "tau=0/source intrinsic ratio, median (5th–95th percentile) |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The C0 reference is the absolute trapezoidal frequency integral of each family's "
            "tau_V=0 F_nu curve on its native grid; that one value is stored for the family and "
            "used for every tau_V row. The intrinsic-table comparison interpolates the explicitly "
            "Jy-labeled 5000-point source table onto each 119-point tau=0 grid; the ratio spread "
            "is retained as an audit result, not used to rescale the published spectra.",
            "",
            f"Haro 11 source rows: 5000. Runtime rows with λ>5 μm: {haro_n}, "
            f"spanning {haro_lo:g}–{haro_hi:g} μm. The HDF5 retains all source wavelength "
            "and flux rows, including values at λ≤5 μm, while its root datasets contain "
            "only the valid infrared subset.",
            "",
            "The `.dat` flux columns are treated as relative F_nu because their tau=0 shapes "
            "follow the Jy-labeled intrinsic F_nu table by a family-dependent scale. The data "
            "do not provide an absolute Jy calibration for the reddened files.",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/lyu2018_public"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/tengri/data/lyu2018"))
    parser.add_argument(
        "--audit-path", type=Path, default=Path("/tmp/lyu2018_public_data_audit.md")
    )
    args = parser.parse_args()

    families, c0_values = _read_agn_families(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_agn_hdf5(args.output_dir / "lyu2018_agn.h5", families)
    haro = _write_haro_hdf5(args.output_dir / "haro11.h5", args.source_dir)
    ratios = _intrinsic_ratios(args.source_dir, families)
    _write_audit(args.audit_path, families, c0_values, ratios, haro)


if __name__ == "__main__":
    main()
