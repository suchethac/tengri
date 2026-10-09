#!/usr/bin/env python3
"""Package the public Rieke et al. normal-SFG tables as an HDF5 library.

Run from the repository root after placing the 14 public tables directly in
``/tmp/a1_normal_sfg_public``::

    python scripts/build_rieke2009_templates.py --source-dir /tmp/a1_normal_sfg_public

The HDF5 file retains the original table arrays and adds runtime spectra
interpolated onto the first table's wavelength grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

_SOURCE_REPOSITORY = "https://github.com/karlan/AGN_templates"
_SOURCE_DIRECTORY = "host_galaxy/normal_SFGs"
_LOG_L_IR_TEMPLATE = np.arange(9.75, 13.0001, 0.25)
_NEGATIVE_WAVELENGTHS_UM = np.array([5.18203, 5.19398, 5.20595])


def _read_tables(source_dir: Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Read and check the 14 public source tables."""
    tables = []
    headers = []
    filenames = []

    for log_l_ir in _LOG_L_IR_TEMPLATE:
        filename = f"rieke-F{log_l_ir:05.2f}_temp.txt"
        path = source_dir / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        table = np.loadtxt(path, skiprows=1, ndmin=2)
        expected_rows = 3000 if filename == "rieke-F12.50_temp.txt" else 5000
        if table.shape != (expected_rows, 2) or not np.all(np.isfinite(table)):
            raise ValueError(f"Expected {expected_rows} finite two-column rows in {path}.")
        if np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError(f"Wavelengths must increase in {path}.")

        negative = table[:, 1] < 0
        expected_negative = log_l_ir <= 11.00
        if expected_negative:
            if negative.sum() != 3 or not np.array_equal(
                table[negative, 0], _NEGATIVE_WAVELENGTHS_UM
            ):
                raise ValueError(f"Unexpected negative shortwave samples in {path}.")
        elif np.any(negative):
            raise ValueError(f"Unexpected negative flux values in {path}.")

        tables.append(table)
        headers.append(lines[0] if lines else "")
        filenames.append(filename)

    if sum(np.count_nonzero(table[:, 1] < 0) for table in tables) != 18:
        raise ValueError("The public source set must contain exactly 18 negative samples.")
    return tables, headers, filenames


def _write_hdf5(
    path: Path, tables: list[np.ndarray], headers: list[str], filenames: list[str]
) -> None:
    """Write original source rows and the common-grid runtime arrays."""
    wavelength_um = tables[0][:, 0]
    runtime_flux = np.stack(
        [
            np.interp(
                wavelength_um,
                table[:, 0],
                np.maximum(table[:, 1], 0.0),
                left=0.0,
                right=0.0,
            )
            for table in tables
        ]
    )
    if np.any(runtime_flux < 0) or not np.all(np.isfinite(runtime_flux)):
        raise ValueError("Runtime spectra must be finite and nonnegative.")

    temporary = path.with_name(f".{path.name}.tmp")
    with h5py.File(temporary, "w") as h5:
        h5.attrs["source_repository"] = _SOURCE_REPOSITORY
        h5.attrs["source_directory"] = _SOURCE_DIRECTORY
        h5.attrs["retrieved_date"] = "2026-10-08"
        h5.attrs["source_log_l_ir_definition"] = "log10(L_IR / L_sun)"
        h5.attrs["source_flux_units"] = "Not specified in the public table files."
        h5.attrs["runtime_flux_convention"] = (
            "The tabulated second column is used as an F_nu shape and normalized to the "
            "requested L_ir on the model wavelength grid."
        )
        h5.attrs["runtime_negative_sample_treatment"] = (
            "The 18 finite negative shortwave subtraction residuals in the six "
            "lowest-luminosity tables are clipped to zero for runtime interpolation. Original "
            "source rows are retained unchanged under source/."
        )

        h5.create_dataset("log_L_ir_template", data=_LOG_L_IR_TEMPLATE)
        wavelength = h5.create_dataset("wavelength_aa", data=wavelength_um * 1e4)
        wavelength.attrs["units"] = "Angstrom"
        flux = h5.create_dataset("flux_nu_relative", data=runtime_flux)
        flux.attrs["units"] = "relative; the source tables specify no absolute unit"
        flux.attrs["density"] = "F_nu shape"

        source = h5.create_group("source")
        for index, (log_l_ir, table, header, filename) in enumerate(
            zip(_LOG_L_IR_TEMPLATE, tables, headers, filenames, strict=True)
        ):
            group = source.create_group(f"template_{index:02d}")
            group.attrs["log_L_ir_template"] = log_l_ir
            group.attrs["filename"] = filename
            group.attrs["header"] = header
            group.attrs["columns"] = "wavelength_um, unlabeled second column"
            group.create_dataset("wavelength_um", data=table[:, 0])
            group.create_dataset("second_column", data=table[:, 1])

    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/a1_normal_sfg_public"))
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("src/tengri/data/lyu2018/rieke2009.h5"),
    )
    args = parser.parse_args()
    tables, headers, filenames = _read_tables(args.source_dir)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    _write_hdf5(args.output_file, tables, headers, filenames)


if __name__ == "__main__":
    main()
