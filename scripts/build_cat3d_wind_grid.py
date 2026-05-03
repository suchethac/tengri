#!/usr/bin/env python3
"""Build ``data/cat3d_wind_torus_grid.h5`` from AGNfitter-rX's CAT3D pickle.

The CAT3D-Wind (Hönig & Kishimoto 2017) clumpy-disc-plus-polar-wind torus
library is shipped inside AGNfitter-rX (branch ``AGNfitter-rX_v0.1``) at
``models/TORUS/CAT3D_mean_3p.pickle``. That pickle is a **pandas
DataFrame** whose rows enumerate the three-parameter sub-library of the
full Hönig+Kishimoto model:

- ``incl-values`` — inclination angle [deg].
- ``a-values``    — radial cloud-distribution power-law index.
- ``fwd-values``  — polar-wind mass fraction.
- ``wavelength``  — per-row array of ``log10(nu / Hz)``. (Mislabelled:
  the stored values are log10 frequency, not wavelength.)
- ``SED``         — per-row F_nu template array.

AGNfitter slices the DataFrame starting at row 210 for the ``a`` axis
(``CAT3Ddict['a-values'][210:].unique()``) because two disjoint
sub-libraries were concatenated; rows 0..209 belong to a different grid
and must be dropped for the 3-parameter view.  This build script
reproduces that slicing exactly so the h5 output matches what AGNfitter
feeds its fitter.

Safety — pickle.load on external data
-------------------------------------
Uses a restricted :class:`pickle.Unpickler` that allow-lists only:

- NumPy array primitives (``ndarray``, ``dtype``, ``multiarray._reconstruct``,
  ``scalar``).
- Pandas container primitives (``DataFrame``, ``Index``, ``_new_Index``,
  ``RangeIndex``, ``BlockManager``).
- ``builtins.slice`` (pandas stores slice objects inside index metadata).
- ``_codecs.encode`` (used to decode byte-string column labels).

Any other GLOBAL raises ``UnpicklingError``. A preflight opcode dump
verifies the pickle only references the allow-listed set before the
unpickler runs.

HDF5 schema
-----------
``/cat3d_wind``

===================  =====================  ============================================
Dataset              Shape                   Description
===================  =====================  ============================================
``incl_axis``        ``(n_incl,)``           inclination [deg], ascending
``a_axis``           ``(n_a,)``              radial power-law index, ascending
``fwd_axis``         ``(n_fwd,)``            wind fraction, ascending
``wavelength``       ``(n_wave,)``           common wavelength grid [Å], ascending
``template``         ``(n_incl, n_a, n_fwd, n_wave)``  F_nu template (unnormalised)
===================  =====================  ============================================

Templates are shape-only; the runtime module
(:mod:`tengri.components.agn.cat3d_wind`) applies per-L_sun normalisation.

References
----------
- Hönig, S. F. & Kishimoto, M., "The dusty heart of nearby active
  galaxies. II. From clumpy torus models to a unified model," ApJL 838,
  L20 (2017). arXiv:1702.08691. Citation details must be verified against
  the original paper before publication-time references.
- Martínez-Ramírez et al. 2024 / Zhuang et al. 2024, arXiv:2405.12111
  (AGNfitter-rX).

Usage
-----
::

    git clone --branch AGNfitter-rX_v0.1 \\
        https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX
    python scripts/build_cat3d_wind_grid.py \\
        --input /tmp/AGNfitter-rX/models/TORUS/CAT3D_mean_3p.pickle \\
        --output data/cat3d_wind_torus_grid.h5
"""

from __future__ import annotations

import argparse
import importlib
import io
import pickle
import pickletools
from pathlib import Path

import h5py
import numpy as np

_SAFE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        # NumPy primitives
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        # Pandas DataFrame container primitives
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        # Python builtins occurring in pandas indices
        ("builtins", "slice"),
        # Codec used for byte-string column labels
        ("_codecs", "encode"),
    }
)

# Map Python-2 pickle module names to their Python-3 equivalents.
_PY2_MODULE_ALIASES: dict[str, str] = {"__builtin__": "builtins"}

_C_LIGHT_M_S = 2.99792458e8


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler allow-listed to numpy / pandas / basic builtins."""

    def find_class(self, module: str, name: str):
        module = _PY2_MODULE_ALIASES.get(module, module)
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in the safe allow-list."
            )
        return getattr(importlib.import_module(module), name)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Abort if any GLOBAL reference is outside the allow-list."""
    seen: set[tuple[str, str]] = set()
    with pickle_path.open("rb") as fh:
        out = io.StringIO()
        pickletools.dis(fh, annotate=0, out=out)
    for line in out.getvalue().splitlines():
        if "GLOBAL" not in line:
            continue
        try:
            qual = line.split("'", 1)[1].rsplit("'", 1)[0]
        except IndexError:
            continue
        parts = qual.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        mod, name = parts
        mod = _PY2_MODULE_ALIASES.get(mod, mod)
        seen.add((mod, name))
    unexpected = seen - _SAFE_CLASSES
    if unexpected:
        raise RuntimeError(
            f"Unexpected GLOBAL references in {pickle_path}: {sorted(unexpected)}. "
            "Refusing to proceed."
        )


def _safe_load(pickle_path: Path):
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        return _RestrictedUnpickler(fh, encoding="latin1").load()


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    return (_C_LIGHT_M_S / nu_hz) * 1e10


def build(input_pickle: Path, output_h5: Path, n_wave: int = 4096) -> None:
    """Read CAT3D_mean_3p.pickle and emit tengri's ``cat3d_wind_torus_grid.h5``."""
    df = _safe_load(input_pickle)
    for col in ("incl-values", "a-values", "fwd-values", "wavelength", "SED"):
        if col not in df.columns:
            raise KeyError(f"CAT3D pickle DataFrame missing column '{col}'.")

    # Mirror AGNfitter's exact slicing: the `a` axis is drawn from the
    # second sub-library starting at row index 210 (see
    # MODEL_AGNfitter.TORUS CAT3D_3P branch).
    if len(df) < 210:
        raise RuntimeError(
            f"CAT3D DataFrame has only {len(df)} rows; AGNfitter's [210:] "
            "slice cannot be reproduced. The upstream pickle format may have changed."
        )
    incl_axis = np.sort(np.asarray(df["incl-values"].unique(), dtype=np.float64))
    a_axis = np.sort(np.asarray(df["a-values"][210:].unique(), dtype=np.float64))
    fwd_axis = np.sort(np.asarray(df["fwd-values"].unique(), dtype=np.float64))

    # Build a per-row lookup keyed on the exact float triple (incl, a, fwd).
    by_key: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for _, row in df.iloc[210:].iterrows():
        key = (
            float(row["incl-values"]),
            float(row["a-values"]),
            float(row["fwd-values"]),
        )
        by_key[key] = (
            np.asarray(row["wavelength"], dtype=np.float64).ravel(),
            np.asarray(row["SED"], dtype=np.float64).ravel(),
        )

    # Determine a common wavelength grid from the intersection of per-row extents.
    all_waves = [_log_nu_to_wavelength_angstrom(ln) for ln, _ in by_key.values()]
    w_min = max(float(w.min()) for w in all_waves)
    w_max = min(float(w.max()) for w in all_waves)
    common_wave = np.geomspace(w_min, w_max, n_wave)

    template = np.zeros((incl_axis.size, a_axis.size, fwd_axis.size, n_wave), dtype=np.float64)
    populated = np.zeros((incl_axis.size, a_axis.size, fwd_axis.size), dtype=bool)
    for i, incl in enumerate(incl_axis):
        for j, a in enumerate(a_axis):
            for k, fwd in enumerate(fwd_axis):
                row = by_key.get((float(incl), float(a), float(fwd)))
                if row is None:
                    continue
                log_nu, fnu = row
                wave = _log_nu_to_wavelength_angstrom(log_nu)
                order = np.argsort(wave)
                template[i, j, k] = np.interp(common_wave, wave[order], fnu[order])
                populated[i, j, k] = True

    # AGNfitter's CAT3D library is not a full Cartesian product of the
    # three axes.  Tengri's triweight interpolation would smear zero
    # templates into neighbouring cells, producing unphysical SEDs at
    # intermediate parameter values.  Fill missing cells with the
    # nearest-neighbour populated cell in axis-index space — this
    # matches AGNfitter's own nearest-neighbour runtime lookup at every
    # queried populated cell, and produces a smoothly-interpolable grid
    # for tengri's gradient-based inference.
    missing_count = int((~populated).sum())
    if missing_count:
        populated_idx = np.array(np.nonzero(populated)).T  # (n_pop, 3)
        for i, j, k in np.array(np.nonzero(~populated)).T:
            d = np.abs(populated_idx - np.array([i, j, k])).sum(axis=1)
            nn = populated_idx[int(np.argmin(d))]
            template[i, j, k] = template[nn[0], nn[1], nn[2]]

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("cat3d_wind")
        g.create_dataset("incl_axis", data=incl_axis, compression="gzip")
        g.create_dataset("a_axis", data=a_axis, compression="gzip")
        g.create_dataset("fwd_axis", data=fwd_axis, compression="gzip")
        g.create_dataset("wavelength", data=common_wave, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        g.attrs["source_pickle"] = str(input_pickle)
        g.attrs["n_incl"] = incl_axis.size
        g.attrs["n_a"] = a_axis.size
        g.attrs["n_fwd"] = fwd_axis.size
        g.attrs["n_wave"] = n_wave
        g.attrs["missing_grid_points"] = missing_count
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu (relative, per-L_sun normalised at runtime)"

    filled = incl_axis.size * a_axis.size * fwd_axis.size - missing_count
    print(
        f"wrote {output_h5} — {incl_axis.size} incl × {a_axis.size} a × "
        f"{fwd_axis.size} fwd × {n_wave} wavelengths "
        f"({filled} of {incl_axis.size * a_axis.size * fwd_axis.size} grid points populated)"
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/TORUS/CAT3D_mean_3p.pickle"),
        help="Path to AGNfitter-rX's CAT3D_mean_3p.pickle.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/cat3d_wind_torus_grid.h5"),
        help="Destination HDF5 path.",
    )
    p.add_argument(
        "--n-wave",
        type=int,
        default=4096,
        help="Size of the common wavelength grid after regridding.",
    )
    args = p.parse_args()
    build(args.input, args.output, n_wave=args.n_wave)


if __name__ == "__main__":
    _cli()
