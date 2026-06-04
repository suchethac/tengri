#!/usr/bin/env python3
"""Build ``data/skirtor_mean3p_torus_grid.h5`` from AGNfitter-rX's SKIRTOR_mean_3p.pickle.

The SKIRTOR (Stalevski et al. 2016) clumpy torus library is shipped inside
AGNfitter-rX (branch ``AGNfitter-rX_v0.1``) as an averaged-parameter version
at ``models/TORUS/SKIRTOR_mean_3p.pickle``. This is a **pandas DataFrame**
that differs from tengri's existing full-grid SKIRTOR model: AGNfitter
averaged over the clumpiness (p, q) and radial parameters, keeping only the
three-parameter sub-library:

- ``oa-values``      — half-opening angle [deg].
- ``incl-values``    — inclination angle [deg].
- ``tv-values``      — equatorial optical depth (tau_9.7).
- ``wavelength``     — per-row array of ``log10(nu / Hz)``. (Mislabelled:
  the stored values are log10 frequency, not wavelength.)
- ``SED``            — per-row F_nu template array.

The ``Dm-values`` column is also present but unused (derived, not independent
parameter).

This build script reads the pickle, extracts the 3D grid, and emits a
self-describing HDF5 file for runtime interpolation by
:mod:`tengri.components.agn.skirtor_agnfitter`.

Safety — pickle.load on external data
-------------------------------------
Uses a restricted :class:`pickle.Unpickler` that allow-lists only:

- NumPy array primitives (``ndarray``, ``dtype``, ``multiarray._reconstruct``,
  ``scalar``).
- Pandas container primitives (``DataFrame``, ``Index``, ``_new_Index``,
  ``RangeIndex``, ``BlockManager``).
- ``builtins.slice`` (pandas stores slice objects inside index metadata).
- ``_codecs.encode`` (used to decode byte-string column labels).
- ``functools.partial`` (newer pandas pickles may reference this for block
  reconstruction).

Any other GLOBAL raises ``UnpicklingError``. A preflight opcode dump
verifies the pickle only references the allow-listed set before the
unpickler runs.

HDF5 schema
-----------
``/skirtor_mean3p``

===================  ============================  ============================================
Dataset              Shape                          Description
===================  ============================  ============================================
``oa_axis``          ``(n_oa,)``                    half-opening angle [deg], ascending
``incl_axis``        ``(n_incl,)``                  inclination [deg], ascending
``tv_axis``          ``(n_tv,)``                    optical depth, ascending
``wavelength``       ``(n_wave,)``                  common wavelength grid [Å], ascending
``template``         ``(n_oa, n_incl, n_tv, n_wave)``  F_nu template (unnormalised)
===================  ============================  ============================================

Templates are shape-only; the runtime module
(:mod:`tengri.components.agn.skirtor_agnfitter`) applies per-L_sun
normalisation and scales to ``agn_log_lbol`` + ``agn_torus_frac``.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.

Usage
-----
::

    git clone --depth 1 --branch AGNfitter-rX_v0.1 \\
        https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX
    python scripts/build_skirtor_mean3p_grid.py \\
        --input /tmp/AGNfitter-rX/models/TORUS/SKIRTOR_mean_3p.pickle \\
        --output data/skirtor_mean3p_torus_grid.h5
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
        ("pandas.core.series", "Series"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.numeric", "Int64Index"),
        ("pandas.core.indexes.numeric", "Float64Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.managers", "SingleBlockManager"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.internals.blocks", "new_block"),
        # Builtins
        ("builtins", "slice"),
        ("_codecs", "encode"),
        # functools.partial for newer pandas
        ("functools", "partial"),
    }
)

_C_LIGHT_M_S = 2.99792458e8
_PY2_MODULE_ALIASES: dict[str, str] = {"__builtin__": "builtins"}

# AGNfitter's cosmetic scaling factor for torus (TO) templates.
# Templates in the pickle are stored divided by 1e-40, i.e., multiplied by 1e40.
# When we load and regrid them, we keep this scaling intact for runtime use.
_AGNFITTER_TORUS_SCALE = 1e40

# AGNfitter's cosmetic renormalization for torus (TO) templates.
# Pickle stores F_nu / 1e-40 = F_nu * 1e40, so we divide by 1e40 to undo it.
_AGNFITTER_TORUS_COSMETIC_SCALE = 1e40


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows numpy/pandas primitives."""

    def find_class(self, module: str, name: str):
        module = _PY2_MODULE_ALIASES.get(module, module)
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in safe allow-list. "
                "If this is a legitimate numpy/pandas primitive, add it to "
                "_SAFE_CLASSES in build_skirtor_mean3p_grid.py."
            )
        # Special handling for pandas new_block compat (same as agnfitter_driver.py)
        if (module, name) == ("pandas.core.internals.blocks", "new_block"):
            return _new_block_compat
        return getattr(importlib.import_module(module), name)


def _new_block_compat(values, placement, *args, **kwargs):
    """Version-tolerant pandas.new_block for legacy pickles."""
    from pandas._libs.internals import BlockPlacement
    from pandas.core.internals.blocks import new_block as _nb

    if isinstance(placement, slice):
        placement = BlockPlacement(placement)
    return _nb(values, placement, *args, **kwargs)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Fail loudly if the pickle's opcode stream references anything non-numpy/pandas."""
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
            "Refusing to proceed. Vet each entry, then add legitimate "
            "numpy/pandas primitives to _SAFE_CLASSES."
        )


def _safe_load(pickle_path: Path) -> dict:
    """Safe-unpickle the SKIRTOR_mean_3p DataFrame."""
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        obj = _RestrictedUnpickler(fh, encoding="latin1").load()
    if not hasattr(obj, "shape"):
        raise TypeError(
            f"SKIRTOR_mean_3p pickle root is {type(obj).__name__}, expected pandas.DataFrame."
        )
    return obj


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    """Convert AGNfitter's ``log10(nu / Hz)`` axis to wavelength [Å] (ascending)."""
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    wavelength_m = _C_LIGHT_M_S / nu_hz
    return wavelength_m * 1e10


def _regrid_to_common(
    rows_list: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_wave: int = 4096,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Regrid a list of (log_nu, f_nu) pairs onto a common ascending wavelength grid.

    Parameters
    ----------
    rows_list : list of (ndarray, ndarray)
        Each item is (log_nu, f_nu) arrays for one template.
    n_wave : int
        Target grid size.

    Returns
    -------
    wavelength_common : ndarray, shape (n_wave,)
        Common ascending wavelength grid [Å].
    regridded : list of ndarray
        Each item is the F_nu regridded to wavelength_common.
    """
    # Convert all log_nu to wavelength
    per_wave = [_log_nu_to_wavelength_angstrom(ln) for ln, _ in rows_list]

    # Find common range
    w_min = max(float(w.min()) for w in per_wave)
    w_max = min(float(w.max()) for w in per_wave)
    wavelength_common = np.geomspace(w_min, w_max, n_wave)

    # Regrid each template
    regridded = []
    for (_log_nu, f_nu), w in zip(rows_list, per_wave, strict=True):
        order = np.argsort(w)
        f_nu_regrid = np.interp(
            wavelength_common,
            w[order],
            np.asarray(f_nu, dtype=np.float64)[order],
        )
        regridded.append(f_nu_regrid)

    return wavelength_common, regridded


def build(
    input_pickle: Path,
    output_h5: Path,
    n_wave: int = 1024,
    use_float32: bool = True,
) -> None:
    """Read SKIRTOR_mean_3p.pickle and emit tengri's skirtor_mean3p_torus_grid.h5.

    Parameters
    ----------
    input_pickle : Path
        Path to AGNfitter-rX's SKIRTOR_mean_3p.pickle.
    output_h5 : Path
        Destination HDF5 file.
    n_wave : int
        Target wavelength grid size (default 1024, reduced from 4096 for file size).
    use_float32 : bool
        If True (default), store templates as float32 to reduce file size.
    """
    # Load the DataFrame
    df = _safe_load(input_pickle)

    # Extract unique parameter values
    oa_axis = np.sort(np.asarray(df["oa-values"].unique(), dtype=np.float64))
    incl_axis = np.sort(np.asarray(df["incl-values"].unique(), dtype=np.float64))
    tv_axis = np.sort(np.asarray(df["tv-values"].unique(), dtype=np.float64))

    n_oa = len(oa_axis)
    n_incl = len(incl_axis)
    n_tv = len(tv_axis)

    print(f"Grid dimensions: {n_oa} oa × {n_incl} incl × {n_tv} tv = {n_oa * n_incl * n_tv} cells")
    print(f"Wavelength points: {n_wave} (log-spaced, from 4096 for size reduction)")
    print(f"Template dtype: float{'32' if use_float32 else '64'}")

    # Collect (log_nu, f_nu) pairs for regridding
    rows_list = []
    node_to_idx = {}  # map (oa, incl, tv) -> linear index

    for oa_i, oa in enumerate(oa_axis):
        for incl_i, incl in enumerate(incl_axis):
            for tv_i, tv in enumerate(tv_axis):
                mask = (
                    (df["oa-values"] == oa) & (df["incl-values"] == incl) & (df["tv-values"] == tv)
                )
                row = df[mask]
                if len(row) == 0:
                    raise ValueError(f"Missing grid point: oa={oa}, incl={incl}, tv={tv}")
                if len(row) > 1:
                    raise ValueError(f"Duplicate grid points: oa={oa}, incl={incl}, tv={tv}")

                log_nu = np.asarray(row["wavelength"].values.item(), dtype=np.float64)
                f_nu = np.asarray(row["SED"].values.item(), dtype=np.float64)

                rows_list.append((log_nu, f_nu))
                node_to_idx[(oa_i, incl_i, tv_i)] = len(rows_list) - 1

    # Regrid all templates to a common wavelength.
    wavelength_common, regridded = _regrid_to_common(rows_list, n_wave=n_wave)

    # Reshape into 4D array and apply AGNfitter's cosmetic scaling (multiply by 1e40).
    # This ensures runtime output matches AGNfitter's cosmetically-scaled values.
    dtype_template = np.float32 if use_float32 else np.float64
    template_4d = np.empty((n_oa, n_incl, n_tv, n_wave), dtype=dtype_template)
    for (oa_i, incl_i, tv_i), row_idx in node_to_idx.items():
        template_4d[oa_i, incl_i, tv_i, :] = (regridded[row_idx] * _AGNFITTER_TORUS_SCALE).astype(
            dtype_template
        )

    # Write HDF5
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("skirtor_mean3p")
        g.create_dataset("oa_axis", data=oa_axis, compression="gzip")
        g.create_dataset("incl_axis", data=incl_axis, compression="gzip")
        g.create_dataset("tv_axis", data=tv_axis, compression="gzip")
        g.create_dataset("wavelength", data=wavelength_common, compression="gzip")
        g.create_dataset("template", data=template_4d, compression="gzip")

        g.attrs["source_pickle"] = str(input_pickle)
        g.attrs["n_oa"] = n_oa
        g.attrs["n_incl"] = n_incl
        g.attrs["n_tv"] = n_tv
        g.attrs["n_wave"] = n_wave
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu (relative, per-L_sun normalised at runtime)"
        g.attrs["template_dtype"] = str(dtype_template)
        g.attrs["description"] = (
            "SKIRTOR-averaged (mean over clumpiness p,q and radial index) "
            "— AGNfitter-rX 3-parameter subset. Compressed with gzip."
        )

    file_size_mb = output_h5.stat().st_size / (1024 * 1024)
    print(
        f"wrote {output_h5} — {n_oa}×{n_incl}×{n_tv} grid × {n_wave} "
        f"wavelengths ({file_size_mb:.2f} MB)"
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/TORUS/SKIRTOR_mean_3p.pickle"),
        help="Path to AGNfitter-rX's SKIRTOR_mean_3p.pickle.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/skirtor_mean3p_torus_grid.h5"),
        help="Destination HDF5 path.",
    )
    p.add_argument(
        "--n-wave",
        type=int,
        default=1024,
        help="Size of the common wavelength grid (default 1024, reduced from 4096).",
    )
    p.add_argument(
        "--float64",
        action="store_true",
        help="Store templates as float64 (default: float32 for size reduction).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch SKIRTOR_mean_3p.pickle from the AGNfitter GitHub repo "
        "(pinned tag) instead of reading --input. No AGNfitter install needed.",
    )
    args = p.parse_args()
    from _agnfitter_download import resolve

    input_pickle = resolve(
        args.input, "models/TORUS/SKIRTOR_mean_3p.pickle", download=args.download
    )
    build(input_pickle, args.output, n_wave=args.n_wave, use_float32=not args.float64)


if __name__ == "__main__":
    _cli()
