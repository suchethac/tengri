#!/usr/bin/env python3
"""Build tengri's remaining AGNfitter-rX torus-library reductions as HDF5.

AGNfitter-rX ships several averaged-parameter torus sub-libraries in
``models/TORUS/``. tengri already vendors the AGNfitter-faithful defaults of
each family (``NK0_mean_1p``, ``SKIRTOR_mean_3p``, the high-``fwd`` half of
``CAT3D_mean_3p``) via ``scripts/build_nk08_agnfitter_grid.py``,
``scripts/build_skirtor_mean3p_grid.py``, and
``scripts/build_cat3d_wind_grid.py`` respectively. This script fills in the
remaining reductions each family's upstream pickle also carries:

- NK0 2-parameter: ``NK0_mean_2p.pickle`` ->
  ``nenkova_agnfitter_2p_torus_grid.h5`` (group ``nenkova_agnfitter_2p``)
- NK0 3-parameter: ``NK0_mean_3p.pickle`` ->
  ``nenkova_agnfitter_3p_torus_grid.h5`` (group ``nenkova_agnfitter_3p``)
- SKIRTOR 1-parameter: ``SKIRTOR_mean_1p.pickle`` ->
  ``skirtor_mean1p_torus_grid.h5`` (group ``skirtor_mean1p``)
- SKIRTOR 2-parameter: ``SKIRTOR_mean_2p.pickle`` ->
  ``skirtor_mean2p_torus_grid.h5`` (group ``skirtor_mean2p``)
- CAT3D low-fwd: ``CAT3D_mean_3p.pickle`` (rows 0-209) ->
  ``cat3d_wind_lowfwd_torus_grid.h5`` (group ``cat3d_wind_lowfwd``)

Native resolution, no resampling
---------------------------------
Unlike ``build_nk08_agnfitter_grid.py`` (whose ``NK0_mean_1p`` dict stores a
*different* ``log10(nu/Hz)`` axis per inclination bin, requiring a regrid
onto a common wavelength axis), every pickle this script reads stores ONE
``log10(nu/Hz)`` axis shared by every row -- verified per file before
building. There is therefore nothing to resample: each grid is written at
the upstream's native wavelength resolution and in float64, with every axis
stored explicitly and a ``source_sha256`` attribute of the upstream pickle
recorded for provenance.

Safety -- pickle.load on external data
---------------------------------------
Uses a restricted :class:`pickle.Unpickler` that allow-lists only numpy and
pandas container primitives (the same allow-list as
``scripts/build_skirtor_mean3p_grid.py``, since these pickles are all
pandas DataFrames pickled by the same AGNfitter-rX code path). Any other
``GLOBAL`` opcode raises ``UnpicklingError``, and a preflight opcode scan
verifies this before the unpickler runs.

HDF5 schema (per group)
------------------------
One dataset ``<axis>_axis`` per grid axis (ascending, native units -- degrees
for inclination/opening angle, dimensionless for optical depth / CAT3D's
``a``/``fwd``), plus ``wavelength`` (Å, ascending) and ``template`` (shape
``axis_1 x ... x axis_k x n_wave``, unnormalized ``F_nu``). Templates are
*shape-only*; each runtime module (``tengri.components.agn.nenkova_agnfitter_2p``,
etc.) applies per-L_sun normalization at evaluation time.

References
----------
.. [1] M. Nenkova et al., "AGN Dusty Tori. II. Observational Implications of
   Clumpiness," ApJ 685, 160 (2008). arXiv:0806.0512. bibcode:2008ApJ...685..160N.
.. [2] M. Stalevski et al., MNRAS, 458, 2288 (2016). arXiv:1602.06954.
.. [3] S. F. Hönig & M. Kishimoto, ApJL 838, L20 (2017). arXiv:1702.08691.
.. [4] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.

Usage
-----
::

    python scripts/build_agnfitter_torus_reductions.py \\
        --torus-dir /tmp/AGNfitter-rX/models/TORUS
    git add -f data/nenkova_agnfitter_2p_torus_grid.h5 \\
               data/nenkova_agnfitter_3p_torus_grid.h5 \\
               data/skirtor_mean1p_torus_grid.h5 \\
               data/skirtor_mean2p_torus_grid.h5 \\
               data/cat3d_wind_lowfwd_torus_grid.h5
"""

from __future__ import annotations

import argparse
import hashlib
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

_PY2_MODULE_ALIASES: dict[str, str] = {"__builtin__": "builtins"}
_C_LIGHT_M_S = 2.99792458e8


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows numpy/pandas primitives."""

    def find_class(self, module: str, name: str):
        module = _PY2_MODULE_ALIASES.get(module, module)
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in safe allow-list. "
                "If this is a legitimate numpy/pandas primitive, add it to "
                "_SAFE_CLASSES in build_agnfitter_torus_reductions.py."
            )
        if (module, name) == ("pandas.core.internals.blocks", "new_block"):
            return _new_block_compat
        return getattr(importlib.import_module(module), name)


def _new_block_compat(values, placement, *args, **kwargs):
    """Version-tolerant ``pandas.new_block`` for legacy pickles."""
    from pandas._libs.internals import BlockPlacement
    from pandas.core.internals.blocks import new_block as _nb

    if isinstance(placement, slice):
        placement = BlockPlacement(placement)
    return _nb(values, placement, *args, **kwargs)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Fail loudly if the pickle's opcode stream references anything unexpected."""
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


def _safe_load(pickle_path: Path):
    """Safe-unpickle an AGNfitter-rX torus DataFrame."""
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        obj = _RestrictedUnpickler(fh, encoding="latin1").load()
    if not hasattr(obj, "shape"):
        raise TypeError(f"{pickle_path} root is {type(obj).__name__}, expected pandas.DataFrame.")
    return obj


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    """Convert AGNfitter's ``log10(nu / Hz)`` axis to wavelength [Å] (descending)."""
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    wavelength_m = _C_LIGHT_M_S / nu_hz
    return wavelength_m * 1e10


def _sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (upstream-pickle provenance)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_native_grid(
    df,
    axis_cols: list[str],
    output_h5: Path,
    group_name: str,
    source_pickle: Path,
    source_label: str,
    *,
    row_slice: slice | None = None,
) -> None:
    """Reshape a per-row AGNfitter-rX torus DataFrame into a native-resolution grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Loaded (via :func:`_safe_load`) upstream torus DataFrame, columns
        ``"wavelength"`` (per-row ``log10(nu/Hz)``), ``"SED"`` (per-row
        ``F_nu``), and one column per entry of ``axis_cols``.
    axis_cols : list of str
        DataFrame columns to treat as grid axes, in the desired output
        axis order (need not match the DataFrame's own column order).
    output_h5 : Path
        Destination HDF5 file (one group, named ``group_name``).
    group_name : str
        HDF5 group name; also the physical-model label recorded in
        ``source_pickle``'s bookkeeping attribute.
    source_pickle : Path
        Path to the upstream pickle, for provenance attributes.
    source_label : str
        Provenance string for the ``source_pickle`` attribute: the path
        INSIDE the pinned upstream archive (``AGNfitter-rX_v0.1/models/...``).
        Kept separate from ``source_pickle`` because that one is wherever this
        machine holds the file, which must never reach a committed grid.
    row_slice : slice, optional
        If given, ``df = df.iloc[row_slice]`` before building (AGNfitter-rX
        concatenates disjoint sub-libraries into one pickle for CAT3D; see
        ``cat3d_wind_lowfwd``, rows 0-209 of ``CAT3D_mean_3p.pickle``).

    Raises
    ------
    ValueError
        If the (possibly sliced) DataFrame is not a full Cartesian product
        of the ``axis_cols`` unique values, or if its rows do not share one
        common wavelength axis (the assumption that makes native-resolution
        vendoring exact rather than a resample).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy/h5py.
    """
    if row_slice is not None:
        df = df.iloc[row_slice]
    df = df.reset_index(drop=True)

    axis_arrays = [np.sort(np.asarray(df[c].unique(), dtype=np.float64)) for c in axis_cols]
    shape = tuple(int(a.size) for a in axis_arrays)
    n_expected = int(np.prod(shape))
    if len(df) != n_expected:
        raise ValueError(
            f"{source_pickle} ({group_name}): {len(df)} rows but the cartesian "
            f"product of {axis_cols} axes is {n_expected} -- not a full "
            "rectangular grid."
        )

    # Native-resolution vendoring assumes every row shares one wavelength
    # axis (verified, not merely assumed): only then is reshaping into an
    # N-D array exact, with nothing resampled or interpolated away.
    first_log_nu = np.asarray(df["wavelength"].iloc[0], dtype=np.float64).ravel()
    for log_nu in df["wavelength"]:
        if not np.array_equal(np.asarray(log_nu, dtype=np.float64).ravel(), first_log_nu):
            raise ValueError(
                f"{source_pickle} ({group_name}): rows do not share one common "
                "wavelength grid; native-resolution vendoring requires it "
                "(fall back to a regridding builder like build_skirtor_mean3p_grid.py "
                "if this ever changes upstream)."
            )
    wave_desc_aa = _log_nu_to_wavelength_angstrom(first_log_nu)
    order = np.argsort(wave_desc_aa)  # log_nu ascending -> wavelength descending
    wavelength_aa = wave_desc_aa[order]
    n_wave = int(wavelength_aa.size)

    key_cols = [df[c].to_numpy(dtype=np.float64) for c in axis_cols]
    row_by_key: dict[tuple[float, ...], int] = {}
    for i in range(len(df)):
        key = tuple(float(col[i]) for col in key_cols)
        if key in row_by_key:
            raise ValueError(f"{source_pickle} ({group_name}): duplicate grid point {key}")
        row_by_key[key] = i

    sed_values = df["SED"].to_numpy()
    template = np.empty((*shape, n_wave), dtype=np.float64)
    for idx in np.ndindex(*shape):
        key = tuple(float(axis_arrays[d][idx[d]]) for d in range(len(axis_cols)))
        row_i = row_by_key.get(key)
        if row_i is None:
            raise ValueError(f"{source_pickle} ({group_name}): missing grid point {key}")
        sed = np.asarray(sed_values[row_i], dtype=np.float64).ravel()
        template[idx] = sed[order]

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group(group_name)
        for c, arr in zip(axis_cols, axis_arrays, strict=True):
            g.create_dataset(f"{c.replace('-values', '')}_axis", data=arr, compression="gzip")
        g.create_dataset("wavelength", data=wavelength_aa, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        # The path INSIDE the pinned upstream archive, never where this
        # machine happened to keep it: an absolute path here ships a
        # contributor's home directory to every user of the public
        # repository (tools/check_no_local_paths.py). ``source_sha256``
        # below is what pins the actual bytes.
        g.attrs["source_pickle"] = source_label
        g.attrs["source_sha256"] = _sha256_of(source_pickle)
        for c, arr in zip(axis_cols, axis_arrays, strict=True):
            g.attrs[f"n_{c.replace('-values', '')}"] = int(arr.size)
        g.attrs["n_wave"] = n_wave
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu (relative, per-L_sun normalized at runtime)"
        g.attrs["resolution"] = (
            "native (no resampling); every upstream row shares the same wavelength grid"
        )

    size_kb = output_h5.stat().st_size / 1024
    print(
        f"wrote {output_h5} -- group {group_name!r}: shape {(*shape, n_wave)} ({size_kb:.1f} KB)"
    )


def build_nk08_2p(input_pickle: Path, output_h5: Path, source_label: str) -> None:
    """Build ``nenkova_agnfitter_2p_torus_grid.h5`` from ``NK0_mean_2p.pickle``."""
    df = _safe_load(input_pickle)
    _build_native_grid(
        df,
        ["incl-values", "oa-values"],
        output_h5,
        "nenkova_agnfitter_2p",
        input_pickle,
        source_label,
    )


def build_nk08_3p(input_pickle: Path, output_h5: Path, source_label: str) -> None:
    """Build ``nenkova_agnfitter_3p_torus_grid.h5`` from ``NK0_mean_3p.pickle``."""
    df = _safe_load(input_pickle)
    _build_native_grid(
        df,
        ["incl-values", "oa-values", "tv-values"],
        output_h5,
        "nenkova_agnfitter_3p",
        input_pickle,
        source_label,
    )


def build_skirtor_mean1p(input_pickle: Path, output_h5: Path, source_label: str) -> None:
    """Build ``skirtor_mean1p_torus_grid.h5`` from ``SKIRTOR_mean_1p.pickle``."""
    df = _safe_load(input_pickle)
    _build_native_grid(
        df, ["incl-values"], output_h5, "skirtor_mean1p", input_pickle, source_label
    )


def build_skirtor_mean2p(input_pickle: Path, output_h5: Path, source_label: str) -> None:
    """Build ``skirtor_mean2p_torus_grid.h5`` from ``SKIRTOR_mean_2p.pickle``."""
    df = _safe_load(input_pickle)
    _build_native_grid(
        df, ["oa-values", "incl-values"], output_h5, "skirtor_mean2p", input_pickle, source_label
    )


def build_cat3d_lowfwd(input_pickle: Path, output_h5: Path, source_label: str) -> None:
    """Build ``cat3d_wind_lowfwd_torus_grid.h5`` from rows 0-209 of ``CAT3D_mean_3p.pickle``.

    Mirrors ``scripts/build_cat3d_wind_grid.py``'s row-210+ slice for the
    high-``fwd`` sub-library, but for the complementary low-``fwd``
    sub-library at the other end of the concatenated pickle -- a full
    Cartesian product on its own, so (unlike the high-``fwd`` half) no
    nearest-neighbor cell filling is required.
    """
    df = _safe_load(input_pickle)
    if len(df) < 210:
        raise RuntimeError(
            f"CAT3D DataFrame has only {len(df)} rows; the [0:210] low-fwd "
            "slice cannot be reproduced. The upstream pickle format may have changed."
        )
    _build_native_grid(
        df,
        ["incl-values", "a-values", "fwd-values"],
        output_h5,
        "cat3d_wind_lowfwd",
        input_pickle,
        source_label,
        row_slice=slice(0, 210),
    )


_SPECS: tuple[tuple[str, str, str, object], ...] = (
    ("nk08_2p", "NK0_mean_2p.pickle", "nenkova_agnfitter_2p_torus_grid.h5", build_nk08_2p),
    ("nk08_3p", "NK0_mean_3p.pickle", "nenkova_agnfitter_3p_torus_grid.h5", build_nk08_3p),
    (
        "skirtor_mean1p",
        "SKIRTOR_mean_1p.pickle",
        "skirtor_mean1p_torus_grid.h5",
        build_skirtor_mean1p,
    ),
    (
        "skirtor_mean2p",
        "SKIRTOR_mean_2p.pickle",
        "skirtor_mean2p_torus_grid.h5",
        build_skirtor_mean2p,
    ),
    (
        "cat3d_lowfwd",
        "CAT3D_mean_3p.pickle",
        "cat3d_wind_lowfwd_torus_grid.h5",
        build_cat3d_lowfwd,
    ),
)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--which",
        choices=[spec[0] for spec in _SPECS] + ["all"],
        default="all",
        help="Which reduction to build (default: all five).",
    )
    p.add_argument(
        "--torus-dir",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/TORUS"),
        help="Directory holding the upstream AGNfitter-rX TORUS pickles.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Destination directory for the HDF5 grids.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch each pickle from the AGNfitter-rX GitHub repo (pinned tag) "
        "instead of reading --torus-dir. No AGNfitter install needed.",
    )
    args = p.parse_args()
    from _agnfitter_download import archive_relpath, resolve

    for name, pickle_name, out_name, fn in _SPECS:
        if args.which != "all" and args.which != name:
            continue
        repo_relpath = f"models/TORUS/{pickle_name}"
        input_pickle = resolve(args.torus_dir / pickle_name, repo_relpath, download=args.download)
        fn(input_pickle, args.output_dir / out_name, archive_relpath(repo_relpath))


if __name__ == "__main__":
    _cli()
