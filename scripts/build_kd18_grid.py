#!/usr/bin/env python3
"""Build ``data/kd18_agnfitter*_disc_grid.h5`` from AGNfitter-rX's KD18 pickles.

The Kubota & Done (2018) three-zone accretion-disc grid ships inside
AGNfitter-rX (branch ``AGNfitter-rX_v0.1``) as two pickles under
``models/BBB/``:

- ``KD18.pickle`` -- a pandas ``DataFrame`` with one row per
  ``(logBHmass, logEddra)`` node (225 rows: 15 x 15), each row holding its own
  ``wavelength`` (``log10(nu/Hz)``) and ``SED`` (:math:`F_\\nu`, relative
  units) arrays of length 100. The template already carries the hot corona
  (extends to :math:`\\lambda \\approx 0.06` A, hard X-ray): AGNfitter-rX's
  ``KD18`` branch never calls ``add_xrays`` (see ``functions/MODEL_AGNfitter.py``,
  ``PLOTandWRITE_AGNfitter.py`` line 700).
- ``KD18_warmInd.pickle`` -- the same layout with an added ``warmIndex`` axis
  (10 values, 2250 rows: 15 x 15 x 10). ``warmIndex`` is the spectral index of
  the warm Comptonization zone (Kubota & Done 2018 section 2.2; identified as
  a candidate extra free parameter in Martinez-Ramirez et al. 2024 section 5.2
  -- "the spectral index of the warm Comptonization component ... can improve
  the performance"), not exercised by AGNfitter-rX's example configuration.

A one-time measurement compared ``KD18_warmInd`` against plain ``KD18`` at
each of the 10 candidate ``warmIndex`` values, matched node-by-node on
``(logBHmass, logEddra)`` (pinned as a regression in
``tests/crossval/test_kd18_grid_vs_agnfitter.py``): the best match
(``warmIndex=2.611``) still disagrees by up to a factor of
:math:`10^{0.103} \\approx 1.27` at some wavelength, and the worst
(``warmIndex=1.5``) by up to :math:`10^{1.93} \\approx 85\\times`. The two
pickles are NOT interchangeable at any single ``warmIndex`` value, so tengri
vendors **two** grids and registers **two** disc blocks: ``kd18_agnfitter``
(this pickle) and ``kd18_agnfitter_warmindex`` (the 3-axis one).

Both wavelength axes are identical across every node within their own pickle
(verified in the same probe), so building requires no per-node resampling --
just a unit conversion (``log10(nu/Hz)`` -> Angstrom, ascending) and a stack.
Native resolution is kept throughout (100 wavelength points, float64; no
down/up-sampling).

Safety -- pickle.load on external data
---------------------------------------
``KD18.pickle`` / ``KD18_warmInd.pickle`` are pandas ``DataFrame`` pickles (not
plain numpy dicts, unlike SN12), so the allow-list needs pandas' internal
container classes -- the same set ``scripts/build_agnfitter_bbb_reference.py``
uses -- plus a preflight opcode scan (``scripts/build_slone_netzer_grid.py``'s
pattern) that aborts before unpickling if any ``GLOBAL`` reference falls
outside the allow-list. Unlike the reference builder, there is no
``pandas.read_pickle`` fallback: these pickles are untrusted data and are read
ONLY through the restricted unpickler.

HDF5 schema
-----------
``data/kd18_agnfitter_disc_grid.h5`` -- ``/kd18_agnfitter``:

================  ========================  ===============================
Dataset           Shape                     Description
================  ========================  ===============================
``log_mbh``       ``(15,)``                 log10(M_BH / M_sun), ascending
``log_edd``       ``(15,)``                 log10(Mdot / Mdot_Edd), ascending
``wavelength``    ``(100,)``                native wavelength [A], ascending
``template``      ``(15, 15, 100)``         F_nu template (unnormalized)
================  ========================  ===============================

``data/kd18_agnfitter_warmindex_disc_grid.h5`` -- ``/kd18_agnfitter_warmindex``:
adds ``gamma_warm`` ``(10,)`` and ``template`` is ``(15, 15, 10, 100)``.

Both groups carry a ``source_sha256`` attribute (SHA-256 of the exact input
pickle bytes) so the vendored HDF5 can be checked against a specific upstream
download.

References
----------
- Kubota, A. & Done, C., "A physical interpretation of the hard X-ray excess
  in low-luminosity AGN," MNRAS 480, 1247 (2018). arXiv:1804.02334.
- Martinez-Ramirez et al. 2024, A&A 688, A46 (AGNfitter-rX).

Usage
-----
::

    python scripts/build_kd18_grid.py \\
        --input /tmp/AGNfitter-rX/models/BBB/KD18.pickle \\
        --input-warmindex /tmp/AGNfitter-rX/models/BBB/KD18_warmInd.pickle
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import pickle
import pickletools
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

#: Allow-list: numpy/pandas container classes needed to round-trip AGNfitter's
#: legacy-pandas DataFrame pickles. Mirrors
#: ``scripts/build_agnfitter_bbb_reference.py``'s ``_SAFE_CLASSES`` (verified
#: sufficient for KD18.pickle / KD18_warmInd.pickle by the preflight scan
#: below), minus ``functools.partial`` (THB21-only, not present in these two).
_SAFE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.series", "Series"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.blocks", "Block"),
        ("pandas.core.arrays.numpy_", "PandasArray"),
        ("__builtin__", "slice"),
        ("builtins", "slice"),
        ("_codecs", "encode"),
    }
)
_PY2_MODULE_ALIASES: dict[str, str] = {"__builtin__": "builtins"}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler allow-listed to numpy / pandas container classes."""

    def find_class(self, module: str, name: str):
        module = _PY2_MODULE_ALIASES.get(module, module)
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in the safe allow-list."
            )
        return getattr(importlib.import_module(module), name)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Abort if any GLOBAL reference in the pickle is outside the allow-list."""
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
    """Load one KD18 pickle, untrusted-data-safe: scan then restricted-unpickle."""
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        return _RestrictedUnpickler(fh, encoding="latin1").load()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log_nu_to_aa(log_nu_hz: np.ndarray) -> np.ndarray:
    """log10(nu/Hz) -> wavelength [Angstrom]."""
    return _C_AA_PER_S / (10.0 ** np.asarray(log_nu_hz, dtype=np.float64))


def _node_wavelength_aa(df) -> np.ndarray:
    """The single common wavelength axis every row of ``df`` shares.

    Raises
    ------
    ValueError
        If the rows do not in fact share one wavelength axis -- the whole
        no-resampling design (native resolution) depends on this.
    """
    first = np.asarray(df["wavelength"].values[0], dtype=np.float64)
    for row in df["wavelength"].to_numpy()[1:]:
        if not np.array_equal(np.asarray(row, dtype=np.float64), first):
            raise ValueError(
                "KD18 rows do not share one wavelength axis; the native-resolution "
                "no-resampling assumption does not hold for this pickle."
            )
    # Wavelength is inversely proportional to frequency, so sorting the
    # *pre-conversion* log10(nu/Hz) axis (ascending) does not sort the
    # post-conversion Angstrom axis (which is then descending) -- argsort
    # must run on the converted wavelength values themselves.
    wave_all = _log_nu_to_aa(first)
    order = np.argsort(wave_all)
    return wave_all[order], order


def build_plain(input_pickle: Path, output_h5: Path) -> None:
    """Read ``KD18.pickle`` and emit ``kd18_agnfitter_disc_grid.h5``."""
    df = _safe_load(input_pickle)
    for col in ("wavelength", "SED", "logBHmass", "logEddra"):
        if col not in df.columns:
            raise KeyError(f"KD18 pickle missing column '{col}'.")

    wave_aa, order = _node_wavelength_aa(df)
    log_mbh = np.sort(np.asarray(df["logBHmass"].unique(), dtype=np.float64))
    log_edd = np.sort(np.asarray(df["logEddra"].unique(), dtype=np.float64))

    template = np.zeros((log_mbh.size, log_edd.size, wave_aa.size), dtype=np.float64)
    for i, mbh in enumerate(log_mbh):
        for j, edd in enumerate(log_edd):
            row = df[(df["logBHmass"] == mbh) & (df["logEddra"] == edd)]
            if row.empty:
                raise KeyError(f"KD18 pickle missing node (logBHmass={mbh}, logEddra={edd}).")
            sed = np.asarray(row["SED"].values[0], dtype=np.float64)
            template[i, j] = sed[order]

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("kd18_agnfitter")
        g.create_dataset("log_mbh", data=log_mbh, compression="gzip")
        g.create_dataset("log_edd", data=log_edd, compression="gzip")
        g.create_dataset("wavelength", data=wave_aa, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        g.attrs["source_pickle"] = input_pickle.name
        g.attrs["source_sha256"] = _sha256(input_pickle)
        g.attrs["n_mbh"] = log_mbh.size
        g.attrs["n_edd"] = log_edd.size
        g.attrs["n_wave"] = wave_aa.size
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu erg/s/Hz (renormalized at runtime)"

    print(
        f"wrote {output_h5} — {log_mbh.size} M_BH x {log_edd.size} Edd x "
        f"{wave_aa.size} wavelengths; log_mbh [{log_mbh.min():.2f}, {log_mbh.max():.2f}], "
        f"log_edd [{log_edd.min():.2f}, {log_edd.max():.2f}]"
    )


def build_warmindex(input_pickle: Path, output_h5: Path) -> None:
    """Read ``KD18_warmInd.pickle`` and emit ``kd18_agnfitter_warmindex_disc_grid.h5``."""
    df = _safe_load(input_pickle)
    for col in ("wavelength", "SED", "logBHmass", "logEddra", "warmIndex"):
        if col not in df.columns:
            raise KeyError(f"KD18_warmInd pickle missing column '{col}'.")

    wave_aa, order = _node_wavelength_aa(df)
    log_mbh = np.sort(np.asarray(df["logBHmass"].unique(), dtype=np.float64))
    log_edd = np.sort(np.asarray(df["logEddra"].unique(), dtype=np.float64))
    gamma_warm = np.sort(np.asarray(df["warmIndex"].unique(), dtype=np.float64))

    template = np.zeros(
        (log_mbh.size, log_edd.size, gamma_warm.size, wave_aa.size), dtype=np.float64
    )
    for i, mbh in enumerate(log_mbh):
        for j, edd in enumerate(log_edd):
            for k, gw in enumerate(gamma_warm):
                row = df[
                    (df["logBHmass"] == mbh) & (df["logEddra"] == edd) & (df["warmIndex"] == gw)
                ]
                if row.empty:
                    raise KeyError(
                        "KD18_warmInd pickle missing node "
                        f"(logBHmass={mbh}, logEddra={edd}, warmIndex={gw})."
                    )
                sed = np.asarray(row["SED"].values[0], dtype=np.float64)
                template[i, j, k] = sed[order]

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("kd18_agnfitter_warmindex")
        g.create_dataset("log_mbh", data=log_mbh, compression="gzip")
        g.create_dataset("log_edd", data=log_edd, compression="gzip")
        g.create_dataset("gamma_warm", data=gamma_warm, compression="gzip")
        g.create_dataset("wavelength", data=wave_aa, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        g.attrs["source_pickle"] = input_pickle.name
        g.attrs["source_sha256"] = _sha256(input_pickle)
        g.attrs["n_mbh"] = log_mbh.size
        g.attrs["n_edd"] = log_edd.size
        g.attrs["n_gamma_warm"] = gamma_warm.size
        g.attrs["n_wave"] = wave_aa.size
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu erg/s/Hz (renormalized at runtime)"
        g.attrs["gamma_warm_label"] = (
            "AGNfitter-rX 'warmIndex' -- spectral index of the warm "
            "Comptonization zone (Kubota & Done 2018)"
        )

    print(
        f"wrote {output_h5} — {log_mbh.size} M_BH x {log_edd.size} Edd x "
        f"{gamma_warm.size} gamma_warm x {wave_aa.size} wavelengths; "
        f"gamma_warm [{gamma_warm.min():.2f}, {gamma_warm.max():.2f}]"
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/BBB/KD18.pickle"),
        help="Path to AGNfitter-rX's KD18.pickle.",
    )
    p.add_argument(
        "--input-warmindex",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/BBB/KD18_warmInd.pickle"),
        help="Path to AGNfitter-rX's KD18_warmInd.pickle.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/kd18_agnfitter_disc_grid.h5"),
        help="Destination HDF5 path for the plain (logBHmass, logEddra) grid.",
    )
    p.add_argument(
        "--output-warmindex",
        type=Path,
        default=Path("data/kd18_agnfitter_warmindex_disc_grid.h5"),
        help="Destination HDF5 path for the (logBHmass, logEddra, warmIndex) grid.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch both pickles from the AGNfitter GitHub repo (pinned tag) "
        "instead of reading --input/--input-warmindex. No AGNfitter install needed.",
    )
    args = p.parse_args()
    from _agnfitter_download import resolve

    input_plain = resolve(args.input, "models/BBB/KD18.pickle", download=args.download)
    input_warm = resolve(
        args.input_warmindex, "models/BBB/KD18_warmInd.pickle", download=args.download
    )
    build_plain(input_plain, args.output)
    build_warmindex(input_warm, args.output_warmindex)


if __name__ == "__main__":
    _cli()
