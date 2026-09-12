#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build DH02_CE01 cold-dust grid from AGNfitter-rX pickle.

The DH02_CE01 library (Dale & Helou 2002 + Chary & Elbaz 2001) is shipped
with AGNfitter-rX as a single-axis (irlum) template grid: 169 templates, each
carrying its OWN native ``log10(nu/Hz)`` wavelength grid. There are exactly
two distinct native grids in the pickle: a fine 1366-point grid (105 of the
169 templates, the low/mid ``irlum`` rows) and a coarse 193-point grid (the
remaining 64 templates, spanning both the lowest AND highest ``irlum`` rows
-- not only the top edge, as an earlier pass at this defect assumed from an
unrepresentative sample of raw indices).

Native-sampling preservation (#task3-M-B / D2)
-----------------------------------------------
This script preserves every template's own native samples instead of
double-resampling everything onto one foreign 1024-point ``np.logspace`` axis
(the prior approach, which also stored the result as ``float32``), via the
shared :mod:`_grid_native_sampling` helpers (task3 fix round 1 RULING R14),
also used by ``build_agnfitter_bbb_reference.py``'s committed crossval
reference for this same library. The output axis is the **union** of every
distinct native grid, restricted to the wavelength range common to both
grids -- that common range is what every template can actually populate
without extrapolating, and it is almost exactly the historical
``[3.6e3, 1.1e7]`` Å bounds this script used to hardcode (the coarse grid's
own extent). The prior float32 + 1024-point resampling reached ~0.47 dex
(factor ~3x) error at the top ``irlum`` node -- see the ``colddust_radio.md``
D2 finding.

Duplicate-``irlum`` tie-break (task3 fix round 1, item 1)
----------------------------------------------------------
Three ``irlum`` values repeat across raw rows (bit-identical float64
repeats, not merely close ones): 8.3091 (rows [58, 59]), 8.3166 (rows
[55, 56, 57, 60, 61, 62, 63]), 8.324 (rows [53, 54]). AGNfitter-rX's own
``STARBURSTFdict_4plot[str(irlum)] = ...`` construction
(``MODEL_AGNfitter.py::STARBURST``) iterates raw rows in storage order and
assigns into a plain dict, so a repeated key is overwritten by every later
occurrence -- the LAST raw-order row is what AGNfitter-rX actually uses at
runtime. An earlier version of this script kept the FIRST occurrence (a
stable sort by irlum, then ``np.unique(..., return_index=True)``), which
disagreed with upstream by ~0.0254 dex at the affected edge node --
undetected because the committed crossval reference shared the same (wrong)
tie-break. :func:`_grid_native_sampling.dedupe_last_write_wins` is the fix,
shared with the reference builder so both agree with upstream rather than
just with each other.

Reference:
  Dale, D. A. & Helou, G. 2002, ApJ, 576, 159. https://doi.org/10.1086/341632
  Chary, R. & Elbaz, D. 2001, ApJ, 556, 562. https://doi.org/10.1086/321609
"""

import hashlib
import pickle
import pickletools
import warnings
from pathlib import Path

import h5py
import numpy as np
from _grid_native_sampling import dedupe_last_write_wins, native_wavelength_grid, place_on_grid

# --- Configuration ---

# AGNfitter-rX pickle source (requires local clone)
AGNFITTER_PICKLE = Path("/tmp/AGNfitter-rX/models/STARBURST/DH02_CE01.pickle")

# Output grid
OUTPUT_GRID = Path(__file__).resolve().parents[1] / "data" / "dh02_ce01_grid.h5"

# Speed of light
C_AA_PER_S = 2.99792458e18  # [Å·Hz]

#: Allow-list for the untrusted upstream pickle: verified (via a preflight
#: opcode scan, see :func:`_preflight_opcode_scan`) that DH02_CE01.pickle's
#: only GLOBAL references are these three numpy container primitives -- a
#: plain dict of numpy arrays, no pandas DataFrame at all, unlike the BBB/
#: SKIRTOR pickles whose restricted unpicklers (``build_agnfitter_bbb_reference.py``,
#: ``build_skirtor_mean3p_grid.py``) also need a pandas allow-list.
_SAFE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler limited to numpy container primitives (no arbitrary code)."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in DH02_CE01's numpy-only "
                "safe allow-list. If this is a legitimate numpy primitive, add it to "
                "_SAFE_CLASSES in build_dh02_ce01_grid.py."
            )
        return super().find_class(module, name)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Fail loudly if the pickle's opcode stream references anything unexpected.

    Mirrors ``build_skirtor_mean3p_grid.py``'s ``_preflight_opcode_scan``: a
    static pass over the pickle's disassembly, before any unpickling touches
    the file, so an unexpected class reference is reported by name rather
    than only rejected (mid-load) by :class:`_RestrictedUnpickler`.

    Scope, verified rather than assumed: this catches the literal ``GLOBAL``
    opcode (protocol 0-2), which carries the module/name as a string
    argument in the disassembly text this function parses -- and
    DH02_CE01.pickle is confirmed protocol 2. It does NOT catch a
    modern-protocol (4+) pickle's ``STACK_GLOBAL``, whose module/name are two
    separate stack pushes with no argument on the ``STACK_GLOBAL`` opcode
    itself, un-resolved by ``pickletools.dis()``'s plain text output (checked
    directly: a ``__reduce__``-based ``os.system`` payload built with the
    default modern protocol passes this scan silently). The real,
    protocol-agnostic security boundary is
    :class:`_RestrictedUnpickler`.\\ ``find_class`` below, which ``pickle``
    calls to resolve a global reference regardless of which opcode encoded
    it, and which the same payload does not get past. This function is a
    defense-in-depth static check for the actual (protocol-2) file, not a
    substitute for that boundary.
    """
    import io

    with pickle_path.open("rb") as fh:
        out = io.StringIO()
        pickletools.dis(fh, annotate=0, out=out)
    seen: set[tuple[str, str]] = set()
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
        seen.add((parts[0], parts[1]))
    unexpected = seen - _SAFE_CLASSES
    if unexpected:
        raise RuntimeError(
            f"Unexpected GLOBAL references in {pickle_path}: {sorted(unexpected)}. "
            "Refusing to proceed. Vet each entry, then add legitimate numpy "
            "primitives to _SAFE_CLASSES."
        )


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes, for provenance tracking."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_agnfitter_dh02_ce01_pickle(pickle_path):
    r"""Load DH02_CE01 pickle from AGNfitter-rX (restricted unpickler).

    Parameters
    ----------
    pickle_path : Path or str
        Path to DH02_CE01.pickle.

    Returns
    -------
    dict
        Keys: irlum (169,), wavelength (169,) [object array], SED (169,) [object array],
        parameters (list).

    Raises
    ------
    RuntimeError
        If the pickle's opcode stream contains a literal ``GLOBAL`` reference
        outside the numpy-only allow-list (see :func:`_preflight_opcode_scan`;
        this static check's scope is the protocol-0-2 ``GLOBAL`` opcode).
    pickle.UnpicklingError
        If, regardless of protocol, resolving any global reference during the
        actual load names a class outside the allow-list (see
        :class:`_RestrictedUnpickler`; this is the protocol-agnostic
        boundary).

    Notes
    -----
    The pickle stores a list of templates with different wavelength grids.
    ``wavelength[i]`` and ``SED[i]`` are 1D arrays specific to template i.
    This is a BUILD-TIME-ONLY read of a trusted, developer-supplied
    AGNfitter-rX clone at a pinned tag -- the shipped artefact is HDF5 and no
    test ever reads a pickle -- but the untrusted-input discipline (#task3
    global constraints: pickles are read only through the repository's
    restricted-unpickler pattern) applies regardless of trust level.
    """
    pickle_path = Path(pickle_path)
    _preflight_opcode_scan(pickle_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with pickle_path.open("rb") as f:
            data = _RestrictedUnpickler(f, encoding="latin-1").load()
    return data


def build_dh02_ce01_grid(pickle_path, output_h5_path):
    """Build DH02_CE01 grid HDF5 from AGNfitter-rX pickle.

    Parameters
    ----------
    pickle_path : Path or str
        Path to DH02_CE01.pickle.
    output_h5_path : Path or str
        Path to write the output HDF5.
    """
    print(f"Loading pickle from {pickle_path}...")
    data = load_agnfitter_dh02_ce01_pickle(pickle_path)

    irlum_raw = np.asarray(data["irlum-values"], dtype=np.float64)  # (169,) -- not sorted
    wavelength_list = data["wavelength"]  # (169,) object array, log10(nu/Hz) per template
    sed_list = data["SED"]  # (169,) object array

    print(
        f"  irlum: {len(irlum_raw)} templates, "
        f"min={irlum_raw.min():.2f}, max={irlum_raw.max():.2f}"
    )

    # Keep the LAST raw-order occurrence of each duplicated irlum value,
    # matching AGNfitter-rX's own dict-keyed-by-str(irlum) construction (see
    # module docstring). Dedupe BEFORE building the wavelength axis so the
    # kept rows alone (not the discarded duplicates) determine it.
    irlum_unique, kept_raw_index = dedupe_last_write_wins(irlum_raw)
    print(
        f"  Deduplicated (last-write-wins): {len(irlum_unique)} unique irlum "
        f"(was {len(irlum_raw)}); kept raw rows {kept_raw_index.tolist()}"
    )

    wave_aa_rows = [
        C_AA_PER_S / 10.0 ** np.asarray(wavelength_list[i], dtype=np.float64)
        for i in kept_raw_index
    ]
    sed_rows = [np.asarray(sed_list[i], dtype=np.float64) for i in kept_raw_index]

    common_wave, mode = native_wavelength_grid(wave_aa_rows)
    n_wavelength = common_wave.size
    print(
        f"  Native grid mode: {mode}; {n_wavelength}-point axis "
        f"[{common_wave[0]:.1f}, {common_wave[-1]:.2e}] Å "
        f"({common_wave[0] / 1e4:.4f}-{common_wave[-1] / 1e4:.1f} µm)"
    )

    sed_grid = np.zeros((len(irlum_unique), n_wavelength), dtype=np.float64)
    for i in range(len(irlum_unique)):
        if i % 20 == 0:
            print(f"    Template {i + 1}/{len(irlum_unique)}")
        sed_grid[i] = place_on_grid(wave_aa_rows[i], sed_rows[i], common_wave)

    # Write HDF5
    print(f"\nWriting to {output_h5_path}...")
    with h5py.File(output_h5_path, "w") as f:
        grp = f.create_group("dh02_ce01")
        grp.create_dataset("wavelength", data=common_wave, compression="gzip")
        grp.create_dataset("irlum_axis", data=irlum_unique, compression="gzip")
        grp.create_dataset(
            "template",
            data=sed_grid,
            compression="gzip",
            chunks=(10, min(128, n_wavelength)),
        )
        # kept_raw_index[j] is the raw pickle row (0-168) that irlum_axis[j]
        # was built from -- lets a test pin the kept row against upstream's
        # own selection, independent of whether this file and the committed
        # crossval reference happen to agree with each other.
        grp.create_dataset("kept_raw_index", data=kept_raw_index, compression="gzip")
        # Metadata
        grp.attrs["description"] = (
            "Dale & Helou 2002 + Chary & Elbaz 2001 cold-dust templates. "
            "Single-axis (L_IR) grid. Relative L_nu (unnormalized; "
            "model function handles energy-balance normalization)."
        )
        grp.attrs["n_templates"] = len(irlum_unique)
        grp.attrs["n_wavelength"] = n_wavelength
        grp.attrs["wavelength_unit"] = "Angstrom"
        grp.attrs["irlum_unit"] = "log10(L_IR / L_sun)"
        grp.attrs["template_unit"] = "Relative L_nu (unnormalized)"
        grp.attrs["native_sampling"] = mode
        grp.attrs["deduplication_note"] = (
            "169 original templates (3 duplicated irlum values, 8 rows dropped) "
            "reduced to 161 unique, keeping the LAST raw-order occurrence of each "
            "duplicated irlum value (matches MODEL_AGNfitter.py's "
            "dict-keyed-by-str(irlum) construction). Sorted ascending by irlum."
        )
        grp.attrs["source_sha256"] = _sha256(Path(pickle_path))

    print(f"  Success! File: {output_h5_path}")
    print(f"  Size: {output_h5_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build DH02_CE01 grid HDF5 from AGNfitter-rX pickle."
    )
    parser.add_argument(
        "--pickle",
        type=Path,
        default=AGNFITTER_PICKLE,
        help=f"Path to DH02_CE01.pickle (default: {AGNFITTER_PICKLE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_GRID,
        help=f"Output HDF5 path (default: {OUTPUT_GRID})",
    )
    args = parser.parse_args()

    if not args.pickle.is_file():
        raise FileNotFoundError(
            f"Pickle not found: {args.pickle}\n"
            "Clone AGNfitter-rX: git clone --branch AGNfitter-rX_v0.1 "
            "https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_dh02_ce01_grid(args.pickle, args.output)
