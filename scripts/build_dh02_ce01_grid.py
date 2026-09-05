#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build DH02_CE01 cold-dust grid from AGNfitter-rX pickle.

The DH02_CE01 library (Dale & Helou 2002 + Chary & Elbaz 2001) is shipped
with AGNfitter-rX as a single-axis (irlum) template grid: 169 templates, each
carrying its OWN native ``log10(nu/Hz)`` wavelength grid. There are exactly
two distinct native grids in the pickle: a fine 1366-point grid (105 of the
169 templates, the low/mid ``irlum`` rows) and a coarse 193-point grid (the
remaining 64 templates, the *highest*-``irlum`` rows, including the top edge
node the crossval test exercises).

Native-sampling preservation (#task3-M-B / D2)
-----------------------------------------------
This script preserves every template's own native samples instead of
double-resampling everything onto one foreign 1024-point ``np.logspace`` axis
(the prior approach, which also stored the result as ``float32``). The output
axis is the **union** of every distinct native grid, restricted to the
wavelength range common to both grids (``[max of the grids' minima, min of
the grids' maxima]``) — that common range is what every template can actually
populate without extrapolation, and it is almost exactly the historical
``[3.6e3, 1.1e7]`` Å bounds this script used to hardcode (the coarse grid's own
extent). Within that range every row's own tabulated points land exactly on
the output axis (:func:`_place_on_grid` returns the native value unchanged
there); a row is only linearly interpolated at axis points that belong to the
*other* native grid. The prior float32 + 1024-point resampling reached ~0.47
dex (factor ~3x) error at the top ``irlum`` node (the coarse-grid rows) — see
the ``colddust_radio.md`` D2 finding.

Reference:
  Dale, D. A. & Helou, G. 2002, ApJ, 576, 159. https://doi.org/10.1086/341632
  Chary, R. & Elbaz, D. 2001, ApJ, 556, 562. https://doi.org/10.1086/321609
"""

import hashlib
import pickle
import warnings
from pathlib import Path

import h5py
import numpy as np

# --- Configuration ---

# AGNfitter-rX pickle source (requires local clone)
AGNFITTER_PICKLE = Path("/tmp/AGNfitter-rX/models/STARBURST/DH02_CE01.pickle")

# Output grid
OUTPUT_GRID = Path(__file__).resolve().parents[1] / "data" / "dh02_ce01_grid.h5"

# Speed of light
C_AA_PER_S = 2.99792458e18  # [Å·Hz]


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes, for provenance tracking."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_agnfitter_dh02_ce01_pickle(pickle_path):
    r"""Load DH02_CE01 pickle from AGNfitter-rX.

    Parameters
    ----------
    pickle_path : Path or str
        Path to DH02_CE01.pickle.

    Returns
    -------
    dict
        Keys: irlum (169,), wavelength (169,) [object array], SED (169,) [object array],
        parameters (list).

    Notes
    -----
    The pickle stores a list of templates with different wavelength grids.
    wavelength[i] and SED[i] are 1D arrays specific to template i.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with open(pickle_path, "rb") as f:
            data = pickle.load(f, encoding="latin-1")
    return data


def _native_wavelength_grid(wave_rows: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """Return the axis every row should be expressed on, preserving native samples.

    Parameters
    ----------
    wave_rows : list of ndarray
        One per-row wavelength array [Å] (any order; sorted internally).

    Returns
    -------
    grid : ndarray
        Ascending wavelength axis [Å].
    mode : str
        ``"verbatim"`` when every row shares exactly one native grid.
        ``"union(common-range)"`` when rows use several distinct native
        grids: the axis is then the sorted union of every distinct native
        grid, restricted to the wavelength range every grid actually covers
        (so every row can populate every axis point without extrapolating
        past its own native coverage).

    Raises
    ------
    RuntimeError
        If the distinct native grids share no common wavelength range.
    """
    unique_grids: list[np.ndarray] = []
    for w in wave_rows:
        w_sorted = np.sort(np.asarray(w, dtype=np.float64))
        is_new = not any(
            w_sorted.shape == u.shape and np.allclose(w_sorted, u, rtol=1e-10, atol=0.0)
            for u in unique_grids
        )
        if is_new:
            unique_grids.append(w_sorted)
    if len(unique_grids) == 1:
        return unique_grids[0], "verbatim"

    lo = max(g.min() for g in unique_grids)
    hi = min(g.max() for g in unique_grids)
    if lo >= hi:
        raise RuntimeError("Native wavelength grids share no common range; cannot build one axis.")
    pieces = [g[(g >= lo) & (g <= hi)] for g in unique_grids]
    return np.unique(np.concatenate(pieces)), "union(common-range)"


def _place_on_grid(wave_row: np.ndarray, sed_row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Express one row's SED on ``grid``, without resampling away native points.

    If ``wave_row`` (sorted) already equals ``grid``, the row's own tabulated
    values are returned unchanged. Otherwise linear interpolation fills the
    grid points this row's own native grid lacks; every point that IS one of
    this row's own native samples is reproduced exactly, since it is present
    verbatim in ``grid`` and ``np.interp`` returns the exact node value at an
    exact-match query point. ``grid`` is restricted to the common range every
    row covers (see :func:`_native_wavelength_grid`), so no extrapolation
    (``left``/``right`` fill) is ever needed.
    """
    order = np.argsort(wave_row)
    w = wave_row[order]
    s = sed_row[order]
    if w.shape == grid.shape and np.allclose(w, grid, rtol=1e-10, atol=0.0):
        return s
    return np.interp(grid, w, s)


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

    irlum_orig = data["irlum-values"]  # (169,) – not sorted
    wavelength_list = data["wavelength"]  # (169,) object array, log10(nu/Hz) per template
    sed_list = data["SED"]  # (169,) object array

    print(
        f"  irlum: {len(irlum_orig)} templates, "
        f"min={irlum_orig.min():.2f}, max={irlum_orig.max():.2f}"
    )

    wave_aa_rows = [C_AA_PER_S / 10.0 ** np.asarray(w, dtype=np.float64) for w in wavelength_list]

    common_wave, mode = _native_wavelength_grid(wave_aa_rows)
    n_wavelength = common_wave.size
    print(
        f"  Native grid mode: {mode}; {n_wavelength}-point axis "
        f"[{common_wave[0]:.1f}, {common_wave[-1]:.2e}] Å "
        f"({common_wave[0] / 1e4:.4f}-{common_wave[-1] / 1e4:.1f} µm)"
    )

    sed_grid = np.zeros((len(irlum_orig), n_wavelength), dtype=np.float64)
    for i in range(len(irlum_orig)):
        if i % 20 == 0:
            print(f"    Template {i + 1}/{len(irlum_orig)}")
        sed_i = np.asarray(sed_list[i], dtype=np.float64)
        sed_grid[i] = _place_on_grid(wave_aa_rows[i], sed_i, common_wave)

    # Sort by irlum, handling duplicates: keep only first occurrence. A
    # stable sort is required here: several irlum VALUES repeat with
    # slightly different (~1%) template shapes (e.g. the two raw rows at the
    # grid-minimum irlum=8.3091), and the default quicksort is not
    # tie-stable, so which of the near-duplicate rows survives dedup would
    # otherwise depend on numpy's internal partitioning rather than on
    # storage order -- a silent, run-to-run-reproducible-but-arbitrary
    # choice that also disagreed with the committed reference's own
    # nearest-match lookup (which always returns the first raw-order match).
    sort_idx = np.argsort(irlum_orig, kind="stable")
    irlum_sorted = irlum_orig[sort_idx]
    sed_sorted = sed_grid[sort_idx]

    # Deduplicate: keep only first occurrence of each unique irlum
    _unique_irlum, unique_idx_in_sorted = np.unique(irlum_sorted, return_index=True)
    unique_idx_in_sorted = np.sort(unique_idx_in_sorted)  # Restore order
    irlum_unique = irlum_sorted[unique_idx_in_sorted]
    sed_unique = sed_sorted[unique_idx_in_sorted]

    print("\n  After sorting and deduplication:")
    print(f"    Unique irlum: {len(irlum_unique)} (was {len(irlum_orig)})")
    print(f"    irlum range: {irlum_unique.min():.2f}–{irlum_unique.max():.2f}")
    print("    Handling: duplicates at low irlum kept first occurrence (8 dup entries)")

    # Write HDF5
    print(f"\nWriting to {output_h5_path}...")
    with h5py.File(output_h5_path, "w") as f:
        grp = f.create_group("dh02_ce01")
        grp.create_dataset("wavelength", data=common_wave, compression="gzip")
        grp.create_dataset("irlum_axis", data=irlum_unique, compression="gzip")
        grp.create_dataset(
            "template",
            data=sed_unique,
            compression="gzip",
            chunks=(10, min(128, n_wavelength)),
        )
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
            "169 original templates (8 duplicate irlum values at low end) "
            "reduced to 161 unique. Sorted ascending by irlum."
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
