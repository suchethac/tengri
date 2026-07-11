#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build DH02_CE01 cold-dust grid from AGNfitter-rX pickle.

The DH02_CE01 library (Dale & Helou 2002 + Chary & Elbaz 2001) is shipped
with AGNfitter-rX as a single-axis (irlum) template grid. This script
reads the vendored pickle, normalizes and resamples to a standard grid,
and writes an HDF5 archive suitable for use with tengri's tabulated
dust emission models.

Reference:
  Dale, D. A. & Helou, G. 2002, ApJ, 576, 159. https://doi.org/10.1086/341632
  Chary, R. & Elbaz, D. 2001, ApJ, 556, 562. https://doi.org/10.1086/321609
"""

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

# Common resampling grid: 1024 points log-spaced over 3.6 µm to 1100 µm
# (matches the reference grid used in agnfitter_cold_dust_reference.h5)
WAVELENGTH_MIN_AA = 3.6e3  # 0.36 µm
WAVELENGTH_MAX_AA = 1.1e7  # 1100 µm
N_WAVELENGTH = 1024

# Speed of light
C_AA_PER_S = 2.99792458e18  # [Å·Hz]


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


def get_relative_sed(nu_hz, sed_raw):
    """Extract relative SED without normalization (stored as-is from AGNfitter).

    Parameters
    ----------
    nu_hz : ndarray, shape (n,)
        Frequency grid [Hz].
    sed_raw : ndarray, shape (n,)
        Raw flux F_nu from AGNfitter pickle.

    Returns
    -------
    ndarray, shape (n,)
        Relative SED (unnormalized).

    Notes
    -----
    Templates are stored relative; the model function handles normalization.
    This ensures the templates retain their original relative amplitudes
    across the grid.
    """
    return sed_raw


def resample_to_grid(wave_src, spec_src, wave_tgt):
    """Resample spectrum to target wavelength grid via linear interp.

    Parameters
    ----------
    wave_src : ndarray
        Source wavelength grid [Å] (will be sorted internally).
    spec_src : ndarray
        Source spectrum (already normalized).
    wave_tgt : ndarray
        Target wavelength grid [Å] (assumed sorted ascending).

    Returns
    -------
    ndarray
        Resampled spectrum on wave_tgt.
    """
    # Sort source grid if needed (log10(nu) grid gives descending wavelengths)
    sort_idx = np.argsort(wave_src)
    wave_src_sorted = wave_src[sort_idx]
    spec_src_sorted = spec_src[sort_idx]
    return np.interp(wave_tgt, wave_src_sorted, spec_src_sorted, left=0.0, right=0.0)


def build_dh02_ce01_grid(
    pickle_path,
    output_h5_path,
    wavelength_min_aa=WAVELENGTH_MIN_AA,
    wavelength_max_aa=WAVELENGTH_MAX_AA,
    n_wavelength=N_WAVELENGTH,
):
    """Build DH02_CE01 grid HDF5 from AGNfitter-rX pickle.

    Parameters
    ----------
    pickle_path : Path or str
        Path to DH02_CE01.pickle.
    output_h5_path : Path or str
        Path to write the output HDF5.
    wavelength_min_aa : float
        Minimum wavelength [Å]. Default: 3600 Å (0.36 µm).
    wavelength_max_aa : float
        Maximum wavelength [Å]. Default: 1.1e7 Å (1100 µm).
    n_wavelength : int
        Number of wavelength points. Default: 1024.
    """
    print(f"Loading pickle from {pickle_path}...")
    data = load_agnfitter_dh02_ce01_pickle(pickle_path)

    irlum_orig = data["irlum-values"]  # (169,) – not sorted
    wavelength_list = data["wavelength"]  # (169,) object array
    sed_list = data["SED"]  # (169,) object array

    print(
        f"  irlum: {len(irlum_orig)} templates, "
        f"min={irlum_orig.min():.2f}, max={irlum_orig.max():.2f}"
    )
    print("  Original templates have varying wavelength grids")

    # Build target wavelength grid (log-spaced in linear space, but covering wide range)
    wave_tgt = np.logspace(
        np.log10(wavelength_min_aa),
        np.log10(wavelength_max_aa),
        n_wavelength,
        dtype=np.float32,
    )
    print(f"  Resampling to {n_wavelength}-point grid: {wave_tgt[0]:.1f}–{wave_tgt[-1]:.2e} Å")

    # Resample all templates to common grid, normalizing as we go
    sed_grid = np.zeros((len(irlum_orig), n_wavelength), dtype=np.float32)
    for i in range(len(irlum_orig)):
        if i % 20 == 0:
            print(f"    Template {i + 1}/{len(irlum_orig)}")
        log10_nu_i = wavelength_list[i]  # log10(nu/Hz), NOT wavelength!
        sed_i = sed_list[i]
        # Convert log10(nu) to frequency [Hz], then to wavelength [Å]
        nu_hz = 10.0**log10_nu_i
        wave_i = C_AA_PER_S / nu_hz  # wavelength in Angstrom
        # Keep template relative (unnormalized; model function handles normalization)
        sed_relative = get_relative_sed(nu_hz, sed_i)
        # Resample to common wavelength grid (sorted ascending Angstrom)
        sed_resampled = resample_to_grid(wave_i, sed_relative, wave_tgt)
        sed_grid[i] = sed_resampled

    # Sort by irlum, handling duplicates: keep only first occurrence
    sort_idx = np.argsort(irlum_orig)
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
        grp.create_dataset("wavelength", data=wave_tgt, compression="gzip")
        grp.create_dataset("irlum_axis", data=irlum_unique, compression="gzip")
        grp.create_dataset(
            "template",
            data=sed_unique,
            compression="gzip",
            chunks=(10, 128),
        )
        # Metadata
        grp.attrs["description"] = (
            "Dale & Helou 2002 + Chary & Elbaz 2001 cold-dust templates. "
            "Single-axis (L_IR) grid. Relative L_nu (unnormalized; "
            "model function handles energy-balance normalization)."
        )
        grp.attrs["n_templates"] = len(irlum_unique)
        grp.attrs["n_wavelength"] = len(wave_tgt)
        grp.attrs["wavelength_unit"] = "Angstrom"
        grp.attrs["irlum_unit"] = "log10(L_IR / L_sun)"
        grp.attrs["template_unit"] = "Relative L_nu (unnormalized)"
        grp.attrs["deduplication_note"] = (
            "169 original templates (8 duplicate irlum values at low end) "
            "reduced to 161 unique. Sorted ascending by irlum."
        )

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
