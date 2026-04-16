#!/usr/bin/env python3
"""Download and build MAPPINGS V photoionization grids from Flury et al. 2024.

Source
------
Flury et al. 2024, arXiv:2412.06763
  "New Ionization Models and the Shocking Nitrogen Excess at z > 5"
  Zenodo: https://zenodo.org/records/14140949
  DOI: 10.5281/zenodo.14140949

MAPPINGS V v5.2.1 photoionization model grids computed with:
  - Nicholls+2017 empirical stellar abundance patterns (ζ_O = 0.05–2)
  - Jenkins+2009/2014 empirical dust depletion (F★ = 0.43)
  - CHIANTI v10 atomic data

Model types
-----------
  sb99    : Starburst99 (Geneva+WM-Basic/CMFGEN, Kroupa IMF 120 Msun)
  bpass   : BPASS v2.2 (binary stars, Kroupa IMF 120 Msun)
  agn_oxaf: OPTXAGNF AGN SED (disk power-law, Jin+2012 variant)

Density structures
------------------
  cpr : constant pressure (isobaric)  — recommended for most applications
  cdn : constant density (isochoric)  — larger/evolved HII regions

Output HDF5 structure
---------------------
  data/flury2024_grids.h5
  ├── line_names           (N_lines,)           str — PyNeb format
  ├── line_wavelengths_aa  (N_lines,)           float32 — vacuum Å
  ├── sb99/
  │   ├── cpr/
  │   │   ├── z_axis       (N_z,)              ζ_O (solar-relative metallicity)
  │   │   ├── logU_axis    (N_u,)              log10(U) ionization parameter
  │   │   ├── log_age_yr_axis  (N_a,)          log10(age/yr) — inst ages
  │   │   ├── sfh_labels   (N_s,)              str — "inst" or "cont"
  │   │   ├── logn_axis    (N_n,)              log10(density/cm⁻³)
  │   │   ├── logHB_per_logq  (N_z,N_a,N_s,N_u,N_n)  log10(L_Hβ/Q_H [erg/photon])
  │   │   └── line_ratios  (N_z,N_a,N_s,N_u,N_n,N_lines)  line/Hβ
  │   └── cdn/  (same)
  ├── bpass/ (same structure, different age grid)
  └── agn_oxaf/
      ├── cpr/
      │   ├── z_axis, logU_axis, logmbh_axis, logedd_axis, logn_axis
      │   ├── logHB_per_lum   (N_z,N_mbh,N_edd,N_u,N_n)  log10(L_Hβ/L_ion)
      │   └── line_ratios     (N_z,N_mbh,N_edd,N_u,N_n,N_lines)
      └── cdn/

Usage
-----
    python scripts/build_flury2024_grids.py
    python scripts/build_flury2024_grids.py --output data/custom.h5
    python scripts/build_flury2024_grids.py --cache-dir /tmp/flury2024_csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Zenodo source files
# ---------------------------------------------------------------------------

_ZENODO_BASE = "https://zenodo.org/records/14140949/files"

_STELLAR_MODELS = ["sb99", "bpass"]
_DENSITY_STRUCTURES = ["cpr", "cdn"]

_ZENODO_FILES: dict[str, str] = {
    "sb99-cpr": f"{_ZENODO_BASE}/sb99-cpr_fluxes.csv",
    "sb99-cdn": f"{_ZENODO_BASE}/sb99-cdn_fluxes.csv",
    "bpass-cpr": f"{_ZENODO_BASE}/bpass-cpr_fluxes.csv",
    "bpass-cdn": f"{_ZENODO_BASE}/bpass-cdn_fluxes.csv",
    "agn-oxaf-cpr": f"{_ZENODO_BASE}/agn-oxaf-cpr_fluxes.csv",
    "agn-oxaf-cdn": f"{_ZENODO_BASE}/agn-oxaf-cdn_fluxes.csv",
}

# Solar metallicity log10(Z_sun) — Asplund+2009
_LOG10_ZSUN = -1.8477116556169435

# Physical constants
_LSUN_ERG = 3.828e33  # erg/s

# ---------------------------------------------------------------------------
# Wavelength parser for PyNeb-format column names
# ---------------------------------------------------------------------------

# Non-flux parameter columns present in every flux CSV
_PARAM_COLS_STELLAR = {"id", "z", "logu", "age", "sfh", "logq", "logn", "logHB"}
_PARAM_COLS_AGN = {"id", "z", "logu", "mbh", "edd", "lum", "logn", "logHB"}


def _parse_wavelength_aa(col: str) -> float | None:
    """Parse wavelength in Å from a PyNeb-format column name.

    Handles:
      ``O3_5007A``    → 5007.0 Å
      ``Ar2_0698um``  → 6980.0 Å
      ``C2_158um``    → 1_580_000.0 Å  (far-IR, kept for completeness)

    Returns None if the column is a parameter column or unparseable.
    """
    if "_" not in col:
        return None
    suffix = col.split("_", 1)[1]  # "5007A" or "0698um"
    try:
        if suffix.endswith("um"):
            return float(suffix[:-2]) * 1e4  # μm → Å
        elif suffix.endswith("A"):
            return float(suffix[:-1])
        else:
            return None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CSV download & parse
# ---------------------------------------------------------------------------


def _download_csv(url: str, cache_path: Path) -> list[dict]:
    """Download (or load from cache) a Zenodo CSV and return rows as dicts."""
    if cache_path.exists():
        print(f"    Using cache: {cache_path}")
    else:
        print(f"    Downloading: {url}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.urlopen(url, timeout=120)  # noqa: S310
            data = req.read()
        except Exception as exc:
            print(f"\nERROR downloading {url}: {exc}")
            sys.exit(1)
        cache_path.write_bytes(data)
        print(f"    Saved to {cache_path} ({len(data)/1e6:.1f} MB)")

    text = cache_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def _identify_line_cols(row: dict, param_cols: set[str]) -> list[str]:
    """Return column names that correspond to emission lines."""
    return [c for c in row if c not in param_cols and _parse_wavelength_aa(c) is not None]


def _build_stellar_grid(
    rows: list[dict],
    model_key: str,
) -> dict:
    """Build a structured grid dict from stellar (sb99/bpass) CSV rows.

    Returns
    -------
    dict with keys:
        z_axis         (N_z,)
        logU_axis      (N_u,)  — ascending (e.g. -4, -3.5, ..., -0.5)
        log_age_yr_axis (N_a,) — log10(age/yr), inst ages only
        sfh_labels     (N_s,)  — list of str e.g. ["inst", "cont"]
        logn_axis      (N_n,)
        line_names     (N_lines,)
        line_waves_aa  (N_lines,)
        logHB_per_logq (N_z, N_a, N_s, N_u, N_n)
        line_ratios    (N_z, N_a, N_s, N_u, N_n, N_lines)
    """
    line_cols = _identify_line_cols(rows[0], _PARAM_COLS_STELLAR)
    line_waves = np.array([_parse_wavelength_aa(c) for c in line_cols], dtype=np.float32)

    # Discover unique axis values
    z_vals = sorted({float(r["z"]) for r in rows})
    logu_vals = sorted({float(r["logu"]) for r in rows})
    age_vals = sorted({float(r["age"]) for r in rows})  # Myr
    sfh_vals = sorted({r["sfh"] for r in rows})         # "inst", "cont"
    logn_vals = sorted({float(r["logn"]) for r in rows})

    # Convert age Myr → log10(yr)
    log_age_yr_vals = np.log10(np.array(age_vals) * 1e6)

    n_z, n_u, n_a, n_s, n_n = (
        len(z_vals), len(logu_vals), len(age_vals), len(sfh_vals), len(logn_vals)
    )
    n_lines = len(line_cols)

    print(
        f"    {model_key}: z={n_z}, logU={n_u}, age={n_a}, sfh={n_s},"
        f" logn={n_n} → grid ({n_z},{n_a},{n_s},{n_u},{n_n},{n_lines})"
    )

    # Build index maps
    z_idx = {v: i for i, v in enumerate(z_vals)}
    u_idx = {v: i for i, v in enumerate(logu_vals)}
    a_idx = {v: i for i, v in enumerate(age_vals)}
    s_idx = {v: i for i, v in enumerate(sfh_vals)}
    n_idx = {v: i for i, v in enumerate(logn_vals)}

    shape = (n_z, n_a, n_s, n_u, n_n)
    logHB_per_logq = np.full(shape, np.nan, dtype=np.float32)
    line_ratios = np.full((*shape, n_lines), np.nan, dtype=np.float32)

    n_filled = 0
    for row in rows:
        iz = z_idx[float(row["z"])]
        iu = u_idx[float(row["logu"])]
        ia = a_idx[float(row["age"])]
        is_ = s_idx[row["sfh"]]
        in_ = n_idx[float(row["logn"])]

        logHB = float(row["logHB"])
        logq = float(row["logq"])
        logHB_per_logq[iz, ia, is_, iu, in_] = logHB - logq

        for jl, col in enumerate(line_cols):
            v = row[col]
            line_ratios[iz, ia, is_, iu, in_, jl] = float(v) if v else np.nan

        n_filled += 1

    n_total = int(np.prod(shape))
    n_nan = int(np.sum(np.isnan(logHB_per_logq)))
    print(f"    Filled {n_filled} rows; {n_nan}/{n_total} grid points are NaN")

    return {
        "z_axis": np.array(z_vals, dtype=np.float32),
        "logU_axis": np.array(logu_vals, dtype=np.float32),
        "log_age_yr_axis": log_age_yr_vals.astype(np.float32),
        "sfh_labels": sfh_vals,
        "logn_axis": np.array(logn_vals, dtype=np.float32),
        "line_names": line_cols,
        "line_waves_aa": line_waves,
        "logHB_per_logq": logHB_per_logq,
        "line_ratios": line_ratios,
    }


def _build_agn_grid(
    rows: list[dict],
    model_key: str,
) -> dict:
    """Build a structured grid dict from AGN (agn-oxaf) CSV rows.

    Axes: (z, mbh, edd, logU, logn).

    The AGN CSV stores ``lum`` (log10 ionizing luminosity in erg/s) rather
    than ``logq`` (ionizing photon rate). We therefore store
    ``logHB_per_lum = logHB − lum``, the log10 ratio of Hβ luminosity to
    ionizing luminosity (both in erg/s). The backend scales by the AGN
    ionizing luminosity in erg/s, not Q_H in photons/s.
    """
    line_cols = _identify_line_cols(rows[0], _PARAM_COLS_AGN)
    line_waves = np.array([_parse_wavelength_aa(c) for c in line_cols], dtype=np.float32)

    z_vals = sorted({float(r["z"]) for r in rows})
    logu_vals = sorted({float(r["logu"]) for r in rows})
    mbh_vals = sorted({float(r["mbh"]) for r in rows})   # log10(M_BH/Msun)
    edd_vals = sorted({float(r["edd"]) for r in rows})   # log10(L/L_Edd)
    logn_vals = sorted({float(r["logn"]) for r in rows})

    n_z, n_u, n_m, n_e, n_n = (
        len(z_vals), len(logu_vals), len(mbh_vals), len(edd_vals), len(logn_vals)
    )
    n_lines = len(line_cols)

    print(
        f"    {model_key}: z={n_z}, logU={n_u}, logMBH={n_m}, logEdd={n_e},"
        f" logn={n_n} → grid ({n_z},{n_m},{n_e},{n_u},{n_n},{n_lines})"
    )

    z_idx = {v: i for i, v in enumerate(z_vals)}
    u_idx = {v: i for i, v in enumerate(logu_vals)}
    m_idx = {v: i for i, v in enumerate(mbh_vals)}
    e_idx = {v: i for i, v in enumerate(edd_vals)}
    n_idx = {v: i for i, v in enumerate(logn_vals)}

    shape = (n_z, n_m, n_e, n_u, n_n)
    logHB_per_lum = np.full(shape, np.nan, dtype=np.float32)
    line_ratios = np.full((*shape, n_lines), np.nan, dtype=np.float32)

    for row in rows:
        iz = z_idx[float(row["z"])]
        iu = u_idx[float(row["logu"])]
        im = m_idx[float(row["mbh"])]
        ie = e_idx[float(row["edd"])]
        in_ = n_idx[float(row["logn"])]

        logHB = float(row["logHB"])
        lum = float(row["lum"])  # log10(L_ion / erg s^-1) used by MAPPINGS V
        logHB_per_lum[iz, im, ie, iu, in_] = logHB - lum

        for jl, col in enumerate(line_cols):
            v = row[col]
            line_ratios[iz, im, ie, iu, in_, jl] = float(v) if v else np.nan

    n_total = int(np.prod(shape))
    n_nan = int(np.sum(np.isnan(logHB_per_lum)))
    print(f"    Filled {len(rows)} rows; {n_nan}/{n_total} grid points are NaN")

    return {
        "z_axis": np.array(z_vals, dtype=np.float32),
        "logU_axis": np.array(logu_vals, dtype=np.float32),
        "logmbh_axis": np.array(mbh_vals, dtype=np.float32),
        "logedd_axis": np.array(edd_vals, dtype=np.float32),
        "logn_axis": np.array(logn_vals, dtype=np.float32),
        "line_names": line_cols,
        "line_waves_aa": line_waves,
        "logHB_per_lum": logHB_per_lum,
        "line_ratios": line_ratios,
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def _write_group(
    parent: h5py.Group,
    name: str,
    data: dict,
    model_type: str,
) -> None:
    """Write one (model, density_structure) group into the HDF5 file."""
    opts = dict(compression="gzip", compression_opts=5)
    g = parent.create_group(name)

    g.create_dataset("line_names", data=np.array(data["line_names"], dtype="S40"), **opts)
    g.create_dataset("line_wavelengths_aa", data=data["line_waves_aa"], **opts)
    # Stellar grids: logHB_per_logq = log10(L_Hβ/Q_H [erg/photon])
    # AGN grids:     logHB_per_lum  = log10(L_Hβ/L_ion [dimensionless])
    if "logHB_per_logq" in data:
        g.create_dataset("logHB_per_logq", data=data["logHB_per_logq"], **opts)
    else:
        g.create_dataset("logHB_per_lum", data=data["logHB_per_lum"], **opts)
    g.create_dataset("line_ratios", data=data["line_ratios"], **opts)

    g.create_dataset("z_axis", data=data["z_axis"], **opts)
    g.create_dataset("logU_axis", data=data["logU_axis"], **opts)
    g.create_dataset("logn_axis", data=data["logn_axis"], **opts)

    if model_type in ("sb99", "bpass"):
        g.create_dataset("log_age_yr_axis", data=data["log_age_yr_axis"], **opts)
        g.create_dataset(
            "sfh_labels",
            data=np.array(data["sfh_labels"], dtype="S10"),
            **opts,
        )
    elif model_type == "agn_oxaf":
        g.create_dataset("logmbh_axis", data=data["logmbh_axis"], **opts)
        g.create_dataset("logedd_axis", data=data["logedd_axis"], **opts)


def _write_hdf5(
    out_path: Path,
    grids: dict[str, dict],  # key: "sb99-cpr", etc.
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.attrs["description"] = (
            "MAPPINGS V v5.2.1 photoionization grids for tengri. "
            "Source: Flury et al. 2024, arXiv:2412.06763, Zenodo 14140949. "
            "Nicholls+2017 empirical abundances, Jenkins+2009/2014 dust depletion, "
            "CHIANTI v10 atomic data. Built by scripts/build_flury2024_grids.py."
        )
        f.attrs["zenodo_doi"] = "10.5281/zenodo.14140949"
        f.attrs["arxiv"] = "2412.06763"
        f.attrs["mappings_version"] = "5.2.1"

        for key, data in grids.items():
            # key: "sb99-cpr", "bpass-cdn", "agn-oxaf-cpr", ...
            parts = key.split("-")
            if parts[0] == "agn":
                model = "agn_oxaf"        # "agn-oxaf-cpr" → "agn_oxaf"
                density = parts[2]
            else:
                model = parts[0]          # "sb99" or "bpass"
                density = parts[1]        # "cpr" or "cdn"

            if model not in f:
                f.create_group(model)
            model_group = f[model]

            _write_group(model_group, density, data, model)
            print(f"  Wrote {model}/{density}")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "data" / "flury2024_grids.h5"),
        help="Output HDF5 file path (default: data/flury2024_grids.h5)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).parents[1] / "data" / "_flury2024_cache"),
        help="Directory to cache downloaded CSVs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(_ZENODO_FILES.keys()),
        choices=list(_ZENODO_FILES.keys()),
        help="Which model/density combinations to download (default: all)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    print("\n=== Flury+2024 MAPPINGS V photoionization grids ===")
    print(f"  Zenodo : https://zenodo.org/records/14140949")
    print(f"  Output : {out_path}")
    print(f"  Cache  : {cache_dir}\n")

    grids: dict[str, dict] = {}

    for key in args.models:
        url = _ZENODO_FILES[key]
        cache_path = cache_dir / f"{key}_fluxes.csv"

        print(f"[{key}]")
        rows = _download_csv(url, cache_path)
        print(f"    Parsed {len(rows)} rows")

        if key.startswith("agn"):
            grids[key] = _build_agn_grid(rows, key)
        else:
            grids[key] = _build_stellar_grid(rows, key)

        print()

    print("=== Writing HDF5 ===")
    _write_hdf5(out_path, grids)

    print("\nDone. Verify with:")
    print(
        "  python -c \""
        "from tengri.nebular import MappingsPhotoStellarBackend; "
        "b = MappingsPhotoStellarBackend('data/flury2024_grids.h5', 'sb99', 'cpr'); "
        "print(b.line_names[:5])\""
    )


if __name__ == "__main__":
    main()
