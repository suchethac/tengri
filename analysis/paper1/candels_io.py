"""Load and parse CANDELS photometry catalog for Paper I analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_candels_z1() -> dict:
    """Load CANDELS_GDSS_workshop_z1.dat catalog.

    Returns dict with keys:
        id, z, photometry_bands, photometry_values, photometry_errors, flags

    Notes:
        Data path: /Users/suchethacooray/Projects/tengri/analysis/hst_proposal/data/CANDELS_GDSS_workshop_z1.dat
        Missing data indicated by 98.999 or negative errors
        flg1: data quality flag (0 = good, 1 = issues)
    """
    data_path = Path(
        "/Users/suchethacooray/Projects/tengri/"
        "analysis/hst_proposal/data/CANDELS_GDSS_workshop_z1.dat"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"CANDELS data not found at {data_path}")

    with open(data_path) as f:
        header = f.readline().strip("#").strip().split()

    data = np.genfromtxt(data_path, skip_header=1)

    id_idx = header.index("ID")
    z_idx = header.index("zz")
    flg1_idx = header.index("flg1")
    flg2_idx = header.index("flg2")

    bands = []
    for h in header:
        if h.startswith("e"):
            continue
        if h in ["ID", "zz", "flg1", "flg2"]:
            continue
        bands.append(h)

    result = {
        "id": data[:, id_idx].astype(int),
        "z": data[:, z_idx],
        "flg1": data[:, flg1_idx].astype(int),
        "flg2": data[:, flg2_idx],
        "bands": bands,
        "header": header,
    }

    return result


def compute_snr_from_error(err: float, sentinel: float = 98.99) -> float:
    """Compute S/N = 1 / (0.921 * err) for a single band.

    Args:
        err: Photometric error (magnitude space)
        sentinel: Non-detection sentinel value

    Returns:
        S/N if detected, np.nan otherwise

    Notes:
        S/N = 1 / (0.921 * mag_err) is the standard definition
        for converting magnitude errors to flux S/N.
    """
    if err <= 0 or err > 10 or abs(err - sentinel) < 0.1:
        return np.nan
    return 1.0 / (0.921 * err)


def is_detected(mag: float, err: float, sentinel: float = 98.99) -> bool:
    """Check if a band is detected (not a sentinel and has valid error)."""
    return mag < 90 and err > 0 and err < 10 and abs(mag - sentinel) > 0.1


if __name__ == "__main__":
    cat = load_candels_z1()
    print(f"Loaded {len(cat['id'])} galaxies")
    print(f"Redshift range: {cat['z'].min():.3f} - {cat['z'].max():.3f}")
    print(f"Photometric bands: {len(cat['bands'])}")
