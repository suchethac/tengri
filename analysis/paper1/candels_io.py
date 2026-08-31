"""Load and parse CANDELS photometry catalog for Paper I analysis."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: Default catalog location, resolved relative to this file (analysis/paper1/
#: candels_io.py -> paper1 -> analysis -> repository root). Override with the
#: ``TENGRI_CANDELS_CATALOG`` environment variable.
CANDELS_CATALOG = (
    Path(__file__).resolve().parents[2]
    / "analysis"
    / "hst_proposal"
    / "data"
    / "CANDELS_GDSS_workshop_z1.dat"
)


def load_candels_z1() -> dict:
    """Load CANDELS_GDSS_workshop_z1.dat catalog.

    Returns dict with keys:
        id, z, flg1, flg2, bands, header,
        data (the full ``(n_galaxies, n_columns)`` float matrix)

    Notes:
        Data path: resolved relative to this file as ``CANDELS_CATALOG`` (repo
        root / analysis/hst_proposal/data/CANDELS_GDSS_workshop_z1.dat),
        overridable via the ``TENGRI_CANDELS_CATALOG`` environment variable.
        Missing data indicated by 98.999 or negative errors
        flg1: data quality flag (0 = good, 1 = issues)
    """
    data_path = Path(os.environ.get("TENGRI_CANDELS_CATALOG", CANDELS_CATALOG))

    if not data_path.exists():
        raise FileNotFoundError(
            f"CANDELS data not found at {data_path} (resolved from CANDELS_CATALOG="
            f"{CANDELS_CATALOG}; override with the TENGRI_CANDELS_CATALOG environment variable)"
        )

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
        "data": data,
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


#: AB zero point in erg s^-1 cm^-2 Hz^-1: 3631 Jy at 1e-23 erg s^-1 cm^-2 Hz^-1 per Jy.
#: The private copy this replaces read 3.63e-23 (3.63 Jy), 1000x too small; every
#: flux was 1000x too faint and every NUTS transition diverged (#2089).
AB_ZERO_POINT_ERG = 3.631e-20


def ab_mag_to_fnu(mag, mag_err):
    """AB magnitude and error to F_nu and its error, erg s^-1 cm^-2 Hz^-1.

    ``F_nu = F_0 10^(-mag / 2.5)`` with ``F_0 = AB_ZERO_POINT_ERG``; the error
    follows from ``dF / F = (ln 10 / 2.5) dm``. Elementwise on arrays.

    Args:
        mag: AB magnitude(s).
        mag_err: Magnitude error(s).

    Returns:
        (fnu, fnu_err) as float arrays (0-d for scalar input).
    """
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)
    fnu = AB_ZERO_POINT_ERG * 10.0 ** (-mag / 2.5)
    return fnu, fnu * (np.log(10.0) / 2.5) * mag_err


#: Catalog column -> tengri filter name; every value is in ``tengri.list_filters()``.
#: ISAAC_KS and HAWKI_KS use the VISTA Ks curve as a stand-in (the registry
#: carries no ISAAC and no broadband HAWK-I curve). CTIO_U and VIMOS_U are left
#: out on purpose: the registry has no CTIO or VIMOS U curve, and a curve from a
#: different telescope was not adopted for them. Order matters: the Ks columns
#: are taken in this order and the first detected one wins.
CANDELS_TO_TENGRI = {
    "ACS_F435W": "hst_f435w",
    "ACS_F606W": "hst_f606w",
    "ACS_F775W": "hst_f775w",
    "ACS_F814W": "hst_f814w",
    "ACS_F850LP": "hst_f850lp",
    "WFC3_F098M": "hst_f098m",
    "WFC3_F105W": "hst_f105w",
    "WFC3_F125W": "hst_f125w",
    "WFC3_F160W": "hst_f160w",
    "ISAAC_KS": "vista_ks",
    "HAWKI_KS": "vista_ks",
    "IRAC_CH1": "irac_36",
    "IRAC_CH2": "irac_45",
    "IRAC_CH3": "irac_58",
    "IRAC_CH4": "irac_80",
}

#: The catalog carries two Ks measurements; one band per galaxy, ISAAC first.
KS_COLUMNS = ("ISAAC_KS", "HAWKI_KS")


def photometry_for_row(
    header: list[str], row: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Detected bands of one catalog row as (tengri names, fnu, fnu_err).

    Args:
        header: Column names of the catalog, as returned by ``load_candels_z1``.
        row: One row of the data matrix.

    Returns:
        (names, fnu, fnu_err) in ``CANDELS_TO_TENGRI`` order; fluxes in
        erg s^-1 cm^-2 Hz^-1.

    Raises:
        KeyError: a mapped column, or its ``e``-prefixed error column, is not in
            ``header``. The map is checked against the file, never the other way
            around: a silent ``continue`` here dropped the five ACS bands (#2089).
    """
    names: list[str] = []
    fnu: list[float] = []
    fnu_err: list[float] = []
    ks_taken = False
    for column, tengri_name in CANDELS_TO_TENGRI.items():
        for needed in (column, f"e{column}"):
            if needed not in header:
                raise KeyError(f"catalog column {needed!r} is not in the header {header}")
        mag = float(row[header.index(column)])
        mag_err = float(row[header.index(f"e{column}")])
        if not is_detected(mag, mag_err):
            continue
        if column in KS_COLUMNS:
            if ks_taken:
                continue
            ks_taken = True
        flux, flux_err = ab_mag_to_fnu(mag, mag_err)
        names.append(tengri_name)
        fnu.append(float(flux))
        fnu_err.append(float(flux_err))
    return names, np.array(fnu), np.array(fnu_err)


if __name__ == "__main__":
    cat = load_candels_z1()
    print(f"Loaded {len(cat['id'])} galaxies")
    print(f"Redshift range: {cat['z'].min():.3f} - {cat['z'].max():.3f}")
    print(f"Photometric bands: {len(cat['bands'])}")
