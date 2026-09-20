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
#:
#: CTIO_U and VIMOS_U carry their own curves as of 2026-09-20 (CTIO/MosaicII.U
#: and Paranal/VIMOS.U, the two GOODS-S U-band sources). They were previously
#: omitted because the registry held neither, and substituting another
#: telescope's U was rightly declined -- but that dropped the two bluest
#: measurements in the catalog, which at z ~ 1 are the rest-frame ultraviolet
#: and so the ones carrying most of the young-star and attenuation information.
#:
#: Both are kept, unlike the Ks pair. They are genuinely different bandpasses
#: from different telescopes (MosaicII peaks at 3644 A over 3044-4139 A, VIMOS
#: at 3851 A over 3329-4004 A), so they are two independent measurements rather
#: than one measurement twice. ISAAC_KS and HAWKI_KS are the opposite case: both
#: resolve to the same VISTA Ks stand-in curve, so fitting both would enter one
#: response twice with correlated errors. Order matters for those two -- they
#: are taken in this order and the first detected one wins.
CANDELS_TO_TENGRI = {
    "CTIO_U": "ctio_u",
    "VIMOS_U": "vimos_u",
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


#: Rest wavelength of Lyman alpha [A]. Blueward of this the intergalactic medium
#: absorbs; redward of it, at these redshifts, it does not.
LYMAN_ALPHA_REST_A = 1216.0


def igm_break_margins(filter_names: list[str], z_max: float) -> list[tuple[str, float, float]]:
    """Per-band clearance between the observed Lyman break and the band's blue edge.

    The production ``WavePrecomp`` path folds IGM transmission at one wavelength
    per sub-band. That is adequate while the transmission is smooth across the
    band and wrong when a Lyman break falls *inside* it, which is the error the
    exact fold (``eq:igm_exact_fold``) removes.

    Whether that distinction matters is not a property of the survey, the
    configuration, or the redshift alone -- it is one inequality over the band
    set actually being fit:

        1216 * (1 + z_max)  >  blue edge of the bluest fitted band

    Stated as a threshold redshift it has to be restated every time the filter
    set or the sample changes, and both changed here (13 bands to 16 when the
    two U curves landed). Stated as the inequality it re-derives itself.

    Args:
        filter_names: tengri filter names actually entering the likelihood.
        z_max: Highest redshift in the sample.

    Returns:
        ``(name, blue_edge_A, z_at_which_the_break_reaches_it)`` per band,
        bluest first. A margin is negative when the break is already inside.
    """
    import tengri

    rows = []
    for f in tengri.Photometry.from_names(sorted(set(filter_names))).filters:
        wave = np.asarray(f.wave)
        trans = np.asarray(f.trans)
        blue_edge = float(wave[trans > 0].min())
        z_bite = blue_edge / LYMAN_ALPHA_REST_A - 1.0
        rows.append((f.name, blue_edge, z_bite))
    rows.sort(key=lambda r: r[1])
    return rows


def assert_igm_node_fold_adequate(filter_names: list[str], z_max: float) -> tuple[str, float]:
    """Raise if the Lyman break falls inside any fitted band at ``z_max``.

    Returns the bluest band and the redshift at which it would start to bite,
    so a caller can log the headroom rather than merely not crash.
    """
    rows = igm_break_margins(filter_names, z_max)
    name, blue_edge, z_bite = rows[0]
    if z_max >= z_bite:
        raise ValueError(
            f"IGM break is inside the fitted band set: at z_max={z_max:.4f} "
            f"Lyman alpha lands at {LYMAN_ALPHA_REST_A * (1 + z_max):.0f} A, "
            f"but {name} transmits from {blue_edge:.0f} A (bites at z={z_bite:.3f}). "
            f"The WavePrecomp node fold is no longer adequate; the exact fold "
            f"(eq:igm_exact_fold) is required before these fits mean anything."
        )
    return name, z_bite


if __name__ == "__main__":
    cat = load_candels_z1()
    print(f"Loaded {len(cat['id'])} galaxies")
    print(f"Redshift range: {cat['z'].min():.3f} - {cat['z'].max():.3f}")
    print(f"Photometric bands: {len(cat['bands'])}")

    z_max = float(cat["z"].max())
    print(
        f"\nIGM node-fold headroom at z_max={z_max:.4f} "
        f"(Lyman alpha at {LYMAN_ALPHA_REST_A * (1 + z_max):.0f} A):"
    )
    for name, blue_edge, z_bite in igm_break_margins(list(CANDELS_TO_TENGRI.values()), z_max)[:4]:
        print(f"  {name:<12} blue edge {blue_edge:7.0f} A   bites at z={z_bite:.3f}")
    bluest, z_bite = assert_igm_node_fold_adequate(list(CANDELS_TO_TENGRI.values()), z_max)
    print(f"[ok] node fold adequate; {bluest} sets the limit at z={z_bite:.3f}")
