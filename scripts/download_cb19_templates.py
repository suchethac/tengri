#!/usr/bin/env python3
"""Download CB_19 (Charlot & Bruzual 2019) CLOUDY grids from 3MdB_17.

Source
------
CB_19 photoionization grid:
  Martinez-Paredes et al. 2023, MNRAS, arXiv:2308.05604
  "On the origin of IR emission lines in star-forming galaxies"
  2,358,330 entries from CLOUDY c17.01 runs with C&B 2019 SSP/CSF SEDs.
  Stored in 3MdB_17, table tab_17, ref = 'CB_19'.

Status (as of a read-only probe on 2026-09-07, #2198)
------------------------------------------------------
This script currently cannot build the grid: querying ref = 'CB_19' returns
zero rows from 3MdB_17.tab_17 (and from every other database reachable with
these credentials -- 3MdB.tab and 3MdBs.projects -- under every near-miss
spelling tried). This is not a renamed or moved reference to chase: the 3MdB
project page for CB_19
(https://sites.google.com/site/mexicanmillionmodels/the-different-projects/cb_19)
carries a standing notice from the grid's own authors that there is a bug in
how chemical abundances and metallicities are defined, and that a new grid
and an erratum will be produced. That page also names the database as
'3MdB', not '3MdB_17' as queried here -- querying '3MdB.tab' directly finds
no CB_19 rows either. Run with --discover (or without) to see this reported
at the query, with a non-zero exit; the script is left in place because it
is the intended route once 3MdB republishes the grid, not because it works
today. Until then, build a nebular model with neb={'type': 'cue'},
neb={'type': 'cloudy', 'grid': <path>}, or neb={'type': 'ssp'} with a wNE
SSP grid -- see docs/internal/advanced/cb19_grid.md for the full status.

Database access (public credentials, same server as 3MdBs MAPPINGS grids)
---------------------------------------------------------------------------
  host:   3mdb.astro.unam.mx   (alias for 132.248.3.52)
  port:   3306
  user:   OVN_user
  passwd: oiii5007
  db:     3MdB_17              (different from 3MdBs used by MAPPINGS)
  table:  tab_17
  filter: ref = 'CB_19'

Grid axes (Table 2 of Martinez-Paredes+2023)
--------------------------------------------
  SED type  : SSP (41 ages) | CSF (24 ages)
  IMF       : Kroupa01 | top-heavy x030
  M_up      : 100 | 300 M_sun
  log_age   : 6–10 (41 SSP values; 24 CSF values)
  log U     : −4, −3.5, −3, −2.5, −2, −1.5   (6 values)
  log n_H   : 1, 2, 3, 4                       (4 values, cm⁻³)
  log(O/H)  : −5.06, −4.06, −3.45, −3.20, −3.12, −2.86, −2.58  (7 values)
  C/O (log) : −1, −0.36, 0.15                  (3 values)
  ΔN/O      : −0.25, 0, 0.25                   (3 values)
  HbFrac    : radiation-bounded (1.0) + 4 matter-bounded cuts    (6 values)

Unit convention
---------------
CB_19 stores all line fluxes as ratios relative to Hβ (dimensionless).
The CB19Backend converts to L_sun / Q_H at load time using:

    L_line / Q_H = ratio × (L_Hβ / Q_H)

where  L_Hβ / Q_H ≈ 4.78×10⁻¹³ erg photon⁻¹  (Case B recombination,
T_e = 10^4 K, Osterbrock & Ferland 2006, Table 4.4; also eq. 1 of
Byler et al. 2017 ApJ 840 44).  Converting to L_sun / (photon s⁻¹):

    L_Hβ / Q_H = 4.78×10⁻¹³ / 3.828×10³³ L_sun s
               ≈ 1.249×10⁻⁴⁶  L_sun s

This brings CB_19 into the same L_sun/Q_H units used by the FSPS CLOUDY
grids loaded by CloudyGridBackend.

Workflow
--------
1. Discover actual column names (REQUIRED before first download):
       python scripts/download_cb19_templates.py --discover

2. Full download (~20–60 min depending on network):
       python scripts/download_cb19_templates.py --output data/cb19_templates.h5

3. Test with small subset:
       python scripts/download_cb19_templates.py --limit 5000 --output /tmp/cb19_test.h5

After discovery, update _PARAM_COLS and _LINE_MAP if the actual column names
differ from the guesses below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Database constants (same server as MAPPINGS grids, different database)
# ---------------------------------------------------------------------------
_DB_HOST = "3mdb.astro.unam.mx"
_DB_PORT = 3306
_DB_USER = "OVN_user"
_DB_PASSWD = "oiii5007"
_DB_NAME = "3MdB_17"
_DB_TABLE = "tab_17"
_REF = "CB_19"

# ---------------------------------------------------------------------------
# Parameter column names in tab_17
# ---------------------------------------------------------------------------
# IMPORTANT: Run --discover first to verify these against the actual schema.
# Names below follow common 3MdBs naming conventions; they may differ slightly.
#
# Mapping: semantic name → likely DB column name
_PARAM_COLS: dict[str, str] = {
    "ref": "ref",
    "sed_type": "SFH",  # 'SSP' or 'CSF'; may be 'type', 'SEDtype', 'SED_type'
    "imf": "IMF",  # 'Kroupa01' or 'x030'
    "mup": "Mup",  # upper mass limit: 100 or 300 (M_sun); may be 'M_up'
    "log_age": "log_age",  # log10(age/yr); may be 'age' (not log), 'log_t'
    "log_u": "logU",  # log10(U); may be 'log_U', 'LogU'
    "log_nh": "log_nH",  # log10(n_H/cm⁻³); may be 'Hden' (linear!), 'log_dens'
    "log_oh": "log_OH",  # log10(O/H)_total; may be 'log_OHtot', 'logOH', 'abund'
    "log_co": "log_CO",  # log10(C/O); may be 'log_COtot'
    "dno": "log_NO",  # ΔN/O or log(N/O); may be 'dNO', 'NO', 'log_NO'
    "hbfrac": "HbFrac",  # Hβ fraction [0–1]; may be 'hbfrac', 'Hbfrac'
}

# ---------------------------------------------------------------------------
# Line map: (our_name, vacuum_wavelength_aa, db_column)
# ---------------------------------------------------------------------------
# Column names follow typical 3MdBs conventions for emission lines in flat tables.
# Doublets appear on two lines below (sum them via the query or separately).
# IMPORTANT: Verify with --discover; IR line columns especially are uncertain.
#
# References for vacuum wavelengths: NIST ASD, line_catalog.py (tengri)
_LINE_MAP: list[tuple[str, float, str]] = [
    # UV
    ("LyA_1216A", 1215.67, "HI_1216"),
    ("CIV_1549A", 1549.48, "CIV_1549"),
    ("HeII_1640A", 1640.42, "HeII_1640"),
    ("CIII_1909A", 1908.73, "CIII_1909"),
    ("MgII_2796A", 2796.35, "MgII_2796"),
    ("MgII_2803A", 2803.53, "MgII_2803"),
    # Optical
    ("NeIII_3869A", 3869.06, "NeIII_3869"),
    ("OII_3726A", 3726.03, "OII_3726"),
    ("OII_3729A", 3728.82, "OII_3729"),
    ("Hg_4340A", 4340.47, "HI_4340"),
    ("HeII_4686A", 4685.68, "HeII_4686"),
    ("Hb_4861A", 4862.68, "HI_4861"),  # reference line (ratio = 1.0)
    ("O3_4959A", 4960.30, "OIII_4959"),
    ("O3_5007A", 5008.24, "OIII_5007"),
    ("O3_4363A", 4364.44, "OIII_4363"),  # auroral line
    ("HeI_5876A", 5877.25, "HeI_5876"),
    ("OI_6300A", 6302.04, "OI_6300"),
    ("NII_6548A", 6549.86, "NII_6548"),
    ("Ha_6563A", 6564.61, "HI_6563"),
    ("NII_6583A", 6585.27, "NII_6583"),
    ("SII_6716A", 6718.29, "SII_6716"),
    ("SII_6731A", 6732.67, "SII_6731"),
    ("ArIII_7135A", 7137.77, "ArIII_7136"),
    ("NII_5755A", 5756.19, "NII_5755"),  # auroral line
    ("OII_7320A", 7321.99, "OII_7320"),
    ("OII_7330A", 7332.21, "OII_7330"),
    # Near-IR
    ("SIII_9069A", 9071.11, "SIII_9069"),
    ("SIII_9532A", 9533.23, "SIII_9531"),
    # Mid-IR (verify column names with --discover; wavelengths in Å)
    # [NeII] 12.81 μm = 128130 Å; [NeIII] 15.56 μm = 155600 Å
    # [OIV] 25.89 μm = 258870 Å; [OIII] 88.3 μm = 882800 Å
    ("NeII_128100A", 128130.0, "NeII_128100"),
    ("NeIII_155600A", 155600.0, "NeIII_155600"),
    ("OIV_258900A", 258870.0, "OIV_258900"),
    ("OIII_882800A", 882800.0, "OIII_882800"),
]

# Reference Hβ column (ratio denominator — must be in _LINE_MAP)
_HB_COL = "HI_4861"
_LINE_NAMES = [e[0] for e in _LINE_MAP]
_LINE_WAVES = np.array([e[1] for e in _LINE_MAP], dtype=np.float32)
_N_LINES = len(_LINE_NAMES)

# Known discrete axis values (from Table 2 of Martinez-Paredes+2023)
# These are used to label the HDF5 axes; actual values verified from DB.
_KNOWN_SED_TYPES = ["SSP", "CSF"]
_KNOWN_IMFS = ["Kroupa01", "x030"]
_KNOWN_MUPS = [100.0, 300.0]
_KNOWN_LOG_U = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5], dtype=np.float32)
_KNOWN_LOG_NH = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
_KNOWN_LOG_OH = np.array([-5.06, -4.06, -3.45, -3.20, -3.12, -2.86, -2.58], dtype=np.float32)
_KNOWN_LOG_CO = np.array([-1.0, -0.36, 0.15], dtype=np.float32)
_KNOWN_DNO = np.array([-0.25, 0.0, 0.25], dtype=np.float32)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _connect() -> "pymysql.Connection":  # type: ignore[name-defined]
    import pymysql

    return pymysql.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        user=_DB_USER,
        passwd=_DB_PASSWD,
        db=_DB_NAME,
        connect_timeout=60,
        read_timeout=600,
        cursorclass=pymysql.cursors.DictCursor,
    )


#: 3MdB project page for CB_19, which carries the upstream errata notice.
_CB19_PROJECT_PAGE = (
    "https://sites.google.com/site/mexicanmillionmodels/the-different-projects/cb_19"
)

#: Printed (and raised) when ``ref = 'CB_19'`` returns no rows (#2198). This is
#: not a renamed or moved reference to chase: a read-only probe on 2026-09-07
#: found zero CB_19-like rows under every reachable database (3MdB_17.tab_17,
#: 3MdB.tab, 3MdBs.projects) and every near-miss spelling tried, and the
#: project page above -- which names the database as '3MdB', not '3MdB_17' as
#: queried here -- carries a standing notice from the grid's own authors that
#: there is a bug in how chemical abundances and metallicities are defined,
#: and that a new grid and an erratum will be produced.
_NO_ROWS_MESSAGE = (
    f"No rows found with ref='{_REF}' in {_DB_NAME}.{_DB_TABLE}. As of a "
    "read-only probe on 2026-09-07, this is not a column-name or ref-spelling "
    "problem: the same query against 3MdB, 3MdB_17 and 3MdBs (every database "
    "reachable with these credentials) returns zero rows for 'CB_19' and "
    "every near-miss spelling tried (CB19, CB_2019, Bruzual, Charlot). The "
    f"3MdB project page for CB_19 ({_CB19_PROJECT_PAGE}) carries a standing "
    "notice: chemical abundances and metallicities are defined incorrectly "
    "in the grid, and the 3MdB team will produce a new grid and an erratum. "
    "That page also names the database as '3MdB', not '3MdB_17' as queried "
    "here. There is currently no route to build data/cb19_templates.h5 from "
    "3MdB; this script will work again once the 3MdB team republishes the "
    "grid. See docs/internal/advanced/cb19_grid.md for the full status."
)


def _ref_row_count(co) -> int:
    """Count rows with ``ref = _REF`` in ``_DB_TABLE`` (#2198).

    A dedicated, cheap query run before the expensive axis discovery and full
    fetch, so a ``ref`` that has gone missing upstream is reported clearly
    rather than crashing deep inside :func:`_query_axes` on an empty-sequence
    ``max()``.
    """
    cur = co.cursor()
    cur.execute(f"SELECT COUNT(*) AS n FROM {_DB_TABLE} WHERE ref = %s", (_REF,))
    row = cur.fetchone()
    return int(row["n"]) if row else 0


def _discover_schema(co) -> None:
    """Print all column names for CB_19 models in tab_17.

    Run this before the first full download to verify _PARAM_COLS and _LINE_MAP.
    """
    cur = co.cursor()
    sql = f"SELECT * FROM {_DB_TABLE} WHERE ref = %s LIMIT 1"
    cur.execute(sql, (_REF,))
    row = cur.fetchone()

    if row is None:
        print(f"ERROR: No rows found with ref='{_REF}' in {_DB_TABLE}")
        return

    print(f"\n=== Column names in {_DB_TABLE} (ref='{_REF}') ===")
    print(f"Total columns: {len(row)}\n")

    param_like = []
    line_like = []
    other = []

    for col in row:
        val = row[col]
        col_lower = col.lower()
        # Heuristic: line columns have element + wavelength pattern
        if any(
            col.startswith(p)
            for p in (
                "HI_",
                "HeI_",
                "HeII_",
                "OI_",
                "OII_",
                "OIII_",
                "NI_",
                "NII_",
                "NIII_",
                "SII_",
                "SIII_",
                "NeII_",
                "NeIII_",
                "NeV_",
                "ArII_",
                "ArIII_",
                "ArIV_",
                "CII_",
                "CIII_",
                "CIV_",
                "MgI_",
                "MgII_",
                "SiII_",
                "SiIII_",
                "FeII_",
                "FeIII_",
                "CI_",
                "OIV_",
            )
        ):
            line_like.append((col, val))
        elif (
            col_lower
            in (
                "ref",
                "logU",
                "log_u",
                "hden",
                "log_nh",
                "log_nH",
                "log_age",
                "age",
                "sfh",
                "imf",
                "mup",
                "m_up",
                "log_oh",
                "log_co",
                "log_no",
                "dno",
                "hbfrac",
                "log_OHtot",
                "log_COtot",
                "log_NOtot",
                "modelid",
                "model_id",
                "log_t",
                "type",
                "sfh_type",
                "sedtype",
                "sed_type",
            )
            or col_lower == col_lower
        ):
            param_like.append((col, val))
        else:
            other.append((col, val))

    print("--- Likely parameter columns ---")
    for col, val in sorted(param_like):
        print(f"  {col:30s} = {val!r}")

    print("\n--- Likely emission line columns ---")
    for col, val in sorted(line_like):
        print(f"  {col:30s} = {val!r}")

    if other:
        print("\n--- Other columns ---")
        for col, val in sorted(other):
            print(f"  {col:30s} = {val!r}")

    print("\n--- All column names (sorted) ---")
    for col in sorted(row.keys()):
        print(f"  {col}")

    print(
        "\nUpdate _PARAM_COLS and _LINE_MAP at the top of this script to match "
        "the actual column names, then re-run without --discover."
    )


# ---------------------------------------------------------------------------
# Axis discovery
# ---------------------------------------------------------------------------


def _query_axes(co) -> dict[str, np.ndarray | list[str]]:
    """Query DISTINCT axis values for CB_19 grid."""
    cur = co.cursor()
    p = _PARAM_COLS

    def _distinct(col: str) -> list:
        cur.execute(
            f"SELECT DISTINCT `{col}` FROM {_DB_TABLE} WHERE ref=%s ORDER BY `{col}`",
            (_REF,),
        )
        return [r[col] for r in cur.fetchall()]

    print("  Querying distinct axis values (may take 1–2 min for 2.3M rows)...")

    sed_types = _distinct(p["sed_type"])
    imfs = _distinct(p["imf"])
    mups = [float(x) for x in _distinct(p["mup"])]

    log_u_raw = [float(x) for x in _distinct(p["log_u"])]
    log_oh_raw = [float(x) for x in _distinct(p["log_oh"])]
    log_co_raw = [float(x) for x in _distinct(p["log_co"])]
    dno_raw = [float(x) for x in _distinct(p["dno"])]
    hbfrac_raw = [float(x) for x in _distinct(p["hbfrac"])]

    # log_nh: may be stored as linear Hden — detect and convert
    nh_raw = [float(x) for x in _distinct(p["log_nh"])]
    if max(nh_raw) > 10:
        # Stored as linear cm⁻³ (e.g. 10, 100, 1000, 10000) — convert to log
        log_nh = np.sort(np.log10(np.array(nh_raw, dtype=np.float32)))
        print(f"    log_nH (converted from linear): {log_nh.tolist()}")
    else:
        log_nh = np.sort(np.array(nh_raw, dtype=np.float32))
        print(f"    log_nH (already log): {log_nh.tolist()}")

    # log_age: may be stored as linear age in yr — detect and convert
    age_ssp_raw: list[float] = []
    age_csf_raw: list[float] = []
    cur.execute(
        f"SELECT DISTINCT `{p['sed_type']}`, `{p['log_age']}` FROM {_DB_TABLE} "
        f"WHERE ref=%s ORDER BY `{p['sed_type']}`, `{p['log_age']}`",
        (_REF,),
    )
    for r in cur.fetchall():
        age_val = float(r[p["log_age"]])
        if age_val > 20:
            age_val = np.log10(age_val)  # stored as linear age
        if r[p["sed_type"]] == "SSP":
            age_ssp_raw.append(age_val)
        else:
            age_csf_raw.append(age_val)

    axes = {
        "sed_types": sed_types,
        "imfs": imfs,
        "mups": np.array(sorted(mups), dtype=np.float32),
        "log_U": np.array(sorted(log_u_raw), dtype=np.float32),
        "log_nH": log_nh,
        "log_OH": np.array(sorted(log_oh_raw), dtype=np.float32),
        "log_CO": np.array(sorted(log_co_raw), dtype=np.float32),
        "dNO": np.array(sorted(dno_raw), dtype=np.float32),
        "HbFrac": np.array(sorted(hbfrac_raw), dtype=np.float32),
        "log_age_ssp": np.array(sorted(set(age_ssp_raw)), dtype=np.float32),
        "log_age_csf": np.array(sorted(set(age_csf_raw)), dtype=np.float32),
    }

    print(f"    SED types : {axes['sed_types']}")
    print(f"    IMFs      : {axes['imfs']}")
    print(f"    M_up      : {axes['mups'].tolist()}")
    print(f"    log U     : {axes['log_U'].tolist()}")
    print(f"    log n_H   : {axes['log_nH'].tolist()}")
    print(f"    log (O/H) : {axes['log_OH'].tolist()}")
    print(f"    log (C/O) : {axes['log_CO'].tolist()}")
    print(f"    ΔN/O      : {axes['dNO'].tolist()}")
    print(f"    HbFrac    : {axes['HbFrac'].tolist()}")
    print(f"    log_age (SSP): {len(axes['log_age_ssp'])} values")
    print(f"    log_age (CSF): {len(axes['log_age_csf'])} values")

    return axes


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

_CHUNK = 20_000  # rows per fetch


def _fetch_cb19(
    co,
    axes: dict,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Download all CB_19 rows and reshape into dense arrays.

    Returns a dict mapping '{sed_type}/{imf}/{mup}' → array of shape
    (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_HbFrac, N_lines).

    Missing grid combinations are NaN-filled.
    """
    p = _PARAM_COLS

    # Build index maps for fast lookup
    def _idx_map(arr: np.ndarray, tol: float = 1e-4) -> dict[float, int]:
        return {float(v): i for i, v in enumerate(arr)}

    oh_idx = _idx_map(axes["log_OH"])
    u_idx = _idx_map(axes["log_U"])
    co_idx = _idx_map(axes["log_CO"])
    dno_idx = _idx_map(axes["dNO"])
    hb_idx = _idx_map(axes["HbFrac"])

    def _nh_idx(val: float) -> int:
        """Nearest-neighbour snap for log_nH."""
        return int(np.argmin(np.abs(axes["log_nH"] - val)))

    # Collect line column names (filter to those actually in the DB)
    # We'll verify which columns exist from the first row
    line_db_cols = [e[2] for e in _LINE_MAP]
    existing_line_cols: list[int] | None = None  # determined on first row

    # Separate age arrays per SED type
    age_arrays = {
        "SSP": axes["log_age_ssp"],
        "CSF": axes["log_age_csf"],
    }

    def _make_age_idx_map(age_arr: np.ndarray) -> dict[float, int]:
        # Round to 4dp to handle float precision issues
        return {round(float(v), 4): i for i, v in enumerate(age_arr)}

    age_idx_maps = {st: _make_age_idx_map(age_arrays[st]) for st in axes["sed_types"]}

    # Allocate output arrays: keyed by (sed_type, imf, mup_str)
    def _make_key(sed: str, imf: str, mup: float) -> str:
        return f"{sed}/{imf}/mu{int(mup)}"

    def _make_arr(sed: str) -> np.ndarray:
        n_age = len(age_arrays[sed])
        return np.full(
            (
                len(axes["log_OH"]),  # 0: metallicity
                n_age,  # 1: age
                len(axes["log_U"]),  # 2: ionization
                len(axes["log_nH"]),  # 3: density
                len(axes["log_CO"]),  # 4: C/O
                len(axes["dNO"]),  # 5: ΔN/O
                len(axes["HbFrac"]),  # 6: matter-bounded
                _N_LINES,  # 7: lines
            ),
            np.nan,
            dtype=np.float32,
        )

    grids: dict[str, np.ndarray] = {}
    for sed in axes["sed_types"]:
        for imf in axes["imfs"]:
            for mup in axes["mups"]:
                key = _make_key(sed, imf, float(mup))
                grids[key] = _make_arr(sed)

    # Build SELECT query
    param_sel = ", ".join(f"`{v}` AS {k}" for k, v in _PARAM_COLS.items() if k != "ref")
    line_sel = ", ".join(f"`{c}`" for c in line_db_cols)
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    sql = (
        f"SELECT {param_sel}, {line_sel} "
        f"FROM {_DB_TABLE} "
        f"WHERE ref=%s "
        f"ORDER BY `{p['sed_type']}`, `{p['imf']}`, `{p['mup']}`, "
        f"`{p['log_age']}`, `{p['log_u']}`, `{p['log_nh']}`, "
        f"`{p['log_oh']}`, `{p['log_co']}`, `{p['dno']}`, `{p['hbfrac']}` "
        f"{limit_clause}"
    )

    print(f"\nFetching CB_19 rows in chunks of {_CHUNK} ...")
    cur = co.cursor()
    cur.execute(sql, (_REF,))

    n_parsed = n_skip = n_total = 0

    while True:
        rows = cur.fetchmany(_CHUNK)
        if not rows:
            break

        for row in rows:
            n_total += 1

            # --- Decode parameters ---
            try:
                sed = str(row["sed_type"]).strip()
                imf = str(row["imf"]).strip()
                mup = float(row["mup"])
                key = _make_key(sed, imf, mup)

                if key not in grids:
                    n_skip += 1
                    continue

                # Age (handle both log and linear storage)
                age_raw = float(row["log_age"])
                if age_raw > 20:
                    age_raw = np.log10(age_raw)
                age_r = round(age_raw, 4)
                i_age = age_idx_maps[sed].get(age_r)
                if i_age is None:
                    # Nearest-neighbour fallback
                    diffs = np.abs(age_arrays[sed] - age_raw)
                    if diffs.min() < 0.02:
                        i_age = int(np.argmin(diffs))
                    else:
                        n_skip += 1
                        continue

                log_u = float(row["log_u"])
                i_u = u_idx.get(round(log_u, 2))
                if i_u is None:
                    i_u = int(np.argmin(np.abs(axes["log_U"] - log_u)))

                nh_raw = float(row["log_nh"])
                if nh_raw > 10:
                    nh_raw = np.log10(nh_raw)
                i_nh = _nh_idx(nh_raw)

                log_oh = float(row["log_oh"])
                i_oh = oh_idx.get(round(log_oh, 4))
                if i_oh is None:
                    i_oh = int(np.argmin(np.abs(axes["log_OH"] - log_oh)))

                log_co = float(row["log_co"])
                i_co = co_idx.get(round(log_co, 4))
                if i_co is None:
                    i_co = int(np.argmin(np.abs(axes["log_CO"] - log_co)))

                dno = float(row["dno"])
                i_dno = dno_idx.get(round(dno, 4))
                if i_dno is None:
                    i_dno = int(np.argmin(np.abs(axes["dNO"] - dno)))

                hbfrac = float(row["hbfrac"])
                i_hb = hb_idx.get(round(hbfrac, 4))
                if i_hb is None:
                    i_hb = int(np.argmin(np.abs(axes["HbFrac"] - hbfrac)))

                hb_flux = row.get(_HB_COL)
                if hb_flux is None or float(hb_flux) == 0.0:
                    n_skip += 1
                    continue
                hb_ref = float(hb_flux)

                # Determine which line columns actually exist (first valid row)
                if existing_line_cols is None:
                    existing_line_cols = [li for li, col in enumerate(line_db_cols) if col in row]
                    missing = [
                        _LINE_MAP[li][2] for li in range(_N_LINES) if li not in existing_line_cols
                    ]
                    if missing:
                        print(
                            f"\n  WARNING: {len(missing)} line columns not found in DB "
                            f"(will be NaN):\n    {missing[:10]}"
                            + (" ..." if len(missing) > 10 else "")
                        )

                # Fill line ratios
                for li in existing_line_cols:
                    col = line_db_cols[li]
                    val = row.get(col)
                    if val is not None:
                        grids[key][i_oh, i_age, i_u, i_nh, i_co, i_dno, i_hb, li] = (
                            float(val) / hb_ref
                        )
                # Hβ itself = 1.0 by definition
                hb_li = next((li for li, e in enumerate(_LINE_MAP) if e[2] == _HB_COL), None)
                if hb_li is not None and hb_li in existing_line_cols:
                    grids[key][i_oh, i_age, i_u, i_nh, i_co, i_dno, i_hb, hb_li] = 1.0

                n_parsed += 1

            except (KeyError, TypeError, ValueError):
                n_skip += 1
                continue

        if n_total % 100_000 == 0:
            print(f"    {n_total:,} rows processed ({n_parsed:,} ok, {n_skip:,} skip)")

    print(f"\n  Total: {n_total:,} rows | {n_parsed:,} parsed | {n_skip:,} skipped")
    return grids


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

# Case B L_Hβ / Q_H in Lsun per (photon/s) = Lsun·s/photon
# Source: Osterbrock & Ferland 2006, Table 4.4; T_e=10^4 K, n_e=10^2 cm⁻³
# See also Byler+2017 (ApJ 840 44), eq.1 and conversion factor.
# L_Hβ/Q_H = 4.78e-13 erg/photon / 3.828e33 erg/Lsun
_HB_PER_QH_LSUN = 4.78e-13 / 3.828e33  # Lsun per (photon s⁻¹) = Lsun s / photon


def _write_hdf5(
    out_path: Path,
    axes: dict,
    grids: dict[str, np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(compression="gzip", compression_opts=6)

    with h5py.File(out_path, "w") as f:
        # Root metadata
        f.attrs["description"] = (
            "CB_19 CLOUDY photoionization grid for tengri. "
            "Source: Martinez-Paredes et al. 2023 (arXiv:2308.05604). "
            "Models from Charlot & Bruzual 2019 SSP/CSF ionizing SEDs, "
            "run with CLOUDY c17.01 via pyCloudy; stored in 3MdB_17 (ref='CB_19'). "
            "Line fluxes stored as ratios relative to Hβ. "
            "Convert to L_sun/Q_H: multiply by _HB_PER_QH_LSUN = 4.78e-13/3.828e33 "
            "≈ 1.249e-46 L_sun s (Case B, T_e=10^4 K; Osterbrock & Ferland 2006 Table 4.4)."
        )
        f.attrs["source"] = "Martinez-Paredes et al. 2023, MNRAS, arXiv:2308.05604"
        f.attrs["cloudy_version"] = "c17.01"
        f.attrs["ssp_library"] = "Charlot & Bruzual 2019"
        f.attrs["db_host"] = _DB_HOST
        f.attrs["db_name"] = _DB_NAME
        f.attrs["db_table"] = _DB_TABLE
        f.attrs["ref"] = _REF
        f.attrs["hb_per_qh_lsun"] = _HB_PER_QH_LSUN
        f.attrs["hb_per_qh_description"] = (
            "L_Hb/Q_H in Lsun per (photon/s). "
            "Multiply stored ratios by this to get L_line/Q_H in Lsun/(photon/s). "
            "Case B recombination, T_e=10^4 K (Osterbrock & Ferland 2006, Table 4.4)."
        )
        f.attrs["units_line_ratios"] = (
            "dimensionless (ratio relative to Hβ). "
            "For absolute units L_sun/Q_H, multiply by hb_per_qh_lsun."
        )

        # Global axes (shared across SED/IMF/Mup groups)
        ax = f.create_group("axes")
        ax.create_dataset("log_U", data=axes["log_U"], **opts)
        ax["log_U"].attrs["description"] = "log10(ionization parameter U)"
        ax.create_dataset("log_nH", data=axes["log_nH"], **opts)
        ax["log_nH"].attrs["description"] = "log10(n_H / cm⁻³)"
        ax.create_dataset("log_OH_total", data=axes["log_OH"], **opts)
        ax["log_OH_total"].attrs["description"] = (
            "log10(O/H)_total — total (gas + dust) oxygen abundance. "
            "Solar reference: 12+log(O/H)_sun ≈ 8.93 → log(O/H)_sun ≈ −3.07. "
            "Convert to log(Z/Zsun): log(Z/Zsun) ≈ log_OH_total − log_OH_solar."
        )
        ax.create_dataset("log_CO", data=axes["log_CO"], **opts)
        ax["log_CO"].attrs["description"] = "log10(C/O) abundance ratio"
        ax.create_dataset("dNO", data=axes["dNO"], **opts)
        ax["dNO"].attrs["description"] = (
            "ΔN/O — offset in log10(N/O) from the default scaling with O/H "
            "(Nicholls et al. 2017 prescription)"
        )
        ax.create_dataset("HbFrac", data=axes["HbFrac"], **opts)
        ax["HbFrac"].attrs["description"] = (
            "H-beta fraction: ratio of Hβ luminosity of truncated (matter-bounded) "
            "model to radiation-bounded model. HbFrac=1.0 = radiation-bounded; "
            "HbFrac<1 = matter-bounded (ionizing photons escape). "
            "Escape fraction ≈ 1 − HbFrac."
        )
        ax.create_dataset("log_age_yr_ssp", data=axes["log_age_ssp"], **opts)
        ax["log_age_yr_ssp"].attrs["description"] = "log10(age/yr) for SSP models"
        ax.create_dataset("log_age_yr_csf", data=axes["log_age_csf"], **opts)
        ax["log_age_yr_csf"].attrs["description"] = "log10(age/yr) for CSF (constant SFR) models"

        # Line catalog
        dt_str = h5py.special_dtype(vlen=str)
        f.create_dataset("line_names", data=np.array(_LINE_NAMES, dtype="S40"), **opts)
        f.create_dataset("line_wavelengths_aa", data=_LINE_WAVES, **opts)
        f["line_wavelengths_aa"].attrs["units"] = "Angstrom (vacuum)"
        f["line_wavelengths_aa"].attrs["description"] = (
            "Rest-frame vacuum wavelengths matching line_names order"
        )

        # Grid data per (SED type, IMF, M_up) combination
        n_valid_total = 0
        for key, arr in sorted(grids.items()):
            grp = f.create_group(f"grids/{key}")
            parts = key.split("/")
            grp.attrs["sed_type"] = parts[0]  # SSP or CSF
            grp.attrs["imf"] = parts[1]
            grp.attrs["mup_msun"] = float(parts[2].lstrip("mu"))
            grp.attrs["array_axes"] = (
                "0:log_OH_total, 1:log_age_yr, 2:log_U, 3:log_nH, "
                "4:log_CO, 5:dNO, 6:HbFrac, 7:line_index"
            )
            grp.attrs["log_age_key"] = "log_age_yr_ssp" if parts[0] == "SSP" else "log_age_yr_csf"
            ds = grp.create_dataset("line_ratios", data=arr, **opts)
            ds.attrs["units"] = "dimensionless (ratio relative to Hβ)"
            ds.attrs["description"] = (
                "Emission line flux ratios relative to Hβ. "
                "Shape: (N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_HbFrac, N_lines). "
                "NaN where the grid combination was not computed or CLOUDY failed."
            )
            n_valid = int(np.sum(~np.isnan(arr)))
            n_valid_total += n_valid
            n_cells = int(np.prod(arr.shape[:-1]))
            pct = 100 * n_valid / max(1, n_cells * _N_LINES)
            print(f"  grids/{key}: shape={arr.shape}, filled={pct:.1f}%")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB)")
    print(f"Total valid entries: {n_valid_total:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Print all column names for CB_19 rows, then exit (run this first)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "data" / "cb19_templates.h5"),
        help="Output HDF5 path (default: data/cb19_templates.h5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to download (for testing; default: all ~2.3M)",
    )
    args = parser.parse_args()

    try:
        import pymysql  # noqa: F401
    except ImportError:
        print("ERROR: pymysql not installed. Run: pip install pymysql")
        sys.exit(1)

    print(f"\n=== CB_19 CLOUDY grids (Martinez-Paredes+2023) ===")
    print(f"  Database : {_DB_HOST}:{_DB_PORT}/{_DB_NAME}")
    print(f"  Table    : {_DB_TABLE}  (ref='{_REF}')")
    if not args.discover:
        print(f"  Output   : {args.output}")
    if args.limit:
        print(f"  Limit    : {args.limit:,} rows (test mode)")

    print("\nConnecting ... ", end="", flush=True)
    try:
        co = _connect()
        print("OK")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"Checking ref='{_REF}' is populated ... ", end="", flush=True)
    n_rows = _ref_row_count(co)
    print(f"{n_rows:,} rows")
    if n_rows == 0:
        print(f"\nERROR: {_NO_ROWS_MESSAGE}")
        co.close()
        sys.exit(1)

    if args.discover:
        _discover_schema(co)
        co.close()
        return

    print("\nDiscovering grid axes ...")
    axes = _query_axes(co)

    print("\nDownloading grid data ...")
    grids = _fetch_cb19(co, axes, limit=args.limit)
    co.close()

    out_path = Path(args.output)
    print(f"\n=== Writing {out_path} ===")
    _write_hdf5(out_path, axes, grids)

    print("\nDone. Verify with:")
    print(
        '  python -c "'
        "from tengri.nebular.cloudy_cb19 import CB19Backend; "
        "b = CB19Backend('data/cb19_templates.h5'); "
        "print(b.line_names[:5])"
        '"'
    )


if __name__ == "__main__":
    main()
