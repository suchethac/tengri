#!/usr/bin/env python3
"""Download and preprocess the 3MdBs MAPPINGS V shock model grids for tengri.

Source
------
3MdBs — Extensive Online Shock SEDModel Database
  Alarie & Morisset 2019, RMxAA, 55, 377-394
  arXiv:1908.08579 | doi:10.22201/ia.01851101p.2019.55.02.21
  Database portal: http://3mdb.astro.unam.mx

MAPPINGS V shock code:
  Sutherland & Dopita 2017, ApJS, 229, 34; arXiv:1702.07453
  https://mappings.anu.edu.au/code/

Database access (public credentials)
--------------------------------------
  host:   3mdb.astro.unam.mx
  port:   3306
  user:   OVN_user
  passwd: oiii5007
  db:     3MdBs

This script queries the live MySQL database directly — no Zenodo download needed.

Project used
------------
Project 1 (ref='Allen08'): 3992 models × 3 component types
  - 5 abundances: Allen2008_Solar, Allen2008_SMC, Allen2008_LMC,
                  Allen2008_Dopita2005, Allen2008_TwiceSolar
  - 6 densities (Solar only): 0.01, 0.1, 1, 10, 100, 1000 cm⁻³
  - Non-solar abundances: n=1 cm⁻³ only
  - 37 velocities: 100–1000 km/s in 25 km/s steps
  - B-fields: up to 35 values (0.0001–1000 μG); n=1 has 8 values matching
    the original Allen+2008 set (0.0001, 0.5, 1, 2, 3.23, 4, 5, 10 μG)
  - Components: shock, precursor, shock_plus_precursor

Output
------
Writes ``data/mappings_templates.h5`` with a single top-level group::

    mappings_templates.h5
    └── mappings5/
        ├── velocities_kms          (N_v=37,)
        ├── b_field_uG              (N_b,)  union of all B values
        ├── log_density_cm3         (N_n=6,)
        ├── abundance_names         (N_a=5,)
        ├── line_names              (N_lines,)
        ├── line_wavelengths_aa     (N_lines,)
        ├── shock_ratios            (N_a, N_n, N_v, N_b, N_lines)  relative to Hβ
        ├── precursor_ratios        same shape
        ├── combined_ratios         same shape
        └── hbeta_log_lum_erg_s     not yet populated (NULL in DB); shape filled with NaN

    NaN where a (abundance, density, velocity, B) combination was not
    computed (e.g. non-solar abundances only exist at n=1 cm⁻³).

Usage
-----
    python scripts/download_mappings_templates.py
    python scripts/download_mappings_templates.py --output data/custom.h5
    python scripts/download_mappings_templates.py --project 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Database constants (public credentials from 3MdBs Google Group)
# ---------------------------------------------------------------------------
# TODO(release): These credentials are publicly posted on the 3MdBs Google Group
# (groups.google.com/g/3mdbsupport) and are intentionally public for read-only
# access to the shock database. Before public release of tengri, verify that
# they are still intended to be public, or move them to a config file /
# environment variable (e.g. TENGRI_3MDBS_USER / TENGRI_3MDBS_PASSWD).
# ---------------------------------------------------------------------------

_DB_HOST = "3mdb.astro.unam.mx"
_DB_PORT = 3306
_DB_USER = "OVN_user"
_DB_PASSWD = "oiii5007"
_DB_NAME = "3MdBs"
_PROJECT_ID = 1  # Allen08 — MAPPINGS V re-evaluation of Allen+2008 grid

# ---------------------------------------------------------------------------
# Line mapping: (PyNeb-format name, vacuum wavelength Å, table, db_column(s))
# ---------------------------------------------------------------------------
# Each entry: (our_name, wave_aa, table, [col1, col2, ...])
# Fluxes from multiple columns are summed (doublets/blends).
# All fluxes are relative to Hβ = emis_VI.HI_4861.
#
# Table abbreviations: VI=emis_VI, UVB=emis_UVB, UVC=emis_UVC, IR=emis_IR

_LINE_MAP: list[tuple[str, float, str, list[str]]] = [
    # UV (emis_UVB: 938–1527 Å)
    ("LyA_1216A", 1216.0, "UVB", ["HI_1216"]),
    # UV (emis_UVC: 1527–3000 Å)
    ("CIV_1549A", 1549.0, "UVC", ["CIV_1548", "CIV_1551"]),
    ("HeII_1640A", 1640.0, "UVC", ["HeII_1640"]),
    ("CIII_1909A", 1909.0, "UVC", ["CIII_1907", "CIII_1909"]),
    ("MgII_2796A", 2796.0, "UVC", ["MgII_2796"]),
    ("MgII_2803A", 2803.0, "UVC", ["MgII_2803"]),
    # Optical (emis_VI: 3000–7500 Å)
    ("OII_3726A", 3726.0, "VI", ["OII_3726"]),
    ("OII_3729A", 3729.0, "VI", ["OII_3729"]),
    ("Hg_4341A", 4341.0, "VI", ["HI_4340"]),
    ("Hb_4861A", 4861.0, "VI", ["HI_4861"]),  # reference (ratio=1.0)
    ("O3_4959A", 4959.0, "VI", ["OIII_4959"]),
    ("O3_5007A", 5007.0, "VI", ["OIII_5007"]),
    ("HeI_5876A", 5876.0, "VI", ["HeI_5876"]),
    ("OI_6300A", 6300.0, "VI", ["OI_6300"]),
    ("HA_6563A", 6563.0, "VI", ["HI_6563"]),
    ("NII_6548A", 6548.0, "VI", ["NII_6548"]),
    ("NII_6583A", 6583.0, "VI", ["NII_6583"]),
    ("SII_6716A", 6716.0, "VI", ["SII_6716"]),
    ("SII_6731A", 6731.0, "VI", ["SII_6731"]),
    ("ArIII_7135A", 7135.0, "VI", ["ArIII_7136"]),
    ("OII_7320A", 7320.0, "VI", ["OII_7320"]),
    ("OII_7330A", 7330.0, "VI", ["OII_7330"]),
    # Near-IR (emis_IR: >7500 Å)
    ("SIII_9069A", 9069.0, "IR", ["SIII_9069"]),
    ("SIII_9532A", 9532.0, "IR", ["SIII_9531"]),
]

_LINE_NAMES = [e[0] for e in _LINE_MAP]
_LINE_WAVES = np.array([e[1] for e in _LINE_MAP], dtype=np.float32)
_N_LINES = len(_LINE_NAMES)

# Map table abbreviation → actual table name
_TABLE_NAME = {
    "VI": "emis_VI",
    "UVB": "emis_UVB",
    "UVC": "emis_UVC",
    "IR": "emis_IR",
}

# Hβ reference column (always in emis_VI)
_HB_COL = "HI_4861"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _connect() -> "pymysql.Connection":  # type: ignore[name-defined]
    import pymysql  # optional: pip install pymysql

    return pymysql.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        user=_DB_USER,
        passwd=_DB_PASSWD,
        db=_DB_NAME,
        connect_timeout=60,
        read_timeout=300,
    )


def _query_axes(co, project_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (velocities_kms, log_densities_cm3, b_fields_uG, abundance_names)."""
    cur = co.cursor()

    cur.execute(
        "SELECT DISTINCT shck_vel FROM shock_params WHERE ProjectID=%s ORDER BY shck_vel",
        (project_id,),
    )
    velocities = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)

    cur.execute(
        "SELECT DISTINCT preshck_dens FROM shock_params WHERE ProjectID=%s ORDER BY preshck_dens",
        (project_id,),
    )
    densities = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)
    log_densities = np.log10(np.maximum(densities, 1e-30)).astype(np.float32)

    cur.execute(
        "SELECT DISTINCT mag_fld FROM shock_params WHERE ProjectID=%s ORDER BY mag_fld",
        (project_id,),
    )
    b_fields = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)

    cur.execute(
        """SELECT DISTINCT a.AbundID, a.name
           FROM shock_params sp JOIN abundances a ON sp.AbundID=a.AbundID
           WHERE sp.ProjectID=%s ORDER BY a.AbundID""",
        (project_id,),
    )
    abund_rows = cur.fetchall()
    abund_ids = [r[0] for r in abund_rows]
    abund_names = [r[1] for r in abund_rows]

    print(f"  Velocities : {len(velocities)} pts  [{velocities[0]:.0f}–{velocities[-1]:.0f} km/s]")
    print(f"  Densities  : {len(densities)} pts  {list(densities)}")
    print(f"  B-fields   : {len(b_fields)} pts  [{b_fields[0]:.4g}–{b_fields[-1]:.4g} μG]")
    print(f"  Abundances : {abund_names}")

    return velocities, log_densities, b_fields, abund_ids, abund_names


def _fetch_component(
    co,
    project_id: int,
    component: str,  # "shock" | "precursor" | "shock_plus_precursor"
    velocities: np.ndarray,
    log_densities: np.ndarray,
    b_fields: np.ndarray,
    abund_ids: list[int],
) -> np.ndarray:
    """Return ratios array of shape (N_a, N_n, N_v, N_b, N_lines), NaN-padded."""
    n_a = len(abund_ids)
    n_n = len(log_densities)
    n_v = len(velocities)
    n_b = len(b_fields)

    ratios = np.full((n_a, n_n, n_v, n_b, _N_LINES), np.nan, dtype=np.float32)

    # Build {abund_id: axis_index}
    abund_idx = {aid: i for i, aid in enumerate(abund_ids)}
    v_vals = velocities.tolist()
    n_vals = np.power(10.0, log_densities).tolist()  # back to cm⁻³ for matching
    b_vals = b_fields.tolist()

    # Group lines by emis table to minimise queries
    tables_needed: dict[str, list[int]] = {}  # table → list of line indices
    for li, (_, _, tbl, _) in enumerate(_LINE_MAP):
        tables_needed.setdefault(tbl, []).append(li)

    # Always need VI for Hβ normalization
    tables_needed.setdefault("VI", [])

    for tbl, line_indices in tables_needed.items():
        tbl_name = _TABLE_NAME[tbl]

        # Columns to SELECT from this emis table
        db_cols_needed: list[str] = []
        line_col_map: list[tuple[int, list[str]]] = []  # (line_idx, [db_cols])
        for li in line_indices:
            _, _, _, db_cols = _LINE_MAP[li]
            for c in db_cols:
                if c not in db_cols_needed:
                    db_cols_needed.append(c)
            line_col_map.append((li, db_cols))

        # Hβ reference lives in emis_VI
        hb_alias = ""
        if tbl == "VI":
            if _HB_COL not in db_cols_needed:
                db_cols_needed.append(_HB_COL)
            hb_alias = f"e.{_HB_COL}"
        else:
            # JOIN emis_VI for Hβ
            hb_alias = f"vi.{_HB_COL}"

        sel_cols = ", ".join(f"e.{c}" for c in db_cols_needed)
        if tbl != "VI":
            sel_cols = f"vi.{_HB_COL} AS hb_ref, " + sel_cols

        if tbl == "VI":
            sql = f"""
                SELECT sp.AbundID, sp.preshck_dens, sp.mag_fld, sp.shck_vel,
                       {sel_cols}
                FROM shock_params sp
                JOIN {tbl_name} e ON e.ModelID = sp.ModelID
                WHERE sp.ProjectID = %s AND e.model_type = %s
                ORDER BY sp.AbundID, sp.preshck_dens, sp.mag_fld, sp.shck_vel
            """
        else:
            sql = f"""
                SELECT sp.AbundID, sp.preshck_dens, sp.mag_fld, sp.shck_vel,
                       {sel_cols}
                FROM shock_params sp
                JOIN {tbl_name} e ON e.ModelID = sp.ModelID
                JOIN emis_VI vi ON vi.ModelID = sp.ModelID AND vi.model_type = %s
                WHERE sp.ProjectID = %s AND e.model_type = %s
                ORDER BY sp.AbundID, sp.preshck_dens, sp.mag_fld, sp.shck_vel
            """

        cur = co.cursor()
        print(
            f"    Querying {tbl_name} ({len(line_indices)} lines, {component}) ... ",
            end="",
            flush=True,
        )
        if tbl == "VI":
            cur.execute(sql, (project_id, component))
        else:
            cur.execute(sql, (component, project_id, component))
        rows = cur.fetchall()
        print(f"{len(rows)} rows")

        # col offsets: 0=AbundID,1=preshck_dens,2=mag_fld,3=shck_vel, then line cols
        col_offset = 4
        col_names = [d[0] for d in cur.description][col_offset:]
        col_name_idx = {c: i for i, c in enumerate(col_names)}

        # hb column in result
        if tbl != "VI":
            hb_result_idx = col_name_idx.get("hb_ref", None)
        else:
            hb_result_idx = col_name_idx.get(_HB_COL, None)

        n_parsed = n_skip = 0
        for row in rows:
            aid, n_cm3, b_ug, v_kms = row[0], row[1], row[2], row[3]
            vals = row[col_offset:]

            i_a = abund_idx.get(aid)
            if i_a is None:
                n_skip += 1
                continue

            # Nearest-index snap (data is already on the exact grid)
            i_n = int(np.argmin(np.abs(np.array(n_vals) - n_cm3)))
            i_b = int(np.argmin(np.abs(b_fields - b_ug)))
            i_v = int(np.argmin(np.abs(velocities - v_kms)))

            hb = vals[hb_result_idx] if hb_result_idx is not None else None
            if hb is None or hb == 0.0:
                n_skip += 1
                continue

            for li, db_cols in line_col_map:
                flux = 0.0
                for c in db_cols:
                    v_col = vals[col_name_idx[c]]
                    if v_col is not None:
                        flux += float(v_col)
                ratios[i_a, i_n, i_v, i_b, li] = flux / float(hb)

            n_parsed += 1

        if n_skip:
            print(f"      ({n_parsed} ok, {n_skip} skipped — zero/null Hβ)")

    return ratios


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def _write_hdf5(
    out_path: Path,
    velocities: np.ndarray,
    log_densities: np.ndarray,
    b_fields: np.ndarray,
    abund_names: list[str],
    shock_ratios: np.ndarray,
    precursor_ratios: np.ndarray,
    combined_ratios: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"

    with h5py.File(out_path, mode) as f:
        f.attrs["description"] = (
            "MAPPINGS V shock model grids for tengri. "
            "Source: 3MdBs MySQL database (Alarie & Morisset 2019, RMxAA 55 377). "
            "Project 1 = Allen+2008 parameters re-evaluated with MAPPINGS V 5.1.13. "
            "Built by scripts/download_mappings_templates.py"
        )

        if "mappings5" in f:
            del f["mappings5"]
        g = f.create_group("mappings5")
        g.attrs["source"] = (
            "Alarie & Morisset 2019, RMxAA, 55, 377-394 — "
            "3MdBs Extensive Online Shock SEDModel Database (Project 1: Allen08)"
        )
        g.attrs["mappings_code"] = "Sutherland & Dopita 2017, ApJS, 229, 34 — MAPPINGS V 5.1.13"
        g.attrs["mappings_code_url"] = "https://mappings.anu.edu.au/code/"
        g.attrs["doi"] = "10.22201/ia.01851101p.2019.55.02.21"
        g.attrs["db_host"] = _DB_HOST
        g.attrs["project_id"] = _PROJECT_ID
        g.attrs["note"] = (
            "Non-solar abundances only at n=1 cm^-3; "
            "non-n=1 densities only for solar abundance. "
            "Missing combinations are NaN."
        )

        opts = dict(compression="gzip", compression_opts=5)
        g.create_dataset("velocities_kms", data=velocities, **opts)
        g.create_dataset("b_field_uG", data=b_fields, **opts)
        g.create_dataset("log_density_cm3", data=log_densities, **opts)
        g.create_dataset(
            "abundance_names",
            data=np.array(abund_names, dtype="S40"),
            **opts,
        )
        g.create_dataset(
            "line_names",
            data=np.array(_LINE_NAMES, dtype="S30"),
            **opts,
        )
        g.create_dataset("line_wavelengths_aa", data=_LINE_WAVES, **opts)
        g.create_dataset("shock_ratios", data=shock_ratios, **opts)
        g.create_dataset("precursor_ratios", data=precursor_ratios, **opts)
        g.create_dataset("combined_ratios", data=combined_ratios, **opts)
        # Hbeta_Flux is NULL for Project 1 in the DB — fill with NaN placeholder
        g.create_dataset(
            "hbeta_log_lum_erg_s",
            data=np.full(shock_ratios.shape[:-1], np.nan, dtype=np.float32),
            **opts,
        )

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
        default=str(Path(__file__).parents[1] / "data" / "mappings_templates.h5"),
        help="Output HDF5 file path (default: data/mappings_templates.h5)",
    )
    parser.add_argument(
        "--project",
        type=int,
        default=_PROJECT_ID,
        help=f"3MdBs ProjectID to download (default: {_PROJECT_ID} = Allen08)",
    )
    args = parser.parse_args()
    out_path = Path(args.output)

    print("\n=== 3MdBs MAPPINGS V shock grids (Alarie & Morisset 2019) ===")
    print(f"  Database : {_DB_HOST}:{_DB_PORT}/{_DB_NAME}")
    print(f"  Project  : {args.project}")
    print(f"  Output   : {out_path}\n")

    try:
        import pymysql  # noqa: F401
    except ImportError:
        print("ERROR: pymysql not installed. Run: pip install pymysql")
        sys.exit(1)

    print("Connecting to 3MdBs MySQL ... ", end="", flush=True)
    try:
        co = _connect()
        print("OK")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print("\nDiscovering grid axes ...")
    velocities, log_densities, b_fields, abund_ids, abund_names = _query_axes(co, args.project)

    components = [
        ("shock", "shock_ratios"),
        ("precursor", "precursor_ratios"),
        ("shock_plus_precursor", "combined_ratios"),
    ]

    all_ratios: dict[str, np.ndarray] = {}
    for db_component, key in components:
        print(f"\nFetching component: {db_component}")
        all_ratios[key] = _fetch_component(
            co,
            args.project,
            db_component,
            velocities,
            log_densities,
            b_fields,
            abund_ids,
        )

    co.close()

    # Summary
    shape = all_ratios["shock_ratios"].shape
    n_total = int(np.prod(shape))
    n_valid = int(np.sum(~np.isnan(all_ratios["combined_ratios"])))
    print(f"\nGrid shape : {shape}")
    print(f"Populated  : {n_valid:,}/{n_total:,} entries ({100 * n_valid / n_total:.1f}%)")

    print(f"\n=== Writing {out_path} ===")
    _write_hdf5(
        out_path,
        velocities,
        log_densities,
        b_fields,
        abund_names,
        all_ratios["shock_ratios"],
        all_ratios["precursor_ratios"],
        all_ratios["combined_ratios"],
    )

    print("\nDone. Verify with:")
    print(
        '  python -c "from tengri.nebular.shock import shock_line_ratios;'
        ' print(shock_line_ratios(300.0))"'
    )


if __name__ == "__main__":
    main()
