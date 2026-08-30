"""Ingest SED fitting results from participating codes for Paper I analysis."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ART_SEDFITTING_DIR = Path(
    os.environ.get(
        "ART_SEDFITTING_DIR",
        "/Users/suchethacooray/Projects/art_sedfitting",
    )
)

CODE_OUTPUTS = ART_SEDFITTING_DIR / "code_outputs"
CANDELS_PATH = Path(
    "/Users/suchethacooray/Projects/tengri/analysis/hst_proposal/data/CANDELS_GDSS_workshop_z1.dat"
)


def get_candels_ids() -> set[int]:
    """Get the list of CANDELS galaxy IDs."""
    ids = set()
    with open(CANDELS_PATH) as f:
        f.readline()  # skip header
        for line in f:
            gal_id = int(line.split()[0])
            ids.add(gal_id)
    return ids


def parse_prospector_z1() -> list[dict]:
    """Parse Prospector results."""
    filepath = CODE_OUTPUTS / "prospector_output_z1.dat"
    if not filepath.exists():
        print(f"Skipping Prospector: {filepath} not found")
        return []

    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    result = []
    for row in data:
        result.append(
            {
                "id": int(row[0]),
                "logmstar": row[1],
                "logmstar_lo": row[1] - row[3],
                "logmstar_hi": row[1] + row[2],
                "logsfr": row[4],
                "logsfr_lo": row[4] - row[6],
                "logsfr_hi": row[4] + row[5],
                "mass_definition_note": "formed stellar mass",
                "sfr_timescale_note": "100 Myr",
            }
        )
    print(f"Prospector: {len(result)} galaxies")
    return result


def parse_bagpipes_z1() -> list[dict]:
    """Parse BAGPIPES results.

    BAGPIPES stellar_mass values are in log10 units; sfr values are linear
    Msun/yr and must be converted to log10.
    """
    filepath = CODE_OUTPUTS / "bagpipes_11_3_19_z1_noir.cat"
    if not filepath.exists():
        print(f"Skipping BAGPIPES: {filepath} not found")
        return []

    with open(filepath) as f:
        header = f.readline().strip("#").split("\t")

    data = np.genfromtxt(filepath, delimiter="\t", skip_header=1)

    id_idx = header.index("ID")
    mstar_50_idx = header.index("stellar_mass_50")
    mstar_16_idx = header.index("stellar_mass_16")
    mstar_84_idx = header.index("stellar_mass_84")
    sfr_50_idx = header.index("sfr_50")
    sfr_16_idx = header.index("sfr_16")
    sfr_84_idx = header.index("sfr_84")

    result = []
    for row in data:
        result.append(
            {
                "id": int(row[id_idx]),
                "logmstar": row[mstar_50_idx],
                "logmstar_lo": row[mstar_16_idx],
                "logmstar_hi": row[mstar_84_idx],
                "logsfr": np.log10(row[sfr_50_idx]),
                "logsfr_lo": np.log10(row[sfr_16_idx]),
                "logsfr_hi": np.log10(row[sfr_84_idx]),
                "mass_definition_note": "survived stellar mass",
                "sfr_timescale_note": "instantaneous",
            }
        )
    print(f"BAGPIPES: {len(result)} galaxies")
    return result


def parse_cigale_z1() -> list[dict]:
    """Parse CIGALE results."""
    filepath = CODE_OUTPUTS / "cigale_UV_NIR_2020.fits"
    if not filepath.exists():
        print(f"Skipping CIGALE: {filepath} not found")
        return []

    try:
        from astropy.io import fits
    except ImportError:
        print("Skipping CIGALE: astropy not available")
        return []

    with fits.open(filepath) as hdul:
        data = hdul[1].data

    result = []
    for row in data:
        logmstar = row[1]
        mstar_err = row[2]
        logsfr = row[3]
        sfr_err = row[4]

        result.append(
            {
                "id": int(row[0]),
                "logmstar": logmstar,
                "logmstar_lo": logmstar - mstar_err,
                "logmstar_hi": logmstar + mstar_err,
                "logsfr": logsfr,
                "logsfr_lo": logsfr - sfr_err,
                "logsfr_hi": logsfr + sfr_err,
                "mass_definition_note": "survived stellar mass",
                "sfr_timescale_note": "100 Myr mean",
            }
        )
    print(f"CIGALE: {len(result)} galaxies")
    return result


def parse_beagle_z1() -> list[dict]:
    """Parse BEAGLE results."""
    filepath = CODE_OUTPUTS / "BEAGLE_summary_catalogue_z1.fits"
    if not filepath.exists():
        print(f"Skipping BEAGLE: {filepath} not found")
        return []

    try:
        from astropy.io import fits
    except ImportError:
        print("Skipping BEAGLE: astropy not available")
        return []

    with fits.open(filepath) as hdul:
        galprop = hdul[1]
        sf = hdul[2]

        ids = galprop.data["ID      "]
        mstar_median = np.log10(galprop.data["M_star_median"])
        mstar_68 = np.log10(galprop.data["M_star_68.00"])
        sfr_median = np.log10(sf.data["SFR_100_median"])
        sfr_68 = np.log10(sf.data["SFR_100_68.00"])

    result = []
    for i in range(len(ids)):
        result.append(
            {
                "id": int(ids[i]),
                "logmstar": mstar_median[i],
                "logmstar_lo": mstar_68[i, 0],
                "logmstar_hi": mstar_68[i, 1],
                "logsfr": sfr_median[i],
                "logsfr_lo": sfr_68[i, 0],
                "logsfr_hi": sfr_68[i, 1],
                "mass_definition_note": "survived stellar mass",
                "sfr_timescale_note": "100 Myr averaged",
            }
        )
    print(f"BEAGLE: {len(result)} galaxies")
    return result


def parse_dense_basis_z1() -> list[dict]:
    """Parse Dense Basis results.

    Dense_Basis_GOODS-S_v1.2.dat has no header. Columns (from notebook):
    0: ID, 1: z_fit, 2: useflag, 3: logM_gal, 4: logM*, 5: logM*_84,
    6: logM*_16, 7: log_SFR_inst, 8-9: errors, 10: log_SFR_100

    Filter to CANDELS z~1 sample IDs only. Uncertainty floor of 0.1 dex
    is applied (matching art_sedfitting/notebooks/import_scripts/
    import_dense_basis_fits.py lines 33-34).
    """
    filepath = CODE_OUTPUTS / "Dense_Basis_GOODS-S_v1.2.dat"
    if not filepath.exists():
        print(f"Skipping Dense Basis: {filepath} not found")
        return []

    candels_ids = get_candels_ids()

    try:
        data = np.genfromtxt(filepath, delimiter=",")
    except Exception as e:
        print(f"Skipping Dense Basis: parse error: {e}")
        return []

    result = []
    if data.ndim == 1:
        data = data[np.newaxis, :]

    for row in data:
        gal_id = int(row[0])
        if gal_id not in candels_ids:
            continue

        logmstar = row[4]  # logM* median
        logmstar_lo = row[6]  # logM*_16
        logmstar_hi = row[5]  # logM*_84

        # Apply uncertainty floor of 0.1 dex (matching official notebook)
        if logmstar - logmstar_lo < 0.1:
            logmstar_lo = logmstar - 0.1
        if logmstar_hi - logmstar < 0.1:
            logmstar_hi = logmstar + 0.1

        logsfr = row[10]  # log_SFR_100

        result.append(
            {
                "id": gal_id,
                "logmstar": logmstar,
                "logmstar_lo": logmstar_lo,
                "logmstar_hi": logmstar_hi,
                "logsfr": logsfr,
                "logsfr_lo": np.nan,
                "logsfr_hi": np.nan,
                "mass_definition_note": "survived stellar mass",
                "sfr_timescale_note": "100 Myr",
            }
        )
    print(f"Dense Basis: {len(result)} galaxies")
    return result


def ingest_z1_results() -> list[dict]:
    """Ingest all z~1 results."""
    parsers = [
        ("Prospector", parse_prospector_z1),
        ("BAGPIPES", parse_bagpipes_z1),
        ("CIGALE", parse_cigale_z1),
        ("BEAGLE", parse_beagle_z1),
        ("Dense_Basis", parse_dense_basis_z1),
    ]

    all_results = []
    for code_name, parser in parsers:
        parsed = parser()
        for row in parsed:
            row["code"] = code_name
        all_results.extend(parsed)

    all_results.sort(key=lambda x: (x["code"], x["id"]))

    return all_results


def write_csv(results: list[dict], output_path: Path) -> None:
    """Write results to CSV with all columns filled."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "code",
        "id",
        "logmstar",
        "logmstar_lo",
        "logmstar_hi",
        "logsfr",
        "logsfr_lo",
        "logsfr_hi",
        "mass_definition_note",
        "sfr_timescale_note",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "code": row["code"],
                    "id": row["id"],
                    "logmstar": row["logmstar"],
                    "logmstar_lo": row["logmstar_lo"],
                    "logmstar_hi": row["logmstar_hi"],
                    "logsfr": row["logsfr"],
                    "logsfr_lo": row["logsfr_lo"],
                    "logsfr_hi": row["logsfr_hi"],
                    "mass_definition_note": row["mass_definition_note"],
                    "sfr_timescale_note": row["sfr_timescale_note"],
                }
            )


if __name__ == "__main__":
    results = ingest_z1_results()
    print(f"\nTotal records parsed: {len(results)}")
    print(f"Unique galaxies: {len(set(r['id'] for r in results))}")
    print(f"Codes: {sorted(set(r['code'] for r in results))}")

    output_path = Path("analysis/paper1/results/art_sedfitting_z1.csv")
    write_csv(results, output_path)
    print(f"\nWrote to {output_path}")
