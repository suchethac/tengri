"""Select representative galaxies for Paper I analysis.

Selection criteria:
    - Clean flags: flg1 == 0
    - At least 10 detected bands
    - High median S/N
    - Present in all successfully parsed SED fitting codes
    - Three types by color proxy: blue star-forming, red/quiescent, intermediate/dusty
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from candels_io import (
    CANDELS_CATALOG,
    compute_snr_from_error,
    is_detected,
    load_candels_z1,
)
from ingest_art_sedfitting import ingest_z1_results


def compute_color_safe(
    f1_mag: float,
    f1_err: float,
    f2_mag: float,
    f2_err: float,
    fallback1_mag: float | None = None,
    fallback1_err: float | None = None,
    fallback2_mag: float | None = None,
    fallback2_err: float | None = None,
) -> float | None:
    """Compute color from two filters, with fallback if either is non-detected."""
    if is_detected(f1_mag, f1_err) and is_detected(f2_mag, f2_err):
        return f1_mag - f2_mag

    # Try fallback pair
    if (
        fallback1_mag is not None
        and fallback1_err is not None
        and fallback2_mag is not None
        and fallback2_err is not None
        and is_detected(fallback1_mag, fallback1_err)
        and is_detected(fallback2_mag, fallback2_err)
    ):
        return fallback1_mag - fallback2_mag

    return None


def select_z1_galaxies() -> dict:
    """Select three representative z~1 galaxies with highest S/N in each class."""
    # Load CANDELS photometry
    candels = load_candels_z1()
    ids = candels["id"]
    z = candels["z"]
    flg1 = candels["flg1"]

    # Load data
    ingest_results = ingest_z1_results()
    codes = sorted(set(r["code"] for r in ingest_results))
    print(f"Parsed codes: {codes}")

    # Find galaxies present in all codes
    id_by_code = {}
    for code in codes:
        id_by_code[code] = set(r["id"] for r in ingest_results if r["code"] == code)

    common_ids = set.intersection(*id_by_code.values())
    print(f"Galaxies in all codes: {len(common_ids)}")

    # Apply quality cuts
    good_mask = flg1 == 0
    good_ids = set(ids[good_mask])
    selected_ids = sorted(common_ids & good_ids)
    print(f"Good galaxies in all codes: {len(selected_ids)}")

    # Load CANDELS data for color and S/N computation
    candels_file = CANDELS_CATALOG
    data = np.genfromtxt(candels_file, skip_header=1)
    with open(candels_file) as f:
        header_line = f.readline()
    header = header_line.strip("#").strip().split()

    # Find filter columns
    f160w_idx = header.index("WFC3_F160W")
    ef160w_idx = header.index("eWFC3_F160W")
    isaac_ks_idx = header.index("ISAAC_KS")
    eisaac_ks_idx = header.index("eISAAC_KS")
    hawki_ks_idx = header.index("HAWKI_KS")
    ehawki_ks_idx = header.index("eHAWKI_KS")
    irac36_idx = header.index("IRAC_CH1")
    eirac36_idx = header.index("eIRAC_CH1")

    # Build list of all non-error columns for S/N calculation
    band_indices = []
    error_indices = []
    for i, h in enumerate(header):
        if not h.startswith("e") and h not in ["ID", "zz", "flg1", "flg2"]:
            band_indices.append(i)
            # Find corresponding error column
            err_h = "e" + h
            if err_h in header:
                error_indices.append(header.index(err_h))
            else:
                error_indices.append(-1)

    # Compute colors, S/N, and n_detected for all candidates
    candidates = []

    for idx, gal_id in enumerate(ids):
        if gal_id not in selected_ids:
            continue

        # Compute n_detected and median S/N
        snrs = []
        for band_idx, err_idx in zip(band_indices, error_indices):
            mag = data[idx, band_idx]
            if err_idx >= 0:
                err = data[idx, err_idx]
            else:
                err = -1

            if is_detected(mag, err):
                snr = compute_snr_from_error(err)
                if not np.isnan(snr):
                    snrs.append(snr)

        n_detected = len(snrs)
        median_snr = np.median(snrs) if snrs else 0

        # Compute color proxy
        f160w = data[idx, f160w_idx]
        ef160w = data[idx, ef160w_idx]
        isaac_ks = data[idx, isaac_ks_idx]
        eisaac_ks = data[idx, eisaac_ks_idx]
        hawki_ks = data[idx, hawki_ks_idx]
        ehawki_ks = data[idx, ehawki_ks_idx]
        irac36 = data[idx, irac36_idx]
        eirac36 = data[idx, eirac36_idx]

        # Try ISAAC first, then HAWKI, then IRAC
        ks_mag = isaac_ks if abs(isaac_ks - 98.99) > 0.1 else hawki_ks
        eks_mag = eisaac_ks if abs(isaac_ks - 98.99) > 0.1 else ehawki_ks

        color = compute_color_safe(f160w, ef160w, ks_mag, eks_mag, f160w, ef160w, irac36, eirac36)

        if color is None or n_detected < 10:
            continue

        # SFR from ingest
        sfr_rows = [r for r in ingest_results if r["id"] == gal_id]
        if sfr_rows:
            sfr = np.mean([r["logsfr"] for r in sfr_rows if not np.isnan(r["logsfr"])])
        else:
            sfr = np.nan

        color_note = "F160W-Ks" if ks_mag == isaac_ks else "F160W-IRAC1"

        candidates.append(
            {
                "id": gal_id,
                "z": z[idx],
                "n_detected": n_detected,
                "median_snr": median_snr,
                "color": color,
                "color_note": color_note,
                "sfr": sfr,
            }
        )

    print(f"\nCandidates with n_detected >= 10: {len(candidates)}")

    # Classify by color and rank by S/N within each class
    blue_candidates = []
    red_candidates = []
    intermediate_candidates = []

    for cand in candidates:
        color = cand["color"]
        if color < 0.4:
            blue_candidates.append(cand)
        elif color > 1.0:
            red_candidates.append(cand)
        else:
            intermediate_candidates.append(cand)

    # Sort each class by median S/N (descending)
    blue_candidates.sort(key=lambda x: x["median_snr"], reverse=True)
    red_candidates.sort(key=lambda x: x["median_snr"], reverse=True)
    intermediate_candidates.sort(key=lambda x: x["median_snr"], reverse=True)

    print("\nCandidates by color class:")
    print(f"  Blue (color < 0.4): {len(blue_candidates)}")
    print(f"  Red (color > 1.0): {len(red_candidates)}")
    print(f"  Intermediate (0.4-1.0): {len(intermediate_candidates)}")

    # Print top 5 per class
    print("\nTop 5 blue candidates:")
    for i, c in enumerate(blue_candidates[:5]):
        print(
            f"  {i + 1}. ID {c['id']}: S/N={c['median_snr']:.1f}, "
            f"n_det={c['n_detected']}, color={c['color']:.3f}"
        )

    print("\nTop 5 red candidates:")
    for i, c in enumerate(red_candidates[:5]):
        print(
            f"  {i + 1}. ID {c['id']}: S/N={c['median_snr']:.1f}, "
            f"n_det={c['n_detected']}, color={c['color']:.3f}"
        )

    print("\nTop 5 intermediate candidates:")
    for i, c in enumerate(intermediate_candidates[:5]):
        print(
            f"  {i + 1}. ID {c['id']}: S/N={c['median_snr']:.1f}, "
            f"n_det={c['n_detected']}, color={c['color']:.3f}"
        )

    # Select best in each class (prefer 4171 for blue if it qualifies)
    if blue_candidates and blue_candidates[0]["id"] == 4171:
        blue_gal = blue_candidates[0]
        print("\n✓ Using CANDELS 4171 as blue anchor (best S/N in class)")
    elif blue_candidates:
        blue_gal = blue_candidates[0]
        print(f"\n✓ Using ID {blue_gal['id']} as blue (4171 not best S/N)")
    else:
        blue_gal = None
        print("\n✗ No blue candidates found")

    red_gal = red_candidates[0] if red_candidates else None
    intermediate_gal = intermediate_candidates[0] if intermediate_candidates else None

    selected_gals = [
        (blue_gal, "blue_star_forming"),
        (red_gal, "red_quiescent"),
        (intermediate_gal, "intermediate_dusty"),
    ]

    result = {
        "selected_galaxies": [],
        "selection_criteria": {
            "clean_flags": "flg1 == 0",
            "min_detected_bands": 10,
            "present_in_all_codes": True,
            "ranked_by": "median_SNR_within_color_class",
            "codes": codes,
        },
    }

    print("\nFinal selection with real S/N and n_detected:")
    for gal, label in selected_gals:
        if gal is None:
            continue

        print(
            f"  ID {gal['id']}: {label}"
            f" (z={gal['z']:.4f}, S/N={gal['median_snr']:.1f}, "
            f"n_det={gal['n_detected']}, color={gal['color']:.3f})"
        )

        result["selected_galaxies"].append(
            {
                "id": int(gal["id"]),
                "z": float(gal["z"]),
                "n_detected": gal["n_detected"],
                "median_snr": float(gal["median_snr"]),
                "color_proxy": float(gal["color"]),
                "color_proxy_note": gal["color_note"],
                "type_label": label,
                "reason": f"Highest S/N in {label} class",
            }
        )

    return result


if __name__ == "__main__":
    result = select_z1_galaxies()

    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "analysis" / "paper1" / "results" / "selected_galaxies.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nWrote {len(result['selected_galaxies'])} selected galaxies")
