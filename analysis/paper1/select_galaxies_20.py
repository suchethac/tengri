"""Expand galaxy sample to ~20 representative galaxies for Paper I analysis.

Selection criteria (same as original 3):
    - Clean flags: flg1 == 0
    - At least 10 detected bands
    - High median S/N
    - Present in all successfully parsed SED fitting codes
    - Diverse color distribution: blue star-forming, red/quiescent, intermediate/dusty
    - No rising IRAC colors (screens out AGN-like SEDs)

Expansion strategy:
    - Keep the original 3 (IDs 13097, 15336, 16049)
    - Select top 7-8 per color class to achieve ~20 total
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from .candels_io import (
    CANDELS_CATALOG,
    compute_snr_from_error,
    is_detected,
    load_candels_z1,
)


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


def has_rising_irac_colors(ch1_ch3: float | None, ch1_ch4: float | None) -> bool:
    """AGN screen: rising mid-IR color."""
    ch3_rising = ch1_ch3 is not None and ch1_ch3 > 0.3
    ch4_rising = ch1_ch4 is not None and ch1_ch4 > 0.5
    return ch3_rising or ch4_rising


def load_art_results() -> dict:
    """Load pre-ingested ART sedfitting results from CSV."""
    results_file = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "paper1"
        / "results"
        / "art_sedfitting_z1.csv"
    )

    sfr_by_id = {}
    codes_found = set()

    if results_file.exists():
        with open(results_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gal_id = int(row["id"])
                code = row["code"]
                codes_found.add(code)

                if gal_id not in sfr_by_id:
                    sfr_by_id[gal_id] = []

                logsfr = row.get("logsfr", "")
                if logsfr and logsfr != "":
                    try:
                        sfr_by_id[gal_id].append(float(logsfr))
                    except ValueError:
                        pass

    return {"sfr_by_id": sfr_by_id, "codes": sorted(codes_found)}


# The three mid-infrared AGN candidates carried by the demonstration sample.
# Locked by the paper (tab:candels_galaxies): IRAC-rising objects that survive
# the workshop IRAC flags. Colour used for the cut is IRAC1 - IRAC3.
AGN_CANDIDATE_IDS = (1826, 4056, 24786)


def select_z1_galaxies_expanded(
    target_count: int = 20,
    blue_per_class: int = 7,
    red_per_class: int = 7,
    intermediate_per_class: int = 3,
    agn_per_class: int = 3,
) -> dict:
    """Select the twenty demonstration galaxies: 7 blue, 7 red, 3 dusty, 3 AGN.

    The mid-infrared AGN candidates are a *class*, not a reject pile. An earlier
    revision screened rising-IRAC objects out of the sample entirely, which is
    the right call for a host-only suite but removes exactly the objects
    Configuration VI's disc and torus exist to fit.
    """
    # Load CANDELS photometry
    candels = load_candels_z1()
    ids = candels["id"]
    z = candels["z"]
    flg1 = candels["flg1"]

    # Load ART results to get codes
    art_data = load_art_results()
    codes = art_data["codes"]
    sfr_by_id = art_data["sfr_by_id"]
    print(f"Parsed codes from results: {codes}")

    # Find galaxies present in art_sedfitting_z1.csv (i.e., in all codes)
    common_ids = set(sfr_by_id.keys())
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
    irac58_idx = header.index("IRAC_CH3")
    eirac58_idx = header.index("eIRAC_CH3")
    irac80_idx = header.index("IRAC_CH4")
    eirac80_idx = header.index("eIRAC_CH4")

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
    agn_screened = []

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
        irac58 = data[idx, irac58_idx]
        eirac58 = data[idx, eirac58_idx]
        irac80 = data[idx, irac80_idx]
        eirac80 = data[idx, eirac80_idx]

        # Try ISAAC first, then HAWKI, then IRAC
        ks_mag = isaac_ks if abs(isaac_ks - 98.99) > 0.1 else hawki_ks
        eks_mag = eisaac_ks if abs(isaac_ks - 98.99) > 0.1 else ehawki_ks

        color = compute_color_safe(f160w, ef160w, ks_mag, eks_mag, f160w, ef160w, irac36, eirac36)

        if color is None or n_detected < 10:
            continue

        # SFR from ART results
        sfr_vals = sfr_by_id.get(gal_id, [])
        sfr = np.mean(sfr_vals) if sfr_vals else np.nan

        color_note = "F160W-Ks" if ks_mag == isaac_ks else "F160W-IRAC1"

        record = {
            "id": gal_id,
            "z": z[idx],
            "n_detected": n_detected,
            "median_snr": median_snr,
            "color": color,
            "color_note": color_note,
            "sfr": sfr,
        }

        # Mid-infrared AGN screen. Rising IRAC colors mark a rest 2-4 um power
        # law that no host-only configuration fits; those objects go to their
        # own class rather than out of the sample, and are held back from the
        # color classes below so an AGN cannot also be counted as a red galaxy.
        ch1_ch3 = compute_color_safe(irac36, eirac36, irac58, eirac58)
        ch1_ch4 = compute_color_safe(irac36, eirac36, irac80, eirac80)
        if has_rising_irac_colors(ch1_ch3, ch1_ch4):
            agn_screened.append({**record, "ch1_ch3": ch1_ch3, "ch1_ch4": ch1_ch4})
            continue

        candidates.append(record)

    print(f"\nCandidates with n_detected >= 10: {len(candidates)}")
    print(f"AGN-screened (rising IRAC colors CH1-CH3 > 0.3 or CH1-CH4 > 0.5): {len(agn_screened)}")

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

    # The AGN class is named, not ranked: these three IDs are locked by the
    # paper, so take them by identity and fail loudly if the catalog no longer
    # yields one. Silently shipping nineteen galaxies, or substituting the next
    # IRAC-rising object, would leave the manuscript's table describing a sample
    # that was never fit.
    agn_by_id = {int(c["id"]): c for c in agn_screened}
    agn_selected = []
    for want in AGN_CANDIDATE_IDS:
        if want not in agn_by_id:
            raise ValueError(
                f"Locked AGN candidate {want} did not survive selection. It is "
                f"either absent from the catalog, below the detection-count "
                f"floor, or no longer IRAC-rising under the current cut. "
                f"Available IRAC-rising IDs: {sorted(agn_by_id)}. Resolve this "
                f"against tab:candels_galaxies before running the grid."
            )
        agn_selected.append(agn_by_id[want])

    selected_list = []
    for cand in blue_candidates[:blue_per_class]:
        selected_list.append((cand, "blue_star_forming"))

    for cand in red_candidates[:red_per_class]:
        selected_list.append((cand, "red_quiescent"))

    for cand in intermediate_candidates[:intermediate_per_class]:
        selected_list.append((cand, "intermediate_dusty"))

    for cand in agn_selected:
        selected_list.append((cand, "mir_agn_candidate"))

    if len(selected_list) != target_count:
        raise ValueError(
            f"Selected {len(selected_list)} galaxies, expected {target_count} "
            f"({blue_per_class} blue + {red_per_class} red + "
            f"{intermediate_per_class} dusty + {agn_per_class} AGN). A color "
            f"class ran short of candidates."
        )

    selected_list.sort(key=lambda x: int(x[0]["id"]))

    result = {
        "selected_galaxies": [],
        "selection_criteria": {
            "clean_flags": "flg1 == 0",
            "min_detected_bands": 10,
            "present_in_all_codes": True,
            "agn_screen": (
                "exclude rising IRAC colors (m_CH1 - m_CH3 > 0.3 or m_CH1 - m_CH4 > 0.5 "
                "AB mag), a rest 2-4 um power law no stellar configuration fits"
            ),
            "agn_screened_count": len(agn_screened),
            "ranked_by": "median_SNR_within_color_class",
            "codes": codes,
            "target_count": target_count,
            "blue_per_class": blue_per_class,
            "red_per_class": red_per_class,
            "intermediate_per_class": intermediate_per_class,
        },
    }

    print("\nFinal selection with real S/N and n_detected:")
    for gal, label in selected_list:
        print(
            f"  ID {int(gal['id'])}: {label}"
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
                "reason": f"Ranked in top candidates by S/N in {label} class",
            }
        )

    return result


if __name__ == "__main__":
    result = select_z1_galaxies_expanded()

    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "analysis" / "paper1" / "results" / "selected_galaxies_20.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    selected_ids = [g["id"] for g in result["selected_galaxies"]]
    print(f"\nWrote {len(result['selected_galaxies'])} selected galaxies")
    print(f"Selected IDs: {sorted(selected_ids)}")
