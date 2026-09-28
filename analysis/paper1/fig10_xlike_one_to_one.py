#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Figure 10: One-to-one comparison of X-like configurations against published codes.

Creates two-panel scatter plot (left: stellar mass, right: SFR) comparing published
values from five SED-fitting codes against tengri X-like configuration posteriors.
Only adopted cells are drawn; non-adopted cells are counted in the sidecar.

Prospector stellar mass is compared against tengri formed mass; other codes
(BAGPIPES, BEAGLE, CIGALE, Dense_Basis) against surviving mass from the census.
All codes use SFR_100myr (100 Myr average).

CLI:
    python analysis/paper1/fig10_xlike_one_to_one.py \\
        [--results-dir DIR] [--surviving JSON] [--published CSV] \\
        [--out PDF] [--data-out JSON]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adoption import is_adopted

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Map X-like JSON/NPZ keys to published code names in CSV
XLIKE_CODE = {
    "cigale_like": "CIGALE",
    "prospector_like": "Prospector",
    "bagpipes_like": "BAGPIPES",
    "beagle_like": "BEAGLE",
    "dense_basis_like": "Dense_Basis",
}

#: Markers per code (locked to fig06_code_overlay.py)
CODE_MARKERS = {
    "BAGPIPES": "o",
    "BEAGLE": "s",
    "CIGALE": "^",
    "Dense_Basis": "v",
    "Prospector": "D",
}

#: Okabe-Ito colorblind-safe palette
CODE_COLORS = {
    "BAGPIPES": "#E69F00",  # Orange
    "BEAGLE": "#56B4E9",  # Sky blue
    "CIGALE": "#009E73",  # Bluish green
    "Dense_Basis": "#CC79A7",  # Reddish purple
    "Prospector": "#0072B2",  # Blue
}

FIGURE_WIDTH = 7.0
FIGURE_HEIGHT = 3.5
PERCENTILES = (16.0, 50.0, 84.0)


class PublishedValue(NamedTuple):
    """Published value for one code."""

    code: str
    gal_id: int
    logmstar: float
    logmstar_lo: float
    logmstar_hi: float
    logsfr: float
    logsfr_lo: float
    logsfr_hi: float


class TengriValue(NamedTuple):
    """Tengri X-like posterior median with 16-84 interval."""

    gal_id: int
    xlike_key: str
    log_mstar_p16: float
    log_mstar_p50: float
    log_mstar_p84: float
    log_sfr_p16: float
    log_sfr_p50: float
    log_sfr_p84: float
    adopted: bool


def _load_published_csv(csv_path: Path) -> dict[tuple[str, int], PublishedValue]:
    """Load published values indexed by (code, gal_id)."""
    values = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["code"].strip()
            gal_id = int(row["id"].strip())
            values[(code, gal_id)] = PublishedValue(
                code=code,
                gal_id=gal_id,
                logmstar=float(row["logmstar"]),
                logmstar_lo=float(row["logmstar_lo"]),
                logmstar_hi=float(row["logmstar_hi"]),
                logsfr=float(row["logsfr"]),
                logsfr_lo=float(row["logsfr_lo"]),
                logsfr_hi=float(row["logsfr_hi"]),
            )
    return values


def _load_selected_galaxies(json_path: Path) -> set[int]:
    """Load set of selected galaxy IDs."""
    with open(json_path) as f:
        data = json.load(f)
    return {int(g["id"]) for g in data["selected_galaxies"]}


def _log_percentiles(values: np.ndarray) -> tuple[float, float, float]:
    """Percentiles of already-logged values (from NPZ which stores log10 values).

    The NPZ files store log10 values directly, so we just compute percentiles
    without additional logging.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite draws")
    p16, p50, p84 = np.percentile(finite, PERCENTILES)
    return float(p16), float(p50), float(p84)


def _load_surviving_mass_census(json_path: Path) -> dict[str, dict]:
    """Load surviving mass census. Returns dict[f'{gal_id}_{xlike_key}'] = {...}."""
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        data = json.load(f)
    return data.get("cells", {})


def _load_xlike_fits(
    results_dir: Path, selected_gal_ids: set[int], surviving_census: dict
) -> list[TengriValue]:
    """Load X-like fit results, yielding TengriValue for each cell.

    Requires surviving_census to be pre-loaded; raises SystemExit if
    surviving mass is needed but census is empty.
    """
    results = []
    xlike_keys = list(XLIKE_CODE.keys())

    # Collect adopted cells to detect if we have any
    all_adopted_per_code = {code: [] for code in XLIKE_CODE.values()}

    for gal_id in selected_gal_ids:
        for xlike_key in xlike_keys:
            json_path = results_dir / f"{gal_id}_{xlike_key}.json"
            npz_path = results_dir / f"{gal_id}_{xlike_key}.npz"

            if not json_path.exists() or not npz_path.exists():
                continue

            with open(json_path) as f:
                meta = json.load(f)

            # Use is_adopted to get adoption verdict (mimic fig09 approach)
            # For X-like, xlike_key is the "config" in the adoption check
            verdict = is_adopted(meta, xlike_key)
            adopted = verdict.adopted

            code_name = XLIKE_CODE[xlike_key]
            if adopted:
                all_adopted_per_code[code_name].append((gal_id, xlike_key))

            # Load NPZ
            with np.load(npz_path, allow_pickle=True) as npz:
                # Stellar mass: formed for Prospector, survived for others
                if code_name == "Prospector":
                    mass_key = "log_stellar_mass_formed"
                else:
                    mass_key = "log_stellar_mass_survived"

                if mass_key not in npz.files:
                    logger.warning(f"{gal_id}_{xlike_key}: no {mass_key}, skipped")
                    continue

                if "log_sfr_100myr" not in npz.files:
                    logger.warning(f"{gal_id}_{xlike_key}: no log_sfr_100myr, skipped")
                    continue

                try:
                    mass_data = np.asarray(npz[mass_key], dtype=float)
                    sfr_data = np.asarray(npz["log_sfr_100myr"], dtype=float)
                    log_mstar_p16, log_mstar_p50, log_mstar_p84 = _log_percentiles(mass_data)
                    log_sfr_p16, log_sfr_p50, log_sfr_p84 = _log_percentiles(sfr_data)
                except ValueError as exc:
                    logger.warning(f"{gal_id}_{xlike_key}: {exc}, skipped")
                    continue

            results.append(
                TengriValue(
                    gal_id=gal_id,
                    xlike_key=xlike_key,
                    log_mstar_p16=log_mstar_p16,
                    log_mstar_p50=log_mstar_p50,
                    log_mstar_p84=log_mstar_p84,
                    log_sfr_p16=log_sfr_p16,
                    log_sfr_p50=log_sfr_p50,
                    log_sfr_p84=log_sfr_p84,
                    adopted=adopted,
                )
            )

    # Check that we have at least one adopted cell across all codes
    any_adopted = any(len(v) > 0 for v in all_adopted_per_code.values())
    if False:  # MUTATED: any_adopted check removed
        sys.stderr.write("ERROR: no adopted X-like cells found.\n")
        sys.exit(1)

    # Check if we need surviving mass but it's missing
    non_prospector_adopted = [
        (gid, key)
        for code, cells in all_adopted_per_code.items()
        for gid, key in cells
        if code != "Prospector"
    ]
    if non_prospector_adopted and not surviving_census:
        sys.stderr.write(
            "ERROR: Non-Prospector adopted cells found but surviving mass census "
            "is empty. Run:\n"
            "  python analysis/paper1/surviving_mass_census.py "
            "--results-dir analysis/paper1/results/fits_xlike "
            "--out analysis/paper1/results/surviving_mass_xlike.json\n"
        )
        sys.exit(1)

    return results


def _compute_stats(
    values: list[float],
) -> tuple[float, float, float]:
    """Compute median and median absolute deviation."""
    if not values:
        return None, None, None
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.array(values) - median)))
    half_iqr = (np.percentile(values, 84) - np.percentile(values, 16)) / 2
    return median, mad, half_iqr


def _make_figure(
    published: dict[tuple[str, int], PublishedValue],
    tengri_values: list[TengriValue],
    surviving_census: dict,
) -> tuple[dict, str]:
    """Create figure and compute sidecar data.

    Returns (sidecar_dict, figure_path_for_pdf).
    """
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    sidecar = {"codes": {}}

    # Organize by code
    by_code: dict[str, list[tuple[PublishedValue, TengriValue]]] = {
        code: [] for code in XLIKE_CODE.values()
    }
    non_adopted_counts: dict[str, int] = {code: 0 for code in XLIKE_CODE.values()}

    for tengri_val in tengri_values:
        code = XLIKE_CODE[tengri_val.xlike_key]
        pub_key = (code, tengri_val.gal_id)

        if pub_key not in published:
            logger.warning(f"No published value for {code} gal {tengri_val.gal_id}")
            continue

        pub_val = published[pub_key]

        if tengri_val.adopted:
            by_code[code].append((pub_val, tengri_val))
        else:
            non_adopted_counts[code] += 1

    # Plot mass panel (left)
    ax_mass = fig.add_subplot(gs[0, 0])
    for code in sorted(XLIKE_CODE.values()):
        pairs = by_code[code]
        if not pairs:
            sidecar["codes"][code] = {
                "n_adopted": 0,
                "n_non_adopted": non_adopted_counts[code],
                "mass_median_offset": None,
                "mass_scatter": None,
                "sfr_median_offset": None,
                "sfr_scatter": None,
            }
            continue

        # Extract data
        pub_masses = [p[0].logmstar for p in pairs]
        pub_mass_los = [p[0].logmstar_lo for p in pairs]
        pub_mass_his = [p[0].logmstar_hi for p in pairs]
        tengri_masses = [p[1].log_mstar_p50 for p in pairs]
        tengri_mass_lo = [p[1].log_mstar_p16 for p in pairs]
        tengri_mass_hi = [p[1].log_mstar_p84 for p in pairs]

        # Error bars for published (absolute bounds)
        pub_mass_xerr_lo = np.array(pub_masses) - np.array(pub_mass_los)
        pub_mass_xerr_hi = np.array(pub_mass_his) - np.array(pub_masses)

        # Skip x error bar for Dense_Basis if all NaN
        if code == "Dense_Basis" and np.all(np.isnan(pub_mass_xerr_lo)):
            pub_mass_xerr = None
        else:
            pub_mass_xerr = [pub_mass_xerr_lo, pub_mass_xerr_hi]

        # Plot
        ax_mass.errorbar(
            pub_masses,
            tengri_masses,
            xerr=pub_mass_xerr,
            fmt="none",
            ecolor=CODE_COLORS[code],
            elinewidth=0.7,
            capsize=1.5,
            alpha=0.6,
            zorder=2,
        )
        ax_mass.scatter(
            pub_masses,
            tengri_masses,
            s=30,
            marker=CODE_MARKERS[code],
            color=CODE_COLORS[code],
            edgecolors="none",
            label=code,
            zorder=3,
        )

        # Compute statistics
        mass_offsets = np.array(tengri_masses) - np.array(pub_masses)
        mass_median_offset, _, mass_scatter = _compute_stats(mass_offsets.tolist())

        sidecar["codes"][code] = {
            "n_adopted": len(pairs),
            "n_non_adopted": non_adopted_counts[code],
            "mass_median_offset": float(mass_median_offset)
            if mass_median_offset is not None
            else None,
            "mass_scatter": float(mass_scatter) if mass_scatter is not None else None,
            "sfr_median_offset": None,  # Will fill in SFR panel
            "sfr_scatter": None,
        }

    # One-to-one diagonal for mass
    mass_values = []
    for c in by_code:
        if by_code[c]:
            mass_values.extend([p[0].logmstar for p in by_code[c]])
    if mass_values:
        mass_lim = [np.floor(min(mass_values) * 2) / 2, np.ceil(max(mass_values) * 2) / 2]
    else:
        mass_lim = [9, 11]
    ax_mass.plot(mass_lim, mass_lim, "k--", alpha=0.3, linewidth=0.8, zorder=1)
    ax_mass.set_xlim(mass_lim)
    ax_mass.set_ylim(mass_lim)
    ax_mass.set_xlabel(r"Published $\log_{10}(M_{\star}\,/\,M_\odot)$")
    ax_mass.set_ylabel(r"tengri X-like $\log_{10}(M_{\star}\,/\,M_\odot)$")
    ax_mass.tick_params(labelsize=7)
    ax_mass.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # Plot SFR panel (right)
    ax_sfr = fig.add_subplot(gs[0, 1])
    for code in sorted(XLIKE_CODE.values()):
        pairs = by_code[code]
        if not pairs:
            continue

        # Extract data
        pub_sfrs = [p[0].logsfr for p in pairs]
        pub_sfr_los = [p[0].logsfr_lo for p in pairs]
        pub_sfr_his = [p[0].logsfr_hi for p in pairs]
        tengri_sfrs = [p[1].log_sfr_p50 for p in pairs]
        tengri_sfr_lo = [p[1].log_sfr_p16 for p in pairs]
        tengri_sfr_hi = [p[1].log_sfr_p84 for p in pairs]

        # Error bars for published
        pub_sfr_xerr_lo = np.array(pub_sfrs) - np.array(pub_sfr_los)
        pub_sfr_xerr_hi = np.array(pub_sfr_his) - np.array(pub_sfrs)

        # Skip x error bar for Dense_Basis if all NaN
        if code == "Dense_Basis" and np.all(np.isnan(pub_sfr_xerr_lo)):
            pub_sfr_xerr = None
        else:
            pub_sfr_xerr = [pub_sfr_xerr_lo, pub_sfr_xerr_hi]

        # Plot
        ax_sfr.errorbar(
            pub_sfrs,
            tengri_sfrs,
            xerr=pub_sfr_xerr,
            fmt="none",
            ecolor=CODE_COLORS[code],
            elinewidth=0.7,
            capsize=1.5,
            alpha=0.6,
            zorder=2,
        )
        ax_sfr.scatter(
            pub_sfrs,
            tengri_sfrs,
            s=30,
            marker=CODE_MARKERS[code],
            color=CODE_COLORS[code],
            edgecolors="none",
            label=code,
            zorder=3,
        )

        # Compute SFR statistics
        sfr_offsets = np.array(tengri_sfrs) - np.array(pub_sfrs)
        sfr_median_offset, _, sfr_scatter = _compute_stats(sfr_offsets.tolist())

        # Update sidecar
        sidecar["codes"][code]["sfr_median_offset"] = (
            float(sfr_median_offset) if sfr_median_offset is not None else None
        )
        sidecar["codes"][code]["sfr_scatter"] = (
            float(sfr_scatter) if sfr_scatter is not None else None
        )

    # One-to-one diagonal for SFR
    sfr_values = []
    for c in by_code:
        if by_code[c]:
            sfr_values.extend([p[0].logsfr for p in by_code[c]])
    if sfr_values:
        sfr_lim = [np.floor(min(sfr_values) * 2) / 2, np.ceil(max(sfr_values) * 2) / 2]
    else:
        sfr_lim = [-1, 2]
    ax_sfr.plot(sfr_lim, sfr_lim, "k--", alpha=0.3, linewidth=0.8, zorder=1)
    ax_sfr.set_xlim(sfr_lim)
    ax_sfr.set_ylim(sfr_lim)
    ax_sfr.set_xlabel(
        r"Published $\log_{10}(\mathrm{SFR}_{100\,\mathrm{Myr}}\,/\,M_\odot\,\mathrm{yr}^{-1})$"
    )
    ax_sfr.set_ylabel(
        r"tengri X-like $\log_{10}(\mathrm{SFR}_{100\,\mathrm{Myr}}\,/\,M_\odot\,\mathrm{yr}^{-1})$"
    )
    ax_sfr.tick_params(labelsize=7)
    ax_sfr.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # Add legend
    handles = [
        Line2D(
            [0],
            [0],
            marker=CODE_MARKERS[code],
            color="w",
            markerfacecolor=CODE_COLORS[code],
            markersize=6,
            label=code,
        )
        for code in sorted(XLIKE_CODE.values())
        if by_code[code]
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=7, ncol=1)

    return sidecar, fig


def main() -> None:
    """Parse arguments, load data, create figure."""
    parser = argparse.ArgumentParser(
        description="One-to-one X-like configuration comparison figure"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "results" / "fits_xlike",
        help="Directory containing X-like fit results (JSON/NPZ pairs)",
    )
    parser.add_argument(
        "--surviving",
        type=Path,
        default=None,
        help="Path to surviving mass census JSON (optional for Prospector-only)",
    )
    parser.add_argument(
        "--published",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "results" / "art_sedfitting_z1.csv",
        help="Path to published code results CSV",
    )
    parser.add_argument(
        "--selected-galaxies",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "results" / "selected_galaxies_20.json",
        help="Path to selected galaxies JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "paper1_figures" / "figures" / "fig10_xlike_one_to_one.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "results" / "fig10_xlike_one_to_one_data.json",
        help="Output sidecar JSON path",
    )

    args = parser.parse_args()

    # Load published data
    published = _load_published_csv(args.published)
    selected_gal_ids = _load_selected_galaxies(args.selected_galaxies)

    # Load surviving mass census if provided
    surviving_census = {}
    if args.surviving:
        surviving_census = _load_surviving_mass_census(args.surviving)

    # Load X-like fits
    tengri_values = _load_xlike_fits(args.results_dir, selected_gal_ids, surviving_census)

    if not tengri_values:
        sys.stderr.write("ERROR: no X-like cells found in results directory\n")
        sys.exit(1)

    # Create figure
    sidecar, fig = _make_figure(published, tengri_values, surviving_census)

    # Write outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved to {args.out}")
    plt.close(fig)

    args.data_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.data_out, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"Sidecar saved to {args.data_out}")


if __name__ == "__main__":
    main()
