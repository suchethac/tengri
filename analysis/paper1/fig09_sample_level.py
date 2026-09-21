#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 9 - The demonstration grid as a whole.

Twenty CANDELS GOODS-South galaxies fit under each of the six configurations,
summarized in one frame rather than twenty individual panels, because the
quantity of interest is how far apart the configurations place the same galaxy.

Left: posterior medians in the log M* - log SFR plane, one marker per cell,
colored by configuration, with the credible intervals of one representative
galaxy drawn for scale.

Right, upper and lower: the configuration-to-configuration differences galaxy
by galaxy, in log M* and in log SFR. Each cell is plotted as its offset from
that galaxy's cross-configuration median, so the vertical extent of a column is
model-choice spread while the shaded band is the spread across the sample. The
two are directly comparable only in this differenced form.

Every quantity is read from the per-cell posterior NPZ; no model is rebuilt, so
this script does not import tengri and runs in seconds.

CLI:
    python analysis/paper1/fig09_sample_level.py [--results-dir DIR] [--out PDF]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adoption import is_adopted, low_ess_note
from _cell_provenance import audit, banner
from _figure_style import CONFIG_COLORS, CONFIG_ORDER
from _grid_completeness import completeness_note, load_expected_galaxy_ids
from config_metadata import CONFIGS

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RESULTS = REPO_ROOT / "analysis" / "paper1" / "results" / "fits"
#: The locked sample this figure claims to show. Read, never restated.
SELECTION_20 = REPO_ROOT / "analysis" / "paper1" / "results" / "selected_galaxies_20.json"


FIGURE_WIDTH = 7.1
FIGURE_HEIGHT = 3.6
PERCENTILES = (16.0, 50.0, 84.0)

#: Edge color for a cell the bar adopted on too few effective samples. It is
#: drawn filled, because the bar did adopt it, but ringed, because its
#: posterior is not one: without this it is indistinguishable from a
#: converged cell and the eye reads the stamp's warning as applying to
#: nothing in particular.
LOW_ESS_EDGE = "#b22222"


@dataclass(frozen=True)
class Cell:
    """One galaxy-configuration fit, reduced to the quantities this figure draws."""

    gal_id: int
    config: str
    adopted: bool
    #: Set when the cell cleared the bar on too few effective samples. The bar
    #: has no ESS criterion, so this is the only place such a cell announces
    #: itself; left None for every honest cell.
    low_ess: str | None
    log_mstar: tuple[float, float, float]  # (p16, p50, p84)
    log_sfr: tuple[float, float, float]


def _log_percentiles(values: np.ndarray) -> tuple[float, float, float]:
    """Percentiles of log10(values), taken after dropping non-positive draws.

    A non-positive stellar mass or star formation rate cannot be logged. Such
    draws are dropped rather than clipped, and the caller is told how many went,
    because clipping would pile them at an arbitrary floor and bias the p16.
    """
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        raise ValueError("no positive finite draws")
    logged = np.log10(finite)
    p16, p50, p84 = np.percentile(logged, PERCENTILES)
    return float(p16), float(p50), float(p84)


def load_cells(results_dir: Path) -> tuple[list[Cell], list[str]]:
    """Read every cell in results_dir. Returns (cells, warnings)."""
    cells: list[Cell] = []
    warnings: list[str] = []
    for json_path in sorted(results_dir.glob("*_*.json")):
        stem = json_path.stem
        gal_str, _, config = stem.partition("_")
        if config not in CONFIG_COLORS:
            warnings.append(f"{stem}: unrecognized configuration {config!r}, skipped")
            continue
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            warnings.append(f"{stem}: no posterior NPZ beside the JSON, skipped")
            continue
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError as exc:
            warnings.append(f"{stem}: unreadable JSON ({exc}), skipped")
            continue
        with np.load(npz_path, allow_pickle=True) as npz:
            missing = [k for k in ("stellar_mass", "sfr_100myr") if k not in npz.files]
            if missing:
                warnings.append(f"{stem}: NPZ lacks {', '.join(missing)}, skipped")
                continue
            try:
                log_mstar = _log_percentiles(np.asarray(npz["stellar_mass"], dtype=float))
                log_sfr = _log_percentiles(np.asarray(npz["sfr_100myr"], dtype=float))
            except ValueError as exc:
                warnings.append(f"{stem}: {exc}, skipped")
                continue
        # The shared rule, not the raw flag. fig05 and fig06 already judge
        # cells through _adoption.is_adopted; reading meta["adoption_pass"]
        # here made this the one figure with its own copy of the criterion --
        # and it is the sample-level figure, the one an adoption rate would be
        # read off, so it is the worst place for the two to drift.
        verdict = is_adopted(meta, config)
        cells.append(
            Cell(
                gal_id=int(gal_str),
                config=config,
                adopted=verdict.adopted,
                low_ess=low_ess_note(meta, verdict),
                log_mstar=log_mstar,
                log_sfr=log_sfr,
            )
        )
    return cells, warnings


def representative_galaxy(cells: list[Cell]) -> int:
    """The galaxy whose intervals are drawn for scale: the one with the most
    adopted cells, ties broken by the median stellar mass so the choice is
    deterministic and does not depend on filesystem ordering."""
    by_gal: dict[int, list[Cell]] = {}
    for cell in cells:
        by_gal.setdefault(cell.gal_id, []).append(cell)
    ranked = sorted(
        by_gal.items(),
        key=lambda kv: (
            -sum(c.adopted for c in kv[1]),
            -float(np.median([c.log_mstar[1] for c in kv[1]])),
        ),
    )
    return ranked[0][0]


def _offsets(
    cells: list[Cell], attr: str
) -> tuple[list[int], dict[int, float], dict[tuple[int, str], float]]:
    """Per-galaxy median of `attr` and each cell's offset from it.

    Returns (galaxy ids ordered by their median, medians, offsets keyed by cell).
    """
    by_gal: dict[int, list[Cell]] = {}
    for cell in cells:
        by_gal.setdefault(cell.gal_id, []).append(cell)
    medians = {
        gid: float(np.median([getattr(c, attr)[1] for c in group]))
        for gid, group in by_gal.items()
    }
    order = sorted(medians, key=lambda gid: medians[gid])
    offsets = {(c.gal_id, c.config): getattr(c, attr)[1] - medians[c.gal_id] for c in cells}
    return order, medians, offsets


def _draw_plane(ax, cells: list[Cell], rep_gal: int) -> None:
    for config in CONFIG_ORDER:
        group = [c for c in cells if c.config == config]
        if not group:
            continue
        adopted = [c for c in group if c.adopted and not c.low_ess]
        low_ess = [c for c in group if c.adopted and c.low_ess]
        held = [c for c in group if not c.adopted]
        if low_ess:
            ax.scatter(
                [c.log_mstar[1] for c in low_ess],
                [c.log_sfr[1] for c in low_ess],
                s=16,
                color=CONFIG_COLORS[config],
                edgecolors=LOW_ESS_EDGE,
                linewidths=0.9,
                zorder=4,
            )
        if adopted:
            ax.scatter(
                [c.log_mstar[1] for c in adopted],
                [c.log_sfr[1] for c in adopted],
                s=16,
                color=CONFIG_COLORS[config],
                edgecolors="none",
                zorder=3,
            )
        if held:
            ax.scatter(
                [c.log_mstar[1] for c in held],
                [c.log_sfr[1] for c in held],
                s=16,
                facecolors="none",
                edgecolors=CONFIG_COLORS[config],
                linewidths=0.7,
                zorder=2,
            )

    for cell in [c for c in cells if c.gal_id == rep_gal]:
        ax.errorbar(
            cell.log_mstar[1],
            cell.log_sfr[1],
            xerr=[
                [cell.log_mstar[1] - cell.log_mstar[0]],
                [cell.log_mstar[2] - cell.log_mstar[1]],
            ],
            yerr=[[cell.log_sfr[1] - cell.log_sfr[0]], [cell.log_sfr[2] - cell.log_sfr[1]]],
            fmt="none",
            ecolor=CONFIG_COLORS[cell.config],
            elinewidth=0.9,
            capsize=1.5,
            alpha=0.9,
            zorder=4,
        )
    # Held in axes coordinates rather than beside the point: at this density the
    # marker the label describes is surrounded by others, and an offset label
    # either covers them or leaves the panel.
    ax.text(
        0.03,
        0.97,
        f"intervals drawn: galaxy {rep_gal}",
        transform=ax.transAxes,
        fontsize=6.5,
        color="0.25",
        ha="left",
        va="top",
    )

    ax.set_xlabel(r"$\log_{10}(M_\star\,/\,M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{SFR}_{100\,\mathrm{Myr}}\,/\,M_\odot\,\mathrm{yr}^{-1})$")
    ax.tick_params(labelsize=7)


def _draw_offsets(ax, cells: list[Cell], attr: str, ylabel: str, show_xlabel: bool) -> float:
    order, medians, offsets = _offsets(cells, attr)
    index = {gid: i for i, gid in enumerate(order)}
    # Spread across the sample, for scale: the standard deviation of the
    # per-galaxy medians. This is what the model-choice spread is measured against.
    sample_sigma = float(np.std([medians[g] for g in order], ddof=1)) if len(order) > 1 else 0.0
    ax.axhspan(-sample_sigma, sample_sigma, color="0.85", zorder=0, linewidth=0)
    ax.axhline(0.0, color="0.55", linewidth=0.6, zorder=1)
    for cell in cells:
        ax.scatter(
            index[cell.gal_id],
            offsets[(cell.gal_id, cell.config)],
            s=11,
            color=CONFIG_COLORS[cell.config],
            edgecolors=(
                LOW_ESS_EDGE
                if cell.low_ess
                else ("none" if cell.adopted else CONFIG_COLORS[cell.config])
            ),
            facecolors=CONFIG_COLORS[cell.config] if cell.adopted else "none",
            linewidths=0.9 if cell.low_ess else (0.0 if cell.adopted else 0.7),
            zorder=4 if cell.low_ess else 3,
        )
    # An offset is measured against that galaxy's OWN median, so a galaxy with
    # one cell contributes exactly zero -- the value minus itself. Until a
    # second configuration lands, every marker sits on the zero line by
    # construction and the panel shows twenty galaxies in perfect agreement
    # across configurations it does not have. The completeness stamp says the
    # grid is partial; it does not say this panel cannot mean anything yet, and
    # a reader looking at a flat row of points inside a tolerance band will not
    # infer it.
    per_galaxy = Counter(cell.gal_id for cell in cells)
    if per_galaxy and max(per_galaxy.values()) < 2:
        ax.text(
            0.5,
            0.5,
            "one configuration per galaxy:\nevery offset is zero by construction",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.5,
            color="0.35",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.6},
            zorder=6,
        )

    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.8, len(order) - 0.2)
    if show_xlabel:
        ax.set_xlabel("galaxy, ordered by cross-configuration median", fontsize=7.5)
    else:
        ax.set_xticklabels([])
    return sample_sigma


def build_figure(cells: list[Cell]) -> tuple[plt.Figure, dict]:
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.15], hspace=0.12, wspace=0.30)
    ax_plane = fig.add_subplot(grid[:, 0])
    ax_mass = fig.add_subplot(grid[0, 1])
    ax_sfr = fig.add_subplot(grid[1, 1], sharex=ax_mass)

    rep_gal = representative_galaxy(cells)
    _draw_plane(ax_plane, cells, rep_gal)
    sigma_mass = _draw_offsets(ax_mass, cells, "log_mstar", r"$\Delta \log M_\star$", False)
    sigma_sfr = _draw_offsets(ax_sfr, cells, "log_sfr", r"$\Delta \log \mathrm{SFR}$", True)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4, color=CONFIG_COLORS[c], label=c)
        for c in CONFIG_ORDER
        if any(cell.config == c for cell in cells)
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="0.4",
            label="not adopted",
        )
    )
    if any(cell.low_ess for cell in cells):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4,
                markerfacecolor="0.6",
                markeredgecolor=LOW_ESS_EDGE,
                markeredgewidth=0.9,
                label="too few ESS",
            )
        )
    ax_plane.legend(
        handles=handles,
        fontsize=6.5,
        ncol=2,
        frameon=False,
        loc="best",
        handletextpad=0.4,
        columnspacing=1.0,
    )

    stats = {
        "n_cells": len(cells),
        "n_galaxies": len({c.gal_id for c in cells}),
        "n_adopted": sum(c.adopted for c in cells),
        "representative_galaxy": rep_gal,
        "sample_sigma_log_mstar": sigma_mass,
        "sample_sigma_log_sfr": sigma_sfr,
    }
    return fig, stats


def model_choice_spread(cells: list[Cell], attr: str) -> dict:
    """Per-galaxy max-min and stdev across configurations, summarized over galaxies."""
    by_gal: dict[int, list[float]] = {}
    for cell in cells:
        by_gal.setdefault(cell.gal_id, []).append(getattr(cell, attr)[1])
    ranges = [max(v) - min(v) for v in by_gal.values() if len(v) > 1]
    stdevs = [float(np.std(v, ddof=1)) for v in by_gal.values() if len(v) > 1]
    if not ranges:
        return {}
    return {
        "n_galaxies_multi_config": len(ranges),
        "range_min": min(ranges),
        "range_median": float(np.median(ranges)),
        "range_max": max(ranges),
        "stdev_min": min(stdevs),
        "stdev_median": float(np.median(stdevs)),
        "stdev_max": max(stdevs),
    }


def _git_describe() -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=CANONICAL_RESULTS,
        help="Directory of per-cell JSON + NPZ fits (default: the canonical grid).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "figures" / "fig09_sample_level.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        return 2

    cells, warnings = load_cells(results_dir)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not cells:
        print(f"ERROR: no readable cells in {results_dir}", file=sys.stderr)
        return 2

    is_canonical = results_dir == CANONICAL_RESULTS.resolve()

    # Do the cells hold the configurations this repository declares? A
    # redefined configuration leaves its old cells in place under their old
    # names, and the colors and labels below would then say Configuration IV
    # over a model that is not Configuration IV.
    mismatches, notes = audit(results_dir, CONFIGS)
    audit_text = banner(results_dir, mismatches, notes)
    if audit_text:
        print(audit_text, file=sys.stderr)

    # Is the whole declared sample here? The directory check below answers
    # only where the cells came from. A canonical directory holding five of
    # a hundred and twenty cells passes it and is not the figure the caption
    # describes, so completeness is asked separately and against the
    # committed selection rather than against a literal count.
    shortfall = None
    try:
        expected_ids = load_expected_galaxy_ids(SELECTION_20)
        shortfall = completeness_note(
            ((cell.gal_id, cell.config) for cell in cells), expected_ids, CONFIG_ORDER
        )
    except (OSError, ValueError, KeyError) as exc:
        shortfall = f"COMPLETENESS UNVERIFIED - cannot read {SELECTION_20.name}: {exc}"

    # A cell adopted on too few effective samples is not a wrong point, it is
    # an uninformative one, and it is about to be drawn indistinguishably from
    # the rest and counted in the adoption rate. Name it on the figure.
    frozen = [f"{c.gal_id}/{c.config}" for c in cells if c.low_ess]
    if frozen:
        print(f"low effective sample size, adopted anyway: {', '.join(frozen)}", file=sys.stderr)
    if shortfall:
        print(shortfall, file=sys.stderr)
    if not is_canonical:
        print(
            f"NOT THE PRODUCTION GRID -- rendered from {results_dir.name} at {_git_describe()}",
            file=sys.stderr,
        )
    if mismatches:
        print(
            "CONFIGURATION LABELS ARE NOT configs.py's: "
            + "; ".join(f"{m.config} sampled {m.found_prefixes[0]}" for m in mismatches),
            file=sys.stderr,
        )

    fig, stats = build_figure(cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png_path = args.out.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"cells read           : {stats['n_cells']}")
    print(f"galaxies             : {stats['n_galaxies']}")
    print(f"adopted              : {stats['n_adopted']} of {stats['n_cells']}")
    print(f"representative galaxy: {stats['representative_galaxy']}")
    print(f"sample sigma log M*  : {stats['sample_sigma_log_mstar']:.3f} dex")
    print(f"sample sigma log SFR : {stats['sample_sigma_log_sfr']:.3f} dex")
    for attr, label in (("log_mstar", "log M*"), ("log_sfr", "log SFR")):
        spread = model_choice_spread(cells, attr)
        if spread:
            print(
                f"model-choice {label:<7}: range {spread['range_min']:.3f}-{spread['range_max']:.3f} "
                f"(median {spread['range_median']:.3f}), stdev median {spread['stdev_median']:.3f} dex, "
                f"over {spread['n_galaxies_multi_config']} galaxies"
            )
    print(f"wrote {args.out}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
