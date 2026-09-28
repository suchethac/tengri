#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 11 - Bayesian model averaging over Laplace evidences.

This figure demonstrates BMA with a flat prior over valid models within each weight set.
Panel (a): flow diagram of the evidence→weights→posterior pathway.
Panel (b): per-galaxy BMA weights as a heatmap, plus marginal weights for the factorial set.
Panel (c): BMA-averaged stellar mass and SFR versus single-configuration posteriors.

CLI:
    python analysis/paper1/fig11_bma.py [--summary PATH] [--out PDF] [--set SET] [--data-out JSON]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figure_style import CONFIG_COLORS

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_WIDTH = 7.1
FIGURE_HEIGHT = 11.0

# X-like configuration mapping
XLIKE_CODE = {
    "cigale_like": "CIGALE",
    "prospector_like": "Prospector",
    "bagpipes_like": "BAGPIPES",
    "beagle_like": "BEAGLE",
    "dense_basis_like": "Dense_Basis",
}

# Color palette for X-like codes (distinct from CONFIG_COLORS)
XLIKE_COLORS = {
    "cigale_like": "#999999",  # dark gray
    "prospector_like": "#666666",  # medium gray
    "bagpipes_like": "#444444",  # darker gray
    "beagle_like": "#AAAAAA",  # light gray
    "dense_basis_like": "#777777",  # medium-dark gray
}


@dataclass(frozen=True)
class DrawSummary:
    """Summary of what was drawn in the figure."""

    galaxies_drawn: list[int]
    models_drawn: list[str]
    sets_present: list[str]
    close_galaxies: list[int]
    excluded_models: dict[str, list[tuple[str, str]]]  # set_name -> [(model_key, reason)]
    invalid_counts: dict[str, dict[str, int]]  # set_name -> {model_key: count}
    prior_mass: dict[str, Any]  # Prior mass specification
    close_disagreement: list[str]  # Warnings about close flag disagreement


def _get_model_color(model_key: str) -> str:
    """Return color for a model key."""
    if model_key in CONFIG_COLORS:
        return CONFIG_COLORS[model_key]
    if model_key in XLIKE_COLORS:
        return XLIKE_COLORS[model_key]
    # Fallback for unknown keys
    return "#cccccc"


def _validate_weights_sum(weights: dict[str, float], galaxy_id: int, set_name: str) -> None:
    """Validate that valid weights sum to 1 within tolerance.

    Raises ValueError if not.
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Galaxy {galaxy_id} set {set_name}: valid weights sum to {total:.9f}, "
            f"not 1.0 within 1e-6 tolerance"
        )


def _filter_configuration_vi(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove Configuration VI entries and warn if found."""
    removed_count = 0

    for gal_id_str in summary.get("galaxies", {}):
        gal_data = summary["galaxies"][gal_id_str]
        for set_name in list(gal_data.get("sets", {}).keys()):
            w_set = gal_data["sets"][set_name]
            models = w_set.get("models", [])
            # Filter out VI entries
            original_count = len(models)
            models[:] = [m for m in models if m.get("config") != "VI"]
            removed_count += original_count - len(models)

    if removed_count > 0:
        print(
            f"Dropped {removed_count} Configuration VI entries (not part of Section 7)",
            file=sys.stderr,
        )

    return summary


def load_summary(path: Path) -> dict[str, Any]:
    """Load and validate the BMA summary JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    with open(path) as f:
        summary = json.load(f)

    # Basic validation
    if "galaxies" not in summary:
        raise ValueError("Summary missing 'galaxies' key")
    if not summary["galaxies"]:
        raise ValueError("Summary contains no galaxies")

    # Filter out Configuration VI
    summary = _filter_configuration_vi(summary)

    return summary


def _draw_flow_panel(ax: plt.Axes) -> None:
    """Draw panel (a): flow diagram of evidence→weights→posterior."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Box styling
    box_color = "#f0f0f0"
    box_edge = "#333333"
    font_size = 8

    # Define boxes: (x, y, width, height, label)
    boxes = [
        (0.05, 0.75, 0.2, 0.15, "Model configs\n+ noise"),
        (0.05, 0.50, 0.2, 0.15, "Laplace\nMAP fit"),
        (0.05, 0.25, 0.2, 0.15, "Per-config\nevidence"),
        (0.35, 0.50, 0.2, 0.15, "Model\nweights"),
        (0.65, 0.50, 0.25, 0.15, "BMA\nposterior"),
    ]

    for x, y, w, h, label in boxes:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01",
            facecolor=box_color,
            edgecolor=box_edge,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            weight="normal",
        )

    # Arrows
    arrow_props = dict(arrowstyle="-|>", lw=0.8, color=box_edge)
    arrows = [
        ((0.15, 0.50), (0.15, 0.65)),  # configs → laplace
        ((0.15, 0.40), (0.15, 0.25)),  # laplace → evidence
        ((0.25, 0.325), (0.35, 0.575)),  # evidence → weights
        ((0.55, 0.575), (0.65, 0.575)),  # weights → posterior
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, **arrow_props)
        ax.add_patch(arrow)


def _draw_weights_panel(
    ax: plt.Axes,
    summary: dict[str, Any],
    selected_set: str,
) -> tuple[list[int], list[str], list[int], list[tuple[str, str]]]:
    """Draw panel (b): per-galaxy BMA weights heatmap and excluded/invalid cells.

    Returns: (galaxies_drawn, models_drawn, close_galaxies, excluded_with_reasons)
    """
    galaxies = sorted(int(gal_id) for gal_id in summary["galaxies"])

    # Find the weight set
    weight_set = None
    for gal_id_str in summary["galaxies"]:
        gal_data = summary["galaxies"][gal_id_str]
        if "sets" in gal_data and selected_set in gal_data["sets"]:
            weight_set = gal_data["sets"][selected_set]
            break

    if weight_set is None:
        # No data for this set
        ax.text(
            0.5,
            0.5,
            f"No data for set: {selected_set}",
            ha="center",
            va="center",
            fontsize=10,
        )
        return galaxies, [], [], []

    # Collect models from all galaxies in this set
    all_models = set()
    weight_matrix = {}
    validity_matrix = {}  # Track which cells are valid/invalid
    close_galaxies = []
    excluded_with_reasons = []

    for gal_id in galaxies:
        gal_id_str = str(gal_id)
        if gal_id_str in summary["galaxies"]:
            gal_data = summary["galaxies"][gal_id_str]
            if "sets" in gal_data and selected_set in gal_data["sets"]:
                w_set = gal_data["sets"][selected_set]
                models = w_set.get("models", [])

                weights = {}
                validity = {}  # True for valid, False for invalid/excluded

                for model in models:
                    model_key = model.get("model_key")
                    weight = model.get("weight")
                    valid = model.get("valid", True)

                    if model_key:
                        all_models.add(model_key)
                        validity[model_key] = valid
                        if valid and weight is not None:
                            weights[model_key] = weight
                        elif not valid:
                            excluded_with_reasons.append(
                                (model_key, model.get("excluded_reason", "unknown"))
                            )

                # Validate that valid weights sum to 1
                try:
                    _validate_weights_sum(weights, gal_id, selected_set)
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc

                weight_matrix[gal_id] = weights
                validity_matrix[gal_id] = validity

                # Check if close (read from set data)
                is_close = w_set.get("close", False)
                if is_close:
                    close_galaxies.append(gal_id)

    if not all_models:
        ax.text(0.5, 0.5, "No valid model weights found", ha="center", va="center", fontsize=10)
        return galaxies, [], close_galaxies, excluded_with_reasons

    models_sorted = sorted(all_models)

    # Create heatmap data
    n_gal = len(galaxies)
    n_mod = len(models_sorted)
    heatmap = np.full((n_mod, n_gal), np.nan)
    hatch_matrix = np.zeros((n_mod, n_gal), dtype=bool)  # Track which cells should be hatched

    for gal_idx, gal_id in enumerate(galaxies):
        if gal_id in weight_matrix:
            for mod_idx, model_key in enumerate(models_sorted):
                if gal_id in validity_matrix and model_key in validity_matrix[gal_id]:
                    is_valid = validity_matrix[gal_id][model_key]
                    if not is_valid:
                        # Excluded/invalid cell
                        hatch_matrix[mod_idx, gal_idx] = True
                        heatmap[mod_idx, gal_idx] = 0.0  # Use 0 for display
                    else:
                        # Valid cell
                        heatmap[mod_idx, gal_idx] = weight_matrix[gal_id].get(model_key, 0.0)

    # Plot heatmap
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Add hatching for excluded/invalid cells
    for mod_idx in range(n_mod):
        for gal_idx in range(n_gal):
            if hatch_matrix[mod_idx, gal_idx]:
                ax.add_patch(
                    plt.Rectangle(
                        (gal_idx - 0.5, mod_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="gray",
                        linewidth=0.5,
                    )
                )

    ax.set_xticks(range(n_gal))
    ax.set_yticks(range(n_mod))
    ax.set_xticklabels([str(g) for g in galaxies], fontsize=7, rotation=45)
    ax.set_yticklabels(models_sorted, fontsize=7)
    ax.set_xlabel("Galaxy ID", fontsize=8)
    ax.set_ylabel("Model", fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weight", fontsize=7)

    return galaxies, models_sorted, close_galaxies, excluded_with_reasons


def _draw_posterior_panel(
    ax: plt.Axes,
    summary: dict[str, Any],
    selected_set: str,
    quantity: str,  # "log_stellar_mass_survived" or "log_sfr_100myr"
) -> None:
    """Draw panel (c): BMA vs single-config posteriors for a quantity."""
    galaxies = sorted(int(gal_id) for gal_id in summary["galaxies"])

    # Collect data
    x_pos = np.arange(len(galaxies))

    for gal_idx, gal_id in enumerate(galaxies):
        gal_id_str = str(gal_id)
        if gal_id_str not in summary["galaxies"]:
            continue

        gal_data = summary["galaxies"][gal_id_str]
        if "sets" not in gal_data or selected_set not in gal_data["sets"]:
            continue

        w_set = gal_data["sets"][selected_set]
        models = w_set.get("models", [])

        # Draw single-config posteriors
        for model in models:
            valid = model.get("valid", True)
            if not valid:
                continue

            percentiles = model.get("percentiles", {})
            if quantity not in percentiles:
                continue

            p16, p50, p84 = percentiles[quantity]
            config = model.get("config")
            if config:
                color = _get_model_color(config)
                ax.errorbar(
                    gal_idx,
                    p50,
                    yerr=[[p50 - p16], [p84 - p50]],
                    fmt="o",
                    markersize=4,
                    color=color,
                    alpha=0.7,
                    linewidth=0.8,
                    capsize=2,
                )

        # Draw BMA posterior
        bma_perc = w_set.get("bma_percentiles", {})
        if quantity in bma_perc:
            p16, p50, p84 = bma_perc[quantity]
            ax.errorbar(
                gal_idx,
                p50,
                yerr=[[p50 - p16], [p84 - p50]],
                fmt="D",
                markersize=5,
                color="black",
                markeredgecolor="black",
                linewidth=1,
                capsize=2,
                zorder=10,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in galaxies], fontsize=7, rotation=45)
    ax.set_xlabel("Galaxy ID", fontsize=8)

    if quantity == "log_stellar_mass_survived":
        ax.set_ylabel(r"$\log M_{\star, survived}$ [M$_{\odot}$]", fontsize=8)
    elif quantity == "log_sfr_100myr":
        ax.set_ylabel(
            r"$\log \mathrm{SFR}_{100\,\mathrm{Myr}}$ [M$_{\odot}$ yr$^{-1}$]", fontsize=8
        )
    else:
        ax.set_ylabel(quantity, fontsize=8)

    ax.tick_params(labelsize=7)


def build_figure(
    summary: dict[str, Any],
    selected_set: str,
) -> tuple[plt.Figure, DrawSummary]:
    """Build the three-panel BMA figure."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.5, 2.0], hspace=0.35)

    # Panel (a): flow diagram
    ax_flow = fig.add_subplot(gs[0])
    _draw_flow_panel(ax_flow)
    ax_flow.text(
        0.02, 0.95, "(a)", fontsize=10, weight="bold", transform=ax_flow.transAxes, va="top"
    )

    # Panel (b): weights heatmap
    ax_weights = fig.add_subplot(gs[1])
    galaxies, models, close_gal, excluded = _draw_weights_panel(ax_weights, summary, selected_set)
    ax_weights.text(
        0.02, 0.95, "(b)", fontsize=10, weight="bold", transform=ax_weights.transAxes, va="top"
    )

    # Panel (c): two sub-panels for mass and SFR
    gs_c = gs[2].subgridspec(2, 1, hspace=0.3)
    ax_mass = fig.add_subplot(gs_c[0])
    ax_sfr = fig.add_subplot(gs_c[1])

    _draw_posterior_panel(ax_mass, summary, selected_set, "log_stellar_mass_survived")
    _draw_posterior_panel(ax_sfr, summary, selected_set, "log_sfr_100myr")

    ax_mass.text(
        0.02, 0.95, "(c1)", fontsize=10, weight="bold", transform=ax_mass.transAxes, va="top"
    )
    ax_sfr.text(
        0.02, 0.95, "(c2)", fontsize=10, weight="bold", transform=ax_sfr.transAxes, va="top"
    )

    # Collect excluded models
    excluded_by_set = {}
    for model_key, reason in excluded:
        if selected_set not in excluded_by_set:
            excluded_by_set[selected_set] = []
        excluded_by_set[selected_set].append((model_key, reason))

    # Check for close flag disagreement
    close_disagreement = []
    for gal_id in galaxies:
        gal_id_str = str(gal_id)
        if gal_id_str in summary["galaxies"]:
            gal_data = summary["galaxies"][gal_id_str]
            if "sets" in gal_data and selected_set in gal_data["sets"]:
                w_set = gal_data["sets"][selected_set]
                is_close = w_set.get("close", False)
                models = w_set.get("models", [])
                weights = {
                    m.get("model_key"): m.get("weight")
                    for m in models
                    if m.get("valid", True) and m.get("weight") is not None
                }
                max_weight = max(weights.values()) if weights else 0.0
                computed_close = max_weight < 0.9
                if is_close != computed_close:
                    close_disagreement.append(
                        f"Galaxy {gal_id}: close={is_close} but max_weight={max_weight:.4f} "
                        f"(computed_close={computed_close})"
                    )

    # Collect invalid counts and prior mass
    invalid_counts = {}
    prior_mass = {}
    sets_present = []

    first_gal = next(iter(summary["galaxies"].values()))
    if "sets" in first_gal:
        sets_present = list(first_gal["sets"].keys())
        for set_name in sets_present:
            w_set = first_gal["sets"][set_name]
            if "invalid_counts" in w_set:
                invalid_counts[set_name] = w_set["invalid_counts"]

    if "model_prior" in summary:
        prior_mass = summary["model_prior"]

    draw_summary = DrawSummary(
        galaxies_drawn=galaxies,
        models_drawn=models,
        sets_present=sets_present,
        close_galaxies=close_gal,
        excluded_models=excluded_by_set,
        invalid_counts=invalid_counts,
        prior_mass=prior_mass,
        close_disagreement=close_disagreement,
    )

    return fig, draw_summary


def _git_describe() -> str:
    """Get short git SHA."""
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
        "--summary",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "results" / "bma_summary.json",
        help="Path to BMA summary JSON (default: analysis/paper1/results/bma_summary.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "analysis" / "paper1" / "figures" / "fig11_bma.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--set",
        type=str,
        default="named_all",
        help="Weight set to draw: named_all, named_grid, or factorial (default: named_all).",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=None,
        help="Sidecar JSON listing drawn data.",
    )
    args = parser.parse_args(argv)

    # Load summary
    try:
        summary = load_summary(args.summary)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: Cannot read BMA summary. Did bma_combine.py run?\n"
            f"  Expected: {args.summary}\n"
            f"  Error: {exc}",
            file=sys.stderr,
        )
        return 1

    # Determine which set to use
    selected_set = args.set
    first_gal = next(iter(summary["galaxies"].values()))
    available_sets = list(first_gal.get("sets", {}).keys()) if "sets" in first_gal else []

    if selected_set not in available_sets:
        # Fallback logic
        if "named_all" in available_sets:
            selected_set = "named_all"
            print(
                f"WARNING: Set '{args.set}' not found; using 'named_all'",
                file=sys.stderr,
            )
        elif "named_grid" in available_sets:
            selected_set = "named_grid"
            print(
                f"WARNING: Set '{args.set}' not found; using 'named_grid'",
                file=sys.stderr,
            )
        else:
            print("ERROR: No weight sets found in summary", file=sys.stderr)
            return 1

    # Build figure
    try:
        fig, draw_summary = build_figure(summary, selected_set)
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Failed to build figure: {exc}", file=sys.stderr)
        return 1

    # Warn about close flag disagreement
    if draw_summary.close_disagreement:
        for warning in draw_summary.close_disagreement:
            print(f"WARNING: {warning}", file=sys.stderr)

    # Write figure
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    # Write sidecar JSON if requested
    if args.data_out:
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        sidecar = asdict(draw_summary)
        sidecar["summary_file"] = str(args.summary)
        sidecar["selected_set"] = selected_set
        sidecar["code_revision"] = _git_describe()
        with open(args.data_out, "w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"wrote {args.data_out}")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
