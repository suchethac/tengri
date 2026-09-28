#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 11 - Bayesian model averaging over Laplace evidences.

This figure demonstrates BMA with a flat prior over valid models within each weight set.
Panel (a): flow diagram of the evidence -> weights -> posterior pathway.
Panel (b1): per-galaxy weights for the selected named set as a heatmap (models x galaxies),
    with excluded/invalid cells hatched and a small marker on cells whose NUTS chain failed
    adoption even though the Laplace evidence that entered BMA was fine.
Panel (b2): factorial marginal weights, one small heatmap per component axis (sfh, ssp,
    isochrone, spectral_library, attenuation; galaxies on x, axis values on y). The color
    encodes log2(weight / prior) on a diverging scale centered at zero, not the raw weight,
    because a marginal weight only means something in comparison to what a flat prior over
    the 100-model grid already predicts for that axis value. The raw weight and the prior
    used are recorded in the sidecar JSON rather than annotated in-cell, since there is no
    room for text at this width once galaxies and axis values both fill an axis. A galaxy
    with no factorial weight set is drawn as a hatched column in every row.
Panel (c1)/(c2): BMA-averaged stellar mass (survived) and SFR (100 Myr) versus each
    single-configuration posterior, for the selected named set.

Schema note: at the time this figure was written, ``bma_combine.py`` at HEAD was still the
CLI placeholder (no ``combine_bma``); this file therefore consumes the summary schema from
the orchestrator's shared spec (SPEC.md) rather than reverse-engineering an uncommitted,
in-progress combiner. The factorial "marginal" field is read as
``{axis: {value: {"weight": w, "prior": p}}}`` per that spec, but a bare
``{axis: {value: w}}`` (weight only, no prior) is also accepted -- the prior then falls back
to the flat-prior design constant for the balanced 100-model grid (see DEFAULT_PRIOR_MASS).
Both "invalid_counts" and "galaxies_per_model" are read from wherever they are found: a
top-level key, a top-level "sets_summary" keyed by set name, or the first galaxy's own
per-set dict, in that order.

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
FIGURE_HEIGHT = 16.5

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

#: Order panel (b2) draws the factorial marginal heatmaps in.
FACTORIAL_MARGINAL_AXES: tuple[str, ...] = (
    "sfh",
    "ssp",
    "isochrone",
    "spectral_library",
    "attenuation",
)

#: Canonical, domain-meaningful row order per axis (unknown values sort after, alphabetically).
_AXIS_VALUE_ORDER: dict[str, tuple[str, ...]] = {
    "sfh": ("continuity", "dirichlet", "delayed", "dpl", "lnorm"),
    "ssp": ("mist_c3k", "mist_miles", "prsc_c3k", "prsc_miles", "bpass_c3k"),
    "isochrone": ("mist", "prsc", "bpass"),
    "spectral_library": ("c3k", "miles"),
    "attenuation": ("calzetti", "smc", "kriek_conroy_2c", "cf00_2c"),
}

#: Prior mass implied by a flat prior over the balanced 100-model factorial grid
#: (AMENDMENT 1). Used only when a marginal entry does not carry its own "prior".
DEFAULT_PRIOR_MASS: dict[str, dict[str, float]] = {
    "sfh": {"continuity": 0.2, "dirichlet": 0.2, "delayed": 0.2, "dpl": 0.2, "lnorm": 0.2},
    "ssp": {
        "mist_c3k": 0.2,
        "mist_miles": 0.2,
        "prsc_c3k": 0.2,
        "prsc_miles": 0.2,
        "bpass_c3k": 0.2,
    },
    "attenuation": {"calzetti": 0.25, "smc": 0.25, "kriek_conroy_2c": 0.25, "cf00_2c": 0.25},
    "isochrone": {"mist": 0.4, "prsc": 0.4, "bpass": 0.2},
    "spectral_library": {"c3k": 0.6, "miles": 0.4},
}

#: log2(weight / prior) beyond +/-3 (an 8x over- or under-weighting) is clipped for color
#: only; the sidecar always records the unclipped weight and prior.
LOG2_RATIO_CLIP = 3.0


@dataclass(frozen=True)
class DrawSummary:
    """Summary of what was drawn in the figure."""

    galaxies_drawn: list[int]
    models_drawn: list[str]
    sets_present: list[str]
    close_galaxies: list[int]
    excluded_models: dict[str, list[tuple[str, str]]]  # set_name -> [(model_key, reason)]
    invalid_counts: dict[str, dict[str, int]]  # set_name -> {model_key: count}
    galaxies_per_model: dict[str, dict[str, int]]  # set_name -> {model_key: n galaxies}
    prior_mass: dict[str, Any]  # Prior mass specification
    close_disagreement: list[str]  # Warnings about close flag disagreement
    factorial_marginals: dict[str, Any]  # axis -> {value: {"prior", "weight", "log2_ratio"}}


def _resolve_marginal_weight_prior(
    entry: Any, axis: str, value: str
) -> tuple[float | None, float | None]:
    """Return (weight, prior) for one factorial axis value.

    ``entry`` is either ``{"weight": w, "prior": p}`` (the documented combiner schema) or a
    bare float weight (a combiner that has not yet started reporting priors). The prior is
    read from the entry when present; otherwise it falls back to DEFAULT_PRIOR_MASS, the
    flat-prior design constant for the balanced 100-model factorial grid.
    """
    if entry is None:
        return None, DEFAULT_PRIOR_MASS.get(axis, {}).get(value)

    if isinstance(entry, dict):
        weight = entry.get("weight")
        prior = entry.get("prior")
        if prior is None:
            prior = DEFAULT_PRIOR_MASS.get(axis, {}).get(value)
        return weight, prior

    # Bare float: weight only, prior falls back to the design constant.
    return float(entry), DEFAULT_PRIOR_MASS.get(axis, {}).get(value)


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
        (0.05, 0.76, 0.2, 0.18, "Model configs\n+ noise"),
        (0.05, 0.50, 0.2, 0.18, "Laplace\nMAP fit"),
        (0.05, 0.24, 0.2, 0.18, "Per-config\nevidence"),
        (0.35, 0.50, 0.2, 0.18, "Model\nweights"),
        (0.65, 0.50, 0.25, 0.18, "BMA\nposterior"),
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
    # Arrow endpoints sit in the gaps BETWEEN boxes (box top/bottom edges), not inside a box.
    arrow_props = dict(arrowstyle="-|>", lw=0.8, color=box_edge)
    arrows = [
        ((0.15, 0.76), (0.15, 0.68)),  # configs -> laplace
        ((0.15, 0.50), (0.15, 0.42)),  # laplace -> evidence
        ((0.25, 0.33), (0.35, 0.59)),  # evidence -> weights
        ((0.55, 0.59), (0.65, 0.59)),  # weights -> posterior
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, **arrow_props)
        ax.add_patch(arrow)


def _draw_weights_panel(
    ax: plt.Axes,
    summary: dict[str, Any],
    selected_set: str,
) -> tuple[list[int], list[str], list[int], list[tuple[str, str]]]:
    """Draw panel (b1): per-galaxy BMA weights heatmap and excluded/invalid cells.

    Excluded/invalid cells are hatched. A valid cell whose named configuration's NUTS chain
    failed adoption -- which does not gate BMA, since a named configuration's evidence is its
    own Laplace run -- is marked with a small "x" so a reader can tell the two failure modes
    apart at a glance.

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
    nuts_fail_matrix_by_gal = {}  # gal_id -> {model_key: True if adoption explicitly failed}
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
                nuts_fail = {}

                for model in models:
                    model_key = model.get("model_key")
                    weight = model.get("weight")
                    valid = model.get("valid", True)
                    nuts_pass = model.get("nuts_adoption_pass")

                    if model_key:
                        all_models.add(model_key)
                        validity[model_key] = valid
                        nuts_fail[model_key] = valid and nuts_pass is False
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
                nuts_fail_matrix_by_gal[gal_id] = nuts_fail

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
    nuts_fail_matrix = np.zeros((n_mod, n_gal), dtype=bool)

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
                        if nuts_fail_matrix_by_gal.get(gal_id, {}).get(model_key, False):
                            nuts_fail_matrix[mod_idx, gal_idx] = True

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
            if nuts_fail_matrix[mod_idx, gal_idx]:
                ax.text(
                    gal_idx,
                    mod_idx,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color="black",
                )

    ax.set_xticks(range(n_gal))
    ax.set_yticks(range(n_mod))
    ax.set_xticklabels([str(g) for g in galaxies], fontsize=6, rotation=90)
    ax.set_yticklabels(models_sorted, fontsize=7)
    ax.set_xlabel("Galaxy ID", fontsize=8)
    ax.set_ylabel("Model", fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Weight", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.text(
        0.0,
        1.04,
        "hatched = excluded/invalid   x = NUTS adoption failed (Laplace evidence still used)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        style="italic",
    )

    return galaxies, models_sorted, close_galaxies, excluded_with_reasons


def _draw_factorial_marginal_panel(
    fig: plt.Figure,
    gs_cell: Any,
    summary: dict[str, Any],
    galaxies: list[int],
) -> dict[str, Any]:
    """Draw panel (b2): factorial marginal weight vs. prior, one heatmap row per axis.

    Rows are drawn in FACTORIAL_MARGINAL_AXES order (sfh, ssp, isochrone, spectral_library,
    attenuation), sharing the galaxy x-axis with panel (b1) above. The color encodes
    log2(weight / prior) on a diverging colormap centered at zero -- a marginal weight only
    means something in comparison to what a flat prior over the 100-model grid already
    predicts for that axis value, so the ratio is the meaningful quantity, not the raw
    weight. The raw weight and the prior actually used are written to the sidecar instead of
    annotated in-cell: with up to 5 axis values and dozens of galaxies on one row, there is no
    legible room for per-cell numbers.

    A galaxy with no "factorial" weight set is drawn as a hatched column in every row rather
    than silently dropped, so a reader can see which galaxies were not fit against the full
    grid instead of a panel that looks complete but is not.

    Returns a sidecar dict:
        {
            axis: {value: {"prior": p, "weight": {gal_id_str: w}, "log2_ratio": {gal_id_str: r}}},
            "galaxies_missing_factorial": [gal_id, ...],
        }
    """
    sidecar: dict[str, Any] = {axis: {} for axis in FACTORIAL_MARGINAL_AXES}

    missing_galaxies = [
        gid
        for gid in galaxies
        if "factorial" not in summary["galaxies"].get(str(gid), {}).get("sets", {})
    ]
    sidecar["galaxies_missing_factorial"] = missing_galaxies

    if len(missing_galaxies) == len(galaxies):
        ax = fig.add_subplot(gs_cell)
        ax.text(0.5, 0.5, "No factorial set in summary", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return sidecar

    n_gal = len(galaxies)
    # Row heights are proportional to how many values each axis has, so a 2-value axis
    # (spectral_library) isn't stretched to match a 5-value one (sfh, ssp).
    row_heights = [
        max(len(_AXIS_VALUE_ORDER.get(axis, ())), 1) for axis in FACTORIAL_MARGINAL_AXES
    ]
    gs_axes = gs_cell.subgridspec(
        len(FACTORIAL_MARGINAL_AXES), 1, height_ratios=row_heights, hspace=0.55
    )
    created_axes: list[plt.Axes] = []
    shared_ax: plt.Axes | None = None
    last_im = None

    for row, axis in enumerate(FACTORIAL_MARGINAL_AXES):
        ax = fig.add_subplot(gs_axes[row], sharex=shared_ax)
        created_axes.append(ax)
        if shared_ax is None:
            shared_ax = ax

        per_gal_entry: dict[int, dict[str, Any]] = {}
        values_seen: set[str] = set()
        for gid in galaxies:
            if gid in missing_galaxies:
                continue
            gal_data = summary["galaxies"].get(str(gid), {})
            w_set = gal_data.get("sets", {}).get("factorial", {})
            axis_marginal = w_set.get("marginal", {}).get(axis, {})
            per_gal_entry[gid] = axis_marginal
            values_seen.update(axis_marginal.keys())

        ordered_values = [v for v in _AXIS_VALUE_ORDER.get(axis, ()) if v in values_seen]
        ordered_values += sorted(values_seen - set(ordered_values))
        if not ordered_values:
            ordered_values = list(_AXIS_VALUE_ORDER.get(axis, ()))

        n_val = len(ordered_values)
        ratio_matrix = np.full((n_val, n_gal), np.nan)
        hatch_cells = np.zeros((n_val, n_gal), dtype=bool)

        for val_idx, value in enumerate(ordered_values):
            value_sidecar = sidecar[axis].setdefault(
                value, {"prior": None, "weight": {}, "log2_ratio": {}}
            )
            for gal_idx, gid in enumerate(galaxies):
                if gid in missing_galaxies:
                    hatch_cells[val_idx, gal_idx] = True
                    continue

                entry = per_gal_entry.get(gid, {}).get(value)
                weight, prior = _resolve_marginal_weight_prior(entry, axis, value)

                if value_sidecar["prior"] is None and prior is not None:
                    value_sidecar["prior"] = prior
                if weight is not None:
                    value_sidecar["weight"][str(gid)] = weight

                if weight is None or not prior:
                    hatch_cells[val_idx, gal_idx] = True
                    continue

                if weight <= 0:
                    log2_ratio = -LOG2_RATIO_CLIP
                else:
                    log2_ratio = float(np.log2(weight / prior))
                value_sidecar["log2_ratio"][str(gid)] = log2_ratio
                ratio_matrix[val_idx, gal_idx] = np.clip(
                    log2_ratio, -LOG2_RATIO_CLIP, LOG2_RATIO_CLIP
                )

        im = ax.imshow(
            ratio_matrix,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-LOG2_RATIO_CLIP,
            vmax=LOG2_RATIO_CLIP,
        )
        last_im = im

        for val_idx in range(n_val):
            for gal_idx in range(n_gal):
                if hatch_cells[val_idx, gal_idx]:
                    ax.add_patch(
                        plt.Rectangle(
                            (gal_idx - 0.5, val_idx - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="///",
                            edgecolor="gray",
                            linewidth=0.4,
                        )
                    )

        ax.set_yticks(range(n_val))
        ax.set_yticklabels(ordered_values, fontsize=6)
        ax.set_ylabel(axis, fontsize=7, rotation=0, ha="right", va="center", labelpad=8)
        ax.tick_params(axis="y", length=2)

        if row < len(FACTORIAL_MARGINAL_AXES) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xticks(range(n_gal))
            ax.set_xticklabels([str(g) for g in galaxies], fontsize=6, rotation=90)
            ax.set_xlabel("Galaxy ID", fontsize=8)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=created_axes, fraction=0.025, pad=0.02)
        cbar.set_label(r"$\log_2$(weight / prior)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    return sidecar


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
    """Build the four-panel BMA figure: (a) flow, (b1) named weights, (b2) factorial
    marginals, (c1)/(c2) BMA vs. single-config mass and SFR posteriors."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.3, 1.7, 2.3, 2.3], hspace=0.5)

    # Panel (a): flow diagram
    ax_flow = fig.add_subplot(gs[0])
    _draw_flow_panel(ax_flow)
    ax_flow.text(
        0.02, 0.95, "(a)", fontsize=10, weight="bold", transform=ax_flow.transAxes, va="top"
    )

    # Panel (b1): named-set weights heatmap
    ax_weights = fig.add_subplot(gs[1])
    galaxies, models, close_gal, excluded = _draw_weights_panel(ax_weights, summary, selected_set)
    ax_weights.text(
        0.02, 1.15, "(b1)", fontsize=10, weight="bold", transform=ax_weights.transAxes, va="top"
    )

    # Panel (b2): factorial marginals vs. prior, always drawn against the "factorial" set
    # regardless of which named set (b1) is showing.
    factorial_marginals = _draw_factorial_marginal_panel(fig, gs[2], summary, galaxies)
    fig.text(0.02, gs[2].get_position(fig).y1 + 0.005, "(b2)", fontsize=10, weight="bold")

    # Panel (c): two sub-panels for mass and SFR
    gs_c = gs[3].subgridspec(2, 1, hspace=0.6)
    ax_mass = fig.add_subplot(gs_c[0])
    ax_sfr = fig.add_subplot(gs_c[1])

    _draw_posterior_panel(ax_mass, summary, selected_set, "log_stellar_mass_survived")
    _draw_posterior_panel(ax_sfr, summary, selected_set, "log_sfr_100myr")

    ax_mass.text(
        0.02, 1.08, "(c1)", fontsize=10, weight="bold", transform=ax_mass.transAxes, va="top"
    )
    ax_sfr.text(
        0.02, 1.08, "(c2)", fontsize=10, weight="bold", transform=ax_sfr.transAxes, va="top"
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
                set_models = w_set.get("models", [])
                weights = {
                    m.get("model_key"): m.get("weight")
                    for m in set_models
                    if m.get("valid", True) and m.get("weight") is not None
                }
                max_weight = max(weights.values()) if weights else 0.0
                computed_close = max_weight < 0.9
                if is_close != computed_close:
                    close_disagreement.append(
                        f"Galaxy {gal_id}: close={is_close} but max_weight={max_weight:.4f} "
                        f"(computed_close={computed_close})"
                    )

    # Collect invalid counts, galaxies-per-model, and prior mass. Both of the first two may be
    # reported at the top level, under a top-level "sets_summary", or on the first galaxy's own
    # per-set dict -- read them in that order of preference.
    invalid_counts = {}
    galaxies_per_model = {}
    prior_mass = {}
    sets_present = []

    first_gal = next(iter(summary["galaxies"].values()))
    if "sets" in first_gal:
        sets_present = list(first_gal["sets"].keys())

    top_level_invalid = summary.get("invalid_counts", {})
    top_level_galaxies_per_model = summary.get("galaxies_per_model", {})
    sets_summary = summary.get("sets_summary", {})

    for set_name in sets_present:
        w_set = first_gal["sets"].get(set_name, {})

        if set_name in top_level_invalid:
            invalid_counts[set_name] = top_level_invalid[set_name]
        elif "invalid_counts" in sets_summary.get(set_name, {}):
            invalid_counts[set_name] = sets_summary[set_name]["invalid_counts"]
        elif "invalid_counts" in w_set:
            invalid_counts[set_name] = w_set["invalid_counts"]

        if set_name in top_level_galaxies_per_model:
            galaxies_per_model[set_name] = top_level_galaxies_per_model[set_name]
        elif "galaxies_per_model" in sets_summary.get(set_name, {}):
            galaxies_per_model[set_name] = sets_summary[set_name]["galaxies_per_model"]
        elif "galaxies_per_model" in w_set:
            galaxies_per_model[set_name] = w_set["galaxies_per_model"]

    if "model_prior" in summary:
        prior_mass = summary["model_prior"]

    draw_summary = DrawSummary(
        galaxies_drawn=galaxies,
        models_drawn=models,
        sets_present=sets_present,
        close_galaxies=close_gal,
        excluded_models=excluded_by_set,
        invalid_counts=invalid_counts,
        galaxies_per_model=galaxies_per_model,
        prior_mass=prior_mass,
        close_disagreement=close_disagreement,
        factorial_marginals=factorial_marginals,
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
