#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Figure 11 - Bayesian model averaging over Laplace evidences.

This figure demonstrates BMA with a flat prior over valid models within each weight set.
Panel (a): thin strip showing the evidence -> weights -> posterior pathway.
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
Panel (c1)/(c2), side by side: BMA-averaged stellar mass (survived) and SFR (100 Myr)
    against each single-configuration posterior for the selected named set; the legend
    below names the configuration colors.

The axes, their value order, the derived axes, the design priors and the valid model ids
all come from ``_bma_keys`` (the module the evidence runner and the combiner share), and
every axis value and named id is labeled through ``_bma_keys.display_label``. Nothing in
this file restates them, so the figure cannot drift from what the combiner writes.

Each factorial marginal's prior is read from the combiner's output: the per-value
``"prior"`` when a marginal entry carries one, else the summary's top-level
``prior_mass["factorial"]``, else the design prior from ``_bma_keys``. When a galaxy's
factorial set is full (every expected model valid) its prior must equal the design prior;
a disagreement is written to the sidecar under ``prior_warnings`` and to stderr.

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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_style import CONFIG_COLORS
from paper1 import _bma_keys as bk

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_WIDTH = 7.1
FIGURE_HEIGHT = 8.6

#: Gray per X-like id, with a marker each so the five stay distinguishable in grayscale.
XLIKE_COLORS = {
    "cigale_like": "#999999",
    "prospector_like": "#666666",
    "bagpipes_like": "#444444",
    "beagle_like": "#AAAAAA",
    "dense_basis_like": "#777777",
}
XLIKE_MARKERS = dict(zip(bk.XLIKE_IDS, ("s", "^", "v", "P", "X"), strict=True))


def _factorial_marginal_axes() -> tuple[str, ...]:
    """Axes panel (b2) draws: every axis with more than one value, derived axes after ssp."""
    base = [axis for axis in bk.AXES if len(bk.axis_values(axis)) > 1]
    cut = base.index("ssp") + 1
    return (*base[:cut], *bk.DERIVED_AXES, *base[cut:])


#: Order panel (b2) draws the factorial marginal heatmaps in.
FACTORIAL_MARGINAL_AXES: tuple[str, ...] = _factorial_marginal_axes()

#: Panel titles of the marginal rows (axis names are not value labels).
AXIS_TITLES: dict[str, str] = {
    "sfh": "SFH",
    "ssp": "SSP grid",
    "isochrone": "Isochrones",
    "spectral_library": "Spectral library",
    "attenuation": "Dust attenuation",
}

#: log2(weight / prior) beyond +/-3 (an 8x over- or under-weighting) is clipped for color
#: only; the sidecar always records the unclipped weight and prior.
LOG2_RATIO_CLIP = 3.0

#: Agreement required between a reported prior and the design prior [dimensionless].
PRIOR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DrawSummary:
    """Summary of what was drawn in the figure."""

    galaxies_drawn: list[int]
    models_drawn: list[str]
    model_labels: list[str]
    sets_present: list[str]
    close_galaxies: list[int]
    excluded_models: dict[str, list[tuple[str, str]]]  # set_name -> [(model_key, reason)]
    hatched_weight_cells: list[tuple[str, int]]  # (model_key, galaxy) hatched in panel (b1)
    invalid_counts: dict[str, dict[str, int]]  # set_name -> {model_key: count}
    galaxies_per_model: dict[str, dict[str, int]]  # set_name -> {model_key: n galaxies}
    prior_mass: dict[str, Any]  # Prior mass specification
    prior_warnings: list[str]  # A reported factorial prior that disagrees with the design
    close_disagreement: list[str]  # Warnings about close flag disagreement
    marginal_row_labels: dict[str, list[str]]  # axis -> row labels as drawn
    factorial_marginals: dict[str, Any]  # axis -> {value: {"prior", "weight", "log2_ratio"}}


# --- model identity and labels --------------------------------------------


def _named_id(model: dict[str, Any]) -> str | None:
    """Configuration id (``I``.., ``cigale_like``..) of a named model, else ``None``."""
    config = model.get("config")
    if config:
        return config
    key = model.get("model_key", "")
    try:
        return bk.parse_model_key(key).get("config")
    except ValueError:
        return key if key in bk.DISPLAY_LABELS["named"] else None


def _model_label(model: dict[str, Any]) -> str:
    """Reader label of a model row. Raises ``ValueError`` for a model with no label."""
    named = _named_id(model)
    if named is not None:
        return bk.display_label("named", named)
    parsed = bk.parse_model_key(model["model_key"])
    return " / ".join(
        bk.display_label(axis, parsed[axis]) for axis in bk.AXES if len(bk.axis_values(axis)) > 1
    )


def _model_sort_key(model: dict[str, Any], order: dict[str, int]) -> tuple[int, str]:
    return order.get(model["model_key"], len(order)), model["model_key"]


def _row_order() -> dict[str, int]:
    """Position of each model key in the enumeration order of every weight set."""
    keys = list(dict.fromkeys(k for name in bk.WEIGHT_SETS for k in bk.expected_keys(name)))
    return {key: i for i, key in enumerate(keys)}


def _get_model_color(named_id: str) -> str:
    """Return the plot color of a configuration id."""
    if named_id in CONFIG_COLORS:
        return CONFIG_COLORS[named_id]
    return XLIKE_COLORS.get(named_id, "#cccccc")


# --- prior resolution -----------------------------------------------------


def _summary_factorial_prior(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The combiner's reported factorial prior mass (empty when it reports none)."""
    prior_mass = summary.get("prior_mass")
    if not isinstance(prior_mass, dict):
        return {}
    factorial = prior_mass.get("factorial")
    return factorial if isinstance(factorial, dict) else {}


def _resolve_marginal_weight_prior(
    entry: Any,
    axis: str,
    value: str,
    reported: dict[str, dict[str, float]],
    design: dict[str, dict[str, float]],
) -> tuple[float | None, float | None, float | None]:
    """Return (weight, prior, reported_prior) for one factorial axis value.

    ``entry`` is ``{"weight": w, "prior": p}`` or a bare float weight (the combiner's
    schema). The prior is the entry's own, else the summary's ``prior_mass`` for this axis
    value, else the design prior. ``reported_prior`` is the first two (``None`` if neither
    was given), so the caller can compare it with the design prior.
    """
    weight = None
    entry_prior = None
    if isinstance(entry, dict):
        weight = entry.get("weight")
        entry_prior = entry.get("prior")
    elif entry is not None:
        weight = float(entry)

    reported_prior = entry_prior
    if reported_prior is None:
        reported_prior = reported.get(axis, {}).get(value)
    prior = reported_prior if reported_prior is not None else design.get(axis, {}).get(value)
    return weight, prior, reported_prior


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

    if "galaxies" not in summary:
        raise ValueError("Summary missing 'galaxies' key")
    if not summary["galaxies"]:
        raise ValueError("Summary contains no galaxies")

    return _filter_configuration_vi(summary)


# --- panels ---------------------------------------------------------------


def _draw_flow_panel(ax: plt.Axes) -> None:
    """Draw panel (a): a thin strip, evidence -> weights -> posterior."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    labels = (
        "Model set",
        "Laplace MAP fit",
        "Per-model\nevidence",
        "Model weights",
        "BMA posterior",
    )
    width, gap = 0.16, 0.05
    x0 = (1.0 - (len(labels) * width + (len(labels) - 1) * gap)) / 2
    for i, label in enumerate(labels):
        x = x0 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.12),
                width,
                0.76,
                boxstyle="round,pad=0.005,rounding_size=0.06",
                facecolor="#f0f0f0",
                edgecolor="#333333",
                linewidth=0.8,
            )
        )
        ax.text(x + width / 2, 0.5, label, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.004, 0.5),
                    (x + width + gap - 0.004, 0.5),
                    arrowstyle="-|>",
                    lw=0.8,
                    color="#333333",
                    mutation_scale=7,
                )
            )


def _collect_weight_set(summary: dict[str, Any], selected_set: str, galaxies: list[int]):
    """Per-galaxy weights, validity and NUTS flags of the selected set, plus its models."""
    models_by_key: dict[str, dict[str, Any]] = {}
    weights_by_gal: dict[int, dict[str, float]] = {}
    validity_by_gal: dict[int, dict[str, bool]] = {}
    nuts_fail_by_gal: dict[int, dict[str, bool]] = {}
    close_galaxies: list[int] = []
    excluded: list[tuple[str, str]] = []

    for gal_id in galaxies:
        w_set = summary["galaxies"][str(gal_id)].get("sets", {}).get(selected_set)
        if w_set is None:
            continue
        weights: dict[str, float] = {}
        validity: dict[str, bool] = {}
        nuts_fail: dict[str, bool] = {}
        for model in w_set.get("models", []):
            model_key = model.get("model_key")
            if not model_key:
                continue
            models_by_key.setdefault(model_key, model)
            valid = model.get("valid", True)
            validity[model_key] = valid
            nuts_fail[model_key] = valid and model.get("nuts_adoption_pass") is False
            if valid and model.get("weight") is not None:
                weights[model_key] = model["weight"]
            elif not valid:
                excluded.append((model_key, model.get("excluded_reason", "unknown")))
        _validate_weights_sum(weights, gal_id, selected_set)
        weights_by_gal[gal_id] = weights
        validity_by_gal[gal_id] = validity
        nuts_fail_by_gal[gal_id] = nuts_fail
        if w_set.get("close", False):
            close_galaxies.append(gal_id)
    return (
        models_by_key,
        weights_by_gal,
        validity_by_gal,
        nuts_fail_by_gal,
        close_galaxies,
        excluded,
    )


def _draw_weights_panel(
    ax: plt.Axes,
    cax: plt.Axes,
    summary: dict[str, Any],
    selected_set: str,
) -> tuple[
    list[int], list[str], list[str], list[int], list[tuple[str, str]], list[tuple[str, int]]
]:
    """Draw panel (b1): per-galaxy BMA weights heatmap and excluded/invalid cells.

    Excluded/invalid cells are hatched. A valid cell whose named configuration's NUTS chain
    failed adoption -- which does not gate BMA, since a named configuration's evidence is its
    own Laplace run -- is marked with a small "x" so a reader can tell the two failure modes
    apart at a glance.

    Returns (galaxies, model_keys, row_labels, close_galaxies, excluded, hatched_cells).
    """
    galaxies = sorted(int(gal_id) for gal_id in summary["galaxies"])
    cax.axis("off")

    if not any(selected_set in g.get("sets", {}) for g in summary["galaxies"].values()):
        ax.text(0.5, 0.5, f"No data for set: {selected_set}", ha="center", va="center")
        return galaxies, [], [], [], [], []

    models_by_key, weights_by_gal, validity_by_gal, nuts_by_gal, close, excluded = (
        _collect_weight_set(summary, selected_set, galaxies)
    )
    if not models_by_key:
        ax.text(0.5, 0.5, "No valid model weights found", ha="center", va="center")
        return galaxies, [], [], close, excluded, []

    order = _row_order()
    models = sorted(models_by_key.values(), key=lambda m: _model_sort_key(m, order))
    keys = [m["model_key"] for m in models]
    labels = [_model_label(m) for m in models]

    n_gal, n_mod = len(galaxies), len(keys)
    heatmap = np.full((n_mod, n_gal), np.nan)
    hatched: list[tuple[str, int]] = []
    nuts_marks: list[tuple[int, int]] = []
    for gal_idx, gal_id in enumerate(galaxies):
        validity = validity_by_gal.get(gal_id, {})
        for mod_idx, key in enumerate(keys):
            if key not in validity:
                continue
            if not validity[key]:
                heatmap[mod_idx, gal_idx] = 0.0
                hatched.append((key, gal_id))
            else:
                heatmap[mod_idx, gal_idx] = weights_by_gal[gal_id].get(key, 0.0)
                if nuts_by_gal[gal_id].get(key, False):
                    nuts_marks.append((mod_idx, gal_idx))

    cax.axis("on")
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for key, gal_id in hatched:
        _hatch_cell(ax, galaxies.index(gal_id), keys.index(key), linewidth=0.5)
    for mod_idx, gal_idx in nuts_marks:
        ax.text(gal_idx, mod_idx, "x", ha="center", va="center", fontsize=6, fontweight="bold")

    ax.set_xticks(range(n_gal))
    ax.set_yticks(range(n_mod))
    ax.set_xticklabels([str(g) for g in galaxies], fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Galaxy ID", fontsize=7, labelpad=2)
    ax.tick_params(length=2, pad=1.5)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Weight", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)

    ax.text(
        0.0,
        1.03,
        "hatched = excluded/invalid   x = NUTS adoption failed (Laplace evidence still used)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        style="italic",
    )
    return galaxies, keys, labels, close, excluded, hatched


def _hatch_cell(ax: plt.Axes, col: int, row: int, *, linewidth: float) -> None:
    """Hatch one heatmap cell."""
    ax.add_patch(
        plt.Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            hatch="///",
            edgecolor="gray",
            linewidth=linewidth,
        )
    )


def _ordered_axis_values(axis: str, seen: set[str]) -> list[str]:
    """Rows of one marginal heatmap: the whole axis in key-module order, then strays."""
    return [*bk.axis_values(axis), *sorted(seen - set(bk.axis_values(axis)))]


def _prior_disagreements(
    axis: str,
    value: str,
    gid: int,
    reported: float | None,
    design: dict[str, dict[str, float]],
    full_set: bool,
) -> str | None:
    """Warning text when a full set reports a prior that is not the design prior."""
    expected = design.get(axis, {}).get(value)
    if not full_set or reported is None or expected is None:
        return None
    if abs(reported - expected) <= PRIOR_TOLERANCE:
        return None
    return (
        f"galaxy {gid} {axis}/{value}: summary prior {reported:.6g} differs from "
        f"design prior {expected:.6g} on a full factorial set"
    )


def _draw_factorial_marginal_panel(
    fig: plt.Figure,
    gs_cell: Any,
    cax: plt.Axes,
    summary: dict[str, Any],
    galaxies: list[int],
) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    """Draw panel (b2): factorial marginal weight vs. prior, one heatmap row per axis.

    The color encodes log2(weight / prior) on a diverging colormap centered at zero. A
    galaxy with no "factorial" weight set is drawn as a hatched column in every row rather
    than silently dropped.

    Returns (sidecar, prior_warnings, row_labels). The sidecar maps
    ``axis -> {value: {"prior", "weight": {gal: w}, "log2_ratio": {gal: r}}}`` and carries
    ``galaxies_missing_factorial`` and ``hatched_cells`` (``axis -> {value: [gal, ...]}``).
    """
    sidecar: dict[str, Any] = {axis: {} for axis in FACTORIAL_MARGINAL_AXES}
    hatched_cells: dict[str, dict[str, list[int]]] = {a: {} for a in FACTORIAL_MARGINAL_AXES}
    sidecar["hatched_cells"] = hatched_cells
    warnings: list[str] = []
    row_labels: dict[str, list[str]] = {}

    def factorial_set(gid: int) -> dict[str, Any] | None:
        # The combiner writes an empty placeholder set for a galaxy with no factorial cells,
        # so a set without marginals counts as missing.
        w_set = summary["galaxies"].get(str(gid), {}).get("sets", {}).get("factorial")
        return w_set if w_set and w_set.get("marginal") else None

    missing = [gid for gid in galaxies if factorial_set(gid) is None]
    sidecar["galaxies_missing_factorial"] = missing

    if len(missing) == len(galaxies):
        ax = fig.add_subplot(gs_cell)
        ax.text(0.5, 0.5, "No factorial set in summary", ha="center", va="center", fontsize=9)
        ax.axis("off")
        cax.axis("off")
        return sidecar, warnings, row_labels

    reported = _summary_factorial_prior(summary)
    design = bk.factorial_prior_mass()
    n_gal = len(galaxies)

    def seen_values(axis: str) -> set[str]:
        seen: set[str] = set()
        for gid in galaxies:
            w_set = factorial_set(gid)
            if w_set is not None:
                seen.update(w_set.get("marginal", {}).get(axis, {}))
        return seen

    values_by_axis = {a: _ordered_axis_values(a, seen_values(a)) for a in FACTORIAL_MARGINAL_AXES}
    gs_axes = gs_cell.subgridspec(
        len(FACTORIAL_MARGINAL_AXES),
        1,
        height_ratios=[len(values_by_axis[a]) for a in FACTORIAL_MARGINAL_AXES],
        hspace=0.95,
    )
    axes_drawn: list[plt.Axes] = []
    shared_ax: plt.Axes | None = None
    last_im = None

    for row, axis in enumerate(FACTORIAL_MARGINAL_AXES):
        ax = fig.add_subplot(gs_axes[row], sharex=shared_ax)
        shared_ax = shared_ax or ax
        axes_drawn.append(ax)
        values = values_by_axis[axis]
        ratio = np.full((len(values), n_gal), np.nan)
        hatch: list[tuple[int, int]] = []

        for val_idx, value in enumerate(values):
            cell = sidecar[axis].setdefault(value, {"prior": None, "weight": {}, "log2_ratio": {}})
            for gal_idx, gid in enumerate(galaxies):
                w_set = factorial_set(gid)
                entry = (
                    None if w_set is None else w_set.get("marginal", {}).get(axis, {}).get(value)
                )
                weight, prior, rep = _resolve_marginal_weight_prior(
                    entry, axis, value, reported, design
                )
                if w_set is not None:
                    n_expected = w_set.get("n_expected")
                    full = n_expected is not None and w_set.get("n_valid") == n_expected
                    msg = _prior_disagreements(axis, value, gid, rep, design, full)
                    if msg:
                        warnings.append(msg)
                if cell["prior"] is None and prior is not None:
                    cell["prior"] = prior
                if weight is not None:
                    cell["weight"][str(gid)] = weight
                if w_set is None or weight is None or not prior:
                    hatch.append((val_idx, gal_idx))
                    hatched_cells[axis].setdefault(value, []).append(gid)
                    continue
                log2_ratio = -LOG2_RATIO_CLIP if weight <= 0 else float(np.log2(weight / prior))
                cell["log2_ratio"][str(gid)] = log2_ratio
                ratio[val_idx, gal_idx] = np.clip(log2_ratio, -LOG2_RATIO_CLIP, LOG2_RATIO_CLIP)

        last_im = ax.imshow(
            ratio, cmap="RdBu_r", aspect="auto", vmin=-LOG2_RATIO_CLIP, vmax=LOG2_RATIO_CLIP
        )
        for val_idx, gal_idx in hatch:
            _hatch_cell(ax, gal_idx, val_idx, linewidth=0.4)

        labels = [bk.display_label(axis, v) for v in values]
        row_labels[axis] = labels
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(AXIS_TITLES.get(axis, axis), fontsize=6.5, loc="left", pad=1.5)
        ax.tick_params(axis="y", length=2, pad=1.5)
        if row < len(FACTORIAL_MARGINAL_AXES) - 1:
            ax.tick_params(axis="x", labelbottom=False, length=0)
        else:
            ax.set_xticks(range(n_gal))
            ax.set_xticklabels([str(g) for g in galaxies], fontsize=6, rotation=90)
            ax.set_xlabel("Galaxy ID", fontsize=7, labelpad=2)
            ax.tick_params(axis="x", length=2, pad=1.5)

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"$\log_2$(weight / prior)", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    return sidecar, list(dict.fromkeys(warnings)), row_labels


def _draw_posterior_panel(
    ax: plt.Axes,
    summary: dict[str, Any],
    selected_set: str,
    quantity: str,  # "log_stellar_mass_survived" or "log_sfr_100myr"
) -> list[str]:
    """Draw panel (c): BMA vs single-config posteriors for a quantity.

    Returns the configuration ids that were plotted, in first-drawn order.
    """
    galaxies = sorted(int(gal_id) for gal_id in summary["galaxies"])
    drawn: list[str] = []
    order = _row_order()

    for gal_idx, gal_id in enumerate(galaxies):
        w_set = summary["galaxies"][str(gal_id)].get("sets", {}).get(selected_set)
        if w_set is None:
            continue
        models = sorted(w_set.get("models", []), key=lambda m: _model_sort_key(m, order))
        n_models = max(len(models), 1)
        for m_idx, model in enumerate(models):
            named = _named_id(model)
            percentiles = model.get("percentiles", {})
            if not model.get("valid", True) or named is None or quantity not in percentiles:
                continue
            p16, p50, p84 = percentiles[quantity]
            dodge = -0.36 + 0.6 * m_idx / max(n_models - 1, 1)
            ax.errorbar(
                gal_idx + dodge,
                p50,
                yerr=[[p50 - p16], [p84 - p50]],
                fmt=XLIKE_MARKERS.get(named, "o"),
                markersize=3.5,
                color=_get_model_color(named),
                alpha=0.85,
                linewidth=0.7,
                capsize=1.5,
            )
            if named not in drawn:
                drawn.append(named)

        bma = w_set.get("bma_percentiles", {})
        if quantity in bma:
            p16, p50, p84 = bma[quantity]
            ax.errorbar(
                gal_idx + 0.42,
                p50,
                yerr=[[p50 - p16], [p84 - p50]],
                fmt="D",
                markersize=4,
                color="black",
                linewidth=1,
                capsize=1.5,
                zorder=10,
            )

    ax.set_xticks(np.arange(len(galaxies)))
    ax.set_xticklabels([str(g) for g in galaxies], fontsize=6, rotation=90)
    ax.set_xlabel("Galaxy ID", fontsize=7, labelpad=2)
    if quantity == "log_stellar_mass_survived":
        ax.set_ylabel(r"$\log M_{\star,\,\mathrm{survived}}$ [M$_{\odot}$]", fontsize=7)
    elif quantity == "log_sfr_100myr":
        ax.set_ylabel(
            r"$\log\,\mathrm{SFR}_{100\,\mathrm{Myr}}$ [M$_{\odot}$ yr$^{-1}$]", fontsize=7
        )
    else:
        ax.set_ylabel(quantity, fontsize=7)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    return drawn


def _config_legend_handles(named_ids: list[str]) -> list[Line2D]:
    """Legend handles for the configuration colors in panel (c), plus the BMA marker."""
    order = {name: i for i, name in enumerate((*bk.GRID_IDS, *bk.XLIKE_IDS))}
    handles = [
        Line2D(
            [],
            [],
            marker=XLIKE_MARKERS.get(named, "o"),
            linestyle="",
            markersize=4,
            color=_get_model_color(named),
            label=bk.display_label("named", named),
        )
        for named in sorted(named_ids, key=lambda n: order.get(n, len(order)))
    ]
    handles.append(
        Line2D([], [], marker="D", linestyle="", markersize=4, color="black", label="BMA")
    )
    return handles


# --- assembly -------------------------------------------------------------


def _close_disagreements(summary: dict[str, Any], galaxies: list[int], selected_set: str):
    """Warnings for galaxies whose ``close`` flag disagrees with their max weight."""
    out = []
    for gal_id in galaxies:
        w_set = summary["galaxies"][str(gal_id)].get("sets", {}).get(selected_set)
        if w_set is None:
            continue
        weights = [
            m["weight"]
            for m in w_set.get("models", [])
            if m.get("valid", True) and m.get("weight") is not None
        ]
        max_weight = max(weights) if weights else 0.0
        computed_close = max_weight < 0.9
        if w_set.get("close", False) != computed_close:
            out.append(
                f"Galaxy {gal_id}: close={w_set.get('close', False)} but "
                f"max_weight={max_weight:.4f} (computed_close={computed_close})"
            )
    return out


def _per_set_report(summary: dict[str, Any], sets_present: list[str], first_gal: dict[str, Any]):
    """Invalid counts and galaxies-per-model, from the top level, sets_summary or the galaxy."""
    invalid_counts: dict[str, Any] = {}
    galaxies_per_model: dict[str, Any] = {}
    top = {
        "invalid_counts": summary.get("invalid_counts", {}),
        "galaxies_per_model": summary.get("galaxies_per_model", {}),
    }
    sets_summary = summary.get("sets_summary", {})
    for set_name in sets_present:
        w_set = first_gal["sets"].get(set_name, {})
        for field, target in (
            ("invalid_counts", invalid_counts),
            ("galaxies_per_model", galaxies_per_model),
        ):
            if set_name in top[field]:
                target[set_name] = top[field][set_name]
            elif field in sets_summary.get(set_name, {}):
                target[set_name] = sets_summary[set_name][field]
            elif field in w_set:
                target[set_name] = w_set[field]
    return invalid_counts, galaxies_per_model


def build_figure(
    summary: dict[str, Any],
    selected_set: str,
) -> tuple[plt.Figure, DrawSummary]:
    """Build the BMA figure: (a) flow strip, (b1) named weights, (b2) factorial marginals,
    (c1)/(c2) BMA vs. single-config mass and SFR posteriors, side by side."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[1, 0.025],
        height_ratios=[0.42, 1.75, 3.3, 1.75],
        hspace=0.42,
        wspace=0.03,
        left=0.25,
        right=0.93,
        top=0.985,
        bottom=0.11,
    )

    ax_flow = fig.add_subplot(gs[0, :])
    _draw_flow_panel(ax_flow)

    ax_weights = fig.add_subplot(gs[1, 0])
    cax_weights = fig.add_subplot(gs[1, 1])
    galaxies, models, model_labels, close_gal, excluded, hatched = _draw_weights_panel(
        ax_weights, cax_weights, summary, selected_set
    )

    cax_marginal = fig.add_subplot(gs[2, 1])
    factorial_marginals, prior_warnings, row_labels = _draw_factorial_marginal_panel(
        fig, gs[2, 0], cax_marginal, summary, galaxies
    )

    gs_c = gs[3, 0].subgridspec(1, 2, wspace=0.42)
    ax_mass = fig.add_subplot(gs_c[0])
    ax_sfr = fig.add_subplot(gs_c[1])
    fig.add_subplot(gs[3, 1]).axis("off")
    drawn = _draw_posterior_panel(ax_mass, summary, selected_set, "log_stellar_mass_survived")
    _draw_posterior_panel(ax_sfr, summary, selected_set, "log_sfr_100myr")
    fig.legend(
        handles=_config_legend_handles(drawn),
        loc="lower center",
        bbox_to_anchor=(0.59, 0.0),
        ncol=6,
        fontsize=6.5,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
        title="Configuration (c1, c2)",
        title_fontsize=6.5,
    )

    for panel, ax in (("(a)", ax_flow), ("(b1)", ax_weights)):
        fig.text(0.012, ax.get_position().y1 + 0.004, panel, weight="bold")
    for panel, ax in (("(c1)", ax_mass), ("(c2)", ax_sfr)):
        ax.text(0.0, 1.03, panel, transform=ax.transAxes, weight="bold", va="bottom")
    fig.text(0.012, gs[2, 0].get_position(fig).y1 + 0.004, "(b2)", weight="bold")

    excluded_by_set: dict[str, list[tuple[str, str]]] = {}
    for model_key, reason in excluded:
        excluded_by_set.setdefault(selected_set, []).append((model_key, reason))

    first_gal = next(iter(summary["galaxies"].values()))
    sets_present = list(first_gal.get("sets", {}))
    invalid_counts, galaxies_per_model = _per_set_report(summary, sets_present, first_gal)

    draw_summary = DrawSummary(
        galaxies_drawn=galaxies,
        models_drawn=models,
        model_labels=model_labels,
        sets_present=sets_present,
        close_galaxies=close_gal,
        excluded_models=excluded_by_set,
        hatched_weight_cells=hatched,
        invalid_counts=invalid_counts,
        galaxies_per_model=galaxies_per_model,
        prior_mass=summary.get("model_prior", {}),
        prior_warnings=prior_warnings,
        close_disagreement=_close_disagreements(summary, galaxies, selected_set),
        marginal_row_labels=row_labels,
        factorial_marginals=factorial_marginals,
    )
    return fig, draw_summary


def _git_describe() -> str:
    """Get short git SHA."""
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
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

    selected_set = args.set
    first_gal = next(iter(summary["galaxies"].values()))
    available_sets = list(first_gal.get("sets", {}))

    if selected_set not in available_sets:
        fallback = next((s for s in ("named_all", "named_grid") if s in available_sets), None)
        if fallback is None:
            print("ERROR: No weight sets found in summary", file=sys.stderr)
            return 1
        print(f"WARNING: Set '{args.set}' not found; using '{fallback}'", file=sys.stderr)
        selected_set = fallback

    try:
        fig, draw_summary = build_figure(summary, selected_set)
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Failed to build figure: {exc}", file=sys.stderr)
        return 1

    for warning in draw_summary.close_disagreement:
        print(f"WARNING: {warning}", file=sys.stderr)
    for warning in draw_summary.prior_warnings:
        print(f"WARNING: prior mismatch: {warning}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

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
