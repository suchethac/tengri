#!/usr/bin/env python
"""
Figure 3 (speed, main text) and Figure B1 (accuracy, appendix).

fig03_precompute.pdf: forward-model cost, exact vs WavePrecomp, from results/fig03_bench_forward_2026-08-30.json
  (bench/scripts/benchmark_forward_model.py parsed by parse_forward_benchmark.py; see REPRODUCTION_COMMANDS.md).
figB1_lut_accuracy.pdf: LUT accuracy vs redshift (from results/fig03_precompute_data.json).

Usage:
  python fig03_precompute.py
  python fig03_precompute.py --bench-json <path> --accuracy-json <path>
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Generate Figure 3: Precomputation speed and accuracy"
)
parser.add_argument(
    "--bench-json",
    type=str,
    default="analysis/paper1/results/fig03_bench_forward_2026-08-30.json",
    help="Path to the parsed benchmark JSON (parse_forward_benchmark.py output); the May 2026 constants are only a fallback",
)
parser.add_argument(
    "--accuracy-json",
    type=str,
    default="analysis/paper1/results/fig03_precompute_data.json",
    help="Path to accuracy measurement JSON",
)
args = parser.parse_args()

# ============================================================================
# Panel (a): Forward model cost — May 2026 data (provisional)
# ============================================================================

# May 2026 data: (config_label, exact_us, hybrid_us)
MAY_2026_DATA = [
    ("Stellar", 23900, 59),
    ("+Neb", 24700, 58),
    ("+Dust IR", 23300, 153),
    ("+AGN", 24200, 148),
    ("+Radio", 22100, 222),
    ("+X-ray", 26500, 233),
    ("All", 76100, 2450),
]


def plot_panel_a(ax, bench_data=None):
    """Plot forward model cost (panel a) with grouped horizontal bars."""
    # Use provided data or May 2026 default
    if bench_data is None:
        config_labels = [label for label, _, _ in MAY_2026_DATA]
        exact_times = np.array([exact for _, exact, _ in MAY_2026_DATA])
        hybrid_times = np.array([hybrid for _, _, hybrid in MAY_2026_DATA])
    else:
        # Parse from JSON structure if provided
        config_labels = [entry.get("label", f"Config {i}") for i, entry in enumerate(bench_data)]
        exact_times = np.array([entry.get("exact_us", 0) for entry in bench_data])
        hybrid_times = np.array(
            [entry.get("hybrid_us", entry.get("precomp_us", 0)) for entry in bench_data]
        )

    x_pos = np.arange(len(config_labels))
    bar_width = 0.35

    # Plot bars
    ax.barh(
        x_pos - bar_width / 2, exact_times, bar_width, label="Exact", color="#2E86AB", alpha=0.85
    )
    ax.barh(
        x_pos + bar_width / 2,
        hybrid_times,
        bar_width,
        label="WavePrecomp",
        color="#A23B72",
        alpha=0.85,
    )

    # Styling
    ax.set_xscale("log")
    ax.set_xlabel("Time per call (µs)", fontsize=10)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(config_labels, fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, which="both", axis="x")
    ax.set_xlim(10, 1e5)

    # Annotate speedups
    for i, (exact_val, hybrid_val) in enumerate(zip(exact_times, hybrid_times)):
        if exact_val > 0 and hybrid_val > 0:
            speedup = exact_val / hybrid_val
            # Place speedup label at the end of the exact bar
            ax.text(
                exact_val * 1.3,
                i - bar_width / 2,
                (f"{speedup:.1f}×" if speedup < 5 else f"{speedup:.0f}×"),
                va="center",
                fontsize=8,
                color="#F18F01",
            )

    # Add provisional stamp if using default data
    if bench_data is None:
        ax.text(
            0.02,
            0.98,
            "timings: May 2026 run; to be re-measured",
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="top",
            color="gray",
            style="italic",
        )


def plot_panel_b(ax, accuracy_data):
    """Plot LUT accuracy vs redshift (panel b) — envelope of band errors with worst-case band highlighted."""
    measurements = accuracy_data.get("measurements", {})
    filters_list = accuracy_data.get("metadata", {}).get("filters", [])

    if not measurements or not filters_list:
        ax.text(
            0.5,
            0.5,
            "No accuracy data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Extract z values and organize errors by band
    z_values = sorted([float(z) for z in measurements])
    z_array = np.array(z_values)

    band_errors = {}

    for z_str in measurements:
        z_meas = measurements[z_str]
        for band in filters_list:
            if band in z_meas:
                if band not in band_errors:
                    band_errors[band] = []
                band_errors[band].append(z_meas[band].get("err_default_pct", np.nan))

    # Compute envelope (min, median, max) across all bands at each redshift
    all_errors_at_z = []
    for z_idx, z_str in enumerate(sorted([float(z) for z in measurements])):
        z_str_key = str(float(z_str))
        errors_at_z = []
        for band in filters_list:
            if band in band_errors and z_idx < len(band_errors[band]):
                err = band_errors[band][z_idx]
                if not np.isnan(err):
                    errors_at_z.append(err)
        if errors_at_z:
            all_errors_at_z.append(errors_at_z)

    # Compute percentiles at each z
    if all_errors_at_z:
        min_errors = np.array([np.min(e) for e in all_errors_at_z]) / 100.0
        median_errors = np.array([np.median(e) for e in all_errors_at_z]) / 100.0
        max_errors = np.array([np.max(e) for e in all_errors_at_z]) / 100.0

        # Shaded region for envelope (min to max)
        ax.fill_between(z_array, min_errors, max_errors, alpha=0.25, color="#4a90e2", label="Band range")

        # Median line
        (line_median,) = ax.plot(z_array, median_errors, "-", linewidth=2.0, alpha=0.85, color="#2e5c8a", label="Median")

        # Highlight worst-case band with a darker line
        worst_band = "galex_fuv" if "galex_fuv" in band_errors else filters_list[0]
        if worst_band in band_errors:
            worst_errors = np.array(band_errors[worst_band]) / 100.0
            # Format band name for astronomers (e.g. galex_fuv -> GALEX FUV)
            band_fmt = worst_band.replace("galex_fuv", "GALEX FUV").replace("galex_nuv", "GALEX NUV")
            if band_fmt == worst_band:  # No replacement happened
                band_fmt = worst_band.replace("_", " ").upper()
            (line_worst,) = ax.plot(z_array, worst_errors, "--", linewidth=1.5, alpha=0.7, color="#e85d75", label=f"Worst: {band_fmt}")

    # 1% and 0.1% reference lines
    ax.axhline(y=0.01, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="1% threshold")
    ax.axhline(y=0.001, color="orange", linestyle=":", linewidth=0.8, alpha=0.4, label="0.1% threshold")

    # Styling
    ax.set_yscale("log")
    ax.set_xlabel("Redshift", fontsize=10)
    ax.set_ylabel("Relative error", fontsize=10)
    ax.set_xlim(0.5, 3.2)
    ax.set_ylim(1e-5, 2)

    # Legend
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)

    ax.grid(True, alpha=0.3, which="both")


def create_figure(bench_data=None, accuracy_data=None):
    """Create figure 3 with both panels."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.8

    # Two panels side by side at two-column width
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=150)
    fig.subplots_adjust(wspace=0.35)

    plot_panel_a(axes[0], bench_data)
    plot_panel_b(axes[1], accuracy_data)

    # Panel labels
    axes[0].text(-0.13, 1.08, "(a)", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.13, 1.08, "(b)", transform=axes[1].transAxes, fontsize=12, fontweight="bold")

    return fig


# ============================================================================
# Main
# ============================================================================

print("Generating Figure 3...", flush=True)

bench_data = None
accuracy_data = None

if args.bench_json:
    print(f"Loading benchmark data from {args.bench_json}...", flush=True)
    try:
        with open(args.bench_json) as f:
            bench_data = json.load(f)
        if isinstance(bench_data, dict):
            bench_data = bench_data["panel_a"]
    except Exception as e:
        print(f"Warning: Could not load benchmark JSON: {e}", flush=True)

if args.accuracy_json and os.path.exists(args.accuracy_json):
    print(f"Loading accuracy data from {args.accuracy_json}...", flush=True)
    try:
        with open(args.accuracy_json) as f:
            accuracy_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load accuracy JSON: {e}", flush=True)
else:
    print(f"Note: Accuracy data not found at {args.accuracy_json}", flush=True)

figures_dir = "analysis/paper1/figures"
os.makedirs(figures_dir, exist_ok=True)

# The paper uses the two panels as separate single-column figures: the speed
# panel in the main text (fig03) and the accuracy panel in the appendix (figB1).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8

fig_a, ax_a = plt.subplots(figsize=(3.5, 3.0), dpi=150)
plot_panel_a(ax_a, bench_data)
for ext in ("pdf", "png"):
    path = os.path.join(figures_dir, f"fig03_precompute.{ext}")
    fig_a.savefig(path, format=ext, bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")

fig_b, ax_b = plt.subplots(figsize=(4.5, 3.0), dpi=150)
plot_panel_b(ax_b, accuracy_data)
for ext in ("pdf", "png"):
    path = os.path.join(figures_dir, f"figB1_lut_accuracy.{ext}")
    fig_b.savefig(path, format=ext, bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")

# Combined two-panel version kept for reference.
fig = create_figure(bench_data, accuracy_data)
fig.savefig(
    os.path.join(figures_dir, "fig03_precompute_combined.pdf"),
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
print("Saved: combined")
