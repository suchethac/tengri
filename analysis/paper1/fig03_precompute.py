#!/usr/bin/env python
"""
Figure 3 (speed, main text) and Figure B1 (accuracy, appendix).

fig03_precompute.pdf: forward-model cost, exact vs WavePrecomp (provisional from May 2026 benchmark until re-run).
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
    default=None,
    help="Path to benchmark results JSON (for fresh run, otherwise uses May 2026 data)",
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
    ax.legend(fontsize=9, loc="lower right")
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
                f"{speedup:.0f}×",
                va="center",
                fontsize=8,
                color="#F18F01",
            )

    # Add provisional stamp if using default data
    if bench_data is None:
        ax.text(
            0.98,
            0.02,
            "timings: May 2026 run; to be re-measured",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="gray",
            style="italic",
        )


def plot_panel_b(ax, accuracy_data):
    """Plot LUT accuracy vs redshift (panel b)."""
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
    band_errors_32 = {}

    for z_str in measurements:
        z_meas = measurements[z_str]
        for band in filters_list:
            if band in z_meas:
                if band not in band_errors:
                    band_errors[band] = []
                    band_errors_32[band] = []
                band_errors[band].append(z_meas[band].get("err_default_pct", np.nan))
                band_errors_32[band].append(z_meas[band].get("err_n32_pct", np.nan))

    # Find worst flux-carrying band (galex_fuv has largest errors at high z)
    worst_band = "galex_fuv"

    # Plot all bands as thin lines
    for band in filters_list:
        if band in band_errors:
            errors = np.array(band_errors[band]) / 100.0  # Convert percent to fraction
            if band == worst_band:
                ax.plot(
                    z_array,
                    errors,
                    "o-",
                    linewidth=2,
                    markersize=5,
                    alpha=0.9,
                    label=worst_band,
                    color="#2E86AB",
                )
            else:
                ax.plot(z_array, errors, "-", linewidth=0.5, alpha=0.15, color="gray")

    # n_subbands=32 for worst band
    if worst_band in band_errors_32:
        errors_32 = np.array(band_errors_32[worst_band]) / 100.0  # Convert percent to fraction
        ax.plot(
            z_array,
            errors_32,
            "s--",
            linewidth=1.5,
            markersize=4,
            alpha=0.9,
            label=f"{worst_band} ($n_{{sub}}=32$)",
            color="#A23B72",
        )

    # 1% reference line
    ax.axhline(y=0.01, color="red", linestyle="--", linewidth=1, alpha=0.6, label="1% level")

    # Styling
    ax.set_yscale("log")
    ax.set_xlabel("Redshift", fontsize=10)
    ax.set_ylabel(r"Relative error $|F_{\text{LUT}}/F_{\text{exact}} - 1|$", fontsize=10)
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")

    # Annotation for dark bands
    ax.text(
        0.98,
        0.05,
        "band dark beyond z~2 (FUV flux < 1e-3 of optical)",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="right",
        style="italic",
        color="gray",
    )


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

fig_b, ax_b = plt.subplots(figsize=(3.5, 3.0), dpi=150)
plot_panel_b(ax_b, accuracy_data)
for ext in ("pdf", "png"):
    path = os.path.join(figures_dir, f"figB1_lut_accuracy.{ext}")
    fig_b.savefig(path, format=ext, bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")

# Combined two-panel version kept for reference.
fig = create_figure(bench_data, accuracy_data)
fig.savefig(os.path.join(figures_dir, "fig03_precompute_combined.pdf"), format="pdf", bbox_inches="tight", dpi=300)
print("Saved: combined")
