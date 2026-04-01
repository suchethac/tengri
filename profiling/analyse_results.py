"""Analyse profiling results and generate publication-quality plots.

Reads CSV outputs from the pipeline profiler, scaling profiler, and
memory profiler, then generates summary figures and a text report.

Usage::

    python profiling/analyse_results.py --input-dir profiling/outputs
    python profiling/analyse_results.py --pipeline-csv profiling/outputs/pipeline.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Pipeline breakdown plot
# ---------------------------------------------------------------------------


def plot_pipeline_breakdown(csv_path: str, output_dir: str) -> None:
    """Generate horizontal bar chart of pipeline step timings.

    Parameters
    ----------
    csv_path : str
        Path to pipeline_timing.csv from ``profile_pipeline().to_csv()``.
    output_dir : str
        Directory for output figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append({
                "name": row["step"],
                "mean_us": float(row["mean_us"]),
                "pct": float(row["pct"]),
            })

    if not steps:
        print(f"No data in {csv_path}")
        return

    names = [s["name"] for s in steps]
    times = [s["mean_us"] for s in steps]
    pcts = [s["pct"] for s in steps]

    fig, ax = plt.subplots(figsize=(10, max(4, len(steps) * 0.5)))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, times, align="center", color="#4C72B0", edgecolor="white")

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax.text(
            bar.get_width() + max(times) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Time (μs)")
    ax.set_title("Pipeline Step Breakdown")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pipeline_breakdown.pdf")
    fig.savefig(output_dir / "pipeline_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"  Pipeline breakdown → {output_dir}/pipeline_breakdown.pdf")


# ---------------------------------------------------------------------------
# Scaling plots from JSON
# ---------------------------------------------------------------------------


def plot_scaling_from_json(json_path: str, output_dir: str) -> None:
    """Re-generate scaling plots from saved JSON results.

    Parameters
    ----------
    json_path : str
        Path to scaling_results.json.
    output_dir : str
        Directory for output figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        results = json.load(f)

    # Import the plot function from profile_scaling
    from profiling.profile_scaling import plot_scaling

    plot_scaling(results, output_dir)


# ---------------------------------------------------------------------------
# Configuration comparison plot
# ---------------------------------------------------------------------------


def plot_config_comparison(csv_paths: list[str], output_dir: str) -> None:
    """Compare timing across multiple configurations.

    Parameters
    ----------
    csv_paths : list of str
        Paths to pipeline CSV files from different configs.
    output_dir : str
        Directory for output figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for path in csv_paths:
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                total = sum(float(r["mean_us"]) for r in rows)
                config_name = rows[0].get("config", Path(path).stem)
                n_free = rows[0].get("n_free", "?")
                path_type = rows[0].get("path", "?")
                configs.append({
                    "name": config_name,
                    "total_us": total,
                    "n_free": n_free,
                    "path": path_type,
                })

    if not configs:
        print("No configuration data found")
        return

    names = [c["name"] for c in configs]
    totals = [c["total_us"] for c in configs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(configs) * 0.6)))
    y_pos = np.arange(len(names))
    colors = ["#4C72B0" if c["path"] == "FUSED" else "#DD8452" for c in configs]
    bars = ax.barh(y_pos, totals, color=colors, edgecolor="white")

    for bar, c in zip(bars, configs):
        ax.text(
            bar.get_width() + max(totals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"D={c['n_free']} {c['path']}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Total time (μs)")
    ax.set_title("Configuration Comparison")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "config_comparison.pdf")
    fig.savefig(output_dir / "config_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Config comparison → {output_dir}/config_comparison.pdf")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def generate_text_report(output_dir: str) -> str:
    """Generate a summary text report from all profiling outputs."""
    output_dir = Path(output_dir)
    lines = []
    lines.append("=" * 70)
    lines.append("TENGRI PROFILING REPORT")
    lines.append("=" * 70)

    # Scaling results
    json_path = output_dir / "scaling_results.json"
    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)

        if "dimension" in results:
            lines.append("\nDIMENSION SCALING")
            lines.append("-" * 50)
            for d in results["dimension"]:
                lines.append(
                    f"  D={d['dimension']:>4d}:  forward={d['forward_us']:>8.1f} μs  "
                    f"gradient={d['gradient_us']:>8.1f} μs"
                )

        if "bands" in results:
            lines.append("\nBAND SCALING")
            lines.append("-" * 50)
            for d in results["bands"]:
                lines.append(
                    f"  {d['n_bands']:>2d} bands: forward={d['forward_us']:>8.1f} μs  "
                    f"gradient={d['gradient_us']:>8.1f} μs"
                )

    # Pipeline CSV
    for csv_file in sorted(output_dir.glob("pipeline*.csv")):
        lines.append(f"\nPIPELINE: {csv_file.name}")
        lines.append("-" * 50)
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lines.append(
                    f"  {row['step']:<40s} {float(row['mean_us']):>8.1f} μs  "
                    f"{row['pct']:>5s}%"
                )

    report = "\n".join(lines)

    # Write to file
    report_path = output_dir / "profiling_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport → {report_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyse tengri profiling results")
    parser.add_argument("--input-dir", default="profiling/outputs",
                        help="Directory with profiling outputs")
    parser.add_argument("--pipeline-csv", help="Single pipeline CSV to plot")
    parser.add_argument("--scaling-json", help="Scaling results JSON to plot")
    args = parser.parse_args()

    output_dir = args.input_dir

    if args.pipeline_csv:
        plot_pipeline_breakdown(args.pipeline_csv, output_dir)

    if args.scaling_json:
        plot_scaling_from_json(args.scaling_json, output_dir)

    # Auto-discover and process all available outputs
    input_dir = Path(output_dir)
    if input_dir.exists():
        for csv_file in input_dir.glob("pipeline*.csv"):
            plot_pipeline_breakdown(str(csv_file), output_dir)

        json_file = input_dir / "scaling_results.json"
        if json_file.exists():
            plot_scaling_from_json(str(json_file), output_dir)

        generate_text_report(output_dir)
    else:
        print(f"No outputs found in {output_dir}. Run profiling first.")


if __name__ == "__main__":
    main()
