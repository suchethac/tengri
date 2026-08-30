"""
GPU vs CPU cost per galaxy as a function of batch size.

Data transcribed from bench/reports/2026-08-20_cuda_device_matrix.md,
Findings 1-3 (RTX 3060 vs Ryzen 9 5900X, JAX 0.11, float64 and float32).

Two panels: (a) forward pass, (b) gradient; x = batch size (log),
y = wall time per galaxy in microseconds (log); four series (CPU/GPU x f64/f32).
CPU/GPU crossover region is shaded.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# SOURCE DATA
# ==============================================================================
# Transcribed from bench/reports/2026-08-20_cuda_device_matrix.md
# Source path: bench/reports/2026-08-20_cuda_device_matrix.md
# Report date: 2026-08-20
# Platform: RTX 3060 12 GB vs AMD Ryzen 9 5900X, JAX 0.11.0

# Finding 2: Forward pass, predict_photometry_batch, microseconds per galaxy
FORWARD_DATA = {
    "batch": [1, 8, 32, 128, 512, 2048],
    "cpu_f64": [234.0, 113.5, 55.3, 27.6, 45.7, 48.3],
    "cpu_f32": [142.6, 95.9, 53.1, 26.1, 16.7, 20.0],
    "gpu_f64": [7344.4, 927.3, 238.1, 65.7, 22.0, 11.2],
    "gpu_f32": [7361.7, 899.3, 226.9, 56.8, 15.2, 4.5],
}

# Finding 2: Gradient, vmap(grad(sum(predict_photometry))), microseconds per galaxy
GRADIENT_DATA = {
    "batch": [1, 8, 32, 128, 512, 2048],
    "cpu_f64": [499.4, 299.8, 131.2, 211.8, 206.6, 172.2],
    "cpu_f32": [435.6, 164.0, 93.6, 43.7, 67.4, 80.7],
    "gpu_f64": [7584.7, 977.0, 250.8, 77.9, 31.5, 19.8],
    "gpu_f32": [7473.8, 933.1, 233.2, 58.5, 16.3, 5.5],
}

# Finding 1: Single-galaxy times (batch 1, for reference)
FINDING_1 = {
    "forward": {
        "cpu_f64": 227.0,
        "cpu_f32": 162.3,
        "gpu_f64": 7422.0,
        "gpu_f32": 7308.2,
    },
    "gradient": {
        "cpu_f64": 587.0,
        "cpu_f32": 479.9,
        "gpu_f64": 7755.3,
        "gpu_f32": 7398.8,
    },
}

# Finding 3: MAP fit times (seconds), for reference
FINDING_3 = {
    "cpu_f64_warm_s": 0.27,
    "cpu_f32_warm_s": 0.23,
    "gpu_f64_warm_s": 2.38,
    "gpu_f32_warm_s": 2.22,
}


def find_crossover_region(batch, cpu, gpu):
    """Find batch range where CPU-to-GPU crossover occurs."""
    crossovers = []
    for i in range(len(batch) - 1):
        if (cpu[i] > gpu[i] and cpu[i + 1] <= gpu[i + 1]) or (
            cpu[i] <= gpu[i] and cpu[i + 1] > gpu[i + 1]
        ):
            crossovers.append((batch[i], batch[i + 1]))
    return crossovers


def create_figure():
    """Create the two-panel GPU vs CPU comparison figure."""

    batch = np.array(FORWARD_DATA["batch"])

    # Forward data
    fwd_cpu_f64 = np.array(FORWARD_DATA["cpu_f64"])
    fwd_cpu_f32 = np.array(FORWARD_DATA["cpu_f32"])
    fwd_gpu_f64 = np.array(FORWARD_DATA["gpu_f64"])
    fwd_gpu_f32 = np.array(FORWARD_DATA["gpu_f32"])

    # Gradient data
    grad_cpu_f64 = np.array(GRADIENT_DATA["cpu_f64"])
    grad_cpu_f32 = np.array(GRADIENT_DATA["cpu_f32"])
    grad_gpu_f64 = np.array(GRADIENT_DATA["gpu_f64"])
    grad_gpu_f32 = np.array(GRADIENT_DATA["gpu_f32"])

    # Find crossover regions
    fwd_crossover = find_crossover_region(batch, fwd_cpu_f64, fwd_gpu_f64)
    grad_crossover = find_crossover_region(batch, grad_cpu_f64, grad_gpu_f64)

    # Compute GPU advantage ratios at batch 2048
    fwd_f64_ratio = fwd_cpu_f64[-1] / fwd_gpu_f64[-1]
    fwd_f32_ratio = fwd_cpu_f32[-1] / fwd_gpu_f32[-1]
    grad_f64_ratio = grad_cpu_f64[-1] / grad_gpu_f64[-1]
    grad_f32_ratio = grad_cpu_f32[-1] / grad_gpu_f32[-1]

    # Create figure: two panels side by side, vector format
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    # Define line styles and colors
    # Devices: CPU (solid), GPU (dashed)
    # Precision: f64 (black), f32 (gray)
    colors = {
        "cpu_f64": "#1f77b4",  # blue for CPU f64
        "cpu_f32": "#ff7f0e",  # orange for CPU f32
        "gpu_f64": "#2ca02c",  # green for GPU f64
        "gpu_f32": "#d62728",  # red for GPU f32
    }

    # ===== PANEL A: FORWARD PASS =====
    ax1.loglog(
        batch,
        fwd_cpu_f64,
        "o-",
        color=colors["cpu_f64"],
        label="CPU f64",
        linewidth=1.5,
        markersize=4,
    )
    ax1.loglog(
        batch,
        fwd_cpu_f32,
        "s-",
        color=colors["cpu_f32"],
        label="CPU f32",
        linewidth=1.5,
        markersize=4,
    )
    ax1.loglog(
        batch,
        fwd_gpu_f64,
        "o--",
        color=colors["gpu_f64"],
        label="GPU f64",
        linewidth=1.5,
        markersize=4,
    )
    ax1.loglog(
        batch,
        fwd_gpu_f32,
        "s--",
        color=colors["gpu_f32"],
        label="GPU f32",
        linewidth=1.5,
        markersize=4,
    )

    # Shade crossover region for forward (128-512)
    if fwd_crossover:
        x_min, x_max = fwd_crossover[0]
        ax1.axvspan(x_min, x_max, alpha=0.15, color="gray", zorder=0)

    # GPU advantage annotations at batch 2048, upper-right corner
    ratio_text_a = f"f64: CPU/GPU = {fwd_f64_ratio:.1f}x\nf32: CPU/GPU = {fwd_f32_ratio:.1f}x"
    ax1.text(
        0.98,
        0.98,
        ratio_text_a,
        transform=ax1.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    ax1.set_xlabel("Batch size", fontsize=10)
    ax1.set_ylabel("Time per galaxy (µs)", fontsize=10)
    ax1.set_title("(a)", fontsize=10, loc="left", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.2, linestyle="-", linewidth=0.5)
    ax1.set_xticks([1, 10, 100, 1000, 2048])
    ax1.set_xticklabels(["1", "10", "100", "1k", "2k"], fontsize=8)
    ax1.tick_params(labelsize=8)

    # Add hardware note in panel (a) only
    ax1.text(
        0.98,
        0.02,
        "RTX 3060 vs Ryzen 9 5900X, JAX 0.11",
        transform=ax1.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    # ===== PANEL B: GRADIENT =====
    ax2.loglog(
        batch,
        grad_cpu_f64,
        "o-",
        color=colors["cpu_f64"],
        label="CPU f64",
        linewidth=1.5,
        markersize=4,
    )
    ax2.loglog(
        batch,
        grad_cpu_f32,
        "s-",
        color=colors["cpu_f32"],
        label="CPU f32",
        linewidth=1.5,
        markersize=4,
    )
    ax2.loglog(
        batch,
        grad_gpu_f64,
        "o--",
        color=colors["gpu_f64"],
        label="GPU f64",
        linewidth=1.5,
        markersize=4,
    )
    ax2.loglog(
        batch,
        grad_gpu_f32,
        "s--",
        color=colors["gpu_f32"],
        label="GPU f32",
        linewidth=1.5,
        markersize=4,
    )

    # Shade crossover region for gradient (128-512, but GPU crosses at 128 in f64)
    if grad_crossover:
        x_min, x_max = grad_crossover[0]
        ax2.axvspan(x_min, x_max, alpha=0.15, color="gray", zorder=0)

    # GPU advantage annotations at batch 2048, upper-right corner
    ratio_text_b = f"f64: CPU/GPU = {grad_f64_ratio:.1f}x\nf32: CPU/GPU = {grad_f32_ratio:.1f}x"
    ax2.text(
        0.98,
        0.98,
        ratio_text_b,
        transform=ax2.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    ax2.set_xlabel("Batch size", fontsize=10)
    ax2.set_ylabel("Time per galaxy (µs)", fontsize=10)
    ax2.set_title("(b)", fontsize=10, loc="left", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.2, linestyle="-", linewidth=0.5)
    ax2.set_xticks([1, 10, 100, 1000, 2048])
    ax2.set_xticklabels(["1", "10", "100", "1k", "2k"], fontsize=8)
    ax2.tick_params(labelsize=8)

    # Create legend for both panels (place on outside)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="CPU f64",
            markerfacecolor=colors["cpu_f64"],
            linestyle="-",
            linewidth=1.5,
            markersize=4,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="CPU f32",
            markerfacecolor=colors["cpu_f32"],
            linestyle="-",
            linewidth=1.5,
            markersize=4,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="GPU f64",
            markerfacecolor=colors["gpu_f64"],
            linestyle="--",
            linewidth=1.5,
            markersize=4,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="GPU f32",
            markerfacecolor=colors["gpu_f32"],
            linestyle="--",
            linewidth=1.5,
            markersize=4,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=True,
    )

    return fig, (ax1, ax2)


def save_results_json():
    """Save all plotted numbers and source info to JSON."""

    results = {
        "figure": "Figure 4: GPU vs CPU cost per galaxy as a function of batch size",
        "source": {
            "report_path": "bench/reports/2026-08-20_cuda_device_matrix.md",
            "date": "2026-08-20",
            "findings": "1-3",
            "platform": "RTX 3060 12 GB vs AMD Ryzen 9 5900X",
            "jax_version": "0.11.0",
        },
        "hardware_note": "RTX 3060 vs R9 5900X, JAX 0.11",
        "finding_1_single_galaxy_microseconds": FINDING_1,
        "forward_pass_per_galaxy_microseconds": {
            "batches": FORWARD_DATA["batch"],
            "cpu_f64": FORWARD_DATA["cpu_f64"],
            "cpu_f32": FORWARD_DATA["cpu_f32"],
            "gpu_f64": FORWARD_DATA["gpu_f64"],
            "gpu_f32": FORWARD_DATA["gpu_f32"],
            "description": "predict_photometry_batch, warm steady state",
        },
        "gradient_per_galaxy_microseconds": {
            "batches": GRADIENT_DATA["batch"],
            "cpu_f64": GRADIENT_DATA["cpu_f64"],
            "cpu_f32": GRADIENT_DATA["cpu_f32"],
            "gpu_f64": GRADIENT_DATA["gpu_f64"],
            "gpu_f32": GRADIENT_DATA["gpu_f32"],
            "description": "vmap(grad(sum(predict_photometry))), warm steady state",
        },
        "finding_3_map_fit_warm_seconds": FINDING_3,
        "gpu_advantage_at_batch_2048": {
            "forward_f64_ratio": float(FORWARD_DATA["cpu_f64"][-1] / FORWARD_DATA["gpu_f64"][-1]),
            "forward_f32_ratio": float(FORWARD_DATA["cpu_f32"][-1] / FORWARD_DATA["gpu_f32"][-1]),
            "gradient_f64_ratio": float(
                GRADIENT_DATA["cpu_f64"][-1] / GRADIENT_DATA["gpu_f64"][-1]
            ),
            "gradient_f32_ratio": float(
                GRADIENT_DATA["cpu_f32"][-1] / GRADIENT_DATA["gpu_f32"][-1]
            ),
        },
        "crossover_batch_ranges": {
            "forward_approximate_range": "128-512",
            "gradient_approximate_range": "64-256 (f64 at 128), 128-512 (f32)",
            "note": "Extracted from the report; exact crossover batch depends on precision and operation",
        },
    }

    return results


if __name__ == "__main__":
    import os

    # Create output directories
    fig_dir = "analysis/paper1/figures"
    results_dir = "analysis/paper1/results"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create and save figure
    print("Creating Figure 4...")
    fig, axes = create_figure()

    pdf_path = f"{fig_dir}/fig04_gpu_crossover.pdf"
    png_path = f"{fig_dir}/fig04_gpu_crossover.png"

    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    plt.close(fig)

    # Save results JSON
    print("Saving results JSON...")
    results = save_results_json()
    json_path = f"{results_dir}/fig04_gpu_crossover_data.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nDone!")
