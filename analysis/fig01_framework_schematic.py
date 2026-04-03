#!/usr/bin/env python3
"""Figure 1: Tengri framework overview schematic.

Produces a publication-quality pipeline diagram showing:
  Prior θ → SFH model → SED kernel → Observable → Inference → Posterior

Usage:
    python analysis/fig01_framework_schematic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG_DIR, PAPER_FIG_DIR

# ── Color scheme ──────────────────────────────────────────────────
C_PRIOR = "#4878d0"  # blue — priors / inputs
C_SFH = "#6acc65"    # green — SFH model
C_SED = "#ee854a"    # orange — SED physics
C_OBS = "#9370b8"    # purple — observable
C_INF = "#d65f5f"    # red — inference
C_POST = "#956cb4"   # mauve — posterior output
C_ARR = "#444444"    # arrow color


def rounded_box(ax, cx, cy, w, h, header, sublines, color, fontsize=8.5):
    """Draw a rounded rectangle with bold header and lighter sub-lines."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.06",
        facecolor=color,
        edgecolor="white",
        linewidth=1.8,
        alpha=0.92,
        zorder=3,
    )
    ax.add_patch(patch)

    n_sub = len(sublines)
    if n_sub:
        # Header sits in top third, sub-lines fill the rest
        ax.text(
            cx,
            cy + h * 0.22,
            header,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        for i, line in enumerate(sublines):
            ypos = cy - h * 0.05 - i * h * 0.19
            ax.text(
                cx,
                ypos,
                line,
                ha="center",
                va="center",
                fontsize=fontsize - 1.5,
                color="white",
                alpha=0.93,
                zorder=4,
            )
    else:
        ax.text(
            cx,
            cy,
            header,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            zorder=4,
        )


def horiz_arrow(ax, x1, x2, y, color=C_ARR):
    """Draw a horizontal arrow from x1 to x2 at height y."""
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.4,
            mutation_scale=13,
        ),
        zorder=2,
    )


def stage_label(ax, cx, y, text, color="#333333"):
    ax.text(
        cx,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.5,
        color=color,
        fontweight="bold",
        style="italic",
    )


def main():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # ── Box geometry ──────────────────────────────────────────────
    # 6 stages: Prior | SFH | SED | Observable | Inference | Posterior
    # x-centres chosen to leave room for 5 arrows
    xs = [0.85, 2.35, 4.0, 5.75, 7.5, 9.3]
    widths = [1.45, 1.45, 1.85, 1.75, 1.85, 1.35]
    y_main = 1.9
    box_h = 2.5

    # ── Stage 1: Prior θ ─────────────────────────────────────────
    rounded_box(
        ax,
        xs[0],
        y_main,
        widths[0],
        box_h,
        "Prior  θ",
        ["ParamSpec", "SSP templates", "(MILES / MIST)"],
        C_PRIOR,
    )

    # ── Stage 2: SFH model ────────────────────────────────────────
    rounded_box(
        ax,
        xs[1],
        y_main,
        widths[1],
        box_h,
        "SFH model",
        ["Mean DPL (α, β, τ)", r"IFT field  ξ ~ $\mathcal{N}(0,I)$", "PSD prior (σ, τ)"],
        C_SFH,
    )

    # ── Stage 3: SED kernel ───────────────────────────────────────
    rounded_box(
        ax,
        xs[2],
        y_main,
        widths[2],
        box_h,
        "SED kernel",
        ["DSPS  (diff. SPS)", "Dust attenuation", "Nebular · AGN (opt.)"],
        C_SED,
    )

    # ── Stage 4: Observable ───────────────────────────────────────
    rounded_box(
        ax,
        xs[3],
        y_main,
        widths[3],
        box_h,
        "Observable",
        ["Broadband fluxes", "Spectrum  F(λ)", "Noise model"],
        C_OBS,
    )

    # ── Stage 5: Inference ────────────────────────────────────────
    rounded_box(
        ax,
        xs[4],
        y_main,
        widths[4],
        box_h,
        "Inference",
        ["geoVI  (default)", "Ray Tracing / NUTS", "NSS · Hierarchical"],
        C_INF,
    )

    # ── Stage 6: Posterior ────────────────────────────────────────
    rounded_box(
        ax,
        xs[5],
        y_main,
        widths[5],
        box_h,
        "Posterior",
        ["p(θ | D)", "Samples", "Diagnostics"],
        C_POST,
    )

    # ── Horizontal arrows ─────────────────────────────────────────
    gap = 0.06
    for i in range(len(xs) - 1):
        x1 = xs[i] + widths[i] / 2 + gap
        x2 = xs[i + 1] - widths[i + 1] / 2 - gap
        horiz_arrow(ax, x1, x2, y_main)

    # ── Stage labels above boxes ──────────────────────────────────
    labels = ["Prior", "SFH", "Forward model", "Likelihood", "Inference", "Output"]
    label_colors = [C_PRIOR, C_SFH, C_SED, C_OBS, C_INF, C_POST]
    y_label = 3.4
    for x, lbl, col in zip(xs, labels, label_colors):
        stage_label(ax, x, y_label, lbl, color=col)

    # ── Thin separator line under stage labels ────────────────────
    ax.axhline(y=3.2, xmin=0.01, xmax=0.99, color="#cccccc", lw=0.6, zorder=1)

    plt.tight_layout(pad=0.3)

    out_path = FIG_DIR / "fig01_overview.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig01_overview.pdf"
        fig.savefig(paper_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()
