#!/usr/bin/env python3
"""Generate placeholder PDFs for paper figures.

Produces eight placeholder PDFs at specified dimensions with light-grey boxes
and dashed borders, centered text indicating the figure content.

Output: /Users/suchethacooray/writing-workspace/projects/tengri/figures/placeholder_fig*.pdf
"""

from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Output directory
OUTPUT_DIR = Path("/Users/suchethacooray/writing-workspace/projects/tengri/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Placeholder specifications: (fig_num, width_in, height_in, content)
PLACEHOLDERS = [
    (
        1,
        7.0,
        3.2,
        "Architecture: priors -> xi ~ N(0,I) -> ForwardModel component slots -> H(xi) -> any backend",
    ),
    (2, 7.0, 4.8, "Panchromatic SED decomposition, X-ray to radio"),
    (
        3,
        7.0,
        3.2,
        "Precomputation: cost per evaluation by strategy (a); LUT-vs-exact per-band error vs z (b)",
    ),
    (4, 7.0, 3.0, "CPU vs GPU cost per galaxy vs batch size (forward, gradient)"),
    (
        5,
        7.0,
        6.5,
        "Three CANDELS galaxies x three configurations: SED fit, SFH, (M*, SFR) posterior",
    ),
    (6, 7.0, 3.0, "Same galaxies: tengri configurations vs published per-code M* and SFR"),
    (7, 7.0, 3.5, "One galaxy through every recommended backend: marginals and cost"),
    (8, 3.4, 3.0, "Per-band Jacobian sensitivity"),
]


def make_placeholder(width_in: float, height_in: float, content: str) -> plt.Figure:
    """Create a single placeholder figure.

    Parameters
    ----------
    width_in : float
        Figure width in inches
    height_in : float
        Figure height in inches
    content : str
        Text content to display

    Returns
    -------
    fig : matplotlib.figure.Figure
        The placeholder figure
    """
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Light-grey box with dashed border
    rect = patches.Rectangle(
        (0.05, 0.05),
        0.9,
        0.9,
        linewidth=2,
        edgecolor="#888888",
        facecolor="#f5f5f5",
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(rect)

    # Centered text
    text_lines = content.split(" — ")
    if len(text_lines) == 2:
        title, subtitle = text_lines
        ax.text(
            0.5,
            0.65,
            title,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="#333333",
            wrap=True,
        )
        ax.text(
            0.5,
            0.35,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
            wrap=True,
        )
    else:
        ax.text(
            0.5,
            0.5,
            content,
            ha="center",
            va="center",
            fontsize=11,
            color="#333333",
            wrap=True,
        )

    return fig


# Generate all placeholders
for fig_num, width_in, height_in, content in PLACEHOLDERS:
    fig = make_placeholder(width_in, height_in, content)
    output_path = OUTPUT_DIR / f"placeholder_fig{fig_num:02d}.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    print(f"Created {output_path}")
    plt.close(fig)

print(f"\nAll {len(PLACEHOLDERS)} placeholder PDFs created in {OUTPUT_DIR}")
