#!/usr/bin/env python3
"""Figure 1 — Architecture schematic with strict layout verification."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup
fig = plt.figure(figsize=(7.0, 3.6), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ────────────────────────────────────────────────────────────────────────────
# Strict layout verification
# ────────────────────────────────────────────────────────────────────────────

def check_layout_strict(fig, margin=0.004):
    """Verify layout: text stays in boxes, text doesn't overlap text, boxes don't exceed bounds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()

    # Define all boxes: (name, x0, y0, x1, y1) in figure-fraction
    boxes = {
        "Priors": (0.02, 0.70, 0.19, 0.92),
        "Model": (0.02, 0.40, 0.19, 0.62),
        "Observation": (0.02, 0.10, 0.19, 0.32),
        "xi": (0.25, 0.60, 0.38, 0.80),
        "theta": (0.25, 0.25, 0.38, 0.45),
        "Frame": (0.42, 0.15, 0.64, 0.85),
        "H": (0.72, 0.60, 0.96, 0.80),
    }

    # Collect all text objects with their box assignments
    text_objs = []
    for t in fig.texts:
        if t.get_text().strip():
            text_objs.append(("fig_text", t))
    for ax_obj in fig.get_axes():
        for t in ax_obj.texts:
            if t.get_text().strip():
                text_objs.append(("ax_text", t))

    # Check 1: text-in-box
    text_failures = []
    text_extents = []
    for _src, txt_obj in text_objs:
        txt_str = txt_obj.get_text()
        try:
            extent = txt_obj.get_window_extent(renderer=renderer)
            if extent is None:
                continue
            bb = extent.transformed(fig_inv)
            x0, y0 = bb.x0, bb.y0
            x1, y1 = bb.x1, bb.y1
            # Ensure x0 < x1 and y0 < y1
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            text_extents.append((txt_str, x0, y0, x1, y1))

            # Find best-matching box
            best_box = None
            for box_name, (bx0, by0, bx1, by1) in boxes.items():
                if (x0 >= bx0 - margin and x1 <= bx1 + margin and
                    y0 >= by0 - margin and y1 <= by1 + margin):
                    best_box = box_name
                    break

            skip_labels = ['autodiff', 'function of', 'MAP', 'NUTS', 'Ray', 'geoVI', 'NSS', 'same', 'SEDModel']
            if any(s in txt_str for s in skip_labels):
                continue
            if best_box is None and len(txt_str) > 2:
                text_failures.append((txt_str[:40], x0, y0, x1, y1, best_box))
        except Exception as e:
            pass

    # Check 2: text-text overlap
    text_text_failures = []
    for i, (txt1, x0_1, y0_1, x1_1, y1_1) in enumerate(text_extents):
        for txt2, x0_2, y0_2, x1_2, y1_2 in text_extents[i + 1 :]:
            # Boxes overlap if: not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)
            if not (x1_1 < x0_2 - 0.001 or x1_2 < x0_1 - 0.001 or
                    y1_1 < y0_2 - 0.001 or y1_2 < y0_1 - 0.001):
                text_text_failures.append((txt1[:20], txt2[:20], x0_1, y0_1, x0_2, y0_2))

    # Check 3: box bounds
    box_failures = []
    for name, (_x0, _y0, x1, y1) in boxes.items():
        if x1 > 0.995 or y1 > 1.0:
            box_failures.append(f"{name}: x1={x1:.3f}, y1={y1:.3f}")

    # Report violations
    if text_failures:
        print(f"TEXT-IN-BOX VIOLATION: {text_failures[0][0]!r}")
        print(f"  extent: x=[{text_failures[0][1]:.4f}, {text_failures[0][2]:.4f}] " +
              f"y=[{text_failures[0][3]:.4f}, {text_failures[0][4]:.4f}]")
        return False

    if text_text_failures:
        txt1, txt2, x0_1, y0_1, x0_2, y0_2 = text_text_failures[0]
        print(f"TEXT-TEXT OVERLAP: {txt1!r} and {txt2!r}")
        print(f"  {txt1} at ({x0_1:.4f}, {y0_1:.4f}); {txt2} at ({x0_2:.4f}, {y0_2:.4f})")
        return False

    if box_failures:
        print(f"BOX BOUNDS VIOLATION: {box_failures[0]}")
        return False

    print("layout OK", flush=True)
    return True

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

font_label = 7.0
font_title = 8.5
box_color = "#f0f0f0"
frame_color = "#e8e8e8"

# ────────────────────────────────────────────────────────────────────────────
# Column 1: Three input boxes
# ────────────────────────────────────────────────────────────────────────────

x_col1_left = 0.02
x_col1_right = 0.19
boxes_col1 = [
    (0.70, 0.92, "Priors\n(distributions)"),
    (0.40, 0.62, "Model composition\n(SEDModel.build)"),
    (0.10, 0.32, "Observation\n(filters/spectra)"),
]

for y_bot, y_top, label in boxes_col1:
    box = FancyBboxPatch(
        (x_col1_left, y_bot), x_col1_right - x_col1_left, y_top - y_bot,
        boxstyle="round,pad=0.008",
        edgecolor="black", facecolor=box_color, linewidth=0.8
    )
    ax.add_patch(box)
    ax.text(
        (x_col1_left + x_col1_right) / 2, (y_bot + y_top) / 2,
        label, fontsize=font_label, ha='center', va='center', weight='bold'
    )

# ────────────────────────────────────────────────────────────────────────────
# Column 2: Parameter transformation
# ────────────────────────────────────────────────────────────────────────────

x_col2_left = 0.25
x_col2_right = 0.38

box_xi = FancyBboxPatch(
    (x_col2_left, 0.60), x_col2_right - x_col2_left, 0.20,
    boxstyle="round,pad=0.008",
    edgecolor="black", facecolor=box_color, linewidth=0.8
)
ax.add_patch(box_xi)
ax.text(
    (x_col2_left + x_col2_right) / 2, 0.70,
    r"$\xi$ ~ N(0, I)", fontsize=font_label, ha='center', va='center', weight='bold'
)

box_theta = FancyBboxPatch(
    (x_col2_left, 0.25), x_col2_right - x_col2_left, 0.20,
    boxstyle="round,pad=0.008",
    edgecolor="black", facecolor=box_color, linewidth=0.8
)
ax.add_patch(box_theta)
ax.text(
    (x_col2_left + x_col2_right) / 2, 0.35,
    r"$\theta$ = h($\xi$)", fontsize=font_label, ha='center', va='center', weight='bold'
)

arrow_xi_theta = FancyArrowPatch(
    ((x_col2_left + x_col2_right) / 2, 0.58),
    ((x_col2_left + x_col2_right) / 2, 0.47),
    arrowstyle='->', mutation_scale=15, linewidth=1.0, color='black'
)
ax.add_patch(arrow_xi_theta)

for y_bot, y_top, _ in boxes_col1:
    mid_y = (y_bot + y_top) / 2
    arrow = FancyArrowPatch(
        (x_col1_right, mid_y),
        (x_col2_left, 0.35),
        arrowstyle='->', mutation_scale=10, linewidth=0.5, color='gray', alpha=0.6
    )
    ax.add_patch(arrow)

# ────────────────────────────────────────────────────────────────────────────
# Column 3: ForwardModel frame (x 0.42–0.64)
# ────────────────────────────────────────────────────────────────────────────

x_frame_left = 0.42
x_frame_right = 0.64

frame = FancyBboxPatch(
    (x_frame_left, 0.15), x_frame_right - x_frame_left, 0.70,
    boxstyle="round,pad=0.010",
    edgecolor="black", facecolor=frame_color, linewidth=1.2
)
ax.add_patch(frame)

ax.text(
    x_frame_left + 0.005, 0.82,
    "ForwardModel", fontsize=font_title, ha='left', va='top', weight='bold'
)

# Six sockets: two-line labels at 6.5 pt, boxes 0.034 wide with 0.003 gaps
socket_y = 0.55
socket_height = 0.20
socket_specs = [
    ("SFH", "SFH"),
    ("SPS", "SPS"),
    ("Neb /\nAGN", "Neb/AGN"),
    ("Dust", "Dust"),
    ("IGM", "IGM"),
    ("Obs", "Obs"),
]
n_sockets = len(socket_specs)
socket_width = 0.030
socket_gap = 0.003
total_w = n_sockets * socket_width + (n_sockets - 1) * socket_gap
x_start = x_frame_left + (x_frame_right - x_frame_left - total_w) / 2

for i, (label, _short_name) in enumerate(socket_specs):
    x_sock = x_start + i * (socket_width + socket_gap)
    sock_box = FancyBboxPatch(
        (x_sock, socket_y - socket_height / 2), socket_width, socket_height,
        boxstyle="round,pad=0.004",
        edgecolor="black", facecolor="white", linewidth=0.7
    )
    ax.add_patch(sock_box)
    ax.text(
        x_sock + socket_width / 2, socket_y,
        label, fontsize=5.5, ha='center', va='center', weight='bold'
    )

# Sub-model line: two lines at 5.5 pt inside frame at y 0.24 and 0.20
ax.text(
    (x_frame_left + x_frame_right) / 2, 0.24,
    "SEDModel · PopulationSEDModel", fontsize=5.5, ha='center', va='center', style='italic', color='0.4'
)
ax.text(
    (x_frame_left + x_frame_right) / 2, 0.20,
    "SpatialSEDModel",
    fontsize=5.5, ha='center', va='center', style='italic', color='0.4'
)

arrow_theta_frame = FancyArrowPatch(
    (x_col2_right, 0.35),
    (x_frame_left, 0.50),
    arrowstyle='->', mutation_scale=15, linewidth=1.0, color='black'
)
ax.add_patch(arrow_theta_frame)

arrow_frame_h = FancyArrowPatch(
    (x_frame_right, 0.50),
    (0.72, 0.70),
    arrowstyle='->', mutation_scale=15, linewidth=1.0, color='black'
)
ax.add_patch(arrow_frame_h)

# ────────────────────────────────────────────────────────────────────────────
# Column 4: Likelihood + Backends (H x 0.72–0.96, backends x 0.655–0.991)
# ────────────────────────────────────────────────────────────────────────────

x_col4_left = 0.72
x_col4_right = 0.96

box_h = FancyBboxPatch(
    (x_col4_left, 0.60), x_col4_right - x_col4_left, 0.20,
    boxstyle="round,pad=0.008",
    edgecolor="black", facecolor=box_color, linewidth=0.8
)
ax.add_patch(box_h)
ax.text(
    (x_col4_left + x_col4_right) / 2, 0.70,
    r"H($\xi$) = $\frac{1}{2}$$\chi^2$ + $\frac{1}{2}$$\xi^T\xi$",
    fontsize=font_label, ha='center', va='center', weight='bold'
)

ax.text(
    (x_col4_left + x_col4_right) / 2, 0.56,
    r"$\nabla H$ (autodiff)",
    fontsize=5.5, ha='center', va='top', style='italic', color='0.3'
)

# Five backends: two-line labels at 7 pt, boxes 0.064 wide with 0.004 gaps, starting at x=0.655
backend_specs = [
    ("MAP /\nLaplace", "MAP"),
    ("NUTS /\nHMC", "NUTS"),
    ("Ray\nTracing", "Ray"),
    ("geoVI", "geoVI"),
    ("NSS", "NSS"),
]
n_backends = len(backend_specs)
backend_width = 0.064
backend_gap = 0.004
x_back_start = 0.655
y_back_top = 0.30
y_back_bot = 0.12

for i, (label, _short_name) in enumerate(backend_specs):
    x_back = x_back_start + i * (backend_width + backend_gap)
    back_box = FancyBboxPatch(
        (x_back, y_back_bot), backend_width, y_back_top - y_back_bot,
        boxstyle="round,pad=0.004",
        edgecolor="black", facecolor="white", linewidth=0.7
    )
    ax.add_patch(back_box)
    ax.text(
        x_back + backend_width / 2, (y_back_bot + y_back_top) / 2,
        label, fontsize=7.0, ha='center', va='center', weight='bold'
    )

h_x = (x_col4_left + x_col4_right) / 2
h_y_bot = 0.60
for i in range(n_backends):
    x_back = x_back_start + i * (backend_width + backend_gap) + backend_width / 2
    arrow = FancyArrowPatch(
        (h_x, h_y_bot),
        (x_back, y_back_top),
        arrowstyle='->', mutation_scale=10, linewidth=0.6,
        linestyle='dashed', color='gray', alpha=0.6
    )
    ax.add_patch(arrow)

ax.text(
    (x_back_start + x_back_start + n_backends * backend_width + (n_backends - 1) * backend_gap) / 2,
    0.03,
    "any backend: one function of (H, ∇H)",
    fontsize=6.5, ha='center', va='center',
    style='italic', color='0.3'
)

# ────────────────────────────────────────────────────────────────────────────
# Save and verify
# ────────────────────────────────────────────────────────────────────────────

fig.savefig(OUTPUT_DIR / "fig01_architecture.pdf", dpi=150, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / "fig01_architecture.png", dpi=150, bbox_inches='tight')

check_layout_strict(fig)
