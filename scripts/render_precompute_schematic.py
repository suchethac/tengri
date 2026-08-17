# SPDX-License-Identifier: BSD-3-Clause
"""Render the two precompute schematics shown in ``docs/known_limitations.md``.

Figure 1 (``precompute_schematic.png``) contrasts the exact wavelength-grid
path with the ``WavePrecomp`` path, showing which work moves to build time.

Figure 2 (``precompute_subband_error.png``) shows where the residual error
lives: the screen is evaluated at K quadrature nodes and held piecewise
constant across each sub-band, so the gap between the true screen and its
sampled version is the entire approximation.

Panel (a) of Figure 2 uses the real GALEX FUV bandpass, a real SSP template,
and the real :func:`~tengri.utils.grid_interp.subband_quadrature` partition,
so the sub-band edges and nodes drawn are the ones the code actually builds.

Run from the repository root::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/render_precompute_schematic.py
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tengri import load_filter_set, load_ssp_data
from tengri.components.dust.attenuation import calzetti
from tengri.utils.grid_interp import _filter_weight_np, subband_quadrature

# ── Configuration ───────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures"
SSP_PATH = REPO_ROOT / "data" / "fsps_prsc_miles_chabrier.h5"

#: The band the docs quote as the worst case — the attenuation curve is
#: steepest across this bandpass, so it is where the quadrature is stressed.
BAND = "galex_fuv"

#: Sub-band count. The shipped default (``WavePrecomp(n_subbands=5)``).
N_SUBBANDS = 5

#: Diffuse-ISM optical depth for the illustrative screen, matching the
#: tau_diff = 0.7 quoted in the docs' measured accuracy statement.
TAU_DIFF = 0.7

#: Worst-band error against the exact path, per Appendix "Sub-band
#: Quadrature Precomputation" of the methods paper. Reference values, not
#: recomputed here.
CONVERGENCE = {1: 8.7, 3: 1.4, 5: 0.6, 8: 0.3}

#: Single-effective-wavelength scheme (``n_subbands=0``), GALEX FUV.
TAYLOR_ERROR = {"z = 0.05": 45.0, "z = 1": 215.0}

INK = "#1c1c1c"
GREY = "#8a8a8a"
BLUE = "#2b6cb0"
ORANGE = "#c05621"
GREEN = "#2f6f4f"
RED = "#a02c2c"
BUILD_BG = "#eef3f8"
CALL_BG = "#fdf3ea"


# ── Shared drawing helpers ──────────────────────────────────────────


def _box(ax, x, y, w, h, text, *, face="white", edge=INK, size=8.0, weight="normal"):
    """Draw a rounded box centered on ``(x, y)`` and return its center."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        zorder=4,
        weight=weight,
        linespacing=1.45,
    )
    return x, y


def _arrow(ax, start, end, *, color=INK, style="-|>", lw=1.1):
    """Draw a connector between two anchor points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            color=color,
            linewidth=lw,
            shrinkA=2,
            shrinkB=2,
            zorder=2,
        )
    )


def _lane_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=9.5, weight="bold", color=color)


def _guard_canvas(fig, *, max_inches: float = 40.0) -> None:
    """Refuse to save a figure whose tight bounding box has blown up.

    A wrong blended transform places an artist far off-axes without raising —
    ``get_yaxis_transform`` reads x in *axes* fraction where
    ``get_xaxis_transform`` reads it in *data*, so a wavelength passed to the
    former lands ~1500 axes-widths away. ``bbox_inches="tight"`` then grows
    the canvas to contain it and the only symptom is a gigapixel PNG. Fail
    here instead, while the cause is still legible.
    """
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    if bbox.width > max_inches or bbox.height > max_inches:
        raise RuntimeError(
            f"tight bbox is {bbox.width:.0f} x {bbox.height:.0f} inches — an artist "
            "is placed off-axes, almost certainly by a wrong transform."
        )


# ── Figure 1: what moves to build time ──────────────────────────────


def _draw_exact_lane(ax):
    """Top lane — every likelihood call carries the full wavelength grid."""
    y = 0.845
    ax.add_patch(
        FancyBboxPatch(
            (0.035, y - 0.088),
            0.93,
            0.176,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#f6f6f6",
            edgecolor="none",
            zorder=0,
        )
    )
    _lane_label(ax, 0.05, 0.955, "Exact path   approx=None", GREY)

    specs = [
        (0.155, 0.185, "SSP grid  $F_\\nu(Z, t, \\lambda)$\n15 × 93 × 5994", "white"),
        (0.385, 0.185, "apply screen $A(\\lambda)$\nat all 5994 $\\lambda$", "#fdecec"),
        (0.600, 0.145, "× filter $T_b(\\lambda)$", "white"),
        (0.762, 0.105, "$\\int \\mathrm{d}\\lambda$", "white"),
        (0.900, 0.085, "$f_b$", "#eaeaea"),
    ]
    for x, w, text, face in specs:
        _box(ax, x, y, w, 0.115, text, face=face)
    for (xa, wa, _, _), (xb, wb, _, _) in pairwise(specs):
        _arrow(ax, (xa + wa / 2, y), (xb - wb / 2, y))

    ax.text(
        0.5,
        0.727,
        "per likelihood call: $\\mathcal{O}(n_\\lambda)$ — the whole grid, every call",
        ha="center",
        va="center",
        fontsize=8.2,
        style="italic",
        color=RED,
    )


def _draw_build_row(ax):
    """Build-time row of the precompute lane — paid once, at model build."""
    y = 0.520
    ax.add_patch(
        FancyBboxPatch(
            (0.035, y - 0.082),
            0.93,
            0.164,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=BUILD_BG,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        0.05,
        y + 0.106,
        "BUILD ONCE   (SEDModel.build — no free parameters involved)",
        ha="left",
        va="center",
        fontsize=8.2,
        weight="bold",
        color=BLUE,
    )

    specs = [
        (0.150, 0.180, "SSP grid\n+ filter set"),
        (0.385, 0.225, "split each band into\n$K$ sub-bands of\nequal filter mass"),
        (0.633, 0.205, "$\\Phi_{ijbk},\\ \\lambda^{\\star}_{ijbk}$\n15 × 93 × $n_b$ × $K$"),
        (0.872, 0.190, "fold in IGM\n$T(\\lambda^{\\star}(1{+}z), z)$"),
    ]
    for x, w, text in specs:
        _box(ax, x, y, w, 0.108, text, face="white")
    for (xa, wa, _), (xb, wb, _) in pairwise(specs):
        _arrow(ax, (xa + wa / 2, y), (xb - wb / 2, y), color=BLUE)
    return specs[2][0], y


def _draw_call_row(ax, phi_anchor):
    """Runtime row — all that is left per likelihood call."""
    y = 0.235
    ax.add_patch(
        FancyBboxPatch(
            (0.035, y - 0.082),
            0.93,
            0.164,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=CALL_BG,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        0.05,
        y + 0.108,
        "PER LIKELIHOOD CALL",
        ha="left",
        va="center",
        fontsize=8.2,
        weight="bold",
        color=ORANGE,
    )

    specs = [
        (0.180, 0.260, "evaluate $A_{\\rm diff}, A_{\\rm bc}$\nat the $K$ nodes only", 8.0),
        (
            0.530,
            0.350,
            "contract\n"
            "$f_b = \\sum_{ij} w_{ij} \\sum_k \\Phi_{ijbk}\\,"
            "A_{\\rm diff}(\\lambda^{\\star})\\,A_{\\rm bc}(\\lambda^{\\star})^{y_j}$",
            7.6,
        ),
        (0.870, 0.092, "$f_b$", 8.0),
    ]
    for i, (x, w, text, size) in enumerate(specs):
        _box(ax, x, y, w, 0.108, text, face="#eaeaea" if i == 2 else "white", size=size)
    for (xa, wa, _, _), (xb, wb, _, _) in pairwise(specs):
        _arrow(ax, (xa + wa / 2, y), (xb - wb / 2, y), color=ORANGE)

    # The build-time tensor drops into the runtime contraction.
    _arrow(ax, (phi_anchor[0], phi_anchor[1] - 0.056), (0.505, y + 0.056), color=BLUE)
    ax.text(
        0.648,
        0.372,
        "build-time constant",
        ha="left",
        va="center",
        fontsize=7.6,
        style="italic",
        color=BLUE,
    )

    ax.text(
        0.5,
        0.118,
        "per likelihood call: $2K$ dust-law evaluations per band — independent of $n_\\lambda$",
        ha="center",
        va="center",
        fontsize=8.2,
        style="italic",
        color=GREEN,
    )


def _draw_additive_note(ax):
    """Additive emitters bypass the quadrature entirely."""
    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.005),
            0.93,
            0.078,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#f0f5f1",
            edgecolor=GREEN,
            linewidth=0.8,
            zorder=1,
        )
    )
    ax.text(
        0.5,
        0.044,
        "Additive emitters (dust IR, AGN, radio, X-ray) are rank-one:  "
        "$L_\\nu = \\sum_k A_k(\\theta) S_k(\\lambda)  \\Rightarrow  "
        "f_b = \\sum_k A_k(\\theta)\\, R_{kb}$   "
        "— band responses $R_{kb}$ are exact, no quadrature enters.",
        ha="center",
        va="center",
        fontsize=7.3,
        color=INK,
        zorder=2,
    )


def render_flow_diagram(out_path: Path) -> None:
    """Figure 1 — which work moves from the likelihood call to build time."""
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_exact_lane(ax)
    _lane_label(ax, 0.05, 0.672, "Precompute path   approx=WavePrecomp()", BLUE)
    phi_anchor = _draw_build_row(ax)
    _draw_call_row(ax, phi_anchor)
    _draw_additive_note(ax)

    _guard_canvas(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Figure 2: where the residual error lives ────────────────────────


def _load_band_and_template():
    """Real bandpass, real SSP template, on the union quadrature grid.

    Returns the same quantities ``preintegrate_grid`` assembles internally,
    so the partition below is the one the code actually builds.
    """
    filter_waves, filter_trans, _ = load_filter_set([BAND])
    fw = np.asarray(filter_waves[0], dtype=np.float64)
    ft = np.asarray(filter_trans[0], dtype=np.float64)

    ssp = load_ssp_data(str(SSP_PATH))
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    lgmet = np.asarray(ssp.ssp_lgmet)
    lg_age = np.asarray(ssp.ssp_lg_age_gyr)

    # Roughly solar metallicity (log10 Zsun = -1.848) and a young, FUV-bright
    # population — 100 Myr.
    i_met = int(np.argmin(np.abs(lgmet - (-1.848))))
    j_age = int(np.argmin(np.abs(lg_age - (-1.0))))
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)[i_met, j_age]

    # Union grid + Bessell photon-counting weight, as in preintegrate_grid.
    grid = np.sort(np.concatenate([wave, fw]))
    lo = max(np.searchsorted(grid, fw[0]) - 1, 0)
    hi = min(np.searchsorted(grid, fw[-1], side="right") + 1, grid.size)
    grid = grid[lo:hi]

    trans = np.interp(grid, fw, ft, left=0.0, right=0.0)
    tw = trans * _filter_weight_np(grid, "bessell")
    template = np.interp(grid, wave, flux)
    return grid, tw, template, float(lgmet[i_met]), float(10.0 ** lg_age[j_age] * 1e3)


def _partition(grid, tw, template):
    """Call the shipped partition so the drawn edges/nodes are the real ones."""
    denom = float(np.trapezoid(tw, grid))
    eff_wave = float(np.trapezoid(tw * grid, grid) / denom)
    integrand = template * tw
    phi, nodes = subband_quadrature(grid, tw, integrand[None, :], denom, N_SUBBANDS, eff_wave)
    # Edges are the K-quantiles of the cumulative filter weight (Eq. 1 of the
    # appendix). Recomputed here only for drawing the shaded bands.
    cum_w = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (tw[1:] + tw[:-1]))])
    edges = np.interp(np.linspace(0.0, cum_w[-1], N_SUBBANDS + 1), cum_w, grid)
    return phi[0], nodes[0], edges, eff_wave


def _draw_subband_panel(ax, grid, tw, nodes, edges, eff_wave, meta):
    """Panel (a) — bandpass, sub-bands, nodes, and the sampled screen."""
    lgmet, age_myr = meta

    tw_norm = tw / tw.max()
    ax.plot(grid, tw_norm, color=BLUE, lw=1.3, zorder=3, label="$T_b(\\lambda)\\,w(\\lambda)$")

    # Equal-mass sub-bands: the shaded areas below the curve are equal by
    # construction, which is the whole content of the partition.
    for k in range(N_SUBBANDS):
        ax.fill_between(
            grid,
            0,
            tw_norm,
            where=(grid >= edges[k]) & (grid <= edges[k + 1]),
            color=BLUE,
            alpha=0.34 if k % 2 == 0 else 0.14,
            zorder=1,
        )
        ax.text(
            0.5 * (edges[k] + edges[k + 1]),
            0.135,
            "$1/K$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.4,
            color=BLUE,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85),
        )
    for e in edges:
        ax.axvline(e, color=GREY, lw=0.7, ls=(0, (3, 3)), zorder=2)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.32)
    ax.set_xlabel("observed wavelength  [Å]", fontsize=9)
    ax.set_ylabel("filter weight  $T_b w$  (normalized)", fontsize=9, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # The screen: true smooth curve vs the piecewise-constant sampled version.
    ax2 = ax.twinx()
    screen = np.exp(-TAU_DIFF * np.asarray(calzetti(grid)))
    ax2.plot(grid, screen, color=ORANGE, lw=1.8, zorder=5, label="true screen $A(\\lambda)$")

    sampled = np.empty_like(screen)
    for k in range(N_SUBBANDS):
        sel = (grid >= edges[k]) & (grid <= edges[k + 1])
        sampled[sel] = float(np.exp(-TAU_DIFF * np.asarray(calzetti(np.array([nodes[k]])))[0]))
    sampled[grid < edges[0]] = sampled[grid >= edges[0]][0]
    sampled[grid > edges[-1]] = sampled[grid <= edges[-1]][-1]

    ax2.plot(
        grid,
        sampled,
        color=RED,
        lw=1.3,
        ls="--",
        zorder=5,
        label="sampled at the $K$ nodes",
    )
    ax2.fill_between(grid, screen, sampled, color=RED, alpha=0.20, zorder=4)

    for node in nodes:
        a_node = float(np.exp(-TAU_DIFF * np.asarray(calzetti(np.array([node])))[0]))
        ax2.plot([node], [a_node], "o", color=RED, ms=4.5, zorder=6)

    # The retired n_subbands=0 scheme: a first-order Taylor extrapolation of
    # the screen out from lambda_eff, which is what runs away in the rest-UV.
    a_eff = float(np.exp(-TAU_DIFF * np.asarray(calzetti(np.array([eff_wave])))[0]))
    h = 1.0
    slope = (
        float(np.exp(-TAU_DIFF * np.asarray(calzetti(np.array([eff_wave + h])))[0]) - a_eff) / h
    )
    taylor = a_eff + (grid - eff_wave) * slope
    ax2.plot(
        grid,
        taylor,
        color=GREEN,
        lw=1.2,
        ls=(0, (1, 2)),
        zorder=5,
        label="$K{=}0$: Taylor from $\\lambda_{\\rm eff}$",
    )
    ax2.plot([eff_wave], [a_eff], "D", color=GREEN, ms=5, zorder=6)
    ax2.axvline(eff_wave, color=GREEN, lw=0.9, ls=":", zorder=4, alpha=0.8)
    ax2.text(
        eff_wave,
        0.305,
        "$\\lambda_{\\rm eff}$ — the single point\nthe $K{=}0$ scheme sees",
        transform=ax2.get_xaxis_transform(),
        fontsize=7.0,
        color=GREEN,
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, lw=0.5, alpha=0.92),
    )

    ax2.set_ylabel(
        "screen  $A(\\lambda) = e^{-\\tau_{\\rm diff} k(\\lambda)}$",
        fontsize=9,
        color=ORANGE,
    )
    ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=8)
    lo = min(sampled.min(), screen.min(), taylor.min())
    hi = max(sampled.max(), screen.max(), taylor.max())
    ax2.set_ylim(lo - 0.30 * (hi - lo), hi + 0.06 * (hi - lo))

    ax.set_title(
        f"(a)  $K$ = {N_SUBBANDS} equal-mass sub-bands across GALEX FUV\n"
        f"SSP: {age_myr:.0f} Myr, $\\log_{{10}} Z$ = {lgmet:.2f};  "
        f"Calzetti, $\\tau_{{\\rm diff}}$ = {TAU_DIFF}",
        fontsize=9.5,
        pad=9,
    )

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=7.4, loc="upper left", framealpha=0.92)

    ax.text(
        0.5,
        0.052,
        "the shaded gap is the entire approximation",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        style="italic",
        color=RED,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.88),
    )


def _draw_convergence_panel(ax):
    """Panel (b) — worst-band error against the exact path, versus K."""
    ks = np.array(sorted(CONVERGENCE))
    errs = np.array([CONVERGENCE[k] for k in ks])

    guide_k = np.linspace(0.9, 9.0, 100)
    ax.plot(
        guide_k,
        CONVERGENCE[5] * (5.0 / guide_k) ** 2,
        color=GREY,
        lw=1.0,
        ls=(0, (4, 3)),
        zorder=2,
        label="$1/K^2$ (anchored at $K{=}5$)",
    )
    ax.plot(ks, errs, "o-", color=BLUE, lw=1.5, ms=6, zorder=4, label="sub-band quadrature")
    ax.plot([5], [CONVERGENCE[5]], "o", color=BLUE, ms=13, mfc="none", mew=1.6, zorder=5)
    ax.annotate(
        "default\n$K = 5$",
        xy=(5, CONVERGENCE[5]),
        xytext=(5.7, 2.4),
        fontsize=7.8,
        color=BLUE,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )

    for label, err in TAYLOR_ERROR.items():
        ax.plot([0.55], [err], "s", color=RED, ms=6, zorder=5)
        ax.text(0.72, err, f"$K{{=}}0$, {label}", fontsize=7.4, color=RED, va="center")

    ax.axhspan(0.0035, 0.012, color=GREEN, alpha=0.16, zorder=1)
    ax.text(
        8.9,
        0.0068,
        "optical / NIR bands",
        fontsize=7.4,
        color=GREEN,
        ha="right",
        va="center",
    )

    ax.set_yscale("log")
    ax.set_xlim(0.2, 9.3)
    ax.set_ylim(0.003, 500)
    ax.set_xlabel("$K$  (sub-bands per filter)", fontsize=9)
    ax.set_ylabel("worst-band error vs exact path  [%]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.92)
    ax.set_title(
        "(b)  Convergence — worst band is GALEX FUV,\nover $z \\leq 1$ and $\\tau \\leq 2$",
        fontsize=9.5,
        pad=9,
    )


def render_error_figure(out_path: Path) -> None:
    """Figure 2 — the sub-band partition and its convergence."""
    grid, tw, template, lgmet, age_myr = _load_band_and_template()
    _, nodes, edges, eff_wave = _partition(grid, tw, template)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]})
    _draw_subband_panel(axes[0], grid, tw, nodes, edges, eff_wave, (lgmet, age_myr))
    _draw_convergence_panel(axes[1])

    fig.subplots_adjust(left=0.065, right=0.945, bottom=0.135, top=0.855, wspace=0.42)
    _guard_canvas(fig)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    if not SSP_PATH.exists():
        raise FileNotFoundError(
            f"SSP grid not found at {SSP_PATH}. Panel (a) draws a real template; "
            "download the grid with tengri.download_ssp('fsps_prsc_miles_chabrier')."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_flow_diagram(OUT_DIR / "precompute_schematic.png")
    render_error_figure(OUT_DIR / "precompute_subband_error.png")


if __name__ == "__main__":
    main()
