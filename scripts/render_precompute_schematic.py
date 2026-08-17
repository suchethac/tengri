# SPDX-License-Identifier: BSD-3-Clause
"""Render the two precompute schematics shown in ``docs/known_limitations.md``.

Figure 1 (``precompute_schematic.png``) walks a real SSP spectrum through the
``WavePrecomp`` build: the SED under the GALEX and ugriz bandpasses, one band
split into sub-bands of equal filter mass, the collapse of each sub-band to a
single stored number, and the runtime contraction against the dust screen.

Figure 2 (``precompute_subband_error.png``) shows where the residual error
lives: the screen is evaluated at K quadrature nodes and held piecewise
constant across each sub-band, so the gap between the true screen and its
sampled version is the entire approximation.

Every curve is real. The SED is an SSP template off the shipped grid, the
bandpasses are the shipped filter curves, and the sub-band edges and nodes
come from :func:`~tengri.utils.grid_interp.subband_quadrature` itself, so the
partition drawn is the one the code actually builds.

Run from the repository root::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/render_precompute_schematic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tengri import load_filter_set, load_ssp_data
from tengri.components.dust.attenuation import calzetti
from tengri.utils.grid_interp import _filter_weight_np, subband_quadrature

# ── Configuration ───────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures"
SSP_PATH = REPO_ROOT / "data" / "fsps_prsc_miles_chabrier.h5"

DPI = 110

#: Bandpasses drawn under the SED in panel (a) of Figure 1.
PANEL_FILTERS = ["galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

#: The band followed through panels (b)-(d). SDSS u carries the Balmer break
#: at 3646 A inside the bandpass, so the template's own flux-weighted node
#: moves noticeably from one sub-band to the next.
PIPELINE_BAND = "sdss_u"

#: The band Figure 2 stresses — the attenuation curve is steepest across it,
#: so it is where the quadrature error is worst.
WORST_BAND = "galex_fuv"

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

#: One color per bandpass in panel (a), blue-to-red across the set.
BAND_COLORS = {
    "galex_fuv": "#6b46c1",
    "galex_nuv": "#4c51bf",
    "sdss_u": "#2b6cb0",
    "sdss_g": "#2f855a",
    "sdss_r": "#b7791f",
    "sdss_i": "#c05621",
    "sdss_z": "#9b2c2c",
}


# ── Data ────────────────────────────────────────────────────────────


def _load_template():
    """One real SSP template: roughly solar metallicity, 100 Myr."""
    ssp = load_ssp_data(str(SSP_PATH))
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    lgmet = np.asarray(ssp.ssp_lgmet)
    lg_age = np.asarray(ssp.ssp_lg_age_gyr)

    # log10(Zsun) = -1.848 on this grid's absolute-metallicity axis.
    i_met = int(np.argmin(np.abs(lgmet - (-1.848))))
    j_age = int(np.argmin(np.abs(lg_age - (-1.0))))
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)[i_met, j_age]
    return wave, flux, float(lgmet[i_met]), float(10.0 ** lg_age[j_age] * 1e3)


def _band_grid(band, wave, flux):
    """Union quadrature grid for one band, exactly as ``preintegrate_grid`` builds it."""
    filter_waves, filter_trans, _ = load_filter_set([band])
    fw = np.asarray(filter_waves[0], dtype=np.float64)
    ft = np.asarray(filter_trans[0], dtype=np.float64)

    grid = np.sort(np.concatenate([wave, fw]))
    lo = max(np.searchsorted(grid, fw[0]) - 1, 0)
    hi = min(np.searchsorted(grid, fw[-1], side="right") + 1, grid.size)
    grid = grid[lo:hi]

    trans = np.interp(grid, fw, ft, left=0.0, right=0.0)
    tw = trans * _filter_weight_np(grid, "bessell")
    template = np.interp(grid, wave, flux)
    return grid, tw, template


def _partition(grid, tw, template):
    """Call the shipped partition so the drawn edges and nodes are the real ones."""
    denom = float(np.trapezoid(tw, grid))
    eff_wave = float(np.trapezoid(tw * grid, grid) / denom)
    integrand = template * tw
    phi, nodes = subband_quadrature(grid, tw, integrand[None, :], denom, N_SUBBANDS, eff_wave)

    # Edges are the K-quantiles of the cumulative filter weight. Recomputed
    # here only for drawing the shaded bands.
    cum_w = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (tw[1:] + tw[:-1]))])
    edges = np.interp(np.linspace(0.0, cum_w[-1], N_SUBBANDS + 1), cum_w, grid)
    return phi[0], nodes[0], edges, eff_wave, denom


def _screen(wavelength):
    """Calzetti diffuse-ISM transmission, the way the code applies it."""
    return np.exp(-TAU_DIFF * np.asarray(calzetti(np.asarray(wavelength, dtype=np.float64))))


# ── Shared drawing helpers ──────────────────────────────────────────


def _tag(ax, text, color):
    """Tag marking a panel as build-time or runtime work.

    Sits just outside the axes, above the top-right corner: inside the frame
    it competes with data in every panel, and which corner is free differs
    from panel to panel.
    """
    ax.text(
        1.0,
        1.015,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        weight="bold",
        color="white",
        zorder=9,
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"),
    )


def _shade_subbands(ax, grid, curve, edges):
    """Alternating fill so the K sub-bands read as distinct regions."""
    for k in range(N_SUBBANDS):
        ax.fill_between(
            grid,
            0,
            curve,
            where=(grid >= edges[k]) & (grid <= edges[k + 1]),
            color=BLUE,
            alpha=0.34 if k % 2 == 0 else 0.14,
            zorder=1,
        )
    for e in edges:
        ax.axvline(e, color=GREY, lw=0.7, ls=(0, (3, 3)), zorder=2)


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


# ── Figure 1, panel (a): the SED under the bandpasses ────────────────


def _panel_sed(ax, wave, flux, meta):
    """The real SSP spectrum with the GALEX and ugriz bandpasses beneath it."""
    lgmet, age_myr = meta
    sel = (wave >= 900.0) & (wave <= 30000.0)
    w, f = wave[sel], flux[sel]
    ax.plot(w, f / f.max(), color=INK, lw=0.85, zorder=4)

    ax2 = ax.twinx()
    for band in PANEL_FILTERS:
        fw, ft, _ = load_filter_set([band])
        fwv = np.asarray(fw[0], dtype=np.float64)
        ftv = np.asarray(ft[0], dtype=np.float64)
        ftv = ftv / ftv.max()
        color = BAND_COLORS[band]
        chosen = band == PIPELINE_BAND
        ax2.fill_between(fwv, 0, ftv, color=color, alpha=0.55 if chosen else 0.22, zorder=2)
        ax2.plot(fwv, ftv, color=color, lw=1.4 if chosen else 0.8, zorder=3)
        ax2.text(
            fwv[np.argmax(ftv)],
            ftv.max() + 0.06,
            band.split("_")[-1] if band.startswith("sdss") else band.split("_")[-1].upper(),
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=color,
            weight="bold" if chosen else "normal",
            zorder=5,
        )

    ax2.set_ylim(0, 3.4)
    ax2.set_yticks([])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(900, 20000)
    ax.set_ylim(1.2e-3, 3.0)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8.5)
    ax.set_ylabel("$F_\\nu$  (normalized)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title(
        f"(a)  SSP template ({age_myr:.0f} Myr, $\\log_{{10}}Z$ = {lgmet:.2f})\n"
        "under the GALEX + ugriz bandpasses",
        fontsize=9,
        pad=6,
    )
    ax.annotate(
        "the band followed\nin (b)–(d)",
        xy=(3550, 0.0072),
        xytext=(3550, 0.0016),
        fontsize=7.0,
        color=BLUE,
        weight="bold",
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9),
    )
    _tag(ax, "BUILD ONCE", BLUE)


# ── Figure 1, panel (b): the equal-mass split ────────────────────────


def _panel_split(ax, grid, tw, template, edges, nodes):
    """The chosen band cut into K sub-bands of equal filter mass."""
    tw_norm = tw / tw.max()
    _shade_subbands(ax, grid, tw_norm, edges)
    ax.plot(grid, tw_norm, color=BLUE, lw=1.3, zorder=3)

    for k in range(N_SUBBANDS):
        ax.text(
            0.5 * (edges[k] + edges[k + 1]),
            0.075,
            "$1/K$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.2,
            color=BLUE,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85),
        )

    # Nodes are wavelengths, not SED values — draw them as axis ticks rather
    # than as markers riding on a curve, where their height means nothing.
    for node in nodes:
        ax.plot(
            [node, node],
            [0.0, 0.055],
            transform=ax.get_xaxis_transform(),
            color=RED,
            lw=1.6,
            zorder=8,
            clip_on=False,
        )
    ax.plot([], [], color=RED, lw=1.6, label="node $\\lambda^{\\star}_k$")
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.92)

    ax2 = ax.twinx()
    ax2.plot(grid, template / template.max(), color=INK, lw=1.0, zorder=4)
    ax2.set_ylabel("$F_\\nu$  (normalized)", fontsize=8.5)
    ax2.tick_params(labelsize=7.5)
    ax2.set_ylim(0, 1.25)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.22)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8.5)
    ax.set_ylabel("filter weight  $T_b w$", fontsize=8.5, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title(
        "(b)  split into $K = 5$ sub-bands of equal filter mass\n"
        "(the five shaded areas are equal)",
        fontsize=9,
        pad=6,
    )
    _tag(ax, "BUILD ONCE", BLUE)


# ── Figure 1, panel (c): the collapse ───────────────────────────────


def _panel_collapse(ax, grid, tw, template, phi, nodes, edges, denom):
    """Each sub-band's area under the integrand becomes one stored number."""
    integrand = template * tw / denom
    scale = integrand.max()
    _shade_subbands(ax, grid, integrand / scale, edges)
    ax.plot(grid, integrand / scale, color=INK, lw=1.1, zorder=4)

    # Bars whose AREA equals Phi_k, so "curve area -> one number" is literal.
    widths = np.diff(edges)
    heights = phi / widths / scale
    ax.bar(
        edges[:-1],
        heights,
        width=widths,
        align="edge",
        facecolor="none",
        edgecolor=ORANGE,
        lw=1.5,
        zorder=5,
    )
    for k, node in enumerate(nodes):
        ax.plot([node], [heights[k]], "o", color=RED, ms=4.5, zorder=6)
        ax.annotate(
            "",
            xy=(node, 0),
            xytext=(node, heights[k]),
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, ls=(0, (2, 2))),
            zorder=5,
        )
    ax.plot([], [], "o", color=RED, ms=4.5, label="node $\\lambda^{\\star}_k$")
    ax.bar([], [], facecolor="none", edgecolor=ORANGE, lw=1.5, label="stored $\\Phi_k$ (area)")
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.92)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8.5)
    ax.set_ylabel("$F_\\nu T_b w$  (normalized)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title(
        "(c)  each sub-band's area collapses to one number $\\Phi_k$\n"
        "at the template's own centroid $\\lambda^{\\star}_k$",
        fontsize=9,
        pad=6,
    )
    ax.text(
        0.5,
        0.055,
        f"{grid.size} wavelengths  →  $K$ = {N_SUBBANDS} numbers + {N_SUBBANDS} nodes,"
        " per (Z, age, band)",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.2,
        style="italic",
        color=ORANGE,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9),
    )
    _tag(ax, "BUILD ONCE", BLUE)


# ── Figure 1, panel (d): the runtime contraction ────────────────────


def _panel_contract(ax, grid, tw, template, phi, nodes, edges, denom):
    """At runtime the screen is evaluated at K nodes and the bars are summed."""
    widths = np.diff(edges)
    scale = (phi / widths).max()
    a_nodes = _screen(nodes)

    ax.bar(
        edges[:-1],
        phi / widths / scale,
        width=widths,
        align="edge",
        facecolor="#e8edf3",
        edgecolor=GREY,
        lw=0.9,
        zorder=3,
        label="$\\Phi_k$  (stored)",
    )
    ax.bar(
        edges[:-1],
        phi * a_nodes / widths / scale,
        width=widths,
        align="edge",
        facecolor=ORANGE,
        alpha=0.75,
        edgecolor=ORANGE,
        lw=1.0,
        zorder=4,
        label="$\\Phi_k \\, A(\\lambda^{\\star}_k)$",
    )

    ax2 = ax.twinx()
    ax2.plot(grid, _screen(grid), color=RED, lw=1.6, zorder=5, label="screen $A(\\lambda)$")
    ax2.plot(nodes, a_nodes, "o", color=RED, ms=5, zorder=6)
    ax2.set_ylabel("screen  $A(\\lambda)$", fontsize=8.5, color=RED)
    ax2.tick_params(axis="y", labelcolor=RED, labelsize=7.5)
    lo, hi = float(_screen(grid).min()), float(_screen(grid).max())
    ax2.set_ylim(lo - 0.55 * (hi - lo), hi + 0.10 * (hi - lo))

    # The numbers this panel is claiming: precompute vs the exact integral.
    exact = float(np.trapezoid(template * _screen(grid) * tw, grid) / denom)
    precomp = float(np.sum(phi * a_nodes))
    single = float(np.sum(phi) * _screen(np.array([_eff_wave(grid, tw)]))[0])
    ax.text(
        0.985,
        0.235,
        f"$f_b = \\sum_k \\Phi_k A(\\lambda^{{\\star}}_k)$\n"
        f"vs exact integral:  {100 * (precomp / exact - 1):+.3f}%\n"
        f"single $\\lambda_{{\\rm eff}}$ instead:  {100 * (single / exact - 1):+.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=INK,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6, alpha=0.95),
    )

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=7.0, loc="upper left", framealpha=0.92)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 2.15)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8.5)
    ax.set_ylabel("band flux contribution", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title(
        "(d)  evaluate $A$ at the $K$ nodes,\nscale the stored bars, and sum",
        fontsize=9,
        pad=6,
    )
    _tag(ax, "PER CALL", ORANGE)


def _eff_wave(grid, tw):
    return float(np.trapezoid(tw * grid, grid) / np.trapezoid(tw, grid))


def render_pipeline_figure(out_path: Path) -> None:
    """Figure 1 — a real SED walked through the WavePrecomp build."""
    wave, flux, lgmet, age_myr = _load_template()
    grid, tw, template = _band_grid(PIPELINE_BAND, wave, flux)
    phi, nodes, edges, _eff, denom = _partition(grid, tw, template)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    _panel_sed(axes[0, 0], wave, flux, (lgmet, age_myr))
    _panel_split(axes[0, 1], grid, tw, template, edges, nodes)
    _panel_collapse(axes[1, 0], grid, tw, template, phi, nodes, edges, denom)
    _panel_contract(axes[1, 1], grid, tw, template, phi, nodes, edges, denom)

    fig.suptitle(
        f"Build-time photometric precomputation — {PIPELINE_BAND.replace('_', ' ')}, "
        f"Calzetti $\\tau_{{\\rm diff}}$ = {TAU_DIFF}",
        fontsize=10.5,
        weight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.068, right=0.932, bottom=0.075, top=0.885, wspace=0.42, hspace=0.46)
    _guard_canvas(fig)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Figure 2: where the residual error lives ────────────────────────


def _draw_subband_panel(ax, grid, tw, nodes, edges, eff_wave, meta):
    """Panel (a) — bandpass, sub-bands, nodes, and the sampled screen."""
    lgmet, age_myr = meta

    tw_norm = tw / tw.max()
    ax.plot(grid, tw_norm, color=BLUE, lw=1.3, zorder=3, label="$T_b(\\lambda)\\,w(\\lambda)$")
    _shade_subbands(ax, grid, tw_norm, edges)
    for k in range(N_SUBBANDS):
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

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.32)
    ax.set_xlabel("observed wavelength  [Å]", fontsize=9)
    ax.set_ylabel("filter weight  $T_b w$  (normalized)", fontsize=9, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    ax2 = ax.twinx()
    screen = _screen(grid)
    ax2.plot(grid, screen, color=ORANGE, lw=1.8, zorder=5, label="true screen $A(\\lambda)$")

    sampled = np.empty_like(screen)
    for k in range(N_SUBBANDS):
        sel = (grid >= edges[k]) & (grid <= edges[k + 1])
        sampled[sel] = float(_screen(np.array([nodes[k]]))[0])
    sampled[grid < edges[0]] = sampled[grid >= edges[0]][0]
    sampled[grid > edges[-1]] = sampled[grid <= edges[-1]][-1]

    ax2.plot(grid, sampled, color=RED, lw=1.3, ls="--", zorder=5, label="sampled at the $K$ nodes")
    ax2.fill_between(grid, screen, sampled, color=RED, alpha=0.20, zorder=4)
    for node in nodes:
        ax2.plot([node], [float(_screen(np.array([node]))[0])], "o", color=RED, ms=4.5, zorder=6)

    # The retired n_subbands=0 scheme: a first-order Taylor extrapolation of
    # the screen out from lambda_eff, which is what runs away in the rest-UV.
    a_eff = float(_screen(np.array([eff_wave]))[0])
    slope = float(_screen(np.array([eff_wave + 1.0]))[0] - a_eff)
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
    ax.text(8.9, 0.0068, "optical / NIR bands", fontsize=7.4, color=GREEN, ha="right", va="center")

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
    wave, flux, lgmet, age_myr = _load_template()
    grid, tw, template = _band_grid(WORST_BAND, wave, flux)
    _phi, nodes, edges, eff_wave, _denom = _partition(grid, tw, template)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]})
    _draw_subband_panel(axes[0], grid, tw, nodes, edges, eff_wave, (lgmet, age_myr))
    _draw_convergence_panel(axes[1])

    fig.subplots_adjust(left=0.065, right=0.945, bottom=0.135, top=0.855, wspace=0.42)
    _guard_canvas(fig)
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    if not SSP_PATH.exists():
        raise FileNotFoundError(
            f"SSP grid not found at {SSP_PATH}. The figures draw a real template; "
            "download the grid with tengri.download_ssp('fsps_prsc_miles_chabrier')."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_pipeline_figure(OUT_DIR / "precompute_schematic.png")
    render_error_figure(OUT_DIR / "precompute_subband_error.png")


if __name__ == "__main__":
    main()
