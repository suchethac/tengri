# SPDX-License-Identifier: BSD-3-Clause
"""Render the two precompute schematics shown in ``docs/known_limitations.md``.

Figure 1 (``precompute_schematic.png``) walks a real SSP grid through the
``WavePrecomp`` build and out the other side: the spectra under the GALEX and
ugriz bandpasses, one band split into sub-bands of equal filter mass, the
per-age-bin quadrature nodes, the stored tensor, the SFH age weights, and the
composite-stellar-population sum that turns all of it into one band flux.

The point the figure has to carry is that the K stored numbers exist **per age
bin** (and per metallicity), and that the SFH integration is the weighted sum
over those age bins — not a single set of K numbers for the whole galaxy.

Figure 2 (``precompute_subband_error.png``) shows where the residual error
lives: the screen is evaluated at K quadrature nodes and held piecewise
constant across each sub-band, so the gap between the true screen and its
sampled version is the entire approximation.

Every curve is real. The spectra come off the shipped SSP grid, the bandpasses
are the shipped filter curves, the sub-band edges and nodes come from
:func:`~tengri.utils.grid_interp.subband_quadrature`, and the age weights come
from the shipped cloud-in-cell kernel ``_age_weights_cic`` — so what is drawn
is what the code builds.

Run from the repository root::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/render_precompute_schematic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from tengri import load_filter_set, load_ssp_data
from tengri.components.dust.attenuation import calzetti
from tengri.components.stellar.component import _age_weights_cic
from tengri.utils.grid_interp import _filter_weight_np, subband_quadrature

# ── Configuration ───────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures"
SSP_PATH = REPO_ROOT / "data" / "fsps_prsc_miles_chabrier.h5"

DPI = 110

#: Bandpasses drawn under the spectra in panel (a).
PANEL_FILTERS = ["galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

#: The band followed through the rest of Figure 1. SDSS u carries the Balmer
#: break at 3646 A inside the bandpass, so the template's own flux-weighted
#: node moves visibly from one age bin to the next.
PIPELINE_BAND = "sdss_u"

#: The band Figure 2 stresses — the attenuation curve is steepest across it,
#: so it is where the quadrature error is worst.
WORST_BAND = "galex_fuv"

#: Sub-band count. The shipped default (``WavePrecomp(n_subbands=5)``).
N_SUBBANDS = 5

#: Diffuse-ISM optical depth for the illustrative screen, matching the
#: tau_diff = 0.7 quoted in the docs' measured accuracy statement.
TAU_DIFF = 0.7

#: SSP ages highlighted in panels (a) and (c) [Gyr].
SHOWCASE_AGES_GYR = (0.001, 0.01, 0.1, 1.0, 10.0)

#: Illustrative delayed-tau star formation history.
SFH_T_FORM_GYR = 10.0
SFH_TAU_GYR = 3.0
T_OBS_GYR = 13.8

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

AGE_CMAP = plt.get_cmap("plasma")


# ── Data ────────────────────────────────────────────────────────────


def _load_grid():
    """The SSP grid at roughly solar metallicity: every age bin, one Z."""
    ssp = load_ssp_data(str(SSP_PATH))
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    lgmet = np.asarray(ssp.ssp_lgmet)
    lg_age_gyr = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64)

    # log10(Zsun) = -1.848 on this grid's absolute-metallicity axis.
    i_met = int(np.argmin(np.abs(lgmet - (-1.848))))
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)[i_met]  # (n_age, n_wave)
    return wave, flux, lg_age_gyr, float(lgmet[i_met]), flux.shape[0], lgmet.size


def _band_grid(band, wave, flux_2d):
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
    templates = np.stack([np.interp(grid, wave, f) for f in flux_2d])
    return grid, tw, templates


def _partition(grid, tw, templates):
    """Call the shipped partition so the drawn edges and nodes are the real ones.

    ``templates`` is (n_age, m), so ``phi`` and ``nodes`` come back (n_age, K):
    one set of K numbers and K nodes *per age bin*, which is the whole point.
    """
    denom = float(np.trapezoid(tw, grid))
    eff_wave = float(np.trapezoid(tw * grid, grid) / denom)
    phi, nodes = subband_quadrature(grid, tw, templates * tw, denom, N_SUBBANDS, eff_wave)

    # Edges are the K-quantiles of the cumulative filter weight. They depend
    # on the filter alone, so there is one set for the whole band.
    cum_w = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (tw[1:] + tw[:-1]))])
    edges = np.interp(np.linspace(0.0, cum_w[-1], N_SUBBANDS + 1), cum_w, grid)
    return np.asarray(phi), np.asarray(nodes), edges, eff_wave, denom


def _sfh_age_weights(lg_age_gyr):
    """Age weights from the shipped cloud-in-cell kernel, for a delayed-tau SFH."""
    lookback_yr = np.logspace(5.0, np.log10(SFH_T_FORM_GYR * 1e9), 600)
    since_form_gyr = SFH_T_FORM_GYR - lookback_yr / 1e9
    sfr = np.where(
        since_form_gyr > 0.0,
        (since_form_gyr / SFH_TAU_GYR) * np.exp(-since_form_gyr / SFH_TAU_GYR),
        0.0,
    )
    weights, _total = _age_weights_cic(
        jnp.asarray(lookback_yr),
        jnp.asarray(sfr),
        jnp.asarray(10.0**lg_age_gyr * 1e9),
        T_OBS_GYR,
    )
    return lookback_yr, sfr, np.asarray(weights)


def _screen(wavelength):
    """Calzetti diffuse-ISM transmission, the way the code applies it."""
    return np.exp(-TAU_DIFF * np.asarray(calzetti(np.asarray(wavelength, dtype=np.float64))))


def _age_indices(lg_age_gyr):
    """Grid indices closest to the showcase ages."""
    return [int(np.argmin(np.abs(lg_age_gyr - np.log10(a)))) for a in SHOWCASE_AGES_GYR]


def _age_color(lg_age, lg_lo, lg_hi):
    return AGE_CMAP(0.08 + 0.84 * (lg_age - lg_lo) / (lg_hi - lg_lo))


# ── Shared drawing helpers ──────────────────────────────────────────


def _tag(ax, text, color):
    """Tag marking a panel as build-time or runtime work.

    Uses matplotlib's right-aligned title slot rather than a floating text
    box: inside the frame the tag competes with data (and which corner is
    free differs per panel), while a free-floating box above the axes
    collides with the centered title.
    """
    ax.set_title(text, loc="right", fontsize=6.4, weight="bold", color=color, pad=6)


def _shade_subbands(ax, grid, curve, edges):
    """Alternating fill so the K sub-bands read as distinct regions."""
    for k in range(N_SUBBANDS):
        ax.fill_between(
            grid,
            0,
            curve,
            where=(grid >= edges[k]) & (grid <= edges[k + 1]),
            color=BLUE,
            alpha=0.32 if k % 2 == 0 else 0.13,
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


# ── Figure 1 panels ─────────────────────────────────────────────────


def _panel_sed(ax, wave, flux_2d, lg_age_gyr):
    """(a) SSP spectra at several ages, under the GALEX and ugriz bandpasses."""
    sel = (wave >= 900.0) & (wave <= 20000.0)
    idx = _age_indices(lg_age_gyr)
    lg_lo, lg_hi = lg_age_gyr[idx[0]], lg_age_gyr[idx[-1]]

    for j in idx:
        f = flux_2d[j][sel]
        age_gyr = 10.0 ** lg_age_gyr[j]
        label = f"{age_gyr * 1e3:.0f} Myr" if age_gyr < 1 else f"{age_gyr:.0f} Gyr"
        ax.plot(
            wave[sel],
            f / f.max(),
            color=_age_color(lg_age_gyr[j], lg_lo, lg_hi),
            lw=1.0,
            zorder=4,
            label=label,
        )

    ax2 = ax.twinx()
    for band in PANEL_FILTERS:
        fw, ft, _ = load_filter_set([band])
        fwv = np.asarray(fw[0], dtype=np.float64)
        ftv = np.asarray(ft[0], dtype=np.float64)
        ftv = ftv / ftv.max()
        color = BAND_COLORS[band]
        chosen = band == PIPELINE_BAND
        ax2.fill_between(fwv, 0, ftv, color=color, alpha=0.5 if chosen else 0.2, zorder=2)
        ax2.plot(fwv, ftv, color=color, lw=1.3 if chosen else 0.8, zorder=3)
        ax2.text(
            fwv[np.argmax(ftv)],
            1.06,
            band.split("_")[-1] if band.startswith("sdss") else band.split("_")[-1].upper(),
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=color,
            weight="bold" if chosen else "normal",
            zorder=5,
        )

    ax2.set_ylim(0, 3.6)
    ax2.set_yticks([])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(900, 20000)
    ax.set_ylim(8e-4, 3.0)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8)
    ax.set_ylabel("$F_\\nu$  (normalized)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(
        fontsize=6.4, loc="lower right", framealpha=0.92, title="SSP age", title_fontsize=6.4
    )
    ax.set_title(
        "(a)  the SSP grid — one spectrum per age bin —\nunder the GALEX + ugriz bandpasses",
        fontsize=8.4,
        pad=6,
    )
    _tag(ax, "BUILD ONCE", BLUE)


def _panel_split(ax, grid, tw, edges):
    """(b) The band cut into K sub-bands of equal filter mass."""
    tw_norm = tw / tw.max()
    _shade_subbands(ax, grid, tw_norm, edges)
    ax.plot(grid, tw_norm, color=BLUE, lw=1.3, zorder=3)

    for k in range(N_SUBBANDS):
        ax.text(
            0.5 * (edges[k] + edges[k + 1]),
            0.09,
            f"$1/K$\n#{k + 1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.0,
            color=BLUE,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85),
        )

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8)
    ax.set_ylabel("filter weight  $T_b w$", fontsize=8, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(
        "(b)  the band split into $K = 5$ sub-bands\n"
        "of equal filter mass (the five areas are equal)",
        fontsize=8.4,
        pad=6,
    )
    _tag(ax, "BUILD ONCE", BLUE)


def _panel_per_age(ax, grid, tw, templates, nodes, edges, lg_age_gyr):
    """(c) Each age bin gets its own integrand, and its own K nodes."""
    idx = _age_indices(lg_age_gyr)
    lg_lo, lg_hi = lg_age_gyr[idx[0]], lg_age_gyr[idx[-1]]

    # Light banding only — a full-height fill drowns the five spectra.
    for k in range(N_SUBBANDS):
        ax.axvspan(
            edges[k], edges[k + 1], color=BLUE, alpha=0.15 if k % 2 == 0 else 0.02, zorder=0
        )
    for e in edges:
        ax.axvline(e, color=GREY, lw=0.7, ls=(0, (3, 3)), zorder=2)

    for row, j in enumerate(idx):
        integrand = templates[j] * tw
        color = _age_color(lg_age_gyr[j], lg_lo, lg_hi)
        age_gyr = 10.0 ** lg_age_gyr[j]
        ax.plot(grid, integrand / integrand.max(), color=color, lw=1.1, zorder=4)

        y_row = 1.13 + 0.105 * row
        ax.plot(nodes[j], np.full(N_SUBBANDS, y_row), "|", color=color, ms=8, mew=1.8, zorder=6)
        ax.text(
            grid[-1] - 8,
            y_row,
            f"{age_gyr * 1e3:.0f} Myr" if age_gyr < 1 else f"{age_gyr:.0f} Gyr",
            ha="right",
            va="center",
            fontsize=5.8,
            color=color,
            zorder=7,
        )

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.72)
    ax.set_xlabel("rest wavelength  [Å]", fontsize=8)
    ax.set_ylabel("$F_\\nu T_b w$  (each normalized)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(
        0.5,
        0.055,
        "the nodes $\\lambda^{\\star}_{jk}$ move with age — each is that\n"
        "age bin's own flux-weighted centroid in its sub-band",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.8,
        style="italic",
        color=INK,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9),
    )
    ax.set_title(
        "(c)  every age bin is integrated separately\n"
        "(ticks: that bin's $K$ nodes, colors as in (a))",
        fontsize=8.4,
        pad=6,
    )
    _tag(ax, "BUILD ONCE", BLUE)


def _panel_tensor(ax, fig, phi, lg_age_gyr, n_age, n_met):
    """(d) The stored tensor: K numbers for every age bin, every metallicity."""
    age_gyr = 10.0**lg_age_gyr
    top = float(np.max(phi))
    floor = top * 1e-4
    mesh = ax.pcolormesh(
        np.arange(1, N_SUBBANDS + 1),
        age_gyr,
        np.maximum(phi, floor),
        cmap="viridis",
        norm=LogNorm(vmin=floor, vmax=top),
        shading="nearest",
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("$\\Phi_{jk}$", fontsize=8)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_yscale("log")
    ax.set_xticks(range(1, N_SUBBANDS + 1))
    ax.set_xlabel("sub-band  $k$", fontsize=8)
    ax.set_ylabel("SSP age  [Gyr]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        "(d)  stored: $K$ numbers per age bin\n"
        f"$\\Phi_{{ijbk}}$: {n_met} $Z$ × {n_age} ages × $n_b$ × {N_SUBBANDS}",
        fontsize=8.4,
        pad=6,
    )
    # Drawn inside the heatmap: outside, it lands on the colorbar.
    y_mark = age_gyr[int(0.80 * n_age)]
    ax.annotate(
        "",
        xy=(0.62, y_mark),
        xytext=(N_SUBBANDS + 0.38, y_mark),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.4),
        zorder=6,
    )
    ax.text(
        0.5 * (N_SUBBANDS + 1),
        age_gyr[int(0.88 * n_age)],
        "one row of $K$ numbers\nper age bin",
        ha="center",
        va="center",
        fontsize=6.6,
        color=INK,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.88),
    )
    _tag(ax, "BUILD ONCE", BLUE)


def _panel_sfh(ax, lookback_yr, sfr, weights, lg_age_gyr):
    """(e) The SFH supplies one mass weight per age bin."""
    age_gyr = 10.0**lg_age_gyr
    ax.plot(
        lookback_yr / 1e9, sfr / sfr.max(), color=GREY, lw=1.4, zorder=3, label="SFH: SFR($t$)"
    )
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 12.0)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("lookback age  [Gyr]", fontsize=8)
    ax.set_ylabel("SFR  (normalized)", fontsize=8, color=GREY)
    ax.tick_params(axis="y", labelcolor=GREY, labelsize=7)
    ax.tick_params(axis="x", labelsize=7)

    ax2 = ax.twinx()
    ax2.vlines(age_gyr, 0, weights, color=GREEN, lw=1.5, zorder=4, label="age weight $w_j$")
    ax2.set_ylabel("$w_j$  (sums to 1)", fontsize=8, color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN, labelsize=7)
    ax2.set_ylim(0, float(weights.max()) * 1.35)

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=6.8, loc="upper left", framealpha=0.92)
    ax.set_title(
        "(e)  the SFH integrated onto the same age grid\n"
        "(cloud-in-cell kernel → one weight $w_j$ per bin)",
        fontsize=8.4,
        pad=6,
    )
    _tag(ax, "PER CALL", ORANGE)


def _panel_contract(ax, grid, tw, templates, phi, nodes, weights, lg_age_gyr):
    """(f) The CSP sum: weight each age bin's K numbers and add them all up."""
    age_gyr = 10.0**lg_age_gyr
    a_nodes = _screen(nodes)  # (n_age, K)
    per_age_bare = np.sum(phi, axis=1)
    per_age_screened = np.sum(phi * a_nodes, axis=1)

    ax.fill_between(
        age_gyr,
        0,
        weights * per_age_bare,
        color=GREY,
        alpha=0.30,
        zorder=2,
        label="$w_j \\sum_k \\Phi_{jk}$  (no dust)",
    )
    ax.fill_between(
        age_gyr,
        0,
        weights * per_age_screened,
        color=ORANGE,
        alpha=0.65,
        zorder=3,
        label="$w_j \\sum_k \\Phi_{jk} A(\\lambda^{\\star}_{jk})$",
    )
    ax.plot(age_gyr, weights * per_age_screened, color=ORANGE, lw=1.1, zorder=4)

    f_b = float(np.sum(weights * per_age_screened))
    screen_grid = _screen(grid)
    denom = float(np.trapezoid(tw, grid))
    exact_per_age = np.array(
        [float(np.trapezoid(t * screen_grid * tw, grid) / denom) for t in templates]
    )
    exact = float(np.sum(weights * exact_per_age))

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 12.0)
    ax.set_ylim(0, float(np.max(weights * per_age_bare)) * 1.55)
    ax.set_xlabel("SSP age bin  [Gyr]", fontsize=8)
    ax.set_ylabel("contribution to $f_b$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.8, loc="upper left", framealpha=0.92)
    ax.text(
        0.975,
        0.955,
        "$f_b = \\sum_j w_j \\sum_k \\Phi_{jk}\\, A(\\lambda^{\\star}_{jk})$\n"
        f"sum over {lg_age_gyr.size} age bins × {N_SUBBANDS} sub-bands\n"
        f"vs exact integral:  {100 * (f_b / exact - 1):+.3f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.9,
        color=INK,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6, alpha=0.95),
    )
    ax.set_title(
        "(f)  per call: screen at each bin's nodes,\nweight by $w_j$, sum — the CSP integral",
        fontsize=8.4,
        pad=6,
    )
    _tag(ax, "PER CALL", ORANGE)


def render_pipeline_figure(out_path: Path) -> None:
    """Figure 1 — a real SSP grid walked through the WavePrecomp build."""
    wave, flux_2d, lg_age_gyr, lgmet, n_age, n_met = _load_grid()
    grid, tw, templates = _band_grid(PIPELINE_BAND, wave, flux_2d)
    phi, nodes, edges, _eff, _denom = _partition(grid, tw, templates)
    lookback_yr, sfr, weights = _sfh_age_weights(lg_age_gyr)

    fig, axes = plt.subplots(3, 2, figsize=(11.4, 11.2))
    _panel_sed(axes[0, 0], wave, flux_2d, lg_age_gyr)
    _panel_split(axes[0, 1], grid, tw, edges)
    _panel_per_age(axes[1, 0], grid, tw, templates, nodes, edges, lg_age_gyr)
    _panel_tensor(axes[1, 1], fig, phi, lg_age_gyr, n_age, n_met)
    _panel_sfh(axes[2, 0], lookback_yr, sfr, weights, lg_age_gyr)
    _panel_contract(axes[2, 1], grid, tw, templates, phi, nodes, weights, lg_age_gyr)

    fig.suptitle(
        "Build-time photometric precomputation — "
        f"{PIPELINE_BAND.replace('_', ' ')}, $\\log_{{10}}Z$ = {lgmet:.2f}, "
        f"Calzetti $\\tau_{{\\rm diff}}$ = {TAU_DIFF}",
        fontsize=10.5,
        weight="bold",
        y=0.994,
    )
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.052, top=0.925, wspace=0.42, hspace=0.52)
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
        f"one age bin: {age_myr:.0f} Myr, $\\log_{{10}} Z$ = {lgmet:.2f};  "
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
    wave, flux_2d, lg_age_gyr, lgmet, _n_age, _n_met = _load_grid()
    grid, tw, templates = _band_grid(WORST_BAND, wave, flux_2d)
    _phi, nodes, edges, eff_wave, _denom = _partition(grid, tw, templates)

    j_age = int(np.argmin(np.abs(lg_age_gyr - (-1.0))))
    age_myr = float(10.0 ** lg_age_gyr[j_age] * 1e3)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]})
    _draw_subband_panel(axes[0], grid, tw, nodes[j_age], edges, eff_wave, (lgmet, age_myr))
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
            f"SSP grid not found at {SSP_PATH}. The figures draw real templates; "
            "download the grid with tengri.download_ssp('fsps_prsc_miles_chabrier')."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_pipeline_figure(OUT_DIR / "precompute_schematic.png")
    render_error_figure(OUT_DIR / "precompute_subband_error.png")


if __name__ == "__main__":
    main()
