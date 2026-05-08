"""Render the population-VI scaling figure to analysis/figures/.

Reads bench/results/vi_scaling_benchmark.json (written by bench/scripts/benchmark_vi_xlarge.py)
and produces a 6-panel figure: wall-time, peak ΔRSS, VI iterations, σ_PSD
recovery, τ_PSD recovery, and σ-half-width vs N.

Usage
-----
Render once::

    .venv/bin/python analysis/render_vi_scaling.py

Watch and re-render every WATCH_S seconds while the benchmark runs::

    .venv/bin/python analysis/render_vi_scaling.py --watch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH_BASIC = REPO_ROOT / "data" / "vi_scaling_benchmark.json"
RESULTS_PATH_RICH = REPO_ROOT / "data" / "vi_scaling_benchmark_rich.json"
RESULTS_PATH_SPEC = REPO_ROOT / "data" / "vi_scaling_benchmark_spec.json"
RESULTS_PATH_JOINT = REPO_ROOT / "data" / "vi_scaling_benchmark_joint.json"
FIG_DIR = REPO_ROOT / "analysis" / "figures"
WATCH_S = 60

# Resolved at runtime from CLI flags.
RESULTS_PATH = RESULTS_PATH_BASIC
FIG_PNG = FIG_DIR / "vi_scaling.png"
FIG_PDF = FIG_DIR / "vi_scaling.pdf"

TRUTH_SIGMA = 2.0
TRUTH_TAU = 20.0

METHODS = ("native_vi_linear", "native_vi_nonlinear")
LABELS = {"native_vi_linear": "MGVI (linear)", "native_vi_nonlinear": "geoVI (nonlinear)"}
COLORS = {"native_vi_linear": "#1f77b4", "native_vi_nonlinear": "#d62728"}
LINESTYLES = {1: "-", 2: "--", 4: "-.", 8: ":", 16: (0, (1, 1)),
              32: (0, (3, 1, 1, 1)), 64: (0, (5, 1)), 128: (0, (1, 3))}


def _series(rows, method, k, key):
    sel = [r for r in rows if r["method"] == method
           and r["forward_chunk_size"] == k
           and r.get("wall_s_warm", -1) > 0]
    sel.sort(key=lambda r: r["n_gal"])
    if not sel:
        return np.array([]), np.array([])
    return (
        np.array([r["n_gal"] for r in sel]),
        np.array([r[key] for r in sel], dtype=float),
    )


def _constraint_series(rows, method, key, k=1):
    sel = [r for r in rows if r["method"] == method
           and r["forward_chunk_size"] == k
           and r.get(key)]
    sel.sort(key=lambda r: r["n_gal"])
    if not sel:
        return (np.array([]),) * 4
    n = np.array([r["n_gal"] for r in sel])
    return (
        n,
        np.array([r[key]["median"] for r in sel]),
        np.array([r[key]["p16"] for r in sel]),
        np.array([r[key]["p84"] for r in sel]),
    )


def render(rows: list[dict]) -> plt.Figure:
    rows = [r for r in rows if not r.get("error") and r.get("wall_s_warm", -1) > 0]
    ks = sorted({r["forward_chunk_size"] for r in rows})

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax_t, ax_m, ax_i = axes[0]
    ax_sig, ax_tau, ax_sig_err = axes[1]

    # Panel 1: wall-time
    for method in METHODS:
        for k in ks:
            n, t = _series(rows, method, k, "wall_s_warm")
            if n.size == 0:
                continue
            ax_t.plot(n, t, color=COLORS[method], linestyle=LINESTYLES.get(k, "-"),
                      marker="o", markersize=4, label=f"{LABELS[method]}, K={k}")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_ylim(10, 1000)
    ax_t.set_xlabel("N (galaxies)")
    ax_t.set_ylabel("warm wall-time [s]")
    ax_t.set_title("Wall-time scaling")
    ax_t.grid(True, which="both", alpha=0.3)
    # Reference horizontal lines at human time-scales.
    for sec, lbl in (
        (30, "30 s"), (60, "1 min"), (120, "2 min"),
        (180, "3 min"), (300, "5 min"),
    ):
        ax_t.axhline(sec, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_t.text(
            0.98, sec, lbl,
            color="gray", fontsize=7, va="bottom", ha="right",
            transform=ax_t.get_yaxis_transform(),
        )

    # Asymptotic O(N) reference line, anchored to the K=1 MGVI cell at the
    # largest N >= 2048. In log-log, slope-1 means t = a * N → log t =
    # log a + log N. Drawn for N >= 2048 only — the regime where the
    # forward model dominates the per-iter cost and per-iter time should
    # be linear in N.
    anchor: tuple[float, float] | None = None
    for method in METHODS:
        n, t = _series(rows, method, 1, "wall_s_warm")
        if n.size == 0:
            continue
        big = n >= 2048
        if not big.any():
            continue
        i = np.argmax(n[big])
        n0 = float(n[big][i])
        t0 = float(t[big][i])
        if anchor is None or t0 > anchor[1]:
            anchor = (n0, t0)
    if anchor is not None:
        n0, t0 = anchor
        # Slope fixed at 1 (linear in N) but anchored to the largest-N point;
        # extend the line across the full plotted N range so the eye can
        # see how the small-N points sit above the asymptote.
        all_n = np.concatenate([
            _series(rows, m, k, "wall_s_warm")[0]
            for m in METHODS for k in ks
            if _series(rows, m, k, "wall_s_warm")[0].size > 0
        ])
        if all_n.size > 0:
            n_ref = np.array([all_n.min(), all_n.max()], dtype=float)
            t_ref = t0 * (n_ref / n0)
            ax_t.plot(n_ref, t_ref, color="black", linestyle="-",
                      linewidth=1.4, alpha=0.8,
                      label=r"$\propto N$ (anchored at $N \geq 2048$)")

    ax_t.legend(fontsize=7, loc="upper left", ncol=2)

    # Panel 2: ΔRSS
    for method in METHODS:
        for k in ks:
            n, m = _series(rows, method, k, "rss_delta_gb")
            if n.size == 0:
                continue
            ax_m.plot(n, m, color=COLORS[method], linestyle=LINESTYLES.get(k, "-"),
                      marker="s", markersize=4, label=f"{LABELS[method]}, K={k}")
    ax_m.axhline(30.0, color="k", linestyle=":", alpha=0.5, label="30 GB budget")
    ax_m.set_xscale("log", base=2)
    ax_m.set_xlabel("N (galaxies)")
    ax_m.set_ylabel("peak ΔRSS [GB]")
    ax_m.set_title("Memory scaling")
    ax_m.grid(True, which="both", alpha=0.3)
    ax_m.legend(fontsize=7, loc="upper left", ncol=2)

    # Panel 3: iters
    for method in METHODS:
        for k in ks:
            n, it = _series(rows, method, k, "n_iters_used_warm")
            if n.size == 0:
                continue
            ax_i.plot(n, it, color=COLORS[method], linestyle=LINESTYLES.get(k, "-"),
                      marker="^", markersize=4, label=f"{LABELS[method]}, K={k}")
    not_conv = [r for r in rows if not r.get("converged", False)]
    if not_conv:
        ax_i.scatter([r["n_gal"] for r in not_conv],
                     [r["n_iters_used_warm"] for r in not_conv],
                     marker="x", color="red", s=80, zorder=5,
                     label="hit cap (NOT converged)")
    cap = max((r["n_iters_max"] for r in rows), default=50)
    ax_i.axhline(cap, color="k", linestyle=":", alpha=0.5, label=f"cap = {cap}")
    ax_i.set_xscale("log", base=2)
    ax_i.set_xlabel("N (galaxies)")
    ax_i.set_ylabel("VI iterations used (warm)")
    ax_i.set_title("Convergence")
    ax_i.grid(True, which="both", alpha=0.3)
    ax_i.legend(fontsize=7, loc="upper left", ncol=2)

    # Panels 4+5: σ, τ posterior vs N (K=1 only, both methods)
    for ax, key, truth, ylabel in (
        (ax_sig, "psd_sigma_summary", TRUTH_SIGMA, r"$\sigma_{\rm PSD}$ posterior"),
        (ax_tau, "psd_tau_summary", TRUTH_TAU, r"$\tau_{\rm PSD}$ [Myr] posterior"),
    ):
        for method in METHODS:
            n, med, p16, p84 = _constraint_series(rows, method, key, k=1)
            if n.size == 0:
                continue
            ax.fill_between(n, p16, p84, color=COLORS[method], alpha=0.2)
            ax.plot(n, med, color=COLORS[method], marker="o", markersize=4,
                    label=LABELS[method])
        ax.axhline(truth, color="k", linestyle="--", alpha=0.7, label="truth")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N (galaxies)")
        ax.set_ylabel(ylabel)
        ax.set_title("Hyperparameter recovery (K=1)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Panel 6: σ 68% half-width vs N (1/sqrt(N) reference)
    for method in METHODS:
        n, med, p16, p84 = _constraint_series(rows, method, "psd_sigma_summary", k=1)
        if n.size == 0:
            continue
        width = (p84 - p16) / 2.0
        ax_sig_err.plot(n, width, color=COLORS[method], marker="o", markersize=4,
                        label=LABELS[method])
    # Anchor 1/sqrt(N) to whichever method has data first.
    for method in METHODS:
        ref_n, _, p16r, p84r = _constraint_series(rows, method, "psd_sigma_summary", k=1)
        if ref_n.size >= 2:
            w0 = (p84r[0] - p16r[0]) / 2.0
            ref = w0 * np.sqrt(ref_n[0] / ref_n)
            ax_sig_err.plot(ref_n, ref, color="gray", linestyle=":",
                            label=r"$1/\sqrt{N}$ reference")
            break
    ax_sig_err.set_xscale("log", base=2)
    ax_sig_err.set_yscale("log")
    ax_sig_err.set_xlabel("N (galaxies)")
    ax_sig_err.set_ylabel(r"$\sigma_{\rm PSD}$ 68% half-width")
    ax_sig_err.set_title("Constraint scaling")
    ax_sig_err.grid(True, which="both", alpha=0.3)
    ax_sig_err.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"PopulationFitter scaling   ({len(rows)} cells)",
        fontsize=13,
    )
    return fig


def _render_placeholder() -> None:
    """Emit a placeholder PNG when JSON is absent — keeps docs builds happy."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        f"Benchmark JSON not yet generated:\n  {RESULTS_PATH.name}\n\n"
        "Run bench/scripts/benchmark_vi_xlarge.py to produce it.",
        ha="center", va="center", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round,pad=0.7", fc="#f6f6f6", ec="#999"),
    )
    fig.savefig(FIG_PNG, dpi=120, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[placeholder] {FIG_PNG} (no JSON yet)")


def render_once() -> None:
    if not RESULTS_PATH.exists():
        _render_placeholder()
        return
    with RESULTS_PATH.open() as f:
        rows = json.load(f)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = render(rows)
    fig.savefig(FIG_PNG, dpi=120, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] {len(rows)} rows -> {FIG_PNG}")


def watch_loop() -> None:
    last_mtime = -1.0
    while True:
        try:
            mtime = RESULTS_PATH.stat().st_mtime if RESULTS_PATH.exists() else -1.0
            if mtime != last_mtime:
                render_once()
                last_mtime = mtime
            time.sleep(WATCH_S)
        except KeyboardInterrupt:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="Re-render whenever the JSON file changes "
                         f"(every {WATCH_S}s).")
    ap.add_argument("--rich-obs", action="store_true",
                    help="Render the 10-band rich-observation benchmark "
                         "(vi_scaling_benchmark_rich.json).")
    ap.add_argument("--spec-obs", action="store_true",
                    help="Render the spectroscopy benchmark "
                         "(vi_scaling_benchmark_spec.json).")
    ap.add_argument("--joint-obs", action="store_true",
                    help="Render the joint photometry+lines benchmark "
                         "(vi_scaling_benchmark_joint.json).")
    args = ap.parse_args()

    global RESULTS_PATH, FIG_PNG, FIG_PDF
    if args.spec_obs:
        RESULTS_PATH = RESULTS_PATH_SPEC
        FIG_PNG = FIG_DIR / "vi_scaling_spec.png"
        FIG_PDF = FIG_DIR / "vi_scaling_spec.pdf"
    elif args.joint_obs:
        RESULTS_PATH = RESULTS_PATH_JOINT
        FIG_PNG = FIG_DIR / "vi_scaling_joint.png"
        FIG_PDF = FIG_DIR / "vi_scaling_joint.pdf"
    elif args.rich_obs:
        RESULTS_PATH = RESULTS_PATH_RICH
        FIG_PNG = FIG_DIR / "vi_scaling_rich.png"
        FIG_PDF = FIG_DIR / "vi_scaling_rich.pdf"

    if args.watch:
        watch_loop()
    else:
        render_once()


if __name__ == "__main__":
    main()
