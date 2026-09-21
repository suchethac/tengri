#!/usr/bin/env python
"""Render the CPU-vs-GPU device-matrix figures from the merged Slurm JSON.

Reads every ``device_matrix_*.json`` written by ``bench/slurm/_run_cells.sh``
(one file per Slurm arm: one CPU SKU, one GPU SKU each) and renders the figure
set plus a machine-readable table.

Deliberately imports nothing from ``tengri`` and never touches JAX: the arms are
measured inside Slurm allocations, and rendering is a separate, cheap step that
must not depend on a CUDA-capable interpreter. numpy + matplotlib only.

Encoding
--------
Hue carries the **device** (the entity), never its rank -- adding a GPU SKU
must not repaint the others, so the hue is looked up by name. Precision is a
**secondary encoding** (solid f64 / dashed f32) rather than four more hues:
eight categorical hues cannot clear the CVD separation floor, and the pairs
that matter here are within-device.

The device hues are validated (OKLab dE, protan/deutan/tritan, adjacent pairs,
light surface). With the two arms measured so far -- CPU and H100 -- the active
pair ``#0072B2``/``#D55E00`` passes every check including 3:1 contrast. The
third and fourth slots (``#CC79A7``, ``#56B4E9``) fall below that contrast
floor, so once a third arm is added the direct line-end labels and the
CSV/Markdown table become required relief rather than convenience.

Usage::

    python bench/scripts/plot_device_matrix.py \
        --results-dir /scratch/$USER/tengri_bench/results \
        --out-dir bench/reports/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────

#: Hue per device, assigned in fixed order and looked up by name. Validated as
#: a 4-slot categorical palette on a light surface.
#: The H100 takes the vermillion slot because it is the subject of the figure
#: and, with only two arms present, #0072B2/#D55E00 is the pair that clears
#: every check including contrast (the two-blue assignment passed CVD but left
#: the GPU on a 2.25:1 fill). Keys are device identities, so adding an L40S
#: arm later cannot repaint these two.
DEVICE_COLORS = {
    "cpu": "#0072B2",
    "gpu:h100_sxm5": "#D55E00",
    "gpu:l40s": "#CC79A7",
    "gpu:v100_sxm2": "#56B4E9",
}
FALLBACK_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9"]

#: Precision as the secondary encoding, not a fifth and sixth hue.
PREC_STYLE = {"f64": dict(ls="-", marker="o"), "f32": dict(ls="--", marker="s")}

GRID = dict(color="#d9d9d9", lw=0.6, alpha=0.9)
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"


def setup_matplotlib():
    """Publication defaults, matching ``analysis/common.py``."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "axes.edgecolor": "#8a8a8a",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ── loading ───────────────────────────────────────────────────────────────


#: Tag suffixes that mark a *continuation* of an arm rather than a new arm:
#: `_ext` extends shape C to larger batches, `_thr` re-runs shape I with
#: bigger chunks. They must merge into the base arm -- plotted separately they
#: would draw one physical device as two series in two hues, which is exactly
#: the "colour follows rank, not entity" failure.
_CONTINUATION_SUFFIXES = ("_ext", "_thr")


def arm_key(meta: dict, rows: list[dict]) -> str:
    """``cpu`` or ``gpu:<sku>`` -- the identity the hue is keyed on."""
    tag = (meta.get("tag") or "").lower()
    for suf in _CONTINUATION_SUFFIXES:
        if tag.endswith(suf):
            tag = tag[: -len(suf)]
    tag = tag.replace("node", "")
    if tag.startswith("gpu_"):
        return "gpu:" + tag[len("gpu_") :]
    if tag.startswith("cpu"):
        return "cpu"
    dev = next((r.get("device") for r in rows if r.get("device")), "cpu")
    return "gpu:unknown" if dev == "gpu" else "cpu"


def merge_rows(rows: list[dict]) -> list[dict]:
    """Fold continuation rows into one row per (shape, precision).

    Sweep points are keyed on their x value so a re-measured batch size
    replaces rather than duplicates; scalar shapes keep the first row seen.
    """
    merged: dict[tuple, dict] = {}
    for r in rows:
        if "error" in r or r.get("shape") is None:
            continue
        key = (r["shape"], r.get("precision"))
        if key not in merged:
            merged[key] = json.loads(json.dumps(r))
            continue
        base = merged[key]
        if "sweep" in r and "sweep" in base:
            # Shape I points are identified by (chunk_size, n_total): the same
            # chunk measured at 1e3 and at 1e6 galaxies are two results, not a
            # duplicate to collapse.
            def ident(c):
                if r["shape"] == "C":
                    return c.get("n")
                return (c.get("chunk_size"), c.get("n_total"))

            by_x = {ident(c): c for c in base["sweep"] if "error" not in c}
            for c in r["sweep"]:
                if "error" not in c:
                    by_x[ident(c)] = c
            sort_key = "n" if r["shape"] == "C" else "chunk_size"
            base["sweep"] = sorted(by_x.values(), key=lambda c: (c.get(sort_key) or 0))
    return list(merged.values())


def arm_label(key: str, meta: dict) -> str:
    """Human label: the SKU, because a bare 'GPU' row is unreadable."""
    if key == "cpu":
        cpu = (meta.get("cpu") or "CPU").replace("(R)", "").replace("(TM)", "")
        cpu = " ".join(cpu.split())
        for noise in ("CPU", "Processor", "@"):
            cpu = cpu.replace(noise, " ")
        return "CPU " + " ".join(cpu.split())[:28]
    gpu = (meta.get("gpu") or "").split(",")[0].strip()
    return gpu or key.split(":", 1)[1].upper()


def arm_short(key: str) -> str:
    """A 3-6 char tag for the direct line-end label.

    Taking the last word of the full label produced "8462Y+" and "HBM3" --
    the least identifying token in each string.
    """
    if key == "cpu":
        return "CPU"
    sku = key.split(":", 1)[1]
    return {"h100_sxm5": "H100", "l40s": "L40S", "v100_sxm2": "V100"}.get(sku, sku.upper()[:6])


def load(results_dir: pathlib.Path) -> list[dict]:
    """One dict per arm: ``{key, label, meta, rows}``, continuations merged."""
    by_key: dict[str, dict] = {}
    for path in sorted(results_dir.glob("device_matrix_*.json")):
        try:
            blob = json.load(open(path))
        except Exception as exc:
            print(f"  ! skipping unreadable {path.name}: {exc}")
            continue
        meta, rows = blob.get("meta", {}), blob.get("rows", [])
        key = arm_key(meta, rows)
        ok = sum(1 for r in rows if "error" not in r)
        print(f"  loaded {path.name}: arm={key} rows={len(rows)} ({ok} ok)")
        if key in by_key:
            by_key[key]["rows"].extend(rows)
            by_key[key]["paths"].append(path.name)
        else:
            by_key[key] = {
                "key": key,
                "label": arm_label(key, meta),
                "meta": meta,
                "rows": list(rows),
                "paths": [path.name],
            }
    arms = list(by_key.values())
    for a in arms:
        a["rows"] = merge_rows(a["rows"])
    # Fixed order so a re-render does not permute the legend.
    arms.sort(key=lambda a: (a["key"] != "cpu", a["key"]))
    for i, a in enumerate(arms):
        a["color"] = DEVICE_COLORS.get(a["key"], FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
    return arms


def find(rows: list[dict], shape: str, precision: str) -> dict | None:
    for r in rows:
        if r.get("shape") == shape and r.get("precision") == precision and "error" not in r:
            return r
    return None


def sweep_xy(row: dict | None, xkey: str, ykey: str):
    """Pull a clean (x, y) pair out of a sweep, dropping errored/missing points."""
    if not row:
        return np.array([]), np.array([])
    xs, ys = [], []
    for c in row.get("sweep", []):
        if "error" in c or c.get(ykey) is None or c.get(xkey) is None:
            continue
        xs.append(c[xkey])
        ys.append(c[ykey])
    order = np.argsort(xs) if xs else []
    return np.asarray(xs, float)[order], np.asarray(ys, float)[order]


def precision_spread(arm: dict, ykey: str) -> float | None:
    """Max fractional |f64 - f32| gap for one arm, over their common batches.

    ``None`` when either precision is missing or they share no batch size.
    """
    x64, y64 = sweep_xy(find(arm["rows"], "C", "f64"), "n", ykey)
    x32, y32 = sweep_xy(find(arm["rows"], "C", "f32"), "n", ykey)
    if len(x64) == 0 or len(x32) == 0:
        return None
    common = np.intersect1d(x64, x32)
    if len(common) == 0:
        return None
    a = np.interp(common, x64, y64)
    b = np.interp(common, x32, y32)
    return float(np.max(np.abs(a - b) / np.maximum(a, b)))


def end_label(ax, x, y, text, color):
    """Direct label at the line end -- the relief the contrast WARN obliges."""
    if len(x) == 0:
        return
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
        ha="left",
        clip_on=False,
    )


def style_axes(ax, xlabel, ylabel, title=None):
    ax.grid(True, which="both", **GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")


def save(fig, out_dir: pathlib.Path, name: str):
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ── figures ───────────────────────────────────────────────────────────────


def fig_batch_scaling(arms, out_dir):
    """Shape C: per-galaxy cost vs batch size. The GPU's best case."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    panels = [
        (axes[0], "forward_us_per_gal", "forward  predict_photometry"),
        (axes[1], "grad_us_per_gal", "gradient  d/dtheta sum(photometry)"),
    ]
    drew = False
    for ax, ykey, title in panels:
        for arm in arms:
            for prec, st in PREC_STYLE.items():
                x, y = sweep_xy(find(arm["rows"], "C", prec), "n", ykey)
                if len(x) == 0:
                    continue
                drew = True
                ax.plot(
                    x, y, color=arm["color"], lw=2.0, markersize=4.5, alpha=0.95,
                    label=f"{arm_short(arm['key'])} · {prec}", **st,
                )
                if prec == "f64":
                    end_label(ax, x, y, arm_short(arm["key"]), arm["color"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(ax, "galaxies per vmapped call", "microseconds per galaxy", title)
    if not drew:
        plt.close(fig)
        return False

    # Where an arm's two precision curves coincide, say so on the figure. They
    # are drawn one over the other and would otherwise read as a single series
    # someone forgot to plot twice -- and for the H100 that coincidence IS the
    # result (precision is ~free when the device never leaves its latency floor).
    for arm in arms:
        spread = precision_spread(arm, "forward_us_per_gal")
        if spread is not None and spread < 0.10:
            axes[0].annotate(
                f"{arm_short(arm['key'])}: f64 and f32 coincide to within {spread * 100:.0f}%",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8,
                color=arm["color"],
            )
            break

    axes[0].legend(frameon=False, loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.12))
    fig.suptitle(
        "tengri forward model: per-galaxy cost vs batch size  (solid f64 · dashed f32)",
        x=0.01, ha="left", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_batch_scaling")
    return True


def fig_latency_floor(arms, out_dir):
    """Per-CALL wall time vs batch -- the finding the per-galaxy view hides.

    Dividing by n turns a constant into a 1/n slope, which looks like scaling.
    Plotted undivided, a launch-latency-bound device is a flat line and a
    compute-bound one rises. Same numbers, and the only view in which the two
    regimes are told apart rather than inferred.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    drew = False
    for arm in arms:
        for prec, st in PREC_STYLE.items():
            row = find(arm["rows"], "C", prec)
            x, y = sweep_xy(row, "n", "forward_us_per_gal")
            if len(x) == 0:
                continue
            drew = True
            ms = x * y / 1e3  # us/gal * n -> us per call -> ms
            ax.plot(x, ms, color=arm["color"], lw=2.0, markersize=4.5,
                    label=f"{arm_short(arm['key'])} · {prec}", **st)
            if prec == "f64":
                end_label(ax, x, ms, arm_short(arm["key"]), arm["color"])
    if not drew:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "galaxies per vmapped call", "wall clock per call  (ms, log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(
        "Flat = launch-latency bound; rising = compute bound  (solid f64 · dashed f32)",
        loc="left",
    )
    fig.tight_layout()
    save(fig, out_dir, "fig_latency_floor")
    return True


def fig_speedup(arms, out_dir):
    """GPU / CPU speedup vs batch size -- the ratio, stated per precision."""
    cpu = next((a for a in arms if a["key"] == "cpu"), None)
    gpus = [a for a in arms if a["key"] != "cpu"]
    if cpu is None or not gpus:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    drew = False
    for ax, ykey, title in [
        (axes[0], "forward_us_per_gal", "forward"),
        (axes[1], "grad_us_per_gal", "gradient"),
    ]:
        for prec, st in PREC_STYLE.items():
            cx, cy = sweep_xy(find(cpu["rows"], "C", prec), "n", ykey)
            if len(cx) == 0:
                continue
            for arm in gpus:
                gx, gy = sweep_xy(find(arm["rows"], "C", prec), "n", ykey)
                if len(gx) == 0:
                    continue
                common = np.intersect1d(cx, gx)
                if len(common) == 0:
                    continue
                ratio = np.interp(common, cx, cy) / np.interp(common, gx, gy)
                drew = True
                ax.plot(
                    common, ratio, color=arm["color"], lw=2.0, markersize=4.5,
                    label=f"{arm_short(arm['key'])} · {prec}", **st,
                )
                if prec == "f64":
                    end_label(ax, common, ratio, f"{ratio[-1]:.1f}x", arm["color"])
        ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=":")
        ax.annotate(
            "parity with CPU", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_MUTED,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(ax, "galaxies per vmapped call", "speedup over CPU  (x)", title)
    if not drew:
        plt.close(fig)
        return False
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "GPU speedup over the CPU arm  (solid f64 · dashed f32; >1 favours the GPU)",
        x=0.01, ha="left", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_speedup")
    return True


def fig_single_galaxy(arms, out_dir):
    """Shapes A/B/D: latency where there is no batch to fill the card."""
    metrics = [
        ("A", "steady_us", 1e-3, "forward\n(1 galaxy)"),
        ("B", "steady_us", 1e-3, "gradient\n(1 galaxy)"),
        ("D", "warm_s", 1e3, "MAP fit\n(adam, warm)"),
    ]
    series, labels, colors, hatches = [], [], [], []
    for arm in arms:
        for prec in ("f64", "f32"):
            vals = []
            for shape, key, scale, _ in metrics:
                r = find(arm["rows"], shape, prec)
                vals.append(r[key] * scale if r and r.get(key) is not None else np.nan)
            if np.all(np.isnan(vals)):
                continue
            series.append(vals)
            labels.append(f"{arm['label']} · {prec}")
            colors.append(arm["color"])
            hatches.append("" if prec == "f64" else "//")
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.6))
    n = len(series)
    width = 0.8 / n
    xs = np.arange(len(metrics))
    for i, (vals, lab, col, hat) in enumerate(zip(series, labels, colors, hatches)):
        pos = xs + (i - (n - 1) / 2) * width
        bars = ax.bar(
            pos, vals, width * 0.9, label=lab, color=col, hatch=hat,
            edgecolor="white", linewidth=1.0,
        )
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.3g}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=7, color=INK_MUTED,
                )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([m[3] for m in metrics])
    style_axes(ax, "", "wall clock  (milliseconds, log)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title(
        "Single-galaxy latency: no batch to fill the card  (hatched = float32)", loc="left"
    )
    fig.tight_layout()
    save(fig, out_dir, "fig_single_galaxy")
    return True


def fig_throughput(arms, out_dir):
    """Shape I: chunked forward throughput at catalog scale."""
    rows = []
    for arm in arms:
        for prec in ("f64", "f32"):
            r = find(arm["rows"], "I", prec)
            if not r:
                continue
            best = max(
                (c for c in r.get("sweep", []) if "error" not in c and c.get("gal_per_s")),
                key=lambda c: (c.get("n_total", 0), c["gal_per_s"]),
                default=None,
            )
            if best:
                rows.append((f"{arm['label']} · {prec}", arm["color"],
                             "" if prec == "f64" else "//", best["gal_per_s"], best["n_total"]))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2.2))
    ypos = np.arange(len(rows))
    bars = ax.barh(
        ypos, [r[3] for r in rows], color=[r[1] for r in rows],
        hatch=[r[2] for r in rows], edgecolor="white", linewidth=1.0, height=0.7,
    )
    for b, r in zip(bars, rows):
        ax.annotate(
            f"  {r[3]:,.0f} gal/s   (N={r[4]:,})", xy=(b.get_width(), b.get_y() + b.get_height() / 2),
            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.margins(x=0.35)
    style_axes(ax, "galaxies per second  (log)", "")
    ax.set_title("Forward throughput at catalog scale, chunked  (hatched = float32)", loc="left")
    fig.tight_layout()
    save(fig, out_dir, "fig_throughput")
    return True


def fig_compile(arms, out_dir):
    """Compile time is a fixed toll a short job cannot amortise -- show it."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    drew = False
    for arm in arms:
        for prec, st in PREC_STYLE.items():
            x, y = sweep_xy(find(arm["rows"], "C", prec), "n", "forward_compile_ms")
            if len(x) == 0:
                continue
            drew = True
            ax.plot(x, y, color=arm["color"], lw=2.0, markersize=4.5,
                    label=f"{arm_short(arm['key'])} · {prec}", **st)
    if not drew:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "galaxies per vmapped call", "first-call wall clock  (ms, log)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("JIT compile of the batched forward pass  (solid f64 · dashed f32)", loc="left")
    fig.tight_layout()
    save(fig, out_dir, "fig_compile")
    return True


# ── the table view (the contrast WARN obliges one) ────────────────────────


def write_tables(arms, out_dir):
    recs = []
    for arm in arms:
        for prec in ("f64", "f32"):
            for shape, label in [("A", "forward_1gal"), ("B", "gradient_1gal")]:
                r = find(arm["rows"], shape, prec)
                if r:
                    recs.append(dict(arm=arm["label"], precision=prec, metric=label,
                                     value_us=r.get("steady_us"), compile_ms=r.get("compile_ms"),
                                     aa_ratio=r.get("aa_ratio"), flops=r.get("flops"), n=1))
            r = find(arm["rows"], "D", prec)
            if r:
                recs.append(dict(arm=arm["label"], precision=prec, metric="map_fit_warm",
                                 value_us=r.get("warm_s", 0) * 1e6,
                                 compile_ms=r.get("cold_s", 0) * 1e3, aa_ratio=None,
                                 flops=None, n=r.get("n_steps")))
            r = find(arm["rows"], "C", prec)
            for c in (r or {}).get("sweep", []):
                if "error" in c:
                    continue
                recs.append(dict(arm=arm["label"], precision=prec, metric="batch_forward",
                                 value_us=c.get("forward_us_per_gal"),
                                 compile_ms=c.get("forward_compile_ms"),
                                 aa_ratio=c.get("forward_aa"), flops=None, n=c.get("n")))
                recs.append(dict(arm=arm["label"], precision=prec, metric="batch_gradient",
                                 value_us=c.get("grad_us_per_gal"), compile_ms=None,
                                 aa_ratio=c.get("grad_aa"), flops=None, n=c.get("n")))
            r = find(arm["rows"], "I", prec)
            for c in (r or {}).get("sweep", []):
                if "error" in c:
                    continue
                recs.append(dict(arm=arm["label"], precision=prec, metric="throughput",
                                 value_us=c.get("us_per_gal"), compile_ms=c.get("compile_ms"),
                                 aa_ratio=None, flops=c.get("gal_per_s"), n=c.get("n_total")))
    if not recs:
        return
    cols = ["arm", "precision", "metric", "n", "value_us", "compile_ms", "aa_ratio", "flops"]
    with open(out_dir / "device_matrix_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)
    print(f"  wrote device_matrix_table.csv ({len(recs)} rows)")

    # A compact Markdown view of the headline shapes only.
    lines = ["| arm | prec | forward 1gal (us) | grad 1gal (us) | batch fwd us/gal (best) | MAP warm (s) |",
             "|---|---|---|---|---|---|"]
    for arm in arms:
        for prec in ("f64", "f32"):
            a = find(arm["rows"], "A", prec)
            b = find(arm["rows"], "B", prec)
            c = find(arm["rows"], "C", prec)
            d = find(arm["rows"], "D", prec)
            if not any([a, b, c, d]):
                continue
            _, cy = sweep_xy(c, "n", "forward_us_per_gal")
            cells = [
                arm["label"],
                prec,
                f"{a['steady_us']:.3g}" if a else "-",
                f"{b['steady_us']:.3g}" if b else "-",
                f"{cy.min():.3g}" if len(cy) else "-",
                f"{d['warm_s']:.4g}" if d else "-",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "device_matrix_table.md").write_text("\n".join(lines) + "\n")
    print("  wrote device_matrix_table.md")


# ── driver ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", default="bench/reports/figures")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    print(f"loading from {results_dir}")
    arms = load(results_dir)
    if not arms:
        raise SystemExit(f"no device_matrix_*.json under {results_dir}")

    print("rendering:")
    made = {
        "batch_scaling": fig_batch_scaling(arms, out_dir),
        "latency_floor": fig_latency_floor(arms, out_dir),
        "speedup": fig_speedup(arms, out_dir),
        "single_galaxy": fig_single_galaxy(arms, out_dir),
        "throughput": fig_throughput(arms, out_dir),
        "compile": fig_compile(arms, out_dir),
    }
    write_tables(arms, out_dir)

    skipped = [k for k, v in made.items() if not v]
    if skipped:
        print(f"\nskipped (no data): {', '.join(skipped)}")
    print(f"\nfigures in {out_dir}")


if __name__ == "__main__":
    main()
