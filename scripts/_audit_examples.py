"""Audit each example by instrumenting matplotlib to report axes state.

For every plot_*.py we run, wrap plt.show() to inspect figures: print
each axes' data extent, xlim, ylim, and number of plotted data points.
A downstream grep can flag:
  * EMPTY axes  (no lines/collections)
  * OUT-OF-RANGE data (min/max outside xlim/ylim by >10x)
  * TINY data-to-frame ratio (data spans <1/100 of frame on log axis)

Usage:
    python scripts/_audit_examples.py examples/agn/plot_agn_hierarchy.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def _axis_data_range(ax):
    """Collect numeric data extent from lines + scatter collections."""
    xs = []
    ys = []
    for line in ax.get_lines():
        xd, yd = line.get_xdata(), line.get_ydata()
        xs.append(np.asarray(xd, dtype=float))
        ys.append(np.asarray(yd, dtype=float))
    for c in ax.collections:
        off = getattr(c, "_offsets", None)
        if off is None:
            continue
        arr = np.asarray(off)
        if arr.ndim == 2 and arr.shape[1] == 2:
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])
    for p in ax.patches:
        bb = p.get_window_extent  # not used; we want data coords
        try:
            path = p.get_path().transformed(p.get_patch_transform())
            pts = path.vertices
            if pts.size:
                xs.append(pts[:, 0])
                ys.append(pts[:, 1])
        except Exception:
            pass
    for im in ax.images:
        extent = im.get_extent()
        if extent:
            xs.append(np.array(extent[:2]))
            ys.append(np.array(extent[2:]))
    if not xs:
        return None
    xall = np.concatenate(xs)
    yall = np.concatenate(ys)
    xall = xall[np.isfinite(xall)]
    yall = yall[np.isfinite(yall)]
    return xall, yall


def _report(script: str):
    fignums = plt.get_fignums()
    for fnum in fignums:
        fig = plt.figure(fnum)
        for i, ax in enumerate(fig.axes):
            tag = f"[{script}] fig{fnum}.ax{i}"
            data = _axis_data_range(ax)
            if data is None:
                print(f"AUDIT {tag}: EMPTY (no data)")
                continue
            xd, yd = data
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xscale = ax.get_xscale()
            yscale = ax.get_yscale()
            pos_yd = yd[yd > 0] if yscale == "log" else yd
            if len(pos_yd) == 0 and yscale == "log":
                print(f"AUDIT {tag}: LOG-Y BUT NO POSITIVE DATA")
                continue
            ymin = pos_yd.min() if yscale == "log" else yd.min()
            ymax = yd.max()
            xmin = xd.min()
            xmax = xd.max()
            # Check data vs lims
            off_y = (ymax < ylim[0] * 0.1) or (ymin > ylim[1] * 10)
            off_x = (xmax < xlim[0] * 0.1) or (xmin > xlim[1] * 10)
            print(
                f"AUDIT {tag}: x=[{xmin:.3g},{xmax:.3g}] xlim={xlim} {xscale} | "
                f"y=[{ymin:.3g},{ymax:.3g}] ylim={ylim} {yscale}"
                + (" OFF-Y" if off_y else "")
                + (" OFF-X" if off_x else "")
            )


def main():
    if len(sys.argv) < 2:
        print("usage: _audit_examples.py <script.py>")
        sys.exit(1)
    script = sys.argv[1]
    path = Path(script).resolve()
    # Monkey-patch plt.show to report before (actually) doing nothing.
    orig_show = plt.show

    def _patched_show(*args, **kwargs):
        _report(path.name)

    plt.show = _patched_show
    # Also monkey-patch savefig to skip heavy I/O during audit
    plt.savefig = lambda *a, **k: None

    # Exec in its own namespace with __file__ set
    ns = {"__name__": "__main__", "__file__": str(path)}
    # Pick the cwd that makes data/ resolve correctly:
    #   - examples/<cat>/plot_*.py expect cwd = examples/<cat>
    #   - notebooks/NN_*.py expect cwd = project root
    if path.parent.name == "notebooks":
        # walk up to find the project root (contains data/*.h5 SSP files)
        proj = path.parent.parent
        os.chdir(proj)
    else:
        os.chdir(path.parent)
    try:
        code = compile(path.read_text(), str(path), "exec")
        exec(code, ns)
    except SystemExit:
        pass
    except Exception as e:
        print(f"AUDIT [{path.name}]: SCRIPT FAILED — {type(e).__name__}: {e}")
        return
    # If the script never called plt.show, still report:
    _report(path.name)


if __name__ == "__main__":
    main()
