#!/usr/bin/env python
"""Silent-no-op detector for sweep examples.

Runs a sweep script, captures every plotted line, groups lines by length, and
flags any group of >=3 same-length curves that are all ~identical (the swept
parameter did not move the output — the qsogen_ebv / cat3d class of bug).
"""

import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def audit(path):
    # Neutralise file writes so scripts that savefig(label) don't error or
    # litter; we only care about the in-memory line data.
    plt.savefig = lambda *a, **k: None
    plt.Figure.savefig = lambda *a, **k: None
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        return f"ERROR {type(e).__name__}: {str(e)[:80]}"
    verdict = []
    for num in plt.get_fignums():
        for ax in plt.figure(num).axes:
            by_len = {}
            for line in ax.lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                if y.size > 5 and np.isfinite(y).any():
                    by_len.setdefault(y.size, []).append(y)
            for n, ys in by_len.items():
                if len(ys) < 3:
                    continue
                stack = np.vstack(ys)
                with np.errstate(all="ignore"):
                    spread = np.nanmax(np.nanstd(stack, axis=0))
                    scale = np.nanmedian(np.abs(stack)) + 1e-300
                rel = spread / scale
                if rel < 1e-6:
                    verdict.append(f"NOOP {len(ys)}curves@{n} rel_spread={rel:.1e}")
    plt.close("all")
    return "; ".join(verdict) if verdict else "ok (curves differ)"


if __name__ == "__main__":
    print(audit(sys.argv[1]), flush=True)
